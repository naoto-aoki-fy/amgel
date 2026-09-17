#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <mutex>
#include <new>
#include <unordered_set>
#include <map>
#include <algorithm>
#include <vector>
#include <unistd.h>
#include <cerrno>
#include <climits>
#include <string>

#include <sys/stat.h>

#include <dlfcn.h>
#include <libelf.h>
#include <gelf.h>
#include <fcntl.h>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <frida-gum.h>

#include <atlc/check_x.hpp>
#include <atlc/check_mpi.hpp>
#include <atlc/check_cuda.hpp>
#include <atlc/check_frida.hpp>

namespace nccl_fold {

    static GumInterceptor *interceptor = NULL;

    typedef struct {
        cudaIpcMemHandle_t handle;
        uint64_t offset;
    } handleOffset;

    typedef struct {
        handleOffset memory;
        cudaIpcEventHandle_t ready;
        uint64_t bytes;
        uint64_t sequence;
    } readyMessage;

    typedef struct {
        cudaIpcEventHandle_t done;
        uint64_t sequence;
    } doneMessage;

    typedef struct {
        cudaIpcMemHandle_t handle;
        void* pointer;
    } importedMemory;

    typedef struct {
        void* buff;
        uint64_t count;
        int datatype;
        int peer;
        cudaStream_t stream;
        uint64_t sequence;
    } sendRecvArgs_t;

    struct EventSlot {
        cudaEvent_t event;
        bool leased;
        EventSlot(cudaEvent_t event_) : event(event_), leased(true) {}
    };

    struct ImportedEvent {
        cudaIpcEventHandle_t handle;
        cudaEvent_t event;
    };

    enum CollectiveKind { Broadcast, AllGather, Reduce, AllReduce, ReduceScatter };
    struct CollectiveArgs {
        CollectiveKind kind;
        const void* sendbuff;
        void* recvbuff;
        size_t count;
        ncclDataType_t datatype;
        ncclRedOp_t op;
        int root;
        cudaStream_t stream;
    };

    struct VirtualComm {
        static constexpr uint64_t MAGIC = UINT64_C(0x4e43434c464f4c44);
        uint64_t magic;
        ncclUniqueId unique_id;
        std::vector<EventSlot> event_pool;
        std::vector<ImportedEvent> imported_event_cache;
        std::vector<importedMemory> ipc_mappings;
        /* IPC-exported grouped-send snapshots cannot be reclaimed while a
         * remote cached mapping may remain open.  They are not reduction scratch. */
        std::vector<void*> retained_ipc_allocations;
        std::vector<uint64_t> next_send_sequence;
        std::vector<uint64_t> next_recv_sequence;
        size_t event_pool_peak;
        size_t scratch_in_flight;
        size_t scratch_high_water;
        size_t scratch_allocations_submitted;
        uint64_t collective_sequence;
        MPI_Comm mpi_comm;
        int rank;
        int ndev;
        std::mutex sequence_mutex;

        VirtualComm() : magic(MAGIC), event_pool_peak(0), scratch_in_flight(0),
            scratch_high_water(0), scratch_allocations_submitted(0), collective_sequence(0),
            mpi_comm(MPI_COMM_NULL), rank(-1), ndev(0) {}
    };

    struct RuntimeState {
        std::map<uintptr_t, size_t> allocations;
        std::mutex pointer_mutex;
        std::unordered_set<VirtualComm*> communicators;
        std::mutex communicator_mutex;
        decltype(&::cudaMalloc<void>) origCudaMalloc;
        decltype(&::cudaFree) origCudaFree;
        cudaError_t (*origCudaMallocAsync)(void**, size_t, cudaStream_t);
        cudaError_t (*origCudaFreeAsync)(void*, cudaStream_t);
        decltype(&::cudaSetDevice) origCudaSetDevice;
        decltype(&::ncclGetUniqueId) origNcclGetUniqueId;
        decltype(&::ncclCommInitRank) origNcclCommInitRank;
        decltype(&::ncclGroupStart) origNcclGroupStart;
        decltype(&::ncclGroupEnd) origNcclGroupEnd;
        decltype(&::ncclSend) origNcclSend;
        decltype(&::ncclRecv) origNcclRecv;
        decltype(&::ncclBroadcast) origNcclBroadcast;
        decltype(&::ncclBcast) origNcclBcast;
        decltype(&::ncclAllGather) origNcclAllGather;
        decltype(&::ncclReduce) origNcclReduce;
        decltype(&::ncclAllReduce) origNcclAllReduce;
        decltype(&::ncclReduceScatter) origNcclReduceScatter;
        decltype(&::ncclCommDestroy) origNcclCommDestroy;
        decltype(&::ncclCommCount) origNcclCommCount;
        decltype(&::ncclCommUserRank) origNcclCommUserRank;
        decltype(&::cudaEventDestroy) cudaEventDestroy;
        decltype(&::cudaIpcCloseMemHandle) cudaIpcCloseMemHandle;
        decltype(&::cudaIpcGetMemHandle) cudaIpcGetMemHandle;
        decltype(&::cudaIpcOpenMemHandle) cudaIpcOpenMemHandle;
        decltype(&::cudaMemcpyAsync) cudaMemcpyAsync;
        decltype(&::cudaEventCreateWithFlags) cudaEventCreateWithFlags;
        decltype(&::cudaEventRecord) cudaEventRecord;
        decltype(&::cudaIpcGetEventHandle) cudaIpcGetEventHandle;
        decltype(&::cudaIpcOpenEventHandle) cudaIpcOpenEventHandle;
        decltype(&::cudaStreamWaitEvent) cudaStreamWaitEvent;
    };

    static RuntimeState runtime;

    enum GroupOpKind { GroupSend, GroupRecv, GroupCollective };
    struct GroupOp {
        GroupOpKind kind;
        VirtualComm* comm;
        sendRecvArgs_t p2p;
        CollectiveArgs collective;
    };
    struct GroupState {
        unsigned int depth = 0;
        std::vector<GroupOp> operations;
    };
    static thread_local GroupState group_state;

    static VirtualComm* getVirtualComm(ncclComm_t handle) {
        if (handle == NULL) return NULL;
        VirtualComm* comm = reinterpret_cast<VirtualComm*>(handle);
        std::lock_guard<std::mutex> lock(runtime.communicator_mutex);
        if (runtime.communicators.count(comm) == 0 || comm->magic != VirtualComm::MAGIC) return NULL;
        return comm;
    }

    static cudaError_t cudaMalloc(void **devPtr, size_t size) {
        cudaError_t const ret = runtime.origCudaMalloc(devPtr, size);
        if (ret == cudaSuccess) {
            std::lock_guard<std::mutex> lock(runtime.pointer_mutex);
            runtime.allocations[(uintptr_t)*devPtr] = size;
        }
        return ret;
    }

    static cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
        cudaError_t const ret = runtime.origCudaMalloc(devPtr, size);
        /* We cannot use buffer allocated with cudaMalloAsync for cudaIpcGetMemHandle */
        if (ret == cudaSuccess) {
            std::lock_guard<std::mutex> lock(runtime.pointer_mutex);
            runtime.allocations[(uintptr_t)*devPtr] = size;
        }
        return ret;
    }

    static cudaError_t cudaFree(void* ptr) {
        cudaError_t ret = runtime.origCudaFree(ptr);
        if (ret == cudaSuccess) { std::lock_guard<std::mutex> lock(runtime.pointer_mutex); runtime.allocations.erase((uintptr_t)ptr); }
        return ret;
    }

    static cudaError_t cudaFreeAsync(void* ptr, cudaStream_t stream) {
        cudaError_t ret = runtime.origCudaFreeAsync(ptr, stream);
        if (ret == cudaSuccess) { std::lock_guard<std::mutex> lock(runtime.pointer_mutex); runtime.allocations.erase((uintptr_t)ptr); }
        return ret;
    }


    static cudaError_t cudaSetDevice(int device) {
        cudaError_t const ret = runtime.origCudaSetDevice(0);
        return ret;

    }

    static inline uint64_t sizeofNcclDataType(int datatype) {
        switch (datatype) {
            case ncclInt8: return sizeof(int8_t);
            case ncclUint8: return sizeof(uint8_t);
            case ncclInt32: return sizeof(int32_t);
            case ncclUint32: return sizeof(uint32_t);
            case ncclInt64: return sizeof(int64_t);
            case ncclUint64: return sizeof(uint64_t);
            case ncclFloat16: return sizeof(__half);
            case ncclFloat32: return sizeof(float);
            case ncclFloat64: return sizeof(double);
            case ncclBfloat16: return sizeof(__nv_bfloat16);
            default: return 0;
        }
        return 0;
    }

    static ncclResult_t getUniqueId(ncclUniqueId* nccl_id) {
        if (nccl_id == NULL) return ncclInvalidArgument;
        static std::atomic<uint64_t> counter(0);
        uint64_t seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        seed ^= (uint64_t)getpid() << 32;
        seed ^= ++counter;
        unsigned char* bytes = reinterpret_cast<unsigned char*>(nccl_id);
        for (size_t i = 0; i < sizeof(*nccl_id); ++i) {
            seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27;
            bytes[i] = (unsigned char)((seed * UINT64_C(2685821657736338717)) >> 56);
        }
        return ncclSuccess;
    }

    /* ncclCommInitRank has no MPI communicator argument, so membership has to
     * be bootstrapped out of band.  NCCL Fold is single-host: small, atomically
     * published files let only the participating processes rendezvous without
     * involving non-members in an MPI_COMM_WORLD collective. */
    struct BootstrapRecord {
        uint64_t magic;
        ncclUniqueId id;
        int ndev;
        int nccl_rank;
        int world_rank;
    };

    static uint64_t hashId(const ncclUniqueId& id, uint64_t seed) {
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&id);
        uint64_t hash = seed;
        for (size_t i = 0; i < sizeof(id); ++i) { hash ^= bytes[i]; hash *= UINT64_C(1099511628211); }
        return hash;
    }

    static bool makeDirectory(const std::string& path) {
        return mkdir(path.c_str(), 0700) == 0 || errno == EEXIST;
    }

    static bool writeAll(int fd, const void* data, size_t bytes) {
        const char* position = static_cast<const char*>(data);
        while (bytes) {
            ssize_t written = write(fd, position, bytes);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) return false;
            position += written; bytes -= (size_t)written;
        }
        return true;
    }

    static bool readRecord(const std::string& path, BootstrapRecord* record) {
        int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return false;
        char* position = reinterpret_cast<char*>(record);
        size_t remaining = sizeof(*record);
        while (remaining) {
            ssize_t got = read(fd, position, remaining);
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) { close(fd); return false; }
            position += got; remaining -= (size_t)got;
        }
        char extra;
        bool exact = read(fd, &extra, 1) == 0;
        close(fd);
        return exact;
    }

    static bool publishFile(const std::string& path, const void* data, size_t bytes) {
        static std::atomic<uint64_t> serial(0);
        std::string temporary = path + ".tmp-" + std::to_string((long long)getpid()) + "-" +
                                std::to_string((unsigned long long)++serial);
        int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) return false;
        bool ok = writeAll(fd, data, bytes) && fsync(fd) == 0;
        close(fd);
        if (ok) ok = link(temporary.c_str(), path.c_str()) == 0;
        unlink(temporary.c_str());
        return ok;
    }

    static bool sameBootstrap(const BootstrapRecord& record, const ncclUniqueId& id,
                              int ndev, int rank) {
        return record.magic == UINT64_C(0x414d47454c425354) && record.ndev == ndev &&
               record.nccl_rank == rank && std::memcmp(&record.id, &id, sizeof(id)) == 0;
    }

    static ncclResult_t bootstrapComm(MPI_Comm* result, const ncclUniqueId& id, int ndev, int rank) {
        int world_rank = -1, world_size = 0;
        if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS ||
            MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS || ndev > world_size)
            return ncclInvalidArgument;

        const char* configured = std::getenv("NCCL_FOLD_BOOTSTRAP_DIR");
        std::string root = configured && *configured ? configured : "/tmp/ncclfold-bootstrap-" + std::to_string((long long)getuid());
        if (!makeDirectory(root)) return ncclSystemError;
        uint64_t h1 = hashId(id, UINT64_C(1469598103934665603));
        uint64_t h2 = hashId(id, UINT64_C(7809847782465536322));
        char name[64];
        std::snprintf(name, sizeof(name), "/comm-%016llx-%016llx",
                      (unsigned long long)h1, (unsigned long long)h2);
        std::string directory = root + name;
        if (!makeDirectory(directory)) return ncclSystemError;

        BootstrapRecord local = {UINT64_C(0x414d47454c425354), id, ndev, rank, world_rank};
        std::string rank_path = directory + "/rank-" + std::to_string(rank);
        if (!publishFile(rank_path, &local, sizeof(local))) return ncclInvalidUsage;

        std::vector<int> members(ndev, -1);
        for (;;) {
            bool complete = true;
            for (int r = 0; r < ndev; ++r) {
                if (members[r] >= 0) continue;
                BootstrapRecord peer;
                if (!readRecord(directory + "/rank-" + std::to_string(r), &peer)) { complete = false; continue; }
                if (!sameBootstrap(peer, id, ndev, r) || peer.world_rank < 0 || peer.world_rank >= world_size)
                    return ncclInvalidUsage;
                members[r] = peer.world_rank;
            }
            if (complete) break;
            usleep(1000);
        }
        std::vector<int> sorted = members;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return ncclInvalidUsage;

        int* tag_upper_bound = NULL, present = 0;
        if (MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_upper_bound, &present) != MPI_SUCCESS ||
            !present || tag_upper_bound == NULL || *tag_upper_bound < 0) return ncclSystemError;
        int tag = -1;
        std::string tag_selection = directory + "/tag";
        if (rank == 0) {
            uint64_t range = (uint64_t)*tag_upper_bound + 1;
            for (uint64_t attempt = 0; attempt < range; ++attempt) {
                int candidate = (int)((h1 + attempt) % range);
                std::string reservation = root + "/tag-" + std::to_string(candidate);
                int fd = open(reservation.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
                if (fd < 0) { if (errno == EEXIST) continue; return ncclSystemError; }
                bool ok = writeAll(fd, &local, sizeof(local)); close(fd);
                if (!ok || !publishFile(tag_selection, &candidate, sizeof(candidate))) {
                    unlink(reservation.c_str()); return ncclSystemError;
                }
                tag = candidate; break;
            }
            if (tag < 0) return ncclSystemError;
        } else {
            for (;;) {
                int fd = open(tag_selection.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd >= 0) {
                    ssize_t got = read(fd, &tag, sizeof(tag)); close(fd);
                    if (got == (ssize_t)sizeof(tag)) break;
                }
                usleep(1000);
            }
        }

        MPI_Group world_group = MPI_GROUP_NULL, member_group = MPI_GROUP_NULL;
        int error = MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        if (error == MPI_SUCCESS) error = MPI_Group_incl(world_group, ndev, members.data(), &member_group);
        if (error == MPI_SUCCESS) error = MPI_Comm_create_group(MPI_COMM_WORLD, member_group, tag, result);
        if (member_group != MPI_GROUP_NULL) MPI_Group_free(&member_group);
        if (world_group != MPI_GROUP_NULL) MPI_Group_free(&world_group);
        if (error != MPI_SUCCESS || *result == MPI_COMM_NULL) return ncclSystemError;

        /* Ensure no later communicator can reuse this creation tag until every
         * member has left MPI_Comm_create_group. */
        if (MPI_Barrier(*result) != MPI_SUCCESS) { MPI_Comm_free(result); return ncclSystemError; }
        if (rank == 0) {
            unlink((root + "/tag-" + std::to_string(tag)).c_str());
            unlink(tag_selection.c_str());
            for (int r = 0; r < ndev; ++r) unlink((directory + "/rank-" + std::to_string(r)).c_str());
            rmdir(directory.c_str());
        }
        return ncclSuccess;
    }

    void* getAllocation(void* pointer_input, size_t bytes, uint64_t* offset) {
        std::lock_guard<std::mutex> lock(runtime.pointer_mutex);
        uintptr_t p = (uintptr_t)pointer_input;
        std::map<uintptr_t, size_t>::iterator it = runtime.allocations.upper_bound(p);
        if (it == runtime.allocations.begin()) return NULL;
        --it;
        size_t delta = p - it->first;
        if (delta > it->second || bytes > it->second - delta) return NULL;
        if (offset) *offset = delta;
        return (void*)it->first;
    }

    static ncclResult_t commInitRank(ncclComm_t* comm, int ndev, ncclUniqueId nccl_id, int rank) {
        if (comm == NULL || ndev <= 0 || rank < 0 || rank >= ndev) return ncclInvalidArgument;
        *comm = NULL;
        VirtualComm* virtual_comm = new (std::nothrow) VirtualComm;
        if (virtual_comm == NULL) return ncclSystemError;
        virtual_comm->unique_id = nccl_id;
        ncclResult_t bootstrap = bootstrapComm(&virtual_comm->mpi_comm, nccl_id, ndev, rank);
        if (bootstrap != ncclSuccess) {
            delete virtual_comm;
            return bootstrap;
        }
        virtual_comm->rank = rank;
        virtual_comm->ndev = ndev;
        int mpi_rank = -1, mpi_size = 0;
        MPI_Comm_rank(virtual_comm->mpi_comm, &mpi_rank);
        MPI_Comm_size(virtual_comm->mpi_comm, &mpi_size);
        if (mpi_rank != rank || mpi_size != ndev) { MPI_Comm_free(&virtual_comm->mpi_comm); delete virtual_comm; return ncclSystemError; }
        virtual_comm->next_send_sequence.assign(ndev, 0);
        virtual_comm->next_recv_sequence.assign(ndev, 0);
        {
            std::lock_guard<std::mutex> lock(runtime.communicator_mutex);
            runtime.communicators.insert(virtual_comm);
        }
        *comm = reinterpret_cast<ncclComm_t>(virtual_comm);
        return ncclSuccess;
    }

    static ncclResult_t groupStart() {
        if (group_state.depth == 0) group_state.operations.clear();
        ++group_state.depth;
        return ncclSuccess;
    }

    static bool debugEnabled() {
        static int enabled = std::getenv("NCCL_FOLD_DEBUG_P2P") != NULL;
        return enabled != 0;
    }

    static ncclResult_t cudaCheck(cudaError_t error, const char* operation) {
        if (error == cudaSuccess) return ncclSuccess;
        std::fprintf(stderr, "NCCL Fold: %s failed: %s\n", operation, cudaGetErrorString(error));
        return ncclUnhandledCudaError;
    }

    static ncclResult_t mpiCheck(int error, const char* operation) {
        if (error == MPI_SUCCESS) return ncclSuccess;
        std::fprintf(stderr, "NCCL Fold: %s failed with MPI error %d\n", operation, error);
        return ncclSystemError;
    }

    static bool resourceStatsEnabled() {
        static int enabled = std::getenv("NCCL_FOLD_RESOURCE_STATS") != NULL;
        return enabled != 0;
    }

    /* A slot stays leased from before it is exported until the control-plane
     * protocol proves that every importer has submitted its wait.  Failures do
     * not release slots: uncertain generations are conservatively quarantined
     * until communicator destruction. */
    static ncclResult_t leaseEvent(VirtualComm* comm, size_t* slot, cudaEvent_t* event) {
        for (size_t i = 0; i < comm->event_pool.size(); ++i) {
            if (!comm->event_pool[i].leased) {
                comm->event_pool[i].leased = true;
                *slot = i; *event = comm->event_pool[i].event;
                return ncclSuccess;
            }
        }
        cudaEvent_t created = NULL;
        ncclResult_t result = cudaCheck(runtime.cudaEventCreateWithFlags(
            &created, cudaEventInterprocess | cudaEventDisableTiming), "cudaEventCreateWithFlags(pool)");
        if (result != ncclSuccess) return result;
        comm->event_pool.push_back(EventSlot(created));
        comm->event_pool_peak = std::max(comm->event_pool_peak, comm->event_pool.size());
        *slot = comm->event_pool.size() - 1; *event = created;
        return ncclSuccess;
    }

    static void releaseEvent(VirtualComm* comm, size_t slot) {
        if (slot < comm->event_pool.size()) comm->event_pool[slot].leased = false;
    }

    static ncclResult_t openEvent(VirtualComm* comm, cudaIpcEventHandle_t const& handle,
                                  cudaEvent_t* event, const char* operation) {
        for (size_t i = 0; i < comm->imported_event_cache.size(); ++i) {
            if (std::memcmp(&comm->imported_event_cache[i].handle, &handle, sizeof(handle)) == 0) {
                *event = comm->imported_event_cache[i].event;
                return ncclSuccess;
            }
        }
        ncclResult_t result = cudaCheck(runtime.cudaIpcOpenEventHandle(event, handle), operation);
        if (result == ncclSuccess) comm->imported_event_cache.push_back({handle, *event});
        return result;
    }

    enum { readyTag = 17001, doneTag = 17002, ackTag = 17003 };

    static ncclResult_t openMemory(VirtualComm* comm, cudaIpcMemHandle_t const& handle, void** pointer) {
        for (size_t i = 0; i < comm->ipc_mappings.size(); ++i) {
            if (std::memcmp(&comm->ipc_mappings[i].handle, &handle, sizeof(handle)) == 0) {
                *pointer = comm->ipc_mappings[i].pointer;
                return ncclSuccess;
            }
        }
        ncclResult_t result = cudaCheck(runtime.cudaIpcOpenMemHandle(pointer, handle, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle");
        if (result == ncclSuccess) comm->ipc_mappings.push_back({handle, *pointer});
        return result;
    }

    /* MPI is only the control plane.  Ready slots are released after done
     * metadata proves the receiver submitted its ready wait.  Done slots use an
     * explicit acknowledgement after the sender submits its done wait. */
    static ncclResult_t enqueueP2P(VirtualComm* comm, const std::vector<sendRecvArgs_t>& send_args,
                                    const std::vector<sendRecvArgs_t>& recv_args) {
        const size_t send_count = send_args.size();
        const size_t recv_count = recv_args.size();
        std::vector<readyMessage> outgoing(send_count);
        std::vector<readyMessage> incoming(recv_count);
        std::vector<doneMessage> outgoing_done(recv_count);
        std::vector<doneMessage> incoming_done(send_count);
        std::vector<MPI_Request> requests(send_count + recv_count);
        std::vector<size_t> ready_slots(send_count), done_slots(recv_count);

        for (size_t i = 0; i < send_count; ++i) {
            sendRecvArgs_t const& op = send_args[i];
            readyMessage& message = outgoing[i];
            uint64_t type_size = sizeofNcclDataType(op.datatype);
            if (type_size == 0 || op.count > UINT64_MAX / type_size) return ncclInvalidArgument;
            void* allocation = getAllocation(op.buff, op.count * type_size, &message.memory.offset);
            if (allocation == NULL) return ncclInvalidArgument;
            ncclResult_t result = cudaCheck(runtime.cudaIpcGetMemHandle(&message.memory.handle, allocation), "cudaIpcGetMemHandle");
            if (result != ncclSuccess) return result;
            cudaEvent_t ready = NULL;
            result = leaseEvent(comm, &ready_slots[i], &ready);
            if (result != ncclSuccess) return result;
            result = cudaCheck(runtime.cudaEventRecord(ready, op.stream), "runtime.cudaEventRecord(ready)");
            if (result != ncclSuccess) return result;
            result = cudaCheck(runtime.cudaIpcGetEventHandle(&message.ready, ready), "runtime.cudaIpcGetEventHandle(ready)");
            if (result != ncclSuccess) return result;
            message.bytes = op.count * type_size;
            message.sequence = op.sequence;
            if (debugEnabled()) std::fprintf(stderr, "NCCL Fold comm=%p rank=%d peer=%d seq=%llu ready=%p stream=%p send\n",
                (void*)comm, comm->rank, op.peer, (unsigned long long)op.sequence, (void*)ready, (void*)op.stream);
        }
        for (size_t i = 0; i < send_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Isend(&outgoing[i], sizeof(readyMessage), MPI_BYTE,
                send_args[i].peer, readyTag, comm->mpi_comm, &requests[i]), "MPI_Isend(ready)");
            if (result != ncclSuccess) return result;
        }
        for (size_t i = 0; i < recv_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Irecv(&incoming[i], sizeof(incoming[i]), MPI_BYTE,
                recv_args[i].peer, readyTag, comm->mpi_comm, &requests[send_count + i]), "MPI_Irecv(ready)");
            if (result != ncclSuccess) return result;
        }
        if (!requests.empty()) {
            ncclResult_t result = mpiCheck(MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall(ready metadata)");
            if (result != ncclSuccess) return result;
        }

        requests.assign(send_count + recv_count, MPI_REQUEST_NULL);
        ncclResult_t deferred_error = ncclSuccess;
        for (size_t i = 0; i < recv_count; ++i) {
            sendRecvArgs_t const& op = recv_args[i];
            readyMessage const& message = incoming[i];
            uint64_t const type_size = sizeofNcclDataType(op.datatype);
            if (type_size == 0 || op.count > UINT64_MAX / type_size) return ncclInvalidArgument;
            uint64_t const recv_bytes = op.count * type_size;
            if (message.sequence != op.sequence || message.bytes != recv_bytes) deferred_error = ncclInvalidArgument;

            cudaEvent_t ready = NULL;
            cudaEvent_t done = NULL;
            void* source = NULL;
            ncclResult_t result = openEvent(comm, message.ready, &ready, "cudaIpcOpenEventHandle(ready)");
            if (result != ncclSuccess) return result;
            result = openMemory(comm, message.memory.handle, &source);
            if (result != ncclSuccess) return result;
            result = cudaCheck(runtime.cudaStreamWaitEvent(op.stream, ready, 0), "runtime.cudaStreamWaitEvent(ready)");
            if (result != ncclSuccess) return result;
            if (message.bytes == recv_bytes) {
                result = cudaCheck(runtime.cudaMemcpyAsync(op.buff, (char*)source + message.memory.offset, recv_bytes,
                    cudaMemcpyDeviceToDevice, op.stream), "runtime.cudaMemcpyAsync(P2P)");
                if (result != ncclSuccess) return result;
            }
            result = leaseEvent(comm, &done_slots[i], &done);
            if (result != ncclSuccess) return result;
            result = cudaCheck(runtime.cudaEventRecord(done, op.stream), "runtime.cudaEventRecord(done)");
            if (result != ncclSuccess) return result;
            result = cudaCheck(runtime.cudaIpcGetEventHandle(&outgoing_done[i].done, done), "runtime.cudaIpcGetEventHandle(done)");
            if (result != ncclSuccess) return result;
            outgoing_done[i].sequence = message.sequence;
            if (debugEnabled()) std::fprintf(stderr, "NCCL Fold comm=%p rank=%d peer=%d seq=%llu done=%p stream=%p recv\n",
                (void*)comm, comm->rank, op.peer, (unsigned long long)message.sequence, (void*)done, (void*)op.stream);
        }
        for (size_t i = 0; i < recv_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Isend(&outgoing_done[i], sizeof(doneMessage), MPI_BYTE,
                recv_args[i].peer, doneTag, comm->mpi_comm, &requests[send_count + i]), "MPI_Isend(done)");
            if (result != ncclSuccess) return result;
        }
        for (size_t i = 0; i < send_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Irecv(&incoming_done[i], sizeof(doneMessage), MPI_BYTE,
                send_args[i].peer, doneTag, comm->mpi_comm, &requests[i]), "MPI_Irecv(done)");
            if (result != ncclSuccess) return result;
        }
        if (!requests.empty()) {
            ncclResult_t result = mpiCheck(MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall(done metadata)");
            if (result != ncclSuccess) return result;
        }
        /* Matching done metadata implies that the receiver issued the ready
         * wait.  A mismatched/error generation remains quarantined. */
        for (size_t i = 0; i < send_count; ++i)
            if (incoming_done[i].sequence == send_args[i].sequence) releaseEvent(comm, ready_slots[i]);
        for (size_t i = 0; i < send_count; ++i) {
            if (incoming_done[i].sequence != send_args[i].sequence) deferred_error = ncclInvalidArgument;
            cudaEvent_t done = NULL;
            ncclResult_t result = openEvent(comm, incoming_done[i].done, &done, "cudaIpcOpenEventHandle(done)");
            if (result != ncclSuccess) return result;
            result = cudaCheck(runtime.cudaStreamWaitEvent(send_args[i].stream, done, 0), "runtime.cudaStreamWaitEvent(done)");
            if (result != ncclSuccess) return result;
        }
        /* Acknowledgements are sent only after all local waits above have been
         * submitted.  Receivers may then safely re-record their done slots. */
        requests.assign(send_count + recv_count, MPI_REQUEST_NULL);
        for (size_t i = 0; i < send_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Isend(&send_args[i].sequence, 1, MPI_UINT64_T,
                send_args[i].peer, ackTag, comm->mpi_comm, &requests[i]), "MPI_Isend(done ack)");
            if (result != ncclSuccess) return result;
        }
        std::vector<uint64_t> acknowledgements(recv_count);
        for (size_t i = 0; i < recv_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Irecv(&acknowledgements[i], 1, MPI_UINT64_T,
                recv_args[i].peer, ackTag, comm->mpi_comm, &requests[send_count + i]), "MPI_Irecv(done ack)");
            if (result != ncclSuccess) return result;
        }
        if (!requests.empty()) {
            ncclResult_t result = mpiCheck(MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall(done ack)");
            if (result != ncclSuccess) return result;
        }
        for (size_t i = 0; i < recv_count; ++i) {
            if (acknowledgements[i] != recv_args[i].sequence) deferred_error = ncclInvalidArgument;
            else releaseEvent(comm, done_slots[i]);
        }
        return deferred_error;
    }

    /* Grouped P2P metadata is exchanged as one control-plane batch before any
     * grouped operation is enqueued. Event records and data movement stay in
     * GroupOp order; send completion waits are safely deferred via snapshots. */
    struct PreparedGroupP2P {
        readyMessage ready;
        doneMessage done;
        cudaEvent_t local_event;
        size_t event_slot;
        void* send_snapshot;
        PreparedGroupP2P() : local_event(NULL), event_slot(0), send_snapshot(NULL) { std::memset(&ready, 0, sizeof(ready)); std::memset(&done, 0, sizeof(done)); }
    };

    static ncclResult_t prepareGroupedP2P(std::vector<GroupOp> const& operations, std::vector<PreparedGroupP2P>& prepared) {
        prepared.resize(operations.size());
        std::vector<MPI_Request> requests;
        requests.reserve(operations.size() * 2);
        for (size_t i = 0; i < operations.size(); ++i) {
            GroupOp const& grouped = operations[i];
            if (grouped.kind == GroupCollective) continue;
            sendRecvArgs_t const& op = grouped.p2p;
            PreparedGroupP2P& p = prepared[i];
            uint64_t const type_size = sizeofNcclDataType(op.datatype);
            if (type_size == 0 || op.count > UINT64_MAX / type_size) return ncclInvalidArgument;
            uint64_t const bytes = op.count * type_size;
            MPI_Request request = MPI_REQUEST_NULL;
            ncclResult_t result;
            if (grouped.kind == GroupSend) {
                if (getAllocation(op.buff, bytes, NULL) == NULL) return ncclInvalidArgument;
                if (bytes) {
                    if ((result = cudaCheck(runtime.origCudaMalloc(&p.send_snapshot, bytes), "cudaMalloc(group send snapshot)")) != ncclSuccess) return result;
                    grouped.comm->retained_ipc_allocations.push_back(p.send_snapshot);
                    if ((result = cudaCheck(runtime.cudaIpcGetMemHandle(&p.ready.memory.handle, p.send_snapshot), "cudaIpcGetMemHandle(group send snapshot)")) != ncclSuccess) return result;
                }
                if ((result = leaseEvent(grouped.comm, &p.event_slot, &p.local_event)) != ncclSuccess) return result;
                if ((result = cudaCheck(runtime.cudaIpcGetEventHandle(&p.ready.ready, p.local_event), "cudaIpcGetEventHandle(group send ready)")) != ncclSuccess) return result;
                p.ready.bytes = bytes; p.ready.sequence = op.sequence;
                if ((result = mpiCheck(MPI_Isend(&p.ready, sizeof(p.ready), MPI_BYTE, op.peer, readyTag, grouped.comm->mpi_comm, &request), "MPI_Isend(group ready)")) != ncclSuccess) return result;
                requests.push_back(request);
                if ((result = mpiCheck(MPI_Irecv(&p.done, sizeof(p.done), MPI_BYTE, op.peer, doneTag, grouped.comm->mpi_comm, &request), "MPI_Irecv(group done)")) != ncclSuccess) return result;
                requests.push_back(request);
            } else {
                if (bytes && getAllocation(op.buff, bytes, NULL) == NULL) return ncclInvalidArgument;
                if ((result = leaseEvent(grouped.comm, &p.event_slot, &p.local_event)) != ncclSuccess) return result;
                if ((result = cudaCheck(runtime.cudaIpcGetEventHandle(&p.done.done, p.local_event), "cudaIpcGetEventHandle(group recv done)")) != ncclSuccess) return result;
                p.done.sequence = op.sequence;
                if ((result = mpiCheck(MPI_Irecv(&p.ready, sizeof(p.ready), MPI_BYTE, op.peer, readyTag, grouped.comm->mpi_comm, &request), "MPI_Irecv(group ready)")) != ncclSuccess) return result;
                requests.push_back(request);
                if ((result = mpiCheck(MPI_Isend(&p.done, sizeof(p.done), MPI_BYTE, op.peer, doneTag, grouped.comm->mpi_comm, &request), "MPI_Isend(group done)")) != ncclSuccess) return result;
                requests.push_back(request);
            }
        }
        if (requests.empty()) return ncclSuccess;
        return mpiCheck(MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall(group P2P metadata)");
    }

    static ncclResult_t enqueuePreparedGroupP2P(GroupOp const& grouped, PreparedGroupP2P const& p) {
        sendRecvArgs_t const& op = grouped.p2p;
        ncclResult_t result;
        if (grouped.kind == GroupSend) {
            if (p.done.sequence != op.sequence) return ncclInvalidArgument;
            uint64_t const bytes = op.count * sizeofNcclDataType(op.datatype);
            if (bytes && (result = cudaCheck(runtime.cudaMemcpyAsync(p.send_snapshot, op.buff, bytes, cudaMemcpyDeviceToDevice, op.stream), "cudaMemcpyAsync(group send snapshot)")) != ncclSuccess) return result;
            if ((result = cudaCheck(runtime.cudaEventRecord(p.local_event, op.stream), "cudaEventRecord(group send ready)")) != ncclSuccess) return result;
            return ncclSuccess;
        }
        uint64_t const type_size = sizeofNcclDataType(op.datatype);
        uint64_t const bytes = op.count * type_size;
        if (p.ready.sequence != op.sequence || p.ready.bytes != bytes) return ncclInvalidArgument;
        cudaEvent_t ready = NULL;
        void* source = NULL;
        if ((result = openEvent(grouped.comm, p.ready.ready, &ready, "cudaIpcOpenEventHandle(group ready)")) != ncclSuccess) return result;
        if (bytes && (result = openMemory(grouped.comm, p.ready.memory.handle, &source)) != ncclSuccess) return result;
        if ((result = cudaCheck(runtime.cudaStreamWaitEvent(op.stream, ready, 0), "cudaStreamWaitEvent(group ready)")) != ncclSuccess) return result;
        if (bytes && (result = cudaCheck(runtime.cudaMemcpyAsync(op.buff, (char*)source + p.ready.memory.offset, bytes, cudaMemcpyDeviceToDevice, op.stream), "cudaMemcpyAsync(group P2P)")) != ncclSuccess) return result;
        return cudaCheck(runtime.cudaEventRecord(p.local_event, op.stream), "cudaEventRecord(group recv done)");
    }

    static ncclResult_t enqueuePreparedGroupSendCompletion(GroupOp const& grouped, PreparedGroupP2P const& p) {
        if (grouped.kind != GroupSend) return ncclSuccess;
        cudaEvent_t done = NULL;
        ncclResult_t result = openEvent(grouped.comm, p.done.done, &done, "cudaIpcOpenEventHandle(group done)");
        if (result != ncclSuccess) return result;
        return cudaCheck(runtime.cudaStreamWaitEvent(grouped.p2p.stream, done, 0), "cudaStreamWaitEvent(group done)");
    }

    struct CollectiveDescriptor {
        handleOffset memory;
        cudaIpcEventHandle_t ready;
        uint64_t bytes;
        uint64_t sequence;
        int kind;
        int datatype;
        int op;
        int root;
    };

    template <typename T> __device__ T reduceValue(T a, T b, int op) {
        if (op == ncclSum) return a + b;
        if (op == ncclProd) return a * b;
        if (op == ncclMin) return a < b ? a : b;
        return a > b ? a : b;
    }
    template <> __device__ __half reduceValue(__half a, __half b, int op) {
        float x = __half2float(a), y = __half2float(b);
        float z = op == ncclSum ? x+y : op == ncclProd ? x*y : op == ncclMin ? fminf(x,y) : fmaxf(x,y);
        return __float2half(z);
    }
    template <> __device__ __nv_bfloat16 reduceValue(__nv_bfloat16 a, __nv_bfloat16 b, int op) {
        float x = __bfloat162float(a), y = __bfloat162float(b);
        float z = op == ncclSum ? x+y : op == ncclProd ? x*y : op == ncclMin ? fminf(x,y) : fmaxf(x,y);
        return __float2bfloat16(z);
    }
    template <typename T> __global__ void reductionKernel(const void* const* sources, T* output,
                                                            size_t count, size_t source_offset,
                                                            int nranks, int op) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        T value = static_cast<const T*>(sources[0])[source_offset + i];
        for (int rank = 1; rank < nranks; ++rank)
            value = reduceValue(value, static_cast<const T*>(sources[rank])[source_offset + i], op);
        output[i] = value;
    }

    static bool validReduction(ncclRedOp_t op) {
        return op == ncclSum || op == ncclProd || op == ncclMin || op == ncclMax;
    }

    template <typename T> static ncclResult_t launchReduction(void** device_sources, void* output,
            size_t count, size_t source_offset, int nranks, ncclRedOp_t op, cudaStream_t stream) {
        if (count != 0) reductionKernel<T><<<(count + 255) / 256, 256, 0, stream>>>(
            (const void* const*)device_sources, (T*)output, count, source_offset, nranks, (int)op);
        return cudaCheck(cudaGetLastError(), "reduction kernel launch");
    }

    static ncclResult_t enqueueCollective(VirtualComm* comm, const CollectiveArgs& args) {
        const uint64_t element_size = sizeofNcclDataType(args.datatype);
        const bool reduction = args.kind == Reduce || args.kind == AllReduce || args.kind == ReduceScatter;
        if (!element_size || (reduction && !validReduction(args.op))) return ncclInvalidArgument;
        if ((args.kind == Broadcast || args.kind == Reduce) && (args.root < 0 || args.root >= comm->ndev)) return ncclInvalidArgument;
        size_t source_count = args.kind == ReduceScatter ? args.count * (size_t)comm->ndev : args.count;
        bool produces_output = args.kind != Reduce || comm->rank == args.root;
        bool has_source = args.kind != Broadcast || comm->rank == args.root;
        if (source_count && ((has_source && !args.sendbuff) || (produces_output && !args.recvbuff))) return ncclInvalidArgument;
        if (source_count > SIZE_MAX / element_size) return ncclInvalidArgument;
        size_t source_bytes = source_count * element_size;

        CollectiveDescriptor local = {};
        local.bytes = source_bytes; local.sequence = comm->collective_sequence++;
        local.kind = args.kind; local.datatype = args.datatype; local.op = reduction ? args.op : 0; local.root = args.root;
        void* allocation = has_source ? getAllocation(const_cast<void*>(args.sendbuff), source_bytes, &local.memory.offset) : NULL;
        if (source_bytes && has_source && !allocation) return ncclInvalidArgument;
        if (args.kind == AllGather && source_bytes > SIZE_MAX / (size_t)comm->ndev) return ncclInvalidArgument;
        size_t output_bytes = args.kind == AllGather ? source_bytes * comm->ndev : args.count * element_size;
        if (produces_output && output_bytes && !getAllocation(args.recvbuff, output_bytes, NULL)) return ncclInvalidArgument;
        if (source_bytes && has_source) {
            ncclResult_t result = cudaCheck(runtime.cudaIpcGetMemHandle(&local.memory.handle, allocation), "cudaIpcGetMemHandle(collective)");
            if (result != ncclSuccess) return result;
        }
        cudaEvent_t ready = NULL;
        size_t ready_slot = 0;
        ncclResult_t result = leaseEvent(comm, &ready_slot, &ready);
        if (result != ncclSuccess) return result;
        if ((result = cudaCheck(runtime.cudaEventRecord(ready, args.stream), "cudaEventRecord(collective ready)")) != ncclSuccess) return result;
        if ((result = cudaCheck(runtime.cudaIpcGetEventHandle(&local.ready, ready), "cudaIpcGetEventHandle(collective ready)")) != ncclSuccess) return result;

        std::vector<CollectiveDescriptor> descriptors(comm->ndev);
        if ((result = mpiCheck(MPI_Allgather(&local, sizeof(local), MPI_BYTE, descriptors.data(), sizeof(local), MPI_BYTE, comm->mpi_comm), "MPI_Allgather(collective metadata)")) != ncclSuccess) return result;
        std::vector<void*> sources(comm->ndev);
        for (int rank = 0; rank < comm->ndev; ++rank) {
            const CollectiveDescriptor& d = descriptors[rank];
            if (d.sequence != local.sequence || d.kind != local.kind || d.datatype != local.datatype ||
                d.op != local.op || d.root != local.root || d.bytes != local.bytes) return ncclInvalidUsage;
            bool need_rank = args.kind != Broadcast || rank == args.root;
            if (!need_rank) continue;
            if (rank == comm->rank) sources[rank] = const_cast<void*>(args.sendbuff);
            else if (source_bytes) {
                cudaEvent_t event = NULL; void* base = NULL;
                if ((result = openEvent(comm, d.ready, &event, "cudaIpcOpenEventHandle(collective ready)")) != ncclSuccess) return result;
                if ((result = cudaCheck(runtime.cudaStreamWaitEvent(args.stream, event, 0), "cudaStreamWaitEvent(collective ready)")) != ncclSuccess) return result;
                if ((result = openMemory(comm, d.memory.handle, &base)) != ncclSuccess) return result;
                sources[rank] = (char*)base + d.memory.offset;
            }
        }
        if ((result = mpiCheck(MPI_Barrier(comm->mpi_comm), "MPI_Barrier(collective ready leases)")) != ncclSuccess) return result;
        releaseEvent(comm, ready_slot);

        if (args.kind == Broadcast) {
            if (source_bytes && args.recvbuff != sources[args.root])
                result = cudaCheck(runtime.cudaMemcpyAsync(args.recvbuff, sources[args.root], source_bytes, cudaMemcpyDeviceToDevice, args.stream), "cudaMemcpyAsync(Broadcast)");
        } else if (args.kind == AllGather) {
            for (int rank = 0; rank < comm->ndev && result == ncclSuccess; ++rank) {
                void* destination = (char*)args.recvbuff + rank * source_bytes;
                if (source_bytes && destination != sources[rank]) result = cudaCheck(runtime.cudaMemcpyAsync(destination, sources[rank], source_bytes, cudaMemcpyDeviceToDevice, args.stream), "cudaMemcpyAsync(AllGather)");
            }
        } else if (args.kind != Reduce || comm->rank == args.root) {
            void** device_sources = NULL;
            if ((result = cudaCheck(runtime.origCudaMallocAsync((void**)&device_sources, sizeof(void*) * comm->ndev, args.stream), "cudaMallocAsync(reduction sources)")) != ncclSuccess) return result;
            ++comm->scratch_in_flight;
            ++comm->scratch_allocations_submitted;
            comm->scratch_high_water = std::max(comm->scratch_high_water, comm->scratch_in_flight);
            result = cudaCheck(runtime.cudaMemcpyAsync(device_sources, sources.data(), sizeof(void*) * comm->ndev, cudaMemcpyHostToDevice, args.stream), "cudaMemcpyAsync(reduction sources)");
            if (result != ncclSuccess) {
                if (runtime.origCudaFreeAsync(device_sources, args.stream) == cudaSuccess) --comm->scratch_in_flight;
                return result;
            }
            size_t offset = args.kind == ReduceScatter ? args.count * (size_t)comm->rank : 0;
            switch (args.datatype) {
                case ncclInt8: result=launchReduction<int8_t>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                case ncclUint8: result=launchReduction<uint8_t>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                case ncclInt32: result=launchReduction<int32_t>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                case ncclUint32: result=launchReduction<uint32_t>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                case ncclInt64: result=launchReduction<int64_t>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                case ncclUint64: result=launchReduction<uint64_t>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                case ncclFloat16: result=launchReduction<__half>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                case ncclFloat32: result=launchReduction<float>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                case ncclFloat64: result=launchReduction<double>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                case ncclBfloat16: result=launchReduction<__nv_bfloat16>(device_sources,args.recvbuff,args.count,offset,comm->ndev,args.op,args.stream); break;
                default: result=ncclInvalidArgument;
            }
            cudaError_t free_error = runtime.origCudaFreeAsync(device_sources, args.stream);
            if (free_error == cudaSuccess) --comm->scratch_in_flight;
            if (result == ncclSuccess) result = cudaCheck(free_error, "cudaFreeAsync(reduction sources)");
        }
        if (result != ncclSuccess) return result;

        /* Host metadata exchange is blocking, but GPU execution remains asynchronous.
         * Done events make later work on every source stream wait until all remote
         * readers have enqueued their use, matching NCCL's buffer-reuse ordering. */
        cudaEvent_t done = NULL;
        size_t done_slot = 0;
        if ((result = leaseEvent(comm, &done_slot, &done)) != ncclSuccess) return result;
        if ((result = cudaCheck(runtime.cudaEventRecord(done, args.stream), "cudaEventRecord(collective done)")) != ncclSuccess) return result;
        cudaIpcEventHandle_t local_done;
        if ((result = cudaCheck(runtime.cudaIpcGetEventHandle(&local_done, done), "cudaIpcGetEventHandle(collective done)")) != ncclSuccess) return result;
        std::vector<cudaIpcEventHandle_t> dones(comm->ndev);
        if ((result = mpiCheck(MPI_Allgather(&local_done, sizeof(local_done), MPI_BYTE, dones.data(), sizeof(local_done), MPI_BYTE, comm->mpi_comm), "MPI_Allgather(collective done)")) != ncclSuccess) return result;
        for (int rank=0; rank<comm->ndev; ++rank) if (rank != comm->rank) {
            cudaEvent_t event = NULL;
            if ((result = openEvent(comm, dones[rank], &event, "cudaIpcOpenEventHandle(collective done)")) != ncclSuccess) return result;
            if ((result = cudaCheck(runtime.cudaStreamWaitEvent(args.stream, event, 0), "cudaStreamWaitEvent(collective done)")) != ncclSuccess) return result;
        }
        if ((result = mpiCheck(MPI_Barrier(comm->mpi_comm), "MPI_Barrier(collective done leases)")) != ncclSuccess) return result;
        releaseEvent(comm, done_slot);
        return ncclSuccess;
    }

    static ncclResult_t groupEnd() {
        if (group_state.depth == 0) return ncclInvalidUsage;
        if (--group_state.depth != 0) return ncclSuccess;

        /* Detach first so every error path leaves this thread's group reusable. */
        std::vector<GroupOp> operations;
        operations.swap(group_state.operations);
        for (size_t i = 0; i < operations.size(); ++i)
            if (getVirtualComm(reinterpret_cast<ncclComm_t>(operations[i].comm)) == NULL)
                return ncclInvalidArgument;

        std::vector<PreparedGroupP2P> prepared;
        ncclResult_t result = prepareGroupedP2P(operations, prepared);
        if (result != ncclSuccess) return result;
        for (size_t i = 0; i < operations.size(); ++i) {
            GroupOp const& op = operations[i];
            result = op.kind == GroupCollective
                ? enqueueCollective(op.comm, op.collective)
                : enqueuePreparedGroupP2P(op, prepared[i]);
            if (result != ncclSuccess) return result;
        }
        /* Snapshots make grouped sends consume their user buffers at the send's
         * exact stream position.  Completion waits can therefore be appended
         * after all group work, avoiding Send/collective/Recv dependency cycles. */
        for (size_t i = 0; i < operations.size(); ++i) {
            result = enqueuePreparedGroupSendCompletion(operations[i], prepared[i]);
            if (result != ncclSuccess) return result;
        }
        /* Both endpoints acknowledge every grouped P2P operation after their
         * ready/done waits have been submitted.  Only then can either endpoint
         * re-record its local event generation. */
        std::vector<MPI_Request> acknowledgements;
        std::vector<uint64_t> received(operations.size());
        acknowledgements.reserve(operations.size() * 2);
        for (size_t i = 0; i < operations.size(); ++i) {
            if (operations[i].kind == GroupCollective) continue;
            MPI_Request request = MPI_REQUEST_NULL;
            GroupOp const& op = operations[i];
            if ((result = mpiCheck(MPI_Isend(&op.p2p.sequence, 1, MPI_UINT64_T, op.p2p.peer,
                    ackTag, op.comm->mpi_comm, &request), "MPI_Isend(group event ack)")) != ncclSuccess) return result;
            acknowledgements.push_back(request);
            if ((result = mpiCheck(MPI_Irecv(&received[i], 1, MPI_UINT64_T, op.p2p.peer,
                    ackTag, op.comm->mpi_comm, &request), "MPI_Irecv(group event ack)")) != ncclSuccess) return result;
            acknowledgements.push_back(request);
        }
        if (!acknowledgements.empty() && (result = mpiCheck(MPI_Waitall((int)acknowledgements.size(),
                acknowledgements.data(), MPI_STATUSES_IGNORE), "MPI_Waitall(group event ack)")) != ncclSuccess) return result;
        for (size_t i = 0; i < operations.size(); ++i) if (operations[i].kind != GroupCollective) {
            if (received[i] != operations[i].p2p.sequence) return ncclInvalidArgument;
            releaseEvent(operations[i].comm, prepared[i].event_slot);
        }
        return ncclSuccess;
    }

    static ncclResult_t collective(CollectiveKind kind, const void* sendbuff, void* recvbuff,
            size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root,
            ncclComm_t handle, cudaStream_t stream) {
        VirtualComm* comm = getVirtualComm(handle);
        if (!comm) return ncclInvalidArgument;
        CollectiveArgs args = {kind, sendbuff, recvbuff, count, datatype, op, root, stream};
        if (group_state.depth != 0) {
            GroupOp grouped = {};
            grouped.kind = GroupCollective; grouped.comm = comm; grouped.collective = args;
            group_state.operations.push_back(grouped);
            return ncclSuccess;
        }
        return enqueueCollective(comm, args);
    }

    static ncclResult_t broadcast(const void* sendbuff, void* recvbuff, size_t count,
            ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
        return collective(Broadcast, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream);
    }
    static ncclResult_t bcast(void* buff, size_t count, ncclDataType_t datatype, int root,
            ncclComm_t comm, cudaStream_t stream) {
        return broadcast(buff, buff, count, datatype, root, comm, stream);
    }
    static ncclResult_t allGather(const void* sendbuff, void* recvbuff, size_t count,
            ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
        return collective(AllGather, sendbuff, recvbuff, count, datatype, ncclSum, -1, comm, stream);
    }
    static ncclResult_t reduce(const void* sendbuff, void* recvbuff, size_t count,
            ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
        return collective(Reduce, sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    }
    static ncclResult_t allReduce(const void* sendbuff, void* recvbuff, size_t count,
            ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
        return collective(AllReduce, sendbuff, recvbuff, count, datatype, op, -1, comm, stream);
    }
    static ncclResult_t reduceScatter(const void* sendbuff, void* recvbuff, size_t count,
            ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
        return collective(ReduceScatter, sendbuff, recvbuff, count, datatype, op, -1, comm, stream);
    }

    static ncclResult_t send(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
                             ncclComm_t handle, cudaStream_t stream) {
        VirtualComm* comm = getVirtualComm(handle);
        if (comm == NULL) return ncclInvalidArgument;
        if (peer < 0 || peer >= comm->ndev) return ncclInvalidArgument;
        uint64_t sequence;
        {
            std::lock_guard<std::mutex> lock(comm->sequence_mutex);
            sequence = comm->next_send_sequence[peer]++;
        }
        sendRecvArgs_t op = {const_cast<void*>(sendbuff), (uint64_t)count, (int)datatype, peer, stream, sequence};
        if (group_state.depth != 0) {
            GroupOp grouped = {};
            grouped.kind = GroupSend; grouped.comm = comm; grouped.p2p = op;
            group_state.operations.push_back(grouped);
            return ncclSuccess;
        }
        std::vector<sendRecvArgs_t> sends(1, op), recvs;
        return enqueueP2P(comm, sends, recvs);
    }

    static ncclResult_t recv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
                             ncclComm_t handle, cudaStream_t stream) {
        VirtualComm* comm = getVirtualComm(handle);
        if (comm == NULL) return ncclInvalidArgument;
        if (peer < 0 || peer >= comm->ndev) return ncclInvalidArgument;
        uint64_t sequence;
        {
            std::lock_guard<std::mutex> lock(comm->sequence_mutex);
            sequence = comm->next_recv_sequence[peer]++;
        }
        sendRecvArgs_t op = {recvbuff, (uint64_t)count, (int)datatype, peer, stream, sequence};
        if (group_state.depth != 0) {
            GroupOp grouped = {};
            grouped.kind = GroupRecv; grouped.comm = comm; grouped.p2p = op;
            group_state.operations.push_back(grouped);
            return ncclSuccess;
        }
        std::vector<sendRecvArgs_t> sends, recvs(1, op);
        return enqueueP2P(comm, sends, recvs);
    }

    static ncclResult_t commCount(ncclComm_t handle, int* count) {
        VirtualComm* comm = getVirtualComm(handle);
        if (comm == NULL || count == NULL) return ncclInvalidArgument;
        *count = comm->ndev;
        return ncclSuccess;
    }

    static ncclResult_t commUserRank(ncclComm_t handle, int* rank) {
        VirtualComm* comm = getVirtualComm(handle);
        if (comm == NULL || rank == NULL) return ncclInvalidArgument;
        *rank = comm->rank;
        return ncclSuccess;
    }

    static ncclResult_t commDestroy(ncclComm_t handle) {
        if (handle == NULL) return ncclInvalidArgument;
        VirtualComm* comm = reinterpret_cast<VirtualComm*>(handle);
        {
            std::lock_guard<std::mutex> lock(runtime.communicator_mutex);
            std::unordered_set<VirtualComm*>::iterator found = runtime.communicators.find(comm);
            if (found == runtime.communicators.end() || comm->magic != VirtualComm::MAGIC)
                return ncclInvalidArgument;
            runtime.communicators.erase(found);
            comm->magic = 0;
        }

        ncclResult_t result = ncclSuccess;
        if (resourceStatsEnabled())
            std::fprintf(stderr, "NCCL Fold resources comm=%p rank=%d owned_events_created=%zu "
                "event_pool_peak=%zu imported_events_opened=%zu imported_event_cache=%zu "
                "ipc_memory_mappings=%zu reduction_scratch_allocations=%zu "
                "reduction_scratch_high_water=%zu retained_ipc_allocations=%zu\n",
                (void*)comm, comm->rank, comm->event_pool.size(), comm->event_pool_peak,
                comm->imported_event_cache.size(), comm->imported_event_cache.size(),
                comm->ipc_mappings.size(), comm->scratch_allocations_submitted,
                comm->scratch_high_water,
                comm->retained_ipc_allocations.size());
        for (size_t i = 0; i < comm->event_pool.size(); ++i)
            if (runtime.cudaEventDestroy(comm->event_pool[i].event) != cudaSuccess) result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->imported_event_cache.size(); ++i)
            if (runtime.cudaEventDestroy(comm->imported_event_cache[i].event) != cudaSuccess) result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->ipc_mappings.size(); ++i)
            if (runtime.cudaIpcCloseMemHandle(comm->ipc_mappings[i].pointer) != cudaSuccess) result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->retained_ipc_allocations.size(); ++i)
            if (runtime.origCudaFree(comm->retained_ipc_allocations[i]) != cudaSuccess) result = ncclUnhandledCudaError;
        if (comm->mpi_comm != MPI_COMM_NULL && MPI_Comm_free(&comm->mpi_comm) != MPI_SUCCESS && result == ncclSuccess)
            result = ncclSystemError;
        delete comm;
        return result;
    }

    static void* find_symbol_offset(const char* exe_path, const char* symbol_name) {
        int fd = open(exe_path, O_RDONLY);
        elf_version(EV_CURRENT);
        Elf* e = elf_begin(fd, ELF_C_READ, NULL);

        size_t shstrndx;
        elf_getshdrstrndx(e, &shstrndx);
        
        Elf_Scn* scn = NULL;
        GElf_Shdr shdr;

        while ((scn = elf_nextscn(e, scn)) != NULL) {
            gelf_getshdr(scn, &shdr);
            if (shdr.sh_type == SHT_SYMTAB) {
                Elf_Data* data = elf_getdata(scn, NULL);
                int count = shdr.sh_size / shdr.sh_entsize;

                for (int i = 0; i < count; ++i) {
                    GElf_Sym sym;
                    gelf_getsym(data, i, &sym);
                    const char* name = elf_strptr(e, shdr.sh_link, sym.st_name);
                    if (strcmp(name, symbol_name) == 0) {
                        return (void*)sym.st_value;
                    }
                }
            }
        }
        return 0;
    } 

    static void* find_symbol_offset_or_dlsym(char const* exe_path, const char* symbol_name) {
        void* addr = find_symbol_offset(exe_path, symbol_name);
        if (addr) { return addr; }
        return dlsym(NULL, symbol_name);
    }

    __attribute__((constructor))
    void init() {

        gum_init_embedded();
        interceptor = gum_interceptor_obtain();
        gum_interceptor_begin_transaction(interceptor);

        runtime.cudaEventDestroy = &::cudaEventDestroy;
        runtime.cudaIpcCloseMemHandle = &::cudaIpcCloseMemHandle;
        runtime.cudaIpcGetMemHandle = &::cudaIpcGetMemHandle;
        runtime.cudaIpcOpenMemHandle = &::cudaIpcOpenMemHandle;
        runtime.cudaMemcpyAsync = &::cudaMemcpyAsync;
        runtime.cudaEventCreateWithFlags = &::cudaEventCreateWithFlags;
        runtime.cudaEventRecord = &::cudaEventRecord;
        runtime.cudaIpcGetEventHandle = &::cudaIpcGetEventHandle;
        runtime.cudaIpcOpenEventHandle = &::cudaIpcOpenEventHandle;
        runtime.cudaStreamWaitEvent = &::cudaStreamWaitEvent;

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaMalloc"), (gpointer)nccl_fold::cudaMalloc, NULL, (gpointer*)&runtime.origCudaMalloc);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaMallocAsync"), (gpointer)nccl_fold::cudaMallocAsync, NULL, (gpointer*)&runtime.origCudaMallocAsync);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaFree"), (gpointer)nccl_fold::cudaFree, NULL, (gpointer*)&runtime.origCudaFree);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaFreeAsync"), (gpointer)nccl_fold::cudaFreeAsync, NULL, (gpointer*)&runtime.origCudaFreeAsync);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaSetDevice"), (gpointer)nccl_fold::cudaSetDevice, NULL, (gpointer*)&runtime.origCudaSetDevice);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclGetUniqueId, (gpointer)nccl_fold::getUniqueId, NULL, (gpointer*)&runtime.origNcclGetUniqueId);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclCommInitRank, (gpointer)nccl_fold::commInitRank, NULL, (gpointer*)&runtime.origNcclCommInitRank);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclGroupStart, (gpointer)nccl_fold::groupStart, NULL, (gpointer*)&runtime.origNcclGroupStart);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclGroupEnd, (gpointer)nccl_fold::groupEnd, NULL, (gpointer*)&runtime.origNcclGroupEnd);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclSend, (gpointer)nccl_fold::send, NULL, (gpointer*)&runtime.origNcclSend);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclRecv, (gpointer)nccl_fold::recv, NULL, (gpointer*)&runtime.origNcclRecv);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclBroadcast, (gpointer)nccl_fold::broadcast, NULL, (gpointer*)&runtime.origNcclBroadcast);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclBcast, (gpointer)nccl_fold::bcast, NULL, (gpointer*)&runtime.origNcclBcast);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclAllGather, (gpointer)nccl_fold::allGather, NULL, (gpointer*)&runtime.origNcclAllGather);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclReduce, (gpointer)nccl_fold::reduce, NULL, (gpointer*)&runtime.origNcclReduce);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclAllReduce, (gpointer)nccl_fold::allReduce, NULL, (gpointer*)&runtime.origNcclAllReduce);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclReduceScatter, (gpointer)nccl_fold::reduceScatter, NULL, (gpointer*)&runtime.origNcclReduceScatter);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclCommDestroy, (gpointer)nccl_fold::commDestroy, NULL, (gpointer*)&runtime.origNcclCommDestroy);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclCommCount, (gpointer)nccl_fold::commCount, NULL, (gpointer*)&runtime.origNcclCommCount);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclCommUserRank, (gpointer)nccl_fold::commUserRank, NULL, (gpointer*)&runtime.origNcclCommUserRank);

        gum_interceptor_end_transaction(interceptor);
    }

    __attribute__((destructor))
    void deinit() {
        gum_interceptor_begin_transaction(interceptor);
        // gum_interceptor_revert(interceptor, (void*)(func));  /* specify original address */
        gum_interceptor_end_transaction(interceptor);
    
        g_object_unref(interceptor);
        gum_deinit_embedded();
    }

}
