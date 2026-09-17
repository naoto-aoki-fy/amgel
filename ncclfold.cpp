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
#include <thread>
#include <poll.h>

#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>

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

namespace nccl_fold
{

#if CUDART_VERSION < 11030
#error "NCCL Fold requires CUDA 11.3 or newer for memory-pool IPC"
#endif

    static GumInterceptor *interceptor = NULL;

    enum AllocationKind : uint32_t
    {
        LegacyAllocation = 1,
        PoolAllocation = 2
    };

    typedef struct
    {
        uint32_t kind;
        uint32_t reserved;
        union
        {
            cudaIpcMemHandle_t legacy;
            cudaMemPoolPtrExportData pool;
        } handle;
        uint64_t offset;
    } memoryDescriptor;

    typedef struct
    {
        memoryDescriptor memory;
        cudaIpcEventHandle_t ready;
        uint64_t bytes;
        uint64_t count;
        uint64_t sequence;
        int datatype;
    } readyMessage;

    typedef struct
    {
        cudaIpcEventHandle_t done;
        uint64_t count;
        uint64_t sequence;
        int datatype;
        int status;
    } doneMessage;

    typedef struct
    {
        cudaIpcMemHandle_t handle;
        void *pointer;
    } importedMemory;

    typedef struct
    {
        void *buff;
        uint64_t count;
        int datatype;
        int peer;
        cudaStream_t stream;
        uint64_t sequence;
    } sendRecvArgs_t;

    struct EventSlot
    {
        cudaEvent_t event;
        bool leased;
        EventSlot(cudaEvent_t event_) : event(event_), leased(true) {}
    };

    struct ImportedEvent
    {
        cudaIpcEventHandle_t handle;
        cudaEvent_t event;
    };

    enum CollectiveKind
    {
        Broadcast,
        AllGather,
        Reduce,
        AllReduce,
        ReduceScatter,
        AlltoAll,
        Gather,
        Scatter
    };
    struct CollectiveArgs
    {
        CollectiveKind kind;
        const void *sendbuff;
        void *recvbuff;
        size_t count;
        ncclDataType_t datatype;
        ncclRedOp_t op;
        int root;
        cudaStream_t stream;
    };

    struct VirtualComm
    {
        static constexpr uint64_t MAGIC = UINT64_C(0x4e43434c464f4c44);
        uint64_t magic;
        ncclUniqueId unique_id;
        std::vector<EventSlot> event_pool;
        std::vector<ImportedEvent> imported_event_cache;
        std::vector<importedMemory> ipc_mappings;
        std::vector<cudaMemPool_t> imported_pools;
        /* IPC-exported grouped-send snapshots cannot be reclaimed while a
         * remote cached mapping may remain open.  They are not reduction scratch. */
        std::vector<void *> retained_ipc_allocations;
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
        uint64_t fingerprint;
        std::string diagnostic_directory;
        std::mutex sequence_mutex;

        VirtualComm() : magic(MAGIC), event_pool_peak(0), scratch_in_flight(0),
                        scratch_high_water(0), scratch_allocations_submitted(0), collective_sequence(0),
                        mpi_comm(MPI_COMM_NULL), rank(-1), ndev(0), fingerprint(0) {}
    };

    struct RuntimeState
    {
        struct Allocation
        {
            size_t size;
            AllocationKind kind;
        };
        std::map<uintptr_t, Allocation> allocations;
        std::mutex pointer_mutex;
        cudaMemPool_t export_pool;
        cudaError_t pool_status;
        std::once_flag pool_once;
        std::unordered_set<VirtualComm *> communicators;
        std::mutex communicator_mutex;
        decltype(&::cudaMalloc<void>) origCudaMalloc;
        decltype(&::cudaFree) origCudaFree;
        cudaError_t (*origCudaMallocAsync)(void **, size_t, cudaStream_t);
        cudaError_t (*origCudaFreeAsync)(void *, cudaStream_t);
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
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
        decltype(&::ncclAlltoAll) origNcclAlltoAll;
        decltype(&::ncclGather) origNcclGather;
        decltype(&::ncclScatter) origNcclScatter;
#endif
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
        RuntimeState() : export_pool(NULL), pool_status(cudaErrorNotSupported) {}
    };

    static RuntimeState runtime;

    enum GroupOpKind
    {
        GroupSend,
        GroupRecv,
        GroupCollective
    };
    struct GroupOp
    {
        GroupOpKind kind;
        VirtualComm *comm;
        sendRecvArgs_t p2p;
        CollectiveArgs collective;
    };
    struct GroupState
    {
        unsigned int depth = 0;
        std::vector<GroupOp> operations;
    };
    static thread_local GroupState group_state;

    static VirtualComm *getVirtualComm(ncclComm_t handle)
    {
        if (handle == NULL)
            return NULL;
        VirtualComm *comm = reinterpret_cast<VirtualComm *>(handle);
        std::lock_guard<std::mutex> lock(runtime.communicator_mutex);
        if (runtime.communicators.count(comm) == 0 || comm->magic != VirtualComm::MAGIC)
            return NULL;
        return comm;
    }

    static void initializeExportPool()
    {
        int pools = 0, handles = 0;
        cudaError_t error = cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, 0);
        if (error == cudaSuccess)
            error = cudaDeviceGetAttribute(&handles, cudaDevAttrMemoryPoolSupportedHandleTypes, 0);
        if (error != cudaSuccess || !pools || !(handles & cudaMemHandleTypePosixFileDescriptor))
        {
            runtime.pool_status = error == cudaSuccess ? cudaErrorNotSupported : error;
            return;
        }
        cudaMemPoolProps properties = {};
        properties.allocType = cudaMemAllocationTypePinned;
        properties.handleTypes = cudaMemHandleTypePosixFileDescriptor;
        properties.location.type = cudaMemLocationTypeDevice;
        properties.location.id = 0;
        runtime.pool_status = cudaMemPoolCreate(&runtime.export_pool, &properties);
        /* The exporting pool deliberately has process lifetime.  Destroying it
         * from a library destructor is unsafe during CUDA runtime teardown. */
    }

    static cudaError_t getExportPool(cudaMemPool_t *pool)
    {
        std::call_once(runtime.pool_once, initializeExportPool);
        if (runtime.pool_status == cudaSuccess)
            *pool = runtime.export_pool;
        return runtime.pool_status;
    }

    static cudaError_t cudaMalloc(void **devPtr, size_t size)
    {
        cudaError_t const ret = runtime.origCudaMalloc(devPtr, size);
        if (ret == cudaSuccess)
        {
            std::lock_guard<std::mutex> lock(runtime.pointer_mutex);
            runtime.allocations[(uintptr_t)*devPtr] = {size, LegacyAllocation};
        }
        return ret;
    }

    static cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
    {
        cudaMemPool_t pool = NULL;
        cudaError_t ret = getExportPool(&pool);
        if (ret == cudaSuccess)
            ret = cudaMallocFromPoolAsync(devPtr, size, pool, stream);
        if (ret == cudaSuccess)
        {
            std::lock_guard<std::mutex> lock(runtime.pointer_mutex);
            runtime.allocations[(uintptr_t)*devPtr] = {size, PoolAllocation};
        }
        return ret;
    }

    static cudaError_t cudaFree(void *ptr)
    {
        cudaError_t ret = runtime.origCudaFree(ptr);
        if (ret == cudaSuccess)
        {
            std::lock_guard<std::mutex> lock(runtime.pointer_mutex);
            runtime.allocations.erase((uintptr_t)ptr);
        }
        return ret;
    }

    static cudaError_t cudaFreeAsync(void *ptr, cudaStream_t stream)
    {
        cudaError_t ret = runtime.origCudaFreeAsync(ptr, stream);
        if (ret == cudaSuccess)
        {
            std::lock_guard<std::mutex> lock(runtime.pointer_mutex);
            runtime.allocations.erase((uintptr_t)ptr);
        }
        return ret;
    }

    static cudaError_t cudaSetDevice(int device)
    {
        cudaError_t const ret = runtime.origCudaSetDevice(0);
        return ret;
    }

    static inline uint64_t sizeofNcclDataType(int datatype)
    {
        switch (datatype)
        {
        case ncclInt8:
            return sizeof(int8_t);
        case ncclUint8:
            return sizeof(uint8_t);
        case ncclInt32:
            return sizeof(int32_t);
        case ncclUint32:
            return sizeof(uint32_t);
        case ncclInt64:
            return sizeof(int64_t);
        case ncclUint64:
            return sizeof(uint64_t);
        case ncclFloat16:
            return sizeof(__half);
        case ncclFloat32:
            return sizeof(float);
        case ncclFloat64:
            return sizeof(double);
        case ncclBfloat16:
            return sizeof(__nv_bfloat16);
        default:
            return 0;
        }
        return 0;
    }

    static ncclResult_t getUniqueId(ncclUniqueId *nccl_id)
    {
        if (nccl_id == NULL)
            return ncclInvalidArgument;
        static std::atomic<uint64_t> counter(0);
        uint64_t seed = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
        seed ^= (uint64_t)getpid() << 32;
        seed ^= ++counter;
        unsigned char *bytes = reinterpret_cast<unsigned char *>(nccl_id);
        for (size_t i = 0; i < sizeof(*nccl_id); ++i)
        {
            seed ^= seed >> 12;
            seed ^= seed << 25;
            seed ^= seed >> 27;
            bytes[i] = (unsigned char)((seed * UINT64_C(2685821657736338717)) >> 56);
        }
        return ncclSuccess;
    }

    /* ncclCommInitRank has no MPI communicator argument, so membership has to
     * be bootstrapped out of band.  NCCL Fold is single-host: small, atomically
     * published files let only the participating processes rendezvous without
     * involving non-members in an MPI_COMM_WORLD collective. */
    struct BootstrapRecord
    {
        uint64_t magic;
        ncclUniqueId id;
        int ndev;
        int nccl_rank;
        int world_rank;
    };

    static uint64_t hashId(const ncclUniqueId &id, uint64_t seed)
    {
        const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&id);
        uint64_t hash = seed;
        for (size_t i = 0; i < sizeof(id); ++i)
        {
            hash ^= bytes[i];
            hash *= UINT64_C(1099511628211);
        }
        return hash;
    }

    static uint64_t timeoutMilliseconds()
    {
        static uint64_t value = []() -> uint64_t
        {
            const char *text = std::getenv("NCCL_FOLD_TIMEOUT_MS");
            if (!text || !*text)
                return 0;
            char *end = NULL;
            errno = 0;
            unsigned long long parsed = std::strtoull(text, &end, 10);
            if (*text == '-' || errno || *end || parsed > UINT64_MAX)
            {
                std::fprintf(stderr, "NCCL Fold: ignoring invalid NCCL_FOLD_TIMEOUT_MS=%s\n", text);
                return 0;
            }
            return (uint64_t)parsed;
        }();
        return value;
    }

    enum DiagnosticClass : uint32_t
    {
        DiagBootstrap,
        DiagCollective,
        DiagSend,
        DiagRecv,
        DiagGroup
    };
    enum DiagnosticState : uint32_t
    {
        DiagEntered,
        DiagWaiting,
        DiagCompleted
    };
    struct DiagnosticSnapshot
    {
        uint64_t magic;
        uint32_t version, record_size;
        uint64_t fingerprint, sequence, count;
        int32_t rank, size, operation, state, kind, peer, datatype, reduction, root, group_index;
        char phase[48];
    };
    static const uint64_t DIAGNOSTIC_MAGIC = UINT64_C(0x4e43464c44494147);
    static bool writeAll(int fd, const void *data, size_t bytes);

    static const char *collectiveName(int kind)
    {
        static const char *names[] = {"Broadcast", "AllGather", "Reduce", "AllReduce",
                                      "ReduceScatter", "AlltoAll", "Gather", "Scatter"};
        return kind >= 0 && kind < 8 ? names[kind] : "unknown";
    }
    static const char *datatypeName(int datatype)
    {
        switch (datatype)
        {
        case ncclInt8:
            return "ncclInt8";
        case ncclUint8:
            return "ncclUint8";
        case ncclInt32:
            return "ncclInt32";
        case ncclUint32:
            return "ncclUint32";
        case ncclInt64:
            return "ncclInt64";
        case ncclUint64:
            return "ncclUint64";
        case ncclFloat16:
            return "ncclFloat16";
        case ncclFloat32:
            return "ncclFloat32";
        case ncclFloat64:
            return "ncclFloat64";
        case ncclBfloat16:
            return "ncclBfloat16";
        default:
            return "unknown";
        }
    }
    static const char *reductionName(int op)
    {
        switch (op)
        {
        case ncclSum:
            return "ncclSum";
        case ncclProd:
            return "ncclProd";
        case ncclMin:
            return "ncclMin";
        case ncclMax:
            return "ncclMax";
        default:
            return "n/a";
        }
    }

    static bool atomicReplace(const std::string &path, const void *data, size_t bytes)
    {
        static std::atomic<uint64_t> serial(0);
        std::string temporary = path + ".tmp-" + std::to_string((long long)getpid()) + "-" +
                                std::to_string((unsigned long long)++serial);
        int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0)
            return false;
        bool ok = writeAll(fd, data, bytes) && fsync(fd) == 0;
        close(fd);
        if (ok)
            ok = rename(temporary.c_str(), path.c_str()) == 0;
        if (!ok)
            unlink(temporary.c_str());
        return ok;
    }

    static std::string snapshotPath(const VirtualComm *comm, int rank)
    {
        return comm->diagnostic_directory + "/snapshot-" + std::to_string(rank);
    }
    static void publishSnapshot(VirtualComm *comm, DiagnosticClass operation, uint64_t sequence,
                                const char *phase, DiagnosticState state, int kind = -1, uint64_t count = 0,
                                int datatype = -1, int reduction = -1, int root = -1, int peer = -1, int group_index = -1)
    {
        if (!timeoutMilliseconds() || !comm || comm->diagnostic_directory.empty())
            return;
        DiagnosticSnapshot snapshot = {};
        snapshot.magic = DIAGNOSTIC_MAGIC;
        snapshot.version = 1;
        snapshot.record_size = sizeof(snapshot);
        snapshot.fingerprint = comm->fingerprint;
        snapshot.sequence = sequence;
        snapshot.count = count;
        snapshot.rank = comm->rank;
        snapshot.size = comm->ndev;
        snapshot.operation = operation;
        snapshot.state = state;
        snapshot.kind = kind;
        snapshot.peer = peer;
        snapshot.datatype = datatype;
        snapshot.reduction = reduction;
        snapshot.root = root;
        snapshot.group_index = group_index;
        std::snprintf(snapshot.phase, sizeof(snapshot.phase), "%s", phase ? phase : "unknown");
        atomicReplace(snapshotPath(comm, comm->rank), &snapshot, sizeof(snapshot));
    }
    static bool readSnapshot(const VirtualComm *comm, int rank, DiagnosticSnapshot *snapshot)
    {
        int fd = open(snapshotPath(comm, rank).c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            return false;
        size_t left = sizeof(*snapshot);
        char *out = reinterpret_cast<char *>(snapshot);
        while (left)
        {
            ssize_t n = read(fd, out, left);
            if (n < 0 && errno == EINTR)
                continue;
            if (n <= 0)
            {
                close(fd);
                return false;
            }
            out += n;
            left -= (size_t)n;
        }
        close(fd);
        return snapshot->magic == DIAGNOSTIC_MAGIC && snapshot->version == 1 &&
               snapshot->record_size == sizeof(*snapshot) && snapshot->fingerprint == comm->fingerprint &&
               snapshot->rank == rank && snapshot->size == comm->ndev;
    }
    [[noreturn]] static void timeoutAbort(VirtualComm *comm, MPI_Comm mpi_comm,
                                          const char *phase, uint64_t sequence)
    {
        std::fprintf(stderr, "NCCL Fold timeout: phase=%s comm=%016llx seq=%llu timeout_ms=%llu\n\n",
                     phase, (unsigned long long)(comm ? comm->fingerprint : 0),
                     (unsigned long long)sequence, (unsigned long long)timeoutMilliseconds());
        if (comm)
            for (int rank = 0; rank < comm->ndev; ++rank)
            {
                DiagnosticSnapshot s;
                if (!readSnapshot(comm, rank, &s))
                {
                    std::fprintf(stderr, "rank %d: no valid/current snapshot\n", rank);
                    continue;
                }
                const char *state = s.state == DiagCompleted ? "last completed" : s.state == DiagWaiting ? "waiting"
                                                                                                         : "entered";
                if (s.operation == DiagCollective)
                    std::fprintf(stderr, "rank %d: %s collective seq=%llu %s count=%llu dtype=%s op=%s root=%d phase=%s\n",
                                 rank, state, (unsigned long long)s.sequence, collectiveName(s.kind),
                                 (unsigned long long)s.count, datatypeName(s.datatype), reductionName(s.reduction), s.root, s.phase);
                else
                    std::fprintf(stderr, "rank %d: %s %s seq=%llu peer=%d count=%llu dtype=%s phase=%s\n",
                                 rank, state, s.operation == DiagSend ? "send" : s.operation == DiagRecv ? "recv"
                                                                             : s.operation == DiagGroup  ? "group"
                                                                                                         : "bootstrap",
                                 (unsigned long long)s.sequence,
                                 s.peer, (unsigned long long)s.count, datatypeName(s.datatype), s.phase);
            }
        std::fprintf(stderr, "\npossible protocol divergence: not all ranks reached the same control-plane rendezvous\n"
                             "NCCL Fold: aborting after diagnostic timeout; recovery is not supported\n");
        std::fflush(stderr);
        MPI_Abort(mpi_comm == MPI_COMM_NULL ? MPI_COMM_WORLD : mpi_comm, 124);
        _exit(124);
    }

    static ncclResult_t timedWait(MPI_Request *requests, int count, VirtualComm *comm,
                                  MPI_Comm mpi_comm, const char *phase, uint64_t sequence)
    {
        if (!count)
            return ncclSuccess;
        if (!timeoutMilliseconds())
            return MPI_Waitall(count, requests, MPI_STATUSES_IGNORE) == MPI_SUCCESS ? ncclSuccess : ncclSystemError;
        const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() +
                                                               std::chrono::milliseconds(timeoutMilliseconds());
        for (;;)
        {
            int complete = 0;
            int error = MPI_Testall(count, requests, &complete, MPI_STATUSES_IGNORE);
            if (error != MPI_SUCCESS)
                return ncclSystemError;
            if (complete)
                return ncclSuccess;
            if (std::chrono::steady_clock::now() >= deadline)
                timeoutAbort(comm, mpi_comm, phase, sequence);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    static ncclResult_t timedBarrier(VirtualComm *comm, MPI_Comm mpi_comm, const char *phase, uint64_t sequence)
    {
        if (!timeoutMilliseconds())
            return MPI_Barrier(mpi_comm) == MPI_SUCCESS ? ncclSuccess : ncclSystemError;
        MPI_Request request = MPI_REQUEST_NULL;
        if (MPI_Ibarrier(mpi_comm, &request) != MPI_SUCCESS)
            return ncclSystemError;
        return timedWait(&request, 1, comm, mpi_comm, phase, sequence);
    }
    static ncclResult_t timedAllgather(const void *send, int send_count, MPI_Datatype send_type,
                                       void *receive, int receive_count, MPI_Datatype receive_type, VirtualComm *comm,
                                       const char *phase, uint64_t sequence)
    {
        if (!timeoutMilliseconds())
            return MPI_Allgather(send, send_count, send_type, receive,
                                 receive_count, receive_type, comm->mpi_comm) == MPI_SUCCESS
                       ? ncclSuccess
                       : ncclSystemError;
        MPI_Request request = MPI_REQUEST_NULL;
        if (MPI_Iallgather(send, send_count, send_type, receive, receive_count, receive_type,
                           comm->mpi_comm, &request) != MPI_SUCCESS)
            return ncclSystemError;
        return timedWait(&request, 1, comm, comm->mpi_comm, phase, sequence);
    }

    static bool makeDirectory(const std::string &path)
    {
        return mkdir(path.c_str(), 0700) == 0 || errno == EEXIST;
    }

    static bool writeAll(int fd, const void *data, size_t bytes)
    {
        const char *position = static_cast<const char *>(data);
        while (bytes)
        {
            ssize_t written = write(fd, position, bytes);
            if (written < 0 && errno == EINTR)
                continue;
            if (written <= 0)
                return false;
            position += written;
            bytes -= (size_t)written;
        }
        return true;
    }

    static bool readRecord(const std::string &path, BootstrapRecord *record)
    {
        int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            return false;
        char *position = reinterpret_cast<char *>(record);
        size_t remaining = sizeof(*record);
        while (remaining)
        {
            ssize_t got = read(fd, position, remaining);
            if (got < 0 && errno == EINTR)
                continue;
            if (got <= 0)
            {
                close(fd);
                return false;
            }
            position += got;
            remaining -= (size_t)got;
        }
        char extra;
        bool exact = read(fd, &extra, 1) == 0;
        close(fd);
        return exact;
    }

    static bool publishFile(const std::string &path, const void *data, size_t bytes)
    {
        static std::atomic<uint64_t> serial(0);
        std::string temporary = path + ".tmp-" + std::to_string((long long)getpid()) + "-" +
                                std::to_string((unsigned long long)++serial);
        int fd = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0)
            return false;
        bool ok = writeAll(fd, data, bytes) && fsync(fd) == 0;
        close(fd);
        if (ok)
            ok = link(temporary.c_str(), path.c_str()) == 0;
        unlink(temporary.c_str());
        return ok;
    }

    static bool sameBootstrap(const BootstrapRecord &record, const ncclUniqueId &id,
                              int ndev, int rank)
    {
        return record.magic == UINT64_C(0x414d47454c425354) && record.ndev == ndev &&
               record.nccl_rank == rank && std::memcmp(&record.id, &id, sizeof(id)) == 0;
    }

    static ncclResult_t bootstrapComm(MPI_Comm *result, const ncclUniqueId &id, int ndev, int rank)
    {
        int world_rank = -1, world_size = 0;
        if (MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS ||
            MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS || ndev > world_size)
            return ncclInvalidArgument;

        const char *configured = std::getenv("NCCL_FOLD_BOOTSTRAP_DIR");
        std::string root = configured && *configured ? configured : "/tmp/ncclfold-bootstrap-" + std::to_string((long long)getuid());
        if (!makeDirectory(root))
            return ncclSystemError;
        uint64_t h1 = hashId(id, UINT64_C(1469598103934665603));
        uint64_t h2 = hashId(id, UINT64_C(7809847782465536322));
        char name[64];
        std::snprintf(name, sizeof(name), "/comm-%016llx-%016llx",
                      (unsigned long long)h1, (unsigned long long)h2);
        std::string directory = root + name;
        if (!makeDirectory(directory))
            return ncclSystemError;

        BootstrapRecord local = {UINT64_C(0x414d47454c425354), id, ndev, rank, world_rank};
        std::string rank_path = directory + "/rank-" + std::to_string(rank);
        if (!publishFile(rank_path, &local, sizeof(local)))
            return ncclInvalidUsage;

        const std::chrono::steady_clock::time_point bootstrap_deadline = std::chrono::steady_clock::now() +
                                                                         std::chrono::milliseconds(timeoutMilliseconds());
        std::vector<int> members(ndev, -1);
        for (;;)
        {
            bool complete = true;
            for (int r = 0; r < ndev; ++r)
            {
                if (members[r] >= 0)
                    continue;
                BootstrapRecord peer;
                if (!readRecord(directory + "/rank-" + std::to_string(r), &peer))
                {
                    complete = false;
                    continue;
                }
                if (!sameBootstrap(peer, id, ndev, r) || peer.world_rank < 0 || peer.world_rank >= world_size)
                    return ncclInvalidUsage;
                members[r] = peer.world_rank;
            }
            if (complete)
                break;
            if (timeoutMilliseconds() && std::chrono::steady_clock::now() >= bootstrap_deadline)
                timeoutAbort(NULL, MPI_COMM_WORLD, "bootstrap-rank-files", 0);
            usleep(1000);
        }
        std::vector<int> sorted = members;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            return ncclInvalidUsage;

        int *tag_upper_bound = NULL, present = 0;
        if (MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_upper_bound, &present) != MPI_SUCCESS ||
            !present || tag_upper_bound == NULL || *tag_upper_bound < 0)
            return ncclSystemError;
        int tag = -1;
        std::string tag_selection = directory + "/tag";
        if (rank == 0)
        {
            uint64_t range = (uint64_t)*tag_upper_bound + 1;
            for (uint64_t attempt = 0; attempt < range; ++attempt)
            {
                int candidate = (int)((h1 + attempt) % range);
                std::string reservation = root + "/tag-" + std::to_string(candidate);
                int fd = open(reservation.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
                if (fd < 0)
                {
                    if (errno == EEXIST)
                        continue;
                    return ncclSystemError;
                }
                bool ok = writeAll(fd, &local, sizeof(local));
                close(fd);
                if (!ok || !publishFile(tag_selection, &candidate, sizeof(candidate)))
                {
                    unlink(reservation.c_str());
                    return ncclSystemError;
                }
                tag = candidate;
                break;
            }
            if (tag < 0)
                return ncclSystemError;
        }
        else
        {
            for (;;)
            {
                int fd = open(tag_selection.c_str(), O_RDONLY | O_CLOEXEC);
                if (fd >= 0)
                {
                    ssize_t got = read(fd, &tag, sizeof(tag));
                    close(fd);
                    if (got == (ssize_t)sizeof(tag))
                        break;
                }
                if (timeoutMilliseconds() && std::chrono::steady_clock::now() >= bootstrap_deadline)
                    timeoutAbort(NULL, MPI_COMM_WORLD, "bootstrap-tag-file", 0);
                usleep(1000);
            }
        }

        MPI_Group world_group = MPI_GROUP_NULL, member_group = MPI_GROUP_NULL;
        int error = MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        if (error == MPI_SUCCESS)
            error = MPI_Group_incl(world_group, ndev, members.data(), &member_group);
        if (error == MPI_SUCCESS)
            error = MPI_Comm_create_group(MPI_COMM_WORLD, member_group, tag, result);
        if (member_group != MPI_GROUP_NULL)
            MPI_Group_free(&member_group);
        if (world_group != MPI_GROUP_NULL)
            MPI_Group_free(&world_group);
        if (error != MPI_SUCCESS || *result == MPI_COMM_NULL)
            return ncclSystemError;

        /* Ensure no later communicator can reuse this creation tag until every
         * member has left MPI_Comm_create_group. */
        if (timedBarrier(NULL, *result, "bootstrap-barrier", 0) != ncclSuccess)
        {
            MPI_Comm_free(result);
            return ncclSystemError;
        }
        if (rank == 0)
        {
            unlink((root + "/tag-" + std::to_string(tag)).c_str());
            unlink(tag_selection.c_str());
            for (int r = 0; r < ndev; ++r)
                unlink((directory + "/rank-" + std::to_string(r)).c_str());
            rmdir(directory.c_str());
        }
        return ncclSuccess;
    }

    static bool sendFd(int socket, int fd, int rank)
    {
        struct iovec iov = {&rank, sizeof(rank)};
        char control[CMSG_SPACE(sizeof(int))] = {};
        struct msghdr message = {};
        message.msg_iov = &iov;
        message.msg_iovlen = 1;
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        struct cmsghdr *header = CMSG_FIRSTHDR(&message);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(header), &fd, sizeof(fd));
        ssize_t sent;
        do
        {
            sent = sendmsg(socket, &message, 0);
        } while (sent < 0 && errno == EINTR);
        return sent == (ssize_t)sizeof(rank);
    }

    static int receiveFd(int socket, int *rank)
    {
        char control[CMSG_SPACE(sizeof(int))] = {};
        struct iovec iov = {rank, sizeof(*rank)};
        struct msghdr message = {};
        message.msg_iov = &iov;
        message.msg_iovlen = 1;
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        ssize_t received;
        do
        {
            received = recvmsg(socket, &message, 0);
        } while (received < 0 && errno == EINTR);
        if (received != (ssize_t)sizeof(*rank) || (message.msg_flags & MSG_CTRUNC))
            return -1;
        struct cmsghdr *header = CMSG_FIRSTHDR(&message);
        if (!header || header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS ||
            header->cmsg_len != CMSG_LEN(sizeof(int)))
            return -1;
        int fd = -1;
        std::memcpy(&fd, CMSG_DATA(header), sizeof(fd));
        return fd;
    }

    static ncclResult_t exchangePools(VirtualComm *comm)
    {
        cudaMemPool_t pool = NULL;
        int available = getExportPool(&pool) == cudaSuccess ? 1 : 0;
        int available_count = 0;
        if (timeoutMilliseconds())
        {
            MPI_Request request = MPI_REQUEST_NULL;
            if (MPI_Iallreduce(&available, &available_count, 1, MPI_INT, MPI_SUM, comm->mpi_comm, &request) != MPI_SUCCESS ||
                timedWait(&request, 1, comm, comm->mpi_comm, "pool-capability-allreduce", 0) != ncclSuccess)
                return ncclSystemError;
        }
        else if (MPI_Allreduce(&available, &available_count, 1, MPI_INT, MPI_SUM, comm->mpi_comm) != MPI_SUCCESS)
            return ncclSystemError;
        comm->imported_pools.assign(comm->ndev, NULL);
        /* Legacy-only programs remain usable on devices without pool IPC, but
         * cudaMallocAsync itself returns cudaErrorNotSupported (never a sync fallback). */
        if (available_count == 0)
            return ncclSuccess;
        if (available_count != comm->ndev)
            return ncclSystemError;
        int pool_fd = -1;
        if (cudaMemPoolExportToShareableHandle(&pool_fd, pool,
                                               cudaMemHandleTypePosixFileDescriptor, 0) != cudaSuccess)
            return ncclUnhandledCudaError;

        const char *configured = std::getenv("NCCL_FOLD_BOOTSTRAP_DIR");
        std::string root = configured && *configured ? configured : "/tmp/ncclfold-bootstrap-" + std::to_string((long long)getuid());
        uint64_t h1 = hashId(comm->unique_id, UINT64_C(1469598103934665603));
        uint64_t h2 = hashId(comm->unique_id, UINT64_C(7809847782465536322));
        char name[64];
        std::snprintf(name, sizeof(name), "/pool-%016llx-%016llx",
                      (unsigned long long)h1, (unsigned long long)h2);
        std::string directory = root + name;
        if (!makeDirectory(root) || !makeDirectory(directory))
        {
            close(pool_fd);
            return ncclSystemError;
        }
        std::string path = directory + "/rank-" + std::to_string(comm->rank);
        int listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        struct sockaddr_un address = {};
        address.sun_family = AF_UNIX;
        if (path.size() >= sizeof(address.sun_path))
        {
            close(pool_fd);
            if (listener >= 0)
                close(listener);
            return ncclSystemError;
        }
        std::strncpy(address.sun_path, path.c_str(), sizeof(address.sun_path) - 1);
        unlink(path.c_str());
        if (listener < 0 || bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
            listen(listener, comm->ndev) != 0)
        {
            if (listener >= 0)
                close(listener);
            close(pool_fd);
            return ncclSystemError;
        }
        if (timedBarrier(comm, comm->mpi_comm, "pool-listener-barrier", 0) != ncclSuccess)
        {
            close(listener);
            close(pool_fd);
            unlink(path.c_str());
            return ncclSystemError;
        }
        ncclResult_t result = ncclSuccess;
        const std::chrono::steady_clock::time_point socket_deadline = std::chrono::steady_clock::now() +
                                                                      std::chrono::milliseconds(timeoutMilliseconds());
        for (int peer = 0; peer < comm->ndev && result == ncclSuccess; ++peer)
        {
            if (peer == comm->rank)
                continue;
            int channel = -1;
            if (peer < comm->rank)
            {
                if (timeoutMilliseconds())
                {
                    for (;;)
                    {
                        struct pollfd descriptor = {listener, POLLIN, 0};
                        int ready = poll(&descriptor, 1, 1);
                        if (ready > 0)
                        {
                            channel = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
                            if (channel >= 0)
                                break;
                        }
                        if (ready < 0 && errno != EINTR)
                            break;
                        if (std::chrono::steady_clock::now() >= socket_deadline)
                            timeoutAbort(comm, comm->mpi_comm, "pool-socket-accept", 0);
                    }
                }
                else
                    do
                    {
                        channel = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
                    } while (channel < 0 && errno == EINTR);
            }
            else
            {
                channel = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
                struct sockaddr_un remote = {};
                remote.sun_family = AF_UNIX;
                std::string remote_path = directory + "/rank-" + std::to_string(peer);
                std::strncpy(remote.sun_path, remote_path.c_str(), sizeof(remote.sun_path) - 1);
                while (channel >= 0 && connect(channel, reinterpret_cast<sockaddr *>(&remote), sizeof(remote)) != 0)
                {
                    if (errno != EINTR && errno != ENOENT && errno != ECONNREFUSED)
                    {
                        close(channel);
                        channel = -1;
                        break;
                    }
                    if (timeoutMilliseconds() && std::chrono::steady_clock::now() >= socket_deadline)
                        timeoutAbort(comm, comm->mpi_comm, "pool-socket-connect", 0);
                    usleep(1000);
                }
            }
            if (channel >= 0 && timeoutMilliseconds())
            {
                struct timeval timeout = {(time_t)(timeoutMilliseconds() / 1000),
                                          (suseconds_t)((timeoutMilliseconds() % 1000) * 1000)};
                setsockopt(channel, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
                setsockopt(channel, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            }
            int remote_fd = -1, remote_rank = -1;
            bool sent = channel >= 0 && sendFd(channel, pool_fd, comm->rank);
            if (timeoutMilliseconds() && channel >= 0 && !sent && (errno == EAGAIN || errno == EWOULDBLOCK))
                timeoutAbort(comm, comm->mpi_comm, "pool-socket-send", 0);
            if (sent)
                remote_fd = receiveFd(channel, &remote_rank);
            if (timeoutMilliseconds() && sent && remote_fd < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
                timeoutAbort(comm, comm->mpi_comm, "pool-socket-receive", 0);
            if (!sent || remote_fd < 0 || remote_rank < 0 ||
                remote_rank >= comm->ndev || remote_rank == comm->rank ||
                comm->imported_pools[remote_rank])
                result = ncclSystemError;
            if (channel >= 0)
                close(channel);
            if (result == ncclSuccess)
            {
                cudaError_t error = cudaMemPoolImportFromShareableHandle(&comm->imported_pools[remote_rank],
                                                                         &remote_fd, cudaMemHandleTypePosixFileDescriptor, 0);
                close(remote_fd);
                if (error != cudaSuccess)
                    result = ncclUnhandledCudaError;
            }
            else if (remote_fd >= 0)
                close(remote_fd);
        }
        close(pool_fd);
        close(listener);
        unlink(path.c_str());
        if (timedBarrier(comm, comm->mpi_comm, "pool-cleanup-barrier", 0) != ncclSuccess)
            return ncclSystemError;
        if (comm->rank == 0)
            rmdir(directory.c_str());
        return result;
    }

    void *getAllocation(void *pointer_input, size_t bytes, uint64_t *offset,
                        AllocationKind *kind = NULL)
    {
        std::lock_guard<std::mutex> lock(runtime.pointer_mutex);
        uintptr_t p = (uintptr_t)pointer_input;
        std::map<uintptr_t, RuntimeState::Allocation>::iterator it = runtime.allocations.upper_bound(p);
        if (it == runtime.allocations.begin())
            return NULL;
        --it;
        size_t delta = p - it->first;
        if (delta > it->second.size || bytes > it->second.size - delta)
            return NULL;
        if (offset)
            *offset = delta;
        if (kind)
            *kind = it->second.kind;
        return (void *)it->first;
    }

    static ncclResult_t commInitRank(ncclComm_t *comm, int ndev, ncclUniqueId nccl_id, int rank)
    {
        if (comm == NULL || ndev <= 0 || rank < 0 || rank >= ndev)
            return ncclInvalidArgument;
        *comm = NULL;
        VirtualComm *virtual_comm = new (std::nothrow) VirtualComm;
        if (virtual_comm == NULL)
            return ncclSystemError;
        virtual_comm->unique_id = nccl_id;
        ncclResult_t bootstrap = bootstrapComm(&virtual_comm->mpi_comm, nccl_id, ndev, rank);
        if (bootstrap != ncclSuccess)
        {
            delete virtual_comm;
            return bootstrap;
        }
        virtual_comm->rank = rank;
        virtual_comm->ndev = ndev;
        virtual_comm->fingerprint = hashId(nccl_id, UINT64_C(1469598103934665603));
        if (timeoutMilliseconds())
        {
            const char *configured = std::getenv("NCCL_FOLD_BOOTSTRAP_DIR");
            std::string root = configured && *configured ? configured : "/tmp/ncclfold-bootstrap-" + std::to_string((long long)getuid());
            char name[48];
            std::snprintf(name, sizeof(name), "/diagnostics-%016llx",
                          (unsigned long long)virtual_comm->fingerprint);
            virtual_comm->diagnostic_directory = root + name;
            if (!makeDirectory(root) || !makeDirectory(virtual_comm->diagnostic_directory))
                timeoutAbort(virtual_comm, virtual_comm->mpi_comm, "diagnostic-bootstrap", 0);
            publishSnapshot(virtual_comm, DiagBootstrap, 0, "pool-exchange", DiagWaiting);
        }
        int mpi_rank = -1, mpi_size = 0;
        MPI_Comm_rank(virtual_comm->mpi_comm, &mpi_rank);
        MPI_Comm_size(virtual_comm->mpi_comm, &mpi_size);
        if (mpi_rank != rank || mpi_size != ndev)
        {
            MPI_Comm_free(&virtual_comm->mpi_comm);
            delete virtual_comm;
            return ncclSystemError;
        }
        virtual_comm->next_send_sequence.assign(ndev, 0);
        virtual_comm->next_recv_sequence.assign(ndev, 0);
        ncclResult_t pool_exchange = exchangePools(virtual_comm);
        if (pool_exchange != ncclSuccess)
        {
            MPI_Comm_free(&virtual_comm->mpi_comm);
            delete virtual_comm;
            return pool_exchange;
        }
        {
            std::lock_guard<std::mutex> lock(runtime.communicator_mutex);
            runtime.communicators.insert(virtual_comm);
        }
        *comm = reinterpret_cast<ncclComm_t>(virtual_comm);
        return ncclSuccess;
    }

    static ncclResult_t groupStart()
    {
        if (group_state.depth == 0)
            group_state.operations.clear();
        ++group_state.depth;
        return ncclSuccess;
    }

    static bool debugEnabled()
    {
        static int enabled = std::getenv("NCCL_FOLD_DEBUG_P2P") != NULL;
        return enabled != 0;
    }

    static ncclResult_t cudaToNccl(cudaError_t error) noexcept
    {
        return error == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
    }

    static ncclResult_t mpiToNccl(int error) noexcept
    {
        return error == MPI_SUCCESS ? ncclSuccess : ncclSystemError;
    }

    static bool resourceStatsEnabled()
    {
        static int enabled = std::getenv("NCCL_FOLD_RESOURCE_STATS") != NULL;
        return enabled != 0;
    }

    /* A slot stays leased from before it is exported until the control-plane
     * protocol proves that every importer has submitted its wait.  Failures do
     * not release slots: uncertain generations are conservatively quarantined
     * until communicator destruction. */
    static ncclResult_t leaseEvent(VirtualComm *comm, size_t *slot, cudaEvent_t *event)
    {
        for (size_t i = 0; i < comm->event_pool.size(); ++i)
        {
            if (!comm->event_pool[i].leased)
            {
                comm->event_pool[i].leased = true;
                *slot = i;
                *event = comm->event_pool[i].event;
                return ncclSuccess;
            }
        }
        cudaEvent_t created = NULL;
        ncclResult_t result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaEventCreateWithFlags, &created, cudaEventInterprocess | cudaEventDisableTiming));
        if (result != ncclSuccess)
            return result;
        comm->event_pool.push_back(EventSlot(created));
        comm->event_pool_peak = std::max(comm->event_pool_peak, comm->event_pool.size());
        *slot = comm->event_pool.size() - 1;
        *event = created;
        return ncclSuccess;
    }

    static void releaseEvent(VirtualComm *comm, size_t slot)
    {
        if (slot < comm->event_pool.size())
            comm->event_pool[slot].leased = false;
    }

    static ncclResult_t openEvent(VirtualComm *comm, cudaIpcEventHandle_t const &handle,
                                  cudaEvent_t *event)
    {
        for (size_t i = 0; i < comm->imported_event_cache.size(); ++i)
        {
            if (std::memcmp(&comm->imported_event_cache[i].handle, &handle, sizeof(handle)) == 0)
            {
                *event = comm->imported_event_cache[i].event;
                return ncclSuccess;
            }
        }
        ncclResult_t result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaIpcOpenEventHandle, event, handle));
        if (result == ncclSuccess)
            comm->imported_event_cache.push_back({handle, *event});
        return result;
    }

    enum
    {
        readyTag = 17001,
        doneTag = 17002,
        ackTag = 17003
    };

    static ncclResult_t exportMemory(void *allocation, AllocationKind kind,
                                     memoryDescriptor *descriptor, uint64_t offset)
    {
        descriptor->kind = kind;
        descriptor->reserved = 0;
        descriptor->offset = offset;
        cudaError_t error = kind == LegacyAllocation
                                ? ATLC_LOG_CUDA(runtime.cudaIpcGetMemHandle, &descriptor->handle.legacy, allocation)
                                : ATLC_LOG_CUDA(cudaMemPoolExportPointer, &descriptor->handle.pool, allocation);
        return cudaToNccl(error);
    }

    static ncclResult_t openMemory(VirtualComm *comm, cudaIpcMemHandle_t const &handle, void **pointer)
    {
        for (size_t i = 0; i < comm->ipc_mappings.size(); ++i)
        {
            if (std::memcmp(&comm->ipc_mappings[i].handle, &handle, sizeof(handle)) == 0)
            {
                *pointer = comm->ipc_mappings[i].pointer;
                return ncclSuccess;
            }
        }
        ncclResult_t result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaIpcOpenMemHandle, pointer, handle, cudaIpcMemLazyEnablePeerAccess));
        if (result == ncclSuccess)
            comm->ipc_mappings.push_back({handle, *pointer});
        return result;
    }

    static ncclResult_t importMemory(VirtualComm *comm, int exporter,
                                     memoryDescriptor const &descriptor, void **pointer, bool *temporary)
    {
        *temporary = false;
        if (descriptor.kind == LegacyAllocation)
            return openMemory(comm, descriptor.handle.legacy, pointer);
        if (descriptor.kind != PoolAllocation || exporter < 0 || exporter >= comm->ndev ||
            !comm->imported_pools[exporter])
            return ncclInvalidArgument;
        ncclResult_t result = cudaToNccl(ATLC_LOG_CUDA(cudaMemPoolImportPointer, pointer,
                                                       comm->imported_pools[exporter], &descriptor.handle.pool));
        if (result == ncclSuccess)
            *temporary = true;
        return result;
    }

    /* MPI is only the control plane.  Ready slots are released after done
     * metadata proves the receiver submitted its ready wait.  Done slots use an
     * explicit acknowledgement after the sender submits its done wait. */
    static ncclResult_t enqueueP2P(VirtualComm *comm, const std::vector<sendRecvArgs_t> &send_args,
                                   const std::vector<sendRecvArgs_t> &recv_args)
    {
        const size_t send_count = send_args.size();
        const size_t recv_count = recv_args.size();
        std::vector<readyMessage> outgoing(send_count);
        std::vector<readyMessage> incoming(recv_count);
        std::vector<doneMessage> outgoing_done(recv_count);
        std::vector<doneMessage> incoming_done(send_count);
        std::vector<MPI_Request> requests(send_count + recv_count);
        std::vector<size_t> ready_slots(send_count), done_slots(recv_count);

        for (size_t i = 0; i < send_count; ++i)
        {
            sendRecvArgs_t const &op = send_args[i];
            readyMessage &message = outgoing[i];
            uint64_t type_size = sizeofNcclDataType(op.datatype);
            if (type_size == 0 || op.count > UINT64_MAX / type_size)
                return ncclInvalidArgument;
            AllocationKind allocation_kind;
            void *allocation = getAllocation(op.buff, op.count * type_size, &message.memory.offset, &allocation_kind);
            if (allocation == NULL)
                return ncclInvalidArgument;
            ncclResult_t result = exportMemory(allocation, allocation_kind, &message.memory, message.memory.offset);
            if (result != ncclSuccess)
                return result;
            cudaEvent_t ready = NULL;
            result = leaseEvent(comm, &ready_slots[i], &ready);
            if (result != ncclSuccess)
                return result;
            result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaEventRecord, ready, op.stream));
            if (result != ncclSuccess)
                return result;
            result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaIpcGetEventHandle, &message.ready, ready));
            if (result != ncclSuccess)
                return result;
            message.bytes = op.count * type_size;
            message.count = op.count;
            message.sequence = op.sequence;
            message.datatype = op.datatype;
            if (debugEnabled())
                std::fprintf(stderr, "NCCL Fold comm=%p rank=%d peer=%d seq=%llu ready=%p stream=%p send\n",
                             (void *)comm, comm->rank, op.peer, (unsigned long long)op.sequence, (void *)ready, (void *)op.stream);
        }
        for (size_t i = 0; i < send_count; ++i)
        {
            ncclResult_t result = mpiToNccl(ATLC_LOG_MPI(MPI_Isend, &outgoing[i], sizeof(readyMessage), MPI_BYTE,
                                                         send_args[i].peer, readyTag, comm->mpi_comm, &requests[i]));
            if (result != ncclSuccess)
                return result;
        }
        for (size_t i = 0; i < recv_count; ++i)
        {
            ncclResult_t result = mpiToNccl(ATLC_LOG_MPI(MPI_Irecv, &incoming[i], sizeof(incoming[i]), MPI_BYTE,
                                                         recv_args[i].peer, readyTag, comm->mpi_comm, &requests[send_count + i]));
            if (result != ncclSuccess)
                return result;
        }
        if (!requests.empty())
        {
            publishSnapshot(comm, send_count ? DiagSend : DiagRecv,
                            send_count ? send_args[0].sequence : recv_args[0].sequence, "p2p-ready-metadata", DiagWaiting,
                            -1, send_count ? send_args[0].count : recv_args[0].count,
                            send_count ? send_args[0].datatype : recv_args[0].datatype, -1, -1,
                            send_count ? send_args[0].peer : recv_args[0].peer);
            ncclResult_t result = timedWait(requests.data(), (int)requests.size(), comm, comm->mpi_comm,
                                            "p2p-ready-metadata", send_count ? send_args[0].sequence : recv_args[0].sequence);
            if (result != ncclSuccess)
                return result;
        }

        requests.assign(send_count + recv_count, MPI_REQUEST_NULL);
        ncclResult_t deferred_error = ncclSuccess;
        for (size_t i = 0; i < recv_count; ++i)
        {
            sendRecvArgs_t const &op = recv_args[i];
            readyMessage const &message = incoming[i];
            uint64_t const type_size = sizeofNcclDataType(op.datatype);
            if (type_size == 0 || op.count > UINT64_MAX / type_size)
                return ncclInvalidArgument;
            uint64_t const recv_bytes = op.count * type_size;
            bool mismatch = message.sequence != op.sequence || message.count != op.count ||
                            message.datatype != op.datatype || message.bytes != recv_bytes;
            if (mismatch)
            {
                deferred_error = ncclInvalidUsage;
                std::fprintf(stderr, "NCCL Fold P2P mismatch: comm=%016llx ranks=%d/%d\n"
                                     "send: seq=%llu count=%llu dtype=%s bytes=%llu\n"
                                     "recv: seq=%llu count=%llu dtype=%s bytes=%llu\n",
                             (unsigned long long)comm->fingerprint, op.peer, comm->rank,
                             (unsigned long long)message.sequence, (unsigned long long)message.count,
                             datatypeName(message.datatype), (unsigned long long)message.bytes,
                             (unsigned long long)op.sequence, (unsigned long long)op.count,
                             datatypeName(op.datatype), (unsigned long long)recv_bytes);
            }

            cudaEvent_t ready = NULL;
            cudaEvent_t done = NULL;
            void *source = NULL;
            bool temporary_source = false;
            ncclResult_t result = openEvent(comm, message.ready, &ready);
            if (result != ncclSuccess)
                return result;
            result = importMemory(comm, op.peer, message.memory, &source, &temporary_source);
            if (result != ncclSuccess)
                return result;
            result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaStreamWaitEvent, op.stream, ready, 0));
            if (result != ncclSuccess)
                return result;
            if (!mismatch && message.bytes == recv_bytes)
            {
                result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaMemcpyAsync, op.buff, (char *)source + message.memory.offset, recv_bytes,
                                                  cudaMemcpyDeviceToDevice, op.stream));
                if (result != ncclSuccess)
                    return result;
            }
            if (temporary_source)
            {
                result = cudaToNccl(ATLC_LOG_CUDA(runtime.origCudaFreeAsync, source, op.stream));
                if (result != ncclSuccess)
                    return result;
            }
            result = leaseEvent(comm, &done_slots[i], &done);
            if (result != ncclSuccess)
                return result;
            result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaEventRecord, done, op.stream));
            if (result != ncclSuccess)
                return result;
            result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaIpcGetEventHandle, &outgoing_done[i].done, done));
            if (result != ncclSuccess)
                return result;
            outgoing_done[i].sequence = message.sequence;
            outgoing_done[i].count = op.count;
            outgoing_done[i].datatype = op.datatype;
            outgoing_done[i].status = mismatch ? ncclInvalidUsage : ncclSuccess;
            if (debugEnabled())
                std::fprintf(stderr, "NCCL Fold comm=%p rank=%d peer=%d seq=%llu done=%p stream=%p recv\n",
                             (void *)comm, comm->rank, op.peer, (unsigned long long)message.sequence, (void *)done, (void *)op.stream);
        }
        for (size_t i = 0; i < recv_count; ++i)
        {
            ncclResult_t result = mpiToNccl(ATLC_LOG_MPI(MPI_Isend, &outgoing_done[i], sizeof(doneMessage), MPI_BYTE,
                                                         recv_args[i].peer, doneTag, comm->mpi_comm, &requests[send_count + i]));
            if (result != ncclSuccess)
                return result;
        }
        for (size_t i = 0; i < send_count; ++i)
        {
            ncclResult_t result = mpiToNccl(ATLC_LOG_MPI(MPI_Irecv, &incoming_done[i], sizeof(doneMessage), MPI_BYTE,
                                                         send_args[i].peer, doneTag, comm->mpi_comm, &requests[i]));
            if (result != ncclSuccess)
                return result;
        }
        if (!requests.empty())
        {
            ncclResult_t result = timedWait(requests.data(), (int)requests.size(), comm, comm->mpi_comm,
                                            "p2p-done-metadata", send_count ? send_args[0].sequence : recv_args[0].sequence);
            if (result != ncclSuccess)
                return result;
        }
        /* Matching done metadata implies that the receiver issued the ready
         * wait.  A mismatched/error generation remains quarantined. */
        for (size_t i = 0; i < send_count; ++i)
            if (incoming_done[i].sequence == send_args[i].sequence)
                releaseEvent(comm, ready_slots[i]);
        for (size_t i = 0; i < send_count; ++i)
        {
            if (incoming_done[i].sequence != send_args[i].sequence || incoming_done[i].count != send_args[i].count ||
                incoming_done[i].datatype != send_args[i].datatype || incoming_done[i].status != ncclSuccess)
                deferred_error = ncclInvalidUsage;
            cudaEvent_t done = NULL;
            ncclResult_t result = openEvent(comm, incoming_done[i].done, &done);
            if (result != ncclSuccess)
                return result;
            result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaStreamWaitEvent, send_args[i].stream, done, 0));
            if (result != ncclSuccess)
                return result;
        }
        /* Acknowledgements are sent only after all local waits above have been
         * submitted.  Receivers may then safely re-record their done slots. */
        requests.assign(send_count + recv_count, MPI_REQUEST_NULL);
        for (size_t i = 0; i < send_count; ++i)
        {
            ncclResult_t result = mpiToNccl(ATLC_LOG_MPI(MPI_Isend, &send_args[i].sequence, 1, MPI_UINT64_T,
                                                         send_args[i].peer, ackTag, comm->mpi_comm, &requests[i]));
            if (result != ncclSuccess)
                return result;
        }
        std::vector<uint64_t> acknowledgements(recv_count);
        for (size_t i = 0; i < recv_count; ++i)
        {
            ncclResult_t result = mpiToNccl(ATLC_LOG_MPI(MPI_Irecv, &acknowledgements[i], 1, MPI_UINT64_T,
                                                         recv_args[i].peer, ackTag, comm->mpi_comm, &requests[send_count + i]));
            if (result != ncclSuccess)
                return result;
        }
        if (!requests.empty())
        {
            ncclResult_t result = timedWait(requests.data(), (int)requests.size(), comm, comm->mpi_comm,
                                            "p2p-done-ack", send_count ? send_args[0].sequence : recv_args[0].sequence);
            if (result != ncclSuccess)
                return result;
        }
        for (size_t i = 0; i < recv_count; ++i)
        {
            if (acknowledgements[i] != recv_args[i].sequence)
                deferred_error = ncclInvalidUsage;
            else
                releaseEvent(comm, done_slots[i]);
        }
        publishSnapshot(comm, send_count ? DiagSend : DiagRecv,
                        send_count ? send_args[0].sequence : recv_args[0].sequence, "p2p-complete", DiagCompleted,
                        -1, send_count ? send_args[0].count : recv_args[0].count,
                        send_count ? send_args[0].datatype : recv_args[0].datatype, -1, -1,
                        send_count ? send_args[0].peer : recv_args[0].peer);
        return deferred_error;
    }

    /* Grouped P2P metadata is exchanged as one control-plane batch before any
     * grouped operation is enqueued. Event records and data movement stay in
     * GroupOp order; send completion waits are safely deferred via snapshots. */
    struct PreparedGroupP2P
    {
        readyMessage ready;
        doneMessage done;
        cudaEvent_t local_event;
        size_t event_slot;
        void *send_snapshot;
        PreparedGroupP2P() : local_event(NULL), event_slot(0), send_snapshot(NULL)
        {
            std::memset(&ready, 0, sizeof(ready));
            std::memset(&done, 0, sizeof(done));
        }
    };

    static ncclResult_t prepareGroupedP2P(std::vector<GroupOp> const &operations, std::vector<PreparedGroupP2P> &prepared)
    {
        prepared.resize(operations.size());
        std::vector<MPI_Request> requests;
        requests.reserve(operations.size() * 2);
        for (size_t i = 0; i < operations.size(); ++i)
        {
            GroupOp const &grouped = operations[i];
            if (grouped.kind == GroupCollective)
                continue;
            sendRecvArgs_t const &op = grouped.p2p;
            PreparedGroupP2P &p = prepared[i];
            uint64_t const type_size = sizeofNcclDataType(op.datatype);
            if (type_size == 0 || op.count > UINT64_MAX / type_size)
                return ncclInvalidArgument;
            uint64_t const bytes = op.count * type_size;
            MPI_Request request = MPI_REQUEST_NULL;
            ncclResult_t result;
            if (grouped.kind == GroupSend)
            {
                if (getAllocation(op.buff, bytes, NULL) == NULL)
                    return ncclInvalidArgument;
                if (bytes)
                {
                    if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.origCudaMalloc, &p.send_snapshot, bytes))) != ncclSuccess)
                        return result;
                    grouped.comm->retained_ipc_allocations.push_back(p.send_snapshot);
                    if ((result = exportMemory(p.send_snapshot, LegacyAllocation, &p.ready.memory, 0)) != ncclSuccess)
                        return result;
                }
                if ((result = leaseEvent(grouped.comm, &p.event_slot, &p.local_event)) != ncclSuccess)
                    return result;
                if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaIpcGetEventHandle, &p.ready.ready, p.local_event))) != ncclSuccess)
                    return result;
                p.ready.bytes = bytes;
                p.ready.count = op.count;
                p.ready.sequence = op.sequence;
                p.ready.datatype = op.datatype;
                if ((result = mpiToNccl(ATLC_LOG_MPI(MPI_Isend, &p.ready, sizeof(p.ready), MPI_BYTE, op.peer, readyTag, grouped.comm->mpi_comm, &request))) != ncclSuccess)
                    return result;
                requests.push_back(request);
                if ((result = mpiToNccl(ATLC_LOG_MPI(MPI_Irecv, &p.done, sizeof(p.done), MPI_BYTE, op.peer, doneTag, grouped.comm->mpi_comm, &request))) != ncclSuccess)
                    return result;
                requests.push_back(request);
            }
            else
            {
                if (bytes && getAllocation(op.buff, bytes, NULL) == NULL)
                    return ncclInvalidArgument;
                if ((result = leaseEvent(grouped.comm, &p.event_slot, &p.local_event)) != ncclSuccess)
                    return result;
                if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaIpcGetEventHandle, &p.done.done, p.local_event))) != ncclSuccess)
                    return result;
                p.done.sequence = op.sequence;
                p.done.count = op.count;
                p.done.datatype = op.datatype;
                p.done.status = ncclSuccess;
                if ((result = mpiToNccl(ATLC_LOG_MPI(MPI_Irecv, &p.ready, sizeof(p.ready), MPI_BYTE, op.peer, readyTag, grouped.comm->mpi_comm, &request))) != ncclSuccess)
                    return result;
                requests.push_back(request);
                if ((result = mpiToNccl(ATLC_LOG_MPI(MPI_Isend, &p.done, sizeof(p.done), MPI_BYTE, op.peer, doneTag, grouped.comm->mpi_comm, &request))) != ncclSuccess)
                    return result;
                requests.push_back(request);
            }
        }
        if (requests.empty())
            return ncclSuccess;
        VirtualComm *comm = operations.empty() ? NULL : operations[0].comm;
        uint64_t sequence = operations.empty() ? 0 : operations[0].p2p.sequence;
        if (comm)
            publishSnapshot(comm, DiagGroup, sequence, "group-p2p-metadata", DiagWaiting);
        return timedWait(requests.data(), (int)requests.size(), comm,
                         comm ? comm->mpi_comm : MPI_COMM_WORLD, "group-p2p-metadata", sequence);
    }

    static ncclResult_t enqueuePreparedGroupP2P(GroupOp const &grouped, PreparedGroupP2P const &p)
    {
        sendRecvArgs_t const &op = grouped.p2p;
        ncclResult_t result;
        if (grouped.kind == GroupSend)
        {
            if (p.done.sequence != op.sequence || p.done.count != op.count || p.done.datatype != op.datatype)
            {
                std::fprintf(stderr, "NCCL Fold grouped P2P mismatch: comm=%016llx peer=%d "
                                     "send(seq=%llu count=%llu dtype=%s) recv(seq=%llu count=%llu dtype=%s)\n",
                             (unsigned long long)grouped.comm->fingerprint, op.peer,
                             (unsigned long long)op.sequence, (unsigned long long)op.count, datatypeName(op.datatype),
                             (unsigned long long)p.done.sequence, (unsigned long long)p.done.count, datatypeName(p.done.datatype));
                return ncclInvalidUsage;
            }
            uint64_t const bytes = op.count * sizeofNcclDataType(op.datatype);
            if (bytes && (result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaMemcpyAsync, p.send_snapshot, op.buff, bytes, cudaMemcpyDeviceToDevice, op.stream))) != ncclSuccess)
                return result;
            if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaEventRecord, p.local_event, op.stream))) != ncclSuccess)
                return result;
            return ncclSuccess;
        }
        uint64_t const type_size = sizeofNcclDataType(op.datatype);
        uint64_t const bytes = op.count * type_size;
        if (p.ready.sequence != op.sequence || p.ready.count != op.count ||
            p.ready.datatype != op.datatype || p.ready.bytes != bytes)
        {
            std::fprintf(stderr, "NCCL Fold grouped P2P mismatch: comm=%016llx peer=%d "
                                 "send(seq=%llu count=%llu dtype=%s bytes=%llu) recv(seq=%llu count=%llu dtype=%s bytes=%llu)\n",
                         (unsigned long long)grouped.comm->fingerprint, op.peer,
                         (unsigned long long)p.ready.sequence, (unsigned long long)p.ready.count,
                         datatypeName(p.ready.datatype), (unsigned long long)p.ready.bytes,
                         (unsigned long long)op.sequence, (unsigned long long)op.count,
                         datatypeName(op.datatype), (unsigned long long)bytes);
            return ncclInvalidUsage;
        }
        cudaEvent_t ready = NULL;
        void *source = NULL;
        bool temporary_source = false;
        if ((result = openEvent(grouped.comm, p.ready.ready, &ready)) != ncclSuccess)
            return result;
        if (bytes && (result = importMemory(grouped.comm, op.peer, p.ready.memory, &source, &temporary_source)) != ncclSuccess)
            return result;
        if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaStreamWaitEvent, op.stream, ready, 0))) != ncclSuccess)
            return result;
        if (bytes && (result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaMemcpyAsync, op.buff, (char *)source + p.ready.memory.offset, bytes, cudaMemcpyDeviceToDevice, op.stream))) != ncclSuccess)
            return result;
        if (temporary_source && (result = cudaToNccl(ATLC_LOG_CUDA(runtime.origCudaFreeAsync, source, op.stream))) != ncclSuccess)
            return result;
        return cudaToNccl(ATLC_LOG_CUDA(runtime.cudaEventRecord, p.local_event, op.stream));
    }

    static ncclResult_t enqueuePreparedGroupSendCompletion(GroupOp const &grouped, PreparedGroupP2P const &p)
    {
        if (grouped.kind != GroupSend)
            return ncclSuccess;
        cudaEvent_t done = NULL;
        ncclResult_t result = openEvent(grouped.comm, p.done.done, &done);
        if (result != ncclSuccess)
            return result;
        return cudaToNccl(ATLC_LOG_CUDA(runtime.cudaStreamWaitEvent, grouped.p2p.stream, done, 0));
    }

    struct CollectiveDescriptor
    {
        memoryDescriptor memory;
        cudaIpcEventHandle_t ready;
        uint64_t bytes;
        uint64_t count;
        uint64_t sequence;
        int kind;
        int datatype;
        int op;
        int root;
    };

    static void reportCollectiveMismatch(VirtualComm *comm,
                                         const std::vector<CollectiveDescriptor> &descriptors)
    {
        if (comm->rank != 0 || descriptors.empty())
            return;
        const CollectiveDescriptor &first = descriptors[0];
        bool sequence = false, kind = false, count = false, datatype = false, op = false, root = false, bytes = false;
        std::fprintf(stderr, "NCCL Fold collective mismatch: comm=%016llx seq=%llu\n\n",
                     (unsigned long long)comm->fingerprint, (unsigned long long)first.sequence);
        for (int rank = 0; rank < comm->ndev; ++rank)
        {
            const CollectiveDescriptor &d = descriptors[rank];
            std::fprintf(stderr, "rank %d: seq=%llu %-13s count=%llu dtype=%s op=%s root=%d bytes=%llu\n",
                         rank, (unsigned long long)d.sequence, collectiveName(d.kind),
                         (unsigned long long)d.count, datatypeName(d.datatype), reductionName(d.op), d.root,
                         (unsigned long long)d.bytes);
            sequence |= d.sequence != first.sequence;
            kind |= d.kind != first.kind;
            count |= d.count != first.count;
            datatype |= d.datatype != first.datatype;
            op |= d.op != first.op;
            root |= d.root != first.root;
            bytes |= d.bytes != first.bytes;
        }
        std::fprintf(stderr, "\ndifference:");
        if (sequence)
            std::fprintf(stderr, " sequence");
        if (kind)
            std::fprintf(stderr, " collective-kind");
        if (count)
            std::fprintf(stderr, " count");
        if (datatype)
            std::fprintf(stderr, " datatype");
        if (op)
            std::fprintf(stderr, " reduction-op");
        if (root)
            std::fprintf(stderr, " root");
        if (bytes)
            std::fprintf(stderr, " byte-count");
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }

    template <ncclRedOp_t Op>
    struct ReductionValue;

    template <>
    struct ReductionValue<ncclSum>
    {
        template <typename T>
        __device__ static T apply(T a, T b)
        {
            return a + b;
        }
        __device__ static __half apply(__half a, __half b)
        {
            return __float2half(__half2float(a) + __half2float(b));
        }
        __device__ static __nv_bfloat16 apply(__nv_bfloat16 a, __nv_bfloat16 b)
        {
            return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
        }
    };

    template <>
    struct ReductionValue<ncclProd>
    {
        template <typename T>
        __device__ static T apply(T a, T b)
        {
            return a * b;
        }
        __device__ static __half apply(__half a, __half b)
        {
            return __float2half(__half2float(a) * __half2float(b));
        }
        __device__ static __nv_bfloat16 apply(__nv_bfloat16 a, __nv_bfloat16 b)
        {
            return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b));
        }
    };

    template <>
    struct ReductionValue<ncclMin>
    {
        template <typename T>
        __device__ static T apply(T a, T b)
        {
            return a < b ? a : b;
        }
        __device__ static __half apply(__half a, __half b)
        {
            return __float2half(fminf(__half2float(a), __half2float(b)));
        }
        __device__ static __nv_bfloat16 apply(__nv_bfloat16 a, __nv_bfloat16 b)
        {
            return __float2bfloat16(fminf(__bfloat162float(a), __bfloat162float(b)));
        }
    };

    template <>
    struct ReductionValue<ncclMax>
    {
        template <typename T>
        __device__ static T apply(T a, T b)
        {
            return a > b ? a : b;
        }
        __device__ static __half apply(__half a, __half b)
        {
            return __float2half(fmaxf(__half2float(a), __half2float(b)));
        }
        __device__ static __nv_bfloat16 apply(__nv_bfloat16 a, __nv_bfloat16 b)
        {
            return __float2bfloat16(fmaxf(__bfloat162float(a), __bfloat162float(b)));
        }
    };

    template <typename T, ncclRedOp_t Op>
    __device__ T reduceValue(T a, T b)
    {
        return ReductionValue<Op>::apply(a, b);
    }
    template <typename T, ncclRedOp_t Op>
    __global__ void reductionKernel(const void *const *sources, T *output,
                                    size_t count, size_t source_offset,
                                    int nranks)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count)
            return;
        T value = static_cast<const T *>(sources[0])[source_offset + i];
        for (int rank = 1; rank < nranks; ++rank)
            value = reduceValue<T, Op>(value, static_cast<const T *>(sources[rank])[source_offset + i]);
        output[i] = value;
    }

    static bool validReduction(ncclRedOp_t op)
    {
        return op == ncclSum || op == ncclProd || op == ncclMin || op == ncclMax;
    }

    template <typename T, ncclRedOp_t Op>
    static ncclResult_t launchReduction(void **device_sources, void *output,
                                        size_t count, size_t source_offset, int nranks, cudaStream_t stream)
    {
        if (count != 0)
            reductionKernel<T, Op><<<(count + 255) / 256, 256, 0, stream>>>(
                (const void *const *)device_sources, (T *)output, count, source_offset, nranks);
        return cudaToNccl(ATLC_LOG_CUDA(cudaGetLastError));
    }

    template <ncclRedOp_t Op>
    static ncclResult_t dispatchReduction(ncclDataType_t datatype, void **device_sources, void *output,
                                          size_t count, size_t source_offset, int nranks, cudaStream_t stream)
    {
        switch (datatype)
        {
        case ncclInt8:
            return launchReduction<int8_t, Op>(device_sources, output, count, source_offset, nranks, stream);
        case ncclUint8:
            return launchReduction<uint8_t, Op>(device_sources, output, count, source_offset, nranks, stream);
        case ncclInt32:
            return launchReduction<int32_t, Op>(device_sources, output, count, source_offset, nranks, stream);
        case ncclUint32:
            return launchReduction<uint32_t, Op>(device_sources, output, count, source_offset, nranks, stream);
        case ncclInt64:
            return launchReduction<int64_t, Op>(device_sources, output, count, source_offset, nranks, stream);
        case ncclUint64:
            return launchReduction<uint64_t, Op>(device_sources, output, count, source_offset, nranks, stream);
        case ncclFloat16:
            return launchReduction<__half, Op>(device_sources, output, count, source_offset, nranks, stream);
        case ncclFloat32:
            return launchReduction<float, Op>(device_sources, output, count, source_offset, nranks, stream);
        case ncclFloat64:
            return launchReduction<double, Op>(device_sources, output, count, source_offset, nranks, stream);
        case ncclBfloat16:
            return launchReduction<__nv_bfloat16, Op>(device_sources, output, count, source_offset, nranks, stream);
        default:
            return ncclInvalidArgument;
        }
    }

    static ncclResult_t enqueueCollective(VirtualComm *comm, const CollectiveArgs &args)
    {
        const uint64_t element_size = sizeofNcclDataType(args.datatype);
        const bool reduction = args.kind == Reduce || args.kind == AllReduce || args.kind == ReduceScatter;
        if (!element_size || (reduction && !validReduction(args.op)))
            return ncclInvalidArgument;
        const bool rooted = args.kind == Broadcast || args.kind == Reduce ||
                            args.kind == Gather || args.kind == Scatter;
        if (rooted && (args.root < 0 || args.root >= comm->ndev))
            return ncclInvalidArgument;
        const bool rank_wide_source = args.kind == ReduceScatter || args.kind == AlltoAll ||
                                      args.kind == Scatter;
        if (rank_wide_source && args.count > SIZE_MAX / (size_t)comm->ndev)
            return ncclInvalidArgument;
        size_t source_count = rank_wide_source ? args.count * (size_t)comm->ndev : args.count;
        bool produces_output = (args.kind != Reduce && args.kind != Gather) || comm->rank == args.root;
        bool has_source = (args.kind != Broadcast && args.kind != Scatter) || comm->rank == args.root;
        if (source_count && ((has_source && !args.sendbuff) || (produces_output && !args.recvbuff)))
            return ncclInvalidArgument;
        if (source_count > SIZE_MAX / element_size)
            return ncclInvalidArgument;
        size_t source_bytes = source_count * element_size;
        if (args.count > SIZE_MAX / element_size)
            return ncclInvalidArgument;
        size_t block_bytes = args.count * element_size;

        /* NCCL explicitly does not support the equal-pointer AlltoAll in-place
         * form.  Gather and Scatter's root-slot aliases are supported below. */
        if (args.kind == AlltoAll && source_bytes && args.sendbuff == args.recvbuff)
            return ncclInvalidArgument;

        CollectiveDescriptor local = {};
        local.bytes = source_bytes;
        local.count = args.count;
        local.sequence = comm->collective_sequence++;
        local.kind = args.kind;
        local.datatype = args.datatype;
        local.op = reduction ? args.op : 0;
        local.root = args.root;
        AllocationKind allocation_kind = LegacyAllocation;
        void *allocation = has_source ? getAllocation(const_cast<void *>(args.sendbuff), source_bytes,
                                                      &local.memory.offset, &allocation_kind)
                                      : NULL;
        if (source_bytes && has_source && !allocation)
            return ncclInvalidArgument;
        const bool rank_wide_output = args.kind == AllGather || args.kind == AlltoAll || args.kind == Gather;
        if (rank_wide_output && block_bytes > SIZE_MAX / (size_t)comm->ndev)
            return ncclInvalidArgument;
        size_t output_bytes = rank_wide_output ? block_bytes * comm->ndev : block_bytes;
        if (produces_output && output_bytes && !getAllocation(args.recvbuff, output_bytes, NULL))
            return ncclInvalidArgument;
        if (source_bytes && has_source)
        {
            ncclResult_t result = exportMemory(allocation, allocation_kind, &local.memory, local.memory.offset);
            if (result != ncclSuccess)
                return result;
        }
        cudaEvent_t ready = NULL;
        size_t ready_slot = 0;
        ncclResult_t result = leaseEvent(comm, &ready_slot, &ready);
        if (result != ncclSuccess)
            return result;
        if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaEventRecord, ready, args.stream))) != ncclSuccess)
            return result;
        if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaIpcGetEventHandle, &local.ready, ready))) != ncclSuccess)
            return result;

        std::vector<CollectiveDescriptor> descriptors(comm->ndev);
        publishSnapshot(comm, DiagCollective, local.sequence, "collective-metadata", DiagWaiting,
                        args.kind, args.count, args.datatype, reduction ? args.op : -1, args.root);
        if ((result = timedAllgather(&local, sizeof(local), MPI_BYTE, descriptors.data(), sizeof(local),
                                     MPI_BYTE, comm, "collective-metadata", local.sequence)) != ncclSuccess)
            return result;
        bool descriptor_mismatch = false;
        for (int rank = 0; rank < comm->ndev; ++rank)
        {
            const CollectiveDescriptor &d = descriptors[rank];
            descriptor_mismatch |= d.sequence != local.sequence || d.kind != local.kind ||
                                   d.count != local.count || d.datatype != local.datatype || d.op != local.op ||
                                   d.root != local.root || d.bytes != local.bytes;
        }
        if (descriptor_mismatch)
        {
            reportCollectiveMismatch(comm, descriptors);
            return ncclInvalidUsage;
        }
        std::vector<void *> sources(comm->ndev);
        std::vector<void *> imported_pool_pointers;
        for (int rank = 0; rank < comm->ndev; ++rank)
        {
            const CollectiveDescriptor &d = descriptors[rank];
            bool need_rank = (args.kind != Broadcast && args.kind != Scatter) || rank == args.root;
            if (!need_rank)
                continue;
            if (rank == comm->rank)
                sources[rank] = const_cast<void *>(args.sendbuff);
            else if (source_bytes)
            {
                cudaEvent_t event = NULL;
                void *base = NULL;
                bool temporary = false;
                if ((result = openEvent(comm, d.ready, &event)) != ncclSuccess)
                    return result;
                if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaStreamWaitEvent, args.stream, event, 0))) != ncclSuccess)
                    return result;
                if ((result = importMemory(comm, rank, d.memory, &base, &temporary)) != ncclSuccess)
                    return result;
                if (temporary)
                    imported_pool_pointers.push_back(base);
                sources[rank] = (char *)base + d.memory.offset;
            }
        }
        if ((result = timedBarrier(comm, comm->mpi_comm, "collective-ready-leases", local.sequence)) != ncclSuccess)
            return result;
        releaseEvent(comm, ready_slot);

        if (args.kind == Broadcast)
        {
            if (source_bytes && args.recvbuff != sources[args.root])
                result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaMemcpyAsync, args.recvbuff, sources[args.root], source_bytes, cudaMemcpyDeviceToDevice, args.stream));
        }
        else if (args.kind == AllGather)
        {
            for (int rank = 0; rank < comm->ndev && result == ncclSuccess; ++rank)
            {
                void *destination = (char *)args.recvbuff + rank * source_bytes;
                if (source_bytes && destination != sources[rank])
                    result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaMemcpyAsync, destination, sources[rank], source_bytes, cudaMemcpyDeviceToDevice, args.stream));
            }
        }
        else if (args.kind == AlltoAll)
        {
            for (int rank = 0; rank < comm->ndev && result == ncclSuccess; ++rank)
            {
                if (block_bytes)
                {
                    const void *source = (const char *)sources[rank] + (size_t)comm->rank * block_bytes;
                    void *destination = (char *)args.recvbuff + (size_t)rank * block_bytes;
                    result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaMemcpyAsync, destination, source, block_bytes, cudaMemcpyDeviceToDevice, args.stream));
                }
            }
        }
        else if (args.kind == Gather)
        {
            if (comm->rank == args.root)
            {
                for (int rank = 0; rank < comm->ndev && result == ncclSuccess; ++rank)
                {
                    void *destination = (char *)args.recvbuff + (size_t)rank * block_bytes;
                    if (block_bytes && destination != sources[rank])
                        result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaMemcpyAsync, destination, sources[rank], block_bytes, cudaMemcpyDeviceToDevice, args.stream));
                }
            }
        }
        else if (args.kind == Scatter)
        {
            if (block_bytes)
            {
                const void *source = (const char *)sources[args.root] + (size_t)comm->rank * block_bytes;
                if (args.recvbuff != source)
                    result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaMemcpyAsync, args.recvbuff, source, block_bytes, cudaMemcpyDeviceToDevice, args.stream));
            }
        }
        else if (args.kind != Reduce || comm->rank == args.root)
        {
            void **device_sources = NULL;
            if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.origCudaMallocAsync, (void **)&device_sources, sizeof(void *) * comm->ndev, args.stream))) != ncclSuccess)
                return result;
            ++comm->scratch_in_flight;
            ++comm->scratch_allocations_submitted;
            comm->scratch_high_water = std::max(comm->scratch_high_water, comm->scratch_in_flight);
            result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaMemcpyAsync, device_sources, sources.data(), sizeof(void *) * comm->ndev, cudaMemcpyHostToDevice, args.stream));
            if (result != ncclSuccess)
            {
                if (runtime.origCudaFreeAsync(device_sources, args.stream) == cudaSuccess)
                    --comm->scratch_in_flight;
                return result;
            }
            size_t offset = args.kind == ReduceScatter ? args.count * (size_t)comm->rank : 0;
            switch (args.op)
            {
            case ncclSum:
                result = dispatchReduction<ncclSum>(args.datatype, device_sources, args.recvbuff, args.count, offset, comm->ndev, args.stream);
                break;
            case ncclProd:
                result = dispatchReduction<ncclProd>(args.datatype, device_sources, args.recvbuff, args.count, offset, comm->ndev, args.stream);
                break;
            case ncclMin:
                result = dispatchReduction<ncclMin>(args.datatype, device_sources, args.recvbuff, args.count, offset, comm->ndev, args.stream);
                break;
            case ncclMax:
                result = dispatchReduction<ncclMax>(args.datatype, device_sources, args.recvbuff, args.count, offset, comm->ndev, args.stream);
                break;
            default:
                result = ncclInvalidArgument;
                break;
            }
            cudaError_t free_error = ATLC_LOG_CUDA(runtime.origCudaFreeAsync, device_sources, args.stream);
            if (free_error == cudaSuccess)
                --comm->scratch_in_flight;
            if (result == ncclSuccess)
                result = cudaToNccl(free_error);
        }
        if (result != ncclSuccess)
            return result;

        /* Pool imports are operation-scoped.  Queue each importing free after
         * its final read and before the done event exported below. */
        for (size_t i = 0; i < imported_pool_pointers.size(); ++i)
        {
            result = cudaToNccl(ATLC_LOG_CUDA(runtime.origCudaFreeAsync, imported_pool_pointers[i], args.stream));
            if (result != ncclSuccess)
                return result;
        }

        /* Host metadata exchange is blocking, but GPU execution remains asynchronous.
         * Done events make later work on every source stream wait until all remote
         * readers have enqueued their use, matching NCCL's buffer-reuse ordering. */
        cudaEvent_t done = NULL;
        size_t done_slot = 0;
        if ((result = leaseEvent(comm, &done_slot, &done)) != ncclSuccess)
            return result;
        if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaEventRecord, done, args.stream))) != ncclSuccess)
            return result;
        cudaIpcEventHandle_t local_done;
        if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaIpcGetEventHandle, &local_done, done))) != ncclSuccess)
            return result;
        std::vector<cudaIpcEventHandle_t> dones(comm->ndev);
        if ((result = timedAllgather(&local_done, sizeof(local_done), MPI_BYTE, dones.data(), sizeof(local_done),
                                     MPI_BYTE, comm, "collective-done", local.sequence)) != ncclSuccess)
            return result;
        for (int rank = 0; rank < comm->ndev; ++rank)
            if (rank != comm->rank)
            {
                cudaEvent_t event = NULL;
                if ((result = openEvent(comm, dones[rank], &event)) != ncclSuccess)
                    return result;
                if ((result = cudaToNccl(ATLC_LOG_CUDA(runtime.cudaStreamWaitEvent, args.stream, event, 0))) != ncclSuccess)
                    return result;
            }
        if ((result = timedBarrier(comm, comm->mpi_comm, "collective-done-leases", local.sequence)) != ncclSuccess)
            return result;
        releaseEvent(comm, done_slot);
        publishSnapshot(comm, DiagCollective, local.sequence, "collective-complete", DiagCompleted,
                        args.kind, args.count, args.datatype, reduction ? args.op : -1, args.root);
        return ncclSuccess;
    }

    static ncclResult_t groupEnd()
    {
        if (group_state.depth == 0)
            return ncclInvalidUsage;
        if (--group_state.depth != 0)
            return ncclSuccess;

        /* Detach first so every error path leaves this thread's group reusable. */
        std::vector<GroupOp> operations;
        operations.swap(group_state.operations);
        for (size_t i = 0; i < operations.size(); ++i)
            if (getVirtualComm(reinterpret_cast<ncclComm_t>(operations[i].comm)) == NULL)
                return ncclInvalidArgument;

        std::vector<PreparedGroupP2P> prepared;
        ncclResult_t result = prepareGroupedP2P(operations, prepared);
        if (result != ncclSuccess)
            return result;
        for (size_t i = 0; i < operations.size(); ++i)
        {
            GroupOp const &op = operations[i];
            result = op.kind == GroupCollective
                         ? enqueueCollective(op.comm, op.collective)
                         : enqueuePreparedGroupP2P(op, prepared[i]);
            if (result != ncclSuccess)
                return result;
        }
        /* Snapshots make grouped sends consume their user buffers at the send's
         * exact stream position.  Completion waits can therefore be appended
         * after all group work, avoiding Send/collective/Recv dependency cycles. */
        for (size_t i = 0; i < operations.size(); ++i)
        {
            result = enqueuePreparedGroupSendCompletion(operations[i], prepared[i]);
            if (result != ncclSuccess)
                return result;
        }
        /* Both endpoints acknowledge every grouped P2P operation after their
         * ready/done waits have been submitted.  Only then can either endpoint
         * re-record its local event generation. */
        std::vector<MPI_Request> acknowledgements;
        std::vector<uint64_t> received(operations.size());
        acknowledgements.reserve(operations.size() * 2);
        for (size_t i = 0; i < operations.size(); ++i)
        {
            if (operations[i].kind == GroupCollective)
                continue;
            MPI_Request request = MPI_REQUEST_NULL;
            GroupOp const &op = operations[i];
            if ((result = mpiToNccl(ATLC_LOG_MPI(MPI_Isend, &op.p2p.sequence, 1, MPI_UINT64_T, op.p2p.peer,
                                                 ackTag, op.comm->mpi_comm, &request))) != ncclSuccess)
                return result;
            acknowledgements.push_back(request);
            if ((result = mpiToNccl(ATLC_LOG_MPI(MPI_Irecv, &received[i], 1, MPI_UINT64_T, op.p2p.peer,
                                                 ackTag, op.comm->mpi_comm, &request))) != ncclSuccess)
                return result;
            acknowledgements.push_back(request);
        }
        if (!acknowledgements.empty())
        {
            VirtualComm *comm = operations.empty() ? NULL : operations[0].comm;
            uint64_t sequence = operations.empty() ? 0 : operations[0].p2p.sequence;
            result = timedWait(acknowledgements.data(), (int)acknowledgements.size(), comm,
                               comm ? comm->mpi_comm : MPI_COMM_WORLD, "group-event-ack", sequence);
            if (result != ncclSuccess)
                return result;
        }
        for (size_t i = 0; i < operations.size(); ++i)
            if (operations[i].kind != GroupCollective)
            {
                if (received[i] != operations[i].p2p.sequence)
                    return ncclInvalidUsage;
                releaseEvent(operations[i].comm, prepared[i].event_slot);
            }
        return ncclSuccess;
    }

    static ncclResult_t collective(CollectiveKind kind, const void *sendbuff, void *recvbuff,
                                   size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root,
                                   ncclComm_t handle, cudaStream_t stream)
    {
        VirtualComm *comm = getVirtualComm(handle);
        if (!comm)
            return ncclInvalidArgument;
        CollectiveArgs args = {kind, sendbuff, recvbuff, count, datatype, op, root, stream};
        if (group_state.depth != 0)
        {
            GroupOp grouped = {};
            grouped.kind = GroupCollective;
            grouped.comm = comm;
            grouped.collective = args;
            group_state.operations.push_back(grouped);
            return ncclSuccess;
        }
        return enqueueCollective(comm, args);
    }

    static ncclResult_t broadcast(const void *sendbuff, void *recvbuff, size_t count,
                                  ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
    {
        return collective(Broadcast, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream);
    }
    static ncclResult_t bcast(void *buff, size_t count, ncclDataType_t datatype, int root,
                              ncclComm_t comm, cudaStream_t stream)
    {
        return broadcast(buff, buff, count, datatype, root, comm, stream);
    }
    static ncclResult_t allGather(const void *sendbuff, void *recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
    {
        return collective(AllGather, sendbuff, recvbuff, count, datatype, ncclSum, -1, comm, stream);
    }
    static ncclResult_t reduce(const void *sendbuff, void *recvbuff, size_t count,
                               ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream)
    {
        return collective(Reduce, sendbuff, recvbuff, count, datatype, op, root, comm, stream);
    }
    static ncclResult_t allReduce(const void *sendbuff, void *recvbuff, size_t count,
                                  ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
    {
        return collective(AllReduce, sendbuff, recvbuff, count, datatype, op, -1, comm, stream);
    }
    static ncclResult_t reduceScatter(const void *sendbuff, void *recvbuff, size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream)
    {
        return collective(ReduceScatter, sendbuff, recvbuff, count, datatype, op, -1, comm, stream);
    }

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    static ncclResult_t alltoAll(const void *sendbuff, void *recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
    {
        return collective(AlltoAll, sendbuff, recvbuff, count, datatype, ncclSum, -1, comm, stream);
    }
    static ncclResult_t gather(const void *sendbuff, void *recvbuff, size_t count,
                               ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
    {
        return collective(Gather, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream);
    }
    static ncclResult_t scatter(const void *sendbuff, void *recvbuff, size_t count,
                                ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
    {
        return collective(Scatter, sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream);
    }
#endif

    static ncclResult_t send(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
                             ncclComm_t handle, cudaStream_t stream)
    {
        VirtualComm *comm = getVirtualComm(handle);
        if (comm == NULL)
            return ncclInvalidArgument;
        if (peer < 0 || peer >= comm->ndev)
            return ncclInvalidArgument;
        uint64_t sequence;
        {
            std::lock_guard<std::mutex> lock(comm->sequence_mutex);
            sequence = comm->next_send_sequence[peer]++;
        }
        sendRecvArgs_t op = {const_cast<void *>(sendbuff), (uint64_t)count, (int)datatype, peer, stream, sequence};
        if (group_state.depth != 0)
        {
            GroupOp grouped = {};
            grouped.kind = GroupSend;
            grouped.comm = comm;
            grouped.p2p = op;
            group_state.operations.push_back(grouped);
            return ncclSuccess;
        }
        std::vector<sendRecvArgs_t> sends(1, op), recvs;
        return enqueueP2P(comm, sends, recvs);
    }

    static ncclResult_t recv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
                             ncclComm_t handle, cudaStream_t stream)
    {
        VirtualComm *comm = getVirtualComm(handle);
        if (comm == NULL)
            return ncclInvalidArgument;
        if (peer < 0 || peer >= comm->ndev)
            return ncclInvalidArgument;
        uint64_t sequence;
        {
            std::lock_guard<std::mutex> lock(comm->sequence_mutex);
            sequence = comm->next_recv_sequence[peer]++;
        }
        sendRecvArgs_t op = {recvbuff, (uint64_t)count, (int)datatype, peer, stream, sequence};
        if (group_state.depth != 0)
        {
            GroupOp grouped = {};
            grouped.kind = GroupRecv;
            grouped.comm = comm;
            grouped.p2p = op;
            group_state.operations.push_back(grouped);
            return ncclSuccess;
        }
        std::vector<sendRecvArgs_t> sends, recvs(1, op);
        return enqueueP2P(comm, sends, recvs);
    }

    static ncclResult_t commCount(ncclComm_t handle, int *count)
    {
        VirtualComm *comm = getVirtualComm(handle);
        if (comm == NULL || count == NULL)
            return ncclInvalidArgument;
        *count = comm->ndev;
        return ncclSuccess;
    }

    static ncclResult_t commUserRank(ncclComm_t handle, int *rank)
    {
        VirtualComm *comm = getVirtualComm(handle);
        if (comm == NULL || rank == NULL)
            return ncclInvalidArgument;
        *rank = comm->rank;
        return ncclSuccess;
    }

    static ncclResult_t commDestroy(ncclComm_t handle)
    {
        if (handle == NULL)
            return ncclInvalidArgument;
        VirtualComm *comm = reinterpret_cast<VirtualComm *>(handle);
        {
            std::lock_guard<std::mutex> lock(runtime.communicator_mutex);
            std::unordered_set<VirtualComm *>::iterator found = runtime.communicators.find(comm);
            if (found == runtime.communicators.end() || comm->magic != VirtualComm::MAGIC)
                return ncclInvalidArgument;
            runtime.communicators.erase(found);
            comm->magic = 0;
        }

        ncclResult_t result = ncclSuccess;
        if (!comm->diagnostic_directory.empty())
        {
            unlink(snapshotPath(comm, comm->rank).c_str());
            /* The directory removal is best-effort: another rank may still be
             * destroying, and leaving an empty directory is harmless. */
            rmdir(comm->diagnostic_directory.c_str());
        }
        if (resourceStatsEnabled())
            std::fprintf(stderr, "NCCL Fold resources comm=%p rank=%d owned_events_created=%zu "
                                 "event_pool_peak=%zu imported_events_opened=%zu imported_event_cache=%zu "
                                 "ipc_memory_mappings=%zu reduction_scratch_allocations=%zu "
                                 "reduction_scratch_high_water=%zu retained_ipc_allocations=%zu\n",
                         (void *)comm, comm->rank, comm->event_pool.size(), comm->event_pool_peak,
                         comm->imported_event_cache.size(), comm->imported_event_cache.size(),
                         comm->ipc_mappings.size(), comm->scratch_allocations_submitted,
                         comm->scratch_high_water,
                         comm->retained_ipc_allocations.size());
        for (size_t i = 0; i < comm->event_pool.size(); ++i)
            if (runtime.cudaEventDestroy(comm->event_pool[i].event) != cudaSuccess)
                result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->imported_event_cache.size(); ++i)
            if (runtime.cudaEventDestroy(comm->imported_event_cache[i].event) != cudaSuccess)
                result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->ipc_mappings.size(); ++i)
            if (runtime.cudaIpcCloseMemHandle(comm->ipc_mappings[i].pointer) != cudaSuccess)
                result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->imported_pools.size(); ++i)
            if (comm->imported_pools[i] && cudaMemPoolDestroy(comm->imported_pools[i]) != cudaSuccess)
                result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->retained_ipc_allocations.size(); ++i)
            if (runtime.origCudaFree(comm->retained_ipc_allocations[i]) != cudaSuccess)
                result = ncclUnhandledCudaError;
        if (comm->mpi_comm != MPI_COMM_NULL && MPI_Comm_free(&comm->mpi_comm) != MPI_SUCCESS && result == ncclSuccess)
            result = ncclSystemError;
        delete comm;
        return result;
    }

    static void *find_symbol_offset(const char *exe_path, const char *symbol_name)
    {
        int fd = open(exe_path, O_RDONLY);
        elf_version(EV_CURRENT);
        Elf *e = elf_begin(fd, ELF_C_READ, NULL);

        size_t shstrndx;
        elf_getshdrstrndx(e, &shstrndx);

        Elf_Scn *scn = NULL;
        GElf_Shdr shdr;

        while ((scn = elf_nextscn(e, scn)) != NULL)
        {
            gelf_getshdr(scn, &shdr);
            if (shdr.sh_type == SHT_SYMTAB)
            {
                Elf_Data *data = elf_getdata(scn, NULL);
                int count = shdr.sh_size / shdr.sh_entsize;

                for (int i = 0; i < count; ++i)
                {
                    GElf_Sym sym;
                    gelf_getsym(data, i, &sym);
                    const char *name = elf_strptr(e, shdr.sh_link, sym.st_name);
                    if (strcmp(name, symbol_name) == 0)
                    {
                        return (void *)sym.st_value;
                    }
                }
            }
        }
        return 0;
    }

    static void *find_symbol_offset_or_dlsym(char const *exe_path, const char *symbol_name)
    {
        void *addr = find_symbol_offset(exe_path, symbol_name);
        if (addr)
        {
            return addr;
        }
        return dlsym(NULL, symbol_name);
    }

    __attribute__((constructor)) void init()
    {

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

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaMalloc"), (gpointer)nccl_fold::cudaMalloc, NULL, (gpointer *)&runtime.origCudaMalloc);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaMallocAsync"), (gpointer)nccl_fold::cudaMallocAsync, NULL, (gpointer *)&runtime.origCudaMallocAsync);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaFree"), (gpointer)nccl_fold::cudaFree, NULL, (gpointer *)&runtime.origCudaFree);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaFreeAsync"), (gpointer)nccl_fold::cudaFreeAsync, NULL, (gpointer *)&runtime.origCudaFreeAsync);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaSetDevice"), (gpointer)nccl_fold::cudaSetDevice, NULL, (gpointer *)&runtime.origCudaSetDevice);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclGetUniqueId, (gpointer)nccl_fold::getUniqueId, NULL, (gpointer *)&runtime.origNcclGetUniqueId);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclCommInitRank, (gpointer)nccl_fold::commInitRank, NULL, (gpointer *)&runtime.origNcclCommInitRank);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclGroupStart, (gpointer)nccl_fold::groupStart, NULL, (gpointer *)&runtime.origNcclGroupStart);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclGroupEnd, (gpointer)nccl_fold::groupEnd, NULL, (gpointer *)&runtime.origNcclGroupEnd);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclSend, (gpointer)nccl_fold::send, NULL, (gpointer *)&runtime.origNcclSend);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclRecv, (gpointer)nccl_fold::recv, NULL, (gpointer *)&runtime.origNcclRecv);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclBroadcast, (gpointer)nccl_fold::broadcast, NULL, (gpointer *)&runtime.origNcclBroadcast);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclBcast, (gpointer)nccl_fold::bcast, NULL, (gpointer *)&runtime.origNcclBcast);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclAllGather, (gpointer)nccl_fold::allGather, NULL, (gpointer *)&runtime.origNcclAllGather);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclReduce, (gpointer)nccl_fold::reduce, NULL, (gpointer *)&runtime.origNcclReduce);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclAllReduce, (gpointer)nccl_fold::allReduce, NULL, (gpointer *)&runtime.origNcclAllReduce);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclReduceScatter, (gpointer)nccl_fold::reduceScatter, NULL, (gpointer *)&runtime.origNcclReduceScatter);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclAlltoAll, (gpointer)nccl_fold::alltoAll, NULL, (gpointer *)&runtime.origNcclAlltoAll);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclGather, (gpointer)nccl_fold::gather, NULL, (gpointer *)&runtime.origNcclGather);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclScatter, (gpointer)nccl_fold::scatter, NULL, (gpointer *)&runtime.origNcclScatter);
#endif

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclCommDestroy, (gpointer)nccl_fold::commDestroy, NULL, (gpointer *)&runtime.origNcclCommDestroy);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclCommCount, (gpointer)nccl_fold::commCount, NULL, (gpointer *)&runtime.origNcclCommCount);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, nccl_fold::interceptor, (gpointer)ncclCommUserRank, (gpointer)nccl_fold::commUserRank, NULL, (gpointer *)&runtime.origNcclCommUserRank);

        gum_interceptor_end_transaction(interceptor);
    }

    __attribute__((destructor)) void deinit()
    {
        gum_interceptor_begin_transaction(interceptor);
        // gum_interceptor_revert(interceptor, (void*)(func));  /* specify original address */
        gum_interceptor_end_transaction(interceptor);

        g_object_unref(interceptor);
        gum_deinit_embedded();
    }

}
