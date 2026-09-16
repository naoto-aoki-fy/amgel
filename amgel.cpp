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

namespace amgel {

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
        static constexpr uint64_t MAGIC = UINT64_C(0x414d47454c434f4d);
        uint64_t magic;
        ncclUniqueId unique_id;
        std::vector<cudaEvent_t> owned_events;
        std::vector<cudaEvent_t> imported_events;
        std::vector<importedMemory> ipc_mappings;
        std::vector<uint64_t> next_send_sequence;
        std::vector<uint64_t> next_recv_sequence;
        std::vector<void*> scratch_allocations;
        uint64_t collective_sequence;
        MPI_Comm mpi_comm;
        int rank;
        int ndev;
        std::mutex sequence_mutex;

        VirtualComm() : magic(MAGIC), collective_sequence(0), mpi_comm(MPI_COMM_NULL), rank(-1), ndev(0) {}
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

    struct GroupEntry {
        VirtualComm* comm;
        std::vector<sendRecvArgs_t> send_args;
        std::vector<sendRecvArgs_t> recv_args;
        std::vector<CollectiveArgs> collective_args;
    };
    struct GroupState {
        bool active = false;
        std::vector<GroupEntry> entries;
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
        if (MPI_Comm_dup(MPI_COMM_WORLD, &virtual_comm->mpi_comm) != MPI_SUCCESS) {
            delete virtual_comm;
            return ncclSystemError;
        }
        virtual_comm->rank = rank;
        virtual_comm->ndev = ndev;
        int mpi_rank = -1, mpi_size = 0;
        MPI_Comm_rank(virtual_comm->mpi_comm, &mpi_rank);
        MPI_Comm_size(virtual_comm->mpi_comm, &mpi_size);
        if (mpi_rank != rank || mpi_size != ndev) { MPI_Comm_free(&virtual_comm->mpi_comm); delete virtual_comm; return ncclInvalidArgument; }
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
        if (group_state.active) return ncclInvalidUsage;
        group_state.entries.clear();
        group_state.active = true;
        return ncclSuccess;
    }

    static bool debugEnabled() {
        static int enabled = std::getenv("AMGEL_DEBUG_P2P") != NULL;
        return enabled != 0;
    }

    static ncclResult_t cudaCheck(cudaError_t error, const char* operation) {
        if (error == cudaSuccess) return ncclSuccess;
        std::fprintf(stderr, "AMGeL: %s failed: %s\n", operation, cudaGetErrorString(error));
        return ncclUnhandledCudaError;
    }

    static ncclResult_t mpiCheck(int error, const char* operation) {
        if (error == MPI_SUCCESS) return ncclSuccess;
        std::fprintf(stderr, "AMGeL: %s failed with MPI error %d\n", operation, error);
        return ncclSystemError;
    }

    enum { readyTag = 17001, doneTag = 17002 };

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

    /*
     * MPI is only the control plane here.  Every exported event is a fresh event,
     * and its record has been submitted before its handle is sent.  Events and IPC
     * mappings deliberately remain owned by the communicator: reusing/destroying
     * them at group end could race a wait or copy which is still queued remotely.
     */
    static ncclResult_t enqueueP2P(VirtualComm* comm, const std::vector<sendRecvArgs_t>& send_args,
                                    const std::vector<sendRecvArgs_t>& recv_args) {
        const size_t send_count = send_args.size();
        const size_t recv_count = recv_args.size();
        std::vector<readyMessage> outgoing(send_count);
        std::vector<readyMessage> incoming(recv_count);
        std::vector<doneMessage> outgoing_done(recv_count);
        std::vector<doneMessage> incoming_done(send_count);
        std::vector<MPI_Request> requests(send_count + recv_count);

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
            result = cudaCheck(runtime.cudaEventCreateWithFlags(&ready, cudaEventInterprocess | cudaEventDisableTiming), "runtime.cudaEventCreateWithFlags(ready)");
            if (result != ncclSuccess) return result;
            comm->owned_events.push_back(ready);
            result = cudaCheck(runtime.cudaEventRecord(ready, op.stream), "runtime.cudaEventRecord(ready)");
            if (result != ncclSuccess) return result;
            result = cudaCheck(runtime.cudaIpcGetEventHandle(&message.ready, ready), "runtime.cudaIpcGetEventHandle(ready)");
            if (result != ncclSuccess) return result;
            message.bytes = op.count * type_size;
            message.sequence = op.sequence;
            if (debugEnabled()) std::fprintf(stderr, "AMGeL comm=%p rank=%d peer=%d seq=%llu ready=%p stream=%p send\n",
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
            ncclResult_t result = cudaCheck(runtime.cudaIpcOpenEventHandle(&ready, message.ready), "runtime.cudaIpcOpenEventHandle(ready)");
            if (result != ncclSuccess) return result;
            comm->imported_events.push_back(ready);
            result = openMemory(comm, message.memory.handle, &source);
            if (result != ncclSuccess) return result;
            result = cudaCheck(runtime.cudaStreamWaitEvent(op.stream, ready, 0), "runtime.cudaStreamWaitEvent(ready)");
            if (result != ncclSuccess) return result;
            if (message.bytes == recv_bytes) {
                result = cudaCheck(runtime.cudaMemcpyAsync(op.buff, (char*)source + message.memory.offset, recv_bytes,
                    cudaMemcpyDeviceToDevice, op.stream), "runtime.cudaMemcpyAsync(P2P)");
                if (result != ncclSuccess) return result;
            }
            result = cudaCheck(runtime.cudaEventCreateWithFlags(&done, cudaEventInterprocess | cudaEventDisableTiming), "runtime.cudaEventCreateWithFlags(done)");
            if (result != ncclSuccess) return result;
            comm->owned_events.push_back(done);
            result = cudaCheck(runtime.cudaEventRecord(done, op.stream), "runtime.cudaEventRecord(done)");
            if (result != ncclSuccess) return result;
            result = cudaCheck(runtime.cudaIpcGetEventHandle(&outgoing_done[i].done, done), "runtime.cudaIpcGetEventHandle(done)");
            if (result != ncclSuccess) return result;
            outgoing_done[i].sequence = message.sequence;
            if (debugEnabled()) std::fprintf(stderr, "AMGeL comm=%p rank=%d peer=%d seq=%llu done=%p stream=%p recv\n",
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
        for (size_t i = 0; i < send_count; ++i) {
            if (incoming_done[i].sequence != send_args[i].sequence) deferred_error = ncclInvalidArgument;
            cudaEvent_t done = NULL;
            ncclResult_t result = cudaCheck(runtime.cudaIpcOpenEventHandle(&done, incoming_done[i].done), "runtime.cudaIpcOpenEventHandle(done)");
            if (result != ncclSuccess) return result;
            comm->imported_events.push_back(done);
            result = cudaCheck(runtime.cudaStreamWaitEvent(send_args[i].stream, done, 0), "runtime.cudaStreamWaitEvent(done)");
            if (result != ncclSuccess) return result;
        }
        return deferred_error;
    }

    static GroupEntry* groupEntry(VirtualComm* comm) {
        for (size_t i = 0; i < group_state.entries.size(); ++i) {
            if (group_state.entries[i].comm == comm) return &group_state.entries[i];
        }
        group_state.entries.push_back(GroupEntry());
        group_state.entries.back().comm = comm;
        return &group_state.entries.back();
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
        ncclResult_t result = cudaCheck(runtime.cudaEventCreateWithFlags(&ready, cudaEventInterprocess | cudaEventDisableTiming), "cudaEventCreateWithFlags(collective ready)");
        if (result != ncclSuccess) return result;
        comm->owned_events.push_back(ready);
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
                if ((result = cudaCheck(runtime.cudaIpcOpenEventHandle(&event, d.ready), "cudaIpcOpenEventHandle(collective ready)")) != ncclSuccess) return result;
                comm->imported_events.push_back(event);
                if ((result = cudaCheck(runtime.cudaStreamWaitEvent(args.stream, event, 0), "cudaStreamWaitEvent(collective ready)")) != ncclSuccess) return result;
                if ((result = openMemory(comm, d.memory.handle, &base)) != ncclSuccess) return result;
                sources[rank] = (char*)base + d.memory.offset;
            }
        }

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
            if ((result = cudaCheck(runtime.origCudaMalloc((void**)&device_sources, sizeof(void*) * comm->ndev), "cudaMalloc(reduction sources)")) != ncclSuccess) return result;
            comm->scratch_allocations.push_back(device_sources);
            if ((result = cudaCheck(runtime.cudaMemcpyAsync(device_sources, sources.data(), sizeof(void*) * comm->ndev, cudaMemcpyHostToDevice, args.stream), "cudaMemcpyAsync(reduction sources)")) != ncclSuccess) return result;
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
        }
        if (result != ncclSuccess) return result;

        /* Host metadata exchange is blocking, but GPU execution remains asynchronous.
         * Done events make later work on every source stream wait until all remote
         * readers have enqueued their use, matching NCCL's buffer-reuse ordering. */
        cudaEvent_t done = NULL;
        if ((result = cudaCheck(runtime.cudaEventCreateWithFlags(&done, cudaEventInterprocess | cudaEventDisableTiming), "cudaEventCreateWithFlags(collective done)")) != ncclSuccess) return result;
        comm->owned_events.push_back(done);
        if ((result = cudaCheck(runtime.cudaEventRecord(done, args.stream), "cudaEventRecord(collective done)")) != ncclSuccess) return result;
        cudaIpcEventHandle_t local_done;
        if ((result = cudaCheck(runtime.cudaIpcGetEventHandle(&local_done, done), "cudaIpcGetEventHandle(collective done)")) != ncclSuccess) return result;
        std::vector<cudaIpcEventHandle_t> dones(comm->ndev);
        if ((result = mpiCheck(MPI_Allgather(&local_done, sizeof(local_done), MPI_BYTE, dones.data(), sizeof(local_done), MPI_BYTE, comm->mpi_comm), "MPI_Allgather(collective done)")) != ncclSuccess) return result;
        for (int rank=0; rank<comm->ndev; ++rank) if (rank != comm->rank) {
            cudaEvent_t event = NULL;
            if ((result = cudaCheck(runtime.cudaIpcOpenEventHandle(&event, dones[rank]), "cudaIpcOpenEventHandle(collective done)")) != ncclSuccess) return result;
            comm->imported_events.push_back(event);
            if ((result = cudaCheck(runtime.cudaStreamWaitEvent(args.stream, event, 0), "cudaStreamWaitEvent(collective done)")) != ncclSuccess) return result;
        }
        return ncclSuccess;
    }

    static ncclResult_t groupEnd() {
        if (!group_state.active) return ncclInvalidUsage;
        ncclResult_t result = ncclSuccess;
        for (size_t i = 0; i < group_state.entries.size(); ++i) {
            GroupEntry& entry = group_state.entries[i];
            if (getVirtualComm(reinterpret_cast<ncclComm_t>(entry.comm)) == NULL) {
                if (result == ncclSuccess) result = ncclInvalidArgument;
                continue;
            }
            ncclResult_t current = enqueueP2P(entry.comm, entry.send_args, entry.recv_args);
            if (result == ncclSuccess && current != ncclSuccess) result = current;
            for (size_t j = 0; j < entry.collective_args.size(); ++j) {
                current = enqueueCollective(entry.comm, entry.collective_args[j]);
                if (result == ncclSuccess && current != ncclSuccess) result = current;
                if (current != ncclSuccess) break;
            }
        }
        group_state.entries.clear();
        group_state.active = false;
        return result;
    }

    static ncclResult_t collective(CollectiveKind kind, const void* sendbuff, void* recvbuff,
            size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root,
            ncclComm_t handle, cudaStream_t stream) {
        VirtualComm* comm = getVirtualComm(handle);
        if (!comm) return ncclInvalidArgument;
        CollectiveArgs args = {kind, sendbuff, recvbuff, count, datatype, op, root, stream};
        if (group_state.active) { groupEntry(comm)->collective_args.push_back(args); return ncclSuccess; }
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
        if (group_state.active) {
            groupEntry(comm)->send_args.push_back(op);
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
        if (group_state.active) {
            groupEntry(comm)->recv_args.push_back(op);
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
        for (size_t i = 0; i < comm->owned_events.size(); ++i)
            if (runtime.cudaEventDestroy(comm->owned_events[i]) != cudaSuccess) result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->imported_events.size(); ++i)
            if (runtime.cudaEventDestroy(comm->imported_events[i]) != cudaSuccess) result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->ipc_mappings.size(); ++i)
            if (runtime.cudaIpcCloseMemHandle(comm->ipc_mappings[i].pointer) != cudaSuccess) result = ncclUnhandledCudaError;
        for (size_t i = 0; i < comm->scratch_allocations.size(); ++i)
            if (runtime.origCudaFree(comm->scratch_allocations[i]) != cudaSuccess) result = ncclUnhandledCudaError;
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

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaMalloc"), (gpointer)amgel::cudaMalloc, NULL, (gpointer*)&runtime.origCudaMalloc);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaMallocAsync"), (gpointer)amgel::cudaMallocAsync, NULL, (gpointer*)&runtime.origCudaMallocAsync);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaFree"), (gpointer)amgel::cudaFree, NULL, (gpointer*)&runtime.origCudaFree);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaFreeAsync"), (gpointer)amgel::cudaFreeAsync, NULL, (gpointer*)&runtime.origCudaFreeAsync);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaSetDevice"), (gpointer)amgel::cudaSetDevice, NULL, (gpointer*)&runtime.origCudaSetDevice);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclGetUniqueId, (gpointer)amgel::getUniqueId, NULL, (gpointer*)&runtime.origNcclGetUniqueId);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclCommInitRank, (gpointer)amgel::commInitRank, NULL, (gpointer*)&runtime.origNcclCommInitRank);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclGroupStart, (gpointer)amgel::groupStart, NULL, (gpointer*)&runtime.origNcclGroupStart);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclGroupEnd, (gpointer)amgel::groupEnd, NULL, (gpointer*)&runtime.origNcclGroupEnd);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclSend, (gpointer)amgel::send, NULL, (gpointer*)&runtime.origNcclSend);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclRecv, (gpointer)amgel::recv, NULL, (gpointer*)&runtime.origNcclRecv);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclBroadcast, (gpointer)amgel::broadcast, NULL, (gpointer*)&runtime.origNcclBroadcast);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclBcast, (gpointer)amgel::bcast, NULL, (gpointer*)&runtime.origNcclBcast);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclAllGather, (gpointer)amgel::allGather, NULL, (gpointer*)&runtime.origNcclAllGather);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclReduce, (gpointer)amgel::reduce, NULL, (gpointer*)&runtime.origNcclReduce);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclAllReduce, (gpointer)amgel::allReduce, NULL, (gpointer*)&runtime.origNcclAllReduce);
        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclReduceScatter, (gpointer)amgel::reduceScatter, NULL, (gpointer*)&runtime.origNcclReduceScatter);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclCommDestroy, (gpointer)amgel::commDestroy, NULL, (gpointer*)&runtime.origNcclCommDestroy);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclCommCount, (gpointer)amgel::commCount, NULL, (gpointer*)&runtime.origNcclCommCount);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclCommUserRank, (gpointer)amgel::commUserRank, NULL, (gpointer*)&runtime.origNcclCommUserRank);

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
