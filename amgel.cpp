#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>

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

    decltype(&::cudaIpcGetMemHandle) cudaIpcGetMemHandle;
    decltype(&::cudaIpcOpenMemHandle) cudaIpcOpenMemHandle;
    decltype(&::cudaMemcpyAsync) cudaMemcpyAsync;
    decltype(&::cudaEventCreateWithFlags) cudaEventCreateWithFlags;
    decltype(&::cudaEventRecord) cudaEventRecord;
    decltype(&::cudaIpcGetEventHandle) cudaIpcGetEventHandle;
    decltype(&::cudaIpcOpenEventHandle) cudaIpcOpenEventHandle;
    decltype(&::cudaStreamWaitEvent) cudaStreamWaitEvent;

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

    struct commStruct {
        bool in_group;
        std::vector<sendRecvArgs_t> send_args;
        std::vector<sendRecvArgs_t> recv_args;
        std::vector<uint64_t> pointer_list;
        std::vector<cudaEvent_t> owned_events;
        std::vector<cudaEvent_t> imported_events;
        std::vector<importedMemory> ipc_mappings;
        std::vector<uint64_t> next_send_sequence;
        std::vector<uint64_t> next_recv_sequence;
        MPI_Comm mpi_comm;
        int rank;
        int ndev;

        decltype(&::cudaMalloc<void>) origCudaMalloc;
        cudaError_t (*origCudaMallocAsync)(void**, size_t, cudaStream_t);
        decltype(&::cudaSetDevice) origCudaSetDevice;
        decltype(&::ncclGetUniqueId) origNcclGetUniqueId;
        decltype(&::ncclCommInitRank) origNcclCommInitRank;
        decltype(&::ncclGroupStart) origNcclGroupStart;
        decltype(&::ncclGroupEnd) origNcclGroupEnd;
        decltype(&::ncclSend) origNcclSend;
        decltype(&::ncclRecv) origNcclRecv;
    };

    static amgel::commStruct commStructPrivate;

    static cudaError_t cudaMalloc(void **devPtr, size_t size) {
        cudaError_t const ret = amgel::commStructPrivate.origCudaMalloc(devPtr, size);
        if (ret == cudaSuccess) {
            amgel::commStructPrivate.pointer_list.push_back((uint64_t)*devPtr);
        }
        return ret;
    }

    static cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
        cudaError_t const ret = amgel::commStructPrivate.origCudaMalloc(devPtr, size);
        /* We cannot use buffer allocated with cudaMalloAsync for cudaIpcGetMemHandle */
        if (ret == cudaSuccess) {
            amgel::commStructPrivate.pointer_list.push_back((uint64_t)*devPtr);
        }
        return ret;
    }


    static cudaError_t cudaSetDevice(int device) {
        cudaError_t const ret = amgel::commStructPrivate.origCudaSetDevice(0);
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
            default:
                throw datatype;
                return 0;
        }
        return 0;
    }

    static ncclResult_t getUniqueId(ncclUniqueId* nccl_id) {
        *(int*)nccl_id = 1;
        return ncclSuccess;
    }

    void* getClosestPointer(void* pointer_input, uint64_t* offset) {
        uint64_t num_ptrs = amgel::commStructPrivate.pointer_list.size();
        uint64_t distance_closest = (uint64_t)(-1);
        uint64_t pointer_closest = 0;
        for (uint64_t ptr_num = 0; ptr_num < num_ptrs; ptr_num++) {
            uint64_t pointer = amgel::commStructPrivate.pointer_list[ptr_num];
            // fprintf(stderr, "[%d] pointer=%p pointer_input=%p\n", __LINE__, pointer, pointer_input);
            if ((uint64_t)pointer_input < pointer) {
                continue;
            }
            uint64_t const distance = (uint64_t)pointer_input - pointer;
            if (distance < distance_closest) {
                distance_closest = distance;
                pointer_closest = pointer;
            }
        }
        if (pointer_closest != 0 && offset != 0) {
            *offset = distance_closest;
        }
        return (void*)pointer_closest;
    }

    static ncclResult_t commInitRank(ncclComm_t* comm, int ndev, ncclUniqueId nccl_id, int rank) {
        (void)nccl_id;
        if (MPI_Comm_dup(MPI_COMM_WORLD, &amgel::commStructPrivate.mpi_comm) != MPI_SUCCESS) {
            return ncclSystemError;
        }
        *comm = (ncclComm_t)(void*)&amgel::commStructPrivate;
        amgel::commStructPrivate.in_group = false;
        amgel::commStructPrivate.rank = rank;
        amgel::commStructPrivate.ndev = ndev;
        amgel::commStructPrivate.next_send_sequence.assign(ndev, 0);
        amgel::commStructPrivate.next_recv_sequence.assign(ndev, 0);
        return ncclSuccess;
    }

    static ncclResult_t groupStart() {
        amgel::commStructPrivate.send_args.clear();
        amgel::commStructPrivate.recv_args.clear();
        amgel::commStructPrivate.in_group = true;
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

    static ncclResult_t openMemory(commStruct* comm, cudaIpcMemHandle_t const& handle, void** pointer) {
        for (size_t i = 0; i < comm->ipc_mappings.size(); ++i) {
            if (std::memcmp(&comm->ipc_mappings[i].handle, &handle, sizeof(handle)) == 0) {
                *pointer = comm->ipc_mappings[i].pointer;
                return ncclSuccess;
            }
        }
        ncclResult_t result = cudaCheck(cudaIpcOpenMemHandle(pointer, handle, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle");
        if (result == ncclSuccess) comm->ipc_mappings.push_back({handle, *pointer});
        return result;
    }

    /*
     * MPI is only the control plane here.  Every exported event is a fresh event,
     * and its record has been submitted before its handle is sent.  Events and IPC
     * mappings deliberately remain owned by the communicator: reusing/destroying
     * them at group end could race a wait or copy which is still queued remotely.
     */
    static ncclResult_t enqueueP2P(commStruct* comm) {
        const size_t send_count = comm->send_args.size();
        const size_t recv_count = comm->recv_args.size();
        std::vector<readyMessage> outgoing(send_count);
        std::vector<readyMessage> incoming(recv_count);
        std::vector<doneMessage> outgoing_done(recv_count);
        std::vector<doneMessage> incoming_done(send_count);
        std::vector<MPI_Request> requests(send_count + recv_count);

        for (size_t i = 0; i < send_count; ++i) {
            sendRecvArgs_t const& op = comm->send_args[i];
            readyMessage& message = outgoing[i];
            void* allocation = getClosestPointer(op.buff, &message.memory.offset);
            if (allocation == NULL) return ncclInvalidArgument;
            ncclResult_t result = cudaCheck(cudaIpcGetMemHandle(&message.memory.handle, allocation), "cudaIpcGetMemHandle");
            if (result != ncclSuccess) return result;
            cudaEvent_t ready = NULL;
            result = cudaCheck(cudaEventCreateWithFlags(&ready, cudaEventInterprocess | cudaEventDisableTiming), "cudaEventCreateWithFlags(ready)");
            if (result != ncclSuccess) return result;
            comm->owned_events.push_back(ready);
            result = cudaCheck(cudaEventRecord(ready, op.stream), "cudaEventRecord(ready)");
            if (result != ncclSuccess) return result;
            result = cudaCheck(cudaIpcGetEventHandle(&message.ready, ready), "cudaIpcGetEventHandle(ready)");
            if (result != ncclSuccess) return result;
            message.bytes = op.count * sizeofNcclDataType(op.datatype);
            message.sequence = op.sequence;
            if (debugEnabled()) std::fprintf(stderr, "AMGeL comm=%p rank=%d peer=%d seq=%llu ready=%p stream=%p send\n",
                (void*)comm, comm->rank, op.peer, (unsigned long long)op.sequence, (void*)ready, (void*)op.stream);
        }
        for (size_t i = 0; i < send_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Isend(&outgoing[i], sizeof(readyMessage), MPI_BYTE,
                comm->send_args[i].peer, readyTag, comm->mpi_comm, &requests[i]), "MPI_Isend(ready)");
            if (result != ncclSuccess) return result;
        }
        for (size_t i = 0; i < recv_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Irecv(&incoming[i], sizeof(incoming[i]), MPI_BYTE,
                comm->recv_args[i].peer, readyTag, comm->mpi_comm, &requests[send_count + i]), "MPI_Irecv(ready)");
            if (result != ncclSuccess) return result;
        }
        if (!requests.empty()) {
            ncclResult_t result = mpiCheck(MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall(ready metadata)");
            if (result != ncclSuccess) return result;
        }

        requests.assign(send_count + recv_count, MPI_REQUEST_NULL);
        ncclResult_t deferred_error = ncclSuccess;
        for (size_t i = 0; i < recv_count; ++i) {
            sendRecvArgs_t const& op = comm->recv_args[i];
            readyMessage const& message = incoming[i];
            uint64_t const recv_bytes = op.count * sizeofNcclDataType(op.datatype);
            if (message.sequence != op.sequence || message.bytes != recv_bytes) deferred_error = ncclInvalidArgument;

            cudaEvent_t ready = NULL;
            cudaEvent_t done = NULL;
            void* source = NULL;
            ncclResult_t result = cudaCheck(cudaIpcOpenEventHandle(&ready, message.ready), "cudaIpcOpenEventHandle(ready)");
            if (result != ncclSuccess) return result;
            comm->imported_events.push_back(ready);
            result = openMemory(comm, message.memory.handle, &source);
            if (result != ncclSuccess) return result;
            result = cudaCheck(cudaStreamWaitEvent(op.stream, ready, 0), "cudaStreamWaitEvent(ready)");
            if (result != ncclSuccess) return result;
            if (message.bytes == recv_bytes) {
                result = cudaCheck(cudaMemcpyAsync(op.buff, (char*)source + message.memory.offset, recv_bytes,
                    cudaMemcpyDeviceToDevice, op.stream), "cudaMemcpyAsync(P2P)");
                if (result != ncclSuccess) return result;
            }
            result = cudaCheck(cudaEventCreateWithFlags(&done, cudaEventInterprocess | cudaEventDisableTiming), "cudaEventCreateWithFlags(done)");
            if (result != ncclSuccess) return result;
            comm->owned_events.push_back(done);
            result = cudaCheck(cudaEventRecord(done, op.stream), "cudaEventRecord(done)");
            if (result != ncclSuccess) return result;
            result = cudaCheck(cudaIpcGetEventHandle(&outgoing_done[i].done, done), "cudaIpcGetEventHandle(done)");
            if (result != ncclSuccess) return result;
            outgoing_done[i].sequence = message.sequence;
            if (debugEnabled()) std::fprintf(stderr, "AMGeL comm=%p rank=%d peer=%d seq=%llu done=%p stream=%p recv\n",
                (void*)comm, comm->rank, op.peer, (unsigned long long)message.sequence, (void*)done, (void*)op.stream);
        }
        for (size_t i = 0; i < recv_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Isend(&outgoing_done[i], sizeof(doneMessage), MPI_BYTE,
                comm->recv_args[i].peer, doneTag, comm->mpi_comm, &requests[send_count + i]), "MPI_Isend(done)");
            if (result != ncclSuccess) return result;
        }
        for (size_t i = 0; i < send_count; ++i) {
            ncclResult_t result = mpiCheck(MPI_Irecv(&incoming_done[i], sizeof(doneMessage), MPI_BYTE,
                comm->send_args[i].peer, doneTag, comm->mpi_comm, &requests[i]), "MPI_Irecv(done)");
            if (result != ncclSuccess) return result;
        }
        if (!requests.empty()) {
            ncclResult_t result = mpiCheck(MPI_Waitall((int)requests.size(), requests.data(), MPI_STATUSES_IGNORE), "MPI_Waitall(done metadata)");
            if (result != ncclSuccess) return result;
        }
        for (size_t i = 0; i < send_count; ++i) {
            if (incoming_done[i].sequence != comm->send_args[i].sequence) deferred_error = ncclInvalidArgument;
            cudaEvent_t done = NULL;
            ncclResult_t result = cudaCheck(cudaIpcOpenEventHandle(&done, incoming_done[i].done), "cudaIpcOpenEventHandle(done)");
            if (result != ncclSuccess) return result;
            comm->imported_events.push_back(done);
            result = cudaCheck(cudaStreamWaitEvent(comm->send_args[i].stream, done, 0), "cudaStreamWaitEvent(done)");
            if (result != ncclSuccess) return result;
        }
        return deferred_error;
    }

    static ncclResult_t groupEnd() {
        commStruct* comm = &amgel::commStructPrivate;
        if (!comm->in_group) return ncclInvalidUsage;
        ncclResult_t result = enqueueP2P(comm);
        comm->in_group = false;
        return result;
    }

    static ncclResult_t send(void* sendbuff, uint64_t count, int datatype, int peer, amgel::commStruct* comm, cudaStream_t stream) {
        if (comm->in_group) {
            if (peer < 0 || peer >= comm->ndev) return ncclInvalidArgument;
            comm->send_args.push_back({sendbuff, count, datatype, peer, stream, comm->next_send_sequence[peer]++});
        } else {
            if (peer < 0 || peer >= comm->ndev) return ncclInvalidArgument;
            comm->send_args.assign(1, {sendbuff, count, datatype, peer, stream, comm->next_send_sequence[peer]++});
            comm->recv_args.clear();
            ncclResult_t result = enqueueP2P(comm);
            comm->send_args.clear();
            return result;
        }
        return ncclSuccess;
    }

    static ncclResult_t recv(void* recvbuff, uint64_t count, int datatype, int peer, amgel::commStruct* comm, cudaStream_t stream) {
        if (comm->in_group) {
            if (peer < 0 || peer >= comm->ndev) return ncclInvalidArgument;
            comm->recv_args.push_back({recvbuff, count, datatype, peer, stream, comm->next_recv_sequence[peer]++});
        } else {
            if (peer < 0 || peer >= comm->ndev) return ncclInvalidArgument;
            comm->recv_args.assign(1, {recvbuff, count, datatype, peer, stream, comm->next_recv_sequence[peer]++});
            comm->send_args.clear();
            ncclResult_t result = enqueueP2P(comm);
            comm->recv_args.clear();
            return result;
        }
        return ncclSuccess;
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

        cudaIpcGetMemHandle = &::cudaIpcGetMemHandle;
        cudaIpcOpenMemHandle = &::cudaIpcOpenMemHandle;
        cudaMemcpyAsync = &::cudaMemcpyAsync;
        cudaEventCreateWithFlags = &::cudaEventCreateWithFlags;
        cudaEventRecord = &::cudaEventRecord;
        cudaIpcGetEventHandle = &::cudaIpcGetEventHandle;
        cudaIpcOpenEventHandle = &::cudaIpcOpenEventHandle;
        cudaStreamWaitEvent = &::cudaStreamWaitEvent;

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaMalloc"), (gpointer)amgel::cudaMalloc, NULL, (gpointer*)&amgel::commStructPrivate.origCudaMalloc);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaMallocAsync"), (gpointer)amgel::cudaMallocAsync, NULL, (gpointer*)&amgel::commStructPrivate.origCudaMallocAsync);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)find_symbol_offset_or_dlsym("/proc/self/exe", "cudaSetDevice"), (gpointer)amgel::cudaSetDevice, NULL, (gpointer*)&amgel::commStructPrivate.origCudaSetDevice);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclGetUniqueId, (gpointer)amgel::getUniqueId, NULL, (gpointer*)&amgel::commStructPrivate.origNcclGetUniqueId);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclCommInitRank, (gpointer)amgel::commInitRank, NULL, (gpointer*)&amgel::commStructPrivate.origNcclCommInitRank);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclGroupStart, (gpointer)amgel::groupStart, NULL, (gpointer*)&amgel::commStructPrivate.origNcclGroupStart);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclGroupEnd, (gpointer)amgel::groupEnd, NULL, (gpointer*)&amgel::commStructPrivate.origNcclGroupEnd);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclSend, (gpointer)amgel::send, NULL, (gpointer*)&amgel::commStructPrivate.origNcclSend);

        ATLC_CHECK_FRIDA_GUM_REPLACE(gum_interceptor_replace, amgel::interceptor, (gpointer)ncclRecv, (gpointer)amgel::recv, NULL, (gpointer*)&amgel::commStructPrivate.origNcclRecv);

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
