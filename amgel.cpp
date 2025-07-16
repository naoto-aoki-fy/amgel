#include <cstdio>
#include <cstring>
#include <cstdint>

#include <dlfcn.h>
#include <libelf.h>
#include <gelf.h>
#include <fcntl.h>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <atlc/check_x.hpp>
#include <atlc/check_mpi.hpp>
#include <atlc/check_cuda.hpp>
#include <frida-gum.h>

namespace amgel {

    static GumInterceptor *interceptor = NULL;

    decltype(&::cudaIpcGetMemHandle) cudaIpcGetMemHandle;
    decltype(&::cudaIpcOpenMemHandle) cudaIpcOpenMemHandle;
    decltype(&::cudaIpcCloseMemHandle) cudaIpcCloseMemHandle;
    decltype(&::cudaMemcpyAsync) cudaMemcpyAsync;
    decltype(&::cudaStreamSynchronize) cudaStreamSynchronize;

    typedef struct {
        cudaIpcMemHandle_t handle;
        uint64_t offset;
    } handleOffset;

    typedef struct {
        void* buff;
        uint64_t count;
        int datatype;
        int peer;
        cudaStream_t stream;
    } sendRecvArgs_t;

    struct commStruct {
        bool in_group;
        std::vector<sendRecvArgs_t> send_args;
        std::vector<sendRecvArgs_t> recv_args;
        std::vector<handleOffset> src_buff_handle_list;
        std::vector<void*> src_buff_list;
        std::vector<MPI_Request> mpi_request_list;
        std::vector<uint64_t> pointer_list;

        decltype(&::cudaMalloc<void>) origCudaMalloc;
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
        amgel::commStructPrivate.pointer_list.push_back((uint64_t)*devPtr);
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
        *comm = (ncclComm_t)(void*)&amgel::commStructPrivate;
        amgel::commStructPrivate.in_group = false;
        return ncclSuccess;
    }

    static ncclResult_t groupStart() {
        amgel::commStructPrivate.send_args.clear();
        amgel::commStructPrivate.recv_args.clear();
        amgel::commStructPrivate.src_buff_handle_list.clear();
        amgel::commStructPrivate.src_buff_list.clear();
        amgel::commStructPrivate.mpi_request_list.clear();
        amgel::commStructPrivate.in_group = true;
        return ncclSuccess;
    }

    static ncclResult_t groupEnd() {
        if (!amgel::commStructPrivate.in_group) {
            return ncclInvalidUsage;
        }

        amgel::commStructPrivate.mpi_request_list.resize(amgel::commStructPrivate.send_args.size() + amgel::commStructPrivate.recv_args.size());
        int mpi_request_idx = 0;

        /* get memhandle and send it to peer */
        for (size_t i = 0; i < amgel::commStructPrivate.send_args.size(); ++i) {
            handleOffset handle_offset;
            void* const buffer = amgel::getClosestPointer(amgel::commStructPrivate.send_args[i].buff, &handle_offset.offset);
            ATLC_CHECK_CUDA(amgel::cudaIpcGetMemHandle, &handle_offset.handle, buffer);
            ATLC_CHECK_MPI(MPI_Isend, &handle_offset, sizeof(handleOffset), MPI_BYTE, amgel::commStructPrivate.send_args[i].peer, 0, MPI_COMM_WORLD, &amgel::commStructPrivate.mpi_request_list[mpi_request_idx]);
            mpi_request_idx++;
        }

        /* recv memhandle from peer */
        amgel::commStructPrivate.src_buff_handle_list.resize(amgel::commStructPrivate.recv_args.size());
        for (size_t i = 0; i < amgel::commStructPrivate.send_args.size(); ++i) {
            ATLC_CHECK_MPI(MPI_Irecv, &amgel::commStructPrivate.src_buff_handle_list[i], sizeof(handleOffset), MPI_BYTE, amgel::commStructPrivate.recv_args[i].peer, 0, MPI_COMM_WORLD, &amgel::commStructPrivate.mpi_request_list[mpi_request_idx]);
            mpi_request_idx++;
        }

        /* wait for all send/recv requests to finish */
        ATLC_CHECK_MPI(MPI_Waitall, amgel::commStructPrivate.mpi_request_list.size(), amgel::commStructPrivate.mpi_request_list.data(), MPI_STATUSES_IGNORE);

        /* open memhandle and copy data */
        amgel::commStructPrivate.src_buff_list.resize(amgel::commStructPrivate.src_buff_handle_list.size());
        for (size_t i = 0; i < amgel::commStructPrivate.src_buff_handle_list.size(); ++i) {
            // fprintf(stderr, "debug: %d\n", __LINE__);
            ATLC_CHECK_CUDA(amgel::cudaIpcOpenMemHandle, &amgel::commStructPrivate.src_buff_list[i], amgel::commStructPrivate.src_buff_handle_list[i].handle, cudaIpcMemLazyEnablePeerAccess);
            ATLC_CHECK_CUDA(
                amgel::cudaMemcpyAsync,
                amgel::commStructPrivate.recv_args[i].buff,
                /* amgel::commStructPrivate.src_buff_list[i] */
                (void*)((uint64_t)amgel::commStructPrivate.src_buff_list[i] + amgel::commStructPrivate.src_buff_handle_list[i].offset),
                amgel::commStructPrivate.recv_args[i].count * amgel::sizeofNcclDataType(amgel::commStructPrivate.recv_args[i].datatype),
                cudaMemcpyDeviceToDevice,
                amgel::commStructPrivate.recv_args[i].stream
            );
        }

        /* synchronize streams */
        ATLC_CHECK_CUDA(amgel::cudaStreamSynchronize, amgel::commStructPrivate.recv_args[0].stream);
        for (size_t i = 1; i < amgel::commStructPrivate.send_args.size(); ++i) {
            if (amgel::commStructPrivate.recv_args[i].stream != amgel::commStructPrivate.recv_args[i - 1].stream) {
                ATLC_CHECK_CUDA(amgel::cudaStreamSynchronize, amgel::commStructPrivate.recv_args[i].stream);
            }
        }

        /* close memhandle */
        for (size_t i = 0; i < amgel::commStructPrivate.src_buff_list.size(); ++i) {
            ATLC_CHECK_CUDA(amgel::cudaIpcCloseMemHandle, amgel::commStructPrivate.src_buff_list[i]);
        }

        /* free send/recv args */

        /* synchronize */
        ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);

        amgel::commStructPrivate.in_group = false;
        return ncclSuccess;
    }

    static ncclResult_t send(void* sendbuff, uint64_t count, int datatype, int peer, amgel::commStruct* comm, cudaStream_t stream) {
        if (comm->in_group) {
            comm->send_args.push_back({sendbuff, count, datatype, peer, stream});
        } else {
            // not yet tested
            handleOffset handle_offset;
            void* const buffer = amgel::getClosestPointer(sendbuff, &handle_offset.offset);
            cudaIpcMemHandle_t handle;
            ATLC_CHECK_CUDA(amgel::cudaIpcGetMemHandle, &handle, buffer);
            MPI_Send(&handle_offset, sizeof(handleOffset), MPI_BYTE, peer, 0, MPI_COMM_WORLD);
        }
        return ncclSuccess;
    }

    static ncclResult_t recv(void* recvbuff, uint64_t count, int datatype, int peer, amgel::commStruct* comm, cudaStream_t stream) {
        if (comm->in_group) {
            comm->recv_args.push_back({recvbuff, count, datatype, peer, stream});
        } else {
            // not yet tested
            handleOffset handle_offset;
            ATLC_CHECK_MPI(MPI_Recv, &handle_offset, sizeof(handleOffset), MPI_BYTE, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            void* sendbuff;
            ATLC_CHECK_CUDA(amgel::cudaIpcOpenMemHandle, &sendbuff, handle_offset.handle, cudaIpcMemLazyEnablePeerAccess);
            ATLC_CHECK_CUDA(amgel::cudaMemcpyAsync, recvbuff, (void*)((uint64_t)sendbuff + handle_offset.offset), count * amgel::sizeofNcclDataType(datatype), cudaMemcpyDeviceToDevice, stream);
            ATLC_CHECK_CUDA(amgel::cudaIpcCloseMemHandle, sendbuff);
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
        cudaIpcCloseMemHandle = &::cudaIpcCloseMemHandle;
        cudaMemcpyAsync = &::cudaMemcpyAsync;
        cudaStreamSynchronize = &::cudaStreamSynchronize;

        void *exe = NULL;

        amgel::commStructPrivate.origCudaMalloc = (cudaError_t (*)(void **, size_t))find_symbol_offset_or_dlsym("/proc/self/exe", "cudaMalloc");
        gum_interceptor_replace(amgel::interceptor, (gpointer)amgel::commStructPrivate.origCudaMalloc, (gpointer)amgel::cudaMalloc, NULL, (gpointer*)&amgel::commStructPrivate.origCudaMalloc);

        amgel::commStructPrivate.origCudaSetDevice = (cudaError_t (*)(int))find_symbol_offset_or_dlsym("/proc/self/exe", "cudaSetDevice");
        gum_interceptor_replace(amgel::interceptor, (gpointer)amgel::commStructPrivate.origCudaSetDevice, (gpointer)amgel::cudaSetDevice, NULL, (gpointer*)&amgel::commStructPrivate.origCudaSetDevice);

        amgel::commStructPrivate.origNcclGetUniqueId = ncclGetUniqueId;
        gum_interceptor_replace(amgel::interceptor, (gpointer)amgel::commStructPrivate.origNcclGetUniqueId, (gpointer)amgel::getUniqueId, NULL, (gpointer*)&amgel::commStructPrivate.origNcclGetUniqueId);

        amgel::commStructPrivate.origNcclCommInitRank = ncclCommInitRank;
        gum_interceptor_replace(amgel::interceptor, (gpointer)amgel::commStructPrivate.origNcclCommInitRank, (gpointer)amgel::commInitRank, NULL, (gpointer*)&amgel::commStructPrivate.origNcclCommInitRank);

        amgel::commStructPrivate.origNcclGroupStart = ncclGroupStart;
        gum_interceptor_replace(amgel::interceptor, (gpointer)amgel::commStructPrivate.origNcclGroupStart, (gpointer)amgel::groupStart, NULL, (gpointer*)&amgel::commStructPrivate.origNcclGroupStart);

        amgel::commStructPrivate.origNcclGroupEnd = ncclGroupEnd;
        gum_interceptor_replace(amgel::interceptor, (gpointer)amgel::commStructPrivate.origNcclGroupEnd, (gpointer)amgel::groupEnd, NULL, (gpointer*)&amgel::commStructPrivate.origNcclGroupEnd);

        amgel::commStructPrivate.origNcclSend = ncclSend;
        gum_interceptor_replace(amgel::interceptor, (gpointer)amgel::commStructPrivate.origNcclSend, (gpointer)amgel::send, NULL, (gpointer*)&amgel::commStructPrivate.origNcclSend);

        amgel::commStructPrivate.origNcclRecv = ncclRecv;
        gum_interceptor_replace(amgel::interceptor, (gpointer)amgel::commStructPrivate.origNcclRecv, (gpointer)amgel::recv, NULL, (gpointer*)&amgel::commStructPrivate.origNcclRecv);

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