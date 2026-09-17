#include "ncclfold.hpp"

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nccl_fold
{
    template <ncclRedOp_t Op>
    struct ReductionValue;

    template <>
    struct ReductionValue<ncclSum>
    {
        template <typename T>
        __device__ static T apply(T a, T b) { return a + b; }
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
        __device__ static T apply(T a, T b) { return a * b; }
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
        __device__ static T apply(T a, T b) { return a < b ? a : b; }
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
        __device__ static T apply(T a, T b) { return a > b ? a : b; }
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
    __global__ void reductionKernel(const void *const *sources, void *output,
                                    size_t count, size_t source_offset, int nranks)
    {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count)
            return;
        T value = static_cast<const T *>(sources[0])[source_offset + i];
        for (int rank = 1; rank < nranks; ++rank)
            value = reduceValue<T, Op>(value, static_cast<const T *>(sources[rank])[source_offset + i]);
        static_cast<T *>(output)[i] = value;
    }

    template <ncclRedOp_t Op>
    static ReductionKernel reductionKernelForDatatype(ncclDataType_t datatype)
    {
        switch (datatype)
        {
        case ncclInt8: return reductionKernel<int8_t, Op>;
        case ncclUint8: return reductionKernel<uint8_t, Op>;
        case ncclInt32: return reductionKernel<int32_t, Op>;
        case ncclUint32: return reductionKernel<uint32_t, Op>;
        case ncclInt64: return reductionKernel<int64_t, Op>;
        case ncclUint64: return reductionKernel<uint64_t, Op>;
        case ncclFloat16: return reductionKernel<__half, Op>;
        case ncclFloat32: return reductionKernel<float, Op>;
        case ncclFloat64: return reductionKernel<double, Op>;
        case ncclBfloat16: return reductionKernel<__nv_bfloat16, Op>;
        default: return nullptr;
        }
    }

    ReductionKernel getReductionKernel(ncclDataType_t datatype, ncclRedOp_t op)
    {
        switch (op)
        {
        case ncclSum: return reductionKernelForDatatype<ncclSum>(datatype);
        case ncclProd: return reductionKernelForDatatype<ncclProd>(datatype);
        case ncclMin: return reductionKernelForDatatype<ncclMin>(datatype);
        case ncclMax: return reductionKernelForDatatype<ncclMax>(datatype);
        default: return nullptr;
        }
    }
}
