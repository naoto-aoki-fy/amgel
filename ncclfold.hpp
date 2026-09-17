#pragma once

#include <cstddef>

#include <nccl.h>

namespace nccl_fold
{
    using ReductionKernel = void (*)(const void *const *sources, void *output,
                                     size_t count, size_t source_offset, int nranks);

    ReductionKernel getReductionKernel(ncclDataType_t datatype, ncclRedOp_t op);
}
