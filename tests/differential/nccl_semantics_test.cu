#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <atlc/check_cuda.hpp>
#include <atlc/check_nccl.hpp>
#include <cstdio>
#include <exception>
#include <algorithm>
#include <string>
#include <vector>
#include <sys/stat.h>

static int rank_, nranks_;
static FILE *out;
static std::string mode_, filter_;
static void die(const char *s)
{
    fprintf(stderr, "rank %d: %s\n", rank_, s);
    MPI_Abort(MPI_COMM_WORLD, 2);
}

template <class T>
__host__ __device__ T cv(double x) { return (T)x; }
template <>
__host__ __device__ __half cv(double x) { return __float2half((float)x); }
template <>
__host__ __device__ __nv_bfloat16 cv(double x) { return __float2bfloat16((float)x); }
template <class T>
__host__ __device__ double dv(T x) { return (double)x; }
template <>
__host__ __device__ double dv(__half x) { return __half2float(x); }
template <>
__host__ __device__ double dv(__nv_bfloat16 x) { return __bfloat162float(x); }
template <class T>
__global__ void produce(T *p, int n, int r)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = cv<T>((r + 1) * 10 + i + 1);
}
template <class T>
__global__ void consume(T *dst, const T *src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = cv<T>(dv(src[i]) + 1);
}
template <class T>
__global__ void overwrite(T *p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = cv<T>(-7);
}

static const char *dtname(ncclDataType_t d)
{
    switch (d)
    {
    case ncclInt8:
        return "int8";
    case ncclUint8:
        return "uint8";
    case ncclInt32:
        return "int32";
    case ncclUint32:
        return "uint32";
    case ncclInt64:
        return "int64";
    case ncclUint64:
        return "uint64";
    case ncclFloat16:
        return "float16";
    case ncclFloat32:
        return "float32";
    case ncclFloat64:
        return "float64";
    case ncclBfloat16:
        return "bfloat16";
    default:
        return "unknown";
    }
}
static const char *opname(ncclRedOp_t o) { return o == ncclSum ? "sum" : o == ncclProd ? "prod"
                                                                     : o == ncclMin    ? "min"
                                                                                       : "max"; }
static bool selected(const std::string &id) { return filter_.empty() || filter_ == id; }
template <class T>
static std::vector<T> input(int n)
{
    std::vector<T> v(n);
    for (int i = 0; i < n; i++)
        v[i] = cv<T>((rank_ + 1) * 2 + (i % 5) + 1);
    return v;
}
template <class T>
static void record(const std::string &id, const char *op, ncclDataType_t dt, const char *red, int root, int peer, int count, bool in_place, const char *stream, const std::vector<T> &v, const char *state = "pass", const char *checks = "values")
{
    if (!selected(id))
        return;
    fprintf(out, "{\"case_id\":\"%s\",\"operation\":\"%s\",\"ranks\":%d,\"rank\":%d,\"datatype\":\"%s\",\"reduction\":", id.c_str(), op, nranks_, rank_, dtname(dt));
    if (red)
        fprintf(out, "\"%s\"", red);
    else
        fputs("null", out);
    fputs(",\"root\":", out);
    if (root < 0)
        fputs("null", out);
    else
        fprintf(out, "%d", root);
    fputs(",\"peer\":", out);
    if (peer < 0)
        fputs("null", out);
    else
        fprintf(out, "%d", peer);
    fprintf(out, ",\"count\":%d,\"in_place\":%s,\"stream\":\"%s\",\"checks\":\"%s\",\"state\":\"%s\",\"mode\":\"%s\",\"values\":[", count, in_place ? "true" : "false", stream, checks, state, mode_.c_str());
    for (size_t i = 0; i < v.size(); i++)
    {
        if (i)
            fputc(',', out);
        fprintf(out, "%.17g", dv(v[i]));
    }
    fputs("]}\n", out);
    fflush(out);
}
template <class T>
static void basic_case(ncclComm_t comm, ncclDataType_t dt, const std::string &fam, int count, bool ip, int root, ncclRedOp_t rop)
{
    std::string id = fam + "/" + dtname(dt) + "/c" + std::to_string(count) + "/" + (ip ? "in" : "out");
    if (fam == "broadcast" || fam == "gather" || fam == "scatter" || fam == "reduce")
        id += "/root" + std::to_string(root);
    if (fam == "reduce" || fam == "allreduce" || fam == "reducescatter")
        id += "/" + opname(rop);
    if (!selected(id))
        return;
    int sendn = count, recvn = count;
    if (fam == "allgather" || fam == "gather")
        recvn = count * nranks_;
    if (fam == "alltoall" || fam == "scatter")
        sendn = count * nranks_;
    if (fam == "reducescatter")
        sendn = count * nranks_;
    std::vector<T> h = input<T>(sendn), got(recvn, cv<T>(-99));
    if ((fam == "reduce" || fam == "allreduce" || fam == "reducescatter") && rop == ncclProd)
        for (int i = 0; i < sendn; i++)
            h[i] = cv<T>(rank_ == 0 ? 2 : 1);
    T *s, *r;
    ATLC_CHECK_CUDA(cudaMalloc, &s, sizeof(T) * sendn);
    ATLC_CHECK_CUDA(cudaMalloc, &r, sizeof(T) * recvn);
    ATLC_CHECK_CUDA(cudaMemcpy, s, h.data(), sizeof(T) * sendn, cudaMemcpyHostToDevice);
    ATLC_CHECK_CUDA(cudaMemset, r, 0, sizeof(T) * recvn);
    cudaStream_t st;
    ATLC_CHECK_CUDA(cudaStreamCreate, &st);
    const void *sp = s;
    void *rp = r;
    if (ip)
    {
        if (fam == "broadcast" || fam == "allreduce")
        {
            ATLC_CHECK_CUDA(cudaFree, r);
            r = s;
            rp = s;
        }
        else if (fam == "allgather")
        {
            ATLC_CHECK_CUDA(cudaFree, s);
            s = r;
            sp = r + rank_ * count;
            ATLC_CHECK_CUDA(cudaMemcpy, (void *)sp, h.data(), sizeof(T) * count, cudaMemcpyHostToDevice);
        }
        else if (fam == "gather" && rank_ == root)
        {
            ATLC_CHECK_CUDA(cudaFree, s);
            s = r;
            sp = r + root * count;
            ATLC_CHECK_CUDA(cudaMemcpy, (void *)sp, h.data(), sizeof(T) * count, cudaMemcpyHostToDevice);
        }
        else if (fam == "scatter" && rank_ == root)
        {
            ATLC_CHECK_CUDA(cudaFree, r);
            r = s;
            rp = s + root * count;
        }
        else if (fam == "reduce" && rank_ == root)
        {
            ATLC_CHECK_CUDA(cudaFree, r);
            r = s;
            rp = s;
        }
        else if (fam == "reducescatter")
        {
            ATLC_CHECK_CUDA(cudaFree, r);
            r = s + rank_ * count;
            rp = r;
        }
    }
    if (fam == "broadcast")
    {
        if (ip)
            ATLC_CHECK_NCCL(ncclBcast, rp, count, dt, root, comm, st);
        else
            ATLC_CHECK_NCCL(ncclBroadcast, sp, rp, count, dt, root, comm, st);
    }
    else if (fam == "allgather")
        ATLC_CHECK_NCCL(ncclAllGather, sp, rp, count, dt, comm, st);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    else if (fam == "alltoall")
        ATLC_CHECK_NCCL(ncclAlltoAll, sp, rp, count, dt, comm, st);
    else if (fam == "gather")
        ATLC_CHECK_NCCL(ncclGather, sp, rp, count, dt, root, comm, st);
    else if (fam == "scatter")
        ATLC_CHECK_NCCL(ncclScatter, sp, rp, count, dt, root, comm, st);
#endif
    else if (fam == "reduce")
        ATLC_CHECK_NCCL(ncclReduce, sp, rp, count, dt, rop, root, comm, st);
    else if (fam == "allreduce")
        ATLC_CHECK_NCCL(ncclAllReduce, sp, rp, count, dt, rop, comm, st);
    else if (fam == "reducescatter")
        ATLC_CHECK_NCCL(ncclReduceScatter, sp, rp, count, dt, rop, comm, st);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, st);
    int visible = (fam == "reduce" || fam == "gather") && rank_ != root ? 0 : recvn;
    if (fam == "scatter" || fam == "reducescatter")
        visible = count;
    got.resize(visible);
    if (visible)
        ATLC_CHECK_CUDA(cudaMemcpy, got.data(), r, sizeof(T) * visible, cudaMemcpyDeviceToHost);
    record(id, (fam == "broadcast" && ip) ? "bcast" : fam.c_str(), dt, (fam == "reduce" || fam == "allreduce" || fam == "reducescatter") ? opname(rop) : nullptr, root, -1, count, ip, "nondefault", got);
    ATLC_CHECK_CUDA(cudaStreamDestroy, st);
    bool interior = ip && ((fam == "reducescatter") || (fam == "scatter" && rank_ == root));
    if (r != s && !interior)
        ATLC_CHECK_CUDA(cudaFree, r);
    ATLC_CHECK_CUDA(cudaFree, s);
}
template <class T>
static void matrix_t(ncclComm_t c, ncclDataType_t d, const std::string &f)
{
    for (int count : {1, 3, 7})
    {
        std::vector<int> roots = {0, nranks_ - 1};
        if (nranks_ > 2)
            roots.push_back(1);
        if (f == "broadcast" || f == "gather" || f == "scatter")
        {
            for (int root : roots)
                for (bool ip : {false, true})
                    basic_case<T>(c, d, f, count, ip, root, ncclSum);
        }
        else if (f == "reduce")
        {
            for (int root : roots)
                for (ncclRedOp_t o : {ncclSum, ncclProd, ncclMin, ncclMax})
                    for (bool ip : {false, true})
                        basic_case<T>(c, d, f, count, ip, root, o);
        }
        else if (f == "allreduce" || f == "reducescatter")
        {
            for (ncclRedOp_t o : {ncclSum, ncclProd, ncclMin, ncclMax})
                for (bool ip : {false, true})
                    basic_case<T>(c, d, f, count, ip, -1, o);
        }
        else
            for (bool ip : {false, true})
            {
                if (f == "alltoall" && ip)
                    continue;
                basic_case<T>(c, d, f, count, ip, -1, ncclSum);
            }
    }
}
static void matrix(ncclComm_t c, const std::string &f)
{
    matrix_t<int8_t>(c, ncclInt8, f);
    matrix_t<uint8_t>(c, ncclUint8, f);
    matrix_t<int32_t>(c, ncclInt32, f);
    matrix_t<uint32_t>(c, ncclUint32, f);
    matrix_t<int64_t>(c, ncclInt64, f);
    matrix_t<uint64_t>(c, ncclUint64, f);
    matrix_t<__half>(c, ncclFloat16, f);
    matrix_t<float>(c, ncclFloat32, f);
    matrix_t<double>(c, ncclFloat64, f);
    matrix_t<__nv_bfloat16>(c, ncclBfloat16, f);
}
static void p2p(ncclComm_t c)
{
    for (int ring = 0; ring < 2; ring++)
    {
        int peer = ring ? (rank_ + 1) % nranks_ : (rank_ ^ 1);
        if (peer >= nranks_)
            peer = rank_;
        int from = ring ? (rank_ - 1 + nranks_) % nranks_ : (rank_ ^ 1);
        if (from >= nranks_)
            from = rank_;
        std::string id = ring ? "p2p/ring" : "p2p/pairwise";
        float *s, *r;
        ATLC_CHECK_CUDA(cudaMalloc, &s, 3 * sizeof(float));
        ATLC_CHECK_CUDA(cudaMalloc, &r, 3 * sizeof(float));
        produce<<<1, 32>>>(s, 3, rank_);
        ATLC_CHECK_NCCL(ncclGroupStart);
        ATLC_CHECK_NCCL(ncclSend, s, 3, ncclFloat32, peer, c, 0);
        ATLC_CHECK_NCCL(ncclRecv, r, 3, ncclFloat32, from, c, 0);
        ATLC_CHECK_NCCL(ncclGroupEnd);
        std::vector<float> v(3);
        ATLC_CHECK_CUDA(cudaMemcpy, v.data(), r, sizeof(float) * 3, cudaMemcpyDeviceToHost);
        record(id, "send_recv", ncclFloat32, nullptr, -1, peer, 3, false, "default", v);
        ATLC_CHECK_CUDA(cudaFree, s);
        ATLC_CHECK_CUDA(cudaFree, r);
    }
}
static void ordering(ncclComm_t c)
{
    std::string id = "ordering/produce-consume-reuse";
    float *s, *r, *canary;
    cudaStream_t st;
    ATLC_CHECK_CUDA(cudaStreamCreate, &st);
    ATLC_CHECK_CUDA(cudaMalloc, &s, 4 * sizeof(float));
    ATLC_CHECK_CUDA(cudaMalloc, &r, 4 * sizeof(float));
    ATLC_CHECK_CUDA(cudaMalloc, &canary, 4 * sizeof(float));
    produce<<<1, 32, 0, st>>>(s, 4, rank_);
    ATLC_CHECK_NCCL(ncclAllReduce, s, r, 4, ncclFloat32, ncclSum, c, st);
    consume<<<1, 32, 0, st>>>(canary, r, 4);
    overwrite<<<1, 32, 0, st>>>(s, 4);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, st);
    std::vector<float> v(8);
    ATLC_CHECK_CUDA(cudaMemcpy, v.data(), canary, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    ATLC_CHECK_CUDA(cudaMemcpy, v.data() + 4, s, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    record(id, "allreduce", ncclFloat32, "sum", -1, -1, 4, false, "nondefault", v, "pass", "pre_input,post_consume,source_reuse,cross_rank");
    ATLC_CHECK_CUDA(cudaFree, s);
    ATLC_CHECK_CUDA(cudaFree, r);
    ATLC_CHECK_CUDA(cudaFree, canary);
    ATLC_CHECK_CUDA(cudaStreamDestroy, st);
}
static void groupcase(ncclComm_t c)
{
    std::string id = "group/nested-mixed-order";
    float *a, *b, *rx;
    ATLC_CHECK_CUDA(cudaMalloc, &a, sizeof(float));
    ATLC_CHECK_CUDA(cudaMalloc, &b, sizeof(float));
    ATLC_CHECK_CUDA(cudaMalloc, &rx, sizeof(float));
    produce<<<1, 1>>>(a, 1, rank_);
    int next = (rank_ + 1) % nranks_, prev = (rank_ - 1 + nranks_) % nranks_;
    ATLC_CHECK_NCCL(ncclGroupStart);
    ATLC_CHECK_NCCL(ncclGroupStart);
    ATLC_CHECK_NCCL(ncclAllReduce, a, b, 1, ncclFloat32, ncclSum, c, 0);
    ATLC_CHECK_NCCL(ncclGroupEnd);
    ATLC_CHECK_NCCL(ncclSend, a, 1, ncclFloat32, next, c, 0);
    ATLC_CHECK_NCCL(ncclRecv, rx, 1, ncclFloat32, prev, c, 0);
    ATLC_CHECK_NCCL(ncclBroadcast, b, b, 1, ncclFloat32, nranks_ - 1, c, 0);
    ATLC_CHECK_NCCL(ncclGroupEnd);
    std::vector<float> v(2);
    ATLC_CHECK_CUDA(cudaMemcpy, v.data(), b, sizeof(float), cudaMemcpyDeviceToHost);
    ATLC_CHECK_CUDA(cudaMemcpy, v.data() + 1, rx, sizeof(float), cudaMemcpyDeviceToHost);
    record(id, "group", ncclFloat32, "sum", nranks_ - 1, next, 1, true, "default", v, "pass", "nested,deferred,multiple,mixed_p2p_collective,issuance_order");
    ATLC_CHECK_CUDA(cudaFree, a);
    ATLC_CHECK_CUDA(cudaFree, b);
    ATLC_CHECK_CUDA(cudaFree, rx);
}
static void commcase(ncclComm_t c)
{
    int n = -1, r = -1;
    ATLC_CHECK_NCCL(ncclCommCount, c, &n);
    ATLC_CHECK_NCCL(ncclCommUserRank, c, &r);
    std::vector<int> v = {n, r};
    record("communicator/query-primary", "comm_query", ncclInt32, nullptr, -1, -1, 2, false, "default", v, "pass", "count,user_rank");
    ncclUniqueId id;
    if (rank_ == 0)
        ATLC_CHECK_NCCL(ncclGetUniqueId, &id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t reversed;
    ATLC_CHECK_NCCL(ncclCommInitRank, &reversed, nranks_, id, nranks_ - 1 - rank_);
    int rn = -1, rr = -1;
    ATLC_CHECK_NCCL(ncclCommCount, reversed, &rn);
    ATLC_CHECK_NCCL(ncclCommUserRank, reversed, &rr);
    float *s, *a, *b;
    ATLC_CHECK_CUDA(cudaMalloc, &s, sizeof(float));
    ATLC_CHECK_CUDA(cudaMalloc, &a, sizeof(float));
    ATLC_CHECK_CUDA(cudaMalloc, &b, sizeof(float));
    produce<<<1, 1>>>(s, 1, rank_);
    ATLC_CHECK_NCCL(ncclGroupStart);
    ATLC_CHECK_NCCL(ncclAllReduce, s, a, 1, ncclFloat32, ncclSum, c, 0);
    ATLC_CHECK_NCCL(ncclAllReduce, s, b, 1, ncclFloat32, ncclMax, reversed, 0);
    ATLC_CHECK_NCCL(ncclGroupEnd);
    std::vector<float> x(4);
    x[0] = rn;
    x[1] = rr;
    ATLC_CHECK_CUDA(cudaMemcpy, &x[2], a, sizeof(float), cudaMemcpyDeviceToHost);
    ATLC_CHECK_CUDA(cudaMemcpy, &x[3], b, sizeof(float), cudaMemcpyDeviceToHost);
    record("communicator/reversed-two-communicators", "comm_query_group", ncclFloat32, nullptr, -1, -1, 4, false, "default", x, "pass", "rank_order,isolation,multi_comm_group");
    ATLC_CHECK_CUDA(cudaFree, s);
    ATLC_CHECK_CUDA(cudaFree, a);
    ATLC_CHECK_CUDA(cudaFree, b);
    ATLC_CHECK_NCCL(ncclCommDestroy, reversed);
}
static void async_case(ncclComm_t c)
{
    int supported = 0;
    ATLC_CHECK_CUDA(cudaDeviceGetAttribute, &supported, cudaDevAttrMemoryPoolsSupported, 0);
    std::string id = "malloc_async/allreduce";
    if (!supported)
    {
        std::vector<float> v;
        record(id, "allreduce", ncclFloat32, "sum", -1, -1, 3, false, "nondefault", v, "skip", "memory_pool_ipc_unavailable");
        return;
    }
    cudaStream_t st;
    ATLC_CHECK_CUDA(cudaStreamCreate, &st);
    float *s, *r;
    ATLC_CHECK_CUDA(cudaMallocAsync, &s, 12, st);
    ATLC_CHECK_CUDA(cudaMallocAsync, &r, 12, st);
    produce<<<1, 32, 0, st>>>(s, 3, rank_);
    ATLC_CHECK_NCCL(ncclAllReduce, s, r, 3, ncclFloat32, ncclSum, c, st);
    std::vector<float> v(3);
    ATLC_CHECK_CUDA(cudaMemcpyAsync, v.data(), r, 12, cudaMemcpyDeviceToHost, st);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, st);
    record(id, "allreduce", ncclFloat32, "sum", -1, -1, 3, false, "nondefault", v);
    ATLC_CHECK_CUDA(cudaFreeAsync, s, st);
    ATLC_CHECK_CUDA(cudaFreeAsync, r, st);
    ATLC_CHECK_CUDA(cudaStreamSynchronize, st);
    ATLC_CHECK_CUDA(cudaStreamDestroy, st);
}
// Error cases run in their own runner subprocess.  A boolean result keeps the
// artifact stable if NCCL changes the numeric value of ncclInvalidUsage.
static void validation_case(ncclComm_t c)
{
    float *s, *r;
    ATLC_CHECK_CUDA(cudaMalloc, &s, 2 * sizeof(float));
    ATLC_CHECK_CUDA(cudaMalloc, &r, 2 * sizeof(float));
    produce<<<1, 2>>>(s, 2, rank_);
    ncclResult_t e = ncclAllReduce(s, r, rank_ == 0 ? 1 : 2, ncclFloat32, ncclSum, c, 0);
    record("validation/collective-count", "collective_metadata_mismatch", ncclInt32, nullptr, -1, -1, 1, false, "default", std::vector<int>{e == ncclInvalidUsage}, "pass", "all_ranks_invalid_usage");
    ATLC_CHECK_NCCL(ncclGroupStart);
    ATLC_CHECK_NCCL(ncclSend, s, 1, ncclFloat32, (rank_ + 1) % nranks_, c, 0);
    ATLC_CHECK_NCCL(ncclRecv, r, 1, ncclInt32, (rank_ - 1 + nranks_) % nranks_, c, 0);
    e = ncclGroupEnd();
    record("validation/p2p-datatype", "p2p_datatype_mismatch", ncclInt32, nullptr, -1, -1, 1, false, "default", std::vector<int>{e == ncclInvalidUsage}, "pass", "equal_size_distinct_datatype");
    ATLC_CHECK_CUDA(cudaFree, s);
    ATLC_CHECK_CUDA(cudaFree, r);
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks_);
    try
    {
        std::string family, dir;
        mode_ = "unknown";
        for (int i = 1; i < argc; i++)
        {
            std::string a = argv[i];
            if (i + 1 < argc && a == "--family")
                family = argv[++i];
            else if (i + 1 < argc && a == "--output")
                dir = argv[++i];
            else if (i + 1 < argc && a == "--mode")
                mode_ = argv[++i];
        }
        const char *f = getenv("NCCL_SEMANTICS_CASE");
        if (f)
            filter_ = f;
        ATLC_CHECK_CUDA(cudaSetDevice, rank_);
        ncclUniqueId id;
        if (rank_ == 0)
            ATLC_CHECK_NCCL(ncclGetUniqueId, &id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclComm_t comm;
        ATLC_CHECK_NCCL(ncclCommInitRank, &comm, nranks_, id, rank_);
        std::string output_path = dir + "/rank-" + std::to_string(rank_) + ".jsonl";
        out = fopen(output_path.c_str(), "w");
        if (!out)
            die("cannot open result file");
#if NCCL_VERSION_CODE < NCCL_VERSION(2, 28, 0)
        if (family == "alltoall" || family == "gather" || family == "scatter")
        {
            std::vector<int> v;
            record(family + "/unavailable", family.c_str(), ncclInt32, nullptr, -1, -1, 0, false, "default", v, "skip", "requires_nccl_2_28");
        }
        else
#endif
            if (family == "p2p")
            p2p(comm);
        else if (family == "ordering")
            ordering(comm);
        else if (family == "group")
            groupcase(comm);
        else if (family == "communicator")
            commcase(comm);
        else if (family == "malloc_async")
            async_case(comm);
        else if (family == "validation")
            validation_case(comm);
        else
            matrix(comm, family);
        ATLC_CHECK_CUDA(cudaDeviceSynchronize);
        ATLC_CHECK_NCCL(ncclCommDestroy, comm);
        fclose(out);
        MPI_Finalize();
        return 0;
    }
    catch (const std::exception &error)
    {
        fprintf(stderr, "rank %d: %s\n", rank_, error.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }
}
