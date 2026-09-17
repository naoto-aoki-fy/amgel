# NCCL Fold

NCCL Fold runs CUDA/MPI/NCCL programs written for one GPU per MPI process on a
single physical NVIDIA GPU. It is a correctness-oriented emulation layer, not
a performance model for a multi-GPU system.

## Build

CUDA, MPI, NCCL, libelf, and `nvcc` are required. Prepare
[`atlc`](https://github.com/naoto-aoki-fy/atlc) separately; it is not included
as a submodule.

Define the environment-specific compiler, linker, and GPU architecture options
in `config.mk`, which the Makefile automatically includes. For example:

```make
CFLAGS_VENDOR = -I/path/to/mpi/include -I/path/to/nccl/include -I/path/to/libelf/include -I/path/to/atlc/include
LDFLAGS_VENDOR = -L/path/to/mpi/lib -L/path/to/nccl/lib -L/path/to/libelf/lib
GENCODE_FLAGS = -gencode=arch=compute_xx,code=sm_xx
```

The `atlc/include` path may instead be supplied through `CPATH`. Once the
required include and library paths are configured, build NCCL Fold with:

```sh
make
```

The first build downloads and extracts the configured Frida development kits.

Run an MPI/NCCL application with the interposer preloaded, for example:

```sh
mpirun -n 4 env LD_PRELOAD="$PWD/nccl-fold.so" ./all_reduce_example
```

An ordinary call such as
`ncclAllReduce(send, receive, count, ncclFloat32, ncclSum, comm, stream)` is
then executed across four logical ranks on physical device zero.

## NCCL compatibility

| NCCL API | Status | Notes |
| --- | --- | --- |
| `ncclSend` | Supported | CUDA IPC copy |
| `ncclRecv` | Supported | CUDA IPC copy |
| `ncclBroadcast` | Supported | Any valid root |
| `ncclBcast` | Supported | In-place broadcast alias |
| `ncclAllGather` | Supported | Rank-ordered output |
| `ncclReduce` | Supported | Any valid root |
| `ncclAllReduce` | Supported | Typed reduction kernel |
| `ncclReduceScatter` | Supported | Rank-dependent reduced segment |

The collective datatype set is `ncclInt8`, `ncclUint8`, `ncclInt32`,
`ncclUint32`, `ncclInt64`, `ncclUint64`, `ncclFloat16`, `ncclFloat32`,
`ncclFloat64`, and `ncclBfloat16`. Reduction collectives support `ncclSum`,
`ncclProd`, `ncclMin`, and `ncclMax`. Other datatypes and operators return
`ncclInvalidArgument`; NCCL Fold does not silently execute an approximate fallback.

Documented in-place layouts work naturally: a shared broadcast buffer,
AllReduce with identical input/output, AllGather with the input in the calling
rank's output slot, Reduce with identical root input/output, and ReduceScatter
with the output pointing at the calling rank's input segment. Every referenced
range must belong to a live CUDA allocation tracked by NCCL Fold.

## Implementation and stream semantics

Each successful `ncclCommInitRank` creates an independent virtual communicator
containing exactly the processes which initialize it with the same
`ncclUniqueId`. Its MPI ranks are ordered by the requested NCCL ranks, so
subsets and reordered ranks work without involving other `MPI_COMM_WORLD`
processes. Each communicator has its own MPI communicator, rank, size, sequence
counters, CUDA IPC mappings, events, and reduction scratch storage. Communicators
therefore do not share a singleton operation state. CUDA allocation metadata records both
base and size and is removed by intercepted `cudaFree`/`cudaFreeAsync` calls.

For each collective, ranks record a ready event and exchange allocation IPC
handles, offsets, operation metadata, and event handles using MPI. Broadcast
copies from the root mapping. AllGather copies each rank into its ordered output
slot. Reduce, AllReduce, and ReduceScatter launch an explicit typed CUDA kernel
over mapped rank inputs; ReduceScatter selects the local rank's segment. A
second event exchange adds stream dependencies that prevent premature source
reuse. GPU copies and kernels are enqueued in the user-provided stream; NCCL Fold
does **not** call `cudaStreamSynchronize` or synchronize the device.

There is one important difference from native NCCL asynchronous behavior: the
MPI metadata exchanges block the calling host thread until every communicator
rank enters the same collective. `ncclGroupEnd` performs this exchange for
queued grouped operations. GPU completion remains asynchronous after the API
returns. Groups may include existing P2P operations, and queues remain separated
by communicator.

P2P uses the same ready/copy/done dependency scheme. Set `NCCL_FOLD_DEBUG_P2P=1`
to log communicator, rank, peer, sequence, event, stream, and direction.

### Known limitations

* Communicator discovery uses a single-host filesystem rendezvous (in
  `/tmp/nccl-fold-bootstrap-<uid>` by default, or `NCCL_FOLD_BOOTSTRAP_DIR`). All
  requested ranks must initialize with the same unique ID, size, and distinct
  NCCL ranks. Normal completion removes rendezvous files; an abnormally killed
  job can leave stale files that may be removed manually.
* MPI provides host-side control-plane progress, so NCCL Fold does not reproduce
  NCCL's nonblocking host progress, algorithms, topology, or performance.
* IPC mappings, events, and reduction pointer arrays are retained until
  `ncclCommDestroy`, favoring safe asynchronous lifetime over bounded scratch
  resource use.
* NCCL reduction operators other than sum/product/min/max are unsupported.
