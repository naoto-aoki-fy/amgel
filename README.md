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
mpirun -n 4 env LD_PRELOAD="$PWD/ncclfold.so" ./all_reduce_example
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

## Semantic Contract

NCCL Fold is intended as a correctness-oriented execution environment for the
supported API subset below. Equivalence means equality of the communicated
values and the relevant CUDA-stream dependencies for a **supported execution**;
it does not mean that NCCL Fold is a complete NCCL implementation or that it
reproduces how a native multi-GPU run makes progress.

NCCL Fold preserves the following observable properties for supported
executions:

* **Communicator membership and rank.** A successful `ncclCommInitRank` creates
  a communicator from the processes presenting the same `ncclUniqueId`, `ndev`,
  and distinct logical ranks. `ncclCommCount` and `ncclCommUserRank` report that
  logical size and rank. Subsets and rank orderings need not match
  `MPI_COMM_WORLD`.
* **Collective value semantics.** Broadcast, AllGather, Reduce, AllReduce, and
  ReduceScatter produce the documented rank-selected or rank-ordered values for
  the datatypes and reduction operators in the compatibility table. Reductions
  are evaluated in increasing logical-rank order. This is mathematical/value
  equivalence, subject to the floating-point caveat below, not byte-for-byte
  equivalence with an arbitrary native NCCL algorithm.
* **P2P value and matching semantics.** Send/Recv transfers the requested byte
  range. Operations are matched by communicator, peer, direction, and a
  monotonically increasing per-peer sequence number. The two sides must issue
  compatible counts and datatypes; NCCL Fold checks byte counts, but does not
  diagnose different datatypes having the same size.
* **Collective order and metadata agreement.** Collectives have a
  per-communicator sequence. All ranks must enter them in the same order and
  agree on kind, root, datatype, reduction operator, and byte count; disagreement
  returns an error rather than intentionally selecting one rank's metadata.
  P2P ordering is the per-peer sequence ordering described above. Ordering of
  mixed operations inside groups is limited as described under known
  deviations.
* **CUDA stream ordering and inter-rank happens-before edges.** A ready event is
  recorded in the operation's stream before a remote read. The consuming stream
  waits for that event before its copy or reduction. A completion event and
  reciprocal stream wait keep subsequent work in each source stream behind
  remote consumption. The copy/kernel itself is enqueued in the stream supplied
  to the NCCL call. Thus ordinary work ordered before and after a supported call
  in that stream observes the intended buffer dependencies; this claim does not
  extend to every CUDA event API or to graph capture.
* **Buffer reuse timing on the device.** After the returned operation in the
  source stream has completed, all remote uses that NCCL Fold enqueued for that
  operation have completed. Destination data is available to later work in its
  destination stream. The host API return alone is not a completion signal, and
  freeing an allocation or destroying a communicator before its queued work is
  complete is outside the contract.
* **Documented in-place layouts.** The in-place forms listed above preserve their
  value semantics when pointers and allocation lifetimes satisfy this section's
  assumptions. Other overlaps or aliases are outside the contract.
* **Communicator isolation.** Operation sequences, MPI control-plane messages,
  CUDA IPC mappings, events, and reduction scratch allocations are maintained
  per virtual communicator. Correctly constructed communicators therefore do
  not intentionally match one another's operations.

NCCL Fold does **not** preserve the following:

* execution time, latency, bandwidth, throughput, or any other performance
  characteristic;
* NCCL algorithm/protocol selection, chunking, channel behavior, launch shape,
  or reduction tree/order;
* physical topology or transport behavior, including NVLink, PCIe, NIC, GPUDirect,
  peer-access, NUMA, and failure characteristics;
* physical multi-GPU concurrency, scheduling, contention, memory capacity, or
  GPU-to-GPU isolation: every logical rank uses physical CUDA device zero;
* native NCCL communication progress, deadlock timing, or host-side nonblocking
  behavior and API return timing: MPI metadata rendezvous can block the calling
  host thread until matching ranks enter an operation (or `ncclGroupEnd`); or
* native NCCL diagnostics, error timing/recovery, asynchronous-error behavior,
  environment-variable tuning, or behavior of APIs not explicitly interposed.

### Assumptions / Scope

* There is one MPI process per logical NCCL rank, MPI is initialized, all
  participating processes are on one host, and all logical ranks are mapped to
  physical CUDA device zero. A communicator's ranks use the same unique ID and
  size exactly once, use distinct valid logical ranks, and collectively complete
  initialization.
* The supported communication surface is `ncclSend`, `ncclRecv`,
  `ncclBroadcast`, `ncclBcast`, `ncclAllGather`, `ncclReduce`, `ncclAllReduce`,
  and `ncclReduceScatter`, with the datatypes and operators in the table above.
  The interposed management/query surface is `ncclGetUniqueId`,
  `ncclCommInitRank`, `ncclCommCount`, `ncclCommUserRank`, `ncclCommDestroy`, and
  non-nested `ncclGroupStart`/`ncclGroupEnd`. No equivalence claim is made for
  other NCCL entry points, including communicator split/abort/finalize,
  asynchronous error queries, registered buffers, user-defined reduction
  operators, or NCCL device APIs.
* Every nonempty referenced range is within a live allocation observed through
  the interposed `cudaMalloc` or `cudaMallocAsync`; untracked allocations
  (including CUDA driver-API, managed, host, externally imported, or custom
  allocator memory) are unsupported. The allocation must be CUDA-IPC-exportable.
* Ranks issue compatible operations. Collectives occur in the same order on all
  communicator ranks; Send and Recv are balanced and ordered compatibly for each
  peer. Concurrent host threads must not race operations on the same communicator
  (the collective sequence is not thread-safe), and a thread may have at most one
  active, non-nested group.
* Streams, allocations, and communicators remain valid until all queued work that
  refers to them has completed. Applications use CUDA stream/event
  synchronization rather than NCCL API return as evidence of GPU completion.
* The contract applies to successful calls. Process failure, malformed or stale
  bootstrap state, MPI/CUDA failures, cancellation, and recovery after a partial
  error are outside its scope.

### Known Semantic Deviations

* **Stream-ordered allocation is changed.** Interposed `cudaMallocAsync` calls
  synchronous `cudaMalloc`, so allocation visibility, allocation-pool behavior,
  failure timing, and the synchronization/progress effects of allocation differ
  from CUDA. This can hide bugs that rely incorrectly on stream-ordered
  allocation or introduce ordering/performance behavior absent from the native
  run. `cudaFreeAsync` remains asynchronous but NCCL Fold removes the allocation
  from its tracking table as soon as the free is accepted, which can reject a
  later operation even while CUDA still considers earlier stream-ordered uses
  valid.
* **Grouping is only a subset of NCCL grouping semantics.** Groups are
  thread-local and cannot nest. Calls are deferred until `ncclGroupEnd`, then
  processed per communicator with all queued P2P operations before queued
  collectives; interleaving between P2P and collectives, between communicators,
  and exact call-order/atomic-launch behavior is not preserved. A group error
  can occur after some work has already been enqueued. These differences can
  introduce deadlocks or errors, or hide ordering bugs, relative to native NCCL.
* **Host blocking differs.** Ungrouped operations perform blocking MPI metadata
  exchanges in the NCCL call, and grouped operations do so in `ncclGroupEnd`.
  This may introduce host deadlocks in code that depends on native nonblocking
  return, while the extra rendezvous can also hide host/GPU progress races.
* **CUDA Graph capture is unsupported.** Event creation, CUDA IPC setup, MPI
  calls, and host-side allocation occur while an operation is submitted; no
  graph-capture-compatible path or replay semantics are implemented. Capture may
  fail or behave differently rather than exposing a native-NCCL graph bug.
* **Reduction results need not be bitwise reproducible.** NCCL Fold reduces in
  increasing rank order with its own kernel; native NCCL may use another
  association, instructions, or precision. In particular, half and bfloat16
  values are converted through float for each pairwise step. Floating-point
  results (including NaN and signed-zero handling) may differ while still
  representing the requested reduction, so bitwise comparisons are not a valid
  equivalence test.
* **Device selection is collapsed.** Every intercepted `cudaSetDevice` request
  selects device zero, and its requested ordinal is not validated. This can hide
  invalid-device and device-placement bugs and cannot reproduce per-device
  contexts or properties.
* **Resource lifetime differs.** Imported IPC mappings, event handles, owned
  events, and reduction pointer arrays are retained until `ncclCommDestroy`.
  Destroy does not first synchronize outstanding GPU work. Compared with native
  NCCL this can increase resource consumption, delay failures, or cause unsafe
  teardown if the application destroys a communicator too early.
* **Bootstrap has stronger environmental constraints.** Communicator discovery
  polls a single-host filesystem directory and has no timeout. Abnormal
  termination can leave stale rank or tag files; a later run may fail, wait
  forever, or consume stale identity data. This is not native NCCL's bootstrap
  or failure behavior.
* **Validation is not identical to NCCL.** P2P checks transferred byte counts but
  not datatype identity; zero-count edge cases and error codes/timing have not
  been established as equivalent. Unsupported calls may reach the real NCCL
  library with virtual communicator handles, so applications must restrict
  themselves to the documented interposed subset.

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
  `/tmp/ncclfold-bootstrap-<uid>` by default, or `NCCL_FOLD_BOOTSTRAP_DIR`). All
  requested ranks must initialize with the same unique ID, size, and distinct
  NCCL ranks. Normal completion removes rendezvous files; an abnormally killed
  job can leave stale files that may be removed manually.
* MPI provides host-side control-plane progress, so NCCL Fold does not reproduce
  NCCL's nonblocking host progress, algorithms, topology, or performance.
* IPC mappings, events, and reduction pointer arrays are retained until
  `ncclCommDestroy`, favoring safe asynchronous lifetime over bounded scratch
  resource use.
* NCCL reduction operators other than sum/product/min/max are unsupported.
