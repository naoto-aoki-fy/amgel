# NCCL Fold

NCCL Fold runs CUDA/MPI/NCCL programs written for one GPU per MPI process on a
single physical NVIDIA GPU. It is a correctness-oriented emulation layer, not
a performance model for a multi-GPU system.

## Build

CUDA, MPI, NCCL, libelf, and `nvcc` are required. Prepare
[`atlc`](https://github.com/naoto-aoki-fy/atlc) separately; it is not included
as a submodule. NCCL 2.28 or newer is required to build support for the host
`ncclAlltoAll`, `ncclGather`, and `ncclScatter` entry points. With older NCCL
headers NCCL Fold still builds, but those three unavailable symbols are not
interposed.

CUDA 11.3 or newer is required. NCCL Fold's interposed `cudaMallocAsync` uses
the CUDA memory-pool IPC API introduced in CUDA 11.3.

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
| `ncclAlltoAll` | Supported (NCCL 2.28+) | Rank-ordered source blocks; in-place unsupported |
| `ncclGather` | Supported (NCCL 2.28+) | Rank-ordered output on root; root-slot in-place supported |
| `ncclScatter` | Supported (NCCL 2.28+) | Root's rank-ordered input blocks; root-slot in-place supported |
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
rank's output slot, Reduce with identical root input/output, ReduceScatter with
the output pointing at the calling rank's input segment, Gather with the root's
input at `recvbuff + root * count`, and Scatter with the root's output at
`sendbuff + root * count` (offsets are in datatype elements). NCCL documents
AlltoAll in-place operation as unsupported; NCCL Fold rejects equal, nonempty
AlltoAll send and receive pointers with `ncclInvalidArgument`. Every referenced
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
* **Collective value semantics.** Broadcast, AllGather, AlltoAll, Gather,
  Scatter, Reduce, AllReduce, and ReduceScatter produce the documented
  rank-selected or rank-ordered values for the datatypes and reduction
  operators in the compatibility table. Gather places rank `i` in root output
  slot `i`; Scatter sends root input slot `i` to rank `i`; and rank `r`'s
  AlltoAll output slot `i` receives source rank `i`'s slot `r`. Reductions are
  evaluated in increasing logical-rank order. This is mathematical/value
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
  P2P ordering is the per-peer sequence ordering described above. Grouped calls
  retain their issuing order across operation types and communicators.
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
  `ncclReduceScatter`, and (when built against NCCL 2.28 or newer)
  `ncclAlltoAll`, `ncclGather`, and `ncclScatter`, with the datatypes and
  operators in the table above.
  The interposed management/query surface is `ncclGetUniqueId`,
  `ncclCommInitRank`, `ncclCommCount`, `ncclCommUserRank`, `ncclCommDestroy`, and
  nested `ncclGroupStart`/`ncclGroupEnd`. No equivalence claim is made for
  other NCCL entry points, including communicator split/abort/finalize,
  asynchronous error queries, registered buffers, user-defined reduction
  operators, or NCCL device APIs.
* Every nonempty referenced range is within a live allocation observed through
  the interposed `cudaMalloc` or `cudaMallocAsync`; untracked allocations
  (including CUDA driver-API, managed, host, externally imported, or custom
  allocator memory) are unsupported. Legacy allocations must be CUDA-IPC-exportable.
  Async allocations come from NCCL Fold's explicit exportable pool and require
  Linux/POSIX-file-descriptor memory-pool IPC. Direct user calls to
  `cudaMallocFromPoolAsync` and allocations from user-created/custom pools are
  outside the supported communication-buffer surface.
* Ranks issue compatible operations. Collectives occur in the same order on all
  communicator ranks; Send and Recv are balanced and ordered compatibly for each
  peer. Concurrent host threads must not race operations on the same communicator
  (the collective sequence is not thread-safe). Groups are thread-local and may
  be nested; only the outermost `ncclGroupEnd` submits their accumulated work.
* Streams, allocations, and communicators remain valid until all queued work that
  refers to them has completed. Applications use CUDA stream/event
  synchronization rather than NCCL API return as evidence of GPU completion.
* The contract applies to successful calls. Process failure, malformed or stale
  bootstrap state, MPI/CUDA failures, cancellation, and recovery after a partial
  error are outside its scope.

### Known Semantic Deviations

* **Allocator coverage is limited.** Interposed `cudaMallocAsync` is genuinely
  stream ordered: it calls `cudaMallocFromPoolAsync` on an NCCL Fold-owned,
  IPC-capable explicit pool. It no longer substitutes synchronous `cudaMalloc`.
  This resolves that synchronization deviation, but does not provide complete
  CUDA allocator equivalence: custom/user pools, direct
  `cudaMallocFromPoolAsync`, pool attributes, and graph capture are unsupported.
  Devices must report memory-pool support and POSIX-FD handle support; otherwise
  `cudaMallocAsync` returns an error with no synchronous fallback. NCCL Fold
  removes tracking metadata when CUDA accepts `cudaFree`/`cudaFreeAsync`, so a
  subsequently submitted communication call cannot refer to that pointer.
* **Grouping is only a subset of NCCL grouping semantics.** Groups are
  thread-local, may nest, and defer submission until the outermost
  `ncclGroupEnd`. Submission preserves call order across P2P, collectives, and
  communicators. A group error can still occur after some work has already been
  enqueued, and NCCL Fold does not reproduce native atomic-launch behavior.
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
* **Resource lifetime differs.** IPC-capable events are communicator-owned and
  pooled; imported event objects are cached by IPC handle. Reduction source
  pointer arrays use stream-ordered allocation and free. Legacy imported IPC
  memory mappings remain cached until `ncclCommDestroy`. In contrast, imported
  pool pointers are freed asynchronously in the consuming operation stream
  before its done event, and the exporter waits on that event before later
  source-stream work. Imported pool objects are communicator-owned; the local
  export pool intentionally has process lifetime. Grouped-send snapshots are
  legacy IPC-exported allocations and are conservatively retained. Destroy does not first synchronize
  outstanding GPU work, so destroying a communicator before its work completes
  remains unsupported.
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
counters, CUDA IPC mappings, event pools, and reduction scratch accounting. Communicators
therefore do not share a singleton operation state. CUDA allocation metadata records
base, size, and whether the allocation is legacy or belongs to NCCL Fold's async
pool, and is removed by intercepted `cudaFree`/`cudaFreeAsync` calls.

At communicator initialization, every rank exports its process-global pool as a
POSIX file descriptor. The descriptors are transferred over communicator-unique
Unix-domain sockets using `SCM_RIGHTS` (the integer descriptor values are never
sent through MPI), imported once per communicator, and closed after import.
Pointer export records are opaque tagged metadata carried by MPI alongside
legacy `cudaIpcMemHandle_t` records. This permits legacy and async-pool buffers,
including interior pointers, to participate in the same operation.

For each collective, ranks record a ready event and exchange allocation IPC
handles, offsets, operation metadata, and event handles using MPI. Broadcast
copies from the root mapping. AllGather and Gather copy each contributing rank
into its ordered output slot; Scatter selects the destination rank's block from
the root mapping; and AlltoAll selects the destination rank's block from every
source mapping and places it in source-rank order. Reduce, AllReduce, and
ReduceScatter launch an explicit typed CUDA kernel
over mapped rank inputs; ReduceScatter selects the local rank's segment. A
second event exchange adds stream dependencies that prevent premature source
reuse. GPU copies and kernels are enqueued in the user-provided stream; NCCL Fold
does **not** call `cudaStreamSynchronize` or synchronize the device.

There is one important difference from native NCCL asynchronous behavior: the
MPI metadata exchanges block the calling host thread until every communicator
rank enters the same collective. `ncclGroupEnd` performs this exchange for
queued grouped operations. GPU completion remains asynchronous after the API
returns. Before submitting a group's CUDA work in original call order, NCCL
Fold batches the MPI P2P metadata rendezvous across the whole group. This keeps
mutually dependent Send/Recv control-plane exchanges from blocking ordered
submission. Grouped sends take a stream-ordered device snapshot at their call
position, so their completion waits can be appended after the ordered group
without permitting premature reuse of the user's source buffer.

P2P uses the same ready/copy/done dependency scheme. Set `NCCL_FOLD_DEBUG_P2P=1`
to log communicator, rank, peer, sequence, event, stream, and direction.

Event slots have an explicit host-side lease. For ordinary P2P, receipt of done
metadata proves that the receiver submitted its ready wait; a subsequent ACK
proves that the sender submitted its done wait. Grouped P2P exchanges symmetric
ACKs after all waits are submitted. Collectives use communicator-wide host
barriers after ready waits and after done waits. These handshakes do not wait for
GPU completion: CUDA waits bind to the event generation current when the wait is
submitted, so the exporter may then safely re-record the slot. Imported events
are opened once per distinct pooled handle. Reduction source-pointer arrays are
allocated, copied, consumed, and freed in order on the operation stream using
the original (uninterposed) `cudaMallocAsync`/`cudaFreeAsync` functions.

Set `NCCL_FOLD_RESOURCE_STATS=1` to print per-communicator teardown accounting
for owned event creation/pool peak, imported event opens/cache size, IPC memory
mappings, reduction-scratch high-water mark, and retained IPC-exported grouped
send snapshots. Normal execution does not print these statistics.

### Known limitations

* Communicator discovery uses a single-host filesystem rendezvous (in
  `/tmp/ncclfold-bootstrap-<uid>` by default, or `NCCL_FOLD_BOOTSTRAP_DIR`). All
  requested ranks must initialize with the same unique ID, size, and distinct
  NCCL ranks. Normal completion removes rendezvous files; an abnormally killed
  job can leave stale files that may be removed manually.
* MPI provides host-side control-plane progress, so NCCL Fold does not reproduce
  NCCL's nonblocking host progress, algorithms, topology, or performance.
* IPC memory mappings remain cached until `ncclCommDestroy`; their normal cache
  cardinality follows distinct remote allocations, not collective operation
  count. IPC-exported grouped-send snapshots are a special retained allocation
  class. Event pools/caches instead follow peak concurrently leased events, and
  reduction pointer arrays are stream-ordered allocations returned immediately
  after their kernels are enqueued.
* NCCL reduction operators other than sum/product/min/max are unsupported.
