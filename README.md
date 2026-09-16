# A's Multi-Gpus Emulation Layer

## P2P synchronization

AMGeL uses MPI only to match peer operations and exchange CUDA IPC handles.  For
each matched send/receive, it constructs this GPU-side dependency chain:

1. the sender records a fresh interprocess CUDA **ready** event after earlier
   work on the send stream;
2. the receiver stream waits for that event and performs the asynchronous IPC
   device-to-device copy;
3. the receiver records a fresh interprocess **done** event after the copy; and
4. the sender stream waits for the done event before later work can reuse the
   source buffer.

The event records are submitted before their IPC handles are sent, so a wait
never accidentally targets an event's old or future generation.  Events and
opened memory mappings are retained by the communicator rather than recycled
while remote GPU work may still reference them.  `ncclGroupEnd()` waits only for
the handle/sequence-number control messages needed to build this graph; it does
not synchronize a CUDA stream or wait for the copy to finish.  Set
`AMGEL_DEBUG_P2P=1` to log communicator, rank, peer, sequence, event, stream, and
direction information.

Each successful `ncclCommInitRank()` owns an independent virtual communicator,
including its duplicated MPI communicator, P2P sequence numbers, CUDA IPC
mappings, and events. `ncclGroupStart()` queues operations in thread-local group
state and keeps queues for different communicators separate. Communicators must
be released with `ncclCommDestroy()`.

This remains an emulation rather than native NCCL: transfers are CUDA IPC
copies initiated on the receiver, MPI metadata exchange may block the host
until matching calls arrive, IPC resources are retained for the communicator's
lifetime, and communicator membership/subsets still follow the existing
`MPI_COMM_WORLD` duplication model.
