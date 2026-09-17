# NCCL Fold single-GPU semantic suite

This is the reproducible correctness suite for the compatibility table and
Semantic Contract in the top-level README. It launches multiple logical MPI
ranks on **one physical NVIDIA GPU**, always preloads NCCL Fold, and compares
the deterministic device results with a value oracle in `run_semantic.py`.
It is not a benchmark and does not require a native multi-GPU machine.

## Build and run

Configure `config.mk` and build NCCL Fold as described in the top-level README,
then run:

```sh
make test
```

The default is equivalent to:

```sh
python3 tests/semantic/run_semantic.py --ranks 2,4 \
  --ncclfold ./ncclfold.so
```

The Make target builds `ncclfold.so` and the self-contained CUDA workload
first. Dependencies are the project's existing CUDA, NCCL, MPI, and Python 3
standard library dependencies. Each family is an isolated, timeout-bounded
`mpirun` subprocess with `LD_PRELOAD` set to the absolute interposer path and
`CUDA_VISIBLE_DEVICES=0`. Set `MPIRUN` for a different launcher, or use:

```sh
make test TEST_RANKS=2 TEST_ARGS='--mpi-arg=--allow-run-as-root --timeout 300'
python3 tests/semantic/run_semantic.py --ranks 4 --families p2p,group
```

`--device`, `--executable`, `--mpirun`, and repeatable `--mpi-arg` options
support site-specific setups. `--artifacts DIR` retains command, stdout,
stderr, status, and per-rank JSON Lines files. Failed runs are always retained.
If no GPU is visible, the runner prints a clear suite-level skip and succeeds;
launcher or workload errors are failures, not skips.

## Coverage

The systematic matrix covers:

* `ncclSend`/`ncclRecv` pair and ring matching;
* `ncclBroadcast` and its in-place `ncclBcast` alias;
* `ncclAllGather`, `ncclReduce`, `ncclAllReduce`, and `ncclReduceScatter`;
* `ncclAlltoAll`, `ncclGather`, and `ncclScatter` when compiled with NCCL
  2.28 or newer;
* all ten documented datatypes, all four reduction operators, counts 1/3/7,
  roots zero/last/interior, and documented in-place and out-of-place layouts;
* non-default streams and consecutive matrix operations;
* communicator size/user-rank queries, reverse logical rank numbering, and
  isolation of two communicators used in one group;
* nested group deferral and ordering across collective and P2P operations;
* GPU-produced input, destination consumption, cross-rank happens-before, and
  source-buffer reuse after an AllReduce on a non-default stream;
* ordinary allocations plus stream-ordered allocations when CUDA memory-pool
  IPC is available; and
* negative collective-count metadata and equal-size/distinct-datatype P2P
  mismatches, both required to report `ncclInvalidUsage` on every rank.

The workload emits a skip record for the three NCCL 2.28 APIs when the build
headers do not declare them, rather than conditionally hiding the tests. Async
allocation is similarly skipped when the CUDA device lacks memory-pool
support. These skips appear in the final `PASS/FAIL/SKIP` summary.

Integer and non-reduction copy results require exact equality. Floating-point
reductions use absolute and relative tolerances chosen for small,
well-conditioned inputs: 2e-3 (float16), 2e-2 (bfloat16), 2e-5 (float32), and
1e-12 (float64). A mismatch identifies case, logical rank, API, datatype,
element, expected value, observed value, and tolerance.

Timeout/fail-stop diagnostics for missing peers are intentionally not provoked
by the default suite: they terminate the MPI job and are environment-sensitive.
Every normal family and validation family is nevertheless already isolated in
its own bounded subprocess, so a regression cannot terminate or hang the rest
of the run. CUDA Graphs, unsupported allocation sources, lifetime violations,
host-progress behavior, and performance remain outside the documented
Semantic Contract and outside this suite.
