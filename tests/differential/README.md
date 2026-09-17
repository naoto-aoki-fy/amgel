# NCCL Fold differential semantic tests

This correctness suite runs one CUDA/MPI/NCCL executable twice.  **Native**
maps MPI world rank `r` to physical GPU `r`; **fold** makes the same calls with
`ncclfold.so` preloaded, where intercepted `cudaSetDevice(r)` maps all logical
ranks to the single GPU exposed by `--fold-device`.  Native NCCL is the oracle.
The suite intentionally measures neither performance nor host progress.

## Prerequisites and build

The prerequisites and `config.mk` settings are the same as the top-level
build: CUDA (including `nvcc`), MPI, and NCCL.  NCCL 2.28 or newer is needed to
compile the AlltoAll/Gather/Scatter families.  Build the workload with:

```sh
make differential
```

Building the interposer remains `make`.  No Python package is required; the
runner uses Python 3's standard library.  `mpirun` must be able to launch local
processes (Open MPI containers may additionally need `--mpi-arg
--allow-run-as-root`).

## Running

```sh
# Both sides, requiring respectively 2, 4, and 8 physical GPUs for the oracle
python3 tests/differential/run_differential.py --ranks 2 --ncclfold ./ncclfold.so
python3 tests/differential/run_differential.py --ranks 2,4,8 --ncclfold ./ncclfold.so

# One side, selected families, permanent artifacts, and a 60-second bound
python3 tests/differential/run_differential.py --mode native --ranks 4
python3 tests/differential/run_differential.py --mode fold --ranks 8 --families p2p,ordering
python3 tests/differential/run_differential.py --ranks 2 --families allreduce \
  --timeout 60 --artifacts artifacts/differential
```

Valid families are `p2p`, `broadcast`, `allgather`, `alltoall`, `gather`,
`scatter`, `reduce`, `allreduce`, `reducescatter`, `ordering`, `group`,
`communicator`, and `malloc_async`.  Use `--case CASE_ID` with a family to
reproduce just one matrix case.  Every failure prints such a runner command.
`--mpi-arg` may be repeated for a site-specific MPI launcher option.

The runner queries `nvidia-smi`.  If fewer than N GPUs are visible, an N-rank
run requiring the native oracle is **SKIP**, not a failure.  A Fold-only run
requires one visible GPU.  The `malloc_async` workload writes a skip record if
the CUDA device reports no memory-pool support; availability of NCCL Fold's
POSIX-FD IPC path is otherwise tested by executing the call.  Launcher errors
are failures rather than false skips.

Each mode/family is a separate subprocess under `--timeout` (default 180
seconds).  On nonzero exit or timeout, its directory is retained with
`command.txt`, `stdout.txt`, `stderr.txt`, `status.json`, and any result files
already written.  `--artifacts DIR` retains all runs and `--keep-passing`
retains an automatically selected directory.

## Artifact and comparison contract

Each MPI rank writes `rank-R.jsonl`; stdout is never used as test data.  A row
contains case ID, API operation, world size/rank, datatype, reduction operator,
root/peer, element count, in-place flag, stream kind, named canary checks,
state, mode, and resulting values.  Inputs are deterministic functions of rank
and element index.  `(case_id, rank)` is the unique comparison key, and every
metadata or value mismatch is reported with its precise field.  Small buffers
are deliberate.

Copies and integer reductions are exact.  Floating reductions use the one
central policy in `run_differential.py`: `(absolute, relative)` tolerance is
`(2e-3,2e-3)` for float16, `(2e-2,2e-2)` for bfloat16,
`(2e-5,2e-5)` for float32, and `(1e-12,1e-12)` for float64.  Inputs are small,
positive, and well-conditioned.  A failure includes both values, absolute and
relative error, both tolerances, element index, rank, and case ID.

## Semantic coverage and intentional exclusions

The generated matrix covers all README datatypes; Sum, Product, Min, and Max;
counts 1, 3, and 7; roots zero, last, and (where possible) one; documented
in-place and out-of-place layouts; and non-default streams.  P2P covers paired
and ring traffic.  Focused stream tests enqueue GPU input production before
AllReduce, output consumption and source reuse after it, without an intervening
host synchronization.  Group tests cover nesting, outermost deferral, multiple
ordered operations, and the communicator family groups operations across two
isolated communicators with reversed logical rank numbering.  Query cases check
`ncclCommCount` and `ncclCommUserRank`.  Ordinary `cudaMalloc` is used throughout;
`malloc_async` separately exercises communication from async allocations.

As required by the top-level semantic contract, this suite does **not** test
CUDA Graph capture; custom/driver/managed/host allocations; custom pools;
unsupported NCCL entry points, datatypes, operators, aliases, or mismatched
calls; failure recovery; communicator lifetime violations; host return/progress;
algorithm/protocol/topology choices; timing, bandwidth, concurrency, memory
pressure, or device-ordinal validation.  Pool accounting and reuse-policy
differences are not compared.  Floating results are not required to be bitwise
identical.
