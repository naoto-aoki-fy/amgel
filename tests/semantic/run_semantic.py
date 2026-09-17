#!/usr/bin/env python3
"""Self-contained, Fold-only semantic oracle for the deterministic workload."""
import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

FAMILIES = ("p2p", "broadcast", "allgather", "alltoall", "gather", "scatter",
            "reduce", "allreduce", "reducescatter", "ordering", "group",
            "communicator", "malloc_async", "validation")
FLOAT_TOL = {"float16": (2e-3, 2e-3), "bfloat16": (2e-2, 2e-2),
             "float32": (2e-5, 2e-5), "float64": (1e-12, 1e-12)}


def gpu_count():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=10)
        return len([line for line in result.stdout.splitlines() if line.strip()]) \
            if result.returncode == 0 else 0
    except (OSError, subprocess.TimeoutExpired):
        return 0


def input_value(rank, index):
    return (rank + 1) * 2 + (index % 5) + 1


def reduced_value(row, index):
    values = ([2] + [1] * (row["ranks"] - 1) if row["reduction"] == "prod"
              else [input_value(r, index) for r in range(row["ranks"])])
    if row["reduction"] == "sum": return sum(values)
    if row["reduction"] == "prod": return math.prod(values)
    if row["reduction"] == "min": return min(values)
    if row["reduction"] == "max": return max(values)
    raise ValueError("unknown reduction " + str(row["reduction"]))


def expected(row):
    op, n, rank, count = row["operation"], row["ranks"], row["rank"], row["count"]
    if row["state"] == "skip": return None
    if row["operation"] in ("collective_metadata_mismatch", "p2p_datatype_mismatch"):
        return [1]
    if row["case_id"] == "ordering/produce-consume-reuse":
        return [sum((r + 1) * 10 + i + 1 for r in range(n)) + 1
                for i in range(4)] + [-7] * 4
    if op == "send_recv":
        source = (rank - 1 + n) % n if row["case_id"].endswith("ring") else rank ^ 1
        return [(source + 1) * 10 + i + 1 for i in range(count)]
    if op in ("broadcast", "bcast"):
        return [input_value(row["root"], i) for i in range(count)]
    if op == "allgather":
        return [input_value(r, i) for r in range(n) for i in range(count)]
    if op == "alltoall":
        return [input_value(source, rank * count + i)
                for source in range(n) for i in range(count)]
    if op == "gather":
        return ([] if rank != row["root"] else
                [input_value(r, i) for r in range(n) for i in range(count)])
    if op == "scatter":
        return [input_value(row["root"], rank * count + i) for i in range(count)]
    if op in ("reduce", "allreduce", "reducescatter"):
        if op == "reduce" and rank != row["root"]: return []
        offset = rank * count if op == "reducescatter" else 0
        return [reduced_value(row, offset + i) for i in range(count)]
    if op == "group":
        return [sum((r + 1) * 10 + 1 for r in range(n)),
                ((rank - 1 + n) % n + 1) * 10 + 1]
    if op == "comm_query": return [n, rank]
    if op == "comm_query_group":
        produced = [(r + 1) * 10 + 1 for r in range(n)]
        return [n, n - 1 - rank, sum(produced), max(produced)]
    raise ValueError("no oracle for operation " + op)


def compare(row):
    want = expected(row)
    if want is None: return []
    got = row["values"]
    prefix = (f"rank={row['rank']} operation={row['operation']} "
              f"datatype={row['datatype']} case={row['case_id']}")
    if len(got) != len(want):
        return [f"{prefix}: expected {len(want)} values, observed {len(got)}"]
    tolerance = FLOAT_TOL.get(row["datatype"]) if row["reduction"] else None
    errors = []
    for i, (observed, target) in enumerate(zip(got, want)):
        equal = (math.isclose(observed, target, rel_tol=tolerance[1],
                              abs_tol=tolerance[0]) if tolerance
                 else observed == target)
        if not equal:
            detail = (f", abs_tol={tolerance[0]}, rel_tol={tolerance[1]}"
                      if tolerance else " (exact equality required)")
            errors.append(f"{prefix} element={i}: expected={target!r}, "
                          f"observed={observed!r}{detail}")
    return errors


def run_family(args, ranks, family, root):
    directory = root / f"ranks-{ranks}" / family
    directory.mkdir(parents=True)
    command = ([args.mpirun, "-n", str(ranks)] + args.mpi_arg +
               [str(pathlib.Path(args.executable).resolve()), "--family", family,
                "--output", str(directory), "--mode", "fold"])
    env = os.environ.copy()
    preload = str(pathlib.Path(args.ncclfold).resolve())
    env["LD_PRELOAD"] = preload + ((":" + env["LD_PRELOAD"]) if env.get("LD_PRELOAD") else "")
    env["CUDA_VISIBLE_DEVICES"] = args.device
    (directory / "command.txt").write_text(" ".join(command) + "\n")
    started = time.time()
    try:
        process = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=args.timeout)
    except subprocess.TimeoutExpired as error:
        (directory / "stdout.txt").write_text(error.stdout or "")
        (directory / "stderr.txt").write_text(error.stderr or "")
        return [], [], f"timeout after {args.timeout}s"
    (directory / "stdout.txt").write_text(process.stdout)
    (directory / "stderr.txt").write_text(process.stderr)
    (directory / "status.json").write_text(json.dumps(
        {"returncode": process.returncode, "seconds": time.time() - started}, indent=2) + "\n")
    if process.returncode: return [], [], f"exit {process.returncode} (artifacts: {directory})"
    rows = []
    for path in sorted(directory.glob("rank-*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    if not rows: return [], [], f"no result records (artifacts: {directory})"
    failures = [error for row in rows for error in compare(row)]
    cases = {(row["case_id"], row["state"]) for row in rows}
    passed = sorted(case for case, state in cases if state == "pass")
    skipped = sorted(case for case, state in cases if state == "skip")
    return passed, skipped, failures


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ranks", default="2,4")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--executable", default="tests/semantic/nccl_semantics_test")
    parser.add_argument("--ncclfold", default="./ncclfold.so")
    parser.add_argument("--mpirun", default=os.environ.get("MPIRUN", "mpirun"))
    parser.add_argument("--mpi-arg", action="append", default=[])
    parser.add_argument("--device", default="0")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--artifacts")
    parser.add_argument("--keep-passing", action="store_true")
    args = parser.parse_args()
    ranks = [int(value) for value in args.ranks.split(",")]
    families = args.families.split(",")
    unknown = set(families) - set(FAMILIES)
    if unknown: parser.error("unknown families: " + ", ".join(sorted(unknown)))
    if gpu_count() < 1:
        print(f"SKIP: no visible NVIDIA GPU ({len(ranks) * len(families)} family runs skipped)")
        return 0
    temporary = args.artifacts is None
    root = pathlib.Path(args.artifacts or tempfile.mkdtemp(prefix="ncclfold-semantic-"))
    total_pass = total_skip = total_fail = 0
    for rank_count in ranks:
        for family in families:
            passed, skipped, errors = run_family(args, rank_count, family, root)
            if errors:
                total_fail += 1
                print(f"FAIL ranks={rank_count} family={family}", file=sys.stderr)
                for error in errors: print("  " + error, file=sys.stderr)
            else:
                total_pass += len(passed); total_skip += len(skipped)
                label = "SKIP" if skipped and not passed else "PASS"
                print(f"{label} ranks={rank_count} family={family} "
                      f"({len(passed)} passed, {len(skipped)} skipped)")
    print(f"\nSemantic test summary: PASS={total_pass} FAIL={total_fail} SKIP={total_skip}")
    if total_fail or args.keep_passing or not temporary:
        print("Artifacts: " + str(root))
    elif temporary:
        shutil.rmtree(root)
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
