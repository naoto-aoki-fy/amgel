#!/usr/bin/env python3
"""Run identical NCCL semantic workloads natively and through NCCL Fold."""
import argparse, json, math, os, pathlib, shutil, subprocess, sys, tempfile, time

FAMILIES = ("p2p", "broadcast", "allgather", "alltoall", "gather", "scatter",
            "reduce", "allreduce", "reducescatter", "ordering", "group",
            "communicator", "malloc_async")
FLOAT_TOL = {"float16": (2e-3, 2e-3), "bfloat16": (2e-2, 2e-2),
             "float32": (2e-5, 2e-5), "float64": (1e-12, 1e-12)}

def gpu_count():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
        if p.returncode == 0: return len([x for x in p.stdout.splitlines() if x.strip()])
    except (OSError, subprocess.TimeoutExpired): pass
    return 0

def load_records(directory):
    out = {}
    for path in sorted(directory.glob("rank-*.jsonl")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            try: row = json.loads(line)
            except ValueError as e: raise RuntimeError(f"invalid JSON {path}:{line_no}: {e}")
            key = (row["case_id"], row["rank"])
            if key in out: raise RuntimeError(f"duplicate result {key} in {path}")
            out[key] = row
    return out

def compare(native, fold):
    errors = []
    if native.keys() != fold.keys():
        for k in sorted(native.keys() - fold.keys()): errors.append(f"{k}: missing from fold")
        for k in sorted(fold.keys() - native.keys()): errors.append(f"{k}: missing from native")
    ignored = {"mode", "values"}
    for key in sorted(native.keys() & fold.keys()):
        a, b = native[key], fold[key]
        for field in sorted((a.keys() | b.keys()) - ignored):
            if a.get(field) != b.get(field):
                errors.append(f"case={key[0]} rank={key[1]} field={field}: native={a.get(field)!r} fold={b.get(field)!r}")
        av, bv = a.get("values", []), b.get("values", [])
        if len(av) != len(bv): errors.append(f"case={key[0]} rank={key[1]} field=values.length: native={len(av)} fold={len(bv)}"); continue
        tol = FLOAT_TOL.get(a.get("datatype")) if a.get("reduction") not in (None, "none") else None
        for i, (x, y) in enumerate(zip(av, bv)):
            if tol:
                ae, re = abs(x-y), abs(x-y)/max(abs(x), abs(y), 1e-300)
                if not math.isclose(x, y, rel_tol=tol[1], abs_tol=tol[0]):
                    errors.append(f"case={key[0]} rank={key[1]} element={i}: native={x!r} fold={y!r} abs_error={ae:.17g} rel_error={re:.17g} abs_tol={tol[0]} rel_tol={tol[1]}")
            elif x != y: errors.append(f"case={key[0]} rank={key[1]} element={i}: native={x!r} fold={y!r}")
    return errors

def run_one(args, mode, ranks, family, root):
    dest = root / f"ranks-{ranks}" / mode / family; dest.mkdir(parents=True, exist_ok=True)
    exe = str(pathlib.Path(args.executable).resolve())
    cmd = [args.mpirun, "-n", str(ranks)] + args.mpi_arg + [exe, "--ranks", str(ranks), "--family", family, "--output", str(dest), "--mode", mode]
    env = os.environ.copy()
    if mode == "fold":
        env["LD_PRELOAD"] = str(pathlib.Path(args.ncclfold).resolve()) + ((":" + env["LD_PRELOAD"]) if env.get("LD_PRELOAD") else "")
        env["CUDA_VISIBLE_DEVICES"] = args.fold_device
    (dest/"command.txt").write_text(" ".join(cmd)+"\n")
    started=time.time()
    try:
        p=subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=args.timeout)
        status={"returncode":p.returncode,"timeout":False,"seconds":time.time()-started}
    except subprocess.TimeoutExpired as e:
        p=e; status={"returncode":None,"timeout":True,"seconds":time.time()-started}
    (dest/"stdout.txt").write_text(p.stdout or ""); (dest/"stderr.txt").write_text(p.stderr or "")
    (dest/"status.json").write_text(json.dumps(status,indent=2)+"\n")
    if status["timeout"]: return None, f"TIMEOUT after {args.timeout}s (artifacts: {dest})"
    if status["returncode"]: return None, f"exit {status['returncode']} (artifacts: {dest})"
    try: return load_records(dest), None
    except Exception as e: return None, f"artifact error: {e} (artifacts: {dest})"

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ranks",default="2,4,8"); ap.add_argument("--mode",choices=("both","native","fold"),default="both")
    ap.add_argument("--families",default=",".join(FAMILIES)); ap.add_argument("--case",help="exact case id (passed as a family-independent filter)")
    ap.add_argument("--ncclfold",default="./ncclfold.so"); ap.add_argument("--executable",default="tests/differential/nccl_semantics_test")
    ap.add_argument("--mpirun",default=os.environ.get("MPIRUN","mpirun")); ap.add_argument("--mpi-arg",action="append",default=[])
    ap.add_argument("--timeout",type=float,default=180); ap.add_argument("--artifacts",default=None); ap.add_argument("--keep-passing",action="store_true")
    ap.add_argument("--fold-device",default="0",help="physical CUDA device exposed to Fold ranks")
    args=ap.parse_args(); ranks=[int(x) for x in args.ranks.split(",")]; families=args.families.split(",")
    bad=set(families)-set(FAMILIES)
    if bad: ap.error("unknown families: "+",".join(sorted(bad)))
    if args.case: os.environ["NCCL_SEMANTICS_CASE"] = args.case
    root=pathlib.Path(args.artifacts or tempfile.mkdtemp(prefix="ncclfold-differential-")); root.mkdir(parents=True,exist_ok=True)
    available=gpu_count(); total_pass=total_fail=total_skip=0; summaries=[]
    modes=("native","fold") if args.mode=="both" else (args.mode,)
    for n in ranks:
        if available < 1:
            total_skip+=len(families); summaries.append((n,"SKIP","no visible physical GPUs")); continue
        if "native" in modes and available<n:
            total_skip+=len(families); summaries.append((n,"SKIP",f"insufficient physical GPUs ({available} available, {n} required)")); continue
        rank_pass=rank_fail=rank_skip=0
        for family in families:
            results={}; errs=[]
            for mode in modes:
                results[mode], err=run_one(args,mode,n,family,root)
                if err: errs.append(f"{mode} {family}: {err}")
            if not errs and args.mode=="both": errs=compare(results["native"],results["fold"])
            if errs:
                rank_fail+=1; print(f"FAIL ranks={n} family={family}",file=sys.stderr)
                for e in errs: print("  "+e,file=sys.stderr)
                print(f"  reproduce: {sys.executable} {' '.join(sys.argv[1:])} --ranks {n} --families {family}",file=sys.stderr)
            else:
                rows=next(iter(results.values())); case_rows={row["case_id"]:row for row in rows.values()}
                skipped=sum(row.get("state")=="skip" for row in case_rows.values()); passed=len(case_rows)-skipped
                rank_pass+=passed; rank_skip+=skipped
                label="SKIP" if skipped and not passed else "PASS"
                print(f"{label} ranks={n} family={family} ({passed} cases passed, {skipped} skipped)")
        total_pass+=rank_pass; total_fail+=rank_fail; total_skip+=rank_skip
        summaries.append((n,"FAIL" if rank_fail else "PASS",f"{rank_pass} cases, {rank_fail} failed families, {rank_skip} skipped"))
    print("\nDifferential test summary")
    for n,s,d in summaries: print(f"  ranks={n}: {s}  {d}")
    print(f"\n  passed: {total_pass}\n  failed: {total_fail}\n  skipped: {total_skip}\n  artifacts: {root}")
    if not args.keep_passing and not total_fail and not args.artifacts: shutil.rmtree(root)
    return 1 if total_fail else 0
if __name__=="__main__": sys.exit(main())
