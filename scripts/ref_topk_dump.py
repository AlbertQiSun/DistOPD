#!/usr/bin/env python3
"""delta-OPD step 1: pi_ref's top-K' at every action position of a frozen rollout cache. (GPU)

    python scripts/ref_topk_dump.py --line miro --cache precomputed_miro_full \\
        --pi-ref $M/pi_ref_miro --k 32 --align none
    python scripts/ref_topk_dump.py --report-only --cache precomputed_miro_full      # progress / report

WHY A PREFILL AND NOT THE ROLLOUT'S OWN LOGPROBS. The cache records hold input_ids, token_type,
teacher_logp, meta, terminated, num_search_calls -- and nothing from the student: the policy
sampler never asked vLLM for logprobs (no SamplingParams(logprobs=...) anywhere in opd/, envs/
or teacher/; the only logprob request in library code is the teacher server's
prompt_logprobs=0).  So pi_ref's row has to be produced here, one HF forward per trajectory
over the FULL cached context, lm_head applied only at the action rows
(fulldist_probe.chunked_logprobs: no [seq, V] tensor is ever built).

SHARDED AND RESUMABLE.  The caches hold 38,400 trajectories per line (tens of millions of
action positions; the moe line ~1,500 per trajectory), which the first version accumulated in
RAM and wrote once at the end -- a 3 h walltime would have killed all three jobs with nothing
on disk.  Now every --shard-size trajectories become one file
    <cache>/ref_topk/part_<start:06d>.npz        (written to .tmp, then renamed: atomic)
and a re-run skips the parts that exist, so a killed job loses at most one shard and a chain
of short jobs (backfill-friendly) finishes the cache.  Arrays in each part, aligned to the
cache's own indexing (cache_paths order, ACTION positions with pos > 0):
    traj, pos, y                int32
    top_ids [K'], top_lp [K']   int32 / fp32, pi_ref rank order (log_softmax in fp32 on bf16 weights)
    lp_y                        fp32, pi_ref's log-prob at the realised token from the FULL row,
                                whether or not y is inside the top-K' (the bucket case is recoverable)
    y_rank                      int32, y's rank in the full row (1 = argmax)
    topk_mass                   fp32, Σ exp(top_lp)
    row_sum                     fp32, Σ exp(full row) -- the log-sum-exp check, kept for every row
plus a `summary` (json) with the per-shard counts the report is assembled from.  When every part
exists, <cache>/ref_topk.npz is written with meta, gates, the report and the part list -- no
arrays; iter_parts() is the reader.

GATES.  Part 0 is gated AS SOON AS IT EXISTS, before any further GPU time is spent:
  first 100 positions:  Σ exp(top-K') <= 1 (+1e-6)  AND  y inside top-K' on >= 0.999 of them
  alignment:            at every (traj, pos) the tail_slope dump also scored (it read
                        cache_paths[:1000], i.e. part 0 at the default shard size), |lp_y here -
                        pi_ref's log-prob at y there| < 1e-3 and the token ids equal, else FATAL.
                        (The cache itself has no pi_ref number to align to; the tail_slope dump --
                        same weights, same chunked_logprobs path, top-1000 -- is the nearest.)
A failed gate keeps the part, writes <cache>/ref_topk/gates.json with passed=false and exits 1.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIELDS = ("traj", "pos", "y", "top_ids", "top_lp", "lp_y", "y_rank", "topk_mass", "row_sum")
N_FIRST = 100
Y_IN_K_MIN = 0.999
SUM_TOL = 1e-6
ALIGN_TOL = 1e-3
BIN_NAMES = ["[-4,0]", "(-8,-4)", "(-13,-8]", "(-inf,-13]"]


# ---------------------------------------------------------------- pure functions (tested)
def x2_bins(top_lp):
    x2 = np.asarray(top_lp)[:, 1].astype(np.float64)
    return np.select([x2 >= -4.0, x2 > -8.0, x2 > -13.0], [0, 1, 2], default=3)


def gates(top_lp, y_rank, row_sum, k, n_first=N_FIRST):
    """The first-`n_first` assertions as written, plus the rates over what was handed in."""
    top_lp, y_rank, row_sum = (np.asarray(v) for v in (top_lp, y_rank, row_sum))
    n = min(n_first, len(y_rank))
    mass = np.exp(top_lp[:n].astype(np.float64)).sum(1)
    in_k_first = float((y_rank[:n] <= k).mean()) if n else float("nan")
    ok_mass = bool((mass <= 1.0 + SUM_TOL).all()) if n else False
    ok_ink = bool(in_k_first >= Y_IN_K_MIN) if n else False
    return dict(n_first=n, max_topk_mass_first=float(mass.max()) if n else float("nan"), y_in_k_first=in_k_first,
                y_in_k_part0=float((y_rank <= k).mean()) if len(y_rank) else float("nan"),
                row_sum_min=float(row_sum.min()) if len(row_sum) else float("nan"),
                row_sum_max=float(row_sum.max()) if len(row_sum) else float("nan"),
                passed=ok_mass and ok_ink)


def align(traj, pos, y, lp_y, tail_traj, tail_pos, tail_y, tail_lp_ref, tail_tok_ids, tail_rank, tol=ALIGN_TOL, traj_max=None):
    """Compare this dump's lp_y with the tail_slope dump's pi_ref log-prob at y on the shared
    (traj, pos).  y is LOOKED UP in the dump's row (y_rank_ref can sit off its topk slot under
    fp32 ties).  Token inequality is an alignment failure (FATAL); |diff| >= tol fails the gate.
    traj_max restricts the comparison to the trajectories this dump covers."""
    where = {(int(t), int(p)): i for i, (t, p) in enumerate(zip(traj, pos))}
    diffs, n_tok_mismatch, n_skipped_rank, n_in_range = [], 0, 0, 0
    for j, (t, p) in enumerate(zip(tail_traj, tail_pos)):
        if traj_max is not None and int(t) >= traj_max:
            continue
        n_in_range += 1
        i = where.get((int(t), int(p)))
        if i is None:
            n_tok_mismatch += 1; continue                    # a scored position this dump does not have
        if int(y[i]) != int(tail_y[j]):
            n_tok_mismatch += 1; continue
        hit = np.where(tail_tok_ids[j] == tail_y[j])[0]
        if hit.size == 0:
            n_skipped_rank += 1; continue                    # y below the dump's top-K: nothing to compare
        diffs.append(float(lp_y[i]) - float(tail_lp_ref[j, hit[0]]))
    d = np.abs(np.asarray(diffs, np.float64))
    return dict(n_in_range=n_in_range, n_shared=len(diffs), n_token_mismatch=n_tok_mismatch, n_skipped_y_below_topK=n_skipped_rank,
                max_abs=float(d.max()) if d.size else float("nan"), mean_abs=float(d.mean()) if d.size else float("nan"),
                n_over_tol=int((d >= tol).sum()), tol=tol,
                passed=bool(n_tok_mismatch == 0 and d.size > 0 and d.max() < tol))


def dump_trajectory(model, ids, tt, k, device, row_chunk=64):
    """pi_ref's top-k and the realised-token quantities at every ACTION position (pos > 0) of one
    cached trajectory.  Rows come from fulldist_probe.chunked_logprobs in fp32; sums in float64."""
    import torch
    from opd.trajectory import TokenType
    from scripts.fulldist_probe import chunked_logprobs
    pos = torch.nonzero(tt == int(TokenType.ACTION), as_tuple=False).squeeze(-1)
    pos = pos[pos > 0]
    if pos.numel() == 0:
        return None
    rows, ys = pos - 1, ids[pos]
    out = {f: [] for f in FIELDS if f != "traj"}
    i0 = 0
    for chunk in chunked_logprobs(model, ids, rows, device, row_chunk):
        la = chunk.float().cpu()
        n = la.shape[0]; yy = ys[i0:i0 + n]
        r = torch.arange(n)
        top = torch.topk(la, k, dim=-1)
        lp_y = la[r, yy]
        out["top_ids"].append(top.indices.int().numpy()); out["top_lp"].append(top.values.numpy())
        out["lp_y"].append(lp_y.numpy())
        out["y_rank"].append(((la > lp_y[:, None]).sum(-1) + 1).int().numpy())
        out["topk_mass"].append(top.values.double().exp().sum(-1).float().numpy())
        out["row_sum"].append(la.double().exp().sum(-1).float().numpy())
        out["pos"].append(pos[i0:i0 + n].int().numpy()); out["y"].append(yy.int().numpy())
        i0 += n
    return {f: np.concatenate(v) for f, v in out.items()}


def part_summary(Z, start, end, n_empty, k, wall_s):
    b = x2_bins(Z["top_lp"]); ink = Z["y_rank"] <= k
    return dict(start=start, end=end, n_traj=end - start - n_empty, n_empty=n_empty, n_pos=int(len(Z["pos"])), k=k, wall_s=wall_s,
                bin_n=[int((b == i).sum()) for i in range(4)], bin_in_k=[int((ink & (b == i)).sum()) for i in range(4)],
                row_sum_min=float(Z["row_sum"].min()), row_sum_max=float(Z["row_sum"].max()))


def part_path(out_dir, start):
    return os.path.join(out_dir, f"part_{start:06d}.npz")


def plan_parts(n, shard):
    return list(range(0, n, shard))


def existing_parts(out_dir):
    return sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(os.path.join(out_dir, "part_??????.npz")))


def iter_parts(cache):
    """Reader for later steps: yields (summary dict, arrays dict) per part in cache order."""
    out_dir = os.path.join(cache, "ref_topk")
    for s in existing_parts(out_dir):
        z = np.load(part_path(out_dir, s), allow_pickle=True)
        yield json.loads(str(z["summary"])), {f: z[f] for f in FIELDS}


def assemble_report(meta, summaries, g, al, complete, adjudication=None):
    n_pos = sum(s["n_pos"] for s in summaries); n_traj = sum(s["n_traj"] for s in summaries)
    bin_n = np.sum([s["bin_n"] for s in summaries], 0); bin_k = np.sum([s["bin_in_k"] for s in summaries], 0)
    rs_min, rs_max = min(s["row_sum_min"] for s in summaries), max(s["row_sum_max"] for s in summaries)
    ok = bool(g and al and g["passed"] and al["passed"])
    L = [f"===== REF TOP-K DUMP  line={meta['line']}  cache={meta['cache']}  pi_ref={os.path.basename(meta['pi_ref'])}  K'={meta['k']}"
         f"  {'COMPLETE' if complete else 'PARTIAL'}: {len(summaries)}/{meta['n_parts']} parts",
         f"trajectories {n_traj} scored (+{sum(s['n_empty'] for s in summaries)} with no action position) of {meta['n_cache']}; "
         f"positions {n_pos}; template_fp(cache manifest) {meta['template_fp']}; GPU-seconds {sum(s['wall_s'] for s in summaries):.0f}",
         f"rows: fp32 log_softmax on bf16 weights; Σ exp(full row) in [{rs_min:.5f}, {rs_max:.5f}]", ""]
    if g:
        L += [f"GATE first {g['n_first']}: max Σ exp(top-K') {g['max_topk_mass_first']:.6f} (<= 1 + {SUM_TOL:g})   "
              f"y inside top-K' {g['y_in_k_first']:.3f} (>= {Y_IN_K_MIN})   -> {'PASS' if g['passed'] else 'FAIL'}"]
    if al:
        L += [f"ALIGNMENT vs tail_slope dump (trajectories < {meta['shard']}): dump positions in range {al['n_in_range']}, "
              f"compared {al['n_shared']}, token/position mismatches {al['n_token_mismatch']}, skipped (y below the dump's top-1000) "
              f"{al['n_skipped_y_below_topK']}",
              f"    |lp_y - lp_ref(y)|: max {al['max_abs']:.2e}  mean {al['mean_abs']:.2e}  n >= {al['tol']:g}: {al['n_over_tol']}   "
              f"-> {'PASS' if al['passed'] else 'FAIL (FATAL)'}"]
    L += ["", f"y inside top-K' over ALL scored positions: {bin_k.sum() / max(n_pos, 1):.4f}", "by x2 = log pi_ref(rank-2):"]
    L += [f"    {BIN_NAMES[i]:<12} n {int(bin_n[i]):>10}   y in K' {bin_k[i] / bin_n[i] if bin_n[i] else float('nan'):.4f}" for i in range(4)]
    verdict_line = "PASSED" if ok else ("FAILED, ADJUDICATED: " + adjudication["note"] if adjudication else "FAILED or not yet run")
    L += ["", f"GATES {verdict_line} -- parts in {meta['out_dir']}"
              + ("" if complete else "  (resubmit the same job to continue; it skips finished parts)")]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- the run
def flagged_positions(part0, T, tol=ALIGN_TOL, traj_max=None):
    """(traj, pos, y, lp_y_dump, lp_ref_tail) for every shared position with |diff| >= tol."""
    where = {(int(t), int(p)): i for i, (t, p) in enumerate(zip(part0["traj"], part0["pos"]))}
    out = []
    for j, (t, p) in enumerate(zip(T["traj"], T["pos"])):
        if traj_max is not None and int(t) >= traj_max:
            continue
        i = where.get((int(t), int(p)))
        if i is None or int(part0["y"][i]) != int(T["y"][j]):
            continue
        hit = np.where(T["tok_ids"][j] == T["y"][j])[0]
        if hit.size == 0:
            continue
        d = float(part0["lp_y"][i]) - float(T["lp_ref"][j, hit[0]])
        if abs(d) >= tol:
            out.append(dict(traj=int(t), pos=int(p), y=int(T["y"][j]), lp_dump=float(part0["lp_y"][i]), lp_tail=float(T["lp_ref"][j, hit[0]]),
                            diff=d, x2_tail=float(T["lp_ref"][j, 1]), tail_top1=float(T["lp_ref"][j, 0])))
    return out


def diagnose_align(out_dir, pi_ref, align_npz, shard, dtype, k):
    """ROOT CAUSE BEFORE ANY TOLERANCE MOVES. Re-score each flagged trajectory with
    row_chunk 32 (tail_slope's default) and 64 (this dump's), twice each, on THIS node, and print
    the dump's, the tail's and the four fresh numbers at every flagged position, with the row's
    top-2 so a near-tie is visible. Same numbers twice under one chunk and different across
    chunks = the lm_head kernel; different across repeats = a non-deterministic kernel; all four
    equal to one side = a genuine cross-node / cross-run difference."""
    import torch
    import transformers
    from scripts.cross_score_cache import cache_paths
    from scripts.fulldist_probe import chunked_logprobs
    part0 = np.load(part_path(out_dir, 0), allow_pickle=True)
    T = np.load(align_npz, allow_pickle=True)
    flagged = flagged_positions(part0, T, traj_max=shard)
    print(f"[diag] {len(flagged)} flagged positions (|diff| >= {ALIGN_TOL}) among trajectories < {shard}")
    if not flagged:
        json.dump([], open(os.path.join(out_dir, "gates_diag.json"), "w")); return 0
    cache = os.path.dirname(os.path.abspath(out_dir)); paths = cache_paths(cache)
    dev = "cuda:0" if torch.cuda.device_count() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(pi_ref, dtype=getattr(torch, dtype), device_map={"": dev},
                                                              low_cpu_mem_usage=True).eval()
    rows_out = []
    for f in flagged:
        rec = torch.load(paths[f["traj"]], weights_only=False, map_location="cpu")
        ids = rec["input_ids"].long(); q = f["pos"]; y = int(ids[q])
        assert y == f["y"], f"token at traj {f['traj']} pos {q} is {y}, tail says {f['y']}"
        tt = rec["token_type"].long()
        from opd.trajectory import TokenType
        pos = torch.nonzero(tt == int(TokenType.ACTION), as_tuple=False).squeeze(-1); pos = pos[pos > 0]
        j = int((pos == q).nonzero()[0, 0])
        fresh = {}
        for chunk in (32, 64):
            for rep in (1, 2):
                la = torch.cat([c.float().cpu() for c in chunked_logprobs(model, ids, pos - 1, dev, chunk)], 0)
                top2 = torch.topk(la[j], 2).values.tolist()
                fresh[f"chunk{chunk}_rep{rep}"] = dict(lp_y=float(la[j, y]), top1=top2[0], top2=top2[1],
                                                       y_rank=int((la[j] > la[j, y]).sum()) + 1)
        line = dict(**f, fresh=fresh, node=os.environ.get("SLURMD_NODENAME", "?"), gpu=torch.cuda.get_device_name(0) if dev != "cpu" else "cpu")
        rows_out.append(line)
        print(f"[diag] traj {f['traj']} pos {q} y {y}: dump {f['lp_dump']:+.6f}  tail {f['lp_tail']:+.6f}  (tail top1 {f['tail_top1']:+.4f}, x2 {f['x2_tail']:+.4f})")
        for kk, v in fresh.items():
            print(f"        {kk}: lp_y {v['lp_y']:+.6f}  top1 {v['top1']:+.4f}  top2 {v['top2']:+.4f}  y_rank {v['y_rank']}")
    json.dump(rows_out, open(os.path.join(out_dir, "gates_diag.json"), "w"), indent=1)
    print(f"[diag] wrote {os.path.join(out_dir, 'gates_diag.json')}")
    return 0


def run_gates(out_dir, k, align_npz, shard):
    z = np.load(part_path(out_dir, 0), allow_pickle=True)
    g = gates(z["top_lp"], z["y_rank"], z["row_sum"], k)
    if align_npz and os.path.exists(align_npz):
        T = np.load(align_npz, allow_pickle=True)
        al = align(z["traj"], z["pos"], z["y"], z["lp_y"], T["traj"], T["pos"], T["y"], T["lp_ref"], T["tok_ids"], T["y_rank_ref"],
                   traj_max=shard)
    elif align_npz in (None, "", "none"):
        # A line with no earlier pi_ref dump (the no-SFT line) has nothing to align against;
        # the gate is SKIPPED, not failed, and the report says so.
        al = dict(n_in_range=0, n_shared=0, n_token_mismatch=0, n_skipped_y_below_topK=0, max_abs=float("nan"), mean_abs=float("nan"),
                  n_over_tol=0, tol=ALIGN_TOL, passed=True, skipped=True, note="no alignment reference given (--align none): gate skipped")
    else:
        al = dict(n_in_range=0, n_shared=0, n_token_mismatch=0, n_skipped_y_below_topK=0, max_abs=float("nan"), mean_abs=float("nan"),
                  n_over_tol=0, tol=ALIGN_TOL, passed=False, note=f"no alignment reference at {align_npz}")
    res = dict(gates=g, align=al, passed=bool(g["passed"] and al["passed"]))
    json.dump(res, open(os.path.join(out_dir, "gates.json"), "w"), indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default=""); ap.add_argument("--cache", required=True)
    ap.add_argument("--pi-ref", default=None); ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--align", default=None, help="optional npz of reference top-K rows to cross-check against, or none")
    ap.add_argument("--n", type=int, default=None, help="first n trajectories only (a smoke)")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--row-chunk", type=int, default=64); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--ignore-gates", action="store_true", help="continue a dump whose gates.json says failed (after adjudication)")
    ap.add_argument("--adjudicate", default=None, metavar="NOTE",
                    help="record a human ruling on a failed gate in gates.json (with gates_diag.json attached) and "
                         "re-assemble the report; the gate result itself is never rewritten")
    ap.add_argument("--diagnose-align", action="store_true",
                    help="re-score the positions where part 0 and the tail_slope dump disagree by >= tol, under both row "
                         "chunk sizes and twice each, and print every number side by side; writes gates_diag.json")
    a = ap.parse_args(argv)
    out_dir = os.path.join(a.cache, "ref_topk"); os.makedirs(out_dir, exist_ok=True)
    from scripts.cross_score_cache import cache_paths
    paths = cache_paths(a.cache)
    man = json.load(open(os.path.join(a.cache, "manifest.json"), encoding="utf-8"))
    template_fp = (man.get("backend") or {}).get("template_fp", "?")
    n_cache = len(paths)
    if a.n:
        paths = paths[:a.n]
    parts = plan_parts(len(paths), a.shard_size)
    meta = dict(line=a.line, cache=os.path.abspath(a.cache), pi_ref=os.path.abspath(a.pi_ref) if a.pi_ref else "?", k=a.k,
                n_cache=n_cache, n_planned=len(paths), n_parts=len(parts), shard=a.shard_size, template_fp=template_fp,
                out_dir=os.path.abspath(out_dir), dtype=a.dtype)

    def finish():
        have = existing_parts(out_dir)
        summaries = [json.loads(str(np.load(part_path(out_dir, s), allow_pickle=True)["summary"])) for s in have]
        gj = os.path.join(out_dir, "gates.json")
        G = json.load(open(gj)) if os.path.exists(gj) else {}
        complete = set(have) >= set(parts)
        if not summaries:
            print(f"[reftopk] no parts yet in {out_dir}"); return 1
        txt = assemble_report(meta, summaries, G.get("gates"), G.get("align"), complete, G.get("adjudication"))
        print(txt)
        accepted = bool(G.get("passed", False) or G.get("adjudication"))
        if complete:
            np.savez_compressed(os.path.join(a.cache, "ref_topk.npz"), meta=np.array(json.dumps(meta)), gates=np.array(json.dumps(G)),
                                gates_passed=np.bool_(G.get("passed", False)), adjudicated=np.bool_(bool(G.get("adjudication"))),
                                usable=np.bool_(accepted), report=np.array(txt), parts=np.array(have))
            open(os.path.join(a.cache, "ref_topk_report.txt"), "w", encoding="utf-8").write(txt)
        return 0 if (complete and accepted) else 1

    if a.adjudicate:
        if not os.path.exists(gj_path := os.path.join(out_dir, "gates.json")):
            raise SystemExit(f"FATAL: {gj_path} missing -- nothing to adjudicate")
        G = json.load(open(gj_path))
        diag_p = os.path.join(out_dir, "gates_diag.json")
        G["adjudication"] = dict(note=a.adjudicate, when=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 diag=json.load(open(diag_p)) if os.path.exists(diag_p) else None)
        json.dump(G, open(gj_path, "w"), indent=1)
        print(f"[reftopk] adjudication recorded in {gj_path}: {a.adjudicate}")
    if a.report_only or a.adjudicate:
        done_p = os.path.join(a.cache, "ref_topk.npz")
        if os.path.exists(done_p) and not a.pi_ref:
            meta.update({k: v for k, v in json.loads(str(np.load(done_p, allow_pickle=True)["meta"])).items() if k in ("line", "pi_ref", "dtype")})
        return finish()

    import torch
    import transformers
    if not a.pi_ref:
        raise SystemExit("FATAL: --pi-ref is required for a dump")
    todo = [s for s in parts if s not in set(existing_parts(out_dir))]
    print(f"[reftopk] cache {a.cache}: {len(paths)} trajectories in {len(parts)} parts of {a.shard_size}; "
          f"{len(parts) - len(todo)} done, {len(todo)} to do; K'={a.k}; template_fp {template_fp}", flush=True)
    gj = os.path.join(out_dir, "gates.json")
    if a.diagnose_align:
        return diagnose_align(out_dir, a.pi_ref, a.align, a.shard_size, a.dtype, a.k)
    if 0 not in todo and not os.path.exists(gj):
        res = run_gates(out_dir, a.k, a.align, a.shard_size)
        if not res["passed"]:
            print(f"FATAL: part 0 fails its gates ({json.dumps(res)})", flush=True); finish(); return 1
    if 0 not in todo and os.path.exists(gj) and not json.load(open(gj)).get("passed", False) and not a.ignore_gates:
        # the 2026-09-05 dense/moe chains kept scoring after their part 0 had FAILED the
        # alignment gate: the continuation only asked whether gates.json EXISTED.
        print(f"FATAL: {gj} records a failed gate; a continuation must not spend GPU time on an unverified dump "
              f"(--ignore-gates overrides, once the failure has been adjudicated)", flush=True); finish(); return 1
    if not todo:
        return finish()
    dt = getattr(torch, a.dtype)
    dev = "cuda:0" if torch.cuda.device_count() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(a.pi_ref, dtype=dt, device_map={"": dev}, low_cpu_mem_usage=True).eval()
    print(f"[reftopk] pi_ref {a.pi_ref} on {dev}", flush=True)
    for s in todo:
        t0 = time.time(); e = min(s + a.shard_size, len(paths))
        cols = {f: [] for f in FIELDS}; n_empty = 0
        for ti in range(s, e):
            rec = torch.load(paths[ti], weights_only=False, map_location="cpu")
            d = dump_trajectory(model, rec["input_ids"].long(), rec["token_type"].long(), a.k, dev, a.row_chunk)
            if d is None:
                n_empty += 1; continue
            for f in FIELDS:
                cols[f].append(np.full(len(d["pos"]), ti, np.int32) if f == "traj" else d[f])
        if not cols["pos"]:
            Z = {f: np.zeros((0, a.k) if f in ("top_ids", "top_lp") else 0, np.int32 if f in ("traj", "pos", "y", "top_ids", "y_rank") else np.float32) for f in FIELDS}
            summ = dict(start=s, end=e, n_traj=0, n_empty=n_empty, n_pos=0, k=a.k, wall_s=time.time() - t0, bin_n=[0] * 4, bin_in_k=[0] * 4,
                        row_sum_min=1.0, row_sum_max=1.0)
        else:
            Z = {f: np.concatenate(cols[f]) for f in FIELDS}
            summ = part_summary(Z, s, e, n_empty, a.k, time.time() - t0)
        tmp = part_path(out_dir, s) + ".tmp.npz"
        np.savez_compressed(tmp, **Z, summary=np.array(json.dumps(summ)))
        os.replace(tmp, part_path(out_dir, s))
        print(f"[reftopk] part {s:06d} ({s}-{e}): {summ['n_pos']} positions in {summ['wall_s']:.0f}s  "
              f"[{len(existing_parts(out_dir))}/{len(parts)} parts]", flush=True)
        if s == 0:
            res = run_gates(out_dir, a.k, a.align, a.shard_size)
            print(f"[reftopk] part 0 gates: {json.dumps(res)}", flush=True)
            if not res["passed"]:
                print("FATAL: part 0 fails its gates -- stopping before more GPU time is spent", flush=True); finish(); return 1
    return finish()


if __name__ == "__main__":
    sys.exit(main())
