#!/usr/bin/env python3
"""delta-OPD step 2: the TEACHER HEAD at every soft position of a frozen rollout cache, from the
served teacher (the same engine that scored the cache's teacher_logp).

    python scripts/teacher_head_dump.py --line miro --cache precomputed_miro_full \\
        --teacher $M/MiroThinker-1.7-mini --teacher-endpoint $TEACHER_ENDPOINT --tokenizer $M/pi_ref_miro
    python scripts/teacher_head_dump.py --report-only --cache precomputed_miro_full

SOFT = x2 > -8 read from <cache>/ref_topk/part_*.npz (top_lp[:, 1]); the hard positions keep
the existing delta_y (teacher_logp in the record) and are not scored here.  Per soft position:
    K_q       teacher top-K_q ids + fp32 log q (cover 0.99, cap 32; the server cuts it from one
              prompt_logprobs=k_wide pass, teacher.head.head_from_entry)
    extras    log q at S \\ K_q = pi_ref's top-K' ∪ {y} not in K_q; NaN when the token sits below
              the engine's top-k_wide (then only log q < floor_lp is known; counted)
    log_rest  log(1 − Σ_S q) with unknown extras counted as 0 (an upper bound on the residual)
Written as <cache>/teacher_head/part_<start>.npz with the SAME shard starts as ref_topk (the
delta reader walks both by trajectory), resumable; part 0 is gated before the rest is spent:
    template   the server's BackendConfig.template_fp == the cache manifest's (FATAL otherwise)
    alignment  |lp_y here − record teacher_logp[pos]| < 1e-3 at every soft position (FATAL
               otherwise: it is the same engine, so this is an alignment error, not noise)
Report (assembled from the parts): soft count and share, |K_q| median/IQR/max, |S| mean,
|K_q \\ K'| by role, the residual mass distribution, unknown-extra rate, cache size on disk.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teacher.head import KQ_CAP, KQ_COVER, residual_log_mass  # noqa: E402

SOFT_X2 = -8.0
ALIGN_TOL = 1e-3
FIELDS = ("traj", "pos", "y", "kq", "q_ids", "q_lp", "extra_ids", "extra_lp", "floor_lp", "lp_y", "log_rest",
          "n_unknown", "cover_reached", "s_size", "kq_not_kp", "role_id")


# ---------------------------------------------------------------- pure functions (tested)
def soft_rows(ref):
    return np.asarray(ref["top_lp"])[:, 1] > SOFT_X2


def requests_for(ref, soft, ids_by_traj):
    """[(traj, input_ids tensor, positions, extras)] per trajectory of one ref part, soft rows only."""
    import torch
    out = []
    tr = np.asarray(ref["traj"]); pos = np.asarray(ref["pos"]); y = np.asarray(ref["y"]); top = np.asarray(ref["top_ids"])
    for t in np.unique(tr):
        m = (tr == t) & soft
        if not m.any():
            continue
        exs = [[int(v) for v in top[i]] + [int(y[i])] for i in np.where(m)[0]]
        out.append((int(t), torch.as_tensor(ids_by_traj[int(t)]).long(), [int(p) for p in pos[m]], exs))
    return out


def pack_heads(heads, ref_rows, kp_ids, k_cap=KQ_CAP):
    """One position's head dicts -> the part's per-row arrays (padded)."""
    n = len(heads); E = kp_ids.shape[1] + 1
    q_ids = np.full((n, k_cap), -1, np.int32); q_lp = np.full((n, k_cap), np.nan, np.float32)
    ex_ids = np.full((n, E), -1, np.int32); ex_lp = np.full((n, E), np.nan, np.float32)
    kq = np.zeros(n, np.int32); floor = np.zeros(n, np.float32); lp_y = np.zeros(n, np.float32); rest = np.zeros(n, np.float32)
    unk = np.zeros(n, np.int32); cov = np.zeros(n, bool); s_size = np.zeros(n, np.int32); kq_not = np.zeros(n, np.int32)
    for i, h in enumerate(heads):
        k = int(h["kq"]); kq[i] = k
        q_ids[i, :k] = h["top_ids"]; q_lp[i, :k] = h["top_lp"]
        e = len(h["extra_ids"])
        ex_ids[i, :e] = h["extra_ids"]; ex_lp[i, :e] = [np.nan if v is None else v for v in h["extra_lp"]]
        floor[i] = h["floor_lp"]; lp_y[i] = h["lp_y"]; cov[i] = bool(h["cover_reached"])
        unk[i] = sum(1 for v in h["extra_lp"] if v is None)
        rest[i] = residual_log_mass(h["top_lp"], h["extra_lp"])
        kp = set(int(t) for t in kp_ids[i]); s = kp | set(int(t) for t in h["top_ids"]) | {int(ref_rows["y"][i])}
        s_size[i] = len(s); kq_not[i] = sum(1 for t in h["top_ids"] if int(t) not in kp)
    return dict(kq=kq, q_ids=q_ids, q_lp=q_lp, extra_ids=ex_ids, extra_lp=ex_lp, floor_lp=floor, lp_y=lp_y, log_rest=rest,
                n_unknown=unk, cover_reached=cov, s_size=s_size, kq_not_kp=kq_not)


def align_check(lp_y, cached, tol=ALIGN_TOL):
    d = np.abs(np.asarray(lp_y, np.float64) - np.asarray(cached, np.float64))
    return dict(n=int(d.size), max_abs=float(d.max()) if d.size else float("nan"), mean_abs=float(d.mean()) if d.size else float("nan"),
                n_over_tol=int((d >= tol).sum()), tol=tol, passed=bool(d.size > 0 and d.max() < tol))


def part_path(out_dir, start):
    return os.path.join(out_dir, f"part_{start:06d}.npz")


def existing_parts(out_dir):
    return sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(os.path.join(out_dir, "part_??????.npz")))


def med_iqr(v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return (float("nan"),) * 3
    a, b, c = np.percentile(v, [50, 25, 75]); return float(a), float(b), float(c)


def assemble_report(meta, parts, G, complete, role_vocab):
    n_ref = sum(p["n_ref"] for p in parts); n_soft = sum(p["n_soft"] for p in parts)
    kq = np.concatenate([p["kq"] for p in parts]) if parts else np.zeros(0)
    s_size = np.concatenate([p["s_size"] for p in parts]) if parts else np.zeros(0)
    rest = np.exp(np.concatenate([p["log_rest"] for p in parts])) if parts else np.zeros(0)
    unk = np.concatenate([p["n_unknown"] for p in parts]) if parts else np.zeros(0)
    cov = np.concatenate([p["cover_reached"] for p in parts]) if parts else np.zeros(0)
    knk = np.concatenate([p["kq_not_kp"] for p in parts]) if parts else np.zeros(0)
    role = np.concatenate([p["role_id"] for p in parts]) if parts else np.zeros(0)
    floor = np.exp(np.concatenate([p["floor_lp"] for p in parts]).astype(np.float64)) if parts else np.zeros(0)
    size = sum(p["bytes"] for p in parts)
    f = lambda v: "{:.3f} [{:.3f},{:.3f}]".format(*med_iqr(v))
    L = [f"===== TEACHER HEAD  line={meta['line']}  cache={meta['cache']}  teacher={os.path.basename(meta['teacher'])}  "
         f"k_wide={meta['k_wide']} k_cap={meta['k_cap']} cover={meta['cover']}  {'COMPLETE' if complete else 'PARTIAL'}: {len(parts)}/{meta['n_parts']} parts",
         f"template_fp server={meta.get('server_template_fp')} cache={meta.get('cache_template_fp')}  GPU-seconds {sum(p['wall_s'] for p in parts):.0f}",
         f"soft positions {n_soft} of {n_ref} scored positions ({n_soft / max(n_ref, 1):.4f}); hard positions keep the record's delta_y",
         f"|K_q|: median [IQR] {f(kq)}  max {int(kq.max()) if kq.size else 0}  cover {meta['cover']} reached within top-{meta['k_wide']} on {cov.mean() if cov.size else float('nan'):.4f}",
         f"|S| = |K' ∪ K_q ∪ {{y}}|: mean {s_size.mean() if s_size.size else float('nan'):.2f}  (K' = {meta['k_prime']})",
         f"|K_q \\ K'|: mean {knk.mean() if knk.size else float('nan'):.3f}  median [IQR] {f(knk)}",
         "  by role:"]
    for ri, r in enumerate(role_vocab):
        m = role == ri
        if m.any():
            L.append(f"    {r:<14} n {int(m.sum()):>10}  |K_q \\ K'| mean {knk[m].mean():.3f}  |K_q| median {np.median(kq[m]):.0f}  residual median {np.median(rest[m]):.2e}")
    if (role < 0).any():
        L.append(f"    {'(no role)':<14} n {int((role < 0).sum()):>10}")
    L += [f"teacher residual off S (1 − Σ_S q): median [IQR] {'{:.2e} [{:.2e},{:.2e}]'.format(*med_iqr(rest))}  p99 {np.percentile(rest, 99) if rest.size else float('nan'):.2e}  max {rest.max() if rest.size else float('nan'):.2e}",
          f"extras below the engine's top-{meta['k_wide']} (log q floored): {unk.sum() if unk.size else 0} tokens on {(unk > 0).mean() if unk.size else float('nan'):.4f} of soft positions",
          f"    their mass is bounded by n_unknown x q(rank {meta['k_wide']}): per position median {np.median(unk * floor) if unk.size else float('nan'):.2e}  "
          f"p99 {np.percentile(unk * floor, 99) if unk.size else float('nan'):.2e}  max {(unk * floor).max() if unk.size else float('nan'):.2e}   "
          f"(q at rank {meta['k_wide']}: median {np.median(floor) if floor.size else float('nan'):.2e})",
          f"cache size on disk: {size / 2**30:.2f} GiB in {len(parts)} parts", ""]
    if G:
        al = G.get("align", {})
        L += [f"GATE template: {'PASS' if G.get('template_ok') else 'FAIL'}",
              f"GATE alignment (lp_y vs record teacher_logp, part 0): n {al.get('n')}  max {al.get('max_abs', float('nan')):.2e}  "
              f"mean {al.get('mean_abs', float('nan')):.2e}  n >= {al.get('tol')}: {al.get('n_over_tol')}  -> {'PASS' if al.get('passed') else 'FAIL (FATAL)'}"]
        adj = G.get("adjudication")
        ok = bool(G.get("passed")) or bool(adj)
        L.append(f"GATES {'PASSED' if G.get('passed') else ('FAILED, ADJUDICATED: ' + adj['note'] if adj else 'FAILED')}")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- the run
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default=""); ap.add_argument("--cache", required=True)
    ap.add_argument("--teacher", default=None, help="teacher model dir (tokenizer_name / fingerprint of the served backend)")
    ap.add_argument("--teacher-endpoint", default=None); ap.add_argument("--tokenizer", default=None, help="for roles (pi_ref dir)")
    ap.add_argument("--k-wide", type=int, default=256); ap.add_argument("--k-cap", type=int, default=KQ_CAP); ap.add_argument("--cover", type=float, default=KQ_COVER)
    ap.add_argument("--shard-size", type=int, default=1000); ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--report-only", action="store_true"); ap.add_argument("--adjudicate", default=None, metavar="NOTE")
    ap.add_argument("--ignore-gates", action="store_true"); ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    from scripts.cross_score_cache import cache_paths
    from scripts.role_vocab import ROLE_VOCAB
    ref_dir = os.path.join(a.cache, "ref_topk")
    man = json.load(open(os.path.join(a.cache, "manifest.json"), encoding="utf-8"))
    cache_fp = (man.get("backend") or {}).get("template_fp", "?")
    paths = cache_paths(a.cache)
    if a.n:
        paths = paths[:a.n]
    # A --n that stops INSIDE a shard is a smoke: its half-part must not sit in teacher_head/
    # where a later full run would take it for a finished part (it would, silently).
    smoke = bool(a.n) and (a.n % a.shard_size != 0)
    out_dir = os.path.join(a.cache, "teacher_head_smoke" if smoke else "teacher_head"); os.makedirs(out_dir, exist_ok=True)
    if smoke:
        print(f"[head] --n {a.n} is not a whole shard: writing to {out_dir} (a smoke, never read by the trainer)", flush=True)
    ref_parts = sorted(int(os.path.basename(p)[5:11]) for p in glob.glob(os.path.join(ref_dir, "part_??????.npz")))
    parts = [s for s in ref_parts if s < len(paths)]
    if not parts:
        raise SystemExit(f"FATAL: no ref_topk parts under {ref_dir} -- run scripts/ref_topk_dump.py first")
    gj = os.path.join(out_dir, "gates.json")
    meta_path = os.path.join(out_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.update(dict(line=a.line or meta.get("line", ""), cache=os.path.abspath(a.cache), teacher=a.teacher or meta.get("teacher", "?"),
                     k_wide=a.k_wide, k_cap=a.k_cap, cover=a.cover, n_parts=len(parts), shard=a.shard_size, cache_template_fp=cache_fp,
                     k_prime=meta.get("k_prime")))

    def finish():
        have = existing_parts(out_dir)
        P = []
        for s in have:
            z = np.load(part_path(out_dir, s), allow_pickle=True)
            summ = json.loads(str(z["summary"]))
            P.append(dict(summ, kq=z["kq"], s_size=z["s_size"], log_rest=z["log_rest"], n_unknown=z["n_unknown"], cover_reached=z["cover_reached"],
                          kq_not_kp=z["kq_not_kp"], role_id=z["role_id"], floor_lp=z["floor_lp"], bytes=os.path.getsize(part_path(out_dir, s))))
        if not P:
            print(f"[head] no parts yet in {out_dir}"); return 1
        G = json.load(open(gj)) if os.path.exists(gj) else {}
        complete = set(have) >= set(parts)
        txt = assemble_report(meta, P, G, complete, ROLE_VOCAB); print(txt)
        accepted = bool(G.get("passed")) or bool(G.get("adjudication"))
        if complete:
            np.savez_compressed(os.path.join(a.cache, "teacher_head.npz"), meta=np.array(json.dumps(meta)), gates=np.array(json.dumps(G)),
                                gates_passed=np.bool_(bool(G.get("passed"))), adjudicated=np.bool_(bool(G.get("adjudication"))),
                                usable=np.bool_(accepted), report=np.array(txt), parts=np.array(have))
            open(os.path.join(a.cache, "teacher_head_report.txt"), "w", encoding="utf-8").write(txt)
        return 0 if (complete and accepted) else 1

    if a.adjudicate:
        if not os.path.exists(gj):
            raise SystemExit(f"FATAL: {gj} missing -- nothing to adjudicate")
        G = json.load(open(gj)); G["adjudication"] = dict(note=a.adjudicate, when=time.strftime("%Y-%m-%dT%H:%M:%S"))
        json.dump(G, open(gj, "w"), indent=1); print(f"[head] adjudication recorded: {a.adjudicate}")
    if a.report_only or a.adjudicate:
        return finish()

    import torch
    from teacher.http_scorer import TeacherHTTPScorer
    if not (a.teacher and a.teacher_endpoint):
        raise SystemExit("FATAL: --teacher and --teacher-endpoint are required for a dump")
    todo = [s for s in parts if s not in set(existing_parts(out_dir))]
    if todo and os.path.exists(gj) and not json.load(open(gj)).get("passed", False) and not a.ignore_gates:
        print(f"FATAL: {gj} records a failed gate; not continuing without --ignore-gates", flush=True); finish(); return 1
    scorer = TeacherHTTPScorer(endpoint=a.teacher_endpoint, dtype="bfloat16", tokenizer_name=a.teacher, timeout=a.timeout)
    server_fp = scorer.config.template_fp
    meta["server_template_fp"] = server_fp
    if server_fp != cache_fp:
        json.dump(dict(template_ok=False, passed=False, server_template_fp=server_fp, cache_template_fp=cache_fp), open(gj, "w"), indent=1)
        raise SystemExit(f"FATAL: served teacher template_fp {server_fp} != cache manifest's {cache_fp}")
    tok = None; roles_for = None
    if a.tokenizer:
        import transformers
        from scripts.advantage_by_role import roles_for as _rf
        tok = transformers.AutoTokenizer.from_pretrained(a.tokenizer); roles_for = _rf
    print(f"[head] {len(parts)} parts, {len(parts) - len(todo)} done, {len(todo)} to do; k_wide={a.k_wide} k_cap={a.k_cap} cover={a.cover}; "
          f"template_fp {server_fp} == cache {cache_fp}; roles {'on' if tok else 'off'}", flush=True)
    for s in todo:
        t0 = time.time()
        z = np.load(part_path(ref_dir, s), allow_pickle=True)
        keep = z["traj"] < len(paths)                      # --n may end inside this shard
        ref = {k: z[k][keep] for k in ("traj", "pos", "y", "top_ids", "top_lp")}
        meta["k_prime"] = int(ref["top_ids"].shape[1])
        soft = soft_rows(ref)
        trajs = np.unique(ref["traj"])
        recs = {int(t): torch.load(paths[int(t)], weights_only=False, map_location="cpu") for t in trajs}
        reqs = requests_for(ref, soft, {t: r["input_ids"] for t, r in recs.items()})
        heads_per_traj = scorer.score_head([(ids, pos, ex) for _, ids, pos, ex in reqs], k_wide=a.k_wide, k_cap=a.k_cap, cover=a.cover)
        cols = {f: [] for f in FIELDS}; lp_y_all, cached_all = [], []
        for (t, ids, pos, ex), heads in zip(reqs, heads_per_traj):
            m = (ref["traj"] == t) & soft; rows = np.where(m)[0]
            rr = {k: ref[k][rows] for k in ref}
            packed = pack_heads(heads, rr, rr["top_ids"], a.k_cap)
            for f, v in packed.items():
                cols[f].append(v)
            cols["traj"].append(np.full(len(rows), t, np.int32)); cols["pos"].append(rr["pos"].astype(np.int32)); cols["y"].append(rr["y"].astype(np.int32))
            if roles_for is not None:
                roles = roles_for(ids.tolist(), tok)
                cols["role_id"].append(np.array([ROLE_VOCAB.index(roles[p]) if roles[p] in ROLE_VOCAB else 0 for p in rr["pos"].tolist()], np.int32))
            else:
                cols["role_id"].append(np.full(len(rows), -1, np.int32))
            lp_y_all.append(packed["lp_y"]); cached_all.append(recs[t]["teacher_logp"].float().numpy()[rr["pos"]])
        Z = {f: (np.concatenate(cols[f]) if cols[f] else np.zeros(0)) for f in FIELDS}
        al = align_check(np.concatenate(lp_y_all) if lp_y_all else [], np.concatenate(cached_all) if cached_all else [])
        summ = dict(start=s, n_ref=int(len(ref["pos"])), n_soft=int(soft.sum()), wall_s=time.time() - t0, align=al)
        tmp = part_path(out_dir, s) + ".tmp.npz"
        np.savez_compressed(tmp, **Z, summary=np.array(json.dumps(summ))); os.replace(tmp, part_path(out_dir, s))
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(f"[head] part {s:06d}: {summ['n_soft']} soft of {summ['n_ref']} positions in {summ['wall_s']:.0f}s; alignment max {al['max_abs']:.2e} "
              f"n>=tol {al['n_over_tol']}  [{len(existing_parts(out_dir))}/{len(parts)} parts]", flush=True)
        if s == parts[0] or not os.path.exists(gj):
            G = dict(template_ok=True, align=al, passed=bool(al["passed"]), server_template_fp=server_fp, cache_template_fp=cache_fp)
            json.dump(G, open(gj, "w"), indent=1)
            if not al["passed"]:
                print("FATAL: alignment gate failed on the first part -- stopping (the part is kept)", flush=True); finish(); return 1
        elif not al["passed"] and not a.ignore_gates:
            print(f"FATAL: alignment gate failed on part {s} -- stopping (the part is kept)", flush=True); finish(); return 1
    return finish()


if __name__ == "__main__":
    sys.exit(main())
