#!/usr/bin/env python3
"""Per-token OPD advantage by token role, and by whether the trajectory ever found gold.

    python scripts/advantage_by_role.py --n 200
        --ckpt $OPD_STUDENT_CKPT --label piref
        --ckpt runs/ckpt_offline_p1 --label offline125

THE LINK THIS SUPPLIES. The chain is (A) what the teacher prefers, (B) which way the
gradient therefore points, (C) what the student does. B is the most direct of the three:
the advantage IS the per-token weight on log pi_theta, so its sign on "</answer>" is not
an inference about training, it is training's direction on that token, by definition.

WHY NO RETRAINING. advantage = clamp(teacher_logp - student_logp, +-tau)
(opd/loss.compute_advantage); teacher_logp is already in the cache, so only a forward pass
was missing. The trainer kept adv_mean/std/absmax/clip_rate and the boundary buckets while
the per-token tensor lived in StepOutput and was dropped -- the under-persist
shape.

AND THIS SCRIPT REPEATED IT. Its first version printed the role table and persisted only
those aggregates, so the obvious next question -- are the extreme negatives concentrated
on trajectories that never retrieved gold, where "keep searching" is CORRECT advice rather
than a bias? -- required the whole GPU pass again. Per-token rows now go to disk (role,
adv, student_logp, teacher_logp) next to the trajectory's own union_hit / n_searches /
terminated, so any later regrouping is free. That question decides which paper this is:
extremes only where gold was never found means the teacher is right and the finding is
gradient pollution from unsolvable questions; extremes on BOTH sides means a threshold
bias that fires even when the evidence is already in hand.

SOURCES. A precompute dir (manifest.json + teacher_logp per record) yields advantages. A
plain directory of *.pt records -- the SFT stage's TEACHER rollouts -- has no teacher_logp,
so only the scored model's own logp is reported. That mode is the control for whether 0.72
is a property of the teacher or of the teacher evaluated on someone else's states: point
--ckpt at the teacher and --source at its own trajectories.

EXACTNESS PER ARM. For OFFLINE checkpoints the cache IS the training data, so a recomputed
advantage is the advantage that checkpoint saw. For ONLINE it is a counterfactual on the
same states -- online trained on its own rollouts, which were never persisted -- and is
labelled as such.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import configs.template as T  # noqa: E402
from opd.diagnostics import retrieval_quality  # noqa: E402
from opd.loss import compute_advantage  # noqa: E402
from opd.masking import build_score_mask  # noqa: E402

ROLE_ORDER = ["answer_close", "im_end", "answer_open", "answer_body",
              "search_open", "search_close", "query_body",
              "think_open", "think_close", "think_body", "other"]
SPLIT_ROLES = ["answer_close", "answer_body", "search_open", "query_body"]


def token_spans(ids, tok):
    pieces = tok.batch_decode([[i] for i in ids], skip_special_tokens=False)
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return spans, "".join(pieces)


def _tag_role(tag):
    return {T.ANSWER_OPEN: "answer_open", T.ANSWER_CLOSE: "answer_close",
            T.SEARCH_OPEN: "search_open", T.SEARCH_CLOSE: "search_close",
            T.THINK_OPEN: "think_open", T.THINK_CLOSE: "think_close"}[tag]


def _find_all(text, needle):
    i, out = 0, []
    while True:
        j = text.find(needle, i)
        if j < 0:
            return out
        out.append(j)
        i = j + len(needle)


def region_map(text):
    regions = []
    for open_tag, close_tag, body_role in (
            (T.ANSWER_OPEN, T.ANSWER_CLOSE, "answer_body"),
            (T.SEARCH_OPEN, T.SEARCH_CLOSE, "query_body"),
            (T.THINK_OPEN, T.THINK_CLOSE, "think_body")):
        i = 0
        while True:
            a = text.find(open_tag, i)
            if a < 0:
                break
            b = text.find(close_tag, a + len(open_tag))
            regions.append((a, a + len(open_tag), _tag_role(open_tag)))
            if b < 0:
                regions.append((a + len(open_tag), len(text), body_role))
                break
            regions.append((a + len(open_tag), b, body_role))
            regions.append((b, b + len(close_tag), _tag_role(close_tag)))
            i = b + len(close_tag)
    for m in _find_all(text, T.IM_END):
        regions.append((m, m + len(T.IM_END), "im_end"))
    return regions


def roles_for(ids, tok):
    """Roles in TEXT space -- matching tag ids is the mistake count_span_nesting made twice."""
    spans, text = token_spans(ids, tok)
    regions = region_map(text)
    roles = ["other"] * len(ids)
    for k, (a, b) in enumerate(spans):
        mid = (a + b) / 2.0
        for (ra, rb, role) in regions:
            if ra <= mid < rb:
                roles[k] = role
                break
    return roles


def load_trajs(src, stride, n):
    man_path = os.path.join(src, "manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path, encoding="utf-8"))
        paths = [os.path.join(src, p) for p in man["paths"]]
    else:
        paths = sorted(glob.glob(os.path.join(src, "*.pt")))
    out = []
    for p in paths[::stride][:n]:
        rec = torch.load(p, weights_only=False, map_location="cpu")
        meta = rec.get("meta") or {}
        rq = retrieval_quality(meta) if meta.get("turns") is not None else {}
        tlp = rec.get("teacher_logp")
        ns = rec.get("num_search_calls", meta.get("num_search_calls"))
        out.append({
            "path": os.path.basename(p),
            "ids": rec["input_ids"].long(),
            "tt": rec["token_type"].long(),
            "tlp": None if tlp is None else tlp.float(),
            "union_hit": rq.get("union_hit"),
            "per_turn_hit_rate": rq.get("per_turn_hit_rate"),
            "n_searches": int(ns) if ns is not None else None,
            "terminated": rec.get("terminated") or meta.get("terminated"),
        })
    return out


def summarise(vals):
    if not vals:
        return None
    return {"n": len(vals), "mean": st.mean(vals), "median": st.median(vals),
            "frac_neg": st.mean(1.0 if x < 0 else 0.0 for x in vals),
            "frac_lt_m1": st.mean(1.0 if x < -1.0 else 0.0 for x in vals),
            "sum_neg": sum(x for x in vals if x < 0)}


def weight_gib(ckpt, dtype):
    """GiB the weights will occupy once loaded, from the checkpoint's own index.

    A pre-flight rather than a guess: loading MiroThinker (30.5B) at the student's
    float32 died after claiming 78 GiB, with an allocator message that named neither the
    model nor the dtype. The student wants float32 params (bf16 autocast) because that is
    what training used; a 30B teacher cannot be float32 on this card at all, and should
    not be -- the cache's teacher_logp came from vLLM in bfloat16.
    """
    per = {"float32": 4, "bfloat16": 2, "float16": 2}[dtype]
    cfg_path = os.path.join(ckpt, "config.json")
    stored_name = "bfloat16"
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        stored_name = cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16"
    stored_per = {"float32": 4, "bfloat16": 2, "float16": 2}.get(stored_name, 2)
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        stored_bytes = json.load(open(idx, encoding="utf-8"))["metadata"]["total_size"]
    else:
        stored_bytes = sum(os.path.getsize(os.path.join(ckpt, f))
                           for f in os.listdir(ckpt) if f.endswith(".safetensors"))
    return (stored_bytes / stored_per) * per / 2 ** 30


def preflight(ckpt, dtype):
    need = weight_gib(ckpt, dtype)
    have = (torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            if torch.cuda.is_available() else float("inf"))
    print(f"[adv] {os.path.basename(ckpt)}: weights {need:.1f} GiB as {dtype}, "
          f"card {have:.1f} GiB")
    if need > 0.9 * have:
        raise SystemExit(
            f"FATAL: {ckpt} needs about {need:.1f} GiB as {dtype} but the card has "
            f"{have:.1f} GiB. float32 is the STUDENT's dtype (it matches training: fp32 "
            f"params, bf16 autocast). A 30B teacher must be scored in bfloat16 -- which "
            f"is also the dtype its cached teacher_logp was produced in. Pass "
            f"--dtype bfloat16 for this checkpoint.")


def fmt(x):
    return f"{x:>9.4f}" if isinstance(x, float) else f"{'-':>9}"


def pctf(d, key):
    return f"{100 * d[key]:>6.1f}%" if d else f"{'-':>7}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--dtype", action="append", default=None,
                    help="per --ckpt, in order. float32 (default) matches the STUDENT's "
                         "training numerics; a 30B teacher needs bfloat16, which is also "
                         "how its cached teacher_logp was produced")
    ap.add_argument("--source", default="precomputed",
                    help="precompute dir (advantages) or a dir of *.pt records (logp only)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stride", type=int, default=37)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--config", default="configs/opd_qwen3.yaml")
    ap.add_argument("--out", default="runs/advantage_by_role.json")
    ap.add_argument("--rows-out", default="runs/advantage_rows.jsonl",
                    help="PER-TOKEN rows, so any later regrouping is free")
    args = ap.parse_args()
    assert len(args.ckpt) == len(args.label), "one --label per --ckpt"
    dtypes = args.dtype or ["float32"] * len(args.ckpt)
    assert len(dtypes) == len(args.ckpt), "one --dtype per --ckpt (or none at all)"

    tau = args.tau
    if tau is None:
        from opd.config import load_config
        tau = load_config(args.config)["opd"]["tau"]

    trajs = load_trajs(args.source, args.stride, args.n)
    has_teacher = bool(trajs) and trajs[0]["tlp"] is not None
    n_hit = sum(1 for t in trajs if t["union_hit"] is True)
    n_miss = sum(1 for t in trajs if t["union_hit"] is False)
    mode = "present" if has_teacher else "ABSENT (logp-only mode)"
    print(f"[adv] source={args.source} tau={tau} n={len(trajs)}  teacher_logp {mode}")
    print(f"[adv] union_hit True/False/unknown = {n_hit}/{n_miss}/"
          f"{len(trajs) - n_hit - n_miss}")

    from transformers import AutoTokenizer
    from teacher.scorer import HFScorer

    rows_f = open(args.rows_out, "w", encoding="utf-8")
    out = {}
    for ckpt, label, dt in zip(args.ckpt, args.label, dtypes):
        print("")
        print(f"[adv] === {label}: {ckpt} ({dt}) ===", flush=True)
        preflight(ckpt, dt)
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = HFScorer.from_pretrained(ckpt, dtype=dt, autocast_dtype="bfloat16",
                                         requires_grad=False, device="cuda:0")
        acc = {r: {"adv": [], "slp": [], "tlp": [], "hit": [], "miss": []}
               for r in ROLE_ORDER}
        for k, tr in enumerate(trajs):
            if k and k % 50 == 0:
                print(f"    ... {k}/{len(trajs)}", flush=True)
            slp = model.logprobs(tr["ids"]).float().cpu()
            mask = build_score_mask(tr["tt"])
            adv = (compute_advantage(tr["tlp"], slp, mask, tau)
                   if tr["tlp"] is not None else None)
            roles = roles_for(tr["ids"].tolist(), tok)
            keep = {"role": [], "pos": [], "slp": [], "adv": [], "tlp": []}
            for t in range(len(roles)):
                if not bool(mask[t]):
                    continue
                sv = float(slp[t])
                if sv != sv:
                    continue
                r = roles[t]
                keep["role"].append(r)
                keep["pos"].append(t)
                keep["slp"].append(round(sv, 5))
                acc[r]["slp"].append(sv)
                if adv is not None:
                    av = float(adv[t])
                    tv = float(tr["tlp"][t])
                    keep["adv"].append(round(av, 5))
                    keep["tlp"].append(round(tv, 5))
                    acc[r]["adv"].append(av)
                    acc[r]["tlp"].append(tv)
                    if tr["union_hit"] is True:
                        acc[r]["hit"].append(av)
                    elif tr["union_hit"] is False:
                        acc[r]["miss"].append(av)
            row = {"model": label, "path": tr["path"], "union_hit": tr["union_hit"],
                   "per_turn_hit_rate": tr["per_turn_hit_rate"],
                   "n_searches": tr["n_searches"], "terminated": tr["terminated"]}
            row.update(keep)
            rows_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        del model
        torch.cuda.empty_cache()

        total = sum(len(acc[r]["slp"]) for r in ROLE_ORDER)
        print(f"{'role':<14} {'n':>7} {'share':>7} {'mean A':>9} {'median A':>9} "
              f"{'%A<0':>7} {'%A<-1':>7} {'logp_stu':>9} {'logp_tea':>9}")
        rows = {}
        for r in ROLE_ORDER:
            a = acc[r]
            if not a["slp"]:
                continue
            s = summarise(a["adv"])
            rows[r] = {"n": len(a["slp"]), "share": len(a["slp"]) / max(1, total),
                       "student_logp_mean": st.mean(a["slp"]),
                       "teacher_logp_mean": st.mean(a["tlp"]) if a["tlp"] else None,
                       "adv": s, "adv_hit": summarise(a["hit"]),
                       "adv_miss": summarise(a["miss"])}
            print(f"{r:<14} {rows[r]['n']:>7} {100 * rows[r]['share']:>6.1f}% "
                  f"{fmt(s['mean'] if s else None)} {fmt(s['median'] if s else None)} "
                  f"{pctf(s, 'frac_neg')} {pctf(s, 'frac_lt_m1')} "
                  f"{rows[r]['student_logp_mean']:>9.4f} "
                  f"{fmt(rows[r]['teacher_logp_mean'])}")

        if has_teacher and n_hit and n_miss:
            print("")
            print("  -- split by trajectory union_hit (did this rollout EVER retrieve gold?) --")
            print(f"  {'role':<14} {'group':<5} {'n':>7} {'mean A':>9} {'median A':>9} "
                  f"{'%A<-1':>7} {'share of role neg A':>21}")
            for r in SPLIT_ROLES:
                if r not in rows:
                    continue
                pair = [rows[r]["adv_hit"], rows[r]["adv_miss"]]
                tot_neg = sum(v["sum_neg"] for v in pair if v)
                for gname, v in (("hit", rows[r]["adv_hit"]), ("miss", rows[r]["adv_miss"])):
                    if not v:
                        continue
                    frac = v["sum_neg"] / tot_neg if tot_neg else float("nan")
                    print(f"  {r:<14} {gname:<5} {v['n']:>7} {v['mean']:>9.4f} "
                          f"{v['median']:>9.4f} {100 * v['frac_lt_m1']:>6.1f}% "
                          f"{100 * frac:>20.1f}%")
        out[label] = {"ckpt": ckpt, "dtype": dt, "roles": rows,
                      "n_scored_tokens": total}

    rows_f.close()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"tau": tau, "source": args.source, "n_trajectories": len(trajs),
                   "models": out}, f, indent=1)
    print("")
    print(f"[adv] aggregates -> {args.out}")
    print(f"[adv] PER-TOKEN rows -> {args.rows_out}")
    print("[adv] regroup by union_hit / n_searches / terminated / role / position from")
    print("      that file, without touching a GPU again.")


if __name__ == "__main__":
    main()
