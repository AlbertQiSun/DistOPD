#!/usr/bin/env python3
"""One job, three answers: how big K must be, whether the full-KL arm is worth running, and
how far the co-located HF teacher drifts from the cached vLLM one.

    python scripts/fulldist_probe.py --cache precomputed_miro_full \\
        --teacher /path/to/scratch/models/MiroThinker-1.7-mini \\
        --student /path/to/scratch/models/pi_ref_miro \\
        --n 200 --out runs/probe/miro_fulldist.npz

All three need exactly the same machinery -- the teacher's and the student's FULL
distributions at the scored positions of cached trajectories -- so they share one forward
pass instead of three jobs.

  1. K COVERAGE. The per-token dump stores top-K, and K has to be chosen from measurement,
     not from a guess about how peaked an LM is. Reports cumulative probability mass at
     K = 1/8/64/256/512/1024/4096, per role, as MEDIAN, p90 and p99.
     THE CRITERION, fixed before the numbers exist: the smallest K with median >= 0.999 AND
     p99 >= 0.99 on the DECISION roles (answer_close, answer_open, im_end, search_open).
     Decision points are the high-entropy positions and therefore the worst covered; setting
     K from think_body's median would underestimate it badly. p99 is in the criterion because
     a median that passes while p99 sits at 0.95 means 1% of positions are badly truncated --
     and that 1% is plausibly where the model is most undecided, i.e. exactly the interesting
     part.

  2. THE PRE-CHECK, and it is ASYMMETRIC -- read this before acting on it.
     Reports corr(A_t, KL_t), ||p-q||, and the cosine between the two objectives' gradients,
     per role.
         LOW correlation  => the two objectives disagree per position => worth running.
         HIGH correlation => NOT a reason to cancel.
     The two gradients are not collinear in logit space (one is L_y*(e_y - p), the other
     p*(L - R)), and even where they agree position by position, the AGGREGATE update can
     still differ because the sampled one is a REINFORCE estimator whose variance does not
     show up in a per-position correlation. This pre-check can only ever say GO, never NO-GO.

  3. GATE 0. Co-locating the teacher means running it under transformers instead of the vLLM
     server, which is a different backend and a different fingerprint -- the numbers are not
     the same numbers. Compares this HF teacher's logp at the realized tokens against the
     cached `teacher_logp` and reports mean|delta| and p99. Small (~1e-3 nats against
     advantages of order 0.3) means the change is recordable and harmless; large means the
     new arms remain internally consistent but nothing may be compared to the existing cache.

Also: prints the template/backend fingerprint READ FROM THE MANIFEST FILE (not derived from
config -- reading the wrong cache has been a silent failure here before), and times writing
25 small files so the "GPFS metadata overhead is negligible at a few thousand files" line in
the storage design becomes a measurement instead of an assumption.

SIGN CONVENTIONS, stated once and used everywhere below:
    logp = STUDENT log-probs, logq = TEACHER log-probs
    L_v  = logp_v - logq_v                       (student minus teacher)
    R    = sum_v p_v L_v = KL(student || teacher)
    A_y  = logq_y - logp_y = -L_y                (the single-token advantage in opd/loss.py)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd.trajectory import TokenType
from scripts.advantage_by_role import roles_for
from scripts.cross_score_cache import cache_paths

K_GRID = (1, 8, 64, 256, 512, 1024, 4096)
from scripts.role_vocab import DECISION_ROLES, ROLE_VOCAB  # noqa: E402


def print_manifest(cache: str) -> None:
    """The fingerprint as READ FROM THE FILE. Never derived from config: this project has
    had a silent 'which cache did that actually read' failure before."""
    p = os.path.join(cache, "manifest.json")
    if not os.path.exists(p):
        print(f"[probe] WARNING: no manifest at {p} -- provenance unverified")
        return
    with open(p, encoding="utf-8") as f:
        man = json.load(f)
    flat = json.dumps(man)
    for key in ("template_fp", "engine", "dtype", "gpu_name", "vllm_version"):
        i = flat.find(f'"{key}"')
        if i >= 0:
            print(f"[probe] manifest {flat[i:i + 80].split(',')[0]}")


def row_stats(logp: torch.Tensor, logq: torch.Tensor, y: torch.Tensor) -> dict:
    """Everything the design stores, for one chunk of rows. float64 reductions.

    fp32 summation over 151936 terms carries ~1e-5 relative error, which is the same order as
    the KL values being measured, so the accumulator is float64. The output is a handful of
    scalars per row, so this costs nothing.
    """
    p = logp.exp().double()
    q = logq.exp().double()
    lp, lq = logp.double(), logq.double()
    L = lp - lq

    rev_kl = (p * L).sum(-1)                                   # KL(student || teacher)
    fwd_kl = (q * (lq - lp)).sum(-1)                            # KL(teacher || student)
    tv = 0.5 * (p - q).abs().sum(-1)
    bc = (0.5 * (lp + lq)).exp().sum(-1)                        # -> Hellinger, Renyi 0.5
    logm = torch.logaddexp(lp, lq) - math.log(2.0)
    js = 0.5 * (p * (lp - logm)).sum(-1) + 0.5 * (q * (lq - logm)).sum(-1)
    H_p = -(p * lp).sum(-1)
    H_q = -(q * lq).sum(-1)
    p2 = (p * p).sum(-1)
    q2 = (q * q).sum(-1)
    p2L = (p * p * L).sum(-1)
    p2L2 = (p * p * L * L).sum(-1)
    l2 = ((p - q) ** 2).sum(-1).sqrt()

    rows = torch.arange(logp.shape[0], device=logp.device)
    st_lp = lp[rows, y]
    t_lp = lq[rows, y]
    p_y = p[rows, y]
    L_y = L[rows, y]

    # closed-form cosine between the two objectives' gradients wrt the logits
    dot = L_y * (p_y * (L_y - rev_kl) - (p2L - rev_kl * p2))
    n_full = (p2L2 - 2 * rev_kl * p2L + rev_kl ** 2 * p2).clamp_min(0).sqrt()
    n_samp = L_y.abs() * (1 - 2 * p_y + p2).clamp_min(0).sqrt()
    cos = dot / (n_full * n_samp).clamp_min(1e-30)

    return {"rev_kl": rev_kl, "fwd_kl": fwd_kl, "tv": tv, "js": js, "bc": bc,
            "H_p": H_p, "H_q": H_q, "sum_p2": p2, "sum_q2": q2, "l2_pq": l2,
            "st_lp": st_lp, "t_lp": t_lp, "cos": cos,
            "grad_full_norm": n_full, "grad_samp_norm": n_samp,
            "adv": -L_y}


def chunked_logprobs(model, ids, rows, device, chunk):
    """log_softmax at `rows` only, WITHOUT ever materialising [seq_len, vocab].

    `model(...).logits` builds the whole [seq, 151936] tensor: 4.98 GiB in bf16 for a
    16k-token trajectory, and several transformers versions then cast it to fp32, which
    doubles it again with both copies live. That is the exact shape of the teacher OOM
    recorded in the project notes -- a [n_prompt, vocab] tensor materialised in full when only a
    slice of it is ever read.

    Running the decoder ONCE and applying the LM head only to the scored rows keeps the peak
    at [seq, hidden] + [chunk, vocab]. For Qwen, `model.model` already applies the final norm
    and `model.lm_head` is the projection, so head(body(x)) is logits exactly -- asserted in
    tests against the full-logits path rather than assumed.
    """
    with torch.no_grad():
        body, head = getattr(model, "model", None), getattr(model, "lm_head", None)
        inp = ids.unsqueeze(0).to(device)
        if body is None or head is None:
            lg = model(input_ids=inp).logits[0]
            for a in range(0, rows.numel(), chunk):
                yield torch.log_softmax(lg[rows[a:a + chunk].to(device)].float(), dim=-1)
            return
        h = body(input_ids=inp).last_hidden_state[0]
        # DEVICE-MAP SAFE. Under device_map="auto" (the 30B and 32B teachers on two cards) `h`
        # lands on the LAST decoder layer's card and lm_head may sit on yet another, while
        # `device` is wherever the FIRST parameter lives. Indexing h with rows on `device`
        # raised "indices should be on the same device as the indexed tensor" and killed
        # flip_direction on every line. Index on h's device, project on the head's.
        head_dev = head.weight.device
        for a in range(0, rows.numel(), chunk):
            sel = h[rows[a:a + chunk].to(h.device)].to(head_dev)
            yield torch.log_softmax(head(sel).float(), dim=-1)


def topk_mass(logits_lp: torch.Tensor, kmax: int) -> torch.Tensor:
    """[rows, len(K_GRID)] cumulative probability mass at each K in the grid."""
    vals = torch.topk(logits_lp, k=min(kmax, logits_lp.shape[-1]), dim=-1).values
    csum = vals.double().exp().cumsum(-1)
    idx = torch.tensor([min(k, csum.shape[-1]) - 1 for k in K_GRID], device=csum.device)
    return csum.index_select(-1, idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--row-chunk", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="runs/probe/fulldist_probe.npz")
    args = ap.parse_args()

    print_manifest(args.cache)
    import transformers

    dt = getattr(torch, args.dtype)
    ndev = torch.cuda.device_count()
    d_st = "cuda:0" if ndev else "cpu"
    d_te = f"cuda:{1 if ndev > 1 else 0}" if ndev else "cpu"
    print(f"[probe] {ndev} visible GPUs -- student on {d_st}, teacher on {d_te}")
    # device_map, NOT .to(device). `from_pretrained(...).to(dev)` materialises the WHOLE
    # model in CPU RAM first and then copies it, so the peak is ~2x the model: the 30B
    # teacher was SIGKILLed at shard 246/531 with 256 GB requested. device_map places each
    # shard on the target device as it is read, so CPU peak is one shard.
    student = transformers.AutoModelForCausalLM.from_pretrained(
        args.student, dtype=dt, device_map={"": d_st}, low_cpu_mem_usage=True).eval()
    tok = transformers.AutoTokenizer.from_pretrained(args.student)
    teacher = transformers.AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=dt, device_map={"": d_te}, low_cpu_mem_usage=True).eval()

    paths = cache_paths(args.cache)[:args.n]
    print(f"[probe] {len(paths)} trajectories")
    cols: dict = {}
    role_ids, gate0 = [], []
    masses = []

    for ti, path in enumerate(paths):
        rec = torch.load(path, weights_only=False, map_location="cpu")
        ids = rec["input_ids"].long()
        tt = rec["token_type"].long()
        pos = torch.nonzero(tt == int(TokenType.ACTION), as_tuple=False).squeeze(-1)
        pos = pos[pos > 0]
        if pos.numel() == 0:
            continue
        roles = roles_for(ids.tolist(), tok)
        cached_t = rec["teacher_logp"].float()

        rows = (pos - 1)
        y = ids[pos]
        gen_s = chunked_logprobs(student, ids, rows, d_st, args.row_chunk)
        gen_t = chunked_logprobs(teacher, ids, rows, d_te, args.row_chunk)
        for ci, (lp, lq_dev) in enumerate(zip(gen_s, gen_t)):
            i0 = ci * args.row_chunk
            yy = y[i0:i0 + args.row_chunk].to(d_st)
            lq = lq_dev.to(d_st)
            st = row_stats(lp, lq, yy)
            for k, v in st.items():
                cols.setdefault(k, []).append(v.cpu().numpy())
            masses.append(torch.stack([topk_mass(lp, max(K_GRID)),
                                       topk_mass(lq, max(K_GRID))], 1).cpu().numpy())
            del lp, lq
        for pp in pos.tolist():
            role_ids.append(ROLE_VOCAB.index(roles[pp]) if roles[pp] in ROLE_VOCAB else 0)
        gate0.append(cached_t[pos].numpy())
        if (ti + 1) % 25 == 0:
            print(f"[probe] {ti + 1} trajectories", flush=True)

    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    arrays["role_id"] = np.array(role_ids, dtype=np.int8)
    arrays["mass"] = np.concatenate(masses)                     # [N, 2, len(K_GRID)]
    arrays["cached_t_lp"] = np.concatenate(gate0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, K_GRID=np.array(K_GRID), **arrays)
    print(f"[probe] wrote {args.out}  ({arrays['rev_kl'].shape[0]} positions)")

    report(arrays)
    smallfile_timing(os.path.dirname(os.path.abspath(args.out)))
    return 0


def report(a: dict) -> None:
    role = a["role_id"]
    names = {i: n for i, n in enumerate(ROLE_VOCAB)}

    print("")
    print("===== 1. K COVERAGE (cumulative mass; criterion: median >= 0.999 AND p99 >= 0.99")
    print("        on the DECISION roles, which are the high-entropy, worst-covered ones)")
    for side, si in (("teacher", 1), ("student", 0)):
        print(f"\n  -- {side}")
        print("    " + "role".rjust(13) + "".join(f"{k:>9}" for k in K_GRID))
        for r in sorted(set(role.tolist())):
            m = role == r
            if m.sum() < 30:
                continue
            med = np.median(a["mass"][m, si, :], axis=0)
            p99 = np.percentile(a["mass"][m, si, :], 1, axis=0)   # lower tail = worst 1%
            mark = " *" if names[r] in DECISION_ROLES else "  "
            print(f"  {mark}{names[r]:>11}" + "".join(f"{v:>9.4f}" for v in med))
            print(f"    {'  p99(worst)':>13}" + "".join(f"{v:>9.4f}" for v in p99))

    print("")
    print("===== 2. PRE-CHECK -- LOW corr => worth running; HIGH corr is NOT a no-go")
    print(f"  {'role':>13} {'n':>8} {'corr(A,KL)':>11} {'cos(grad)':>11} "
          f"{'||p-q||2':>10} {'KL(S||T)':>10}")
    for r in sorted(set(role.tolist())):
        m = role == r
        if m.sum() < 30:
            continue
        A, K = a["adv"][m], a["rev_kl"][m]
        c = np.corrcoef(A, K)[0, 1] if A.std() > 0 and K.std() > 0 else float("nan")
        print(f"  {names[r]:>13} {int(m.sum()):>8} {c:>11.3f} "
              f"{np.median(a['cos'][m]):>11.3f} {np.median(a['l2_pq'][m]):>10.4f} "
              f"{np.median(K):>10.4f}")

    print("")
    print("===== 3. GATE 0 -- co-located HF teacher vs the cached vLLM teacher_logp")
    d = np.abs(a["t_lp"] - a["cached_t_lp"])
    print(f"  n={d.size}  mean|delta|={d.mean():.3e}  p50={np.median(d):.3e}  "
          f"p99={np.percentile(d, 99):.3e}  max={d.max():.3e}")
    print("  Compare against advantages of order 0.3 nats: 1e-3 is recordable and harmless,")
    print("  1e-1 means the new arms are internally consistent but not comparable to the")
    print("  existing cache-trained arms.")


def smallfile_timing(d: str) -> None:
    """Turns 'GPFS metadata overhead is negligible at a few thousand files' into a number."""
    t0 = time.time()
    blob = np.zeros(119_296, dtype=np.float32)          # one step's worth of one column
    for i in range(25):
        np.save(os.path.join(d, f".mdtest_{i}.npy"), blob)
    dt = time.time() - t0
    for i in range(25):
        os.remove(os.path.join(d, f".mdtest_{i}.npy"))
    print("")
    print(f"===== 4. small-file write: 25 columns x 477 KB in {dt:.2f} s "
          f"({25 / max(dt, 1e-9):.0f} files/s)")
    print(f"  A 150-step run writes ~25 columns/step; at this rate the per-step metadata cost")
    print(f"  is {dt:.2f} s, i.e. {dt / 20 * 100:.1f}% of a 20 s step (20 s is a PROJECTION).")


if __name__ == "__main__":
    raise SystemExit(main())
