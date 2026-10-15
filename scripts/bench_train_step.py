#!/usr/bin/env python3
"""How long does one training step actually take, and does it fit the card? ONE GPU, no services.

    python scripts/bench_train_step.py --config configs/opd_qwen3.yaml --repeats 3
    python scripts/bench_train_step.py --shapes min,p50,p90        # skip the shape that OOMs

WHY THIS EXISTS. The number that decides the schedule -- seconds per trajectory for the student
forward/backward PLUS the diagnostic pass -- was blocked behind a 2-GPU job needing a teacher, a
retriever and a precomputed cache, on a partition where two co-located H20s are the scarcest
thing there is.

None of that is required to measure it. The teacher's logprobs are an INPUT to the loss: their
VALUES change the gradient, not the cost. Cost is driven by trajectory SHAPE -- length sets the
forward, scored-token count sets the loss and the diagnostic -- and both are known from the real
cache. So: real student, real optimizer, real opd_train_step, real DiagnosticRunner, synthetic
trajectories at the measured shapes, one card, no network.

WHAT IT ANSWERS
  1. seconds per trajectory for fwd/bwd, per length bucket.
  2. seconds for the diagnostic pass, which until recently sat outside every timer.
  3. PEAK GPU MEMORY, split into the FIXED part (weights, gradients, optimizer state, the
     autocast weight-cast cache) and the per-shape part. The first measured run overran an
     80.1 GiB arithmetic estimate by ~13 GiB, so the split matters: when a total is wrong the
     first question is which half was wrong.

WHAT IT DOES NOT ANSWER. Rollout cost (sampling + retrieval + teacher scoring). That needs the
services and comes from the precompute job. Report the two separately and ADD them; blending a
synthetic term into a measured one produces a figure nobody can defend.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from opd.config import load_config, require_input_asset
from opd.diagnostics import DiagnosticRunner
from opd.step import opd_train_step
from opd.trajectory import TokenType, Trajectory

# The measured cache (2026-08-14): length percentiles and their scored-token counts. Named
# buckets rather than a mean, because the tail is where the walltime and the memory risk both
# live -- p90 to max is a 5x jump in length.
SHAPES = [("min", 708, 77), ("p50", 2241, 270), ("p90", 2674, 487), ("max", 13750, 1694)]

GIB = 2 ** 30


def synth(n_tokens: int, n_scored: int, vocab: int, seed: int) -> Trajectory:
    """A trajectory with the right SHAPE. Content is irrelevant to cost: the forward sees
    n_tokens positions and the loss/diagnostic see n_scored of them, whatever the ids are."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n_tokens,), generator=g)
    tt = torch.full((n_tokens,), int(TokenType.PROMPT), dtype=torch.long)
    # one contiguous ACTION span, never touching position 0 (build_score_mask forbids it)
    start = max(1, (n_tokens - n_scored) // 2)
    tt[start:start + n_scored] = int(TokenType.ACTION)
    return Trajectory(ids, tt)


def search_length_ceiling(fits, lo: int, hi: int, tol: int = 256) -> dict:
    """Locate the longest trajectory that fits, or report that there is none IN RANGE.

    `fits(n) -> bool` runs one real trajectory of n tokens.

    WHY THIS IS A SEPARATE FUNCTION, AND WHY IT PROBES `hi` FIRST. The previous version
    bisected [lo, hi] unconditionally and returned the last value that fit. When NOTHING in the
    range OOMs -- which is what actually happens on both cards -- `lo` simply climbs until
    `hi - lo <= tol` and the loop returns `hi - tol`ish. It then printed

        --> longest trajectory that fits on this card: ~16169 tokens

    which is not a property of the card at all: it is `hi` minus the stopping tolerance. The
    A100 and the H20 have different memory budgets and reported the SAME 16169, which is the
    tell -- two different cards cannot share a memory-determined limit to the token.

    So: probe `hi` first. If it fits there is no ceiling to bisect for, and the honest result
    is "> hi", not "= hi - tol". `located` carries which of the two happened, because the two
    are different claims and the number alone cannot distinguish them.
    """
    probes: list[dict] = []

    def probe(n: int) -> bool:
        ok = fits(n)
        probes.append({"tokens": n, "fits": ok})
        return ok

    if probe(hi):
        return {"located": False, "largest_that_fits": hi, "smallest_that_ooms": None,
                "search_upper_bound": hi, "probes": probes}

    best = lo
    while hi - lo > tol:
        mid = (lo + hi) // 2
        if probe(mid):
            best, lo = mid, mid
        else:
            hi = mid
    return {"located": True, "largest_that_fits": best, "smallest_that_ooms": hi,
            "search_upper_bound": probes[0]["tokens"], "probes": probes}


def describe(name, xs, unit="s"):
    if not xs:
        print(f"  {name:28} (none)")
        return
    xs = sorted(xs)
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]  # noqa: E731
    print(f"  {name:28} n={len(xs):>3}  mean={statistics.fmean(xs):.3f}{unit}  "
          f"p50={p(.5):.3f}  p90={p(.9):.3f}  max={xs[-1]:.3f}")


def bench_one(name, n_tok, n_scored, *, vocab, student, opt, ref, diag, tau, warmup,
              repeats, dev):
    """One shape, timed. Raises torch.OutOfMemoryError for the caller to record."""
    traj = synth(n_tok, n_scored, vocab, seed=abs(hash(name)) % 10_000)
    tlp = torch.randn(n_tok, dtype=torch.float32)     # values do not affect cost, only shape
    train_s, diag_s, ref_s = [], [], []
    torch.cuda.reset_peak_memory_stats()

    for i in range(warmup + repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = opd_train_step(traj, tlp, student, tau=tau)
        out.loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)

        # THE pi_ref TRANSFER IS A PER-STEP COST, NOT A PER-TRAJECTORY ONE. OPDTrainer.step
        # pages the reference in ONCE and then loops over all global_batch trajectories, so
        # charging the full 15 GiB round trip to every trajectory -- which the first version of
        # this bench did -- overstated the projection by ~3.5x. Timed separately: the transfer
        # is divided by global_batch downstream, the compute is not.
        t_ref0 = time.perf_counter()
        ref.to(dev)
        torch.cuda.synchronize()
        t_ref1 = time.perf_counter()
        diag.run(0, "offline", traj, out, student, staleness=0, resync_every=1)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        ref.to("cpu")
        torch.cuda.synchronize()
        t_ref2 = time.perf_counter()
        ref_s.append((t_ref1 - t_ref0) + (t_ref2 - t2))

        del out
        if i >= warmup:
            train_s.append(t1 - t0)
            diag_s.append(t2 - t_ref1)          # COMPUTE only; the transfer is ref_s
        else:
            ref_s.pop()

    return {"tokens": n_tok, "scored": n_scored, "train_seconds": train_s,
            "diag_seconds": diag_s, "ref_transfer_seconds": ref_s,
            "peak_gib": torch.cuda.max_memory_allocated() / GIB}


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-trajectory training cost, one GPU, no services.")
    ap.add_argument("--config", default="configs/opd_qwen3.yaml")
    ap.add_argument("--repeats", type=int, default=3, help="timed passes per shape")
    ap.add_argument("--warmup", type=int, default=1, help="untimed passes first (alloc/cudnn)")
    ap.add_argument("--shapes", default="", help="comma list to restrict, e.g. min,p50,p90")
    ap.add_argument("--project", type=int, default=38400, help="trajectories/arm to project to")
    ap.add_argument("--find-max-length", action="store_true",
                    help="bisect the longest trajectory that fits, after the fixed shapes")
    ap.add_argument("--memory-profile", type=int, default=0, metavar="TOKENS",
                    help="snapshot allocated memory at each phase of ONE trajectory")
    ap.add_argument("--weight-by", default="", metavar="DIR",
                    help="project using the REAL length distribution in DIR/traj_*.pt "
                         "instead of the p50 shape alone")
    ap.add_argument("--out", default="runs/bench_train_step.json")
    args = ap.parse_args()

    # Vanilla IS the real-run configuration for the student: the batch-invariant overrides
    # are inference-only (aten::linear_backward has no CUDA kernel -- measured 2026-08-18,
    # the run that retired this bench's --pin-batch-invariant flag), so both arms train
    # vanilla and the online arm brackets out of the overrides per step.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # same pin as train.py
    print("[bench] kernel environment: vanilla torch + CUBLAS_WORKSPACE_CONFIG=:4096:8 "
          "(the real-run student configuration)")

    from teacher.backend import runtime_fingerprint
    from teacher.scorer import HFScorer
    from opd.trainer import snapshot_ref

    cfg = load_config(args.config)
    rt = runtime_fingerprint()
    print(f"[bench] {rt.get('gpu_name')} {rt.get('gpu_sm')}  torch={rt.get('torch_version')}")
    if not torch.cuda.is_available():
        print("[bench] FATAL: no usable CUDA device -- this measures GPU cost", file=sys.stderr)
        return 1
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    shapes = [s for s in SHAPES if not args.shapes or s[0] in args.shapes.split(",")]
    results: dict = {}

    def snapshot(label):
        torch.cuda.synchronize()
        cur = torch.cuda.memory_allocated() / GIB
        print(f"[bench] {label:38} {cur:6.1f} GiB allocated / {total:.1f}")
        return cur

    # THE REAL STUDENT, in the configuration the real run uses: fp32 master weights, bf16
    # autocast, gradient checkpointing. Anything else measures a setup that will not be run.
    # Checked BEFORE loading, because transformers reads a non-existent path as a Hub repo id
    # and raises HFValidationError -- which certify gate 6 then reported under its configured
    # meaning, "the student does not fit this card". Nothing had been measured.
    require_input_asset(cfg["model"]["student"], "the student checkpoint (model.student)")
    print(f"[bench] loading student {cfg['model']['student']}")
    student = HFScorer.from_pretrained(
        cfg["model"]["student"],
        dtype=cfg["model"].get("student_param_dtype", "float32"),
        autocast_dtype=cfg["model"].get("student_autocast_dtype", "bfloat16"),
        requires_grad=True, device="cuda:0")
    vocab = int(student.lm.config.vocab_size)
    DEV = next(student.lm.parameters()).device   # not a hardcoded cuda:0
    results["fixed_weights_gib"] = snapshot("fp32 weights")

    from opd.trainer import OPDConfig, build_adamw

    opt = build_adamw(student.lm.parameters(), OPDConfig(
        lr=cfg["optim"]["lr"], adam_beta1=cfg["optim"]["adam_beta1"],
        adam_beta2=cfg["optim"]["adam_beta2"],
        weight_decay=cfg["optim"].get("weight_decay", 0.0)))
    print(f"[bench] optimizer: AdamW (torch default implementation; fused was tried for "
          f"memory and reverted -- see opd.trainer.build_adamw)")

    # pi_ref: deepcopy then park on the HOST. The COPY briefly doubles the weights on the
    # card, which is itself worth seeing -- a transient peak still has to fit.
    print("[bench] snapshotting pi_ref (deepcopy on device, then parked on host)")
    ref = snapshot_ref(student)
    torch.cuda.empty_cache()
    results["fixed_after_ref_gib"] = snapshot("after pi_ref parked on host")
    results["peak_during_snapshot_gib"] = torch.cuda.max_memory_allocated() / GIB
    print(f"[bench] peak DURING the pi_ref deepcopy: "
          f"{results['peak_during_snapshot_gib']:.1f} GiB")
    diag = DiagnosticRunner(ref_scorer=ref, tau=cfg["opd"]["tau"], embedder=None)

    for name, n_tok, n_scored in shapes:
        print(f"\n=== {name}: {n_tok} tokens, {n_scored} scored ===")
        try:
            r = bench_one(name, n_tok, n_scored, vocab=vocab, student=student, opt=opt,
                          ref=ref, diag=diag, tau=cfg["opd"]["tau"],
                          warmup=args.warmup, repeats=args.repeats, dev=DEV)
        except torch.OutOfMemoryError as e:
            # AN OOM ON ONE SHAPE MUST NOT DISCARD THE OTHERS. The longest shape is the one
            # expected to fail, and the shorter ones are what the schedule actually needs.
            torch.cuda.empty_cache()
            results[name] = {"tokens": n_tok, "scored": n_scored,
                             "oom": str(e).splitlines()[0][:200]}
            print(f"  OOM: {str(e).splitlines()[0][:120]}")
            print("  (continuing -- the shorter shapes still answer the schedule question)")
            continue
        results[name] = r
        describe("student fwd+bwd+step", r["train_seconds"])
        describe("diagnostic COMPUTE", r["diag_seconds"])
        describe("pi_ref transfer (PER STEP)", r["ref_transfer_seconds"])
        print(f"  {'PEAK GPU MEMORY':28} {r['peak_gib']:.1f} / {total:.1f} GiB")

    if args.memory_profile:
        # WHERE DOES 7.2 MB/TOKEN COME FROM? Above 2674 tokens the peak grows at that rate,
        # while the accounted per-token terms (logits bf16 304 KB + its gradient 304 KB +
        # checkpointed layer boundaries 184 KB) come to ~0.8 MB. A 9x gap is not "expensive",
        # it is a term nobody has identified -- and it is what sets the 5458-token ceiling
        # against a 16384 max_len. Snapshot each phase rather than reasoning about it.
        n = args.memory_profile
        n_scored = max(1, int(n * 0.12))
        print(f"\n=== memory profile: ONE trajectory of {n} tokens, {n_scored} scored ===")
        traj = synth(n, n_scored, vocab, seed=5)
        tlp = torch.randn(n, dtype=torch.float32)
        prof: dict = {}

        def mark(label):
            torch.cuda.synchronize()
            cur = torch.cuda.memory_allocated() / GIB
            pk = torch.cuda.max_memory_allocated() / GIB
            prof[label] = {"allocated_gib": cur, "peak_gib": pk}
            print(f"  {label:34} allocated {cur:6.2f}   peak-so-far {pk:6.2f} GiB")

        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mark("baseline (weights + Adam state)")

        # the model forward alone, no logprob machinery
        with torch.no_grad(), student._autocast(DEV):
            _lg = student.lm(traj.input_ids.unsqueeze(0).to(DEV)).logits
        mark(f"logits {tuple(_lg.shape)} {_lg.dtype} (no_grad)")
        del _lg
        torch.cuda.empty_cache()
        mark("after freeing them")

        out = opd_train_step(traj, tlp, student, tau=cfg["opd"]["tau"])
        mark("after forward + loss (graph held)")
        out.loss.backward()
        mark("after backward")
        opt.step()
        mark("after optimizer step")
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        mark("after zero_grad(set_to_none)")
        ref.to(DEV)
        mark("pi_ref paged in")
        diag.run(0, "offline", traj, out, student, staleness=0, resync_every=1)
        mark("after diagnostics")
        ref.to("cpu")
        del out
        torch.cuda.empty_cache()
        mark("pi_ref parked, trajectory released")
        results["memory_profile"] = {"tokens": n, "scored": n_scored, "phases": prof}
        per_tok = (max(p["peak_gib"] for p in prof.values()) - prof[
            "baseline (weights + Adam state)"]["peak_gib"]) * 1024 / n
        print(f"  --> {per_tok:.2f} MiB per token above baseline")

    if args.find_max_length:
        # WHERE EXACTLY DOES IT STOP FITTING? p90 (2674) fits at 73.4 GiB and max (13750)
        # OOMs -- a 5x gap with a hard limit somewhere inside it. That limit is not a
        # curiosity: rollout.max_len is 16384, so the sampler is free to produce trajectories
        # the trainer cannot consume, and nothing downstream would catch it until a job dies
        # hours in. Bisect it, then set max_len from the measurement.
        #
        # scored-token count is held at the measured ratio (~12% of length) so the diagnostic
        # side scales with the forward side, as it does in real trajectories.
        lo, hi = 2674, 16384
        print(f"\n=== searching for the longest trajectory that fits ({lo}..{hi}) ===")

        def _fits(n: int) -> bool:
            try:
                r = bench_one(f"probe{n}", n, max(1, int(n * 0.12)), vocab=vocab,
                              student=student, opt=opt, ref=ref, diag=diag,
                              tau=cfg["opd"]["tau"], warmup=0, repeats=1, dev=DEV)
                print(f"  {n:>6} tokens: fits    (peak {r['peak_gib']:.1f} GiB)")
                return True
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {n:>6} tokens: OOM")
                return False

        search = search_length_ceiling(_fits, lo, hi)
        results["length_search"] = search           # every probe, not just the verdict
        results["max_fitting_tokens"] = search["largest_that_fits"]
        max_len = cfg["rollout"]["max_len"]
        if search["located"]:
            print(f"  --> MEASURED ceiling: {search['largest_that_fits']} tokens fits, "
                  f"{search['smallest_that_ooms']} OOMs.")
            print(f"      rollout.max_len is {max_len}; anything longer is a trajectory the "
                  f"sampler can produce and the trainer cannot consume.")
        else:
            # The case that actually occurs on both cards. Say what was established and what
            # was not -- a ceiling reported here would end up in the paper as a card property.
            print(f"  --> NO CEILING FOUND. {hi} tokens fits, so the limit is ABOVE the "
                  f"largest length this search can reach. The card's ceiling is NOT measured "
                  f"by this run and must not be quoted as one.")
            print(f"      What IS established: every length up to {hi} trains on this card.")
            print(f"      rollout.max_len is {max_len} <= {hi}, so the trainer can consume "
                  f"everything the sampler can produce. That is the decision this gate owed.")

    ok = {k: v for k, v in results.items() if isinstance(v, dict) and "peak_gib" in v}
    print("\n" + "=" * 74)
    if ok:
        worst = max(v["peak_gib"] for v in ok.values())
        print(f"peak {worst:.1f} / {total:.1f} GiB ({worst / total:.0%}), "
              f"headroom {total - worst:.1f} GiB")
        if worst > total * 0.95:
            print("  ^ NO REAL HEADROOM. A longer trajectory or a fragmented allocator will OOM.")
    oomed = [k for k, v in results.items() if isinstance(v, dict) and "oom" in v]
    if oomed:
        print(f"OOM at shape(s): {', '.join(oomed)} -- these do NOT fit this card as configured.")

    if args.weight_by and len(ok) >= 2:
        # A p50-ONLY PROJECTION UNDERSTATES THE TAIL. Cost is strongly superlinear in length
        # (13750 tokens costs 6.85s against 0.84s at 2241 -- 8x the time for 6x the tokens),
        # and the real distribution has a tail that a median cannot represent. Interpolate the
        # measured shapes over the ACTUAL lengths in the cache instead.
        import glob

        lens = sorted(torch.load(p, weights_only=False)["input_ids"].numel()
                      for p in glob.glob(os.path.join(args.weight_by, "traj_*.pt")))
        if lens:
            pts = sorted((v["tokens"], statistics.fmean(v["train_seconds"])
                          + statistics.fmean(v["diag_seconds"])) for v in ok.values())

            def cost(n):                       # piecewise-linear through the measured shapes
                if n <= pts[0][0]:
                    return pts[0][1] * n / pts[0][0]
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    if n <= x1:
                        return y0 + (y1 - y0) * (n - x0) / (x1 - x0)
                return pts[-1][1] * n / pts[-1][0]

            per = statistics.fmean([cost(n) for n in lens])
            gb = int(cfg["optim"].get("global_batch", 1)) or 1
            xf = statistics.fmean(ok["p50"]["ref_transfer_seconds"] or [0.0]) if "p50" in ok else 0.0
            per_total = per + xf / gb
            results["weighted_projection"] = {
                "n_cached": len(lens), "median_tokens": lens[len(lens) // 2],
                "seconds_per_trajectory": per_total}
            print()
            print(f"projection WEIGHTED by the {len(lens)} cached trajectory lengths "
                  f"(median {lens[len(lens) // 2]}, max {lens[-1]}):")
            print(f"  {per_total:.3f}s per trajectory -> "
                  f"{per_total * args.project / 3600:.1f}h per arm")
            print("  This is the number to plan with; the p50-only figure below ignores the "
                  "tail, which is where the cost is.")

    if "p50" in ok:
        gb = int(cfg["optim"].get("global_batch", 1)) or 1
        tr = statistics.fmean(ok["p50"]["train_seconds"])
        dg = statistics.fmean(ok["p50"]["diag_seconds"])
        xf = statistics.fmean(ok["p50"]["ref_transfer_seconds"] or [0.0])
        # THE TRANSFER IS AMORTISED. OPDTrainer.step pages pi_ref in ONCE, loops over all
        # global_batch trajectories, pages it out -- so charging the full 15 GiB round trip to
        # every trajectory (what the first version of this bench did) overstated the
        # projection by ~3.5x on the measured numbers.
        per = tr + dg + xf / gb
        print()
        print(f"projection at the p50 shape, global_batch={gb}:")
        print(f"  train {tr:.3f}s + diag {dg:.3f}s + pi_ref transfer {xf:.3f}s/{gb}"
              f"  =  {per:.3f}s per trajectory")
        print(f"  -> {per * args.project / 3600:.1f}h per arm for {args.project} trajectories")
        if xf > 2 * (tr + dg):
            print(f"  NOTE: the transfer ({xf:.1f}s) dwarfs a single trajectory and is only "
                  f"tolerable BECAUSE it is amortised over {gb}.")
        print("  EXCLUDES rollout (sampling + retrieval + teacher). Measure that in the "
              "precompute job and ADD -- do not blend a synthetic term into a measured one.")

    # WRITTEN EVEN WHEN SHAPES FAILED: a partial measurement is the point of the try/except
    # above, and losing it to a crash at the end would defeat that.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"runtime": rt, "gpu_total_gib": total, "results": results,
               "config": {k: cfg["model"].get(k) for k in
                          ("student", "student_param_dtype", "student_autocast_dtype")}},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\n[bench] raw per-pass timings -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
