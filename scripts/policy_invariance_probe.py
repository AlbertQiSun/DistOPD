#!/usr/bin/env python3
"""Does the SAMPLING policy depend on who it shares a batch with? And what does pinning
that cost? (HPC only, real pi_ref on a real GPU.)

WHY THIS IS NOT THE TEACHER PROBE AGAIN
---------------------------------------
The teacher scores a GIVEN sequence: its output is a continuous logprob, and batch
invariance only affects numbers. The policy SAMPLES: batch composition can change the
tokens themselves, and a changed token cascades -- the rest of the trajectory diverges.
Three consequences shape this script.

1. THE OUTCOME IS DISCRETE, SO "IDENTICAL" IS WEAK EVIDENCE. Ids can match while the
   logits already differ, simply because the perturbation has not yet crossed an argmax /
   sampling-threshold boundary; a longer generation would eventually flip. So every
   comparison also reports the residual on the SAMPLED TOKENS' logprobs. Matching ids with
   a non-zero logprob residual is NOT a pass -- it is a near miss that will bite later.

2. THERE ARE TWO INDEPENDENT CAUSES OF A FLIP, WITH DIFFERENT REMEDIES, so the probe must
   separate them or a FAIL cannot be acted on:
     (i)  NUMERICS -- different tiling -> last-bit logit differences -> argmax crosses.
          Remedy: VLLM_BATCH_INVARIANT=1.
     (ii) RNG STREAM -- the sampler's randomness depends on batch composition/position.
          Remedy: NONE from that flag; it needs per-request derived seeds.
   Separation: run each case at temperature=0 AND at the production temperature.
   Greedy removes RNG entirely, so a greedy flip can only be (i). Greedy clean + sampled
   flip is (ii), and then the cost thresholds below do not apply at all -- the flag is not
   the remedy.

3. THE TWO CONDITIONS NEED TWO PROCESSES. VLLM_BATCH_INVARIANT is read when vllm is
   imported, and two live vLLM V1 engines in one process corrupt each other's state. So a
   run measures ONE setting and writes its JSON; running the other setting afterwards picks
   the first file up and emits the comparison.

IF THE VERDICT IS (ii), THE FIX CARRIES A CONSTRAINT -- RECORD IT BEFORE IT IS NEEDED
--------------------------------------------------------------------------------------
The natural repair is a per-request derived seed, e.g. seed = f(base, question, turn)
instead of one constant. Whatever form it takes, it MUST keep the two arms equally
reproducible: offline samples its trajectories at PRECOMPUTE time (one batching), online
samples them during TRAINING (another). If the derived seed depends on anything that
varies between those two contexts -- wall-clock, process id, position in the batch, a
per-run base -- then online and offline would differ in a way that is NOT "where the
trajectory came from", and that difference is exactly what this whole phase promises does
not exist. Derive only from (question, turn) and a CONSTANT base carried in config.

    # correctness + cost, one setting per run:
    python scripts/policy_invariance_probe.py --model <pi_ref_ckpt> --batch-invariant off
    python scripts/policy_invariance_probe.py --model <pi_ref_ckpt> --batch-invariant on
    # the second run prints the comparison and appends both gate-log entries.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rollout.engine import (  # noqa: E402  (must precede `import vllm`)
    pin_policy_env,
    policy_config_summary,
    policy_engine_kwargs,
)

def parse_args(argv=None):
    """Parsed in main(), not at import: the verdict logic below is what will
    interpret an expensive GPU run, so it has to be unit-testable on CPU."""
    ap = argparse.ArgumentParser(description="Policy batch-invariance: correctness + cost.")
    ap.add_argument("--model", required=True, help="LOCAL pi_ref checkpoint path")
    ap.add_argument("--batch-invariant", choices=["on", "off"], required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--precomputed", default="precomputed",
                 help="dir of traj_*.pt -- REAL prefixes, not synthetic ones")
    ap.add_argument("--filler-lens", default="128,1024,4096")
    ap.add_argument("--max-tokens", type=int, default=256, help="tokens to generate per probe")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8, help="the PRODUCTION temperature")
    ap.add_argument("--cost-questions", type=int, default=32,
                 help="trajectories for the real multi-turn cost run (0 = skip)")
    ap.add_argument("--retriever", default=None, help="endpoint; omit to skip the cost run")
    ap.add_argument("--config", default="configs/opd_qwen3.yaml")
    ap.add_argument("--out-dir", default="runs")
    return ap.parse_args(argv)

args = None            # set by main(); functions read it as a module global


import torch  # noqa: E402

from opd.config import load_config  # noqa: E402
from opd.gate_log import append_gate_run  # noqa: E402
from opd.trajectory import TokenType  # noqa: E402


# ---------------------------------------------------------------------------
def real_prefixes(precomputed_dir: str) -> dict:
    """Two REAL prefixes: the one the policy sees on turn 0, and the one it sees on the
    LAST turn. Synthetic short sequences would probe only one tiling regime -- a turn-0
    prompt is a few hundred tokens while a late-turn prefix carries several thousand
    tokens of injected observations, and that is a different shape entirely."""
    paths = sorted(glob.glob(os.path.join(precomputed_dir, "traj_*.pt")))
    if not paths:
        raise SystemExit(f"no traj_*.pt in {precomputed_dir} -- run a precompute first; "
                         "synthetic prefixes would not represent real tiling")
    out = {}
    for p in paths:
        rec = torch.load(p, weights_only=False)
        ids, tt = rec["input_ids"].long(), rec["token_type"].long()
        act = int(TokenType.ACTION)
        starts = [i for i in range(1, tt.numel())
                  if int(tt[i]) == act and int(tt[i - 1]) != act]
        if len(starts) >= 2:
            out.setdefault("turn0", ids[:starts[0]])
            out.setdefault("late", ids[:starts[-1]])
        if len(out) == 2:
            break
    if len(out) < 2:
        raise SystemExit("could not find both a turn-0 and a late-turn prefix")
    return out


def rand_ids(n: int, vocab: int, gen) -> torch.Tensor:
    return torch.randint(1000, max(1001, vocab - 1000), (n,), generator=gen, dtype=torch.long)


def gen_batch(llm, params, prompts):
    from vllm.inputs import TokensPrompt
    outs = llm.generate([TokensPrompt(prompt_token_ids=p.tolist()) for p in prompts], params)
    res = []
    for o in outs:
        c = o.outputs[0]
        lps = []
        for step_lp, tid in zip(c.logprobs or [], c.token_ids):
            entry = step_lp.get(tid) if isinstance(step_lp, dict) else None
            lps.append(float(getattr(entry, "logprob", float("nan"))) if entry else float("nan"))
        res.append((list(c.token_ids), lps))
    return res


def compare(alone, co):
    """-> (identical, first_divergence, max_logprob_residual). The residual is what makes
    'identical ids' trustworthy rather than lucky."""
    a_ids, a_lp = alone
    c_ids, c_lp = co
    n = min(len(a_ids), len(c_ids))
    first = next((i for i in range(n) if a_ids[i] != c_ids[i]), None)
    if first is None and len(a_ids) != len(c_ids):
        first = n
    resid = 0.0
    for i in range(n if first is None else first):
        x, y = a_lp[i], c_lp[i]
        if x == x and y == y:                       # both finite
            resid = max(resid, abs(x - y))
    return first is None, first, resid


# ---------------------------------------------------------------------------
def run_correctness(llm, prefixes, vocab) -> dict:
    from vllm import SamplingParams

    gen = torch.Generator().manual_seed(args.seed)
    filler_lens = [int(x) for x in args.filler_lens.split(",") if x.strip()]
    assert len(set(filler_lens)) == len(filler_lens) and len(filler_lens) > 1, (
        "fillers must be DIFFERENT lengths -- equal-length padding may leave the tiling "
        "unchanged and pass without testing anything")

    results = {}
    for temp_name, temp in (("greedy", 0.0), ("sampled", args.temperature)):
        for pref_name, a_ids in prefixes.items():
            params = SamplingParams(temperature=temp, top_p=1.0, seed=args.seed,
                                    max_tokens=args.max_tokens, logprobs=0)
            alone = gen_batch(llm, params, [a_ids])[0]
            assert len(alone[0]) > 0, (
                "the probe generated 0 tokens -- an empty generation compares equal to "
                "anything and would pass vacuously")

            fillers = [rand_ids(n, vocab, gen) for n in filler_lens]
            batch, a_at, layout, cursor = [], [], [], 0
            for i, f in enumerate(fillers):
                batch.append(f); layout.append(("F", int(f.numel()), cursor)); cursor += int(f.numel())
                a_at.append(len(batch))
                batch.append(a_ids); layout.append(("A", int(a_ids.numel()), cursor))
                cursor += int(a_ids.numel())
            assert len(batch) > 1, "a batch of one is comparing A with itself"

            flips, firsts, resid = 0, [], 0.0
            for _ in range(args.trials):
                outs = gen_batch(llm, params, batch)
                for k in a_at:
                    same, first, r = compare(alone, outs[k])
                    resid = max(resid, r)
                    if not same:
                        flips += 1
                        firsts.append(first)
            total = args.trials * len(a_at)
            key = f"{temp_name}/{pref_name}"
            results[key] = {
                "prefix_len": int(a_ids.numel()), "generated": len(alone[0]),
                "comparisons": total, "flips": flips, "flip_rate": flips / total,
                "first_divergence": firsts,
                "max_logprob_residual": resid,
                "batch_layout": [{"kind": k, "len": n, "offset": o} for k, n, o in layout],
            }
            print(f"[correctness] {key:16} prefix={int(a_ids.numel()):>6} gen={len(alone[0]):>4} "
                  f"flips={flips}/{total} rate={flips/total:.3f} "
                  f"max|Δlogp|={resid:.3e}"
                  + ("" if flips or resid == 0.0 else "   <-- ids match but logprobs DIFFER"))
    return results


def run_cost(llm, tokenizer, cfg) -> dict:
    """Real multi-turn rollout. Times the POLICY ONLY as well as end-to-end, because
    retrieval latency is identical under both settings and folding it in would compress
    the ratio toward 1 -- making the flag look cheaper than it is."""
    from envs.http_retriever import HTTPRetriever
    from rollout.rollout import generate_trajectories
    from rollout.vllm_policy import VLLMPolicy
    from dataio.dataset import load

    rows = load(cfg["data"]["path"], name=cfg["data"].get("name"),
                limit=args.cost_questions, offset=cfg["data"].get("offset", 0))
    questions = [r["question"] for r in rows]
    policy = VLLMPolicy(llm, tokenizer, temperature=cfg["rollout"]["temperature"],
                        top_p=cfg["rollout"]["top_p"], seed=cfg["run"]["seed"],
                        max_action_tokens=cfg["rollout"]["max_action_tokens"])
    retriever = HTTPRetriever(endpoint=args.retriever)

    stats = {"policy_seconds": 0.0, "act_calls": 0, "batch_sizes": [], "gen_tokens": 0}
    inner = policy.act_batch

    def timed_act_batch(requests):
        # The driver calls act_batch ONCE PER ROUND, so len(requests) IS that round's
        # surviving-trajectory count. Recording it answers, for free, how much the batch
        # actually shrinks in a real rollout -- i.e. how many distinct batch compositions
        # a trajectory meets, which is how many chances the correctness question gets.
        stats["batch_sizes"].append(len(requests))
        stats["act_calls"] += 1
        t0 = time.perf_counter()
        out = inner(requests)
        stats["policy_seconds"] += time.perf_counter() - t0
        stats["gen_tokens"] += sum(int(s.ids.numel()) for s in out)
        return out

    policy.act_batch = timed_act_batch
    t0 = time.perf_counter()
    trajs = generate_trajectories(questions, policy, retriever, tokenizer,
                                  max_turns=cfg["rollout"]["max_turns"],
                                  max_len=cfg["rollout"]["max_len"],
                                  top_k=cfg["rollout"]["top_k"], seed=cfg["run"]["seed"])
    wall = time.perf_counter() - t0
    stats["end_to_end_seconds"] = wall
    stats["n_trajectories"] = len(trajs)
    stats["policy_tokens_per_s"] = stats["gen_tokens"] / max(stats["policy_seconds"], 1e-9)
    print(f"[cost] {len(trajs)} trajectories | policy {stats['policy_seconds']:.1f}s "
          f"({stats['policy_tokens_per_s']:.0f} tok/s) | end-to-end {wall:.1f}s")
    print(f"[cost] batch size per round (survival): {stats['batch_sizes']}")
    return stats


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    global args
    args = parse_args(argv)
    # MUST precede `import vllm` -- the flag is read at import time, so pinning after the
    # import would silently measure whatever the environment happened to carry.
    pin_policy_env(args.batch_invariant == "on")

    import yaml
    from vllm import LLM

    cfg = load_config(args.config)

    engine_kwargs = policy_engine_kwargs(dtype=args.dtype)
    summary = policy_config_summary(dtype=args.dtype)
    print(f"[policy_probe] batch_invariant={args.batch_invariant}")
    print(f"[policy_probe] engine (shared with scripts/train.py make_policy): {summary}")

    llm = LLM(model=args.model, **engine_kwargs)
    from transformers import AutoTokenizer
    from envs.hf_tokenizer import HFTokenizerAdapter
    tokenizer = HFTokenizerAdapter(AutoTokenizer.from_pretrained(args.model))
    vocab = int(tokenizer.tok.vocab_size)

    prefixes = real_prefixes(args.precomputed)
    correctness = run_correctness(llm, prefixes, vocab)
    cost = (run_cost(llm, tokenizer, cfg)
            if args.retriever and args.cost_questions > 0 else None)

    os.makedirs(args.out_dir, exist_ok=True)
    payload = {"batch_invariant": args.batch_invariant, "config": summary,
               "correctness": correctness, "cost": cost, "model": args.model}
    path = os.path.join(args.out_dir, f"policy_probe_{args.batch_invariant}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[policy_probe] wrote {path}")

    greedy_flips = sum(v["flips"] for k, v in correctness.items() if k.startswith("greedy"))
    sampled_flips = sum(v["flips"] for k, v in correctness.items() if k.startswith("sampled"))
    resid = max(v["max_logprob_residual"] for v in correctness.values())
    invariant = (greedy_flips == 0 and sampled_flips == 0 and resid == 0.0)
    append_gate_run(
        "policy_batch_invariance", "PASS" if invariant else "FAIL",
        config=summary,
        details={"batch_invariant": args.batch_invariant, "greedy_flips": greedy_flips,
                 "sampled_flips": sampled_flips, "max_logprob_residual": resid,
                 "per_case": correctness, "cost": cost},
    )

    other = os.path.join(args.out_dir,
                         f"policy_probe_{'off' if args.batch_invariant == 'on' else 'on'}.json")
    if os.path.exists(other):
        with open(other, "r", encoding="utf-8") as f:
            prev = json.load(f)
        _report(payload, prev, summary)


def _agg(run: dict, prefix: str = "") -> dict:
    """flips and the WORST logprob residual over the probes whose key starts with `prefix`."""
    items = [v for k, v in run["correctness"].items() if k.startswith(prefix)]
    return {"flips": sum(v["flips"] for v in items),
            "residual": max(v["max_logprob_residual"] for v in items)}


def classify_cause(off: dict, on: dict) -> tuple:
    """-> (cause_slug, sentence). Pure, so an expensive GPU run's interpretation is testable.

    WHAT DISCRIMINATES (i) FROM (ii) IS THE GREEDY RESIDUAL, NOT THE GREEDY ID COUNT, and the
    difference is not academic -- getting it backwards is what this function did on the A100
    (2026-08-17, runs/policy_probe_off.json). Measured there with the flag OFF:

        greedy/turn0   flips=0/6   max|dlogp|=1.528e-01
        sampled/turn0  flips=6/6   max|dlogp|=1.076e-01

    The old rule read "greedy flips == 0" as "greedy is clean", concluded cause (ii) RNG
    STREAM, and printed that the flag is NOT the remedy and per-request derived seeds are
    needed. Every part of that was wrong, and the evidence was already on screen:

      * greedy was NOT clean. Its LOGITS differed by 0.15; only the argmax happened not to
        cross. This file's own opening paragraph says matching ids with a non-zero residual
        "is NOT a pass -- it is a near miss that will bite later", and under sampling the same
        0.15 does cross a threshold. That is cause (i) exactly.
      * the probe passes ONE constant seed in SamplingParams for every request, so with
        identical prompts and identical seeds the sampler's randomness is fixed by
        construction and the ONLY remaining source of divergence is the logits.
      * the flag demonstrably WAS the remedy: with it on, all four probes gave 0 flips and a
        residual of exactly 0.0.

    The old `else` branch even said the right thing -- "ids matched but logprobs differ ... a
    NEAR MISS ... Treat as cause (i)" -- but `elif s_off > 0` came first and swallowed it.
    """
    g, s = _agg(off, "greedy"), _agg(off, "sampled")
    on_all = _agg(on)
    on_clean = on_all["flips"] == 0 and on_all["residual"] == 0.0

    if g["flips"] == 0 and s["flips"] == 0 and max(g["residual"], s["residual"]) == 0.0:
        return ("already-invariant",
                "policy is ALREADY batch-invariant without the flag -- do not enable it, the "
                "cost thresholds do not apply")

    if g["flips"] > 0 or g["residual"] > 0.0:
        why = ("greedy FLIPPED" if g["flips"] else
               f"greedy ids matched but its logits differed by {g['residual']:.3e} -- a NEAR "
               f"MISS, not invariance, and under sampling that same difference does cross")
        confirm = (" The ON run confirms it: 0 flips, residual exactly 0."
                   if on_clean else
                   " WARNING: the ON run did NOT come back clean, so the flag alone is not "
                   "sufficient here -- read its per-probe lines before trusting this.")
        return ("numerics",
                f"cause (i) NUMERICS: {why}. Tiling changes the logits, so the flag IS the "
                f"remedy -- judge it on the cost ratio below.{confirm}")

    if s["flips"] > 0:
        # Reachable only when greedy was BIT-IDENTICAL and sampling still moved.
        contradiction = ("" if not on_clean else
                         " CONTRADICTION: this says the flag cannot help, yet the ON run came "
                         "back with 0 flips and residual 0.0. One of the two is wrong; the ON "
                         "measurement is the direct evidence and this classification is the "
                         "inference, so re-read the per-probe lines before acting on it.")
        return ("rng-stream",
                "cause (ii) RNG STREAM: greedy was bit-identical (0 flips AND 0 residual) yet "
                "sampling flipped, so the randomness itself depends on batch composition. The "
                "flag is NOT the remedy and the cost thresholds do not apply -- this needs "
                "per-request derived seeds (see the constraint in this file's docstring)."
                + contradiction)

    return ("near-miss",
            "ids matched but logprobs differ -- a NEAR MISS, not invariance; a longer "
            "generation would flip. Treat as cause (i)")


def _report(a: dict, b: dict, summary: str) -> None:
    on = a if a["batch_invariant"] == "on" else b
    off = a if a["batch_invariant"] == "off" else b
    print("=" * 72)
    for name, r in (("OFF", off), ("ON", on)):
        g = sum(v["flips"] for k, v in r["correctness"].items() if k.startswith("greedy"))
        s = sum(v["flips"] for k, v in r["correctness"].items() if k.startswith("sampled"))
        res = max(v["max_logprob_residual"] for v in r["correctness"].values())
        print(f"  batch_invariant={name:3}  greedy_flips={g}  sampled_flips={s}  "
              f"max|Δlogp|={res:.3e}")

    cause, verdict = classify_cause(off, on)
    print(f"  VERDICT: {verdict}")
    # PERSIST THE MATERIALS, not just the sentence. The classification is an INFERENCE over
    # four numbers; recording those numbers is what lets a later reader re-derive it -- or
    # catch it being wrong, which is how this verdict was corrected in the first place.
    append_gate_run("policy_batch_invariance_verdict", cause, config=summary, details={
        "sentence": verdict,
        "off": {"greedy": _agg(off, "greedy"), "sampled": _agg(off, "sampled")},
        "on": {"greedy": _agg(on, "greedy"), "sampled": _agg(on, "sampled")},
    })

    if on.get("cost") and off.get("cost"):
        po, pf = on["cost"]["policy_seconds"], off["cost"]["policy_seconds"]
        eo, ef = on["cost"]["end_to_end_seconds"], off["cost"]["end_to_end_seconds"]
        ratio = po / max(pf, 1e-9)
        band = ("<2x -- keep it on, no discussion" if ratio < 2 else
                "2-4x -- compute total wall clock and check it fits 48h" if ratio < 4 else
                ">4x -- only now consider the fallback, and say so in the paper")
        print(f"  POLICY-ONLY cost ratio (on/off): {ratio:.2f}x   [{band}]")
        print(f"  end-to-end ratio (retrieval-diluted, for wall-clock projection only): "
              f"{eo / max(ef, 1e-9):.2f}x  ({ef:.1f}s -> {eo:.1f}s)")
        print(f"  round survival OFF={off['cost']['batch_sizes']}  ON={on['cost']['batch_sizes']}")
        append_gate_run("policy_batch_invariance_cost", f"{ratio:.2f}x", config=summary,
                        details={"policy_only_ratio": ratio, "policy_seconds_on": po,
                                 "policy_seconds_off": pf, "end_to_end_on": eo,
                                 "end_to_end_off": ef, "band": band,
                                 "round_survival_on": on["cost"]["batch_sizes"],
                                 "round_survival_off": off["cost"]["batch_sizes"]})
    print("=" * 72)


if __name__ == "__main__":
    main()
