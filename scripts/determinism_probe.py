#!/usr/bin/env python3
"""THE determinism probe (HPC-only, real MiroThinker on a real H20).

This is the one gate whose failure has NO code fix: if the teacher's `prompt_logprobs`
are NOT bit-invariant to batch composition, we cannot patch vLLM -- we fall back to a
measured epsilon and gate 3a (teacher-logp bit-identity) is knowingly weakened, which
cascades into the whole online==offline guarantee. So this must be settled FIRST, before
building the SFT/OPD stack, not discovered the night before training.

WHY THIS EXISTS (the source of non-determinism it targets):
  On GPU, a reduction (softmax/logsumexp over the vocab, matmul reductions) is split into
  tiles whose boundaries depend on the batch's TOKEN SHAPE (how many tokens, of what
  lengths, are co-scheduled). Different tiling => different floating-point summation order
  => different last-bit results for the SAME token. `VLLM_BATCH_INVARIANT=1` is supposed to
  pin the tiling so a token's logprob is independent of its batch-mates. vLLM documents this
  for SAMPLING output; `prompt_logprobs` (scoring a GIVEN sequence) is a DIFFERENT code path
  with no such documented guarantee. This probe is the only thing that settles whether the
  flag actually delivers batch-invariance on the path the teacher uses.

THE THREE PINS this probe must satisfy (else it can PASS while missing the real risk):
  (1) CO-BATCH WITH REAL TOKEN-SHAPE DIFFERENCES, not equal-length padding. The fillers use
      DIFFERENT lengths spanning the real trajectory range (default 128 / 1024 / 4096) so the
      batch genuinely exercises different split-K / tiling paths. Equal-length fillers could
      leave the tiling unchanged and FALSELY pass, then blow up under real training where
      offline packs many lengths per batch and online uses small mixed batches. We also place
      SEVERAL copies of the scored sequence A at different batch offsets, each preceded by a
      different-length filler, so A is measured under several distinct tilings in one shot.
  (2) MEASURE THE prompt_logprobs PATH, not sampling. We call SamplingParams(prompt_logprobs=0,
      max_tokens=1, temperature=0) -- the SAME call the served teacher uses -- and compare the
      logp of the PROMPT tokens at ACTION positions (same path, same positions as real scoring
      via opd.masking), never generated tokens.
  (3) USE THE REAL MiroThinker (30B-A3B MoE) on real H20. MoE routing (which experts each token
      visits) can introduce a batch dependence a dense model never shows, and "the vLLM
      batch-invariant family includes qwen3 MoE" is a third-party claim -- this probe verifies
      it on THIS checkpoint. A small dense mock passing proves nothing about the 30B MoE.

OUTPUT: a bit-identical verdict, or the measured max residual. That residual is the PHYSICAL
FLOOR for any fallback epsilon -- gate 3a could only be weakened to >= it. Exit code 0 = PASS
(bit-identical everywhere), 1 = FAIL (report the residual). Use it as a hard gate.

Run (on the teacher node, teacher env, its own card):
    CUDA_VISIBLE_DEVICES=0 python scripts/determinism_probe.py \
      --model /path/to/scratch/.cache/models/MiroThinker-1.7-mini --dtype bfloat16
  optional: --seq-len 512 --filler-lens 128,1024,4096 --a-copies 3 --trials 2
  faithful:  --from-precomputed precomputed/traj_0000.pt   # score a REAL trajectory instead
"""
from __future__ import annotations

import argparse
import os
import sys

# FORCE the determinism source on BEFORE importing vllm -- the whole point is to test whether
# prompt_logprobs is batch-invariant UNDER this flag, exactly as the served teacher pins it.
# setdefault would let a stale env=0 silently make the probe test the wrong thing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from teacher.engine import (  # noqa: E402  (must follow the sys.path insert)
    TEACHER_GPU_MEMORY_UTILIZATION,
    TEACHER_MAX_MODEL_LEN,
    config_summary,
    pin_determinism_env,
    teacher_engine_kwargs,
)

pin_determinism_env()

import torch  # noqa: E402

from opd.masking import build_score_mask, scored_indices, vllm_prompt_logprobs_to_aligned  # noqa: E402
from opd.trajectory import TokenType, TrajectoryBuilder  # noqa: E402


def _rand_ids(n: int, vocab: int, gen: torch.Generator, lo: int = 1000) -> torch.Tensor:
    """n valid token ids in [lo, vocab-lo) -- avoids low/high special-token bands so the
    forward exercises ordinary vocabulary (and, for MoE, ordinary routing)."""
    hi = max(lo + 1, vocab - lo)
    return torch.randint(lo, hi, (n,), generator=gen, dtype=torch.long)


def _synth_trajectory(seq_len: int, vocab: int, gen: torch.Generator):
    """A realistic PROMPT/[ACTION+OBSERVATION]* sequence of ~seq_len tokens, so there ARE
    ACTION positions to score at -- the exact set real teacher scoring reads via build_score_mask.
    Content is random ids (we probe NUMERICS, not semantics); use --from-precomputed for a real one."""
    b = TrajectoryBuilder()
    b.add(_rand_ids(32, vocab, gen), TokenType.PROMPT)
    while b.current_length() < seq_len:
        b.add(_rand_ids(24, vocab, gen), TokenType.ACTION)
        if b.current_length() < seq_len:
            b.add(_rand_ids(40, vocab, gen), TokenType.OBSERVATION)
    b.add(_rand_ids(24, vocab, gen), TokenType.ACTION)  # end on ACTION
    return b.build()


def _load_precomputed(path: str):
    rec = torch.load(path, weights_only=False)
    return rec["input_ids"].long(), rec["token_type"].long()


def _score_batch(llm, params, prompts):
    """Score a LIST of token-id prompts in ONE generate call so they are co-scheduled into
    the same prefill step (co-batched). Returns each output's raw prompt_logprobs, in input
    order (vLLM preserves request order)."""
    from vllm.inputs import TokensPrompt

    # p is a torch.Tensor -- MUST use .tolist() (python ints), NOT list(p) which yields 0-d
    # tensors that msgspec serializes as arrays -> EngineCore rejects them (Expected int, got array).
    def _ids(p):
        return p.tolist() if torch.is_tensor(p) else [int(x) for x in p]

    outs = llm.generate([TokensPrompt(prompt_token_ids=_ids(p)) for p in prompts], params)
    return [o.prompt_logprobs for o in outs]


def main() -> None:
    ap = argparse.ArgumentParser(description="Teacher prompt_logprobs batch-invariance probe (HPC).")
    ap.add_argument("--model", required=True, help="LOCAL MiroThinker path (real 30B MoE; no mock)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seq-len", type=int, default=512, help="length of the scored sequence A")
    ap.add_argument("--filler-lens", default="128,1024,4096",
                    help="comma list of DIFFERENT filler lengths spanning the real traj range")
    ap.add_argument("--a-copies", type=int, default=3,
                    help="how many copies of A to interleave among the fillers (>=1)")
    ap.add_argument("--trials", type=int, default=2, help="repeat the co-batch trial N times")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-precomputed", default=None,
                    help="score a REAL trajectory (a precomputed traj_*.pt) instead of a synthetic one")
    # Defaults come from teacher/engine.py so the probe measures PRODUCTION's engine.
    # Overriding either of these makes the run non-representative -- do it only to
    # deliberately test a configuration you then also ship to teacher_server.
    ap.add_argument("--gpu-memory-utilization", type=float,
                    default=TEACHER_GPU_MEMORY_UTILIZATION)
    ap.add_argument("--max-model-len", type=int, default=TEACHER_MAX_MODEL_LEN)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="Chunked-prefill token budget. DEFAULT (None) sizes it to hold the "
                         "WHOLE batch, which means NO chunking -- that run measures co-batching "
                         "within one prefill step only. Set it BELOW the batch total to force "
                         "chunk boundaries INTO the scored sequence, which is the other half of "
                         "the question and the half that decides how large a budget Task-2 "
                         "step 3 may use (VLLM_BATCH_INVARIANT pins batch COMPOSITION, not "
                         "chunking).")
    args = ap.parse_args()

    filler_lens = [int(x) for x in args.filler_lens.split(",") if x.strip()]
    assert filler_lens, "need at least one filler length"
    assert args.a_copies >= 1

    from vllm import LLM, SamplingParams

    gen = torch.Generator().manual_seed(args.seed)

    # peek vocab from the tokenizer so synthetic/filler ids are valid for THIS checkpoint.
    from transformers import AutoTokenizer
    vocab = int(AutoTokenizer.from_pretrained(args.model).vocab_size)

    if args.from_precomputed:
        a_ids, a_type = _load_precomputed(args.from_precomputed)
        print(f"[probe] scoring REAL trajectory from {args.from_precomputed} (len={a_ids.numel()})")
    else:
        traj = _synth_trajectory(args.seq_len, vocab, gen)
        a_ids, a_type = traj.input_ids, traj.token_type
    action_pos = scored_indices(build_score_mask(a_type))
    assert action_pos.numel() > 0, "no ACTION positions to score -- bad trajectory"

    # co-batch layout: [F0, A, F1, A, F2, A, ...] -- A after a DIFFERENT-length filler each time,
    # so each A copy sits at a distinct cumulative token offset / tiling. batch tokens must all
    # fit ONE prefill step for a genuine co-batch, so size the token budget to the whole batch.
    fillers = [_rand_ids(filler_lens[i % len(filler_lens)], vocab, gen) for i in range(args.a_copies)]
    batch, a_at = [], []
    for i in range(args.a_copies):
        batch.append(fillers[i])
        a_at.append(len(batch)); batch.append(a_ids)     # A right after a filler
    total_tokens = sum(int(p.numel()) for p in batch)
    longest = max(int(args.seq_len), max(filler_lens), int(a_ids.numel()))
    assert longest < args.max_model_len, (
        f"longest probe sequence {longest} does not fit the PRODUCTION max_model_len "
        f"{args.max_model_len}; shorten --seq-len/--filler-lens rather than raising the "
        "ceiling, or the probe stops representing production.")

    # Engine config comes from teacher/engine.py -- the SAME source slurm/teacher_server.py
    # builds from, so this cannot certify a configuration production does not run. The two
    # used to construct engines independently and had drifted (probe 0.90 / derived
    # max_model_len vs production 0.72 / 17408).
    #
    # max_num_batched_tokens / max_num_seqs ARE overridden, deliberately and visibly: the
    # probe's entire job is to co-schedule the batch in ONE prefill step, and production's
    # implicit defaults could chunk it apart and leave the co-batch effect unmeasured -- a
    # probe that cannot co-batch would PASS while proving nothing.
    # Token budget: default holds the whole batch (no chunking); an explicit value below
    # `total_tokens` forces chunk boundaries. Report WHERE the first boundary lands, because
    # "I set a small budget" does not by itself mean the boundary cut the SCORED sequence --
    # if it lands in a filler, the run measures chunking but not the case that matters, and
    # would PASS while leaving the real question open. That is the same vacuity trap as the
    # previous run, whose 6952 tokens simply fit under the 8192 budget.
    budget = args.max_num_batched_tokens or max(total_tokens + 256, 8192)
    if budget >= total_tokens:
        print(f"[chunking] budget {budget} >= batch total {total_tokens} -- NO chunking will "
              "occur. This run measures co-batching within ONE prefill step only; it does NOT "
              "test a chunk boundary inside a scored sequence.")
    else:
        bounds, cursor = [], 0
        for j, p in enumerate(batch):
            bounds.append((cursor, cursor + int(p.numel()), j))
            cursor += int(p.numel())
        hit = next(((s, e, j) for s, e, j in bounds if s < budget < e), None)
        where = ("BETWEEN sequences (aligned) -- boundary does NOT cut any sequence"
                 if hit is None else
                 (f"INSIDE batch item {hit[2]} at offset {budget - hit[0]} of "
                  f"{hit[1] - hit[0]}" + (" -- this IS a scored A copy" if hit[2] in a_at
                                          else " -- a FILLER, not the scored sequence")))
        print(f"[chunking] budget {budget} < batch total {total_tokens} -> first boundary {where}")
        if hit is None or hit[2] not in a_at:
            print("[chunking] WARNING: the first chunk boundary does not fall inside a scored "
                  "sequence, so a PASS here would NOT settle the chunk-boundary question. "
                  "Adjust --seq-len / --filler-lens / --max-num-batched-tokens.")

    engine_kwargs = teacher_engine_kwargs(
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=budget,
        max_num_seqs=len(batch) + 2,
    )
    llm = LLM(model=args.model, **engine_kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)  # SAME as the server

    print(f"[determinism_probe] model={args.model}")
    print(f"[determinism_probe] engine (shared with teacher_server): "
          f"{config_summary(**{k: v for k, v in engine_kwargs.items() if k != 'dtype'}, dtype=args.dtype)}")
    print(f"[probe] A len={int(a_ids.numel())} action_positions={int(action_pos.numel())} "
          f"fillers={filler_lens} a_copies={args.a_copies} batch_tokens={total_tokens} trials={args.trials}")

    def aligned(pl):
        return vllm_prompt_logprobs_to_aligned(pl, a_ids)

    # --- control: A ALONE, scored twice. Must be identical to itself; if not, the
    # non-determinism is NOT batch composition (atomics/kernel), which is itself a finding. ---
    base = aligned(_score_batch(llm, params, [a_ids])[0])
    base2 = aligned(_score_batch(llm, params, [a_ids])[0])
    ctrl = float((base[action_pos] - base2[action_pos]).abs().max())
    print(f"[control] alone vs alone: max|Δ|@action={ctrl:.3e} "
          f"{'OK' if ctrl == 0.0 else 'WARN: alone is itself non-deterministic'}")

    worst = 0.0
    all_bit_identical = True
    for trial in range(args.trials):
        pls = _score_batch(llm, params, batch)
        for k, idx in enumerate(a_at):
            co = aligned(pls[idx])
            d = (base[action_pos] - co[action_pos]).abs()
            mx = float(d.max())
            bit = bool(torch.equal(base[action_pos], co[action_pos]))
            worst = max(worst, mx)
            all_bit_identical &= bit
            print(f"[trial {trial}] A copy #{k} (batch idx {idx}, after filler "
                  f"len {int(batch[idx-1].numel())}) vs alone: max|Δ|@action={mx:.3e} bit_identical={bit}")

    # Record EVERY run, pass or fail. The gate log exists because "when did this last run,
    # at which commit, under which config" was unanswerable from the repo -- which is how
    # this very probe stayed un-executed while the online==offline argument rested on it.
    # Auto-appending is what stops that recurring: a future run cannot forget to log itself.
    from opd.gate_log import append_gate_run
    append_gate_run(
        "determinism_probe",
        "PASS" if (all_bit_identical and ctrl == 0.0) else "FAIL",
        config=config_summary(**{k: v for k, v in engine_kwargs.items() if k != "dtype"},
                              dtype=args.dtype),
        details={"model": args.model, "control_alone_vs_alone": ctrl,
                 "cobatch_max_abs_delta": worst, "all_bit_identical": all_bit_identical,
                 "a_len": int(a_ids.numel()), "action_positions": int(action_pos.numel()),
                 "fillers": filler_lens, "a_copies": args.a_copies,
                 "batch_tokens": total_tokens, "budget": budget,
                 "chunking": budget < total_tokens, "trials": args.trials,
                 "from_precomputed": args.from_precomputed},
    )

    print("=" * 72)
    if all_bit_identical and ctrl == 0.0:
        print("VERDICT: PASS -- prompt_logprobs is bit-identical across batch shapes at action "
              "positions. gate 3a bit-identity holds; the online==offline guarantee stands.")
        sys.exit(0)
    print(f"VERDICT: FAIL -- max residual ε = {worst:.6e} (control {ctrl:.3e}).")
    print("  This ε is the PHYSICAL FLOOR for any fallback tolerance: gate 3a cannot be tighter "
          "than it, and adopting it means gate 3a is KNOWINGLY weakened from bit-identity to ~ε. "
          "Re-evaluate the online==offline claim before training on this backend.")
    sys.exit(1)


if __name__ == "__main__":
    main()
