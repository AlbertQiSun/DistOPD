"""OPD advantage + objective. Single implementation for online AND offline.

Lightning OPD (arXiv:2604.13010):
    A_t = clip( log pi_T(a_t|s_t) - log pi_theta(a_t|s_t), -tau, +tau )   [Eq. 2]
    J_off = E_{q, x}[ sum_t A_t ]                                          [Eq. 4]
with A_t treated as a stop-gradient weight, so the effective gradient is the
advantage-weighted score-function estimator  A_t * grad log pi_theta(a_t).
Default tau = 10 -- WHICH FOLLOWS THEIR PAPER AND DIVERGES FROM THEIR CODE. Verified
exhaustively against github.com/jet-ai-projects/Lightning-OPD at cd6b559 with full history
(2026-08-31): the released trainer applies NO bound to the advantage anywhere. All 22
clamp/clip call sites were inventoried and classified (PPO ratio, low_var_kl estimator, TIS
weights, critic value clip, divide-by-zero guards, FP8 quantisation); the 246-flag argparse
surface contains no advantage-bound flag to leave unset; every consumer of `advantages` /
`returns` was traced; and `git diff 7938152 cd6b559` touches only README.md and two PNGs, so
there is exactly one code state in the repo's history and it never had the clip.

The `[-10, 10]` row entered the paper in arXiv v2 (2026-05-08), the same day the code
shipped. It is not decoration there: Assumption 3.1's finiteness of sigma_A is justified BY
the clip (sigma_A <= T*tau), and Theorem 3.5's bound inherits it -- so their theory needs the
clip their code omits. Keeping tau here is therefore MORE faithful to their analysis than
their implementation is, which is a defensible choice but must not be described as matching
their code.

A NEAR MISS worth naming so nobody re-discovers it as a contradiction: slime's
`ppo_utils.py` does contain `torch.clamp(kl, min=-10, max=10)` -- the only +-10 in the whole
numerical path -- but it bounds the `low_var_kl` estimator taken against the frozen ref
model, not the teacher advantage, and the OPD configs multiply that term by
`--kl-loss-coef 0.00`.

NORMALISATION -- WHY WE DIVIDE BY THE TOKEN COUNT WHEN Eq. 4 WRITES A SUM
-------------------------------------------------------------------------
This is NOT a deviation. Eq. 4's inner `sum_t` is the simplified expression; the
reference implementation normalises, and we match it. Verified against the source, not
recalled:

  * Official repo: https://github.com/jet-ai-projects/Lightning-OPD -- it VENDORS the
    slime trainer (its own `slime/` tree, whose arg names differ from upstream slime:
    `--advantage-estimator on_policy_distillation` vs upstream's `--use-opd`, so upstream
    must not be read in its place).
  * Their `configs/lightning_opd/qwen3-4b-lightning-opd.py` sets `--global-batch-size 256`
    and `--lr 2e-6` (matching the Table 6 values transcribed in configs/opd_qwen3.yaml,
    which is how we know this is the right config) and does NOT set
    `--calculate-per-token-loss`, so slime's default aggregation applies.
  * That default, in the vendored
    `slime/backends/megatron_utils/cp_utils.py::get_sum_of_sample_mean`, is
        (x_i * loss_mask_i).sum() / clamp_min(loss_mask_i.sum(), 1)   summed over samples
    i.e. each sequence divided by ITS OWN scored-token count, then summed over sequences;
    `loss_function` then divides by `step_global_batch_size`, documented as a SAMPLE count.
    The OPD advantage reaches it via `policy_loss_function`'s `sum_of_sample_mean(pg_loss)`.

So the reference implementation is per-sequence token-mean, averaged over sequences --
exactly what this file and OPDTrainer's gradient accumulation do.

WHY NOT THE AGENTIC-RL CONVENTION (token-mean)
----------------------------------------------
The agentic-search line does the opposite, and a reader from there will expect it:
veRL defaults to `loss_agg_mode: token-mean` (verl/trainer/config/actor/actor.yaml), and
Search-R1's `compute_policy_loss` uses `masked_mean(..., eos_mask)` over the whole
micro-batch, i.e. every TOKEN weighted equally. DAPO (arXiv:2503.14476) argues for
token-level loss explicitly, on the grounds that sample-level averaging fails to penalise
bad patterns inside long samples.

We keep sequence weighting anyway, and the reason is specific to what this codebase
MEASURES rather than to what is best for optimisation. Online degradation shows up partly
as LONGER trajectories (more spinning before an answer). Under token-mean those very
trajectories would automatically receive more gradient weight -- the measurement would be
coupled to the effect being measured, and an online-vs-offline gap could be manufactured
by the aggregation rule. DAPO's argument holds in an optimisation setting; ours is a
measurement setting, so the two do not conflict, they answer different questions.

(Length spread here is smaller than the raw budget suggests: rollout.max_len 16384 covers
the WHOLE trajectory including retrieved passages, while only ACTION tokens are scored --
observed n_scored 62..1114, about 18x, comparable to Lightning OPD's 4096-token responses.)

`teacher_logp` and `student_logp` are token-aligned (masking.token_aligned_logprobs)
and only positions where `score_mask` is True (ACTION tokens) contribute.

"""
from __future__ import annotations

import torch

DEFAULT_TAU = 10.0


def compute_advantage(teacher_logp: torch.Tensor, student_logp: torch.Tensor,
                      score_mask: torch.Tensor, tau: float = DEFAULT_TAU) -> torch.Tensor:
    """A_t on scored tokens, 0 elsewhere. Both logps are stop-grad here (advantage
    is a fixed weight). nan entries (e.g. position 0) are guaranteed unscored."""
    diff = teacher_logp.detach() - student_logp.detach()
    zeros = torch.zeros_like(diff)
    adv = torch.where(score_mask, diff, zeros).clamp(-tau, tau)
    # re-zero non-scored positions after clamp (clamp turned nan->nan only where scored=False,
    # but those were already replaced by 0 above; keep this explicit for safety)
    adv = torch.where(score_mask, adv, zeros)
    return adv


def compute_opd_loss(teacher_logp: torch.Tensor, student_logp: torch.Tensor,
                     score_mask: torch.Tensor, tau: float = DEFAULT_TAU):
    """Scalar loss to MINIMIZE (= -J_off), plus the advantage tensor.

    student_logp must carry grad in real training (it is the only term that does);
    teacher_logp and the advantage are detached.
    """
    adv = compute_advantage(teacher_logp, student_logp, score_mask, tau)
    n = score_mask.sum().clamp(min=1)
    # surrogate: maximize sum A_t * log pi_theta  =>  minimize its negative mean
    per_tok = adv.detach() * student_logp
    loss = -(per_tok[score_mask].sum()) / n
    return loss, adv
