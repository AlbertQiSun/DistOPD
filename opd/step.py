"""THE single code path shared by online and offline OPD.

The ONLY thing that differs between the two modes is where `trajectory` and
`teacher_logp` come from -- everything computed inside `opd_train_step` is
identical. This is the code-level guarantee behind the whole phase-1 claim
("offline vs online differ only in rollout source"). Do not fork this.

  - online : trajectory ~ current pi_theta (sampled now); teacher_logp scored live
  - offline: trajectory ~ frozen pi_ref (sampled once); teacher_logp read from disk
"""
from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass

import torch

from .loss import DEFAULT_TAU, compute_opd_loss
from .masking import build_score_mask
from .trajectory import Trajectory


@dataclass
class StepOutput:
    loss: torch.Tensor
    advantage: torch.Tensor
    score_mask: torch.Tensor
    student_logp: torch.Tensor
    teacher_logp: torch.Tensor


def _student_device(student_scorer):
    """The student's CUDA device as a current-device context; nullcontext off-GPU.

    Derived from the model's own parameters -- same rule as HFScorer.logprobs -- so it stays
    correct through snapshot_ref's deepcopy and never needs threading a device attribute.
    """
    lm = getattr(student_scorer, "lm", None)
    try:
        dev = next(lm.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return contextlib.nullcontext()
    if dev.type != "cuda":
        return contextlib.nullcontext()
    return torch.cuda.device(dev)


def opd_train_step(trajectory: Trajectory, teacher_logp: torch.Tensor,
                   student_scorer, tau: float = DEFAULT_TAU) -> StepOutput:
    """Compute advantage + loss for one trajectory. Mode-agnostic on purpose.

    `student_scorer.logprobs(input_ids)` returns token-aligned student logprobs.
    `teacher_logp` is a token-aligned tensor from either the live teacher (online)
    or disk (offline); this function does not know or care which.
    """
    assert teacher_logp.shape == trajectory.input_ids.shape, "teacher_logp length mismatch"
    # REFUSE TO TRAIN UNDER THE BATCH-INVARIANT OVERRIDES. They are inference-only: the
    # direct aten::linear kernel breaks autograd's decomposition and the first backward dies
    # in `linear_backward` (no CUDA kernel) -- and even if it ran, the online student would
    # be training under different kernels than the offline one. Callers wrap training in
    # rollout/engine.student_vanilla_math(); this guard makes a caller that forgot fail
    # HERE, with the cause named, instead of at backward with a dispatcher riddle.
    _bi = sys.modules.get("vllm.model_executor.layers.batch_invariant")
    if _bi is not None and getattr(_bi, "_batch_invariant_MODE", False):
        raise RuntimeError(
            "opd_train_step called while vllm's batch-invariant overrides are live. Training "
            "under them is impossible (aten::linear_backward has no CUDA kernel) and would be "
            "kernel-asymmetric with the offline arm if it were possible. Wrap the training "
            "section in rollout.engine.student_vanilla_math().")
    # RUN UNDER THE STUDENT'S CUDA DEVICE CONTEXT. Load-bearing since both arms train under
    # vLLM's batch-invariant kernel overrides (rollout/engine.pin_student_math): those Triton
    # kernels launch on the thread's CURRENT device, and in the online arm that is cuda:0 --
    # the policy engine's -- while the student lives on cuda:1. The very first online training
    # step died exactly there, in the rotary embedding's bmm, with Triton's misleading
    # "(cpu tensor?)". Vanilla torch never crashed because its ops guard per-call.
    #
    # The guard is per-thread, so it cannot disturb the in-process vLLM engine, whose busy
    # loop runs (and sets its device) on its own thread. On the CPU mock path it is a
    # nullcontext and the single online/offline code path is untouched.
    with _student_device(student_scorer):
        score_mask = build_score_mask(trajectory.token_type)
        student_logp = student_scorer.logprobs(trajectory.input_ids)
        # The student may live on GPU while the trajectory-derived score_mask and the
        # teacher_logp (vLLM output or disk readback) are on CPU. Align the loss inputs to
        # the student's device. StepOutput therefore carries device-consistent tensors for
        # the downstream diagnostics.
        dev = student_logp.device
        teacher_logp = teacher_logp.to(dev)
        score_mask = score_mask.to(dev)
        loss, adv = compute_opd_loss(teacher_logp, student_logp, score_mask, tau)
    return StepOutput(loss=loss, advantage=adv, score_mask=score_mask,
                      student_logp=student_logp, teacher_logp=teacher_logp)
