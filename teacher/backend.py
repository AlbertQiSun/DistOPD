"""Gate 1: backend-consistency constraint.

online (teacher scored live) and offline (teacher precomputed to disk) MUST use
an identical scoring backend -- same engine, dtype, tokenizer, and frozen
template constants. If they diverge (e.g. online=vLLM bf16, offline=HF fp32) the
same token gets different teacher logprobs and the online/offline comparison
silently absorbs a "two backends" confound -- precisely what phase-1 must rule
out. This is enforced by an explicit config + a startup assertion, not by habit.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from configs import template as T


def template_fingerprint() -> str:
    """Hash of every frozen template constant that can shift token boundaries.
    Any drift changes the fingerprint and trips assert_same_backend."""
    parts = [
        str(T.ENABLE_THINKING), str(T.ADD_GENERATION_PROMPT),
        T.THINK_OPEN, T.THINK_CLOSE, T.SEARCH_OPEN, T.SEARCH_CLOSE,
        T.INFO_OPEN, T.INFO_CLOSE, T.ANSWER_OPEN, T.ANSWER_CLOSE,
        T.OBS_PREFIX, T.OBS_SUFFIX, T.IM_START, T.IM_END, T.SYSTEM_PROMPT,
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


def runtime_fingerprint() -> dict:
    """What THIS process is actually running on: GPU model, compute capability, versions.

    WHY THIS IS PART OF THE BACKEND AND NOT TRIVIA. The plan is to open a second execution
    channel (a different GPU type, a different CUDA/vLLM build) so work can go wherever the
    queue is shorter. The hard rule for that plan is that ONE experiment's precompute and
    both its training arms stay inside ONE channel -- an A800-scored teacher logprob and an
    H20-scored one are not the same number, and mixing them puts a hardware confound
    directly into the online-vs-offline comparison the whole phase exists to isolate.

    Before this, that rule had no machine check. BackendConfig recorded engine, dtype,
    tokenizer, template and vLLM runtime -- all of which are IDENTICAL across channels -- so
    a cache scored on A800 and a cache scored on H20 carried the same fingerprint and gate 1
    would accept either. The constraint lived only in whoever remembered it, and every
    failure this project has hit has been the same shape: a discipline that was not written
    down as an assertion got violated before anyone noticed.

    Versions are here for the same reason as the GPU: a channel is (hardware, stack), and
    torch/vLLM differ between channels by construction -- that is why the channel exists.
    """
    out = {"gpu_name": None, "gpu_sm": None, "torch_version": None, "vllm_version": None}
    try:
        import torch

        out["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            out["gpu_name"] = torch.cuda.get_device_name(0)
            out["gpu_sm"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
    except Exception:                      # noqa: BLE001 - a missing/broken torch is not
        pass                               # this function's problem to diagnose
    try:
        import vllm

        out["vllm_version"] = vllm.__version__
    except Exception:                      # noqa: BLE001
        pass
    return out


# Fields describing the SERVER's runtime -- its vLLM settings AND the machine underneath. A
# client cannot know them a priori -- it learns them from /health and stamps them onto
# whatever it writes -- so they are NOT part of the startup equality check. Drift is caught
# at the DISK boundary instead, where a cache scored under one configuration meets a run
# using another.
ENGINE_RUNTIME_FIELDS = ("max_model_len", "max_num_batched_tokens", "gpu_memory_utilization",
                         "gpu_name", "gpu_sm", "torch_version", "vllm_version")


@dataclass(frozen=True)
class BackendConfig:
    engine: str            # "vllm" | "hf" | "mock"
    dtype: str             # "bfloat16" | "float32" | "mock"
    tokenizer_name: str    # e.g. "Qwen/Qwen3-1.7B" | "mock"
    enable_thinking: bool = T.ENABLE_THINKING
    add_generation_prompt: bool = T.ADD_GENERATION_PROMPT
    template_fp: str = None  # filled in post-init
    # --- engine runtime (see ENGINE_RUNTIME_FIELDS) --------------------------------
    # max_model_len and max_num_batched_tokens can CHANGE THE NUMBERS: the second sets the
    # chunked-prefill budget, and VLLM_BATCH_INVARIANT pins batch COMPOSITION, not chunk
    # boundaries -- the determinism probe verified co-batching within ONE prefill step and
    # did not cover a chunk boundary falling inside a scored sequence. Neither was visible
    # to gate 1 before, so a teacher restart under different settings could have shifted
    # every logprob with nothing to notice.
    #
    # gpu_memory_utilization sizes the KV pool, and is MEASURED not to change the generated
    # tokens ON THE SERIAL GENERATION PATH: runs/util_858 vs runs/util_900, 2026-08-22 --
    # Qwen3-32B, 32 questions, util 0.858 against 0.900 with every other engine field and the
    # code version identical, 32/32 trajectories byte-identical compared on every turn's text
    # and token count, the termination reason and the final action. (It read "believed not to
    # affect values" until that ran. The claim was load-bearing: the three teacher lines size
    # this per checkpoint -- 0.80 / 0.80 / 0.858 -- and so do not share it.)
    #
    # WHAT THAT DOES NOT COVER. The test ran serial, prefix caching off, batch invariance off,
    # which closes the two routes by which pool size could reach the numbers: eviction (nothing
    # cached to evict) and block-starved scheduling (one sequence, ~24k tokens of pool against
    # a 17408 ceiling). The BATCHED rollout is not covered. The SCORING path is not covered
    # either and does not need to be -- precompute and both arms read one channel-certified
    # value there, so it cannot differ between the things being compared.
    #
    # Recorded regardless: the cost of carrying it is zero and the cost of being wrong is
    # another silent drift.
    #
    # None means "not explicitly configured -- vLLM derived it from max_model_len". That is
    # a meaningful value, not missing data: if a later run sets an explicit budget the field
    # goes None -> number and the mismatch is caught.
    max_model_len: int = None
    max_num_batched_tokens: int = None
    gpu_memory_utilization: float = None
    # --- the MACHINE the numbers came off, see runtime_fingerprint() ------------------
    # None means "not recorded" -- true of every cache written before these existed. That is
    # deliberately distinguishable from a real value, so an old artifact reads as unknown
    # rather than as a match.
    gpu_name: str = None
    gpu_sm: str = None
    torch_version: str = None
    vllm_version: str = None

    def __post_init__(self):
        if self.template_fp is None:
            object.__setattr__(self, "template_fp", template_fingerprint())


def assert_same_backend(online: BackendConfig, offline: BackendConfig) -> None:
    """Fail loudly if the online and offline teacher backends are not identical."""
    o, f = asdict(online), asdict(offline)
    diffs = [k for k in o if o[k] != f[k]]
    if diffs:
        lines = "\n".join(f"  {k}: online={o[k]!r}  offline={f[k]!r}" for k in diffs)
        raise AssertionError(
            "online and offline teacher backends differ -- this would inject a "
            f"backend confound into the offline-vs-online comparison:\n{lines}"
        )


def assert_same_teacher_model(teacher: BackendConfig, generator: BackendConfig) -> None:
    """Teacher-consistency gate for the SFT milestone (M-SFT Stage 1).

    The model that GENERATES the SFT data MUST be the same model that scores in OPD
    (both MiroThinker). If the SFT data came from any other model, sigma_Delta > 0
    and offline inherits the Theorem-3.11 irreducible bias -- so this is a hard,
    machine-checked red line, not a comment.

    We compare model IDENTITY (dtype, tokenizer/model name, frozen template, thinking
    flags) but deliberately EXCLUDE `engine`: SFT only SAMPLES tokens from the model,
    it never scores logprobs, so a vLLM-sampled generator feeding an HF-scored OPD
    teacher is fine here -- both are the same MiroThinker weights. Engine-level numeric
    identity is a DIFFERENT axis: it is gate 1's concern for the online-vs-offline
    teacher SCORING (assert_same_backend), where the same token must get the same
    logprob. Conflating the two would wrongly force SFT data generation onto the slow
    HF path for no provenance benefit.
    """
    o, f = asdict(teacher), asdict(generator)
    diffs = [k for k in o if k != "engine" and o[k] != f[k]]
    if diffs:
        lines = "\n".join(f"  {k}: teacher={o[k]!r}  generator={f[k]!r}" for k in diffs)
        raise AssertionError(
            "SFT data generator is NOT the OPD teacher model -- this re-introduces "
            "sigma_Delta>0 (SFT distribution != OPD scoring distribution) and gives "
            f"offline a Theorem-3.11 irreducible bias:\n{lines}"
        )
