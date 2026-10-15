"""The masking + logprob-alignment core. This is where confound #1 lives or dies.

Two project-wide invariants are defined here and MUST be used everywhere:

1. `build_score_mask` is the ONE function that decides which tokens are scored.
   Student loss, online teacher scoring, and offline teacher precompute all call
   it. They therefore mask the exact same token set, by construction.

2. `token_aligned_logprobs` fixes the ONE alignment convention:
       logp[t] = log p(input_ids[t] | input_ids[<t]),   logp[0] = nan
   i.e. every per-token logprob array is aligned to the PREDICTED token index t
   (not to the logits row t-1 that produced it). Any backend must emit this
   convention. For HF logits we shift by one here; for vLLM, prompt_logprobs[t]
   is already log p(token_t | <t) with prompt_logprobs[0] = None -- same thing.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .trajectory import TokenType


def build_score_mask(token_type: torch.Tensor) -> torch.Tensor:
    """Single source of the scored-token set: ACTION tokens only.

    Teacher conditions on the FULL context (incl. observation) automatically via
    a full forward pass; we merely restrict *scoring* to ACTION positions here.
    OBSERVATION tokens are 'not scored but still conditioned on' -- that
    distinction is realised by masking here while leaving them in `input_ids`.
    """
    mask = token_type == int(TokenType.ACTION)
    if mask.numel() > 0:
        # An action token can never be at position 0: a prompt always precedes it,
        # and logp[0] is undefined. If this fires, span tagging is broken.
        assert not bool(mask[0]), "ACTION token at position 0 has no prefix to condition on"
    return mask


# Rows of `logits` turned into logprobs per pass. Peak for one chunk is
# chunk x V x 4 bytes, twice (the fp32 cast and the log_softmax output): at V=151936 and
# 1024 rows that is ~1.24 GiB, against ~18.5 GiB for the whole 16k-token sequence at once.
# Chunking the SEQUENCE is exact -- log_softmax normalises within a row, so a row's result
# does not depend on which other rows share the call. See the note in token_aligned_logprobs.
LOGPROB_CHUNK_ROWS = 1024


def _logprob_chunk(logits_rows: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(target_i | row_i) for one slab of rows. The unit that gets checkpointed."""
    lsm = F.log_softmax(logits_rows.float(), dim=-1)          # [rows, V] fp32
    return lsm.gather(1, targets[:, None]).squeeze(1)         # [rows]


def token_aligned_logprobs(logits: torch.Tensor, input_ids: torch.Tensor,
                           chunk_rows: int = LOGPROB_CHUNK_ROWS) -> torch.Tensor:
    """Per-token logprob aligned to the predicted token t (see module docstring).

    logits: [N, V] float. input_ids: [N] long. Returns [N] with entry 0 = nan,
    ON THE SAME DEVICE AS `logits`.

    COMPUTED IN ROW CHUNKS, AND THIS IS EXACT -- not an approximation with a tolerance.
    log_softmax normalises within a row, and the gather picks one entry from that row, so
    row i's output depends on row i alone. Splitting the rows across calls therefore cannot
    change a single bit, and the gate on this asserts bit-identity (values AND gradients),
    not closeness.

    WHY IT IS WORTH DOING. The full-sequence version materialised [N, V] twice in fp32 --
    `logits.float()` and the log_softmax output -- and both survived to backward. At
    N=16384, V=151936 that is 9.27 GiB each on top of the 4.64 GiB bf16 logits, and the
    backward pass needs matching gradient buffers: ~46 GiB for ONE trajectory's scoring
    stage, which is what actually put the student over a 95 GiB card.

    The chunks are wrapped in torch.utils.checkpoint when grad is required. Without that,
    autograd keeps every chunk's log_softmax OUTPUT (its backward is defined in terms of it)
    and the peak is unchanged -- chunking the forward alone would have saved nothing.
    Recomputation is the same op on the same input, so bit-identity survives it.

    The output device is load-bearing, not cosmetic. `torch.full` defaults to CPU, so
    building the result there and filling it from a CUDA gather relied on cross-device
    `copy_` -- which works, silently. The consequence landed one caller up: opd_train_step
    reads `dev = student_logp.device` to align teacher_logp and score_mask, so on a real GPU
    run `dev` was `cpu` and the ENTIRE loss was computed on CPU, with the backward crossing
    the device boundary and forcing a GPU sync per trajectory. HFScorer.logprobs even
    documented the opposite ("stays on `dev`"). Anchor to `logits.device` so the claim is
    true and the loss stays where the model is.
    """
    assert logits.ndim == 2 and input_ids.ndim == 1, "shape"
    assert logits.shape[0] == input_ids.shape[0], "logits/ids length mismatch"
    assert chunk_rows >= 1, "chunk_rows must be positive"
    dev = logits.device
    ids = input_ids.to(dev).long()
    n = int(ids.shape[0])
    logp = torch.full((n,), float("nan"), dtype=torch.float32, device=dev)
    if n < 2:
        return logp                                   # only position 0, which is nan

    # rows 0..n-2 predict tokens 1..n-1; row n-1 has no successor to score
    rows, parts = n - 1, []
    for a in range(0, rows, chunk_rows):
        b = min(a + chunk_rows, rows)
        # targets for rows a..b-1 are ids a+1..b (the align-to-t shift, chunked)
        if logits.requires_grad:
            parts.append(torch.utils.checkpoint.checkpoint(
                _logprob_chunk, logits[a:b], ids[a + 1:b + 1], use_reentrant=False))
        else:
            parts.append(_logprob_chunk(logits[a:b], ids[a + 1:b + 1]))
    logp[1:] = torch.cat(parts) if len(parts) > 1 else parts[0]
    return logp


# Rows per lm_head chunk in head_aligned_logprobs. PINNED IN CODE, deliberately NOT a
# channel knob: unlike log_softmax chunking (exact per row), chunking the head MATMUL
# changes the GEMM's M dimension, and the kernel's blocking by M moves the bits --
# MEASURED at ~5e-7 (1 ulp of the fp32 logits) on CPU MKL,
# tests/test_fused_head_logprobs.py::test_row_chunked_matmul_is_not_bitwise_and_we_know_it.
# A channel file may not change a number (scripts/train.py CHANNEL_ALLOWLIST doctrine), so
# the chunk size that now affects numbers lives here, one value for both arms and both
# channels. 512 rows: ~0.15 GiB bf16 logits + ~0.3 GiB fp32 transients per chunk at
# V=151936 -- the a100-measured-safe size.
HEAD_CHUNK_ROWS = 512


def _head_logprob_chunk(head, hidden_rows: torch.Tensor, targets: torch.Tensor,
                        autocast_dtype, autocast_cache: bool) -> torch.Tensor:
    """lm_head matmul + log p(target) for one slab of rows -- the checkpointed unit.

    The matmul runs under the SAME autocast the full forward used (bf16 logits, weight
    cast per the channel's cache setting); the fp32 widening + log_softmax happen OUTSIDE
    it, exactly as _logprob_chunk always has. Managing autocast INSIDE this function means
    the checkpoint recompute in backward replays it identically by construction.
    """
    if autocast_dtype is not None:
        with torch.autocast(device_type=hidden_rows.device.type, dtype=autocast_dtype,
                            cache_enabled=autocast_cache):
            logits_rows = head(hidden_rows)
    else:
        logits_rows = head(hidden_rows)
    return _logprob_chunk(logits_rows, targets)


def head_aligned_logprobs(head, hidden: torch.Tensor, input_ids: torch.Tensor,
                          chunk_rows: Optional[int] = None,
                          autocast_dtype=None, autocast_cache: bool = True) -> torch.Tensor:
    """token_aligned_logprobs WITHOUT ever materialising the [N, V] logits.

    hidden: [N, H] trunk output. head: the lm_head module. Same output contract as
    token_aligned_logprobs: [N] fp32 on hidden's device, entry 0 = nan. chunk_rows
    defaults to the pinned HEAD_CHUNK_ROWS (see its note for why it is not a knob).

    WHY THIS EXISTS. token_aligned_logprobs chunks everything DOWNSTREAM of the logits,
    but its input is the full [N, V] tensor, already materialised by the model forward --
    bf16 4.35 GiB at N=15360, V=151936 -- and backward allocates a same-sized gradient in
    one piece. On the A100 (79.25 GiB) with the steps>=1 resident base of 64.4 GiB
    (fp32 weights + accumulated grads + Adam moments), that pair is exactly what killed
    the 2026-08-18/19 runs: measured demand ~79.6 GiB, and the cache's census counted 14
    max-length trajectories that hit it deterministically. Folding the lm_head matmul
    INTO the chunk loop caps the transient at chunk-sized pieces: forward peak per chunk
    is [rows, V] bf16 + two fp32 copies, backward regrows only that chunk (checkpoint)
    and the gradient flowing backward is [rows, H] -- the [N, V] never exists.

    NUMERICS, measured not assumed (tests/test_fused_head_logprobs.py):
    - The log_softmax + gather stage is exact per row, as in token_aligned_logprobs.
    - The head MATMUL is NOT bit-identical to the retired full-[N,V] path: the GEMM's
      blocking depends on M, and the measured deviation is ~1 ulp of the fp32 logits
      (~5e-7, CPU MKL). The absolute longhand reference bounds the values; the
      full-path comparison bounds the deviation.
    - What the experiment needs is therefore held by construction plus two assertions:
      BOTH arms run THIS path with the SAME pinned chunk size (no per-channel, per-arm
      or per-config variation can move the bits), and the path is DETERMINISTIC in what
      THIS code controls: values are bit-stable run-to-run (CPU and GPU, asserted), and
      the HEAD-stage gradients are bit-stable in isolation (GEMM + checkpoint replay,
      asserted on GPU). Full-model gradient bit-stability is NOT claimed: the trunk's
      tied-embedding scatter-add and SDPA backward use atomic accumulation, a property
      the retired full-logits path had identically -- measured as same-class in
      tests/test_fused_head_gpu_parity.py.
    - Consequence: numbers from this code version are 1-ulp-class different from the
      full-logits era. Arms must never be paired across that boundary -- the per-record
      `code_version` stamp and scripts/check_channel_parity.py enforce it.
    tests/test_fused_head_gpu_parity.py repeats the measurements on the real checkpoint,
    on the GPU, under bf16 autocast. Recompute-vs-first-pass identity inside the
    checkpoint holds for the same reason it does in token_aligned_logprobs: same op,
    same input, same kernel.

    COST: the head matmul is recomputed once in backward (checkpoint), ~10% of the
    trunk's FLOPs at these shapes -- the same trade gradient checkpointing already makes
    for every decoder layer.
    """
    if chunk_rows is None:
        chunk_rows = HEAD_CHUNK_ROWS      # runtime lookup so tests can pin a small value
    assert hidden.ndim == 2 and input_ids.ndim == 1, "shape"
    assert hidden.shape[0] == input_ids.shape[0], "hidden/ids length mismatch"
    assert chunk_rows >= 1, "chunk_rows must be positive"
    dev = hidden.device
    ids = input_ids.to(dev).long()
    n = int(ids.shape[0])
    logp = torch.full((n,), float("nan"), dtype=torch.float32, device=dev)
    if n < 2:
        return logp                                   # only position 0, which is nan
    needs_grad = torch.is_grad_enabled() and (
        hidden.requires_grad or any(p.requires_grad for p in head.parameters()))
    # rows 0..n-2 predict tokens 1..n-1; row n-1 has no successor to score. The head is
    # never applied to row n-1 at all -- the full-logits path computed it and threw it
    # away; per-row independence makes skipping it value-identical.
    rows, parts = n - 1, []
    for a in range(0, rows, chunk_rows):
        b = min(a + chunk_rows, rows)
        if needs_grad:
            parts.append(torch.utils.checkpoint.checkpoint(
                _head_logprob_chunk, head, hidden[a:b], ids[a + 1:b + 1],
                autocast_dtype, autocast_cache, use_reentrant=False))
        else:
            parts.append(_head_logprob_chunk(head, hidden[a:b], ids[a + 1:b + 1],
                                             autocast_dtype, autocast_cache))
    logp[1:] = torch.cat(parts) if len(parts) > 1 else parts[0]
    return logp


def vllm_prompt_logprobs_to_aligned(prompt_logprobs, input_ids: torch.Tensor) -> torch.Tensor:
    """Convert vLLM `prompt_logprobs` into the project's align-to-t convention.

    vLLM returns a list of length N where entry t is a dict {token_id: Logprob}
    giving log p(token_t | tokens_<t), and entry 0 is None (no prefix). We want
    aligned[t] = log p(input_ids[t] | input_ids[<t]) with aligned[0] = nan --
    which is exactly prompt_logprobs[t][input_ids[t]].logprob. This is the ONLY
    place the vLLM off-by-one is handled; getting it wrong desyncs teacher from
    student on every token.
    """
    n = int(input_ids.shape[0])
    assert len(prompt_logprobs) == n, "prompt_logprobs length != sequence length"
    assert prompt_logprobs[0] is None, "vLLM prompt_logprobs[0] should be None"
    out = torch.full((n,), float("nan"), dtype=torch.float32)
    for t in range(1, n):
        entry = prompt_logprobs[t]
        tid = int(input_ids[t])
        assert tid in entry, f"actual token {tid} missing from prompt_logprobs[{t}]"
        lp = entry[tid]
        out[t] = lp.logprob if hasattr(lp, "logprob") else float(lp)
    return out


def assert_masks_identical(mask_a: torch.Tensor, mask_b: torch.Tensor, what: str = "") -> None:
    assert torch.equal(mask_a, mask_b), f"score masks differ {what}"


def scored_indices(score_mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(score_mask, as_tuple=False).squeeze(-1)


def _head_gather_chunk(head, hidden_rows: torch.Tensor, ids_rows: torch.Tensor,
                       autocast_dtype, autocast_cache: bool):
    """lm_head matmul on one slab of rows, then the fp32 logits GATHERED at `ids_rows` [r, M]
    (-1 = pad, gathered as 0 and masked by the caller) and the row log-sum-exp -- the
    checkpointed unit of head_gathered_logits. Same autocast discipline as
    _head_logprob_chunk: matmul inside, fp32 widening outside."""
    if autocast_dtype is not None:
        with torch.autocast(device_type=hidden_rows.device.type, dtype=autocast_dtype,
                            cache_enabled=autocast_cache):
            logits_rows = head(hidden_rows)
    else:
        logits_rows = head(hidden_rows)
    lg = logits_rows if logits_rows.dtype == torch.float64 else logits_rows.float()   # widen bf16/fp16, never narrow
    lse = torch.logsumexp(lg, dim=-1)
    gathered = torch.gather(lg, 1, ids_rows.clamp_min(0))
    return gathered, lse


def head_gathered_logits(head, hidden: torch.Tensor, rows: torch.Tensor, ids: torch.Tensor,
                         chunk_rows: Optional[int] = None,
                         autocast_dtype=None, autocast_cache: bool = True):
    """The student's fp32 logits at `ids` [N, M] on trunk rows `rows` [N], plus each row's
    log-sum-exp -- everything delta_kl_loss needs from the student, and the [N, V] never
    exists: the head runs on chunk_rows-row slabs exactly as head_aligned_logprobs does
    (same pinned HEAD_CHUNK_ROWS, same per-chunk checkpoint, same autocast placement).
    log p(j) = gathered - lse; the bucket's log p = log(1 - sum_S exp(gathered - lse)).
    ids entries of -1 are pads: gathered as column 0, to be masked by the caller.
    Returns (gathered [N, M] fp32, lse [N] fp32) on hidden's device, grad kept."""
    if chunk_rows is None:
        chunk_rows = HEAD_CHUNK_ROWS
    assert hidden.ndim == 2 and rows.ndim == 1 and ids.ndim == 2 and ids.shape[0] == rows.shape[0], "shape"
    dev = hidden.device
    rows = rows.to(dev).long(); ids = ids.to(dev).long()
    n = int(rows.shape[0])
    if n == 0:
        return (torch.zeros((0, ids.shape[1]), dtype=torch.float32, device=dev),
                torch.zeros((0,), dtype=torch.float32, device=dev))
    needs_grad = torch.is_grad_enabled() and (
        hidden.requires_grad or any(p.requires_grad for p in head.parameters()))
    g_parts, l_parts = [], []
    for a in range(0, n, chunk_rows):
        b = min(a + chunk_rows, n)
        h = hidden[rows[a:b]]
        if needs_grad:
            g, l = torch.utils.checkpoint.checkpoint(_head_gather_chunk, head, h, ids[a:b],
                                                     autocast_dtype, autocast_cache, use_reentrant=False)
        else:
            g, l = _head_gather_chunk(head, h, ids[a:b], autocast_dtype, autocast_cache)
        g_parts.append(g); l_parts.append(l)
    return torch.cat(g_parts), torch.cat(l_parts)
