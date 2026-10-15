"""Teacher / student logprob scorers, mock + real backends.

Interface: `Scorer.logprobs(input_ids) -> token-aligned logprobs` (the align-to-t
convention from opd.masking). Every backend implements the SAME interface and the
SAME alignment, so online and offline are numerically comparable when (gate 1)
their BackendConfig matches.

  - MockLM / Scorer / TeacherScorer : deterministic CPU dry-run.
  - HFScorer   : HuggingFace CausalLM (shift-by-one on logits).
  - VLLMScorer : vLLM prompt_logprobs over a fixed sequence (lazy import; HPC).

TeacherScorer adds OPD modes: score_online() and precompute_and_save()/load().
CRITICAL: precompute scores the FULL trajectory (incl. observation tokens) so the
teacher conditioning is identical to online -- never an action-only sequence.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from opd.masking import (head_aligned_logprobs, head_gathered_logits, token_aligned_logprobs,
                         vllm_prompt_logprobs_to_aligned)
from opd.trajectory import Trajectory

from .backend import BackendConfig

MOCK_CONFIG = BackendConfig(engine="mock", dtype="mock", tokenizer_name="mock")


class MockLM(nn.Module):
    """Deterministic tiny LM. Same seed => same logits => reproducible logprobs."""

    def __init__(self, vocab_size: int, dim: int = 32, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(self.emb.weight.shape, generator=g))
            self.head.weight.copy_(torch.randn(self.head.weight.shape, generator=g) * 0.1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(input_ids))


class Scorer:
    """Wraps a logits-producing LM; exposes token-aligned logprobs."""

    def __init__(self, lm: nn.Module, requires_grad: bool = False,
                 config: BackendConfig = MOCK_CONFIG,
                 autocast_dtype: Optional[str] = None,
                 autocast_cache: bool = True,
                 logprob_chunk_rows: Optional[int] = None) -> None:
        self.lm = lm
        self.requires_grad = requires_grad
        self.config = config
        # MIXED PRECISION, SPLIT DELIBERATELY: parameters/grads/optimizer state stay in the
        # dtype the model was loaded with (fp32 for the trained student), while the FORWARD
        # runs in autocast_dtype (bf16). Keeping the master weights fp32 is not an
        # optimisation -- at lr 2e-6 a bf16 parameter cannot represent the update at all
        # (see opd/trainer.py and tests/test_optimizer_precision.py), so the run would be a
        # no-op. None = no autocast, which is the mock/CPU path and every existing caller.
        self.autocast_dtype = autocast_dtype
        # CHANNEL KNOBS (configs/channels/*.yaml). Both trade speed for memory and NEITHER
        # changes a number: the cache decides whether autocast keeps a bf16 copy of every
        # weight (~7.5 GiB for the 4B) or re-casts per use, and chunk_rows only regroups an
        # exact, per-row computation. They live here rather than as globals so the two
        # hardware channels can differ without either editing the other's config.
        self.autocast_cache = autocast_cache
        self.logprob_chunk_rows = logprob_chunk_rows

    def _chunk_kw(self) -> dict:
        """chunk_rows only when the channel set one, so the default stays in one place."""
        return {} if self.logprob_chunk_rows is None else {"chunk_rows": self.logprob_chunk_rows}

    def _split_lm(self):
        """(trunk, head) when the wrapped lm exposes them separably; None otherwise.

        The split is what lets logprobs() fold the lm_head into the chunk loop
        (opd.masking.head_aligned_logprobs) so the [N, V] logits never materialise.
        MockLM is head(emb(ids)); HFScorer overrides for the HF model/lm_head pair.
        """
        emb = getattr(self.lm, "emb", None)
        head = getattr(self.lm, "head", None)
        if emb is not None and head is not None:
            return emb, head
        return None

    def _head_kw(self) -> dict:
        """Autocast forwarding for the fused path. DELIBERATELY does not forward the
        channel's logprob_chunk_rows: head chunking moves bits (the GEMM's M-blocking,
        measured ~1 ulp -- see opd.masking.HEAD_CHUNK_ROWS), and a channel file may not
        change a number, so the head chunk size is the pinned code constant."""
        return {"autocast_dtype": (None if self.autocast_dtype is None
                                   else getattr(torch, self.autocast_dtype)),
                "autocast_cache": self.autocast_cache}

    def gathered_logits(self, input_ids: torch.Tensor, rows: torch.Tensor, ids: torch.Tensor):
        """The delta arm's student read: fp32 logits at `ids` [N, M] on trunk rows `rows` [N]
        plus each row's log-sum-exp (opd.masking.head_gathered_logits), grad kept, the [N, V]
        never built. Mirrors `logprobs` -- same trunk-under-autocast, same head discipline."""
        dev = next(self.lm.parameters()).device
        input_ids = input_ids.to(dev)
        self.lm.train() if self.requires_grad else self.lm.eval()
        ctx = torch.enable_grad() if self.requires_grad else torch.no_grad()
        split = self._split_lm()
        if split is None:
            raise RuntimeError(f"{type(self.lm).__name__} exposes no trunk/head split; gathered_logits needs the "
                               "chunked head path (the full-logits path would build the [N, V] this arm exists to avoid)")
        trunk, head = split
        with ctx:
            with self._autocast(dev):
                hidden = trunk(input_ids)
            return head_gathered_logits(head, hidden, rows, ids, **self._head_kw())

    def to(self, device):
        """Move the wrapped model. Returns self so call sites read as one statement.

        pi_ref uses this to sit on the HOST during training and come back for the diagnostic
        pass. It is a full fp32 copy of the 4B student -- ~15 GiB -- and it is idle for the
        entire forward/backward, which is exactly when the card has no room to spare."""
        self.lm.to(device)
        return self

    def _autocast(self, device: torch.device):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type,
                              dtype=getattr(torch, self.autocast_dtype),
                              cache_enabled=self.autocast_cache)

    def logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        dev = next(self.lm.parameters()).device
        split = self._split_lm()
        self.lm.train() if self.requires_grad else self.lm.eval()
        ctx = torch.enable_grad() if self.requires_grad else torch.no_grad()
        if split is None:
            # lm does not expose trunk/head separably: full-logits path, chunked only
            # downstream of the logits. This is also the REFERENCE path the fused one is
            # bit-compared against in tests/test_fused_head_logprobs.py.
            with ctx, self._autocast(dev):
                logits = self.lm(input_ids)
            # OUTSIDE the autocast region on purpose: token_aligned_logprobs casts to fp32
            # and normalises there, and that must not be re-narrowed by an enclosing
            # autocast. The logits themselves stay bf16, which is what the memory budget
            # assumes.
            return token_aligned_logprobs(logits, input_ids, **self._chunk_kw())
        trunk, head = split
        with ctx:
            # Trunk under autocast exactly as the full forward was; the head matmul
            # re-enters autocast PER CHUNK inside head_aligned_logprobs, so the [N, V]
            # logits never materialise (see the memory note there). Numerics vs the
            # retired full-logits path: 1-ulp-class GEMM blocking deviation, MEASURED in
            # tests/test_fused_head_logprobs.py; both arms share this path and its pinned
            # chunk size, and determinism is asserted there too.
            with self._autocast(dev):
                hidden = trunk(input_ids)
            return head_aligned_logprobs(head, hidden, input_ids, **self._head_kw())

    def raw_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[N, V] logits, eval + no grad, ON THE MODEL'S DEVICE and in the model's dtype.

        Exists so a caller that needs only SOME rows can take them before anything is
        widened to fp32 or copied to the host. full_logprobs below does neither of those
        cheaply: it casts the whole [N, V] to fp32 and moves it, which for one 13,750-token
        trajectory is 8.4 GB across the bus for the ~1,700 rows the diagnostic actually reads.
        """
        dev = next(self.lm.parameters()).device
        self.lm.eval()
        with torch.no_grad(), self._autocast(dev):
            return self.lm(input_ids.to(dev))

    def log_dist_at(self, input_ids: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """log_softmax of ONLY `rows` of the logits -> [len(rows), V] fp32, on device.

        The diagnostic wants the distribution at the rows that generated SCORED tokens --
        p50 is 270 of 2241, so the full-sequence version computes 8x more than it uses and
        then pays to move all of it. Normalisation is per row, so restricting rows changes
        nothing about the values.
        """
        logits = self.raw_logits(input_ids)
        return torch.log_softmax(logits[rows.to(logits.device)].float(), dim=-1)

    def full_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full next-token log-distribution [N, V] (eval, no grad). Row t is the
        distribution predicting token t+1.

        KEPT for callers that genuinely want every row; the diagnostics no longer do (see
        log_dist_at). Returns on the model's device -- it used to force .cpu(), which is
        what made the per-step transfer bill dominate the step it was measuring."""
        return torch.log_softmax(self.raw_logits(input_ids).float(), dim=-1)


class TeacherScorer(Scorer):
    def __init__(self, lm: nn.Module, config: BackendConfig = MOCK_CONFIG) -> None:
        super().__init__(lm, requires_grad=False, config=config)  # teacher frozen

    def score_online(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logprobs(input_ids)

    def precompute_and_save(self, traj: Trajectory, path: str,
                            store_dtype: str = "float32") -> torch.Tensor:
        logp = self.logprobs(traj.input_ids)  # FULL sequence incl. observation (float32)
        _save_precomputed(traj, logp, self.config, path, store_dtype)
        return logp

    @staticmethod
    def load(path: str) -> dict:
        return torch.load(str(path), weights_only=False)


def _save_precomputed(traj: Trajectory, logp: torch.Tensor, config: BackendConfig,
                      path: str, store_dtype: str) -> None:
    """Serialize precomputed teacher logprobs. dtype is recorded explicitly so the
    reader knows any precision loss was declared, not silent. Default float32."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": traj.input_ids,
            "token_type": traj.token_type,
            "teacher_logp": logp.to(getattr(torch, store_dtype)),
            "logp_dtype": store_dtype,
            "backend": config,
            # frozen rollout context (query/docs/gold per turn) so OFFLINE context-quality
            # is measured on the SAME material online sees -- and stays frozen by construction.
            "meta": traj.meta,
            # confound #2 (termination) raw materials. These are Trajectory FIELDS, not meta,
            # so they were silently dropped here and defaulted (None / 0) on the offline read --
            # making every offline record report reason=null, num_search_calls=0 while ONLINE
            # reported the real values. That is a spurious online-vs-offline difference in the
            # very dimension the diagnostic exists to measure. Store the enum's .value (a plain
            # str) so the payload does not depend on the enum class at unpickle time.
            "terminated": traj.terminated.value if traj.terminated is not None else None,
            "num_search_calls": int(traj.num_search_calls),
        },
        str(p),
    )


def assert_no_active_dropout(model) -> None:
    """Refuse train() mode on a model that would then sample dropout.

    Gradient checkpointing only engages in train() mode (see enable_grad_checkpointing), and
    train() ALSO switches dropout on. Every dropout draw is a fresh random number, so the
    same trajectory would get a different loss on every pass -- online and offline would
    diverge for a reason that is not "where the trajectory came from", which is the one
    difference this phase permits. Qwen3 ships attention_dropout=0.0, but that is a property
    of the CHECKPOINT's config, not a law, so check rather than assume.
    """
    offenders = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Dropout) and float(getattr(mod, "p", 0.0)) > 0:
            offenders.append(f"module {name}: p={mod.p}")
    cfg = getattr(model, "config", None)
    for key in dir(cfg) if cfg is not None else []:
        if "dropout" not in key or key.startswith("_"):
            continue
        val = getattr(cfg, key, None)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
            offenders.append(f"config.{key}={val}")
    if offenders:
        raise AssertionError(
            "this checkpoint has ACTIVE DROPOUT, so putting it in train() mode (which "
            "gradient checkpointing requires) would make every forward pass random and "
            "break the online/offline comparison:\n  " + "\n  ".join(offenders))


def enable_grad_checkpointing(model) -> None:
    """Turn on activation checkpointing for a model that is about to be TRAINED.

    Without this, all 36 layers' activations are kept for backward: at N=16384 that is
    ~58 GiB for one trajectory, which no arrangement of the rest fits on a 95 GiB card. With
    it only the layer boundaries survive (~2.8 GiB) and each block is recomputed, costing
    roughly a third more compute in the backward pass.

    TWO FOOTGUNS, BOTH SILENT, BOTH GATED IN tests/test_grad_checkpointing.py:

    1. IT ONLY ENGAGES IN train() MODE. HF layers check `self.gradient_checkpointing and
       self.training`, and `from_pretrained` hands back a model in EVAL mode -- so calling
       gradient_checkpointing_enable() and stopping there does exactly nothing, with no
       warning and no error. Measured: recomputation happens in train() and not in eval().
    2. train() ALSO ENABLES DROPOUT, which would make the loss random per pass. Guarded
       above, before the mode switch, not after.

    use_reentrant=False is the non-legacy implementation. In THIS configuration the
    reentrant variant is numerically fine -- the checkpointed block's input comes from the
    embedding and does require grad -- so swapping it in does not turn the gate red, and that
    is honest rather than a hole. It is still the wrong default: it is the deprecated path
    (torch warns), and it silently produces NO gradient for any block whose inputs happen not
    to require grad, which is one frozen-embedding experiment away from mattering.
    """
    assert_no_active_dropout(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False        # a KV cache is meaningless under recomputation
    model.train()                             # REQUIRED -- see footgun 1


class HFScorer(Scorer):
    """HuggingFace CausalLM backend. Real teacher/student on HPC."""

    @classmethod
    def from_pretrained(cls, name: str, dtype: str = "bfloat16",
                        requires_grad: bool = False, device: str = "cpu",
                        grad_checkpointing: Optional[bool] = None,
                        autocast_dtype: Optional[str] = None,
                        autocast_cache: bool = True,
                        logprob_chunk_rows: Optional[int] = None):
        """`dtype` is the PARAMETER dtype -- what the optimizer updates and what Adam's
        moments are shaped like. `autocast_dtype` is what the FORWARD computes in.

        For the trained student these MUST differ: fp32 parameters, bf16 compute. At
        lr 2e-6 an AdamW step moves a weight by ~2e-6 while bf16's spacing near |w|=0.02 is
        7.8e-5 -- 39x coarser -- so a bf16 parameter rounds the update away and 95.6% of
        weights never move at all. Loading in bf16 to save memory silently converts training
        into a no-op that looks like a stable flat line. See tests/test_optimizer_precision.py.

        `grad_checkpointing` defaults to requires_grad: the TRAINED model is the only one
        holding activations for a backward pass. The teacher and the pi_ref snapshot score
        under no_grad, where checkpointing would buy nothing and recompute for free."""
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype))
        model.to(device)
        if grad_checkpointing is None:
            grad_checkpointing = requires_grad
        if grad_checkpointing:
            enable_grad_checkpointing(model)
        cfg = BackendConfig(engine="hf", dtype=dtype, tokenizer_name=name)
        return cls(model, requires_grad=requires_grad, config=cfg,
                   autocast_dtype=autocast_dtype, autocast_cache=autocast_cache,
                   logprob_chunk_rows=logprob_chunk_rows)

    def _split_lm(self):
        """HF CausalLM = trunk (`lm.model`) + `lm.lm_head`. Refuses to fall back: the
        full-logits path is the one that OOMs (the [N, V] pair at max_len is ~1 GiB more
        than the A100 has left at steps>=1 -- see head_aligned_logprobs), so silently
        degrading to it would reintroduce a crash that only fires on the 14-in-38400
        longest trajectories, weeks later. An HF model without these attributes is a
        loud error, not a slower run."""
        inner = getattr(self.lm, "model", None)
        head = getattr(self.lm, "lm_head", None)
        if inner is None or head is None:
            raise AttributeError(
                f"{type(self.lm).__name__} does not expose .model/.lm_head; the chunked "
                "lm_head path (head_aligned_logprobs) cannot run and the full-logits "
                "fallback is exactly the [N, V] materialisation that OOMs at max_len.")

        def trunk(ids: torch.Tensor) -> torch.Tensor:
            # ids arrive already on the model's device (logprobs moves them); the batch
            # dim is HF's, stripped again so the head chunks see [rows, H].
            return inner(ids.unsqueeze(0)).last_hidden_state[0]
        return trunk, head

    def logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        # The trajectory ids arrive on CPU; move them to wherever the model lives
        # (GPU in a real run) so the forward matches. Device is DERIVED from the
        # model's own parameters -- no device attribute to thread, and it stays
        # correct through deepcopy (snapshot_ref) which preserves placement.
        dev = next(self.lm.parameters()).device
        input_ids = input_ids.to(dev)
        # SET THE MODE, EVERY TIME. HF gradient checkpointing engages only when
        # `self.training` is True, and raw_logits/full_logprobs (the diagnostic path) put the
        # model in eval() on every call. This override previously inherited neither -- so
        # after the FIRST diagnostic pass the student trained in eval mode forever, silently
        # retaining every activation instead of recomputing them.
        #
        # Measured, before the fix: 41.9 GiB of autograd graph for ONE 5000-token trajectory,
        # against ~2.5 GiB when checkpointing actually runs. That is what capped the trainable
        # trajectory length at 5458 tokens with rollout.max_len at 16384, and it would have
        # OOM'd the real run at step 1 -- after step 0's diagnostics flipped the mode.
        self.lm.train() if self.requires_grad else self.lm.eval()
        ctx = torch.enable_grad() if self.requires_grad else torch.no_grad()
        trunk, head = self._split_lm()
        with ctx:
            # Trunk under autocast as the full forward was; the lm_head runs INSIDE the
            # chunk loop (head_aligned_logprobs) so the [N, V] logits -- and their
            # backward-time gradient twin -- never materialise. fp32 widening +
            # log_softmax stay outside autocast per chunk, same convention as before.
            # NOT bit-identical to the retired full-logits path: GEMM M-blocking moves
            # the bits by ~1 ulp (measured, tests/test_fused_head_logprobs.py). What is
            # held instead: longhand-referenced values, determinism, and both arms on
            # this same path with the pinned chunk size. Never pair arms across the
            # code-version boundary -- the per-record code_version stamp exists for that.
            with self._autocast(dev):
                hidden = trunk(input_ids)
            return head_aligned_logprobs(head, hidden, input_ids,
                                         **self._head_kw())      # on `dev`, grad kept

    def raw_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The ONLY thing HF needs to override: a batch dimension. log_dist_at and
        full_logprobs are inherited, so the row-restricted path cannot drift away from the
        full one -- which is exactly what happened to the previous duplicated
        full_logprobs (it never got autocast, and it forced .cpu())."""
        dev = next(self.lm.parameters()).device
        self.lm.eval()
        with torch.no_grad(), self._autocast(dev):
            return self.lm(input_ids.unsqueeze(0).to(dev)).logits[0]


class TeacherHFScorer(HFScorer):
    """Real HF teacher: same score_online / precompute_and_save surface as the mock
    and vLLM teachers, so train.py can wire engine=hf identically."""

    def score_online(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logprobs(input_ids)

    def precompute_and_save(self, traj: Trajectory, path: str,
                            store_dtype: str = "float32") -> torch.Tensor:
        logp = self.logprobs(traj.input_ids)
        _save_precomputed(traj, logp, self.config, path, store_dtype)
        return logp


class VLLMScorer:
    """vLLM backend for scoring a FIXED sequence via prompt_logprobs (HPC only).

    Lazy-imports vllm so this module loads without it. Emits the same align-to-t
    convention as every other scorer.
    """

    def __init__(self, llm, config: BackendConfig) -> None:
        self.llm = llm
        self.config = config

    @classmethod
    def from_pretrained(cls, name: str, dtype: str = "bfloat16", **vllm_kwargs):
        from vllm import LLM
        llm = LLM(model=name, dtype=dtype, **vllm_kwargs)
        cfg = BackendConfig(engine="vllm", dtype=dtype, tokenizer_name=name)
        return cls(llm, cfg)

    def logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        ids = input_ids.tolist()
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        out = self.llm.generate(TokensPrompt(prompt_token_ids=ids), params)[0]
        return vllm_prompt_logprobs_to_aligned(out.prompt_logprobs, input_ids)


class TeacherVLLMScorer(VLLMScorer):
    def score_online(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.logprobs(input_ids)

    def precompute_and_save(self, traj: Trajectory, path: str,
                            store_dtype: str = "float32") -> torch.Tensor:
        logp = self.logprobs(traj.input_ids)
        _save_precomputed(traj, logp, self.config, path, store_dtype)
        return logp
