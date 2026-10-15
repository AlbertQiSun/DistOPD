"""Unified OPD trainer. ONE loop, ONE step function; a config flag picks the
rollout source. This is the trainer-level embodiment of the phase-1 invariant:
online and offline differ ONLY in where (trajectory, teacher_logp) come from.

  mode="online"  -> OnlineSource : sample pi_theta now, score teacher live.
  mode="offline" -> OfflineSource: sample once, precompute teacher to disk,
                    then read (trajectory, teacher_logp) back from disk.

Both sources hand the trainer the SAME pair type; `OPDTrainer.step` runs the
identical `opd_train_step` + optimizer update regardless of mode. The offline
source re-checks the backend at the disk boundary (gate 1) so a mismatched
precompute cannot silently feed the comparison.
"""
from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from rollout.rollout import generate_trajectories, generate_trajectory
from teacher.backend import BackendConfig, assert_same_backend, runtime_fingerprint
from teacher.scorer import Scorer, TeacherScorer
from opd.delta_cache_reader import DeltaCacheReader
from opd.delta_loss import ABLATIONS, delta_train_step
from opd.loss import DEFAULT_TAU
from opd.step import StepOutput, opd_train_step
from opd.trajectory import Trajectory, TerminationReason


def _git_head() -> Optional[str]:
    """12-hex commit of the repo this module runs from, or None.

    Tries the git binary, then falls back to READING .git/HEAD DIRECTLY. The fallback is
    not paranoia: the A100 cluster's compute nodes ship no git, and the 2026-08-19 offline real
    run (150 steps, 38400 records) stamped code_version=None on every row because of
    it -- discovered only after the run finished, when the artifact could not name its
    own commit. The manual read needs no binary: HEAD is either a raw hash (detached) or
    "ref: <path>", the ref lives as a loose file or a packed-refs line, and a worktree's
    .git is a one-line "gitdir:" pointer file.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import subprocess
    try:
        full = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL,
                                       timeout=10).strip()
        if full:
            return full[:12]
    except Exception:                      # noqa: BLE001 - no git binary is a state
        pass
    try:
        git_dir = os.path.join(root, ".git")
        if os.path.isfile(git_dir):        # worktree: .git is "gitdir: <real dir>"
            line = open(git_dir, encoding="utf-8").read().strip()
            git_dir = line.split(":", 1)[1].strip() if ":" in line else git_dir
        head = open(os.path.join(git_dir, "HEAD"), encoding="utf-8").read().strip()
        if not head.startswith("ref:"):
            return head[:12] or None       # detached HEAD holds the raw hash
        ref = head.split(None, 1)[1].strip()
        loose = os.path.join(git_dir, *ref.split("/"))
        if os.path.exists(loose):
            return open(loose, encoding="utf-8").read().strip()[:12] or None
        packed = os.path.join(git_dir, "packed-refs")
        if os.path.exists(packed):
            for pline in open(packed, encoding="utf-8"):
                pline = pline.strip()
                if pline.endswith(" " + ref):
                    return pline.split()[0][:12]
    except Exception:                      # noqa: BLE001 - truly off-git
        pass
    return None


class _DictRecord:
    """Wraps an already-dict-ified DiagnosticRecord so DiagnosticLogger keeps its
    `.to_dict()` contract while we attach the per-trajectory fields (micro / global_batch)
    that only exist once a step consumes more than one trajectory."""

    __slots__ = ("_d",)

    def __init__(self, d: dict) -> None:
        self._d = d

    def to_dict(self) -> dict:
        return self._d


@dataclass
class OPDConfig:
    mode: str = "online"          # "online" | "offline"
    tau: float = DEFAULT_TAU
    lr: float = 1e-3
    weight_decay: float = 0.0     # AdamW (Lightning OPD Table 6: 0.1)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999     # Table 6 OPD stage: 0.98
    steps: int = 4
    global_batch: int = 1          # trajectories consumed per OPTIMIZER step (grad accum).
                                   # 1 reproduces the pre-batching behaviour exactly.
    seed: int = 0
    max_turns: int = 4
    max_len: int = 4096
    top_k: int = 2
    precompute_dir: Optional[str] = None   # required for offline
    store_dtype: str = "float32"           # dtype teacher_logp is stored on disk
    resync_every: int = 1                  # online: push training weights -> inference
                                           # engine every N steps. 1 = fully on-policy
                                           # (main experiment). offline ignores it.
    loss: str = "opd"                     # "opd" (Lightning advantage, opd/loss.py) | "delta" (opd/delta_loss.py,
                                           # offline only: needs <precompute_dir>/{ref_topk,teacher_head})
    beta: float = 1.0                      # delta loss: q_hat ∝ pi_ref · exp(beta · delta)
    beta_end: Optional[float] = None       # delta loss: linear schedule beta -> beta_end over the run (None = constant)
    data_seed: Optional[int] = None        # offline: permute the cache's trajectory ORDER with this seed (None = cache order)
    ablation: str = "none"                 # delta loss: one of opd.delta_loss.ABLATIONS (teacher_head | no_bucket | no_anchor)
    channel: Optional[str] = None          # which configs/channels/<name>.yaml overlay this
                                           # run applied, stamped onto every diag record.
                                           # Before this, the overlay's identity lived ONLY
                                           # in a job-log print -- the artifact could not
                                           # say which channel produced it (2026-08-18: an
                                           # OOM postmortem had to grep SLURM logs to learn
                                           # whether autocast_cache/chunk_rows were live).


def _rollout(question: str, policy, retriever, tokenizer, cfg: OPDConfig,
             gold: Optional[str] = None) -> Trajectory:
    """Shared trajectory construction. Both sources build the trajectory the SAME
    way -- only teacher scoring (live vs disk) differs downstream."""
    return generate_trajectory(
        question, policy, retriever, tokenizer,
        max_turns=cfg.max_turns, max_len=cfg.max_len, top_k=cfg.top_k, seed=cfg.seed,
        gold_answer=gold,
    )


def _gold_for(golds, i: int) -> Optional[str]:
    return golds[i % len(golds)] if golds else None


MANIFEST_NAME = "manifest.json"


def precompute_offline(cfg: OPDConfig, questions, policy, retriever, tokenizer,
                       teacher: TeacherScorer, golds=None) -> str:
    """Sample + teacher-score every question ONCE, store to disk, write a manifest.

    This is the offline half of the single-path invariant, physically split out so a
    SEPARATE process (scripts/train.py --precompute-only, one GPU calling the served
    teacher) can produce the cache that the training process later only reads. The
    rollout + precompute_and_save loop is byte-for-byte the one OfflineSource used to
    run inline: same _rollout, same span construction (scored_indices inside
    precompute_and_save), same align-to-t. Decoupling changes WHEN the teacher is
    scored, never HOW -- so the offline teacher_logp on disk is the identical tensor
    online would compute live. The manifest records the backend fingerprint so the
    reader can enforce gate 1 at the disk boundary before any traj is loaded."""
    assert cfg.precompute_dir is not None, "offline precompute needs precompute_dir"
    os.makedirs(cfg.precompute_dir, exist_ok=True)
    # REMOVE ANY EXISTING MANIFEST FIRST. The manifest is written LAST, so without this a
    # re-run that crashes part-way leaves the PREVIOUS manifest pointing at traj files some
    # of which this run already overwrote -- a silent mix of two precomputes. Gate 1 would
    # not catch it: each file carries a backend fingerprint, and a re-run under the same
    # teacher has the same one. Deleting it up front makes a crashed run leave NO readable
    # cache, which fails loudly at the next read instead of feeding a blend.
    manifest = os.path.join(cfg.precompute_dir, MANIFEST_NAME)
    if os.path.exists(manifest):
        os.remove(manifest)
        print(f"[precompute] removed the previous manifest -- a crash from here on leaves "
              f"no readable cache rather than a half-overwritten one", flush=True)
    golds = list(golds) if golds else None
    paths: list[str] = []
    # BATCHED, in chunks the size of one optimizer step's rollout. Serially this loop was
    # the offline arm's whole budget: 38,400 trajectories one at a time measured ~60h
    # against ~16h batched -- the difference between fitting 48h and not. Nothing about the
    # cache changes: same generate_trajectories the online source drives (so the same
    # coroutine, per the batching-cannot-change-semantics design), same span construction,
    # same _save_precomputed. Only WHEN requests are issued differs.
    chunk = max(1, int(cfg.global_batch))
    for start in range(0, len(questions), chunk):
        qs = questions[start:start + chunk]
        chunk_golds = [_gold_for(golds, i) for i in range(start, start + len(qs))]
        chunk_paths = [os.path.join(cfg.precompute_dir, f"traj_{start + j:04d}.pt")
                       for j in range(len(qs))]
        # RESUMABLE: a chunk whose files all exist is not regenerated (a 38,400-trajectory
        # precompute is ~16 h; a walltime kill used to cost all of it). The manifest is still
        # written LAST, so a partial directory stays unreadable until the run completes.
        if chunk_paths and all(os.path.exists(pth) for pth in chunk_paths):
            paths.extend(os.path.basename(pth) for pth in chunk_paths)
            print(f"[precompute] {len(paths)}/{len(questions)} trajectories (chunk at {start} already on disk)", flush=True)
            continue
        trajs = generate_trajectories(qs, policy, retriever, tokenizer,
                                      max_turns=cfg.max_turns, max_len=cfg.max_len,
                                      top_k=cfg.top_k, seed=cfg.seed, golds=chunk_golds)
        assert len(trajs) == len(qs), "generate_trajectories returned a different count than questions"
        batch_fn = getattr(teacher, "precompute_and_save_batch", None)
        if batch_fn is not None:
            batch_fn(trajs, chunk_paths, store_dtype=cfg.store_dtype)
        else:
            # mock / in-process teachers keep only the one-at-a-time surface; identical
            # results, just slower -- they are never the 38,400-trajectory path.
            for traj, path in zip(trajs, chunk_paths):
                teacher.precompute_and_save(traj, path, store_dtype=cfg.store_dtype)
        paths.extend(os.path.basename(p) for p in chunk_paths)
        print(f"[precompute] {len(paths)}/{len(questions)} trajectories", flush=True)
    with open(manifest, "w", encoding="utf-8") as f:
        # `backend` says who SCORED. `sampler` says who ROLLED OUT -- a different process
        # on a different card, and the arms must match on BOTH. The teacher's hardware
        # arrives via /health; the sampler's is this process's own, and nothing else records
        # it, so a cache sampled on one channel and consumed by an arm on another would look
        # perfectly consistent.
        json.dump({"backend": asdict(teacher.config), "paths": paths, "n": len(paths),
                   "sampler": runtime_fingerprint()}, f)
    return manifest


class OnlineSource:
    """Sample the trajectory now, score the teacher live."""

    def __init__(self, questions, policy, retriever, tokenizer,
                 teacher: TeacherScorer, cfg: OPDConfig, golds=None) -> None:
        self.questions = list(questions)
        self.golds = list(golds) if golds else None
        self.policy = policy
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.cfg = cfg
        self.last_sync_step = 0        # step at which the inference engine last pulled
                                       # the training weights (drives staleness)

    def resync(self, student, step: int) -> None:
        """Push the CURRENT training weights into the sampling policy's inference
        engine, so the next rollouts are drawn from the up-to-date pi_theta. This
        is the single thing that makes 'online' actually on-policy; a no-op here
        would silently turn online into another frozen (offline-like) run."""
        self.policy.sync_from(student)
        self.last_sync_step = step

    def get(self, step: int) -> tuple[Trajectory, torch.Tensor]:
        return self.get_batch(step, 1)[0]

    def get_batch(self, step: int, n: int) -> list:
        """Sample and score n trajectories for ONE optimizer step.

        Both halves are batched: the rollouts go through generate_trajectories (gathered
        rounds, one coroutine per trajectory) and the teacher scores them through
        logprobs_batch (grouped by token budget). Neither changes a single trajectory --
        the rollout coroutine is the same one the size-1 path drives, and batch
        composition was certified numerically inert for teacher scoring.
        """
        idx = [(step * n + j) % len(self.questions) for j in range(n)]
        qs = [self.questions[i] for i in idx]
        golds = [_gold_for(self.golds, i) for i in idx]
        trajs = generate_trajectories(qs, self.policy, self.retriever, self.tokenizer,
                                      max_turns=self.cfg.max_turns, max_len=self.cfg.max_len,
                                      top_k=self.cfg.top_k, seed=self.cfg.seed, golds=golds)
        batch_fn = getattr(self.teacher, "logprobs_batch", None)
        if batch_fn is not None:
            logps = batch_fn([(t.input_ids, None) for t in trajs])
        else:
            # mock / in-process teachers keep the one-at-a-time surface; scoring one at a
            # time is behaviour-identical, only slower.
            logps = [self.teacher.score_online(t.input_ids) for t in trajs]
        return list(zip(trajs, logps))


class OfflineSource:
    """Precompute (trajectory + teacher_logp) to disk once, then read back.

    The read path reconstructs the trajectory from the stored token_type (single
    source of truth) and asserts the stored backend matches the expected online
    backend -- gate 1 enforced exactly where online and offline could diverge.
    """

    def __init__(self, questions, policy, retriever, tokenizer,
                 teacher: TeacherScorer, cfg: OPDConfig,
                 expected_backend: BackendConfig, golds=None) -> None:
        assert cfg.precompute_dir is not None, "offline mode needs precompute_dir"
        os.makedirs(cfg.precompute_dir, exist_ok=True)
        self.expected_backend = expected_backend
        self.last_sync_step = 0        # frozen: rollouts sampled ONCE at precompute,
                                       # never resynced -> staleness == step by design
        manifest = os.path.join(cfg.precompute_dir, MANIFEST_NAME)
        if not os.path.exists(manifest):
            # No cache yet: precompute in THIS process (first-signal single job, mock
            # tests). The decoupled path (--precompute-only in a separate process) writes
            # the manifest first, so a training process then takes the read-only branch.
            precompute_offline(cfg, questions, policy, retriever, tokenizer, teacher,
                               golds=golds)
        with open(manifest, "r", encoding="utf-8") as f:
            man = json.load(f)
        # disk boundary, enforced BEFORE any traj load: refuse a cache produced by a
        # different backend than the online run would use (gate 1, cross-process form).
        assert_same_backend(self.expected_backend, BackendConfig(**man["backend"]))
        self.paths = [os.path.join(cfg.precompute_dir, p) for p in man["paths"]]
        # DATA ORDER. The cache is walked in manifest order; a data_seed permutes that order
        # once (a seed-2 arm sees the same trajectories in a different order). cache_index
        # stays the RECORD's index, so the delta caches keyed by it are unaffected.
        import numpy as _np
        self.order = (_np.random.default_rng(int(cfg.data_seed)).permutation(len(self.paths))
                      if getattr(cfg, "data_seed", None) is not None else _np.arange(len(self.paths)))
        # WHICH MACHINE ROLLED THIS CACHE OUT. Not asserted here: this process does not
        # sample, so it has nothing to compare against. It is carried onto every offline
        # diagnostic row instead, because the cross-arm check can only happen where both
        # arms are visible -- scripts/check_channel_parity.py, at analysis time.
        self.sampler_runtime = man.get("sampler")

    def resync(self, student, step: int) -> None:
        """No-op by design: offline rollouts are frozen at precompute time, so the
        sampling policy is never resynced. Keeping the method (vs omitting it) lets
        the trainer stay mode-agnostic -- it calls resync on either source."""
        return

    def get(self, step: int) -> tuple[Trajectory, torch.Tensor]:
        idx = int(self.order[step % len(self.paths)])
        path = self.paths[idx]
        rec = TeacherScorer.load(path)
        # disk boundary: refuse a precompute scored by a different backend
        assert_same_backend(self.expected_backend, rec["backend"])
        # Restore the confound-#2 fields too. They are Trajectory FIELDS (not meta), so
        # omitting them here silently defaulted every offline record to reason=null /
        # num_search_calls=0 while online reported the real values -- a fake online-vs-offline
        # gap in the termination diagnostic. .get() keeps PRE-FIX caches readable (they lack
        # the keys and simply fall back to today's defaults), so this is backward compatible;
        # the values only become real once the cache is regenerated by a post-fix precompute.
        term = rec.get("terminated")
        meta = dict(rec.get("meta", {}))
        meta["cache_index"] = idx                        # which record: the delta caches are keyed by it
        traj = Trajectory(rec["input_ids"].long(), rec["token_type"].long(),
                          terminated=TerminationReason(term) if term is not None else None,
                          num_search_calls=int(rec.get("num_search_calls", 0)),
                          meta=meta)  # frozen context travels with the data
        return traj, rec["teacher_logp"]

    def get_batch(self, step: int, n: int) -> list:
        """n cached trajectories for ONE optimizer step. Reading is already cheap, so this
        is a loop over the SAME get() the size-1 path uses -- the index arithmetic must
        match OnlineSource's so both arms walk their data in the same order."""
        return [self.get(step * n + j) for j in range(n)]


def build_source(cfg: OPDConfig, questions, policy, retriever, tokenizer,
                 teacher: TeacherScorer, golds=None):
    """Config flag -> source. The trainer loop is oblivious to which one it got."""
    if cfg.mode == "online":
        return OnlineSource(questions, policy, retriever, tokenizer, teacher, cfg, golds=golds)
    if cfg.mode == "offline":
        return OfflineSource(questions, policy, retriever, tokenizer, teacher, cfg,
                             expected_backend=teacher.config, golds=golds)
    raise ValueError(f"unknown mode {cfg.mode!r} (expected 'online' or 'offline')")


def snapshot_ref(student: Scorer) -> Scorer:
    """Frozen pi_ref = student at training start. Diagnostics measure policy drift
    (KL, importance weight) against THIS snapshot, so it must be taken BEFORE the
    first optimizer step.

    The ref must be the SAME scorer class as the student: on a real run the student
    is an HFScorer (batched forward + device handling), and a base Scorer wrapping an
    HF CausalLM would call it wrong (no batch dim; CausalLMOutput is not a logits
    tensor). deepcopy preserves device, so the ref lives wherever the student does."""
    ref_lm = copy.deepcopy(student.lm)
    for p in ref_lm.parameters():
        p.requires_grad_(False)
    # EVAL, EXPLICITLY. deepcopy carries the student's mode across, and the student is now
    # left in train() by enable_grad_checkpointing. Scorer.logprobs/full_logprobs do set the
    # mode per call, so this is belt-and-braces rather than a fix -- but the ref is the one
    # model that must NEVER train, and inheriting train() by accident is the kind of thing
    # that stops being harmless the moment someone calls the module directly.
    ref_lm.eval()
    # ON THE HOST. The reference is a full copy of the student -- fp32, ~15 GiB for the 4B --
    # and it is untouched during forward/backward, which is precisely when the card is
    # tightest. OPDTrainer.step brings it back for the diagnostic pass and returns it after.
    #
    # KEPT IN THE STUDENT'S PRECISION, not narrowed to bf16 to save host RAM. The KL this
    # measures is ~1e-4 between two nearly identical distributions; bf16 weights carry ~4e-3
    # relative error, which would swamp the signal with rounding noise and make the drift
    # diagnostic report its own quantisation.
    ref_lm.to("cpu")
    return type(student)(ref_lm, requires_grad=False, config=student.config,
                         autocast_dtype=getattr(student, "autocast_dtype", None))


def build_adamw(params, cfg: OPDConfig):
    """AdamW on torch's DEFAULT implementation. One construction point, no kernel selection.

    THIS BRIEFLY REQUESTED fused=True AND THAT WAS REVERTED. The record, so nobody re-adds it
    without a reason:

    The fused kernel was adopted to save ~7.5 GiB. torch's default CUDA path is `foreach`,
    which batches the update across parameters and materialises full-parameter-sized
    temporaries to do it; for a 4B model in fp32 that is real memory, and at the time the peak
    read 75.0 GiB on a 95 GiB card with the longest real trajectory OOM-ing.

    Both halves of that justification then failed:

    1. THE MEMORY PRESSURE WAS A BUG, NOT A CONSTRAINT. The 75.0 GiB peak came from
       HFScorer.logprobs training in eval mode, which stopped gradient checkpointing from
       engaging (see teacher/scorer.py). With that fixed the peak is 71.6 GiB of 95.1 at the
       longest real trajectory, so the default lands near 79 GiB and fits with room. Nothing
       needs the 7.5 GiB.
    2. THE EQUIVALENCE WAS NEVER TESTED. The docstring had claimed the numerics were identical
       either way. They are the same elementwise arithmetic, but different kernels can round
       differently and no comparison was ever run.

    A numerical deviation that solves nothing should be removed, not defended with a new gate
    -- and the gate would cost a GPU allocation to answer a question that disappears on
    reverting. It also matters that this project's claim is a faithful reproduction of
    Lightning OPD: the optimizer is exactly where a self-introduced difference is least
    welcome, having already been bitten by copying the learning rate without the optimizer
    precision that makes it work.

    If throughput ever becomes the binding constraint, come back and add the comparison first.
    """
    return torch.optim.AdamW(
        list(params), lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay)


class OPDTrainer:
    """Single training loop. Same step + optimizer update for online and offline.

    Diagnostics are strictly optional and additive: pass a DiagnosticRunner + logger
    to emit one record per step. They run AFTER the optimizer step (so KL / importance
    weight reflect the just-updated pi_theta vs the frozen pi_ref) and never touch the
    training numerics."""

    def __init__(self, student, source, cfg: OPDConfig,
                 diagnostics=None, logger=None) -> None:
        assert student.requires_grad, "student scorer must carry grad for training"
        self.student = student
        self.source = source
        self.cfg = cfg
        self.diagnostics = diagnostics
        self.logger = logger
        # Stamped once, attached to every record. Per-row rather than a run header because
        # a row that cannot say where it came from is a row a future comparison has to take
        # on trust -- and the comparison this project makes is precisely one that only holds
        # if both arms ran on the same channel.
        self.runtime = runtime_fingerprint()
        # One-slot box for the PREVIOUS step's diagnostic wall clock -- see step().
        self._diag_seconds_prev = [None]
        # Capacity + allocator config for the per-record `memory` block. Stamped per row
        # (not a run header) for the same reason as `runtime` above. The allocator env var
        # is provenance, NOT part of the numeric channel -- it moves where tensors live,
        # not what kernels compute -- so it is deliberately absent from the parity FIELDS.
        _dev = next(student.lm.parameters()).device
        self.gpu_total_gib = (
            torch.cuda.get_device_properties(_dev).total_memory / 2**30
            if _dev.type == "cuda" else None)
        self.cuda_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        # WHICH CODE produced these numbers. Became numerics-relevant on 2026-08-19: the
        # chunked-lm_head path deviates from the retired full-logits path by ~1 ulp
        # (opd/masking.py, head_aligned_logprobs), so two arms trained under different
        # commits are two arms trained under different kernels -- exactly the confound
        # class the channel fingerprint exists for, one level up. None when git is
        # unavailable, recorded as itself.
        self.code_version = _git_head()
        self.optimizer = build_adamw(student.lm.parameters(), cfg)
        # THE DELTA ARM reads two more caches beside the precompute; everything else in this
        # class is untouched, and the online arm cannot select it (no teacher head is scored live).
        self.delta_reader = None
        if getattr(cfg, "loss", "opd") == "delta":
            if cfg.mode != "offline":
                raise ValueError("loss=delta is offline-only: the teacher head is a precomputed cache")
            if cfg.ablation not in ABLATIONS:
                raise ValueError(f"unknown delta ablation {cfg.ablation!r}; one of {sorted(ABLATIONS)}")
            self.delta_reader = DeltaCacheReader(cfg.precompute_dir, tau=cfg.tau, support=ABLATIONS[cfg.ablation]["support"])

    def beta_at(self, step_idx: int) -> float:
        """The delta arm's beta at this optimizer step: constant, or linear from beta to beta_end
        over the run's steps (the annealing arm)."""
        b0, b1 = float(self.cfg.beta), getattr(self.cfg, "beta_end", None)
        if b1 is None:
            return b0
        n = max(1, int(self.cfg.steps) - 1)
        return b0 + (float(b1) - b0) * min(step_idx, n) / n

    def step(self, step_idx: int) -> StepOutput:
        # resync BEFORE sampling so rollouts are drawn from the up-to-date pi_theta.
        # online: real weight push (every resync_every steps); offline: no-op. staleness
        # is then the gap between the sampling weights and the current training weights.
        if step_idx % self.cfg.resync_every == 0:
            self.source.resync(self.student, step_idx)

        n = max(1, int(self.cfg.global_batch))
        # TIME THE ROLLOUT. Two uses, both load-bearing:
        # (1) EARLY WARNING. The wall-clock projection for a real run linearly extrapolates
        #     seconds-per-trajectory from a short probe, which assumes per-trajectory cost
        #     is constant across training. The central hypothesis says otherwise: online
        #     degradation means more spinning, longer trajectories, more expensive rollouts,
        #     so the back half should be slower than the front. Watching this per step beats
        #     discovering it when SLURM kills the job.
        # (2) IT IS ITSELF A POISONING OBSERVABLE. Trajectories lengthening IS a degradation
        #     symptom, so this should move together with rising KL and falling hit@k -- a
        #     third, independent read on the same effect, free.
        _t0 = time.perf_counter()
        pairs = self.source.get_batch(step_idx, n)
        rollout_seconds = time.perf_counter() - _t0
        staleness = step_idx - self.source.last_sync_step

        # GRADIENT ACCUMULATION, SEQUENCE-WEIGHTED. Each trajectory's loss is already a
        # per-token mean over its own scored tokens (opd/loss.py), and dividing by n makes
        # the optimizer step consume the MEAN of those per-trajectory means -- every
        # trajectory weighted equally regardless of length. That is what the Lightning OPD
        # reference implementation does (slime's default sum_of_sample_mean, then a divide
        # by a SAMPLE count); see the normalisation section of opd/loss.py for the full
        # evidence chain and for why we do not follow veRL/Search-R1's token-mean.
        # PER-TRAJECTORY forward/backward timing, not just the step total. The student pass
        # is serial over the batch and its cost scales with trajectory length, which spans
        # ~18x here -- so a step mean hides the tail, and the tail is where the walltime
        # risk lives. Each trajectory's own record carries its own number, next to the
        # n_scored that explains it.
        #
        # CUDA is ASYNCHRONOUS: without a sync, perf_counter measures enqueue time, not
        # work, and the cost would appear to land on whichever call finally blocks. Sync so
        # the attribution is real. (The sync serialises CPU/GPU overlap slightly; for a
        # budget projection, correct attribution is worth more than that.)
        _sync = (torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None))

        # PER-TRAJECTORY PEAK MEMORY, next to the per-trajectory time and for the same
        # reason: the 2026-08-18 online OOM died asking for 3.96 GiB with 3.95 free --
        # a 10 MiB miss on a ~74 GiB-resident card -- and nothing on disk could say how
        # much headroom the surviving steps had. Peak-vs-capacity per trajectory, written
        # into each record beside the length that drives it, turns the next OOM from a
        # guess into a curve. reset+read brackets ONE trajectory, so each entry includes
        # the persistent base (weights + optimizer states + grads) plus that trajectory's
        # own activation peak -- headroom is gpu_total_gib minus this number, directly.
        _dev = next(self.student.lm.parameters()).device
        _mem = _dev.type == "cuda"

        _t1 = time.perf_counter()
        # VANILLA-KERNEL BRACKET around everything that trains. In the online process the
        # batch-invariant overrides are live for the policy; they are inference-only
        # (aten::linear_backward has no CUDA kernel) and the offline arm trains vanilla, so
        # the training math must leave them and provably come back (the exit re-enables and
        # verifies). No-op in the offline process -- same code path, both arms.
        from rollout.engine import student_vanilla_math
        with student_vanilla_math():
            self.optimizer.zero_grad(set_to_none=True)
            outs, per_traj_seconds, per_traj_peak_gib = [], [], []
            for traj, teacher_logp in pairs:
                _tj = time.perf_counter()
                if _mem:
                    torch.cuda.reset_peak_memory_stats(_dev)
                if self.delta_reader is not None:
                    idx = traj.meta.get("cache_index")
                    if idx is None:
                        raise RuntimeError("delta loss needs the offline source's cache_index (OfflineSource.get stamps it)")
                    cache = self.delta_reader.get(int(idx), teacher_logp)
                    ab = ABLATIONS[self.cfg.ablation]
                    out = delta_train_step(traj, teacher_logp, cache, self.student, beta=self.beta_at(step_idx),
                                           include_bucket=ab["include_bucket"], anchor=ab["anchor"])
                else:
                    out = opd_train_step(traj, teacher_logp, self.student, tau=self.cfg.tau)
                (out.loss / n).backward()
                _sync()
                per_traj_seconds.append(time.perf_counter() - _tj)
                per_traj_peak_gib.append(
                    torch.cuda.max_memory_allocated(_dev) / 2**30 if _mem else None)
                outs.append((traj, teacher_logp, out))
            # optimizer.step() has its own temporaries; measured separately so a step-time
            # OOM and a backward-time OOM are distinguishable in the record.
            if _mem:
                torch.cuda.reset_peak_memory_stats(_dev)
            self.optimizer.step()
            _sync()
        opt_step_peak_gib = (
            torch.cuda.max_memory_allocated(_dev) / 2**30 if _mem else None)
        train_seconds = time.perf_counter() - _t1
        # RELEASE THE GRADIENT BUFFER before diagnostics run. set_to_none frees a full
        # parameter-sized fp32 tensor (~15 GiB for the 4B student); zeroing in place would
        # keep it resident through the diagnostic pass, which is the other moment this card
        # is tight (pi_ref pages back in, and kl_importance materialises [N, V] twice).
        self.optimizer.zero_grad(set_to_none=True)

        if self.diagnostics is not None and self.logger is not None:
            # PAGE pi_ref BACK ONTO THE CARD for this block only. It is idle during
            # forward/backward and would otherwise hold ~15 GiB there for nothing. One
            # transfer per OPTIMIZER step, not per trajectory -- sub-second on this link,
            # against 150 steps.
            # THE DIAGNOSTIC PASS WAS OUTSIDE EVERY TIMER. It runs two extra full
            # forwards per trajectory, so a budget projection built on train_seconds alone
            # was measuring the cheaper half of the step -- and would have been used to
            # decide whether the real experiment fits 48h. Timed now, and written into the
            # SAME record, so it cannot be silently omitted again.
            _diag_box = self._diag_seconds_prev
            _ref = getattr(self.diagnostics, "ref", None)
            _dev = next(self.student.lm.parameters()).device
            if _ref is not None and hasattr(_ref, "to"):
                _ref.to(_dev)
            _t2 = time.perf_counter()
            try:
                # Same current-device guard as opd_train_step, same reason: the diagnostic
                # pass runs scorer forwards AND softmax/mean math on the student's card, and
                # under the batch-invariant overrides those launch Triton on the thread's
                # CURRENT device -- cuda:0 in the online arm, where these tensors are not.
                from opd.step import _student_device
                from rollout.engine import student_vanilla_math
                # the diagnostics compute the CURVES the paper compares; they must run under
                # the same vanilla kernels in both arms, or the online-offline difference
                # would carry a kernel term. Same bracket as the training block above.
                with student_vanilla_math(), _student_device(self.student):
                    self._log_diagnostics(step_idx, outs, staleness, n,
                                          rollout_seconds, train_seconds, per_traj_seconds,
                                          per_traj_peak_gib, opt_step_peak_gib,
                                          diag_seconds_ref=_diag_box)
            finally:
                if _ref is not None and hasattr(_ref, "to"):
                    _ref.to("cpu")
                _sync()
                # Filled AFTER the records are written, so it necessarily lands one step
                # late. Reported under a name that says so rather than as this step's own.
                _diag_box[0] = time.perf_counter() - _t2
        return outs[-1][2]

    def _log_diagnostics(self, step_idx, outs, staleness, n,
                         rollout_seconds, train_seconds, per_traj_seconds,
                         per_traj_peak_gib, opt_step_peak_gib,
                         diag_seconds_ref=None) -> None:
        """Split out of step() so pi_ref's page-in/page-out can wrap it in a try/finally --
        a leaked 15 GiB reference on the card is not a thing to leave to control flow."""
        # ONE RECORD PER TRAJECTORY, not per optimizer step. Collapsing 256 trajectories
        # into a step-level mean would discard the distribution -- and the distribution
        # is the point: termination reasons, retrieval hits and the boundary |A_t|
        # buckets are per-sample quantities that cannot be rebuilt from an average
        # without re-running generation (the the project conventions persistence rule).
        for micro, (traj, _tlp, out) in enumerate(outs):
            rec = self.diagnostics.run(step_idx, self.cfg.mode, traj, out, self.student,
                                       staleness=staleness,
                                       resync_every=self.cfg.resync_every)
            d = rec.to_dict() if hasattr(rec, "to_dict") else rec
            if isinstance(d, dict):
                d["micro"] = micro          # position within the optimizer step
                d["global_batch"] = n
                d["runtime"] = self.runtime
                # Which channel overlay shaped this run (autocast_cache, chunk_rows,
                # max_len...). None = no overlay was applied, recorded as itself.
                d["channel"] = self.cfg.channel
                # Which COMMIT computed these numbers -- numerics-relevant since the
                # chunked-lm_head change (~1 ulp vs the full-logits era); arms must
                # never be paired across it. check_channel_parity.py enforces.
                d["code_version"] = self.code_version
                # THE DELTA ARM's per-trajectory numbers: soft / hard loss means, and the
                # argmax agreement with the teacher on the soft positions (student now, and
                # pi_ref from the cache) -- one record per trajectory, like everything else.
                info = getattr(out, "info", None)
                if isinstance(info, dict):
                    d["delta"] = {k: info[k] for k in ("n_soft", "n_hard", "loss_soft", "loss_hard",
                                                       "argmax_student_eq_teacher", "argmax_ref_eq_teacher") if k in info}
                    d["delta"]["beta"] = self.beta_at(step_idx)
                    d["delta"]["ablation"] = self.cfg.ablation
                    d["delta"]["loss"] = float(out.loss.detach())
                # OFFLINE rows carry a SECOND machine: the one that sampled the cache, which
                # is a different process (and possibly a different channel) from the one
                # training now. Online has no such split -- it samples and trains in one
                # process -- so the field is simply absent there, which is itself the honest
                # record of the difference.
                cache_sampler = getattr(self.source, "sampler_runtime", None)
                if cache_sampler is not None:
                    d["cache_sampler_runtime"] = cache_sampler
                # step-level wall clock, repeated on each of the step's records so a
                # per-trajectory reader never has to join against another file
                d["timing"] = {
                    # step-level, repeated on each record so a per-trajectory reader
                    # never has to join against another file
                    "rollout_seconds": rollout_seconds,
                    "train_seconds": train_seconds,
                    "rollout_seconds_per_trajectory": rollout_seconds / n,
                    # THIS trajectory's own forward+backward -- the distribution across
                    # records is what a step mean would hide
                    "train_seconds_this_trajectory": per_traj_seconds[micro],
                    # PREVIOUS step's diagnostic pass -- this step's is still running as
                    # this record is written. None on step 0. Named for what it is: the
                    # diagnostic pass sat outside every timer, so a budget projection built
                    # on train_seconds alone was projecting half the step.
                    "diag_seconds_prev_step":
                        None if diag_seconds_ref is None else diag_seconds_ref[0],
                }
                # Peak-vs-capacity, per trajectory, beside the termination.length that
                # drives it. None (keys still present) on CPU/mock runs.
                d["memory"] = {
                    "peak_gib_this_trajectory": per_traj_peak_gib[micro],
                    "peak_gib_optimizer_step": opt_step_peak_gib,
                    "gpu_total_gib": self.gpu_total_gib,
                    "cuda_alloc_conf": self.cuda_alloc_conf,
                }
                self.logger.log(_DictRecord(d))
            else:                            # pragma: no cover - defensive
                self.logger.log(rec)
        return outs[-1][2]

    def train(self):
        """Yield each optimizer step's output AS IT COMPLETES.

        Deliberately a generator, not a list comprehension. The old eager version ran every
        step before returning, so the caller's per-step print only fired after training had
        finished -- the job log showed a burst of N lines at the end and nothing during the
        run. That is not merely inconvenient: it made a correctly interleaved run LOOK like
        "all rollouts first, then all training", and cost a real round of doubt about
        whether the online arm was on-policy at all.

        Consequence for callers: a bare `trainer.train()` no longer runs anything. Consume
        it (`for out in trainer.train()` / `list(...)`) -- the lazy version fails loudly on
        the assertions that follow rather than silently doing nothing.
        """
        for i in range(self.cfg.steps):
            yield self.step(i)
