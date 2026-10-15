"""THE real-run entrypoint. ONE code path; the config (and --dry_run) only choose
which SEAM implementations feed the trainer.

`--dry_run` flips the FACTORIES (tokenizer / policy / retriever / teacher / student /
dataset) to the mock/tiny CPU versions. Everything from `build_source(...)` downward --
snapshot_ref, DiagnosticRunner, OPDTrainer.train() -- is byte-identical between dry-run
and a real HPC run. The dry-run is therefore the same train.py proving the wiring, which
is exactly what the sbatch preflights invoke. The seams are real objects, never stubbed
green: where a seam can only run on HPC (vLLM, live retriever) the dry-run substitutes
the SOURCE of data (mock), not the code path.

    python scripts/train.py --config configs/opd_qwen3.yaml --mode online --dry_run
    python scripts/train.py --config configs/opd_qwen3.yaml --mode offline   # real (HPC)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd.config import load_config
from opd.diag_logger import DiagnosticLogger
from opd.diagnostics import DiagnosticRunner
from opd.trainer import (
    MANIFEST_NAME,
    OPDConfig,
    OPDTrainer,
    build_source,
    precompute_offline,
    snapshot_ref,
)

# deterministic mock corpus/questions for the CPU dry-run (no files, no downloads)
_DRY_CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
    "Berlin is the capital of Germany.",
]
_DRY_QUESTIONS = ["What is the capital of France?", "How tall is the Eiffel Tower?"]
_DRY_GOLDS = [["Paris"], ["330 metres", "330 m"]]   # gold is a LIST of aliases (OR)


# ---------------------------------------------------------------------------
# factories -- the ONLY thing dry_run vs real swaps
# ---------------------------------------------------------------------------
def _assert_disjoint_from_sft(cfg: dict, questions: list[str]) -> None:
    """Guard: the OPD training slice must not contain any question pi_ref was SFT'd on.

    OPD trains on an UNUSED slice of the SFT *train* file (data.offset past the prefix SFT
    consumed), which keeps the dev split whole so the Stage-3 pi_ref dev numbers stay
    directly reusable as the eval baseline. The offset makes that disjoint by construction
    -- but only if the offset is right, so this re-derives the fact instead of trusting it.

    IMPORTANT -- the SFT-consumed prefix is reconstructed from `sft.data.sources` (the SAME
    config field that actually drove SFT), NOT from `data.offset`. Reconstructing with
    `limit=offset` would be SELF-REFERENTIAL: any contiguous offset trivially fails to
    intersect the first `offset` rows, so a too-SMALL offset would sail through a check that
    looks rigorous. Deriving the count from the SFT config is what makes this a real gate --
    it fails loud on a wrong offset AND on duplicate question strings straddling the
    boundary (a contiguous slice is disjoint by ROW, not by CONTENT).
    """
    from dataio.dataset import load
    from opd.diagnostics import normalize_answer

    path = cfg["data"]["path"]
    sources = ((cfg.get("sft") or {}).get("data") or {}).get("sources") or []
    matches = [s for s in sources if s.get("path") == path]
    if not matches:
        # Legitimate (e.g. pointing OPD at dev, or a dataset SFT never touched). Say so
        # OUT LOUD: a check that can silently no-op is not a check.
        print(f"[data] disjointness: {path} is not an SFT source -> no SFT overlap possible")
        return
    limits = [s.get("limit") for s in matches]
    if any(x is None for x in limits):
        raise ValueError(
            f"{path} is an SFT source with limit=None -- SFT consumed the WHOLE file, so no "
            "offset can produce an unused slice. Point data.path at a different file."
        )
    sft_limit = max(limits)
    sft_rows = load(path, name=cfg["data"].get("name"), limit=sft_limit)
    sft_q = {normalize_answer(r["question"]) for r in sft_rows}
    opd_q = {normalize_answer(q) for q in questions}
    overlap = sft_q & opd_q
    if overlap:
        raise ValueError(
            f"OPD training slice OVERLAPS pi_ref's SFT set: {len(overlap)} shared question(s) "
            f"(data.offset={cfg['data'].get('offset', 0)}, SFT consumed {sft_limit} valid rows "
            f"of {path}). pi_ref memorized these, so the rollout states would not be genuine "
            f"search. Raise data.offset to at least {sft_limit}. Example: {sorted(overlap)[0][:80]!r}"
        )
    print(f"[data] disjointness OK: {len(opd_q)} OPD questions vs the {len(sft_rows)} "
          f"SFT-consumed rows of {path} -- intersection empty")


def make_dataset(cfg: dict, dry_run: bool):
    """-> (questions: list[str], golds: list[list[str]]). golds are alias lists."""
    if dry_run:
        return list(_DRY_QUESTIONS), list(_DRY_GOLDS)
    from dataio.dataset import load
    offset = cfg["data"].get("offset", 0)
    rows = load(cfg["data"]["path"], name=cfg["data"].get("name"),
                limit=cfg["data"].get("limit"), offset=offset)
    if not rows:
        raise ValueError(f"no usable rows in {cfg['data']['path']} (need question + gold)")
    questions = [r["question"] for r in rows]
    _assert_disjoint_from_sft(cfg, questions)
    return questions, [r["gold_answers"] for r in rows]


CHANNEL_ALLOWLIST = {
    ("model", "autocast_cache"),
    ("model", "student_device"),
    ("opd", "logprob_chunk_rows"),
    ("rollout", "max_len"),
    ("teacher", "gpu_memory_utilization"),
}


def apply_channel(cfg: dict, channel: str) -> dict:
    """Merge configs/channels/<channel>.yaml over cfg. Returns cfg (mutated).

    A CHANNEL FILE MAY NOT CHANGE THE SCIENCE. Everything it is allowed to set is in
    CHANNEL_ALLOWLIST, and anything else is refused before the run starts -- without that
    guard there would be two places a run's meaning is decided and the second would win
    quietly. steps, lr, global_batch, epochs, tau, seed and the data slice live in the base
    config and nowhere else.

    What a channel legitimately carries is what the hardware forces: how much of the card the
    teacher may take, the longest trajectory that fits, whether the autocast weight-cast cache
    is affordable, and how many rows a logprob chunk holds. Those are accommodations, not
    choices about the experiment.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs", "channels", f"{channel}.yaml")
    if not os.path.exists(path):
        avail = sorted(f[:-5] for f in os.listdir(os.path.dirname(path)) if f.endswith(".yaml"))
        raise ValueError(f"unknown channel {channel!r} -- have {avail}")
    overlay = load_config(path)

    name = overlay.pop("name", channel)
    offenders = []
    for section, values in overlay.items():
        if not isinstance(values, dict):
            offenders.append(section)
            continue
        for key in values:
            if (section, key) not in CHANNEL_ALLOWLIST:
                offenders.append(f"{section}.{key}")
    if offenders:
        raise ValueError(
            f"channel {name!r} tries to set {offenders}, which is not a hardware "
            f"accommodation. A channel file may only set "
            f"{sorted('.'.join(k) for k in CHANNEL_ALLOWLIST)} -- the science lives in the "
            f"base config, and letting a channel override it would put a run's meaning in two "
            f"places with the second one winning silently.")

    for section, values in overlay.items():
        cfg.setdefault(section, {}).update(values)
    cfg["channel"] = name
    print(f"[channel] {name}: " + "  ".join(
        f"{sec}.{k}={v}" for sec, vals in overlay.items() for k, v in vals.items()), flush=True)
    return cfg


def assert_epochs_declared(cfg: dict, opd_cfg, n_questions: int) -> int:
    """`steps * global_batch` draws over `n_questions` questions must equal a DECLARED
    `data.epochs`. Returns the epoch count.

    THIS GUARDS A CONFOUND, NOT A TYPO. Both sources walk `questions[i % n]`, but they do
    different things on a revisit: ONLINE re-samples the policy and gets a FRESH trajectory,
    OFFLINE re-reads the same cached one. So with L questions and D = steps * global_batch
    draws, online sees D distinct trajectories and offline sees L, each repeated D/L times.
    That ratio IS a second independent variable -- data diversity -- sitting next to the one
    the experiment is about. At the shipped smoke setting (L=64, D=38400) it was 600.

    epochs=1 (L == D) makes it 1: each question drawn once per arm, the only remaining
    difference being when and from which policy the trajectory was sampled. It also pairs the
    arms -- step k consumes the same questions on both sides, which is a stronger comparison
    than two independent draws.

    Any other value is allowed but must be WRITTEN DOWN, because a run at epochs=160 is a
    legitimate smoke test and an illegitimate experiment, and nothing else distinguishes them.
    Deriving `limit` silently would hide exactly the number a reader needs.
    """
    draws = int(opd_cfg.steps) * max(1, int(opd_cfg.global_batch))
    declared = cfg.get("data", {}).get("epochs")
    if n_questions <= 0:
        raise AssertionError("dataset is empty -- nothing to declare epochs over")
    actual = draws / n_questions
    if declared is None:
        raise AssertionError(
            f"data.epochs is not declared. This run makes {draws} draws "
            f"(steps={opd_cfg.steps} x global_batch={opd_cfg.global_batch}) over "
            f"{n_questions} questions = {actual:g} epochs. Offline re-reads the SAME cached "
            f"trajectory on every revisit while online re-samples a fresh one, so that number "
            f"is the data-diversity ratio between the arms -- set data.epochs to it "
            f"deliberately (1 for the real experiment) rather than inheriting it by accident.")
    if abs(actual - float(declared)) > 1e-9:
        raise AssertionError(
            f"data.epochs={declared} but this run makes {draws} draws over {n_questions} "
            f"questions = {actual:g} epochs. steps x global_batch must equal limit x epochs.")
    if float(declared) != 1.0:
        print(f"[data] epochs={declared} -- offline reuses each cached trajectory "
              f"{declared}x while online resamples. NOT an apples-to-apples run.", flush=True)
    else:
        print(f"[data] epochs=1: {n_questions} questions, each drawn once per arm", flush=True)
    return int(actual) if float(actual).is_integer() else actual


def make_tokenizer(cfg: dict, dry_run: bool):
    if dry_run:
        from envs.tokenizer import MockTokenizer
        return MockTokenizer(vocab_size=8192)
    from transformers import AutoTokenizer
    from envs.hf_tokenizer import HFTokenizerAdapter
    return HFTokenizerAdapter(AutoTokenizer.from_pretrained(cfg["model"]["student"]))


def make_retriever(cfg: dict, dry_run: bool):
    if dry_run:
        from envs.retriever import MockRetriever
        return MockRetriever(_DRY_CORPUS, seed=cfg["run"]["seed"])
    from envs.http_retriever import HTTPRetriever
    return HTTPRetriever(endpoint=cfg["retriever"]["endpoint"])


def make_policy(cfg: dict, dry_run: bool, tokenizer):
    if dry_run:
        from rollout.policy import MockPolicy
        return MockPolicy(tokenizer, queries=["capital of France", "Eiffel Tower height"],
                          answer="Paris")
    # real student: vLLM sampling. NOTE the remaining online concern -- keeping this
    # engine's weights in sync with the training student across steps -- is a deeper
    # HPC integration item; M6 proves the id-passthrough + scoring identity, not the
    # weight resync. Left explicit rather than faked.
    # Engine kwargs come from rollout/engine.py -- the SAME definition
    # scripts/policy_invariance_probe.py builds with, so a probe cannot certify a
    # configuration this arm does not run (that has already happened twice in this project:
    # the hot-update test with prefix caching off vs a real arm with it on, and the
    # determinism probe at gpu_memory_utilization 0.90 vs a server at 0.72). Prefix caching
    # OFF and the in-process EngineCore are load-bearing, not perf knobs -- see that module.
    from rollout.engine import pin_policy_env, policy_engine_kwargs
    pin_policy_env()          # MUST precede `import vllm` -- the flag is read at import time
    from vllm import LLM
    from rollout.vllm_policy import VLLMPolicy
    llm = LLM(model=cfg["model"]["student"],
              **policy_engine_kwargs(dtype=cfg["model"]["dtype"]))
    return VLLMPolicy(llm, tokenizer, temperature=cfg["rollout"]["temperature"],
                      top_p=cfg["rollout"]["top_p"], seed=cfg["run"]["seed"],
                      max_action_tokens=cfg["rollout"]["max_action_tokens"])


def make_teacher(cfg: dict, dry_run: bool):
    if dry_run:
        from envs.tokenizer import MockTokenizer
        from teacher.scorer import MockLM, TeacherScorer
        return TeacherScorer(MockLM(MockTokenizer(vocab_size=8192).vocab_size, seed=1234))
    # HARD guard (CI-guard rule): a real (non-dry-run) run MUST score the teacher via
    # the SERVED vLLM process (engine=vllm_http). In-process TeacherVLLMScorer /
    # TeacherHFScorer are CI/dry-run tools only -- using either here would re-admit the
    # two-vLLM-engines-in-one-process contamination that gate 1's isolation forbids.
    engine = cfg["model"]["engine"]
    if engine != "vllm_http":
        raise ValueError(
            f"real training requires teacher engine 'vllm_http' (served); got {engine!r}. "
            "The in-process vllm/hf scorers are CI/dry-run only -- start the teacher "
            "service (slurm/teacher_server.py) and point teacher.endpoint at it."
        )
    from teacher.http_scorer import TeacherHTTPScorer
    return TeacherHTTPScorer(endpoint=cfg["teacher"]["endpoint"],
                             dtype=cfg["model"]["dtype"],
                             tokenizer_name=cfg["model"]["teacher"])


def make_student(cfg: dict, dry_run: bool, tokenizer):
    if dry_run:
        from teacher.scorer import MockLM, Scorer
        return Scorer(MockLM(tokenizer.vocab_size, seed=5678), requires_grad=True)
    from teacher.scorer import HFScorer
    # The student is TRAINED (forward+backward) -- it must be on GPU. policy/teacher are
    # vLLM (own their GPU); the student is a separate HF model, placed here. Device is a
    # config knob so a CPU-only sanity check is still possible without editing code.
    device = cfg["model"].get("student_device", "cuda")
    # NOT cfg.model.dtype. That field is the TEACHER's backend dtype and feeds gate 1's
    # fingerprint -- changing it would reject the existing precompute cache. The student's
    # parameter dtype is a separate decision and must be fp32: see student_param_dtype in
    # the config for the arithmetic.
    return HFScorer.from_pretrained(
        cfg["model"]["student"],
        dtype=cfg["model"].get("student_param_dtype", "float32"),
        autocast_dtype=cfg["model"].get("student_autocast_dtype", "bfloat16"),
        autocast_cache=cfg["model"].get("autocast_cache", True),
        logprob_chunk_rows=cfg.get("opd", {}).get("logprob_chunk_rows"),
        requires_grad=True, device=device)


# ---------------------------------------------------------------------------
def _opd_config(cfg: dict, mode: str, precompute_dir: Optional[str],
                dry_run: bool) -> OPDConfig:
    """`precompute_dir` and `dry_run` are PARAMETERS, not free names.

    They used to be neither: the body referenced them while the signature took only
    (cfg, mode), so every real invocation raised NameError at the one line that reads
    them -- i.e. --precompute-dir never worked once, and nothing noticed because no
    test calls run(). See tests/test_train_run_entrypoint.py.
    """
    return OPDConfig(
        mode=mode,
        tau=cfg["opd"]["tau"],
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"].get("weight_decay", 0.0),
        adam_beta1=cfg["optim"].get("adam_beta1", 0.9),
        adam_beta2=cfg["optim"].get("adam_beta2", 0.999),
        steps=cfg["optim"]["steps"],
        global_batch=cfg["optim"].get("global_batch", 1),
        seed=cfg["run"]["seed"],
        max_turns=cfg["rollout"]["max_turns"],
        max_len=cfg["rollout"]["max_len"],
        top_k=cfg["rollout"]["top_k"],
        precompute_dir=_resolve_precompute_dir(
            precompute_dir or cfg["diagnostics"].get("precompute_dir"), cfg, dry_run),
        resync_every=cfg.get("online", {}).get("resync_every", 1),
        channel=cfg.get("channel"),        # set by apply_channel; None when no overlay
        loss=cfg["opd"].get("loss", "opd"),
        beta=float(cfg["opd"].get("beta", 1.0)),
        beta_end=(None if cfg["opd"].get("beta_end") is None else float(cfg["opd"]["beta_end"])),
        data_seed=(None if cfg["opd"].get("data_seed") is None else int(cfg["opd"]["data_seed"])),
        ablation=str(cfg["opd"].get("ablation", "none")),
    )


CKPT_POINTER = "ckpt_{mode}_latest.json"


def ckpt_slot(step: int, save_every: int) -> Optional[int]:
    """Which of the TWO alternating slots `step` saves into, or None if it does not save.

    Two slots, not one overwritten directory, and not one directory per save.

    * Overwriting a single directory means a job killed mid-write leaves a checkpoint that is
      neither the old one nor a complete new one. `save_pretrained` writes several files; there
      is no moment at which that is atomic.
    * One directory per save would be correct and costs ~16 GiB each (4B parameters in fp32 --
      the periodic save uses the SAME dtype as the final one on purpose, see below).

    Alternating bounds the disk at two while guaranteeing that at any instant one COMPLETE
    checkpoint exists on disk.
    """
    if not save_every or step <= 0 or step % save_every:
        return None
    return (step // save_every) % 2


def save_student_checkpoint(student, tokenizer, ckpt_dir: str) -> str:
    """Write a student checkpoint. THE ONE PLACE that does -- periodic and final both call it.

    Not a refactor for tidiness: a periodic checkpoint that is written differently from the
    final one is a checkpoint you cannot evaluate the same way, and the difference would only
    surface when you try to read it after losing the run that wrote it.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    student.lm.save_pretrained(ckpt_dir)
    save_tok = getattr(tokenizer, "tok", tokenizer)   # unwrap HFTokenizerAdapter
    if hasattr(save_tok, "save_pretrained"):
        save_tok.save_pretrained(ckpt_dir)
    return ckpt_dir


def save_periodic_checkpoint(student, tokenizer, out_dir: str, mode: str,
                             step: int, slot: int) -> str:
    """Save into `slot`, THEN record it. The order is the crash-safety property.

    The pointer is written and fsynced only after save_student_checkpoint returns, so it can
    never name a directory whose write did not finish. A reader that trusts the pointer is
    therefore reading a complete checkpoint even if the job was killed one second later --
    which is the whole reason this exists, the online arm being a ~31-hour job whose only
    other durable output is the per-step diagnostics stream.
    """
    ckpt_dir = os.path.join(out_dir, f"ckpt_{mode}_p{slot}")
    save_student_checkpoint(student, tokenizer, ckpt_dir)
    pointer = os.path.join(out_dir, CKPT_POINTER.format(mode=mode))
    with open(pointer, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "step": step, "dir": os.path.basename(ckpt_dir)}, f)
        f.flush()
        os.fsync(f.fileno())
    return ckpt_dir


def keep_checkpoint(student, tokenizer, out_dir: str, mode: str, step: int) -> str:
    """A permanent, never-overwritten checkpoint directory for one named step."""
    ckpt_dir = os.path.join(out_dir, f"ckpt_{mode}_step{step}")
    save_student_checkpoint(student, tokenizer, ckpt_dir)
    return ckpt_dir


def _resolve_precompute_dir(path, cfg, dry_run):
    """The offline cache directory, refusing a directory that already holds ANOTHER teacher's.

    WHY THE REFUSAL. diagnostics.precompute_dir defaults to a single shared "precomputed/".
    Three teacher lines running at once all write traj_0000.pt there, and the second one
    overwrites the first's manifest too -- so gate 1, which compares a run against the
    manifest it finds, sees a cache that is perfectly consistent with the run that clobbered
    it. The confound is a WRITE collision, not a fingerprint mismatch, and no fingerprint can
    catch it. BackendConfig.tokenizer_name is the teacher's path, so a stale manifest from a
    different teacher IS detectable -- before anything is written, which is the only moment
    it costs nothing.
    """
    if path is None or dry_run:
        return path
    man = os.path.join(path, "manifest.json")
    if os.path.exists(man):
        try:
            with open(man, encoding="utf-8") as f:
                prev = ((json.load(f) or {}).get("backend") or {}).get("tokenizer_name")
        except (ValueError, OSError):
            prev = None
        here = cfg["model"]["teacher"]
        if prev and prev != here:
            raise SystemExit(
                f"FATAL: {man} was written for teacher {prev!r} but this run's teacher is "
                f"{here!r}.\n"
                f"  Writing here would overwrite that line's cache AND its manifest, leaving "
                f"the two consistent with each other and the first line reading the second "
                f"line's trajectories.\n"
                f"  Pass --precompute-dir a directory of this line's own.")
    return path


def run(config_path: str, mode: str, out_dir: str, dry_run: bool,
        steps: Optional[int] = None, limit: Optional[int] = None,
        epochs: Optional[float] = None, channel: Optional[str] = None,
        retriever_endpoint: Optional[str] = None,
        teacher_endpoint: Optional[str] = None,
        precompute_only: bool = False, save_every: int = 0,
        precompute_dir: Optional[str] = None, loss: Optional[str] = None,
        beta: Optional[float] = None, keep_steps: tuple = (), beta_end: Optional[float] = None,
        data_seed: Optional[int] = None, ablation: Optional[str] = None) -> None:
    # BEFORE the config is read. This is a pure argument check -- it needs no file and no
    # site -- and putting it first means a mis-invocation is refused for the reason it is
    # actually wrong, instead of for whatever the config loader happens to notice first.
    if channel is None and not dry_run:
        raise ValueError(
            "a real run must name its hardware channel: --channel h20 | a100. The channel "
            "carries the teacher's memory budget, the trainable length ceiling and the "
            "memory/speed knobs, all of which differ per card -- and one channel must host a "
            "run's precompute AND both of its arms, so leaving it implicit is how a "
            "cross-channel mix happens. See configs/channels/README.md.")

    cfg = load_config(config_path)
    os.makedirs(out_dir, exist_ok=True)
    # --loss / --beta: the delta arm (opd/delta_loss.py). Overrides land in cfg["opd"] so the
    # ONE place that builds OPDConfig (_opd_config) stays the one place.
    if loss is not None:
        cfg["opd"]["loss"] = loss
    if beta is not None:
        cfg["opd"]["beta"] = float(beta)
    # The sbatch passes both ALWAYS (a `${X:+...}` optional flag reads as a stray positional to
    # tests/test_sbatch_wellformed.py): beta_end=nan and data_seed=0 mean "not set".
    if beta_end is not None and beta_end == beta_end:          # nan != nan
        cfg["opd"]["beta_end"] = float(beta_end)
    if data_seed:
        cfg["opd"]["data_seed"] = int(data_seed)
        cfg["run"]["seed"] = int(data_seed)          # the record's seed field says which arm this is
    if ablation is not None:
        cfg["opd"]["ablation"] = ablation

    # BOTH ARMS, SAME KERNEL ENVIRONMENT -- which after measurement means VANILLA for the
    # student everywhere. vLLM's batch-invariant overrides are inference-only (the direct
    # aten::linear kernel leaves autograd asking for linear_backward, which has no CUDA
    # kernel), so the online arm BRACKETS its training/diagnostic sections out of them
    # (OPDTrainer.step -> rollout/engine.student_vanilla_math) and offline never enables
    # them. What still needs equalizing here is cuBLAS algorithm selection: the online
    # process carries CUBLAS_WORKSPACE_CONFIG from vllm's override_envs_for_invariance, and
    # cuBLAS picks algorithms per workspace config -- a bit-level asymmetry the kernel
    # bracket cannot see. Pin the same value in both arms; deterministic algorithm choice is
    # desirable in its own right. tests/test_student_kernel_parity.py measures the result.
    if not dry_run:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # first-signal overrides: cap steps / dataset size from the CLI so the committed
    # Table-6 hyperparameters (steps=150, full set) stay pristine and we never hand-edit
    # the version-controlled config on HPC (which would fight `git pull`).
    # CHANNEL FIRST: everything below reads values it may have overridden.
    if channel is not None:
        apply_channel(cfg, channel)
    if limit is not None:
        cfg["data"]["limit"] = limit
    if epochs is not None:
        cfg["data"]["epochs"] = epochs
    # retriever host is only known at run time (it's whatever node the retriever job landed
    # on). Override here so the committed endpoint stays a REPLACE_ME placeholder, not a
    # per-run hostname the training sbatch would otherwise have to sed into the yaml.
    if retriever_endpoint is not None:
        cfg["retriever"]["endpoint"] = retriever_endpoint
    # teacher host is likewise only known at run time (whatever node the teacher-service
    # job landed on). Override here so the committed teacher.endpoint stays REPLACE_ME.
    if teacher_endpoint is not None:
        cfg.setdefault("teacher", {})["endpoint"] = teacher_endpoint

    tokenizer = make_tokenizer(cfg, dry_run)
    teacher = make_teacher(cfg, dry_run)
    questions, golds = make_dataset(cfg, dry_run)

    opd_cfg = _opd_config(cfg, mode, precompute_dir, dry_run)
    if steps is not None:
        opd_cfg.steps = steps
    if dry_run:                                    # keep the CPU preflight short
        opd_cfg.steps = min(opd_cfg.steps, 4)
    assert_epochs_declared(cfg, opd_cfg, len(questions))

    # The ROLLOUT stack (vLLM policy + retriever) is only needed when we actually SAMPLE:
    # the online arm (live, every step) or a precompute pass. Offline TRAINING that reads an
    # EXISTING manifest never samples -- so skip the vLLM/retriever load entirely. This avoids
    # loading a whole vLLM engine (and a second GPU) the offline read never uses, and makes the
    # offline arm robust to a policy-side vLLM failure (e.g. an NVML device-probe error on a bad
    # card) that has nothing to do with reading a cache. The self-precompute fallback (manifest
    # not yet written) still builds them.
    need_rollout = precompute_only or mode == "online"
    if mode == "offline" and not precompute_only:
        manifest_path = os.path.join(opd_cfg.precompute_dir or "", MANIFEST_NAME)
        need_rollout = not os.path.exists(manifest_path)
    policy = make_policy(cfg, dry_run, tokenizer) if need_rollout else None
    retriever = make_retriever(cfg, dry_run) if need_rollout else None
    if mode == "offline" and not need_rollout:
        print("[offline] manifest present -> read-only; skipped vLLM policy + retriever load")

    # precompute-only: the DECOUPLED offline half. Run the rollout+teacher-score loop in
    # THIS process, write the traj_*.pt cache + manifest, and STOP -- no student, no
    # trainer. A later `--mode offline` training process finds the manifest and only reads
    # it. Calling the same precompute_offline that OfflineSource would call inline keeps
    # the cache byte-identical to the self-precompute path; splitting into two processes
    # changes WHEN the teacher is scored, never HOW.
    if precompute_only:
        manifest = precompute_offline(opd_cfg, questions, policy, retriever, tokenizer,
                                      teacher, golds=golds)
        print(f"[precompute-only] wrote {len(questions)} trajectories + manifest -> {manifest}")
        return

    # student device: with a co-resident vLLM policy (online) vLLM owns cuda:0, so the student
    # goes on cuda:1 (config default). With NO policy built (offline read / self-precompute),
    # nothing holds cuda:0 -- put the student there. Forcing cuda:1 in that case is pointless AND
    # a hard failure ("invalid device ordinal") on a node that exposes only one card.
    if policy is None:
        cfg["model"]["student_device"] = "cuda:0"
    student = make_student(cfg, dry_run, tokenizer)

    # ---- from here down: IDENTICAL to a real run (the phase-1 invariant) ----
    source = build_source(opd_cfg, questions, policy, retriever, tokenizer, teacher, golds=golds)
    ref = snapshot_ref(student)                    # pi_ref = student at step 0
    diagnostics = DiagnosticRunner(ref_scorer=ref, tau=opd_cfg.tau, embedder=None)
    log_path = os.path.join(out_dir, f"diag_{mode}.jsonl")
    print(f"[train mode={mode} dry_run={dry_run}] steps={opd_cfg.steps} "
          f"tau={opd_cfg.tau} -> {log_path}")
    with DiagnosticLogger(log_path) as logger:
        trainer = OPDTrainer(student, source, opd_cfg, diagnostics=diagnostics, logger=logger)
        for i, out in enumerate(trainer.train()):
            print(f"  step {i}: loss={out.loss.item():+.6f} scored={int(out.score_mask.sum())}")
            # PURE SIDE EFFECT. Nothing here reads or writes trainer state, so the numbers are
            # the numbers whether or not this fires -- which is the only version of this
            # feature worth having in a run that produces a paper.
            slot = ckpt_slot(i + 1, save_every) if not dry_run else None
            if slot is not None:
                d = save_periodic_checkpoint(student, tokenizer, out_dir, mode, i + 1, slot)
                print(f"[ckpt] periodic save at step {i + 1} -> {d}", flush=True)
            # KEPT checkpoints: a permanent copy at named steps (the delta arm's step-30 gate
            # reads one while training continues; the alternating slots would overwrite it).
            if (i + 1) in set(keep_steps) and not dry_run:
                d = keep_checkpoint(student, tokenizer, out_dir, mode, i + 1)
                print(f"[ckpt] kept step {i + 1} -> {d}", flush=True)
    print(f"done -> {log_path}")

    # ---- persist the TRAINED student as a HF checkpoint (real runs only) ----
    # The ONLY other artifact this run writes is runs/diag_{mode}.jsonl (the in-run
    # diagnostics). Those diagnostics can NOT answer "did OPD change the model?": online
    # task.em is self-poisoning ONSET, offline task.em is a frozen-input property -- neither
    # is a held-out capability read. To compare the online-arm product vs the offline-arm
    # product (or vs pi_ref) we need the WEIGHTS, eval'd later on a disjoint slice by
    # scripts/stage3_eval.py --model <ckpt>. So write student.lm here, one dir per arm.
    # Bonus: this also exercises the from_pretrained load path on our OWN OPD output --
    # the ckpt round-trip is otherwise never walked, and a silent save/load breakage would
    # only surface at eval time. Guarded on dry_run: MockLM has no save_pretrained.
    if not dry_run:
        ckpt_dir = save_student_checkpoint(student, tokenizer,
                                           os.path.join(out_dir, f"ckpt_{mode}"))
        print(f"[ckpt] saved trained {mode} student -> {ckpt_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified OPD trainer entrypoint.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["online", "offline"], default=None,
                    help="override run.mode in the config")
    ap.add_argument("--out-dir", default=None, help="override run.out_dir")
    ap.add_argument("--dry_run", action="store_true", help="force the mock CPU path")
    ap.add_argument("--steps", type=int, default=None,
                    help="override optim.steps (first-signal cap; config stays pristine)")
    ap.add_argument("--limit", type=int, default=None,
                    help="override data.limit (subset size for the first curve)")
    ap.add_argument("--save-every", type=int, default=0, metavar="N",
                    help="checkpoint the student every N steps (0 = only at the end). The run "
                         "alternates between two directories so a job killed mid-write always "
                         "leaves one COMPLETE checkpoint, and records the finished one in "
                         "ckpt_<mode>_latest.json. ~16 GiB per slot for a 4B fp32 student. The "
                         "per-step diagnostics already survive a crash; this is what makes the "
                         "WEIGHTS survive one, and the weights are what the dev eval needs.")
    ap.add_argument("--channel", default=None,
                    help="hardware channel (configs/channels/<name>.yaml). REQUIRED for a "
                         "real run: it carries the teacher's memory budget, the trainable "
                         "length ceiling and the memory/speed knobs, which differ per card. "
                         "One channel must host a run's precompute AND both arms.")
    ap.add_argument("--epochs", type=float, default=None,
                    help="override data.epochs. REQUIRED alongside --limit on a smoke run: "
                         "steps x global_batch must equal limit x epochs, and epochs is the "
                         "data-diversity ratio between the arms (offline reuses each cached "
                         "trajectory, online resamples). 1 = apples-to-apples.")
    ap.add_argument("--retriever-endpoint", default=None,
                    help="override retriever.endpoint (the retriever job's host:port/retrieve)")
    ap.add_argument("--teacher-endpoint", default=None,
                    help="override teacher.endpoint (the teacher-service job's host:port/score_logprobs)")
    ap.add_argument("--precompute-dir", default=None,
                    help="override diagnostics.precompute_dir. REQUIRED when more than one "
                         "teacher line runs at once: the default is a single shared "
                         "'precomputed/', and a second line silently OVERWRITES the first's "
                         "traj_*.pt AND its manifest, so gate 1 finds the two consistent with "
                         "each other and reports nothing.")
    ap.add_argument("--loss", choices=["opd", "delta"], default=None, help="override opd.loss (delta = the delta-KL arm)")
    ap.add_argument("--beta", type=float, default=None, help="override opd.beta (delta arm's interpolation)")
    ap.add_argument("--keep-steps", default="", help="comma-separated steps to keep a permanent checkpoint at (ckpt_<mode>_step<N>)")
    ap.add_argument("--beta-end", type=float, default=None, help="delta arm: anneal beta linearly to this value over the run (nan = unset)")
    ap.add_argument("--delta-ablation", choices=["none", "teacher_head", "no_bucket", "no_anchor", "supp1", "supp2", "supp4",
                                                 "rho50", "rho80", "rho90", "rho95", "rho99"], default=None,
                    help="delta arm: opd.delta_loss.ABLATIONS -- teacher_head (S = K_q ∪ {y}, no pi_ref top-K'), no_bucket, no_anchor (sg(p_theta) fill), "
                         "supp<n> (nested support of size n: {y} + pi_ref ranks, dropped mass into the bucket)")
    ap.add_argument("--data-seed", type=int, default=None, help="offline: permute the cache order with this seed (a seed-2 arm; 0 = cache order)")
    ap.add_argument("--precompute-only", action="store_true",
                    help="offline DECOUPLE: roll out + teacher-score to disk (manifest), then stop "
                         "(no student/training). A later --mode offline run reads the cache.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg["run"]["mode"]
    out_dir = args.out_dir or cfg["run"]["out_dir"]
    dry_run = args.dry_run or bool(cfg["run"].get("dry_run", False))
    if args.precompute_only and mode != "offline":
        raise SystemExit("--precompute-only only makes sense with --mode offline")
    run(args.config, mode, out_dir, dry_run, steps=args.steps, limit=args.limit,
        epochs=args.epochs, channel=args.channel,
        retriever_endpoint=args.retriever_endpoint, teacher_endpoint=args.teacher_endpoint,
        precompute_only=args.precompute_only, save_every=args.save_every,
        precompute_dir=args.precompute_dir, loss=args.loss, beta=args.beta,
        keep_steps=tuple(int(x) for x in args.keep_steps.split(",") if x.strip()),
        beta_end=args.beta_end, data_seed=args.data_seed, ablation=args.delta_ablation)


if __name__ == "__main__":
    main()
