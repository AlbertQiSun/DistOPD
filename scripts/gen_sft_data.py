"""M-SFT Stage 1 entrypoint: generate MiroThinker SFT data in the OPD env.

ONE code path; `--dry_run` flips the factories to the CPU mock exactly like
scripts/train.py. Everything from generate_sft_dataset(...) downward is byte-identical
between the dry-run and a real HPC run -- the dry-run proves the wiring (generate ->
filter -> action-only mask -> save + teacher-consistency assert) on mock data.

The SFT-data GENERATOR is the OPD TEACHER (MiroThinker), NOT the student. That is the
teacher-consistency red line, and generate_sft_dataset asserts it (model identity) before
writing anything.

    python scripts/gen_sft_data.py --config configs/opd_qwen3.yaml --dry_run
    python scripts/gen_sft_data.py --config configs/opd_qwen3.yaml --out-dir ...   # real (HPC)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import configs.template as T
from opd.config import load_config
from sft.datagen import SFTGenConfig, generate_sft_dataset
from teacher.backend import BackendConfig, template_fingerprint
from teacher.scorer import MOCK_CONFIG
# reuse train.py's factories so the SFT env is bit-identical to the OPD env
from scripts.train import make_dataset, make_retriever, make_tokenizer


def make_generator(cfg: dict, dry_run: bool, tokenizer, temperature: float,
                   max_action_tokens: int = None, max_len: int = None,
                   prefix_caching: bool = None, batch_invariant: bool = None):
    """-> (policy, gen_backend). The generator IS the OPD teacher (MiroThinker).

    `temperature` is passed explicitly (default 0.6, NOT the OPD rollout's 0.8): SFT wants
    MiroThinker's MOST coherent search for pi_ref to imitate, whereas OPD's 0.8 is there to
    make pi_theta EXPLORE. 0.8 raises the unproductive-spinning rate in the SFT data; the
    productive-multihop filter is the backstop, but keeping the temperature moderate reduces
    how much the filter has to throw away. This is an explicit, ablatable choice."""
    if dry_run:
        from rollout.policy import MockPolicy
        pol = MockPolicy(tokenizer, queries=["capital of France", "Eiffel Tower height"],
                         answer="Paris")
        return pol, MOCK_CONFIG, {}
    # BEFORE `import vllm`, and that ordering is the whole point: VLLM_BATCH_INVARIANT is read
    # at IMPORT time, so setting it after the import silently gives the run the other kernels.
    # This block first sat below the import -- where the assert fired on every real run, which
    # is the guard doing exactly its job and is why it is an assert rather than a comment.
    #
    # The VALUE's authority is rollout/engine.py (POLICY_BATCH_INVARIANT and the probe
    # measurement behind it); only the decision to apply it to the SAMPLING engine is local,
    # because the failure it prevents -- batch composition changing the sampled tokens --
    # needs a batch, and this path is serial (sft/datagen drives generate_trajectory, singular).
    #
    # VLLM_ENABLE_V1_MULTIPROCESSING is deliberately NOT bundled in. pin_policy_env sets it
    # because the online policy needs an in-process handle for sync_from's weight hot-update;
    # generation never hot-updates, and quietly changing where the engine lives would be an
    # unmeasured change riding along with a measured one.
    bi = (cfg["rollout"].get("batch_invariant_generation", False)
          if batch_invariant is None else bool(batch_invariant))
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if bi else "0"
    assert "vllm" not in sys.modules, (
        "vllm was imported before VLLM_BATCH_INVARIANT could be set -- the flag is read at "
        "import time, so this run would silently use the other kernels. Something imported "
        "vllm at module scope; find it rather than moving this assert.")

    # real: sample MiroThinker with vLLM (fast). engine here is 'vllm' -- that is fine:
    # assert_same_teacher_model ignores engine (SFT samples tokens, never scores logp).
    from vllm import LLM
    from rollout.vllm_policy import VLLMPolicy
    name = cfg["model"]["teacher"]                       # <- the OPD teacher = the generator
    # PIN THE ENGINE. This call used to be LLM(model=name, dtype=...) and nothing else, so
    # max_model_len came from the checkpoint's own config and gpu_memory_utilization from
    # vLLM's default. That survived only because Stage 1 ran on the 95 GiB H20: MiroThinker
    # declares a 256k context, one request at that length needs ~24 GiB of KV, and 0.9 x 95.1
    # minus 56.9 GiB of weights left 28.7 GiB -- about 4.7 GiB of margin nobody had chosen.
    # On the 79 GiB A100 the same call leaves 13.8 GiB and vLLM refuses to start, and it
    # refuses for ANY teacher, including MiroThinker: this is a channel-portability bug, not
    # a new-model problem. It surfaced on Qwen3-30B-A3B-Thinking only because that line was
    # the first generation ever attempted on this card.
    #
    # max_model_len follows the rollout budget rather than the checkpoint: a trajectory is
    # hard-capped at rollout.max_len and the generator must hold it plus the turn it is
    # sampling, which is the same 2048 of headroom the served teacher runs with (15360 ->
    # 17408). gpu_memory_utilization goes through the same resolver as every other engine
    # (explicit > $OPD_TEACHER_UTIL > channel file > default) so one card's number cannot be
    # silently inherited by another.
    # ... and the utilization is SIZED TO THIS CHECKPOINT rather than inherited. The channel
    # file's 0.80 is a measurement -- of MiroThinker's 56.93 GiB of weights. Qwen3-32B is
    # 61.03 GiB, and that 4.1 GiB is almost exactly the KV pool the certification left, so the
    # engine refuses to start ("estimated maximum model length is 2656"). The first fix here
    # stopped one model's number being inherited by another CARD; this stops it being
    # inherited by another MODEL. Scoring is untouched -- resolve_generation_utilization is
    # for the sampling engine only and says so.
    from teacher.engine import resolve_generation_utilization
    max_model_len = int(max_len or cfg["rollout"]["max_len"]) + 2048
    util = resolve_generation_utilization(name, max_model_len)
    print(f"[gen-sft] engine: max_model_len={max_model_len} "
          f"gpu_memory_utilization={util}", flush=True)
    # enable_prefix_caching is PASSED rather than left to vLLM's default, because it is the
    # leading suspect for Qwen3-32B producing different output across two runs with identical
    # settings (see configs/opd_qwen3.yaml). Passing it explicitly also means the shard records
    # which way it ran -- the two shards that produced the discrepancy do not.
    pfx = (cfg["rollout"].get("enable_prefix_caching", True)
           if prefix_caching is None else bool(prefix_caching))
    llm = LLM(model=name, dtype=cfg["model"]["dtype"],
              max_model_len=max_model_len, gpu_memory_utilization=util,
              enable_prefix_caching=pfx)
    # max_action_tokens was a PYTHON DEFAULT (512) that no call site ever passed, so it had
    # never been chosen -- and it binds the teachers unequally: measured 2026-08-21, 100% of
    # Qwen3-30B-A3B-Thinking's empty_query drops sat exactly at the cap, against a MiroThinker
    # median of 37 tokens per action. An uncontrolled variable that favours one teacher is a
    # confound in a three-teacher comparison, so it is now explicit, overridable, and recorded.
    cap = int(max_action_tokens or cfg["rollout"].get("max_action_tokens") or 512)
    pol = VLLMPolicy(llm, tokenizer, temperature=temperature, max_action_tokens=cap,
                     top_p=cfg["rollout"]["top_p"], seed=cfg["run"]["seed"])
    # record what the engine ACTUALLY ran with, not what the config asked for. These fields
    # were left None, so a shard could not say which utilization or context length produced
    # it -- and this stage now resolves both at runtime, which makes "what was it" a question
    # only the artifact can answer. assert_same_teacher_model ignores `engine` but compares
    # everything else, and the OPD teacher declares None for these, so they are attached
    # AFTER the identity check rather than being part of it.
    gb = BackendConfig(engine="vllm", dtype=cfg["model"]["dtype"], tokenizer_name=name)
    return pol, gb, {"max_model_len": max_model_len, "gpu_memory_utilization": util,
                     "max_action_tokens": cap, "enable_prefix_caching": pfx,
                     "batch_invariant": bi,
                     "max_len": int(max_len or cfg["rollout"]["max_len"])}


def make_sft_dataset(cfg: dict, dry_run: bool):
    """-> (questions, golds, tags). Uses the difficulty-mixed multi-source loader when
    cfg['sft']['data']['sources'] is set (so the hop distribution isn't all-easy);
    otherwise falls back to the single-source OPD dataset factory.

    `tags` is the per-row DECLARED hop class / source label (1hop/2hop/multihop) that
    load_mixed attaches; it is provenance ONLY (never gates) and is threaded downstream so
    the health report can split PER SOURCE. On the single-source / dry-run fallback there is
    no meaningful source label, so tags is all-None."""
    sources = (cfg.get("sft", {}).get("data", {}) or {}).get("sources")
    if dry_run or not sources:
        qs, gs = make_dataset(cfg, dry_run)
        return qs, gs, [None] * len(qs)
    from dataio.dataset import load_mixed
    total = cfg.get("sft", {}).get("data", {}).get("total")
    rows = load_mixed(sources, total=total, seed=cfg["run"]["seed"])
    if not rows:
        raise ValueError("no usable rows from sft.data.sources (need question + gold)")
    return ([r["question"] for r in rows], [r["gold_answers"] for r in rows],
            [r.get("tag") for r in rows])


def opd_teacher_backend(cfg: dict, dry_run: bool) -> BackendConfig:
    """The BackendConfig the OPD teacher will declare -- the target the generator must
    match (model identity). Built WITHOUT loading the heavy model."""
    if dry_run:
        return MOCK_CONFIG
    return BackendConfig(engine=cfg["model"]["engine"], dtype=cfg["model"]["dtype"],
                         tokenizer_name=cfg["model"]["teacher"])


def run(config_path: str, out_dir: str, dry_run: bool,
        limit: Optional[int] = None, retriever_endpoint: Optional[str] = None,
        temperature: Optional[float] = None, max_action_tokens: Optional[int] = None,
        max_len: Optional[int] = None, prefix_caching: Optional[bool] = None,
        batch_invariant: Optional[bool] = None, questions_file: Optional[str] = None,
        source_tag: str = "opd_slice", batch: int = 1) -> None:
    cfg = load_config(config_path)
    if questions_file is not None:
        # SeqKD: the questions come from a file (a seeded sample
        # of the OPD slice, scripts/seqkd_questions.py) instead of the difficulty-mixed sources;
        # everything else -- temperature, caps, drops, records -- is Stage 1 unchanged.
        cfg.setdefault("sft", {})["data"] = {"sources": [{"path": questions_file, "tag": source_tag, "limit": None}], "total": None}
    os.makedirs(out_dir, exist_ok=True)
    if limit is not None:
        cfg["data"]["limit"] = limit                     # single-source fallback path
        # When sft.data.sources is set the questions come from load_mixed, which ignores
        # cfg["data"]["limit"] and reads per-source limits + sft.data.total. So a SMOKE
        # `--limit 32` must ALSO cap the mixed total, or it silently generates the full mix.
        cfg.setdefault("sft", {}).setdefault("data", {})["total"] = limit
    if retriever_endpoint is not None:
        cfg["retriever"]["endpoint"] = retriever_endpoint

    sft_cfg = cfg.get("sft", {}) or {}
    temp = temperature if temperature is not None else sft_cfg.get("temperature", 0.6)
    # spin_min_hops is a REPORT-ONLY knob for scripts/check_sft_stats.py; it no longer gates
    # datagen (zero quality filtering), so it is NOT threaded into SFTGenConfig.
    spin_min_hops = sft_cfg.get("spin_min_hops", 3)

    tokenizer = make_tokenizer(cfg, dry_run)
    retriever = make_retriever(cfg, dry_run)
    generator, gen_backend, engine_kw = make_generator(
        cfg, dry_run, tokenizer, temperature=temp,
        max_action_tokens=max_action_tokens, max_len=max_len,
        prefix_caching=prefix_caching, batch_invariant=batch_invariant)
    teacher_backend = opd_teacher_backend(cfg, dry_run)
    questions, golds, tags = make_sft_dataset(cfg, dry_run)

    gcfg = SFTGenConfig(max_turns=cfg["rollout"]["max_turns"],
                        max_len=int(max_len or cfg["rollout"]["max_len"]),
                        top_k=cfg["rollout"]["top_k"], seed=cfg["run"]["seed"])

    # Announce the prompt arm BEFORE generating. OPD_PROMPT_VARIANT reaches this process
    # through the environment, and this cluster has already eaten one env var silently
    # (--export=NONE stripped OPD_SITE). Discovering after 40 minutes that the treatment arm
    # ran the control prompt costs the whole run; one line at the top costs nothing.
    print(f"[gen_sft dry_run={dry_run}] questions={len(questions)} temp={temp} "
          f"spin_min_hops={spin_min_hops} batch={batch} -> {out_dir}")
    print(f"[gen_sft] PROMPT VARIANT = {T.PROMPT_VARIANT}   "
          f"template_fingerprint = {template_fingerprint()}")
    print(f"[gen_sft] system prompt ({len(T.SYSTEM_PROMPT)} chars): "
          f"{T.SYSTEM_PROMPT[:120]!r}...")
    print(f"[gen_sft] max_action_tokens={engine_kw.get('max_action_tokens')}  "
          f"max_len={gcfg.max_len}  max_turns={gcfg.max_turns}  "
          f"prefix_caching={engine_kw.get('enable_prefix_caching')}  "
          f"batch_invariant={engine_kw.get('batch_invariant')}")
    # WHICH COMMIT MADE THIS SHARD. Both stale-checkout incidents would have been one grep
    # away with this line present, and the shard itself could not say what code produced it.
    from opd.trainer import _git_head
    engine_kw["code_version"] = _git_head()
    print(f"[gen_sft] code_version={engine_kw['code_version']}")
    stats = generate_sft_dataset(
        questions=questions, golds=golds, policy=generator, retriever=retriever,
        tokenizer=tokenizer, cfg=gcfg, gen_backend=gen_backend,
        teacher_backend=teacher_backend, out_dir=out_dir, tags=tags,
        engine_kwargs=engine_kw, batch=batch,
    )
    print(f"  kept={stats.kept}  productive_multihop={stats.productive_multihop}  "
          f"dropped={stats.dropped}  hop_hist={stats.hops}")
    print(f"done -> {out_dir}/manifest.jsonl , {out_dir}/stats.json "
          f"(run scripts/check_sft_stats.py {out_dir} for the health report; "
          f"only no_search can FAIL, the rest is report-only)")


def assert_expectations(cfg: dict, *, resolved_variant: str, cli_cap,
                        expect_cap, expect_variant) -> int:
    """Refuse to generate unless the run matches what the submit command asked for.

    Returns the resolved action cap. Split out of main() so it can be unit-tested: the two
    failures it guards against both produced shards that LOOKED normal, and a guard verified
    only by being run once by hand is the same kind of thing as the shards it is protecting.
    """
    if expect_variant and resolved_variant != expect_variant:
        raise SystemExit(
            f"FATAL: --expect-prompt-variant {expect_variant!r} but configs.template "
            f"resolved to {resolved_variant!r}.\n"
            f"  OPD_PROMPT_VARIANT={os.environ.get('OPD_PROMPT_VARIANT')!r}. If that looks "
            f"right, this checkout predates the variant: git pull.")
    declared = (cfg.get("rollout") or {}).get("max_action_tokens")
    cap = int(cli_cap or declared or 512)
    if expect_cap is not None and cap != expect_cap:
        raise SystemExit(
            f"FATAL: --expect-action-cap {expect_cap} but this run resolved "
            f"max_action_tokens={cap}.\n"
            f"  configs/opd_qwen3.yaml rollout.max_action_tokens = {declared}; "
            f"--max-action-tokens = {cli_cap}.\n"
            f"  A stale checkout is the usual cause -- git pull. Generating under the wrong "
            f"cap does not fail, it produces a shard whose empty_query drops are truncation "
            f"wearing a format failure's name.")
    return cap


def main() -> None:
    ap = argparse.ArgumentParser(description="M-SFT Stage 1: MiroThinker SFT data generation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry_run", action="store_true", help="force the mock CPU path")
    ap.add_argument("--limit", type=int, default=None, help="override data.limit")
    ap.add_argument("--questions-file", default=None,
                    help="JSONL of {question, golden_answers}: replaces sft.data.sources (SeqKD on the OPD slice)")
    ap.add_argument("--source-tag", default="opd_slice", help="source_tag written to every manifest row with --questions-file")
    ap.add_argument("--batch", type=int, default=1,
                    help="questions per gathered generation round (1 = the original one-at-a-time path)")
    ap.add_argument("--retriever-endpoint", default=None,
                    help="override retriever.endpoint (the retriever job's host:port/retrieve)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override sft.temperature (SFT generation temp; default 0.6)")
    ap.add_argument("--max-action-tokens", type=int, default=None,
                    help="per-action sampling cap. THE PROBE KNOB: a distribution measured "
                         "under a binding cap is right-censored at it and cannot be used to "
                         "choose a new one, so measure with this set large, then decide.")
    # WHY THESE EXIST. Twice in one session a job ran against a checkout that predated the
    # change it was meant to test, and produced a shard that looked completely normal: once
    # OPD_PROMPT_VARIANT did not exist yet so both A/B arms silently ran the control prompt,
    # once configs/opd_qwen3.yaml still said max_action_tokens 512 where the run needed 2560.
    # Both times the tell was one line in a 3000-line log that nobody had a reason to reread,
    # and the fixed seed made the wrong result byte-identical to a previous one -- which reads
    # as reproducibility, not as failure. The submit command knows what it intends; these turn
    # that intent into a check that fires in five seconds, before the weights load.
    ap.add_argument("--batch-invariant", dest="batch_invariant", action="store_true",
                    default=None,
                    help="pin VLLM_BATCH_INVARIANT for this generation run. Predicted to be a "
                         "no-op here (generation is serial, so batch composition never "
                         "varies) -- measure the cost before adopting it.")
    ap.add_argument("--no-prefix-caching", dest="prefix_caching", action="store_false",
                    default=None,
                    help="turn vLLM prefix caching OFF for this run. THE DETERMINISM TEST: "
                         "two runs of one teacher on the same questions must come out "
                         "byte-identical, and Qwen3-32B currently does not.")
    ap.add_argument("--expect-action-cap", type=int, default=None,
                    help="FATAL unless the resolved max_action_tokens equals this. Put the "
                         "value the experiment requires in the submit command.")
    ap.add_argument("--expect-prompt-variant", default=None,
                    help="FATAL unless configs.template resolved to this variant.")
    ap.add_argument("--max-len", type=int, default=None,
                    help="whole-trajectory budget override. Raise it ALONGSIDE "
                         "--max-action-tokens: leaving it at 15360 while the cap goes to 4096 "
                         "just moves the censoring from the action to the trajectory, and the "
                         "measurement is uncensored in neither.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dry_run = args.dry_run or bool(cfg["run"].get("dry_run", False))
    # Checked BEFORE run() imports anything heavy, so a stale checkout dies in seconds rather
    # than after a 70-second weight load inside a queued job.
    assert_expectations(cfg, resolved_variant=T.PROMPT_VARIANT,
                        cli_cap=args.max_action_tokens,
                        expect_cap=args.expect_action_cap,
                        expect_variant=args.expect_prompt_variant)
    run(args.config, args.out_dir, dry_run, limit=args.limit,
        retriever_endpoint=args.retriever_endpoint, temperature=args.temperature,
        max_action_tokens=args.max_action_tokens, max_len=args.max_len,
        prefix_caching=args.prefix_caching, batch_invariant=args.batch_invariant,
        questions_file=args.questions_file, source_tag=args.source_tag, batch=args.batch)


if __name__ == "__main__":
    main()
