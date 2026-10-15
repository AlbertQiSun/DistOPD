"""Sizing the generation engine from the checkpoint instead of inheriting another model's number.

THE FAILURE THIS ENCODES. configs/channels/a100.yaml pins gpu_memory_utilization 0.80, and the
comment above it says exactly how that was earned: "63.4 GiB budget - 56.93 GiB weights -> KV
cache 5.1 GiB". It is a measurement of MiroThinker. Qwen3-32B weighs 61.03 GiB, and the 4.1 GiB
difference is almost exactly the KV pool that measurement left over, so the engine refused to
start with "estimated maximum model length is 2656" -- on a card that had just run the same
model fine, under vLLM's old 0.9 default, before the utilization was pinned at all.

That is the same shape as the bug resolve_teacher_utilization was written to stop (one card's
number inherited by another card), one level up: one MODEL's number inherited by another model.

THE ANCHOR. test_kv_arithmetic_matches_what_vllm_independently_reported checks the KV figure
against a number this code did not produce -- vLLM's own start-up error said 4.25 GiB for a
17408-token request on Qwen3-32B. Checking the formula against a re-derivation of itself would
pass for any formula; this is the absolute reference the project conventions asks for.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

import teacher.engine as E

# Qwen3-32B geometry, and what vLLM said about it at start-up on 2026-08-21:
#   "(4.25 GiB KV cache is needed ... available KV cache memory (0.65 GiB) ...
#    estimated maximum model length is 2656"
QWEN3_32B = {"num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8,
             "head_dim": 128, "hidden_size": 5120, "torch_dtype": "bfloat16"}
VLLM_REPORTED_KV_GIB_AT_17408 = 4.25
A100_TOTAL_GIB = 79.2


def _fake_ckpt(tmp_path, cfg, weight_bytes, name="ckpt"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": weight_bytes}, "weight_map": {}}),
        encoding="utf-8")
    return str(d)


@pytest.fixture(autouse=True)
def _no_gpu_and_no_operator_override(monkeypatch):
    """Pin the two things the environment would otherwise decide for us: the card, and whether
    an operator has already overridden the value."""
    monkeypatch.delenv("OPD_TEACHER_UTIL", raising=False)
    monkeypatch.setattr(E, "device_total_gib", lambda index=0: A100_TOTAL_GIB)


def test_kv_arithmetic_matches_what_vllm_independently_reported(tmp_path):
    ck = _fake_ckpt(tmp_path, QWEN3_32B, int(61.03 * 2 ** 30))
    fp = E.checkpoint_footprint(ck, 17408)
    assert fp["kv_gib_per_request"] == pytest.approx(VLLM_REPORTED_KV_GIB_AT_17408, abs=0.02)
    assert fp["kv_bytes_per_token"] == 64 * 8 * 128 * 2 * 2      # layers x kv x dim x (K,V) x bf16
    assert fp["weights_gib"] == pytest.approx(61.03, abs=0.01)
    # ... and therefore the utilization the channel's 0.80 was short of.
    assert fp["required_utilization"] == pytest.approx((61.03 + 1.8 + 4.25) / 79.2, abs=0.002)


def test_head_dim_is_derived_when_the_config_omits_it(tmp_path):
    """Not every Qwen-family config carries head_dim explicitly; a wrong fallback would
    silently mis-size every model that omits it."""
    cfg = {k: v for k, v in QWEN3_32B.items() if k != "head_dim"}
    fp = E.checkpoint_footprint(_fake_ckpt(tmp_path, cfg, 2 ** 30), 1024)
    assert fp["kv_bytes_per_token"] == 64 * 8 * (5120 // 64) * 2 * 2


def test_a_model_the_channel_value_fits_keeps_the_channel_value(tmp_path, monkeypatch, capsys):
    """MiroThinker's own footprint -- the one 0.80 was certified against -- must not be
    perturbed. A sizing function that moved the certified case would be changing numbers on
    the strength of its own estimate."""
    monkeypatch.setenv("OPD_CHANNEL", "a100")
    # geometry chosen to reproduce the channel file's OWN recorded measurement: it says 0.80
    # gave MiroThinker "KV cache 5.1 GiB = 55,696 tokens", i.e. 96 KiB per token.
    miro = {"num_hidden_layers": 48, "num_attention_heads": 40, "num_key_value_heads": 4,
            "head_dim": 128, "hidden_size": 5120, "torch_dtype": "bfloat16"}
    assert 48 * 4 * 128 * 2 * 2 == 96 * 1024
    ck = _fake_ckpt(tmp_path, miro, int(56.93 * 2 ** 30))
    assert E.resolve_generation_utilization(ck, 17408) == 0.80
    assert "is sufficient" in capsys.readouterr().out
    # and the slack it reports is the channel's own: 0.80 x 79.2 - 56.93 - 1.8 - 1.59 = 2.99
    fp = E.checkpoint_footprint(ck, 17408)
    assert fp["kv_gib_per_request"] == pytest.approx(1.59, abs=0.02)


def test_a_heavier_model_raises_the_value_and_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPD_CHANNEL", "a100")
    ck = _fake_ckpt(tmp_path, QWEN3_32B, int(61.03 * 2 ** 30))
    util = E.resolve_generation_utilization(ck, 17408)
    assert util > 0.80
    assert util == pytest.approx((61.03 + 1.8 + 4.25) / 79.2 + 0.02, abs=0.002)
    out = capsys.readouterr().out
    assert "BELOW what this checkpoint needs" in out
    assert "scoring teacher" in out, "the message must say what it did NOT change"


def test_a_model_that_cannot_fit_refuses_instead_of_shrinking_the_context(tmp_path, monkeypatch):
    """The refusal is the point. The tempting fix for "it does not fit" is a smaller
    max_model_len -- which is rollout.max_len + 2048, so it would cap trajectory length on the
    strength of a memory error. the project conventions has that exact failure on record twice."""
    monkeypatch.setenv("OPD_CHANNEL", "a100")
    ck = _fake_ckpt(tmp_path, QWEN3_32B, int(74.0 * 2 ** 30))
    with pytest.raises(SystemExit, match="cannot be served"):
        E.resolve_generation_utilization(ck, 17408)


def test_an_operator_override_is_not_second_guessed(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OPD_CHANNEL", "a100")
    monkeypatch.setenv("OPD_TEACHER_UTIL", "0.66")
    ck = _fake_ckpt(tmp_path, QWEN3_32B, int(61.03 * 2 ** 30))
    # deliberately a value the sizing check would reject: a hand override must win, and must
    # not be quietly corrected upward into something the operator did not ask for.
    assert E.resolve_generation_utilization(ck, 17408) == 0.66
    assert "operator override" in capsys.readouterr().out


def test_an_unreadable_checkpoint_falls_back_loudly_rather_than_crashing(tmp_path,
                                                                        monkeypatch, capsys):
    monkeypatch.setenv("OPD_CHANNEL", "a100")
    d = tmp_path / "empty"
    d.mkdir()
    assert E.resolve_generation_utilization(str(d), 17408) == 0.80
    assert "could not size" in capsys.readouterr().out


def test_the_scoring_path_is_untouched(monkeypatch, capsys):
    """resolve_teacher_utilization is what precompute and both arms call, and it is a
    BackendConfig ENGINE_RUNTIME_FIELD -- gate 1 compares it. If sizing had been added there
    instead, every existing cache would stop matching a fresh run."""
    monkeypatch.setenv("OPD_CHANNEL", "a100")
    monkeypatch.delenv("OPD_TEACHER_UTIL", raising=False)
    assert E.resolve_teacher_utilization() == 0.80
    assert "configs/channels/a100.yaml" in capsys.readouterr().out


def test_stats_json_records_the_engine_the_generation_actually_used(tmp_path):
    """PRODUCER test. The utilization is now RESOLVED at runtime, so a reader cannot recover
    it from the config; delete the `engine` block in generate_sft_dataset and this goes red."""
    from envs.retriever import MockRetriever
    from envs.tokenizer import MockTokenizer
    from rollout.policy import MockPolicy
    from sft.datagen import SFTGenConfig, generate_sft_dataset
    from teacher.backend import BackendConfig

    tok = MockTokenizer(vocab_size=8192)
    mock = BackendConfig(engine="mock", dtype="mock", tokenizer_name="mock")
    out = str(tmp_path / "gen")
    generate_sft_dataset(
        questions=["capital of France?"], golds=["Paris"],
        policy=MockPolicy(tok, queries=["capital"], answer="Paris"),
        retriever=MockRetriever(["Paris is the capital of France."], seed=0), tokenizer=tok,
        cfg=SFTGenConfig(max_turns=4, max_len=4096, top_k=1),
        gen_backend=mock, teacher_backend=mock, out_dir=out,
        engine_kwargs={"max_model_len": 17408, "gpu_memory_utilization": 0.865})
    stats = json.load(open(os.path.join(out, "stats.json"), encoding="utf-8"))
    assert stats["engine"] == {"max_model_len": 17408, "gpu_memory_utilization": 0.865}


def test_stats_json_records_generation_throughput(tmp_path):
    """PRODUCER test. "How fast is this teacher" is a question the three-line comparison asks
    -- a dense 32B reads ~64 GB per decoded token where a 30B-A3B MoE reads ~6 GB -- and until
    now the only way to answer it was subtracting slurm log timestamps by hand. Delete the
    `throughput` block and this goes red."""
    from envs.retriever import MockRetriever
    from envs.tokenizer import MockTokenizer
    from rollout.policy import MockPolicy
    from sft.datagen import SFTGenConfig, generate_sft_dataset
    from teacher.backend import BackendConfig

    tok = MockTokenizer(vocab_size=8192)
    mock = BackendConfig(engine="mock", dtype="mock", tokenizer_name="mock")
    out = str(tmp_path / "gen")
    qs = ["capital of France?", "tallest tower?"]
    generate_sft_dataset(
        questions=qs, golds=["Paris"],
        policy=MockPolicy(tok, queries=["capital"], answer="Paris"),
        retriever=MockRetriever(["Paris is the capital of France."], seed=0), tokenizer=tok,
        cfg=SFTGenConfig(max_turns=4, max_len=4096, top_k=1),
        gen_backend=mock, teacher_backend=mock, out_dir=out)
    tp = json.load(open(os.path.join(out, "stats.json"), encoding="utf-8"))["throughput"]
    assert tp["n_generated"] == len(qs)
    assert tp["elapsed_s"] >= 0.0
    # the token totals must be REAL, not zeros a formatter happened to emit.
    assert tp["n_tokens_total"] > 0
    assert 0 < tp["n_action_tokens_total"] < tp["n_tokens_total"],         "action tokens are a strict subset -- observations are in the sequence too"
    # ... and counted over ALL generated trajectories, not only the kept ones, so a teacher
    # whose output the structural filter likes is not flattered.
    assert tp["n_tokens_total"] == sum(
        json.loads(l)["n_tokens"] for l in
        open(os.path.join(out, "manifest.jsonl"), encoding="utf-8") if l.strip())


def test_the_health_report_monitors_whether_the_action_cap_is_binding(tmp_path, capsys):
    """max_action_tokens=512 clipped 100% of one teacher's empty_query turns and none of
    another's, and read as a teacher-quality difference because nothing ever printed the
    at-cap rate. 2560 was chosen against an observed MAX of 2372 from 222 actions -- a
    3000-question shard will run a longer tail, so the cap has to be watched, not assumed."""
    import scripts.check_sft_stats as C

    # `q` present and non-empty == the turn produced a usable query, which is what makes it
    # PRODUCTIVE. The two kinds of at-cap turn are deliberately both here.
    rows = [{"turns_summary": [{"q": "who is x", "n_tok": 2560},   # productive AND clipped
                               {"q": "", "n_tok": 2560},           # runaway, bounded
                               {"q": "who is y", "n_tok": 40}],
             "final_action_n_tok": 100},
            {"turns_summary": [{"q": "", "n_tok": 2556}],          # runaway, bounded
             "final_action_n_tok": 30}]
    got = C._action_cap_pressure(rows, 2560)
    assert got["n_actions"] == 6
    # 2556 counts: the tolerance is 8 tokens, because a stop string landing one token inside
    # the ceiling is still an action that ran out of room.
    assert got["at_cap"] == 3
    # THE SPLIT, and why it exists. Pooled, this shard is 50% at-cap and the alarm fires. But
    # only ONE of those three is a turn that produced a query and was cut off; the other two
    # emitted nothing and were bounded, which is what a backstop does. On runs/prod_moe_sft
    # the pooled rate was 1.48% -- "regenerate an 11-hour shard" -- against 0.04% productive.
    assert got["n_productive"] == 4          # 2 query turns + 2 final actions
    assert got["at_cap_productive"] == 1
    assert got["at_cap_productive_frac"] == 0.25
    # 2556 counts: the tolerance is 8 tokens, because a stop string that lands one token
    # inside the ceiling is still an action that ran out of room.
    assert got["at_cap_frac"] == 0.5

    # ... and dropped rows are counted too. A clipped action is the usual REASON for a drop,
    # so counting survivors only would hide exactly the cases this exists to find.
    assert C._action_cap_pressure([{"kept": False, "reason": "empty_query",
                                    "turns_summary": [{"n_tok": 2560}]}], 2560)["at_cap"] == 1

    # An unrecorded cap must read as UNKNOWN, never be defaulted to 512: "the cap never bit"
    # and "nobody knows what the cap was" are the two things a reader must not confuse.
    assert C._action_cap_pressure(rows, None)["at_cap_frac"] is None
    d = tmp_path / "old"
    d.mkdir()
    (d / "stats.json").write_text(json.dumps({"gen_config": {"max_len": 15360}}),
                                  encoding="utf-8")
    assert C._read_gen_action_cap(str(d)) is None
    (d / "stats.json").write_text(
        json.dumps({"engine": {"max_action_tokens": 2560}}), encoding="utf-8")
    assert C._read_gen_action_cap(str(d)) == 2560


# --------------------------------------------------------------- the server's own preflight
# THE INCIDENT (job 27001269, 2026-09-03): slurm/teacher_server.py was pointed at Qwen3-32B and
# inherited configs/channels/a100.yaml's 0.80, which was certified by MEASURING it against
# MiroThinker's 56.93 GiB of weights. Qwen3-32B is 61.03 GiB. vLLM loaded for 64 s and died
# with "Available KV cache memory: 0.79 GiB" against the 4.25 GiB one max-length request needs.
# Everything needed to refuse was on disk before a byte was loaded.
def _server_src():
    return (pathlib.Path(__file__).resolve().parents[1] / "slurm" / "teacher_server.py").read_text(encoding="utf-8")


def test_teacher_server_sizes_the_checkpoint_before_building_the_engine():
    """PRODUCER side: the refusal must sit between resolving the kwargs and constructing the
    engine. Deleting the block, or moving it after _build_scorer, makes this red."""
    src = _server_src()
    assert "checkpoint_footprint" in src, "the server must size the checkpoint it was given"
    i_kw = src.index("engine_kwargs = teacher_engine_kwargs(")
    i_chk = src.index("fp = checkpoint_footprint(args.model")
    i_refuse = src.index("cannot start at gpu_memory_utilization=")
    # the CALL, not the def: `_build_scorer(` alone matches the definition 130 lines earlier,
    # which sits before main() entirely and made this assertion compare the wrong two points.
    i_build = src.index("scorer_fn = _build_scorer(")
    assert i_kw < i_chk < i_refuse < i_build, (
        f"sizing must run after the kwargs are resolved and BEFORE the engine is built "
        f"(kwargs@{i_kw} check@{i_chk} refuse@{i_refuse} build@{i_build})")
    assert "OPD_TEACHER_UTIL" in src[i_refuse:i_refuse + 800], (
        "the refusal must name the override that fixes it, not just report a number")


def test_the_refusal_condition_is_the_one_that_fired_in_production():
    """The arithmetic, on the real numbers from that log: at 0.80 the requirement exceeds the
    resolved value, at 0.858 it does not. checkpoint_footprint's own overhead constant is used,
    so this tracks the function rather than restating a constant."""
    total = 80.00                      # nvidia-smi board total, GiB (what device_total_gib reports)
    weights, kv_one = 61.03, 4.25      # both MEASURED in job 27001269's log
    need = weights + E._ENGINE_OVERHEAD_GIB + kv_one
    required = need / total
    assert required > 0.80, f"required {required:.3f} must exceed the channel value that failed"
    assert required <= 0.858, f"required {required:.3f} must be met by the dense line's 0.858"


def test_miro_still_starts_under_the_channel_value():
    """Non-vacuity in the other direction: the refusal must NOT fire for the checkpoint the
    channel value was certified against, or it would block every production teacher."""
    total = 80.00
    weights, kv_one = 56.93, 4.25
    required = (weights + E._ENGINE_OVERHEAD_GIB + kv_one) / total
    assert required < 0.80, f"required {required:.3f} would falsely refuse MiroThinker at 0.80"
