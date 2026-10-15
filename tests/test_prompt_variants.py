"""The prompt A/B mechanism: a switchable SYSTEM_PROMPT that cannot silently move the default.

WHY A SWITCH AT ALL. Deciding whether to replace the SYSTEM_PROMPT costs a full three-line
regeneration (template_fingerprint is part of gate 1, so every arm must be produced under the
same prompt or the comparison is not clean). That decision has to be made on data, which means
running two prompts side by side -- and doing that by editing the frozen constant would leave
no way to tell the two output directories apart afterwards, and no way to get back.

WHAT THESE TESTS PROTECT, in order of how expensive the failure is:

  1. The DEFAULT is byte-identical to what every existing artifact was produced under. The
     fingerprint is asserted as a LITERAL (f9da237f476eb71e), not against a recomputation of
     itself -- a self-comparison would pass no matter what the prompt said. sft_shard1, pi_ref
     and the whole Row 10 precompute are downstream of that string.
  2. The variant is RECORDED where a reader will find it. Two shards generated a day apart are
     otherwise indistinguishable; stats.json must carry the variant name AND the prompt text.
  3. The switch FAILS LOUD on a typo. An unrecognised OPD_PROMPT_VARIANT that fell back to the
     default would produce a control arm labelled as a treatment arm -- the one outcome worse
     than a crash, because it looks like a result.

Reloading a module to test an env var is normally a smell; here the env var is read at import
time ON PURPOSE (SYSTEM_PROMPT must be a constant for the rest of the process), so importlib
is the only honest way to exercise it.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest

import configs.template as T
from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from rollout.policy import MockPolicy
from sft.datagen import SFTGenConfig, generate_sft_dataset
from teacher.backend import BackendConfig, template_fingerprint

# The fingerprint of the CURRENT default. Written out rather than computed: this line is the
# tripwire, and a tripwire that recomputes its own trigger is not one.
#
# 2026-08-22: f9da237f476eb71e (v2_terse) -> 3165062cda7688de (v3_terse_shortans), decided on
# the ab4 round. Every artifact made before that commit -- sft_shard1, pi_ref, the Row 10
# precompute -- is downstream of the OLD string and must be regenerated. That was already
# owed: max_action_tokens moved 512 -> 2560 in the same window.
FROZEN_DEFAULT_FP = "3165062cda7688de"
PREVIOUS_DEFAULT_FP = "f9da237f476eb71e"
CORPUS = ["Paris is the capital of France."]
MOCK_BACKEND = BackendConfig(engine="mock", dtype="mock", tokenizer_name="mock")


def _reload_with(variant):
    """Re-import configs.template under a given OPD_PROMPT_VARIANT and SNAPSHOT what it became,
    then restore the process to the default.

    It returns a snapshot rather than the module because configs.template is a singleton: the
    restoring reload mutates the very object a caller would be holding, so returning `T` would
    hand back the DEFAULT and every assertion about the variant would silently pass on the
    control. (It did, before this comment existed.)"""
    prev = os.environ.get("OPD_PROMPT_VARIANT")
    try:
        if variant is None:
            os.environ.pop("OPD_PROMPT_VARIANT", None)
        else:
            os.environ["OPD_PROMPT_VARIANT"] = variant
        mod = importlib.reload(T)
        return {"variant": mod.PROMPT_VARIANT, "prompt": mod.SYSTEM_PROMPT,
                "variants": dict(mod.PROMPT_VARIANTS),
                "fingerprint": template_fingerprint()}
    finally:
        if prev is None:
            os.environ.pop("OPD_PROMPT_VARIANT", None)
        else:
            os.environ["OPD_PROMPT_VARIANT"] = prev
        importlib.reload(T)


def test_the_default_prompt_and_its_fingerprint_have_not_moved():
    assert T.PROMPT_VARIANT == "v3_terse_shortans"
    assert T.SYSTEM_PROMPT == T.PROMPT_VARIANTS["v3_terse_shortans"]
    assert template_fingerprint() == FROZEN_DEFAULT_FP, (
        "the default SYSTEM_PROMPT changed. Everything downstream of it -- the SFT shard, "
        "pi_ref, the OPD precompute -- was produced under " + FROZEN_DEFAULT_FP + ". If this "
        "change is intended, it invalidates all of them and the constant above must be "
        "updated in the same commit that regenerates them.")
    # ... and the OLD default is still reachable by name, so a shard that records
    # f9da237f476eb71e can still be reproduced without checking out an old commit.
    assert template_fingerprint() != PREVIOUS_DEFAULT_FP
    assert "v2_terse" in T.PROMPT_VARIANTS


def test_an_unset_env_var_is_the_default():
    snap = _reload_with(None)
    assert snap["variant"] == "v3_terse_shortans"
    assert snap["fingerprint"] == FROZEN_DEFAULT_FP


def test_selecting_the_previous_default_reproduces_its_fingerprint_exactly():
    """The old shards are not orphaned: OPD_PROMPT_VARIANT=v2_terse still produces
    f9da237f476eb71e, so anything recorded under it can be regenerated from this commit."""
    snap = _reload_with("v2_terse")
    assert snap["fingerprint"] == PREVIOUS_DEFAULT_FP


def test_selecting_a_variant_changes_the_prompt_and_the_fingerprint():
    snap = _reload_with("v1_asp")
    assert snap["variant"] == "v1_asp"
    assert snap["prompt"] == snap["variants"]["v1_asp"]
    # the fingerprint MUST move -- that is what stops an A/B arm from being mistaken for the
    # control by gate 1 downstream.
    assert snap["fingerprint"] != FROZEN_DEFAULT_FP
    # ... and the process is back on the default afterwards, so test order cannot leak.
    assert template_fingerprint() == FROZEN_DEFAULT_FP


def test_an_unknown_variant_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="not a registered prompt variant"):
        _reload_with("v1-asp")          # a plausible typo: hyphen instead of underscore


def test_every_variant_still_defines_the_protocol_the_env_parses():
    """A variant that forgot a tag would produce a shard of empty_query drops and look like a
    teacher problem. rollout only ever parses <search>/</search> and <answer>/</answer>."""
    for name, prompt in T.PROMPT_VARIANTS.items():
        for tag in (T.SEARCH_OPEN, T.SEARCH_CLOSE, T.ANSWER_OPEN, T.ANSWER_CLOSE,
                    T.INFO_OPEN, T.INFO_CLOSE):
            assert tag in prompt, f"{name} never mentions {tag}"


def test_v3_is_v2_plus_the_answer_content_instruction_and_nothing_else():
    """v3 is the single-variable arm; if it silently became a rewrite it could not attribute
    anything. Its value is precisely that v2 is a PREFIX of it."""
    v2, v3 = T.PROMPT_VARIANTS["v2_terse"], T.PROMPT_VARIANTS["v3_terse_shortans"]
    assert v3.startswith(v2)
    delta = v3[len(v2):]
    assert "short-form" in delta and "nothing else" in delta
    assert T.SEARCH_OPEN not in delta, "v3 must not also change the search instructions"


def test_only_v1_carries_a_worked_example():
    """The example is the largest single difference and the one with a known failure mode (a
    concrete entity in the prompt becomes an attractor the model copies under uncertainty --
    measured at 12.5% with the old <answer>Beijing</answer>). Which variants carry that risk
    must be a fact, not a memory."""
    assert "### Example" in T.PROMPT_VARIANTS["v1_asp"]
    assert "MacMahon" in T.PROMPT_VARIANTS["v1_asp"]
    for name in ("v2_terse", "v3_terse_shortans"):
        assert "### Example" not in T.PROMPT_VARIANTS[name]
    # ... and the answer SLOT stays a placeholder even in v1.
    assert "[your concise short-form final answer goes here]" in T.PROMPT_VARIANTS["v1_asp"]


def test_only_v1_and_v3_say_what_may_be_inside_the_answer_span():
    """The gap under test. v2 says WHERE to put the answer and never says WHAT may be in it --
    which is the hypothesis for 90-word answer spans and deliberation inside the tags."""
    def constrains_answer_content(p):
        low = p.lower()
        return ("short-form" in low or "short phrase" in low) and \
               ("nothing else" in low or "not include any other" in low)
    assert not constrains_answer_content(T.PROMPT_VARIANTS["v2_terse"])
    assert constrains_answer_content(T.PROMPT_VARIANTS["v1_asp"])
    assert constrains_answer_content(T.PROMPT_VARIANTS["v3_terse_shortans"])


def test_stats_json_records_which_prompt_produced_the_shard(tmp_path):
    """PRODUCER test. Delete the `template` block in generate_sft_dataset and this goes red;
    a test that hand-built a stats dict would not notice."""
    tok = MockTokenizer(vocab_size=8192)
    out = str(tmp_path / "gen")
    generate_sft_dataset(
        questions=["capital of France?"], golds=["Paris"],
        policy=MockPolicy(tok, queries=["capital"], answer="Paris"),
        retriever=MockRetriever(CORPUS, seed=0), tokenizer=tok,
        cfg=SFTGenConfig(max_turns=4, max_len=4096, top_k=1),
        gen_backend=MOCK_BACKEND, teacher_backend=MOCK_BACKEND, out_dir=out)
    stats = json.load(open(os.path.join(out, "stats.json"), encoding="utf-8"))
    tpl = stats["template"]
    assert tpl["fingerprint"] == template_fingerprint()
    assert tpl["prompt_variant"] == T.PROMPT_VARIANT
    # the full TEXT, not only the name: the name points at a file that keeps changing.
    assert tpl["system_prompt"] == T.SYSTEM_PROMPT


def test_the_manifest_records_the_answer_text_a_prompt_ab_has_to_read(tmp_path):
    """PRODUCER test for final_answer / final_action_text. The A/B's second question is "did
    the answers get shorter and cleaner"; neither is answerable from an aggregate."""
    tok = MockTokenizer(vocab_size=8192)
    out = str(tmp_path / "gen")
    generate_sft_dataset(
        questions=["capital of France?"], golds=["Paris"],
        policy=MockPolicy(tok, queries=["capital"], answer="Paris"),
        retriever=MockRetriever(CORPUS, seed=0), tokenizer=tok,
        cfg=SFTGenConfig(max_turns=4, max_len=4096, top_k=1),
        gen_backend=MOCK_BACKEND, teacher_backend=MOCK_BACKEND, out_dir=out)
    row = json.loads(open(os.path.join(out, "manifest.jsonl"), encoding="utf-8").readline())
    assert row["kept"] is True
    assert row["final_answer"] == "Paris"
    # the WHOLE final action, so contamination inside the span is visible: the extracted
    # answer is "Paris", the action also carries the think block that preceded it.
    assert row["final_action_text"].startswith(T.THINK_OPEN)
    assert T.ANSWER_OPEN + "Paris" + T.ANSWER_CLOSE in row["final_action_text"]


def test_a_run_refuses_when_it_does_not_match_what_the_submit_command_asked_for():
    """Twice in one session a job ran against a checkout that predated the change it was meant
    to test -- once OPD_PROMPT_VARIANT did not exist so both A/B arms silently ran the control
    prompt, once the config still said max_action_tokens 512 where the run needed 2560. Neither
    failed. Both produced shards that looked normal, and the fixed seed made the wrong result
    byte-identical to an earlier one, which reads as reproducibility rather than as failure.

    The guard cannot live in the reader: by the time anyone reads the shard, the GPU hours are
    spent. It lives in the submit command, which knows what it intends."""
    from scripts.gen_sft_data import assert_expectations

    cfg = {"rollout": {"max_action_tokens": 2560}}
    assert assert_expectations(cfg, resolved_variant="v2_terse", cli_cap=None,
                               expect_cap=2560, expect_variant="v2_terse") == 2560

    # the stale-config case, verbatim: config still says 512, submit asked for 2560
    with pytest.raises(SystemExit, match="expect-action-cap 2560"):
        assert_expectations({"rollout": {"max_action_tokens": 512}},
                            resolved_variant="v2_terse", cli_cap=None,
                            expect_cap=2560, expect_variant=None)

    # the stale-code case: the variant knob did not exist, so everything is the default
    with pytest.raises(SystemExit, match="predates the variant"):
        assert_expectations(cfg, resolved_variant="v2_terse", cli_cap=None,
                            expect_cap=None, expect_variant="v3_terse_shortans")

    # a CLI override still wins over the config, and is what gets checked
    assert assert_expectations({"rollout": {"max_action_tokens": 512}},
                               resolved_variant="v2_terse", cli_cap=4096,
                               expect_cap=4096, expect_variant=None) == 4096

    # ... and with no expectation given, nothing is enforced: the flags are opt-in
    assert assert_expectations({}, resolved_variant="v2_terse", cli_cap=None,
                               expect_cap=None, expect_variant=None) == 512
