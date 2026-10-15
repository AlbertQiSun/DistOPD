"""PRODUCER test: the configured action budget must reach vLLM's SamplingParams.

THE FAILURE THIS ENCODES. `max_action_tokens` lived in configs/opd_qwen3.yaml at 2560 and
as a Python default of 512 in VLLMPolicy.__init__. Only ONE of the four production call
sites passed it. gen_sft_data.py did, so all three teachers' SFT data was generated at
2560; train.py, stage3_eval.py and policy_invariance_probe.py did not, so pi_ref would have
been TRAINED on 2560-token demonstrations and then rolled out and evaluated at 512.

Why that is a confound and not a rounding error -- measured offline on the production
shards (runs/prod_{miro,moe}_sft/manifest.jsonl, 2026-08-23, from the per-action n_tok that
the project conventions's persist rule required be written):

    fraction of teacher actions longer than 512     MiroThinker      1.29%
                                                    Qwen3-30B-A3B   24.68%

A truncated action ends mid-thought with no closing tag; rollout routes every non-answer
action into the search branch, and Gate A scores it a format failure. At 512 the MoE line
would have lost ~19x more actions than the MiroThinker line -- an artifact of the sampler,
in exactly the cross-teacher comparison the whole experiment exists to make.

WHY THESE TESTS AND NOT A GREP. the project conventions: delete the line that writes the field and
something must go red. Here the "field" is a kwarg, so the tests call the real factory
functions with a stubbed engine and read what they actually handed the policy. Asserting on
the source text alone would pass for a call site that passed a hardcoded 512.

test_the_default_is_gone is the structural half: it makes a FUTURE call site that forgets
the argument a TypeError rather than a silent 512.
"""
from __future__ import annotations

import glob
import inspect
import sys
import types

import pytest

CONFIG_CAP = 2560          # what configs/opd_qwen3.yaml pins; asserted below, not assumed


# --------------------------------------------------------------------------- stubs
class _CapturingPolicy:
    """Stands in for VLLMPolicy and records the kwargs the factory chose."""
    last = None

    def __init__(self, llm, tokenizer, **kw):
        _CapturingPolicy.last = kw


@pytest.fixture
def stub_vllm(monkeypatch):
    """Make `from vllm import LLM` and `from rollout.vllm_policy import VLLMPolicy` cheap.

    Both imports happen INSIDE the factories (they are lazy on purpose -- vllm is HPC
    only), so the stubs have to live in sys.modules before the call, not around it.
    """
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = lambda *a, **k: object()
    fake_vllm.SamplingParams = object
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    import rollout.vllm_policy as VP
    monkeypatch.setattr(VP, "VLLMPolicy", _CapturingPolicy)

    import rollout.engine as RE
    monkeypatch.setattr(RE, "pin_policy_env", lambda: None)
    monkeypatch.setattr(RE, "policy_engine_kwargs", lambda **k: {})

    _CapturingPolicy.last = None
    return _CapturingPolicy


def _cfg(cap=CONFIG_CAP):
    return {"model": {"student": "/nonexistent", "dtype": "bfloat16"},
            "run": {"seed": 7},
            "rollout": {"temperature": 0.8, "top_p": 1.0, "max_action_tokens": cap}}


# --------------------------------------------------------------------------- the config
def _raw(path):
    """yaml.safe_load, NOT opd.config.load_config.

    load_config resolves ${SITE_VAR} references and raises when they are unset, so it can
    only run inside a sourced site environment -- which a laptop test run is not. Nothing
    in the rollout block is a site variable, so reading the file directly asks exactly the
    question these tests ask and no more.
    """
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_the_shipped_config_still_pins_the_cap_these_tests_assume():
    """If opd_qwen3.yaml moves, this file's premise moves with it and must be re-read."""
    assert _raw("configs/opd_qwen3.yaml")["rollout"]["max_action_tokens"] == CONFIG_CAP


def test_every_shipped_config_carries_the_key():
    """VLLMPolicy has no default any more, and the call sites index rather than .get(),
    so a config missing the key is a crash at rollout time -- on a GPU node, minutes in."""
    seen = 0
    for path in sorted(glob.glob("configs/*.yaml")):
        roll = (_raw(path) or {}).get("rollout")
        if roll is None:
            continue
        seen += 1
        assert "max_action_tokens" in roll, f"{path} has no rollout.max_action_tokens"
    assert seen >= 2, "expected at least opd_qwen3.yaml and the smoke config to be scanned"


# --------------------------------------------------------------------------- producers
def test_train_passes_the_configured_cap(stub_vllm):
    import scripts.train as train
    train.make_policy(_cfg(), dry_run=False, tokenizer=object())
    assert stub_vllm.last["max_action_tokens"] == CONFIG_CAP


def test_train_follows_the_config_rather_than_a_constant(stub_vllm):
    """A hardcoded 2560 would pass the test above. This one moves the config."""
    import scripts.train as train
    train.make_policy(_cfg(cap=777), dry_run=False, tokenizer=object())
    assert stub_vllm.last["max_action_tokens"] == 777


def test_stage3_eval_passes_the_configured_cap(stub_vllm):
    import scripts.stage3_eval as s3
    s3._make_policy("/nonexistent", object(), False, temperature=0.8, top_p=1.0,
                    dtype="bfloat16", seed=7, max_action_tokens=CONFIG_CAP)
    assert stub_vllm.last["max_action_tokens"] == CONFIG_CAP


def test_stage3_eval_reads_the_cap_out_of_the_config_block():
    """_make_policy takes the cap as an argument, so the test above cannot prove run_eval
    supplies it from the config. Read the source of run_eval for the binding instead --
    the one place where a static check is the honest one, because the alternative is
    standing up a whole eval run.
    """
    import scripts.stage3_eval as s3
    src = inspect.getsource(s3.run_eval)
    assert 'roll["max_action_tokens"]' in src, (
        "run_eval must take the cap from the config block; a literal or a .get() default "
        "would reintroduce the 512 confound this file documents")
    assert "max_action_tokens=max_action_tokens" in src


def test_the_eval_report_records_the_cap_it_ran_under():
    """PERSIST rule. Two Stage 3 reports produced at different caps are otherwise
    indistinguishable, and the cap is exactly what would explain a Gate A difference."""
    import scripts.stage3_eval as s3
    src = inspect.getsource(s3.run_eval)
    assert '"max_action_tokens": max_action_tokens' in src


def test_probe_passes_the_configured_cap():
    """policy_invariance_probe builds its policy inside run_cost, which also loads data and
    a retriever; assert on the construction expression rather than executing all of that."""
    import scripts.policy_invariance_probe as P
    src = inspect.getsource(P.run_cost)
    assert 'max_action_tokens=cfg["rollout"]["max_action_tokens"]' in src


def test_gen_sft_data_still_passes_it():
    """The one call site that was always correct. It is in here so that a refactor which
    'tidies' the cap out of the generator -- the arm that DEFINES the 2560 the others now
    follow -- cannot pass silently."""
    import scripts.gen_sft_data as G
    src = inspect.getsource(G)
    assert "max_action_tokens=cap" in src


# --------------------------------------------------------------------------- structure
def test_the_default_is_gone():
    """The structural half. With a default, a fifth call site added next month inherits
    512 and nothing fails; without one, it is a TypeError at construction."""
    from rollout.vllm_policy import VLLMPolicy
    p = inspect.signature(VLLMPolicy.__init__).parameters["max_action_tokens"]
    assert p.default is inspect.Parameter.empty, (
        "max_action_tokens must stay REQUIRED -- a default is what let three of four "
        "production call sites sample at a number nobody chose")
    assert p.kind is inspect.Parameter.KEYWORD_ONLY


def test_the_cap_reaches_the_sampling_params():
    """End of the chain: the stored attribute must actually become vLLM's max_tokens.
    Everything above proves the value reaches the constructor; this proves the constructor
    does not then drop it."""
    from rollout.vllm_policy import VLLMPolicy

    captured = {}

    class _SP:
        def __init__(self, **kw):
            captured.update(kw)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = _SP
    sys.modules["vllm"] = fake_vllm
    try:
        pol = VLLMPolicy(object(), object(), max_action_tokens=1234)
        pol._sampling_params()
    finally:
        del sys.modules["vllm"]
    assert captured["max_tokens"] == 1234
