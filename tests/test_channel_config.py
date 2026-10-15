"""A channel file carries hardware accommodations and MUST NOT carry the science.

Two config files that can both set a run's meaning is one file too many: the second wins, and
nothing says which one did. So `apply_channel` enforces an allowlist -- a channel may set the
teacher's memory budget, the trainable length ceiling, the student's device, and the two
memory/speed knobs, and nothing else. steps, lr, global_batch, epochs, tau, seed and the data
slice live in the base config alone.

The knobs a channel does set are chosen to be NUMERICALLY INERT. `autocast_cache` decides
whether autocast keeps a bf16 copy of every weight or re-casts per use; `logprob_chunk_rows`
regroups a per-row-exact computation. Both trade memory for speed and neither can move a
number -- which is what makes it acceptable for the two channels to differ, given that their
results are never compared anyway.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import CHANNEL_ALLOWLIST, apply_channel

CHANNEL_DIR = ROOT / "configs" / "channels"
BASE = ROOT / "configs" / "opd_qwen3.yaml"
CHANNELS = sorted(p.stem for p in CHANNEL_DIR.glob("*.yaml"))

# the fields that decide what the experiment MEANS. A channel touching any of these would be
# changing the science from a file named after a GPU.
SCIENCE = [("optim", "steps"), ("optim", "lr"), ("optim", "global_batch"),
           ("optim", "weight_decay"), ("optim", "adam_beta1"), ("optim", "adam_beta2"),
           ("opd", "tau"), ("data", "limit"), ("data", "epochs"), ("data", "offset"),
           ("data", "path"), ("run", "seed"), ("rollout", "temperature"),
           ("rollout", "top_k"), ("rollout", "max_turns"),
           ("online", "resync_every"), ("model", "student"), ("model", "teacher"),
           ("model", "student_param_dtype"), ("model", "engine")]


def _base():
    return yaml.safe_load(BASE.read_text(encoding="utf-8"))


def test_there_are_channels_to_check():
    assert CHANNELS, "no channel files -- this whole file would be vacuous"


@pytest.mark.parametrize("channel", CHANNELS)
def test_a_channel_cannot_change_the_science(channel):
    """THE GATE. Compare every science field before and after the merge."""
    before, after = _base(), apply_channel(_base(), channel)
    for section, key in SCIENCE:
        if section in before and key in before[section]:
            assert after[section][key] == before[section][key], (
                f"channel {channel!r} changed {section}.{key}: "
                f"{before[section][key]!r} -> {after[section][key]!r}")


@pytest.mark.parametrize("channel", CHANNELS)
def test_a_channel_only_sets_allowlisted_keys(channel):
    overlay = yaml.safe_load((CHANNEL_DIR / f"{channel}.yaml").read_text(encoding="utf-8"))
    overlay.pop("name", None)
    for section, values in overlay.items():
        for key in values:
            assert (section, key) in CHANNEL_ALLOWLIST, (
                f"{channel}.yaml sets {section}.{key}, which is not a hardware accommodation")


def test_a_channel_that_oversteps_is_REFUSED(tmp_path, monkeypatch):
    """The allowlist has to bind, or it is a comment. Write a rogue channel and confirm the
    merge refuses it rather than quietly applying it."""
    rogue = CHANNEL_DIR / "_rogue_test.yaml"
    rogue.write_text("name: rogue\noptim:\n  steps: 3\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="not a hardware accommodation"):
            apply_channel(_base(), "_rogue_test")
    finally:
        rogue.unlink()


def test_an_unknown_channel_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unknown channel"):
        apply_channel(_base(), "no_such_card")


@pytest.mark.parametrize("channel", CHANNELS)
def test_the_merge_actually_applies(channel):
    """Non-vacuity: at least one value must differ from the base, or the file does nothing and
    a reader would take its presence as evidence of an accommodation that is not happening."""
    overlay = yaml.safe_load((CHANNEL_DIR / f"{channel}.yaml").read_text(encoding="utf-8"))
    overlay.pop("name", None)
    after = apply_channel(_base(), channel)
    for section, values in overlay.items():
        for key, want in values.items():
            assert after[section][key] == want, f"{channel}: {section}.{key} did not apply"
    assert after["channel"] == channel


def test_the_channels_differ_from_each_other_where_it_matters():
    """If two channels were identical the split would be ceremony. The A100 has 15.9 GiB less
    card than the H20, so its memory knobs must actually be different."""
    if len(CHANNELS) < 2:
        pytest.skip("only one channel defined")
    h20, a100 = apply_channel(_base(), "h20"), apply_channel(_base(), "a100")
    assert h20["model"]["autocast_cache"] is True
    assert a100["model"]["autocast_cache"] is False, (
        "the A100 channel must drop the autocast weight-cast cache -- ~7.5 GiB it does not have")
    assert a100["opd"]["logprob_chunk_rows"] < h20["opd"]["logprob_chunk_rows"]


def test_a_real_run_refuses_to_start_without_a_channel():
    """Leaving it implicit is exactly how a cross-channel mix happens."""
    import scripts.train as T

    with pytest.raises(ValueError, match="must name its hardware channel"):
        T.run(str(BASE), "offline", "runs", dry_run=False, channel=None)


def test_the_scorer_honours_both_knobs():
    """The config saying so proves nothing about what the code does -- the lesson from
    global_batch. Check the values reach the object that uses them."""
    import torch

    from teacher.scorer import MockLM, Scorer

    s = Scorer(MockLM(64, seed=0), requires_grad=False, autocast_dtype="bfloat16",
               autocast_cache=False, logprob_chunk_rows=7)
    ctx = s._autocast(torch.device("cpu"))
    assert getattr(ctx, "_cache_enabled", True) is False, "autocast_cache did not reach autocast"
    assert s._chunk_kw() == {"chunk_rows": 7}

    default = Scorer(MockLM(64, seed=0), requires_grad=False)
    assert default._chunk_kw() == {}, (
        "an unset chunk_rows must fall through to opd.masking's single default rather than "
        "hardcoding a second one here")
