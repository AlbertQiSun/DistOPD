"""`steps x global_batch` must equal `limit x epochs`, declared -- not inherited.

WHAT THIS GUARDS. Both sources walk questions[i % limit], but a revisit means different
things on each side: ONLINE re-samples the policy and gets a fresh trajectory, OFFLINE
re-reads the same cached one. So at E = draws/limit epochs, online sees E distinct
trajectories per question and offline sees one repeated E times. E is a second independent
variable -- data diversity -- sitting right next to the one the experiment is about, and the
combination this repo shipped (limit 64, steps 150, global_batch 256) had E = 600.

E = 1 removes it, and additionally PAIRS the arms: step k draws the same questions on both
sides, which is a stronger comparison than two independent draws.

Any other E is allowed -- a smoke run legitimately reuses a small cache -- but it has to be
written down, because nothing else on disk distinguishes a smoke test from an experiment.

THE COMMITTED CONFIG IS PART OF THE TEST. A rule the code enforces against a config that
violates it is a rule that stops the next real run, so the shipped values are checked here
too. That is the `global_batch` lesson: the config saying the right thing and the code
reading it are two different facts, and so is the config saying the right thing at all.
"""
from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from opd.trainer import OPDConfig
from scripts.train import assert_epochs_declared

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "opd_qwen3.yaml"


def _cfg(epochs):
    return {"data": {"epochs": epochs}} if epochs is not None else {"data": {}}


def test_a_matching_declaration_passes():
    got = assert_epochs_declared(_cfg(1), OPDConfig(steps=60, global_batch=256), 15360)
    assert got == 1


def test_an_undeclared_epoch_count_is_refused():
    with pytest.raises(AssertionError, match="data.epochs is not declared"):
        assert_epochs_declared(_cfg(None), OPDConfig(steps=150, global_batch=256), 64)


def test_the_shipped_confound_is_named_in_the_error():
    """The refusal has to tell the reader WHAT number they were about to inherit -- 600 is the
    diversity ratio, not a typo, and the message is the only place it appears."""
    with pytest.raises(AssertionError) as e:
        assert_epochs_declared(_cfg(None), OPDConfig(steps=150, global_batch=256), 64)
    assert "600 epochs" in str(e.value)
    assert "38400 draws" in str(e.value)


def test_a_wrong_declaration_is_refused():
    with pytest.raises(AssertionError, match="must equal"):
        assert_epochs_declared(_cfg(1), OPDConfig(steps=150, global_batch=256), 64)


def test_a_declared_smoke_run_is_allowed():
    """Reusing a small cache is legitimate when it is stated. 40 x 256 over 64 questions."""
    assert assert_epochs_declared(_cfg(160), OPDConfig(steps=40, global_batch=256), 64) == 160


def test_an_empty_dataset_is_refused():
    with pytest.raises(AssertionError, match="dataset is empty"):
        assert_epochs_declared(_cfg(1), OPDConfig(steps=4, global_batch=1), 0)


def test_the_COMMITTED_config_satisfies_its_own_rule():
    """The rule is worthless if the shipped config breaks it -- that is a run that refuses to
    start, discovered at the front of a queue wait."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    steps, gb = cfg["optim"]["steps"], cfg["optim"]["global_batch"]
    limit, epochs = cfg["data"]["limit"], cfg["data"]["epochs"]
    assert steps * gb == limit * epochs, (
        f"committed config: steps({steps}) x global_batch({gb}) = {steps * gb} but "
        f"limit({limit}) x epochs({epochs}) = {limit * epochs}")
    assert epochs == 1, "the committed config must be the apples-to-apples setting"


def test_the_committed_config_passes_the_real_assertion():
    """Not just arithmetic on the yaml -- the actual function, on the actual values."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    opd = OPDConfig(steps=cfg["optim"]["steps"], global_batch=cfg["optim"]["global_batch"])
    assert assert_epochs_declared(cfg, opd, cfg["data"]["limit"]) == 1


def test_the_planned_scale_fits_the_measured_walltime():
    """steps is a BUDGET decision as much as a scientific one. Measured on H20: 1.348 s of
    training per trajectory (weighted over the real length distribution) plus 1.55 s of
    rollout, and the online arm does both in ONE job against a 48h walltime. If steps is
    raised past what fits, that is a choice to make deliberately -- with resume in place --
    not one to discover when SLURM kills the run.

    48, not 24: 24 was a value put into the sbatch files earlier in development and then
    mistakenly reasoned about as if it were the cluster's limit. The rest of this repo had it
    right all along (rollout/engine.py, opd/trainer.py, scripts/policy_invariance_probe.py
    all budget against 48h)."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    draws = cfg["optim"]["steps"] * cfg["optim"]["global_batch"]
    online_hours = draws * (1.348 + 1.55) / 3600
    assert online_hours < 48, (
        f"the online arm projects to {online_hours:.1f}h for {draws} trajectories, past the "
        "48h walltime. Either lower steps or land checkpoint/resume first.")
