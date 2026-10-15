"""scripts/train.py::run -- the entry point every arm goes through, and nothing called it.

THE FAILURE THIS ENCODES. `_opd_config` was edited to honour --precompute-dir:

    def _opd_config(cfg: dict, mode: str) -> OPDConfig:          # <- signature unchanged
        ...
        precompute_dir=_resolve_precompute_dir(
            precompute_dir or cfg["diagnostics"].get("precompute_dir"), cfg, dry_run),

`precompute_dir` and `dry_run` are locals of run(), not of _opd_config, and there are no
module-level names by those spellings -- so this raised NameError on EVERY invocation, dry or
real. --precompute-dir had therefore never worked once since it was added, and the three-line
comparison depends on it (each teacher's cache must go in its own directory).

WHY IT SURVIVED 617 TESTS. Nothing called run(). scripts/dry_run_smoke.py -- which the
precompute sbatch runs as its node-level preflight, and which passed -- assembles the mocks
itself and calls opd_train_step directly. It exercises the PARTS and never the wiring, so a
break in run() is exactly what it cannot see. The failure surfaced on a GPU node, after the
queue wait, after the retriever and teacher handshakes, four lines from the end of setup.

WHAT THIS FILE DOES. Runs run() end to end in dry_run mode -- CPU, mocks, no vllm, no
retriever -- for the three shapes production uses: online, precompute-only, and offline
reading that cache back. Any name in run()'s call graph that is not actually bound now fails
here, in a second, instead of on a node.
"""
from __future__ import annotations

import json
import os

import pytest

import scripts.train as train

CONFIG = "configs/_smoke_hotpotqa.yaml"

# dry_run substitutes _DRY_QUESTIONS for the dataset, and the smoke config declares
# steps=2 x global_batch=256 draws over them, so epochs must be declared to match or
# assert_epochs_declared refuses -- correctly, and not what this file is testing.
STEPS = 2
EPOCHS = 256.0


@pytest.fixture(autouse=True)
def site_vars(monkeypatch):
    """The config interpolates ${SITE_VAR}s and load_config refuses when they are empty.
    dry_run never opens any of these paths; they only have to be non-empty."""
    for k, v in {"OPD_MODELS_DIR": "/nonexistent/models",
                 "OPD_TEACHER_MODEL": "/nonexistent/models/teacher",
                 "OPD_STUDENT_CKPT": "/nonexistent/models/student",
                 "OPD_SFT_DATA_DIR": "/nonexistent/data"}.items():
        monkeypatch.setenv(k, v)


def test_online_runs_end_to_end(tmp_path):
    """THE regression. Before the fix this raised NameError inside _opd_config."""
    out = tmp_path / "online"
    out.mkdir()
    train.run(CONFIG, "online", str(out), True, steps=STEPS, epochs=EPOCHS)
    diag = out / "diag_online.jsonl"
    assert diag.exists(), "the online arm wrote no diagnostics"
    rows = [json.loads(l) for l in diag.read_text(encoding="utf-8").splitlines() if l.strip()]
    # ONE ROW PER SAMPLE, not per step -- the file carries global_batch rows for each step,
    # which is the whole point of the per-sample persist rule. Assert on the STEP field.
    assert rows, "diagnostics file is empty"
    assert sorted({r["step"] for r in rows}) == list(range(STEPS))
    assert all("task" in r and "advantage" in r for r in rows), (
        "a diagnostic row must carry the per-sample task and advantage blocks a later "
        "metric would be rebuilt from")


def test_precompute_only_writes_a_manifest_where_it_was_told(tmp_path):
    """--precompute-dir is the argument the NameError was introduced to support, so it gets
    its own assertion: the cache must land in the NAMED directory, not in the config's
    default 'precomputed'. Three teacher lines write concurrently; a default shared
    directory is a write collision between arms."""
    out, cache = tmp_path / "pc", tmp_path / "precomputed_miro"
    out.mkdir()
    train.run(CONFIG, "offline", str(out), True, steps=STEPS, epochs=EPOCHS,
              precompute_only=True, precompute_dir=str(cache))
    assert (cache / "manifest.json").exists(), (
        f"nothing written to the requested --precompute-dir; contents: "
        f"{sorted(p.name for p in tmp_path.iterdir())}")
    assert not (tmp_path / "precomputed").exists(), "fell back to the config default"


def test_offline_reads_back_the_cache_that_precompute_wrote(tmp_path):
    """The two halves of the offline arm are separate processes in production. Running them
    back to back here is the only place their agreement is checked without a GPU."""
    cache = tmp_path / "cache"
    pc_out, off_out = tmp_path / "pc", tmp_path / "off"
    pc_out.mkdir()
    off_out.mkdir()
    train.run(CONFIG, "offline", str(pc_out), True, steps=STEPS, epochs=EPOCHS,
              precompute_only=True, precompute_dir=str(cache))
    train.run(CONFIG, "offline", str(off_out), True, steps=STEPS, epochs=EPOCHS,
              precompute_dir=str(cache))
    assert (off_out / "diag_offline.jsonl").exists()


def test_precompute_only_stops_before_training(tmp_path):
    """It must write the cache and RETURN -- a precompute job that also trained would burn
    its walltime and, worse, produce a checkpoint nobody asked for."""
    out, cache = tmp_path / "pc", tmp_path / "cache"
    out.mkdir()
    train.run(CONFIG, "offline", str(out), True, steps=STEPS, epochs=EPOCHS,
              precompute_only=True, precompute_dir=str(cache))
    assert not (out / "diag_offline.jsonl").exists(), "precompute-only entered the train loop"


# --------------------------------------------------------------------------- the specific bug
def test_opd_config_takes_everything_it_reads():
    """Direct and narrow: call _opd_config the way run() does. A body that reaches for a name
    the signature does not bind fails here with the same NameError, but in a second.

    Kept alongside the end-to-end tests rather than instead of them: this one names the
    function that broke, so a future reader sees WHAT to look at, while the tests above are
    what would catch the next break somewhere else in the same call graph.
    """
    import inspect

    params = inspect.signature(train._opd_config).parameters
    assert "precompute_dir" in params and "dry_run" in params, (
        "_opd_config reads these; they must be parameters, not free names")

    cfg = {"opd": {"tau": 10.0},
           "optim": {"steps": 3, "lr": 1e-6, "global_batch": 8},
           "run": {"seed": 0},
           "rollout": {"max_turns": 4, "max_len": 1024, "top_k": 3},
           "diagnostics": {"precompute_dir": "from_the_config"}}
    got = train._opd_config(cfg, "online", None, True)
    assert got.precompute_dir == "from_the_config"
    got = train._opd_config(cfg, "online", "from_the_flag", True)
    assert got.precompute_dir == "from_the_flag", "--precompute-dir must win over the config"


def test_the_node_level_preflight_does_not_cover_this():
    """A note in executable form. scripts/dry_run_smoke.py is what the precompute sbatch runs
    before spending the GPU, and it passed while run() was broken -- because it assembles the
    mocks itself and never calls run(). If that ever changes, this test should be deleted
    along with the paragraph in the module docstring that explains why these tests exist.
    """
    src = open("scripts/dry_run_smoke.py", encoding="utf-8").read()
    assert "train.run(" not in src and "from scripts.train import run" not in src, (
        "dry_run_smoke now exercises run() -- update this file's premise")
