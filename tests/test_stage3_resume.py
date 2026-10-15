"""Nine hours of finished rollouts must never again evaporate at a walltime.

THE FAILURE THIS ENCODES. Job 26845863 (dense offline product eval, 2026-08-27) hit its
9 h TimeLimit around question ~900 of 1000. stage3_eval wrote its report only at the END,
so every completed question was lost -- the same under-persist shape as every the project conventions
incident, in time instead of fields. The writer now appends each finished question to
<report>.partial.jsonl immediately, and a rerun skips what is already there.

THE RESUME MUST BE SAMPLE-IDENTICAL, NOT MERELY CONVENIENT. base_seed is derived from the
question's INDEX in the deterministic decontaminated prefix; skipping by question key
preserves every other question's index, so a resumed run and an uninterrupted run draw the
same rollouts. And a partial written under a DIFFERENT protocol (other K, other max_turns,
other model) must be refused, not merged -- mixing them puts two eval configurations into
one report with nothing visible at the seam.
"""
from __future__ import annotations

import json
import sys

import pytest

import scripts.stage3_eval as E


@pytest.fixture(autouse=True)
def site_vars(monkeypatch):
    """The config interpolates ${SITE_VAR}s and load_config refuses when they are empty.
    dry_run never opens any of these paths; they only have to be non-empty."""
    for k, v in {"OPD_MODELS_DIR": "/nonexistent/models",
                 "OPD_TEACHER_MODEL": "/nonexistent/models/teacher",
                 "OPD_STUDENT_CKPT": "/nonexistent/models/student",
                 "OPD_SFT_DATA_DIR": "/nonexistent/data"}.items():
        monkeypatch.setenv(k, v)


def _run(monkeypatch, out_dir, label="smoke"):
    monkeypatch.setattr(sys, "argv",
                        ["stage3_eval.py", "--dry_run", "--label", label,
                         "--out-dir", str(out_dir)])
    E.main()
    return out_dir / f"stage3_{label}.json"


def test_each_question_lands_in_the_partial_before_the_report_exists(tmp_path, monkeypatch):
    """The partial is the crash artifact; the report is the success artifact."""
    report = _run(monkeypatch, tmp_path)
    assert report.exists()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert len(data["per_question"]) == len(E._DRY_QUESTIONS)


def test_the_partial_is_removed_once_the_report_is_written(tmp_path, monkeypatch):
    """A leftover partial would let a later run 'resume' on top of a FINISHED eval and
    silently skip all of its work."""
    report = _run(monkeypatch, tmp_path)
    assert not (tmp_path / (report.name + ".partial.jsonl")).exists()


def test_a_resumed_run_reuses_the_partial_rows_verbatim(tmp_path, monkeypatch):
    """Plant a partial whose first question carries a marker value the dry-run policy could
    never produce; the final report must contain the marker row (skipped, not re-rolled)
    and a freshly evaluated second question."""
    label = "resume"
    out = tmp_path / f"stage3_{label}.json"
    partial = tmp_path / (out.name + ".partial.jsonl")
    protocol = {"label": label, "model": "MOCK", "num_samples": 4,
                "max_turns": None, "temperature": None, "max_len": None}
    # Protocol values must match what the dry run will compute -- read them by running once.
    _run(monkeypatch, tmp_path, label="probe")
    probe = json.loads((tmp_path / "stage3_probe.json").read_text(encoding="utf-8"))
    protocol = {"label": label, "model": probe["model"],
                "num_samples": probe["num_samples"],
                "max_turns": probe["rollout"]["max_turns"],
                "temperature": probe["temperature"],
                "max_len": probe["rollout"]["max_len"]}
    marker = dict(probe["per_question"][0])
    marker["em"] = 0.123456          # no K=4 mean can be this
    with open(partial, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__protocol__": protocol}) + "\n")
        f.write(json.dumps(marker) + "\n")
    report = _run(monkeypatch, tmp_path, label=label)
    data = json.loads(report.read_text(encoding="utf-8"))
    ems = [r["em"] for r in data["per_question"]]
    assert 0.123456 in ems, "the partial row was re-evaluated instead of reused"
    assert len(data["per_question"]) == len(E._DRY_QUESTIONS)


def test_a_partial_from_a_different_protocol_is_refused_not_merged(tmp_path, monkeypatch):
    label = "clash"
    partial = tmp_path / f"stage3_{label}.json.partial.jsonl"
    with open(partial, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__protocol__": {"label": label, "model": "OTHER",
                                             "num_samples": 8, "max_turns": 10,
                                             "temperature": 0.1, "max_len": 99}}) + "\n")
    monkeypatch.setattr(sys, "argv",
                        ["stage3_eval.py", "--dry_run", "--label", label,
                         "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as e:
        E.main()
    assert "different protocol" in str(e.value)


def test_a_partial_with_a_torn_final_line_resumes_on_what_is_whole(tmp_path, monkeypatch):
    """A walltime kill ends mid-write; the torn line is discarded, the rest is kept."""
    label = "torn"
    _run(monkeypatch, tmp_path, label="probe2")
    probe = json.loads((tmp_path / "stage3_probe2.json").read_text(encoding="utf-8"))
    protocol = {"label": label, "model": probe["model"],
                "num_samples": probe["num_samples"],
                "max_turns": probe["rollout"]["max_turns"],
                "temperature": probe["temperature"],
                "max_len": probe["rollout"]["max_len"]}
    partial = tmp_path / f"stage3_{label}.json.partial.jsonl"
    good = dict(probe["per_question"][0])
    good["em"] = 0.654321
    with open(partial, "w", encoding="utf-8") as f:
        f.write(json.dumps({"__protocol__": protocol}) + "\n")
        f.write(json.dumps(good) + "\n")
        f.write('{"question": "torn mid-wr')
    report = _run(monkeypatch, tmp_path, label=label)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert 0.654321 in [r["em"] for r in data["per_question"]]
    assert len(data["per_question"]) == len(E._DRY_QUESTIONS)
