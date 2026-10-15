"""The OPD training slice must not contain questions pi_ref already memorised in SFT.

WHAT IS BEING PROTECTED. pi_ref is trained on the first N valid rows of
hotpotqa_with_gold.jsonl (sft.data.sources limit 2000), and OPD rolls out on the same file
past data.offset. If the two slices overlap, pi_ref answers those questions from memory and
the rollout states are not genuine search -- which is the ONE thing the whole Row 10
comparison measures. Nothing crashes; the numbers just quietly mean something else.

WHY THIS FILE EXISTS AT ALL. scripts/train.py::_assert_disjoint_from_sft was already written
and already correct, and NOTHING in tests/ mentioned it. Deleting the single line in
make_dataset that calls it left all 575 tests green. That is the the project conventions producer rule one
level up: the check existed, but nothing verified the check RUNS. test_make_dataset_actually_
runs_the_check is the line that closes it.

THE SUBTLE PROPERTY, and the reason a lazier test would be worthless: the check derives how
many rows SFT consumed from the SFT CONFIG (sources[].limit), not from data.offset. Deriving
it from offset would be self-referential -- any contiguous slice starting at `offset` is
trivially disjoint from the first `offset` rows, so a too-SMALL offset would sail through a
check that looks rigorous. test_a_too_small_offset_is_caught is that property, and it is the
one an offset-derived implementation fails.
"""
from __future__ import annotations

import json

import pytest

import scripts.train as train

QUESTIONS = [f"who wrote book number {i}" for i in range(20)]


@pytest.fixture
def pool(tmp_path):
    """A 20-question file. SFT takes the first 8; OPD is configured with an offset."""
    p = tmp_path / "pool_with_gold.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i, q in enumerate(QUESTIONS):
            f.write(json.dumps({"question": q, "golden_answers": [f"author {i}"]}) + "\n")
    return str(p)


def _cfg(path, offset, limit=None, sft_limit=8, sft_path=None):
    return {
        "data": {"path": path, "name": None, "offset": offset, "limit": limit},
        "sft": {"data": {"sources": [{"path": sft_path or path, "limit": sft_limit}]}},
    }


# --------------------------------------------------------------------------- the property
def test_a_correct_offset_passes_and_says_what_it_compared(pool, capsys):
    qs, golds = train.make_dataset(_cfg(pool, offset=8), dry_run=False)
    assert qs == QUESTIONS[8:]
    assert golds[0] == ["author 8"]
    out = capsys.readouterr().out
    assert "disjointness OK" in out
    assert "12 OPD questions" in out and "8" in out


def test_a_too_small_offset_is_caught(pool):
    """THE test. offset=5 with SFT having consumed 8 leaves rows 5,6,7 in both slices. An
    implementation that derived the SFT count from `offset` itself would find no overlap
    here and pass -- which is why the count comes from the SFT config."""
    with pytest.raises(ValueError, match="OVERLAPS"):
        train.make_dataset(_cfg(pool, offset=5), dry_run=False)


def test_zero_offset_against_an_sft_source_is_caught(pool):
    """The default. Forgetting data.offset entirely is the most likely way to get here."""
    with pytest.raises(ValueError, match="OVERLAPS"):
        train.make_dataset(_cfg(pool, offset=0), dry_run=False)


def test_the_error_says_how_to_fix_it(pool):
    with pytest.raises(ValueError) as e:
        train.make_dataset(_cfg(pool, offset=5), dry_run=False)
    msg = str(e.value)
    assert "3 shared question" in msg, "the operator needs the SIZE of the overlap"
    assert "Raise data.offset to at least 8" in msg, "and the number to raise it to"
    assert "book number 5" in msg or "book number 6" in msg or "book number 7" in msg


def test_overlap_is_by_CONTENT_not_by_row_index(tmp_path):
    """A contiguous offset is disjoint by ROW. If the same question text appears twice in
    the file -- once inside the SFT prefix and once after it -- the slices are disjoint by
    index and NOT disjoint in effect. pi_ref memorised the string, not the line number."""
    p = tmp_path / "dup_with_gold.jsonl"
    rows = [{"question": q, "golden_answers": ["x"]} for q in QUESTIONS[:8]]
    rows.append({"question": QUESTIONS[0], "golden_answers": ["x"]})   # a repeat, at row 8
    rows += [{"question": q, "golden_answers": ["x"]} for q in QUESTIONS[8:]]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="OVERLAPS"):
        train.make_dataset(_cfg(str(p), offset=8), dry_run=False)


def test_normalization_catches_a_punctuation_variant(tmp_path):
    """The keys go through the shared SQuAD/DPR normalizer, so 'Who wrote Book Number 1?'
    and 'who wrote book number 1' are the SAME memorised question."""
    p = tmp_path / "var_with_gold.jsonl"
    rows = [{"question": q, "golden_answers": ["x"]} for q in QUESTIONS[:8]]
    rows.append({"question": "Who wrote Book Number 1?", "golden_answers": ["x"]})
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="OVERLAPS"):
        train.make_dataset(_cfg(str(p), offset=8), dry_run=False)


# --------------------------------------------------------------------------- escape hatches
def test_a_file_sft_never_touched_is_declared_not_silently_skipped(pool, tmp_path, capsys):
    """Pointing OPD at dev, or at a dataset SFT never used, is legitimate. It must SAY so:
    a check that can quietly no-op is indistinguishable from a check that passed."""
    other = tmp_path / "other_with_gold.jsonl"
    other.write_text(json.dumps({"question": "unrelated q", "golden_answers": ["y"]}) + "\n",
                     encoding="utf-8")
    train.make_dataset(_cfg(str(other), offset=0, sft_path=pool), dry_run=False)
    out = capsys.readouterr().out
    assert "not an SFT source" in out
    assert "no SFT overlap possible" in out


def test_an_sft_source_with_no_limit_refuses(pool):
    """limit=None means SFT consumed the WHOLE file, so no offset can carve an unused
    slice. Continuing would guarantee overlap."""
    with pytest.raises(ValueError, match="limit=None"):
        train.make_dataset(_cfg(pool, offset=8, sft_limit=None), dry_run=False)


def test_the_largest_limit_wins_when_a_file_is_listed_twice(pool):
    """Two sources naming the same file: the consumed prefix is the LARGER of the two, and
    taking the first or the smaller one would under-report what pi_ref saw."""
    cfg = _cfg(pool, offset=8)
    cfg["sft"]["data"]["sources"] = [{"path": pool, "limit": 4}, {"path": pool, "limit": 12}]
    with pytest.raises(ValueError, match="OVERLAPS"):
        train.make_dataset(cfg, dry_run=False)


# --------------------------------------------------------------------------- the producer
def test_make_dataset_actually_runs_the_check(pool, monkeypatch):
    """THE LINE THIS FILE EXISTS FOR. Delete `_assert_disjoint_from_sft(cfg, questions)`
    from make_dataset and this goes red; without it every test above passes while the
    production path never calls the function at all."""
    seen = {}
    monkeypatch.setattr(train, "_assert_disjoint_from_sft",
                        lambda cfg, questions: seen.update(n=len(questions)))
    train.make_dataset(_cfg(pool, offset=8), dry_run=False)
    assert seen == {"n": 12}, "make_dataset did not call the disjointness check"


def test_the_shipped_config_offset_matches_what_sft_consumed():
    """The live numbers, read from the file rather than remembered: data.offset must be at
    least the hotpotqa limit in sft.data.sources, or the real run trips the check."""
    import yaml
    with open("configs/opd_qwen3.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    path = cfg["data"]["path"]
    consumed = [s["limit"] for s in cfg["sft"]["data"]["sources"] if s["path"] == path]
    assert consumed, f"data.path {path} is no longer an SFT source -- re-read this test"
    assert cfg["data"]["offset"] >= max(consumed), (
        f"data.offset={cfg['data']['offset']} but SFT consumed {max(consumed)} rows of {path}")
