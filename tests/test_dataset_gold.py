"""Seam 3 (local, MUST pass): dataset gold parsing + answer-recall correctness.

The per-hop hit headline is only as correct as this parse. If the gold field is
read wrong, the mechanism headline is silently wrong -- so gold extraction across
the real dataset shapes (Search-R1 unified `golden_answers`, TriviaQA alias dict,
HotpotQA single-string answer) is verified here on committed real-schema fixtures.

Also pins the two answer-recall semantics M6 nailed down:
  * alias hit is OR (any normalized alias in the passage), not AND;
  * alias and passage share the SAME normalize_answer as task EM;
  * per-hop hit is per-hop ANSWER-string recall, never hop-to-gold-doc alignment
    (HotpotQA supporting_facts cannot support that, so we never attempt it).
"""
from __future__ import annotations

import os

from dataio import dataset as D
from opd import diagnostics as DIAG
from opd.diagnostics import has_answer, retrieval_quality

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def test_shared_normalizer_identity():
    # the loader and the hit metric must use the very SAME normalizer object
    assert D.normalize_answer is DIAG.normalize_answer


def test_searchr1_unified_golden_answers():
    rows = D.load(os.path.join(FIX, "searchr1_unified.json"))
    assert len(rows) == 3
    r = {row["question"]: row["gold_answers"] for row in rows}
    assert r["what is the capital of france"] == ["Paris"]
    # multi-alias golden_answers preserved as a list
    assert set(r["who sang who wants to be a millionaire in high society"]) == {
        "Frank Sinatra", "Celeste Holm"}


def test_triviaqa_aliases_parsed_to_list():
    rows = D.load(os.path.join(FIX, "triviaqa_raw.json"), name="triviaqa")
    florida = next(r for r in rows if "Sunshine" in r["question"])["gold_answers"]
    # aliases + value flattened & de-duplicated (normalized) into one alias list
    assert "Florida" in florida
    assert any(a.lower().startswith("state of florida") for a in florida)


def test_hotpotqa_single_answer_string():
    rows = D.load(os.path.join(FIX, "hotpotqa_raw.json"), name="hotpotqa")
    ans = {r["question"]: r["gold_answers"] for r in rows}
    assert ans["What is the height of the tower located in the capital of France?"] == ["330 metres"]


def test_alias_hit_is_or_not_and():
    aliases = ["Florida", "State of Florida", "Fla."]
    # passage contains ONLY one alias -> still a hit (OR, not all-must-appear)
    assert has_answer("I was born in the state of Florida last year.", aliases)
    assert not has_answer("I was born in Georgia last year.", aliases)


def test_per_hop_recall_is_answer_string_based():
    # gold is an alias LIST; per-hop hit computed by answer-string OR per hop,
    # with NO attempt to align hop t to a specific gold document.
    meta = {
        "question": "How tall is the Eiffel Tower?",
        "gold_answer": ["330 metres", "330 m"],
        "turns": [
            {"turn": 0, "query": "eiffel tower height",
             "docs": ["The Eiffel Tower is 330 m tall."]},          # hit via alias "330 m"
            {"turn": 1, "query": "eiffel tower location",
             "docs": ["The tower stands in Paris, France."]},        # no gold string
        ],
    }
    rq = retrieval_quality(meta)
    assert rq["per_turn_hit"] == [True, False]      # per-hop, not smoothed
    assert rq["per_turn_hit_rate"] == 0.5
    assert rq["union_hit"] is True                  # task-success proxy only
    assert rq["first_hit_turn"] == 0
