"""Milestone-5: the diagnostics are MEASUREMENT INSTRUMENTS -- these tests check the
instrument reads correctly (schema well-formed, conventions right, the three pinned
precision fixes actually implemented), NOT that it can see poisoning. Poisoning needs
real pi_theta drift + real retriever on HPC; on the deterministic mock nothing rots,
and a flat curve here is EXPECTED, not a bug.
"""
from __future__ import annotations

import json

import torch

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.diagnostics import (
    DiagnosticRunner, advantage_stats, answer_em_f1, boundary_bucketed_advantage,
    has_answer, kl_importance, normalize_answer, query_health, retrieval_quality,
)
from opd.step import StepOutput, opd_train_step
from opd.trajectory import TokenType, Trajectory
from rollout.policy import MockPolicy
from rollout.rollout import generate_trajectory
from teacher.scorer import MockLM, Scorer, TeacherScorer

CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
    "Berlin is the capital of Germany.",
]


def _traj(gold="Paris"):
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=["capital of France", "Eiffel Tower height"], answer="Paris")
    return generate_trajectory("What is the capital of France?", policy, retr, tok,
                               max_turns=4, max_len=4096, top_k=2, seed=0, gold_answer=gold)


# ---- convention: SQuAD/DPR normalizer shared by task-EM and hit@k ----
def test_normalize_and_has_answer():
    assert normalize_answer("The  Paris!") == "paris"
    assert has_answer("... capital is Paris, France.", "paris")
    assert not has_answer("Berlin is the capital of Germany.", "paris")


# ---- headline TASK metric: EM (strict) + F1 (lenient) on the final answer span ----
def test_answer_em_f1_headline():
    # exact (up to normalizer): EM=1, F1=1, cover_em=1; alias OR
    perfect = answer_em_f1("The Paris.", ["paris", "france"])
    assert perfect["em"] == 1.0 and perfect["f1"] == 1.0 and perfect["answered"]
    assert perfect["cover_em"] == 1.0

    # partial overlap: EM=0 but F1>0 -- the "answers drift to partially-correct" band
    partial = answer_em_f1("Paris France", ["paris"])
    assert partial["em"] == 0.0 and 0.0 < partial["f1"] < 1.0

    # verbose/padded but CORRECT: strict EM=0 and F1 crushed by precision, YET cover_em=1
    # because the gold appears as a substring. THIS is the one distinction cover_em exists for:
    # "found it but chatty" (pi_ref imitating a verbose teacher) must NOT read as a wrong answer.
    verbose = answer_em_f1("the capital city is clearly paris of course", ["paris"])
    assert verbose["em"] == 0.0 and 0.0 < verbose["f1"] < 1.0
    assert verbose["cover_em"] == 1.0

    # genuinely WRONG answer: EM=0, F1=0, AND cover_em=0 (gold absent) -- the other side of the
    # split cover_em draws. answered=True (it IS an answer, not a truncation).
    wrong = answer_em_f1("Berlin", ["paris"])
    assert wrong["em"] == 0.0 and wrong["f1"] == 0.0 and wrong["answered"]
    assert wrong["cover_em"] == 0.0

    # no answer (truncated): answered=False, all three 0
    missing = answer_em_f1(None, ["paris"])
    assert missing["answered"] is False and missing["em"] == 0.0 and missing["f1"] == 0.0
    assert missing["cover_em"] == 0.0


# ---- headline fix #1: per-hop hit rate exposes what the union smooths away ----
def test_per_hop_vs_union():
    meta = {"question": "q", "gold_answer": "paris",
            "turns": [{"query": "a", "docs": ["Paris is the capital."]},   # hop-1 hits
                      {"query": "b", "docs": ["Berlin is in Germany."]}]}   # hop-2 misses
    rq = retrieval_quality(meta)
    assert rq["per_turn_hit"] == [True, False]
    assert rq["per_turn_hit_rate"] == 0.5      # mechanism curve DROPS as later hops rot
    assert rq["union_hit"] is True             # union hides the hop-2 collapse
    assert rq["first_hit_turn"] == 0


def test_query_health_flags_degeneracy():
    meta = {"question": "capital of France", "gold_answer": "paris",
            "turns": [{"query": "capital of France"},   # verbatim repeat of question
                      {"query": ""},                     # empty
                      {"query": "Eiffel Tower height"}]}
    qh = query_health(meta)
    assert qh["n_queries"] == 3
    assert qh["degeneracy_rate"] > 0.5


# ---- pinned fix #3: bucketing is ONE-DIRECTIONAL post-boundary ----
def test_boundary_buckets_are_one_directional():
    P, A, O = int(TokenType.PROMPT), int(TokenType.ACTION), int(TokenType.OBSERVATION)
    #        0  1  2   3  4    5  6  7  8  9 10 11
    tt = [P, A, A,  O, O,   A, A, A, A, A, A, A]
    token_type = torch.tensor(tt, dtype=torch.long)
    n = token_type.shape[0]
    traj = Trajectory(torch.zeros(n, dtype=torch.long), token_type)
    score_mask = token_type == A
    adv = torch.arange(n, dtype=torch.float)
    out = StepOutput(loss=torch.tensor(0.0), advantage=adv, score_mask=score_mask,
                     student_logp=torch.zeros(n), teacher_logp=torch.zeros(n))

    buckets = boundary_bucketed_advantage(traj, out, tau=10.0, max_fine=5)
    # only the 7 action tokens AFTER the observation are bucketed; the 2 pre-obs
    # action tokens (idx 1,2) must NOT appear anywhere.
    assert set(buckets) == {"1", "2", "3", "4", "5", "6+"}
    assert sum(b["count"] for b in buckets.values()) == 7
    assert buckets["6+"]["count"] == 2
    assert buckets["1"]["count"] == 1
    # offset-1 bucket must be the FIRST post-obs token (idx 5), not a pre-obs token
    assert buckets["1"]["mean_adv"] == 5.0


# ---- pinned fix #2: KL is per-token mean at action positions; IW is finite ----
def test_kl_importance_finite_and_actionmasked():
    traj = _traj()
    student = Scorer(MockLM(8192, seed=5678))
    ref = Scorer(MockLM(8192, seed=4321))       # different model => nonzero drift
    score_mask = traj.token_type == int(TokenType.ACTION)
    ki = kl_importance(student, ref, traj, score_mask)
    assert ki["n"] == int(score_mask.sum())
    assert ki["kl_pertok_mean"] is not None and ki["kl_pertok_mean"] > 0
    assert ki["iw_mean"] > 0 and ki["iw_std"] >= 0
    # identical model => KL ~ 0, IW ~ 1 (sanity that the metric means what we claim)
    ki0 = kl_importance(student, student, traj, score_mask)
    assert abs(ki0["kl_pertok_mean"]) < 1e-5
    assert abs(ki0["iw_mean"] - 1.0) < 1e-5


def test_advantage_stats_schema():
    traj = _traj()
    teacher = TeacherScorer(MockLM(8192, seed=1234))
    student = Scorer(MockLM(8192, seed=5678))
    out = opd_train_step(traj, teacher.score_online(traj.input_ids), student)
    a = advantage_stats(out, tau=10.0)
    assert a["n_scored"] > 0
    assert 0.0 <= a["clip_rate"] <= 1.0
    for k in ("adv_mean", "adv_std", "adv_absmax"):
        assert a[k] is not None


# ---- end-to-end: one full record is well-formed and JSON-serializable ----
def test_runner_record_is_serializable():
    traj = _traj()
    teacher = TeacherScorer(MockLM(8192, seed=1234))
    student = Scorer(MockLM(8192, seed=5678))
    ref = Scorer(MockLM(8192, seed=5678))       # snapshot-equivalent
    out = opd_train_step(traj, teacher.score_online(traj.input_ids), student)

    runner = DiagnosticRunner(ref_scorer=ref, tau=10.0, embedder=None)
    rec = runner.run(0, "online", traj, out, student).to_dict()

    for key in ("step", "mode", "task", "retrieval", "query", "termination",
                "kl_importance", "advantage", "boundary", "masking", "policy", "embedding"):
        assert key in rec
    assert rec["embedding"] is None            # no encoder in dry-run
    assert rec["task"] == {"answered": True, "em": 1.0, "f1": 1.0, "cover_em": 1.0, "pred": "paris"}
    assert rec["masking"]["scored_equals_action"] is True
    assert rec["policy"] == {"staleness": 0, "resync_every": 1, "on_policy": True}
    json.dumps(rec)                            # must round-trip through jsonl


# ---- offline context travels frozen with the data (constant reference line) ----
def test_offline_meta_survives_disk_and_is_deterministic():
    import os
    import tempfile
    traj = _traj()
    teacher = TeacherScorer(MockLM(8192, seed=1234))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.pt")
        teacher.precompute_and_save(traj, path)
        rec = TeacherScorer.load(path)
        assert "meta" in rec and rec["meta"]["gold_answer"] == "Paris"
        rq1 = retrieval_quality(rec["meta"])
        rq2 = retrieval_quality(TeacherScorer.load(path)["meta"])
        assert rq1 == rq2                       # frozen => identical every read
