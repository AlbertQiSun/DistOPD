"""M-SFT Stage 1 unit tests. The design invariant, sharpened this milestone:

    SFT takes MiroThinker's teacher distribution AS-IS (zero quality filtering).
    The drop predicate is a function of the trajectory's STRUCTURE / COMPLETENESS,
    NEVER its QUALITY.

So classify_trajectory drops ONLY structurally-broken / structurally-incomplete trajectories
(no_action / empty_query / empty_answer / not_terminated). Everything that is a well-formed,
TERMINATED trajectory is KEPT, INCLUDING low-quality teacher behavior: wrong answers, single
no-search, empty retrieval, repeated queries, high-hop spinning, and low-EM best-effort forced
answers. These tests lock exactly that -- a trajectory that would previously have been
"quality-filtered" must now be KEPT, because any such filter is sigma_Delta.

TERMINATION IS STRUCTURAL, NOT QUALITY. The rollout now has a forced-final-answer seam
(Guard 1): at budget exhaustion the environment injects a warning and takes ONE supervised
answer turn, so essentially every trajectory CAN terminate ANSWERED. A trajectory that still
has no <answer> is therefore structurally INCOMPLETE (no supervised terminal answer span) ->
dropped not_terminated. This is orthogonal to quality: a low-EM best-effort forced answer
terminates ANSWERED and is KEPT.

The action-only mask must still equal build_score_mask, and the teacher-consistency gate
must still fire on a mismatched generator (those are structural, not quality).
"""
from __future__ import annotations

import torch

from opd.masking import build_score_mask
from opd.trajectory import TerminationReason, TokenType, Trajectory
from sft.datagen import build_sft_record, classify_trajectory, productivity_summary
from teacher.backend import BackendConfig, assert_same_teacher_model
from teacher.scorer import MOCK_CONFIG


def _traj(types, *, num_search, terminated, turns, final_answer=None, question="q", gold=None):
    n = len(types)
    ids = torch.arange(1, n + 1, dtype=torch.long)
    tt = torch.tensor([int(t) for t in types], dtype=torch.long)
    return Trajectory(ids, tt, terminated=terminated, num_search_calls=num_search,
                      meta={"question": question, "turns": turns,
                            "final_answer": final_answer, "gold_answer": gold})


P, A, O = TokenType.PROMPT, TokenType.ACTION, TokenType.OBSERVATION


def _coherent(final_answer="Paris"):
    """prompt, action(search), observation, action(answer): a normal 1-hop search."""
    return _traj(
        [P, P, A, A, O, O, A, A],
        num_search=1, terminated=TerminationReason.ANSWERED,
        turns=[{"turn": 0, "query": "capital of France", "docs": ["Paris is ..."]}],
        final_answer=final_answer,
    )


# ---------------------------------------------------------------------------
# structural drops (the ONLY drops): no_action / empty_query / empty_answer
# ---------------------------------------------------------------------------
def test_coherent_trajectory_is_kept():
    assert classify_trajectory(_coherent()).keep


def test_wrong_answer_but_coherent_search_is_kept():
    # coherence != correctness: coherent search + WRONG final answer -> KEPT (no EM gate).
    v = classify_trajectory(_coherent(final_answer="Berlin"))  # wrong, but coherent
    assert v.keep, "must NOT drop a coherent-but-incorrect trajectory"


def test_empty_query_is_dropped():
    # empty_query = teacher emitted an empty <search> (generation-layer STRUCTURAL defect).
    t = _traj([P, P, A, A, O, O], num_search=1, terminated=TerminationReason.TRUNCATED_MAXTURNS,
              turns=[{"turn": 0, "query": "   ", "docs": ["d"]}])
    v = classify_trajectory(t)
    assert not v.keep and v.reason == "empty_query"


def test_empty_answer_is_dropped():
    # terminated ANSWERED but empty <answer> span = STRUCTURAL defect.
    t = _coherent(final_answer="")
    v = classify_trajectory(t)
    assert not v.keep and v.reason == "empty_answer"


def test_no_action_is_dropped():
    # no student ACTION span -> literally nothing to supervise. STRUCTURAL.
    t = _traj([P, P, O], num_search=1, terminated=TerminationReason.TRUNCATED_MAXLEN,
              turns=[{"turn": 0, "query": "q", "docs": ["d"]}])
    v = classify_trajectory(t)
    assert not v.keep and v.reason == "no_action"


# ---------------------------------------------------------------------------
# ZERO QUALITY FILTERING: trajectories that a quality filter WOULD have dropped
# must now be KEPT (dropping them is sigma_Delta). These lock the reversal.
# ---------------------------------------------------------------------------
def test_single_no_search_is_kept():
    # a single blurted (zero-search) answer is REAL teacher behavior -> KEPT. A high BATCH
    # rate is a device red flag (check_sft_stats.py), but one trajectory is not dropped.
    t = _traj([P, P, A, A], num_search=0, terminated=TerminationReason.ANSWERED,
              turns=[], final_answer="Paris")
    assert classify_trajectory(t).keep


def test_retrieval_empty_is_kept():
    # query well-formed but retriever returned no docs = ENVIRONMENT output, not a broken
    # trajectory; the teacher answering under empty retrieval is behavior pi_ref should see.
    # (Terminated ANSWERED so the ONLY property under test is "empty retrieval isn't a drop".)
    t = _traj([P, P, A, A, O, A, A], num_search=1, terminated=TerminationReason.ANSWERED,
              turns=[{"turn": 0, "query": "q1", "docs": []}], final_answer="Paris")
    assert classify_trajectory(t).keep


def test_repeated_query_loop_is_kept():
    # repeating a query is "dumb", not "broken" (a well-formed trajectory) -> KEPT.
    # (Terminated ANSWERED so the ONLY property under test is "a repeated query isn't a drop".)
    t = _traj([P, P, A, A, O, A, A, O, A, A], num_search=2,
              terminated=TerminationReason.ANSWERED,
              turns=[{"turn": 0, "query": "same", "docs": ["d"]},
                     {"turn": 1, "query": "same", "docs": ["d"]}], final_answer="Paris")
    assert classify_trajectory(t).keep


def test_best_effort_forced_answer_is_kept():
    # Guard 1 forced an answer at budget exhaustion; it may be a low-EM guess. That is COHERENT
    # terminal behavior (terminated ANSWERED), NOT quality damage -> KEPT (there is no EM gate).
    t = _traj([P, P, A, A, O, A, A, O, A, A], num_search=2,
              terminated=TerminationReason.ANSWERED,
              turns=[{"turn": 0, "query": "hard q part 1", "docs": ["irrelevant"]},
                     {"turn": 1, "query": "hard q part 2", "docs": ["irrelevant"]}],
              final_answer="MyBestGuess")
    assert classify_trajectory(t).keep


def test_not_terminated_is_dropped():
    # After the forced-final-answer seam, a trajectory that STILL has no <answer> is
    # structurally INCOMPLETE (no supervised terminal answer span) -> dropped not_terminated.
    # This is structural, not quality: the model failed to answer even when the env forced it.
    t = _traj([P, P, A, A, O, O], num_search=1, terminated=TerminationReason.TRUNCATED_MAXTURNS,
              turns=[{"turn": 0, "query": "capital of France", "docs": ["Paris is ..."]}],
              final_answer=None)
    v = classify_trajectory(t)
    assert not v.keep and v.reason == "not_terminated"


def test_truncated_maxlen_without_answer_is_dropped():
    # ran out of token budget before emitting <answer> -> also no terminal answer span ->
    # dropped not_terminated (should be rare; a high count is a monitoring signal).
    t = _traj([P, P, A, A, O, O], num_search=1, terminated=TerminationReason.TRUNCATED_MAXLEN,
              turns=[{"turn": 0, "query": "capital of France", "docs": ["Paris is ..."]}],
              final_answer=None)
    v = classify_trajectory(t)
    assert not v.keep and v.reason == "not_terminated"


# ---------------------------------------------------------------------------
# productivity is a DIAGNOSTIC that is still COMPUTED but NEVER gates.
# ---------------------------------------------------------------------------
def _multihop(*, hops, hit_turns, final_answer="x", gold="Paris"):
    """A `hops`-search trajectory; `hit_turns` = set of turn indices whose docs contain gold."""
    turns = [{"turn": k, "query": f"q{k}",
              "docs": ([f"{gold} appears here"] if k in hit_turns else ["irrelevant text"])}
             for k in range(hops)]
    types = [P, P] + sum(([A, A, O] for _ in range(hops)), []) + [A, A]
    return _traj(types, num_search=hops, terminated=TerminationReason.ANSWERED,
                 turns=turns, final_answer=final_answer, gold=gold)


def test_productive_multihop_is_kept_and_reported():
    # 2 hops, evidence surfaced at hop 1 -> productive multi-hop. KEPT, and the diagnostic
    # reports productive/productive_multihop True (computed, but did not gate the keep).
    t = _multihop(hops=2, hit_turns={1})
    assert classify_trajectory(t).keep
    prod = productivity_summary(t)
    assert prod["productive"] is True and prod["productive_multihop"] is True


def test_productive_but_wrong_answer_is_kept():
    # evidence surfaced (productive) but final answer WRONG -> KEPT (productive != correct).
    t = _multihop(hops=2, hit_turns={0, 1}, final_answer="TotallyWrong", gold="Paris")
    assert classify_trajectory(t).keep
    assert productivity_summary(t)["productive"] is True


def test_high_hop_spinning_is_kept_but_flagged_unproductive():
    # 3 hops, gold evidence NEVER surfaced = spinning. Under zero filtering this is KEPT
    # (real teacher behavior on a hard question); the diagnostic flags productive=False so
    # the check script can REPORT the spinning ratio -- but it does not drop the trajectory.
    t = _multihop(hops=3, hit_turns=set(), gold="Paris")
    assert classify_trajectory(t).keep
    assert productivity_summary(t)["productive"] is False


def test_no_gold_leaves_productivity_unknown():
    # without gold, union_hit is None -> productivity unknown (never forced), still KEPT.
    t = _multihop(hops=4, hit_turns=set(), gold=None)
    assert classify_trajectory(t).keep
    assert productivity_summary(t)["productive"] is None


# ---------------------------------------------------------------------------
# structural invariants: action-only mask and teacher-consistency gate
# ---------------------------------------------------------------------------
def test_action_only_labels_match_score_mask():
    t = _coherent()
    rec = build_sft_record(t)
    mask = build_score_mask(t.token_type)
    # labels are supervised (== ids) exactly on ACTION positions, -100 elsewhere.
    supervised = rec["labels"] != -100
    assert torch.equal(supervised, mask), "supervised set must equal build_score_mask (ACTION)"
    assert torch.equal(rec["labels"][mask], t.input_ids[mask])
    assert rec["meta"]["n_action_tokens"] == int(mask.sum())


def test_record_meta_carries_productivity_as_diagnostic_only():
    # productivity is embedded in meta for later analysis, but it is a pure diagnostic --
    # its presence must not imply any gating (the record exists <=> classify kept it).
    t = _coherent()
    rec = build_sft_record(t)
    assert "productivity" in rec["meta"]
    assert rec["meta"]["productivity"]["hops"] == 1


def test_teacher_consistency_gate_ignores_engine_but_catches_model():
    # same MiroThinker model, different engine (vllm gen vs hf teacher) -> OK.
    teacher = BackendConfig(engine="hf", dtype="bfloat16", tokenizer_name="MiroThinker")
    gen_ok = BackendConfig(engine="vllm", dtype="bfloat16", tokenizer_name="MiroThinker")
    assert_same_teacher_model(teacher, gen_ok)  # must NOT raise

    # different model -> must raise (re-introduces sigma_Delta).
    gen_bad = BackendConfig(engine="vllm", dtype="bfloat16", tokenizer_name="SomeOtherModel")
    try:
        assert_same_teacher_model(teacher, gen_bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected teacher-consistency gate to fire on a different model")

    # mock dry-run path is self-consistent.
    assert_same_teacher_model(MOCK_CONFIG, MOCK_CONFIG)
