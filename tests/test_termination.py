"""Termination-reason tagging (confound #2 hook). Structure must be recorded now
even though the diverge-vs-truncation *judgement* comes later.
"""
from __future__ import annotations

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.trajectory import TerminationReason, TokenType
from rollout.policy import MockPolicy
from rollout.rollout import generate_trajectory

CORPUS = ["Paris is the capital of France.", "Berlin is the capital of Germany."]


def _make(queries, answer, **kw):
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=queries, answer=answer)
    return generate_trajectory("q?", policy, retr, tok, top_k=1, **kw)


def test_answered():
    traj = _make(["a"], "Paris", max_turns=4, max_len=4096)
    assert traj.terminated == TerminationReason.ANSWERED
    assert traj.num_search_calls == 1


def test_truncated_maxturns():
    # Under the prefilled-<answer> forced seam the env pushes the model INTO the answer span,
    # so it can no longer keep searching. The ONLY way a forced turn fails to terminate now is
    # if the model never emits </answer>. forced_answer_closes=False simulates exactly that ->
    # genuinely non-terminating -> TRUNCATED_MAXTURNS.
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=["a", "b"], answer="Paris", forced_answer_closes=False)
    traj = generate_trajectory("q?", policy, retr, tok, top_k=1, max_turns=2, max_len=4096)
    assert traj.terminated == TerminationReason.TRUNCATED_MAXTURNS


def test_forced_prefill_answer_open_is_masked_body_is_supervised():
    # Guard 1 mask-semantics (the load-bearing token-typing rule): on the forced turn the env
    # PREFILLS the <answer> open tag -> it is OBSERVATION (masked, env-forced, NOT the model's
    # decision), while the model's continuation (answer body + </answer> + <|im_end|>) is ACTION
    # (supervised). Supervising the prefilled open tag would teach pi_ref "<answer> appears
    # automatically" instead of "WHEN to emit it".
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=["a", "b"], answer="Paris")   # forced turn closes
    traj = generate_trajectory("q?", policy, retr, tok, top_k=1, max_turns=2, max_len=4096)
    assert traj.terminated == TerminationReason.ANSWERED
    assert traj.meta["forced_answer_fired"] and traj.meta["forced_answer_answered"]

    ao = tok.encode("<answer>")[0]
    ac = tok.encode("</answer>")[0]
    ids = traj.input_ids.tolist()
    tt = traj.token_type.tolist()
    OBS, ACT = int(TokenType.OBSERVATION), int(TokenType.ACTION)
    # the prefilled <answer> open tag exists and is OBSERVATION ...
    assert any(x == ao and y == OBS for x, y in zip(ids, tt)), \
        "prefilled <answer> open tag must be tagged OBSERVATION"
    # ... and is NEVER tagged ACTION (the model must not be supervised to emit it).
    assert not any(x == ao and y == ACT for x, y in zip(ids, tt)), \
        "forced turn must NOT supervise the <answer> open tag"
    # the model-produced </answer> is ACTION (supervised continuation).
    assert any(x == ac and y == ACT for x, y in zip(ids, tt)), \
        "model-produced </answer> must be tagged ACTION"


def test_forced_final_answer_terminates_on_budget_exhaustion():
    # Guard 1: at max_turns the env injects FORCED_ANSWER_WARNING and takes ONE more
    # SUPERVISED turn. With exactly max_turns scripted queries, that forced turn emits
    # <answer> -> ANSWERED (the seam Search-R1/StepSearch/s3 get from do_search=False).
    traj = _make(["a", "b"], "Paris", max_turns=2, max_len=4096)
    assert traj.terminated == TerminationReason.ANSWERED
    assert traj.num_search_calls == 2          # two searches; the forced turn is the answer
    assert traj.meta["final_answer"] == "Paris"


def test_duplicate_query_skips_retrieval():
    # Guard 2: an EXACT-repeat query is not retrieved again (retrieval is deterministic =>
    # zero new docs). The repeat turn is recorded duplicate=True with empty docs; the first
    # occurrence retrieves normally.
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=["same", "same"], answer="Paris")
    traj = generate_trajectory("q?", policy, retr, tok, top_k=1,
                               max_turns=4, max_len=4096)
    turns = traj.meta["turns"]
    assert len(turns) == 2
    assert turns[0].get("duplicate") is not True and len(turns[0]["docs"]) > 0
    assert turns[1].get("duplicate") is True and turns[1]["docs"] == []
    assert traj.terminated == TerminationReason.ANSWERED


def test_truncated_maxlen():
    traj = _make(["a", "b", "c", "d", "e"], "Paris", max_turns=100, max_len=25)
    assert traj.terminated == TerminationReason.TRUNCATED_MAXLEN
    assert traj.length <= 25
