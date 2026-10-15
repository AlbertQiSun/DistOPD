"""Batching must not change a single trajectory (mock layer, zero-GPU).

global_batch=256 forces rollout to be issued in gathered rounds. The design keeps ONE
copy of the per-trajectory control flow (a coroutine that yields ActRequest /
RetrieveRequest) and lets a driver decide when to serve those asks, so batching cannot
reach the control flow. THIS FILE IS THE PROOF THAT THE REFACTOR ITSELF IS FAITHFUL.

Scope, stated precisely: MockPolicy and MockRetriever are DETERMINISTIC, so every
difference observable here is attributable to the scheduler and to nothing else -- there
is no sampling noise and no vLLM batch-composition effect to hide behind. A failure here
means the refactor is wrong; it admits no other explanation. Whether the REAL vLLM policy
is batch-invariant is a separate, GPU-only question (batch composition shifts as
trajectories finish at different turns), settled by its own measurement.

The trajectories are chosen to desynchronise: they answer after different numbers of
turns, and one is driven onto the forced-answer path (Guard 1). If the driver ever
coupled trajectories -- served a reply to the wrong coroutine, ran them in lockstep,
or let a finished one keep contributing -- these are the cases that expose it.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.retriever import MockRetriever  # noqa: E402
from envs.tokenizer import MockTokenizer  # noqa: E402
from rollout.policy import MockPolicy  # noqa: E402
from rollout.rollout import generate_trajectories, generate_trajectory  # noqa: E402

CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
    "Lyon is a city in France known for its cuisine.",
    "The Seine is a river that flows through Paris.",
]

# Different query counts => different turn counts => the batch desynchronises.
# 3 queries with max_turns=3 exhausts the budget and takes the FORCED-answer path.
SCRIPTS = [
    (["capital of France"], "Paris"),
    (["Eiffel Tower height", "Seine river"], "330 metres"),
    (["capital of France", "Lyon cuisine", "Seine river"], "forced"),
    ([], "immediate"),
]
QUESTIONS = [f"question {i}?" for i in range(len(SCRIPTS))]
GOLDS = [["Paris"], ["330"], ["x"], ["y"]]

KW = dict(max_turns=3, max_len=4096, top_k=2, seed=0)


def _contains(haystack: tuple, needle: tuple) -> bool:
    n = len(needle)
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


class _MultiPolicy:
    """One policy object serving all trajectories, dispatching on the prefix it is given.

    The rollout hands the policy only (prefix_ids, turn, forced) -- it does NOT say which
    trajectory is asking, which is correct: the policy is stateless per call. To keep a
    per-trajectory script anyway we identify the caller by the question's TOKEN-ID
    signature inside the prefix (MockTokenizer has no decode, and matching on ids is the
    stricter check anyway). The consequence that matters: a driver which handed a reply to
    the WRONG coroutine would be serving it another trajectory's script, and the resulting
    trajectory would differ -- so mis-pairing surfaces as a failure instead of as
    plausible-looking output.
    """

    def __init__(self, tokenizer, scripts: dict) -> None:
        self.tok = tokenizer
        self.by_q = {q: MockPolicy(tokenizer, queries=qs, answer=a)
                     for q, (qs, a) in scripts.items()}
        self._sig = {q: tuple(tokenizer.encode(q)) for q in scripts}
        self.calls = 0

    def _which(self, prefix_ids: torch.Tensor):
        ids = tuple(prefix_ids.tolist())
        for q, sig in self._sig.items():
            if _contains(ids, sig):
                return self.by_q[q]
        raise AssertionError("prefix does not identify a known question")

    def act(self, prefix_ids, turn, forced=False):
        self.calls += 1
        # the turn index comes from the coroutine's own state, so a per-trajectory script
        # stays correct no matter how the rounds are gathered.
        return self._which(prefix_ids).act(prefix_ids, turn, forced=forced)

    def act_batch(self, requests):
        from rollout.policy import act_batch_via_act
        return act_batch_via_act(self, requests)


def _fresh():
    tok = MockTokenizer(vocab_size=8192)
    scripts = {q: s for q, s in zip(QUESTIONS, SCRIPTS)}
    return tok, _MultiPolicy(tok, scripts), MockRetriever(CORPUS, seed=0)


def _fingerprint(traj):
    """Everything downstream consumes: the frozen ids, the span tagging that drives
    masking, the termination bookkeeping, and the rollout context the diagnostics read."""
    return {
        "input_ids": traj.input_ids.tolist(),
        "token_type": traj.token_type.tolist(),
        "terminated": None if traj.terminated is None else traj.terminated.value,
        "num_search_calls": traj.num_search_calls,
        "meta": traj.meta,
    }


def test_batched_rollout_is_bit_identical_to_one_at_a_time():
    tok, policy, retr = _fresh()
    singles = [generate_trajectory(q, policy, retr, tok, gold_answer=g, **KW)
               for q, g in zip(QUESTIONS, GOLDS)]

    tok, policy, retr = _fresh()
    batched = generate_trajectories(QUESTIONS, policy, retr, tok, golds=GOLDS, **KW)

    assert len(batched) == len(singles)
    for i, (a, b) in enumerate(zip(singles, batched)):
        fa, fb = _fingerprint(a), _fingerprint(b)
        assert fa["input_ids"] == fb["input_ids"], f"traj {i}: token ids differ under batching"
        assert fa["token_type"] == fb["token_type"], f"traj {i}: span tagging differs"
        assert fa["terminated"] == fb["terminated"], f"traj {i}: termination differs"
        assert fa["num_search_calls"] == fb["num_search_calls"], f"traj {i}: search count differs"
        assert fa["meta"] == fb["meta"], f"traj {i}: rollout meta differs"


def test_the_batch_actually_desynchronises():
    """Guard against a vacuous pass: if every trajectory happened to take the same number
    of turns, the test above would hold even for a driver that ran them in lockstep and
    could not survive a real batch. Assert the fixtures exercise the ragged case, including
    a forced-answer trajectory and a zero-search one."""
    tok, policy, retr = _fresh()
    trajs = generate_trajectories(QUESTIONS, policy, retr, tok, golds=GOLDS, **KW)
    searches = [t.num_search_calls for t in trajs]
    assert len(set(searches)) > 1, f"all trajectories took the same turn count: {searches}"
    assert min(searches) == 0, "no zero-search trajectory in the batch"
    assert any(t.meta.get("forced_answer_fired") for t in trajs), "no forced-answer path exercised"


def test_single_path_still_drives_the_same_coroutine():
    """generate_trajectory must be the batched driver at size one, not a second
    implementation -- otherwise the equivalence above compares two code paths that could
    drift apart later."""
    tok, policy, retr = _fresh()
    one = generate_trajectory(QUESTIONS[1], policy, retr, tok, gold_answer=GOLDS[1], **KW)
    tok, policy, retr = _fresh()
    via_batch = generate_trajectories([QUESTIONS[1]], policy, retr, tok, golds=[GOLDS[1]], **KW)[0]
    assert one.input_ids.tolist() == via_batch.input_ids.tolist()
    assert one.token_type.tolist() == via_batch.token_type.tolist()


def test_replies_are_paired_by_index_not_by_content():
    """Two trajectories issuing the SAME query text in the same round must still get their
    own replies. Content-keyed pairing would look correct in most tests and cross the
    replies exactly here."""
    tok = MockTokenizer(vocab_size=8192)
    same = "capital of France"          # BOTH trajectories retrieve this exact query
    qs = ["twin alpha?", "twin beta?"]
    pol = _MultiPolicy(tok, {qs[0]: ([same], "answer-A"), qs[1]: ([same], "answer-B")})
    retr = MockRetriever(CORPUS, seed=0)

    trajs = generate_trajectories(qs, pol, retr, tok, golds=[["a"], ["b"]], **KW)
    assert trajs[0].meta["final_answer"] == "answer-A", "reply crossed between trajectories"
    assert trajs[1].meta["final_answer"] == "answer-B", "reply crossed between trajectories"
    # and both really did issue the identical query in the same round
    assert trajs[0].meta["turns"][0]["query"] == trajs[1].meta["turns"][0]["query"] == same


def test_act_batch_receives_3_TUPLES_and_this_is_the_contract():
    """PIN THE SHAPE the driver hands to act_batch, on CPU.

    rollout/rollout.py::_dispatch builds `[(r.prefix_ids, r.turn, r.forced) for r in acts]`
    and rollout/policy.py::act_batch_via_act unpacks exactly that. Every act_batch override
    in the repo depends on it -- but the two overrides that existed only DELEGATED to
    act_batch_via_act, so nothing ever looked at an element and nothing pinned the shape.

    tests/test_vllm_policy_scoring.py's recorder did look, and read `req.prefix_ids`. Against
    a tuple that raises AttributeError before any assertion runs, so both of its
    verbatim-ids tests -- the gates that protect the precompute cache from vLLM returning
    outputs in the wrong order -- had never executed. They are @slow/GPU-only, so no CPU run
    could see it; certify gate 4 found it the first time it got that far.

    This is the CPU twin: same inspection, mock policy, no GPU. A future refactor that turns
    the request back into an object, or reorders the fields, reddens here instead of on a
    cluster three weeks later.
    """
    import torch

    seen = []

    class _Inspecting(_MultiPolicy):
        def act_batch(self, requests):
            from rollout.policy import act_batch_via_act
            requests = list(requests)
            seen.extend(requests)
            return act_batch_via_act(self, requests)

    tok = MockTokenizer(vocab_size=8192)
    policy = _Inspecting(tok, {q: s for q, s in zip(QUESTIONS, SCRIPTS)})
    generate_trajectories(QUESTIONS, policy, MockRetriever(CORPUS, seed=0), tok,
                          golds=GOLDS, **KW)

    assert seen, "act_batch was never called -- the driver stopped using the batch surface"
    for i, req in enumerate(seen):
        # `type(req) is tuple`, NOT isinstance. The driver's own ActRequest is a NamedTuple,
        # which passes isinstance(_, tuple) AND supports attribute access -- so isinstance
        # cannot tell the contract from the thing the contract deliberately strips. Reverse-
        # verification caught exactly that: replacing the conversion with `list(acts)` left
        # this test green and only reddened the one below.
        assert type(req) is tuple and len(req) == 3, (
            f"request {i} is {type(req).__name__} of len "
            f"{len(req) if hasattr(req, '__len__') else '?'}, not a PLAIN 3-tuple: {req!r}. "
            "_dispatch converts ActRequest to a bare tuple on purpose -- the public contract "
            "is act(prefix_ids, turn, forced), and an external act_batch must not have to "
            "know an internal NamedTuple type.")
        prefix_ids, turn, forced = req
        assert isinstance(prefix_ids, torch.Tensor) and prefix_ids.dtype == torch.long
        assert isinstance(turn, int) and turn >= 0
        assert isinstance(forced, bool)


def test_attribute_access_on_a_request_FAILS():
    """The mistake itself, stated. Kept because the test above passes just as well if the
    contract silently became an object with .prefix_ids -- this is the half that says the
    tuple is the contract and attribute access is the error, not an alternative."""
    seen = []

    class _Inspecting(_MultiPolicy):
        def act_batch(self, requests):
            from rollout.policy import act_batch_via_act
            requests = list(requests)
            seen.extend(requests)
            return act_batch_via_act(self, requests)

    tok = MockTokenizer(vocab_size=8192)
    policy = _Inspecting(tok, {q: s for q, s in zip(QUESTIONS, SCRIPTS)})
    generate_trajectories(QUESTIONS, policy, MockRetriever(CORPUS, seed=0), tok,
                          golds=GOLDS, **KW)

    import pytest as _pytest
    with _pytest.raises(AttributeError):
        _ = seen[0].prefix_ids
