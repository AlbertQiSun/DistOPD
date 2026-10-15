"""Multi-turn agentic-search rollout with token-type tagging.

Builds ONE flat token-id sequence: prompt (rendered once) + interleaved ACTION
spans (student) and OBSERVATION spans (retriever). Never re-renders the chat
template per turn, never detokenizes->retokenizes. Records the terminated reason
(confound #2 hook).

BATCHING IS A SCHEDULER, NOT A SECOND IMPLEMENTATION
----------------------------------------------------
global_batch=256 makes one-trajectory-at-a-time sampling infeasible, but writing a
separate "batched rollout" would fork the per-trajectory control flow (forced-answer
Guard 1, duplicate-query Guard 2, truncation, termination bookkeeping) into two copies
that must then be kept equal by testing. Instead the loop body lives ONCE, as a
COROUTINE: wherever it needs the policy or the retriever it `yield`s a request and is
resumed with the reply. A driver decides whether to serve those requests one at a time
or gathered across trajectories.

So "batching does not change a trajectory's behaviour" is STRUCTURAL, not a property we
verify after the fact -- there is only one copy of the logic, and batching alters only
WHEN a request is issued, never the control flow that consumes it. Same discipline as
the never-detok->retok rule and the assert welded to the disk boundary: make the failure
mode unrepresentable rather than merely tested for.

Trajectories in a batch desynchronise (one answers on turn 2, another runs to max_turns),
so the driver re-gathers whatever each coroutine asks for next, round by round, until all
have returned. A finished trajectory simply stops contributing requests.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from configs import template as T
from opd.trajectory import TerminationReason, TokenType, Trajectory, TrajectoryBuilder


class ActRequest(NamedTuple):
    """A coroutine asking the policy to generate the next ACTION span."""
    prefix_ids: torch.Tensor
    turn: int
    forced: bool


class RetrieveRequest(NamedTuple):
    """A coroutine asking the retriever for this turn's documents."""
    query: str
    k: int
    turn: int


def render_prompt(question: str) -> str:
    """One-shot prompt render (enable_thinking=True, add_generation_prompt=True).

    Mirrors Qwen3 ChatML. In real runs this is tokenizer.apply_chat_template(...);
    here it is the equivalent literal string so the MockTokenizer produces a
    stable prefix. NOTE: under enable_thinking=True the real render's assistant tail
    is `<|im_start|>assistant\\n` -- byte-identical to this literal (no injected
    empty <think></think> block), so mock and real prompts no longer diverge.
    """
    return (f"{T.IM_START}system\n{T.SYSTEM_PROMPT}{T.IM_END}\n"
            f"{T.IM_START}user\n{question}{T.IM_END}\n"
            f"{T.IM_START}assistant\n")


def _extract_between(text: str, open_tag: str, close_tag: str) -> str:
    i = text.find(open_tag)
    j = text.find(close_tag, i + len(open_tag)) if i >= 0 else -1
    if i < 0 or j < 0:
        return ""
    return text[i + len(open_tag):j]


def _trajectory_coro(question: str, tokenizer, *,
                     max_turns: int, max_len: int, top_k: int = 2,
                     seed: int = 0, gold_answer: Optional[str] = None):
    """THE rollout logic -- the single copy. Yields ActRequest / RetrieveRequest and is
    resumed with the corresponding reply; returns the finished Trajectory. Takes no policy
    or retriever: it never calls them, so it cannot depend on how they are dispatched."""
    b = TrajectoryBuilder()
    # real HF tokenizer: render prompt once via apply_chat_template (frozen flags);
    # mock tokenizer: encode the equivalent literal ChatML string.
    if hasattr(tokenizer, "render_prompt_ids"):
        prompt_ids = tokenizer.render_prompt_ids(question)
    else:
        prompt_ids = tokenizer.encode_tensor(render_prompt(question))
    b.add(prompt_ids, TokenType.PROMPT)

    terminated: Optional[TerminationReason] = None
    num_search = 0
    turn = 0
    final_answer: Optional[str] = None    # the <answer>...</answer> span, task EM/F1 headline
    # per-turn {query, docs} recorded for confound-#2/#3 diagnostics (context quality).
    # These are the raw materials for answer-recall@k; captured at rollout time because
    # they cannot be recovered from the token ids alone (docs are retriever outputs).
    turns_meta: list[dict] = []
    # Guard 2 bookkeeping: normalized text of every query actually retrieved this rollout.
    # Retrieval is DETERMINISTIC (same query -> same docs), so an EXACT repeat yields zero
    # new evidence. We dedup on exact normalized match only -- near-match would wrongly kill
    # legitimate progressive query refinement.
    executed_queries: set[str] = set()
    # diagnostics for the forced-final-answer seam (Guard 1): did the env have to force an
    # answer, and did the forced turn actually produce one. Report-only (never gates); read by
    # the health/smoke report to size the forced-answer trigger rate & its success rate.
    forced_answer_fired = False
    forced_answer_answered = False
    # DIAGNOSTIC (report-only): the FULL text the model emitted on the forced turn. Stored in
    # full (not truncated) so we can see whether a failed forced turn was habitually re-searching,
    # a long <think> that got cut before <answer> (the tail is where <answer> would appear), or a
    # genuine format miss. None unless Guard 1 actually took a forced turn.
    forced_answer_text: Optional[str] = None
    # DIAGNOSTIC (report-only): did the teacher's native <think> channel actually fire in ANY
    # ACTION span this rollout. This is the smoke's think-appearance signal -- with
    # enable_thinking=True it should be ~always True; a low batch rate means the think channel
    # is being suppressed again (regression guard for the empty-<think></think> bug).
    any_think = False
    # raw text of the action that ENDED the trajectory, on whichever path it ended.
    final_action_text = None
    final_action_n_tok = None

    while True:
        # feed the EXACT id prefix appended so far as the generation context, then
        # append the returned ids VERBATIM. text is decode(ids) for control flow only
        # -- never re-encoded (that would reintroduce the boundary-merge hazard).
        step = yield ActRequest(b.current_ids(), turn, False)
        action_text = step.text
        b.add(step.ids, TokenType.ACTION)
        if T.THINK_OPEN in action_text:
            any_think = True

        if T.ANSWER_CLOSE in action_text:
            final_answer = _extract_between(action_text, T.ANSWER_OPEN, T.ANSWER_CLOSE)
            # the WHOLE final action, not just the extracted span. final_answer is what the
            # scorer sees; this is what the model actually wrote, and the difference between
            # them IS the measurement -- deliberation left inside the span, a restated
            # question, a second <answer> block. Without it, "is the answer contaminated" can
            # only be asked of eval reports, never of the teacher data pi_ref is trained on.
            final_action_text = action_text
            final_action_n_tok = int(step.ids.numel())
            b.add(tokenizer.encode_tensor(T.IM_END), TokenType.ACTION)  # student learns EOS
            terminated = TerminationReason.ANSWERED
            break

        # NON-ANSWER action. Note this branch does NOT require a <search> tag: anything that
        # is not a closed <answer> lands here and is counted as a search turn. _extract_between
        # returns "" when the tags are absent, so `query == ""` covers THREE distinct failures
        # -- a literal <search></search>, an unclosed <answer>, and free prose with no tool call
        # at all. sft.datagen calls all three "empty_query"; only the raw `text` recorded below
        # can tell them apart, which is why it is persisted (a teacher whose format never fits
        # the protocol and one that merely stops mid-answer need opposite fixes).
        num_search += 1
        query = _extract_between(action_text, T.SEARCH_OPEN, T.SEARCH_CLOSE)
        qnorm = " ".join(query.split()).lower()
        if qnorm and qnorm in executed_queries:
            # Guard 2: exact repeat -> skip retrieval (deterministic => no new docs) and inject
            # a hint OBSERVATION (masked) nudging a different query or an answer.
            turns_meta.append({"turn": turn, "query": query, "docs": [], "duplicate": True,
                               "text": action_text, "n_tok": int(step.ids.numel())})
            obs_text = T.OBS_PREFIX + T.DUPLICATE_QUERY_HINT + T.OBS_SUFFIX
        else:
            if qnorm:
                executed_queries.add(qnorm)
            docs = yield RetrieveRequest(query, top_k, turn)
            # n_tok is len(sampled ids) -- the EXACT number vLLM generated, not a re-encode
            # of the decoded text. It is the only way to tell an action that STOPPED at
            # </search> from one that ran out of max_action_tokens, and those are the same
            # symptom (no usable query) from opposite causes.
            turns_meta.append({"turn": turn, "query": query, "docs": list(docs),
                               "text": action_text, "n_tok": int(step.ids.numel())})
            obs_text = T.OBS_PREFIX + T.INFO_OPEN + "\n".join(docs) + T.INFO_CLOSE + T.OBS_SUFFIX
        b.add(tokenizer.encode_tensor(obs_text), TokenType.OBSERVATION)

        turn += 1
        if b.current_length() >= max_len:
            terminated = TerminationReason.TRUNCATED_MAXLEN
            break
        if turn >= max_turns:
            # Guard 1: forced-final-answer seam. Budget exhausted -> inject a warning
            # OBSERVATION (masked), then take ONE MORE SUPERVISED action that must answer.
            # This is the seam Search-R1/StepSearch/s3 get from do_search=False on the last
            # turn; without it the teacher spins to max_turns and never emits <answer>.
            forced_answer_fired = True
            warn_text = T.OBS_PREFIX + T.FORCED_ANSWER_WARNING + T.OBS_SUFFIX
            b.add(tokenizer.encode_tensor(warn_text), TokenType.OBSERVATION)
            if b.current_length() >= max_len:
                terminated = TerminationReason.TRUNCATED_MAXLEN
                break
            # PREFILL the <answer> open tag as an OBSERVATION span. It is ENVIRONMENT-forced
            # (the env pushes the model into the answer span), NOT the model's own termination
            # decision, so it is MASKED like <information>/<warning>: still conditioned on, but
            # never supervised. Supervising it would teach pi_ref "<answer> appears
            # automatically" instead of "WHEN to autonomously emit <answer>" -- corrupting the
            # very termination decision natural trajectories teach. Only the model's
            # continuation (answer body + </answer> + <|im_end|>) is ACTION.
            b.add(tokenizer.encode_tensor(T.ANSWER_OPEN), TokenType.OBSERVATION)
            step = yield ActRequest(b.current_ids(), turn, True)
            forced_answer_text = step.text         # capture in full for the diagnostic
            final_action_text = step.text           # same field on both termination paths
            final_action_n_tok = int(step.ids.numel())
            b.add(step.ids, TokenType.ACTION)      # answer BODY (+ </answer>) IS supervised
            if T.THINK_OPEN in step.text:
                any_think = True
            if T.ANSWER_CLOSE in step.text:
                # <answer> was prefilled as OBSERVATION (not in step.text); prepend it before
                # extracting so the answer span is well-formed for the EM/F1 headline.
                final_answer = _extract_between(T.ANSWER_OPEN + step.text,
                                                T.ANSWER_OPEN, T.ANSWER_CLOSE)
                b.add(tokenizer.encode_tensor(T.IM_END), TokenType.ACTION)
                terminated = TerminationReason.ANSWERED
                forced_answer_answered = True
            else:
                # model failed to close <answer> even with the tag prefilled -> genuinely
                # non-terminating (no supervised terminal answer span).
                terminated = TerminationReason.TRUNCATED_MAXTURNS
            break

    # count exact-repeat queries Guard 2 intercepted (verbatim-repeat signal for the report).
    n_duplicate_queries = sum(1 for t in turns_meta if t.get("duplicate"))
    traj = b.build(terminated=terminated, num_search_calls=num_search,
                   meta={"question": question, "seed": seed,
                         "gold_answer": gold_answer, "final_answer": final_answer,
                         "turns": turns_meta,
                         "forced_answer_fired": forced_answer_fired,
                         "forced_answer_answered": forced_answer_answered,
                         "forced_answer_text": forced_answer_text,
                         "final_action_text": final_action_text,
                         "final_action_n_tok": final_action_n_tok,
                         "has_think": any_think,
                         "n_duplicate_queries": n_duplicate_queries})

    # hard cap: truncate to max_len; downgrade termination if it cut off the answer
    if traj.length > max_len:
        input_ids = traj.input_ids[:max_len]
        token_type = traj.token_type[:max_len]
        term = traj.terminated
        if term == TerminationReason.ANSWERED:
            term = TerminationReason.TRUNCATED_MAXLEN
        traj = Trajectory(input_ids=input_ids.long(), token_type=token_type.long(),
                          terminated=term, num_search_calls=num_search, meta=traj.meta)
    return traj


# ---------------------------------------------------------------------------
# drivers -- the ONLY thing that differs between one-at-a-time and batched rollout
# ---------------------------------------------------------------------------
def _dispatch(policy, retriever, acts: list, retrieves: list) -> tuple[list, list]:
    """Serve one round of gathered requests.

    `act_batch` / `retrieve_batch` are OPTIONAL overrides, not new requirements. The
    contract is still `act(prefix_ids, turn, forced)` / `retrieve(query, k, step)`
    (rollout/policy.py), so anything implementing only those -- including test stubs --
    keeps working and simply gets served one at a time. Making the batch surface
    mandatory would have quietly redefined the contract that the rest of the codebase,
    and every existing implementation, is written against.
    """
    from envs.retriever import retrieve_batch_via_retrieve
    from rollout.policy import act_batch_via_act

    steps: list = []
    if acts:
        reqs = [(r.prefix_ids, r.turn, r.forced) for r in acts]
        fn = getattr(policy, "act_batch", None)
        steps = fn(reqs) if fn is not None else act_batch_via_act(policy, reqs)
    docs: list = []
    if retrieves:
        reqs = [(r.query, r.k, r.turn) for r in retrieves]
        fn = getattr(retriever, "retrieve_batch", None)
        docs = fn(reqs) if fn is not None else retrieve_batch_via_retrieve(retriever, reqs)
    return steps, docs


def _drive(coros: list, policy, retriever) -> list:
    """Round-by-round scheduler over N rollout coroutines.

    Each round: resume every LIVE coroutine, gather what it asks for, serve the asks in
    ONE batched call per kind, hand the replies back. Trajectories desynchronise freely --
    a coroutine that returned is simply absent from later rounds -- so the batch shrinks
    as trajectories finish rather than forcing them to run in lockstep.

    Request->reply pairing is positional and index-keyed, never by identity: two
    trajectories can legitimately issue the SAME query text in the same round, and
    matching on content would cross their replies.
    """
    results: list = [None] * len(coros)
    pending: dict[int, object] = {}
    for i, coro in enumerate(coros):
        try:
            pending[i] = coro.send(None)                 # prime to the first request
        except StopIteration as done:
            results[i] = done.value

    while pending:
        act_idx = [i for i, r in pending.items() if isinstance(r, ActRequest)]
        ret_idx = [i for i, r in pending.items() if isinstance(r, RetrieveRequest)]
        steps, docs = _dispatch(policy, retriever,
                                [pending[i] for i in act_idx],
                                [pending[i] for i in ret_idx])
        assert len(steps) == len(act_idx), f"act_batch returned {len(steps)} for {len(act_idx)} requests"
        assert len(docs) == len(ret_idx), f"retrieve_batch returned {len(docs)} for {len(ret_idx)} requests"
        replies = dict(zip(act_idx, steps))
        replies.update(zip(ret_idx, docs))

        pending = {}
        for i in sorted(replies):                        # deterministic resume order
            try:
                pending[i] = coros[i].send(replies[i])
            except StopIteration as done:
                results[i] = done.value
    return results


def generate_trajectory(question: str, policy, retriever, tokenizer, *,
                        max_turns: int, max_len: int, top_k: int = 2,
                        seed: int = 0, gold_answer: Optional[str] = None) -> Trajectory:
    """One trajectory. Unchanged signature/behaviour -- it drives the same coroutine the
    batched path drives, with a batch of size one."""
    return generate_trajectories([question], policy, retriever, tokenizer,
                                 max_turns=max_turns, max_len=max_len, top_k=top_k,
                                 seed=seed, golds=[gold_answer])[0]


def generate_trajectories(questions: list[str], policy, retriever, tokenizer, *,
                          max_turns: int, max_len: int, top_k: int = 2,
                          seed: int = 0, golds: Optional[list] = None) -> list[Trajectory]:
    """N trajectories, sampled/retrieved in gathered rounds. Same coroutine as the single
    path, so a trajectory's control flow cannot depend on who it shares a batch with."""
    golds = list(golds) if golds is not None else [None] * len(questions)
    assert len(golds) == len(questions), "golds must align 1:1 with questions"
    coros = [_trajectory_coro(q, tokenizer, max_turns=max_turns, max_len=max_len,
                              top_k=top_k, seed=seed, gold_answer=g)
             for q, g in zip(questions, golds)]
    return _drive(coros, policy, retriever)
