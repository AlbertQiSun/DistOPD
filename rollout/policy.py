"""Student policy contract for the rollout.

THE contract is `act(prefix_ids, turn, forced=False) -> ActionStep(ids, text)`:
  - `forced` : True only on the rollout's forced-final-answer turn (Guard 1), where the
             env has PREFILLED the <answer> open tag. When forced, the policy must emit only
             the answer BODY (+ </answer>) -- never the <answer> open tag again.
  - `ids`  : the token ids that become the ACTION span, appended VERBATIM. For a
             real model these are the sampled ids -- they are never round-tripped
             through text, so the </search>->\\n\\n<information> boundary merge that
             detok->retok would trigger cannot corrupt the frozen sequence.
  - `text` : decode(ids), used ONLY for control flow (answer/search tag detection,
             query extraction). It is never re-encoded back into the trajectory.

`MockPolicy` is the dry-run adapter: it encodes its scripted text ONCE with the
same tokenizer the rollout uses, so `ids == encode(text)` reproduces the exact
M2-M4 trajectories bit-for-bit while still flowing through the id-passthrough path.
The real path (`rollout/vllm_policy.py`) implements the same contract with vLLM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from configs import template as T


@dataclass
class ActionStep:
    ids: torch.Tensor   # LongTensor [n]; appended VERBATIM as the ACTION span
    text: str           # decode(ids); control flow ONLY, never re-encoded


def act_batch_via_act(policy, requests) -> list[ActionStep]:
    """Default `act_batch`: serve a gathered round by calling `act` once per request.

    Every policy gets a batch surface for free, so the rollout driver can be written
    against ONE interface and a backend that cannot batch is not a special case. Because
    this default is a plain loop over the SAME `act`, adding the surface is behaviour-
    preserving by construction -- only a backend that overrides it (vLLM, which submits
    the whole round to its scheduler) actually changes how the work is issued.
    """
    # `forced` is passed ONLY when True, exactly as the pre-batching call sites did
    # (`policy.act(ids, turn)` on the normal path, `forced=True` only on Guard 1's forced
    # turn). Always passing it would change the call convention and break any policy whose
    # `act` does not declare the kwarg -- the contract makes it optional.
    return [policy.act(prefix_ids, turn, forced=True) if forced else policy.act(prefix_ids, turn)
            for prefix_ids, turn, forced in requests]


class MockPolicy:
    """Scripted 'student' for dry-run. Emits <think><search> for a few turns then
    <think><answer>. Deterministic. Encodes text ONCE via the shared tokenizer so the
    produced ACTION ids are identical to the pre-M6 (text-then-encode) pipeline."""

    def __init__(self, tokenizer, queries: list[str], answer: str, *,
                 forced_answer_closes: bool = True) -> None:
        self.tok = tokenizer
        self.queries = list(queries)
        self.answer = answer
        # On the forced turn (rollout Guard 1) the env has PREFILLED the <answer> open tag, so
        # the policy emits only the answer BODY. forced_answer_closes=True appends </answer>
        # (the normal case -> terminates ANSWERED); False emits the body with NO </answer>,
        # simulating a model that fails to close even when forced -> TRUNCATED_MAXTURNS. This
        # knob exists ONLY so tests can exercise the non-terminating forced path.
        self.forced_answer_closes = forced_answer_closes

    def _text(self, turn: int) -> str:
        if turn < len(self.queries):
            q = self.queries[turn]
            return (f"{T.THINK_OPEN}I should look this up.{T.THINK_CLOSE}"
                    f"{T.SEARCH_OPEN}{q}{T.SEARCH_CLOSE}")
        return (f"{T.THINK_OPEN}I can answer now.{T.THINK_CLOSE}"
                f"{T.ANSWER_OPEN}{self.answer}{T.ANSWER_CLOSE}")

    def _forced_text(self) -> str:
        # <answer> open tag is PREFILLED by the env (OBSERVATION); produce only the body
        # (+ closing tag). Never includes the open tag -- that would double it.
        if self.forced_answer_closes:
            return f"{self.answer}{T.ANSWER_CLOSE}"
        return self.answer   # body only, no </answer> -> forced turn fails to terminate

    def act(self, prefix_ids: torch.Tensor, turn: int,
            forced: bool = False) -> ActionStep:  # noqa: ARG002
        text = self._forced_text() if forced else self._text(turn)
        ids = self.tok.encode_tensor(text)
        return ActionStep(ids=ids.long(), text=text)

    def act_batch(self, requests) -> list[ActionStep]:
        return act_batch_via_act(self, requests)

    def sync_from(self, student) -> None:  # noqa: ARG002
        """No-op: the scripted mock has no separate inference weights to update.
        The resync SEMANTICS (dual weights, staleness) are proven with a mock
        inference engine in tests/test_online_resync.py."""
        return
