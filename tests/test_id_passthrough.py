"""Seam 1 (local, zero-hardware): the id-passthrough contract.

Real vLLM emits token ids directly. If the rollout ever round-tripped an action
through text (encode(decode(ids))), the </search>->\\n\\n<information> boundary merge
would corrupt the frozen sequence -- voiding the M2-M4 per-position online/offline
proof. MockPolicy hides this hazard (its ids ARE encode(text)), so we use a
StubIdPolicy whose returned ids DELIBERATELY differ from encode(its text):

  * verbatim append  => ACTION spans equal the stub's ids exactly (this passes).
  * any hidden re-encode => ACTION spans equal encode(text) != stub ids (fails).

It also pins prefix correctness (M6 pin #1): the ids handed to the policy as the
generation context must be bit-for-bit the trajectory prefix appended so far --
an off-by-one there is subtler than detok->retok and only real (id-emitting)
generation would expose it. We force the check to bite locally.
"""
from __future__ import annotations

import torch

from configs import template as T
from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.trajectory import TokenType
from rollout.policy import ActionStep
from rollout.rollout import generate_trajectory, render_prompt

CORPUS = ["Paris is the capital of France.", "Berlin is the capital of Germany."]


class StubIdPolicy:
    """Returns hand-picked ACTION ids that disagree with encode(text). Records the
    prefix it was handed at each step so the test can assert prefix correctness."""

    def __init__(self, action_ids: list[torch.Tensor], texts: list[str]) -> None:
        self.action_ids = action_ids
        self.texts = texts
        self.seen_prefixes: list[torch.Tensor] = []

    def act(self, prefix_ids: torch.Tensor, turn: int) -> ActionStep:
        self.seen_prefixes.append(prefix_ids.clone())
        return ActionStep(ids=self.action_ids[turn].clone(), text=self.texts[turn])


def test_action_ids_appended_verbatim_never_reencoded():
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)

    t0 = f"{T.THINK_OPEN}x{T.THINK_CLOSE}{T.SEARCH_OPEN}capital of France{T.SEARCH_CLOSE}"
    t1 = f"{T.THINK_OPEN}y{T.THINK_CLOSE}{T.ANSWER_OPEN}Paris{T.ANSWER_CLOSE}"
    a0 = torch.tensor([101, 102, 103], dtype=torch.long)   # NOT encode(t0)
    a1 = torch.tensor([201, 202, 203], dtype=torch.long)   # NOT encode(t1)

    # sanity: the stub is genuinely adversarial -- a re-encode would change the ids
    assert tok.encode(t0) != a0.tolist()
    assert tok.encode(t1) != a1.tolist()

    policy = StubIdPolicy([a0, a1], [t0, t1])
    traj = generate_trajectory("q?", policy, retr, tok,
                               max_turns=4, max_len=4096, top_k=1)

    # reconstruct the sequence the id-passthrough contract must produce, exactly:
    #   prompt + a0(ACTION) + obs0(OBS) + a1(ACTION) + im_end(ACTION harness EOS)
    prompt = tok.encode_tensor(render_prompt("q?"))
    docs = retr.retrieve("capital of France", k=1, step=0)
    obs_text = T.OBS_PREFIX + T.INFO_OPEN + "\n".join(docs) + T.INFO_CLOSE + T.OBS_SUFFIX
    obs0 = tok.encode_tensor(obs_text)
    imend = tok.encode_tensor(T.IM_END)
    expected_ids = torch.cat([prompt, a0, obs0, a1, imend])
    assert torch.equal(traj.input_ids, expected_ids)

    # and the ACTION-typed positions are exactly the stub ids (+ EOS), never encode(text)
    action_ids = traj.input_ids[traj.token_type == int(TokenType.ACTION)]
    assert torch.equal(action_ids, torch.cat([a0, a1, imend]))


def test_prefix_fed_to_policy_matches_trajectory_prefix():
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    t0 = f"{T.THINK_OPEN}x{T.THINK_CLOSE}{T.SEARCH_OPEN}capital of France{T.SEARCH_CLOSE}"
    t1 = f"{T.THINK_OPEN}y{T.THINK_CLOSE}{T.ANSWER_OPEN}Paris{T.ANSWER_CLOSE}"
    a0 = torch.tensor([101, 102, 103], dtype=torch.long)
    a1 = torch.tensor([201, 202, 203], dtype=torch.long)
    policy = StubIdPolicy([a0, a1], [t0, t1])
    traj = generate_trajectory("q?", policy, retr, tok,
                               max_turns=4, max_len=4096, top_k=1)

    prompt = tok.encode_tensor(render_prompt("q?"))
    docs = retr.retrieve("capital of France", k=1, step=0)
    obs_text = T.OBS_PREFIX + T.INFO_OPEN + "\n".join(docs) + T.INFO_CLOSE + T.OBS_SUFFIX
    obs0 = tok.encode_tensor(obs_text)

    # step 0 sees the prompt; step 1 sees prompt + action0 + observation0 -- exactly.
    assert torch.equal(policy.seen_prefixes[0], prompt)
    assert torch.equal(policy.seen_prefixes[1], torch.cat([prompt, a0, obs0]))
    # general invariant: every prefix handed to the policy is a true prefix of the
    # final trajectory ids (no off-by-one between generation context and scoring).
    for pref in policy.seen_prefixes:
        assert torch.equal(pref, traj.input_ids[: pref.shape[0]])
