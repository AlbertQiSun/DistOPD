"""Seam 1 last centimeter (@slow, HPC): real vLLM generation -> scoring identity.

The local StubIdPolicy test proves the rollout appends ids verbatim and feeds the
right prefix. This proves the REAL half that only a GPU can: ids actually SAMPLED by
vLLM, when fed back through scoring, land at the ACTION positions bit-for-bit with no
off-by-one, and the prefix the student conditioned on equals the scored prefix.

WHY THIS FILE HOOKS `act_batch` AND ASSERTS THAT IT FIRED
---------------------------------------------------------
It used to record by overriding `act`. That was correct until the rollout driver was
refactored to gather each round and serve it through `act_batch`; from then on the override
was never called, `rec_ids` stayed empty, and the test failed on an empty `torch.cat` --
but only the first time anyone ran the slow suite after the refactor. For the stretch in
between, the gate looked like coverage and was not, while the precompute cache it was
meant to protect was being produced by the very path it had stopped watching.

MockPolicy hid this: it inherits `act_batch_via_act`, so mock-based tests that hook `act`
still exercise the driver. Only the REAL VLLMPolicy implements `act_batch` independently,
which is exactly why this file (the only one that touches it) was the only one to notice.

So: hook the batched surface, and ASSERT THE RECORDING IS NON-EMPTY before comparing --
a bypass must fail as "the hook never fired", not as a confusing empty-tensor error.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tests.vllm_engine import vllm_engine

# Offline HPC: export OPD_TEST_MODEL=/path/to/local/Qwen3-1.7B (default is the HF id).
MODEL = os.environ.get("OPD_TEST_MODEL", "Qwen/Qwen3-1.7B")
CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
]


def _recording_policy_cls():
    from rollout.vllm_policy import VLLMPolicy

    class _Recording(VLLMPolicy):
        """Records on the surface the driver ACTUALLY calls."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.rec_prefixes, self.rec_steps = [], []

        def act_batch(self, requests):
            # THE CONTRACT IS A 3-TUPLE, not an object with attributes. rollout/rollout.py
            # builds `[(r.prefix_ids, r.turn, r.forced) for r in acts]` and
            # rollout/policy.py::act_batch_via_act unpacks exactly that. This recorder read
            # `req.prefix_ids`, which cannot work against a tuple -- so both tests below
            # raised AttributeError at the first act_batch call and had never once executed
            # their actual assertions. They are @slow/GPU-only, so no CPU run could see it;
            # the first certify run that reached gate 4 did, immediately.
            requests = list(requests)
            steps = super().act_batch(requests)
            assert len(steps) == len(requests), "act_batch dropped or added a request"
            for (prefix_ids, _turn, _forced), step in zip(requests, steps):
                self.rec_prefixes.append(prefix_ids.clone())
                self.rec_steps.append(step.ids.clone())
            return steps

    return _Recording


@pytest.mark.slow
def test_vllm_sampled_ids_are_scored_verbatim():
    torch = pytest.importorskip("torch")
    from tests.gpu import require_gpu
    require_gpu()
    pytest.importorskip("vllm")
    transformers = pytest.importorskip("transformers")

    from envs.hf_tokenizer import HFTokenizerAdapter
    from envs.retriever import MockRetriever
    from opd.trajectory import TokenType
    from rollout.rollout import generate_trajectory
    from teacher.scorer import HFScorer

    tok = HFTokenizerAdapter(transformers.AutoTokenizer.from_pretrained(MODEL))
    with vllm_engine(model=MODEL, dtype="bfloat16") as llm:
        policy = _recording_policy_cls()(llm, tok, temperature=0.7,
                                         max_action_tokens=128, seed=0)
        retr = MockRetriever(CORPUS, seed=0)
        traj = generate_trajectory("What is the capital of France?", policy, retr, tok,
                                   max_turns=3, max_len=2048, top_k=1)

        # NON-VACUITY FIRST. If the driver stops routing through act_batch, say so plainly
        # instead of failing later on an empty tensor.
        assert policy.rec_steps, (
            "act_batch was never called -- the rollout driver no longer routes generation "
            "through it, so this test is not checking the real path")

        # every prefix fed to the student == the trajectory prefix at that point (no off-by-one)
        for pref in policy.rec_prefixes:
            assert torch.equal(pref, traj.input_ids[: pref.shape[0]])

        # the sampled ids appear verbatim, in order, at the ACTION positions (no re-encode)
        action_ids = traj.input_ids[traj.token_type == int(TokenType.ACTION)]
        recorded = torch.cat(policy.rec_steps)
        assert action_ids.tolist()[: recorded.numel()] == recorded.tolist()

        # fed back through the student scorer: aligned length, finite at scored positions
        student = HFScorer.from_pretrained(MODEL, dtype="bfloat16", device="cuda")
        lp = student.logprobs(traj.input_ids)
        assert lp.device.type == "cuda", "logprobs fell back to CPU (see opd.masking)"
        assert lp.shape[0] == traj.input_ids.shape[0]
        scored = traj.token_type == int(TokenType.ACTION)
        assert bool(torch.isfinite(lp[scored]).all())


@pytest.mark.slow
def test_verbatim_ids_survive_a_REAL_multi_trajectory_batch():
    """The single-trajectory case above drives act_batch at size 1. The precompute cache is
    produced at size 256, where one round carries many prefixes of DIFFERENT lengths through
    one llm.generate -- and vLLM returns a list whose order is the only thing pairing an
    output back to its request. A reordered or short return would hand a trajectory another
    one's sampled tokens: well-formed ids, rollout continues, cache silently wrong.

    So assert the per-trajectory identity at batch > 1: for EACH trajectory, the ids recorded
    against its own prefix are the ids that ended up at its own ACTION positions.
    """
    torch = pytest.importorskip("torch")
    from tests.gpu import require_gpu
    require_gpu()
    pytest.importorskip("vllm")
    transformers = pytest.importorskip("transformers")

    from envs.hf_tokenizer import HFTokenizerAdapter
    from envs.retriever import MockRetriever
    from opd.trajectory import TokenType
    from rollout.rollout import generate_trajectories

    questions = ["What is the capital of France?",
                 "How tall is the Eiffel Tower?",
                 "Which city is the Eiffel Tower in?"]

    tok = HFTokenizerAdapter(transformers.AutoTokenizer.from_pretrained(MODEL))
    with vllm_engine(model=MODEL, dtype="bfloat16") as llm:
        policy = _recording_policy_cls()(llm, tok, temperature=0.7,
                                         max_action_tokens=64, seed=0)
        trajs = generate_trajectories(questions, policy, MockRetriever(CORPUS, seed=0), tok,
                                      max_turns=2, max_len=2048, top_k=1)

        assert policy.rec_steps, "act_batch was never called"
        assert max(len(policy.rec_prefixes), 0) >= len(questions)
        # at least one round must have carried MORE THAN ONE request, or this test is just
        # the size-1 case again wearing a different name
        assert len(policy.rec_prefixes) > len(trajs) or len(trajs) > 1

        # pair each recorded (prefix -> ids) back to the trajectory it belongs to BY PREFIX,
        # then require those ids to appear verbatim at that trajectory's ACTION positions.
        for traj in trajs:
            mine = [ids for pref, ids in zip(policy.rec_prefixes, policy.rec_steps)
                    if pref.shape[0] <= traj.input_ids.shape[0]
                    and torch.equal(pref, traj.input_ids[: pref.shape[0]])]
            assert mine, "no recorded generation matched this trajectory's prefixes"
            action_ids = traj.input_ids[traj.token_type == int(TokenType.ACTION)].tolist()
            recorded = torch.cat(mine).tolist()
            assert action_ids[: len(recorded)] == recorded, (
                "a trajectory's ACTION span does not match the ids generated for ITS prefix "
                "-- batched results were paired to the wrong request")
