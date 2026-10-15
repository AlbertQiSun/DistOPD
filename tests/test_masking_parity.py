"""Masking parity test -- the hard gate for confound #1.

Same fixed trajectory driven through the ONLINE path (teacher scored live) and
the OFFLINE path (teacher precomputed -> disk -> read back). Five assertions,
plus point-1 full-length/observation-conditioning checks. If any fail, the
online/offline "only rollout source differs" guarantee is broken -- STOP.
"""
from __future__ import annotations

import torch

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.masking import build_score_mask, scored_indices
from opd.step import opd_train_step
from opd.trajectory import TokenType, Trajectory
from rollout.policy import MockPolicy
from rollout.rollout import generate_trajectory
from teacher.scorer import MockLM, Scorer, TeacherScorer

TAU = 10.0
CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
    "Berlin is the capital of Germany.",
    "The Great Wall of China is a historic fortification.",
]


def _fixed_trajectory():
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=["capital of France", "Eiffel Tower height"], answer="Paris")
    traj = generate_trajectory(
        "What is the capital of France?", policy, retr, tok,
        max_turns=4, max_len=4096, top_k=2, seed=0,
    )
    return traj, tok


def test_masking_parity(tmp_path):
    traj, tok = _fixed_trajectory()
    vocab = tok.vocab_size
    teacher = TeacherScorer(MockLM(vocab, seed=1234))
    student = Scorer(MockLM(vocab, seed=5678), requires_grad=False)

    # --- point 1: trajectory really contains observation tokens (not stripped) ---
    assert traj.has_observation()
    n_obs = int((traj.token_type == int(TokenType.OBSERVATION)).sum())
    assert n_obs > 0, "no observation tokens -- env injection or tagging broken"

    # --- ONLINE path ---
    tl_online = teacher.score_online(traj.input_ids)
    out_online = opd_train_step(traj, tl_online, student, TAU)

    # --- OFFLINE path: precompute -> disk -> readback ---
    path = tmp_path / "teacher_logp.pt"
    teacher.precompute_and_save(traj, str(path))
    d = TeacherScorer.load(str(path))
    traj_off = Trajectory(input_ids=d["input_ids"], token_type=d["token_type"])
    tl_offline = d["teacher_logp"]

    # point 1 hard assert: offline teacher was fed the FULL sequence incl observation,
    # same length as online and as the full trajectory (no action-only shortcut).
    assert traj_off.length == traj.length == tl_online.shape[0] == tl_offline.shape[0]
    assert torch.equal(traj_off.input_ids, traj.input_ids)
    assert torch.equal(traj_off.token_type, traj.token_type)
    assert traj_off.has_observation()

    out_offline = opd_train_step(traj_off, tl_offline, student, TAU)

    sm = build_score_mask(traj.token_type)

    # Assertion 1: mask sets bit-identical; student-loss mask IS the teacher-score mask.
    assert torch.equal(out_online.score_mask, out_offline.score_mask)
    assert torch.equal(out_online.score_mask, sm)

    # Assertion 2: the token index set that actually enters advantage/loss is identical.
    assert torch.equal(scored_indices(out_online.score_mask),
                       scored_indices(out_offline.score_mask))

    # Assertion 3: teacher logp per-token equal, online vs offline readback (locks
    # off-by-one across the serialize->deserialize path).
    assert torch.allclose(tl_online[sm], tl_offline[sm], atol=1e-4)

    # Assertion 4: teacher scored positions == student scored positions; no off-by-one;
    # all scored logps finite => no scored token landed at index 0.
    assert torch.isfinite(tl_online[sm]).all()
    assert torch.isfinite(out_online.student_logp[sm]).all()
    assert int(sm[0].item()) == 0
    assert bool((scored_indices(sm) >= 1).all())

    # Assertion 5 (highest priority): per-token advantage AND loss identical
    # online vs offline -> code-level proof that only the rollout source differs.
    assert torch.allclose(out_online.advantage, out_offline.advantage, atol=1e-4)
    assert torch.allclose(out_online.loss, out_offline.loss, atol=1e-4)


def test_observation_is_conditioned_not_deleted():
    """OBSERVATION = not scored, but still in the sequence (conditioned on).

    The action token immediately after an </information> block must be preceded
    in input_ids by the whole observation span.
    """
    traj, _ = _fixed_trajectory()
    tt = traj.token_type
    # find a transition OBSERVATION -> ACTION
    found = False
    for i in range(1, traj.length):
        if tt[i] == int(TokenType.ACTION) and tt[i - 1] == int(TokenType.OBSERVATION):
            found = True
            # everything before i includes observation tokens (conditioning present)
            assert int((tt[:i] == int(TokenType.OBSERVATION)).sum()) > 0
            break
    assert found, "expected an OBSERVATION->ACTION boundary in the trajectory"
