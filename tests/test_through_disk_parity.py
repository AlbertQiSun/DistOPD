"""Milestone-4 acceptance core: assertion 5, upgraded to travel THROUGH DISK.

Assertion 5 (in-memory, milestone 2) proved online and offline produce bit-close
per-token advantage AND loss for the SAME fixed trajectory. Here we prove it
survives real serialization: precompute the teacher logprob, torch.save it, load
it back, rebuild the score_mask from the STORED token_type, and run the identical
opd_train_step. If offline OPD is going to be compared against online, the disk
round-trip must not move the numbers.

Covered:
  - per-token advantage + loss allclose(atol=1e-4) after precompute->disk->readback
  - the stored dtype is recorded in the file (logp_dtype), float32 by default
  - score_mask rebuilt from stored token_type is BIT-equal to the store-time mask
  - teacher logp is identical through disk at scored (action) positions
  - float16 must DECLARE its dtype and its degradation stays bounded (not silent)
"""
from __future__ import annotations

import os
import tempfile

import torch

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.masking import build_score_mask
from opd.step import opd_train_step
from opd.trajectory import Trajectory
from rollout.policy import MockPolicy
from rollout.rollout import generate_trajectory
from teacher.scorer import MockLM, Scorer, TeacherScorer

CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
    "Berlin is the capital of Germany.",
]


def _fixed_trajectory() -> Trajectory:
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=["capital of France", "Eiffel Tower height"], answer="Paris")
    return generate_trajectory("What is the capital of France?", policy, retr, tok,
                               max_turns=4, max_len=4096, top_k=2, seed=0)


def _scorers():
    tok_vocab = 8192
    teacher = TeacherScorer(MockLM(tok_vocab, seed=1234))
    student = Scorer(MockLM(tok_vocab, seed=5678))  # requires_grad=False: parity, not training
    return teacher, student


def test_advantage_and_loss_survive_disk_roundtrip():
    traj = _fixed_trajectory()
    teacher, student = _scorers()

    # ONLINE: teacher scored live, in-memory
    online_teacher_logp = teacher.score_online(traj.input_ids)
    out_online = opd_train_step(traj, online_teacher_logp, student)

    # OFFLINE: precompute -> disk -> readback -> rebuild everything from stored fields
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.pt")
        store_mask = build_score_mask(traj.token_type)  # mask at store time
        teacher.precompute_and_save(traj, path, store_dtype="float32")
        rec = TeacherScorer.load(path)

        assert rec["logp_dtype"] == "float32", "stored dtype must be recorded explicitly"
        assert rec["teacher_logp"].dtype == torch.float32

        traj_off = Trajectory(rec["input_ids"].long(), rec["token_type"].long())
        read_mask = build_score_mask(traj_off.token_type)  # rebuilt from stored token_type
        assert torch.equal(read_mask, store_mask), "score_mask must be bit-equal through disk"

        out_offline = opd_train_step(traj_off, rec["teacher_logp"], student)

    # teacher logp identical through disk at the ONLY positions that matter (scored)
    assert torch.allclose(
        online_teacher_logp[out_online.score_mask],
        out_offline.teacher_logp[out_offline.score_mask],
        atol=1e-6,
    ), "teacher logp drifted through float32 disk round-trip at action positions"

    # the assertion 5 pair, now through real serialization
    assert torch.allclose(out_online.advantage, out_offline.advantage, atol=1e-4)
    assert torch.allclose(out_online.loss, out_offline.loss, atol=1e-4)


def test_float16_declares_dtype_and_bounds_degradation():
    """float16 is allowed ONLY if declared. The file must record logp_dtype so any
    precision loss is a declared choice, not a silent confound; and the loss shift
    from half precision must stay small (bounded), never unbounded."""
    traj = _fixed_trajectory()
    teacher, student = _scorers()

    online_teacher_logp = teacher.score_online(traj.input_ids)
    out_online = opd_train_step(traj, online_teacher_logp, student)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t16.pt")
        teacher.precompute_and_save(traj, path, store_dtype="float16")
        rec = TeacherScorer.load(path)

        assert rec["logp_dtype"] == "float16", "half precision must be DECLARED in the file"
        assert rec["teacher_logp"].dtype == torch.float16

        traj_off = Trajectory(rec["input_ids"].long(), rec["token_type"].long())
        # upcast back to float32 for the shared step (student logp is float32)
        out_offline = opd_train_step(traj_off, rec["teacher_logp"].float(), student)

    # degradation exists but is bounded -- half precision is a declared tradeoff, not free
    assert torch.allclose(out_online.loss, out_offline.loss, atol=1e-2)
