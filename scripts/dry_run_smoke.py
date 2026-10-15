"""End-to-end smoke: build one trajectory, run the SINGLE step function in both
online and offline mode, print the parity-relevant quantities. Not a trainer
(that is milestone 4) -- just proves the pipeline wires together on CPU.

    python scripts/dry_run_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.step import opd_train_step
from opd.trajectory import TokenType, Trajectory
from rollout.policy import MockPolicy
from rollout.rollout import generate_trajectory
from teacher.scorer import MockLM, Scorer, TeacherScorer

CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
    "Berlin is the capital of Germany.",
]


def main() -> None:
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=["capital of France", "Eiffel Tower height"], answer="Paris")
    traj = generate_trajectory("What is the capital of France?", policy, retr, tok,
                               max_turns=4, max_len=4096, top_k=2, seed=0)

    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))
    student = Scorer(MockLM(tok.vocab_size, seed=5678))

    counts = {t.name: int((traj.token_type == int(t)).sum()) for t in TokenType}
    print(f"trajectory: N={traj.length}  terminated={traj.terminated.value}  "
          f"searches={traj.num_search_calls}")
    print(f"token-type counts: {counts}")

    # ONLINE
    out_on = opd_train_step(traj, teacher.score_online(traj.input_ids), student)

    # OFFLINE
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.pt")
        teacher.precompute_and_save(traj, p)
        rec = TeacherScorer.load(p)
        traj_off = Trajectory(rec["input_ids"], rec["token_type"])
        out_off = opd_train_step(traj_off, rec["teacher_logp"], student)

    print(f"scored tokens: {int(out_on.score_mask.sum())}")
    print(f"online  loss={out_on.loss.item():.6f}")
    print(f"offline loss={out_off.loss.item():.6f}")
    print(f"loss parity (allclose): {torch.allclose(out_on.loss, out_off.loss, atol=1e-4)}")
    print(f"advantage parity: {torch.allclose(out_on.advantage, out_off.advantage, atol=1e-4)}")


if __name__ == "__main__":
    main()
