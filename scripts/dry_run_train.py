"""End-to-end dry-run of the unified trainer, both modes, on CPU/mock, WITH the
confound diagnostics attached -- emits runs/diag_{mode}.jsonl so the collection
code + schema are exercised and plot_diagnostics.py has real input.

Runs N steps online and N steps offline through the SAME OPDTrainer. The mock is
deterministic, so the curves will be flat: this validates that logging does not
crash and the records are well-formed -- it CANNOT and is not meant to show
poisoning (that needs real pi_theta drift + real retriever on HPC).

    python scripts/dry_run_train.py [--out-dir runs]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.diag_logger import DiagnosticLogger
from opd.diagnostics import DiagnosticRunner
from opd.trainer import OPDConfig, OPDTrainer, build_source, snapshot_ref
from rollout.policy import MockPolicy
from teacher.scorer import MockLM, Scorer, TeacherScorer

CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
    "Berlin is the capital of Germany.",
]
QUESTIONS = ["What is the capital of France?", "How tall is the Eiffel Tower?"]
GOLDS = ["Paris", "330 metres"]


def _build():
    tok = MockTokenizer(vocab_size=8192)
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(tok, queries=["capital of France", "Eiffel Tower height"], answer="Paris")
    teacher = TeacherScorer(MockLM(tok.vocab_size, seed=1234))
    student = Scorer(MockLM(tok.vocab_size, seed=5678), requires_grad=True)
    return tok, retr, policy, teacher, student


def _run(mode: str, out_dir: str, precompute_dir=None) -> None:
    tok, retr, policy, teacher, student = _build()
    cfg = OPDConfig(mode=mode, steps=4, lr=1e-2, precompute_dir=precompute_dir)
    source = build_source(cfg, QUESTIONS, policy, retr, tok, teacher, golds=GOLDS)

    ref = snapshot_ref(student)                          # pi_ref = student at step 0
    diagnostics = DiagnosticRunner(ref_scorer=ref, tau=cfg.tau, embedder=None)
    log_path = os.path.join(out_dir, f"diag_{mode}.jsonl")
    print(f"[{mode}] steps={cfg.steps} lr={cfg.lr} tau={cfg.tau} -> {log_path}")
    with DiagnosticLogger(log_path) as logger:
        trainer = OPDTrainer(student, source, cfg, diagnostics=diagnostics, logger=logger)
        for i, out in enumerate(trainer.train()):
            print(f"  step {i}: loss={out.loss.item():+.6f}  scored={int(out.score_mask.sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    _run("online", args.out_dir)
    with tempfile.TemporaryDirectory() as d:
        _run("offline", args.out_dir, precompute_dir=d)
    print(f"\ndiagnostics written to {args.out_dir}/diag_online.jsonl and diag_offline.jsonl")
    print("plot with: python scripts/plot_diagnostics.py --run-dir", args.out_dir)


if __name__ == "__main__":
    main()
