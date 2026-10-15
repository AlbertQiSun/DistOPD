"""The two arms' student must produce bit-identical logprobs -- with and without the engine.

WHERE THIS COMES FROM. The online arm's FIRST-EVER student training step (2026-08-17, any
cluster) crashed inside vllm's bmm override: VLLM_BATCH_INVARIANT=1 registers PROCESS-WIDE
torch dispatcher overrides (mm/addmm/matmul/linear on sm_80; bmm and the softmax family
everywhere, plus a torch.bmm monkeypatch), so the online student trained under vLLM's Triton
kernels while the offline student -- whose process never imports vllm -- trained under
vanilla cuBLAS. Different kernels round differently: the arms differed by matmul rounding ON
TOP OF rollout source, a confound in exactly the quantity the experiment measures.

The resolution (second design -- the first, pin-both-arms, died on measurement: the
overrides are inference-only, aten::linear_backward has no CUDA kernel): the student trains
VANILLA in both arms, and the online arm BRACKETS its training out of the overrides per step
(rollout/engine.student_vanilla_math). This file is the gate that the bracket actually
restores vanilla where it matters -- at the BITS of the student's output:

    process A (offline-shaped): pin_student_math, no engine        -> student logprobs
    process B (online-shaped) : pin_student_math + a LIVE vLLM engine in-process
                                                                    -> student logprobs
    assert bit-identical.

Engine presence is the online arm's distinguishing feature, and engine construction touches
global state (allocator, backend flags, compiled artifacts), so "the engine does not change
the student's numbers" is a claim about process state that only a measurement can make.

Two subprocesses, not two phases of one process: one process cannot host "engine absent"
after hosting "engine present" (the KV pool never returns -- scripts/run_slow_isolated.sh
exists because of it).

KNOWN LIMIT, stated: under certify gate 4 there is ONE visible GPU, so both children run on
cuda:0 and the cross-device Triton-launch crash (student on cuda:1, current device cuda:0)
is NOT reproduced here -- opd/step.py's _student_device guard covers it, and its unit tests
plus the real online run are its evidence.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.slow

_CHILD = r"""
import os, sys, json
sys.path.insert(0, sys.argv[4])
mode, ckpt, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

# EXACTLY the arms' order. Online pins the policy env before importing vllm; both arms
# equalize the cuBLAS workspace (train.py does the same).
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from rollout.engine import pin_policy_env, student_vanilla_math
pin_policy_env()

llm = None
if mode == "with-engine":
    from tests.vllm_engine import production_engine_kwargs
    from vllm import LLM
    llm = LLM(**production_engine_kwargs(model=ckpt, gpu_memory_utilization=0.30,
                                         max_model_len=2048))
    # the engine must be LIVE, not merely constructed: sampling is what exercises its state
    from vllm import SamplingParams
    llm.generate(["ping"], SamplingParams(max_tokens=4, temperature=0.0))

import torch
from teacher.scorer import HFScorer
from opd.step import opd_train_step
from opd.trajectory import TokenType, Trajectory

student = HFScorer.from_pretrained(ckpt, dtype="float32", autocast_dtype="bfloat16",
                                   requires_grad=True, device="cuda:0")
g = torch.Generator().manual_seed(7)
vocab = int(student.lm.config.vocab_size)
ids = torch.randint(1000, vocab - 1000, (768,), generator=g)
tt = torch.full((768,), int(TokenType.PROMPT), dtype=torch.long)
tt[100:400] = int(TokenType.ACTION)
tlp = torch.randn(768, generator=g)

# the trainer's bracket, verbatim: with-engine this leaves the overrides and returns;
# no-engine it is a no-op. THIS is the mechanism under test.
with student_vanilla_math():
    out = opd_train_step(Trajectory(ids, tt), tlp, student)
torch.save({"student_logp": out.student_logp.detach().cpu(),
            "loss": out.loss.detach().cpu()}, out_path)
print("CHILD_OK", mode)
"""


def test_engine_presence_does_not_change_student_bits(tmp_path):
    ckpt = os.environ.get("OPD_STUDENT_CKPT")
    if not ckpt or not os.path.isdir(ckpt):
        pytest.skip("OPD_STUDENT_CKPT unset -- source slurm/env.sh")
    if not torch.cuda.is_available():
        pytest.skip("GPU required")

    outs = {}
    for mode in ("no-engine", "with-engine"):
        out = tmp_path / f"{mode}.pt"
        r = subprocess.run(
            [sys.executable, "-c", _CHILD, mode, ckpt, str(out), str(ROOT)],
            capture_output=True, text=True, cwd=ROOT, timeout=1200,
        )
        assert r.returncode == 0 and "CHILD_OK" in r.stdout, (
            f"{mode} child failed:\n--- stdout\n{r.stdout[-3000:]}\n--- stderr\n{r.stderr[-3000:]}")
        outs[mode] = torch.load(out, weights_only=True)

    a, b = outs["no-engine"], outs["with-engine"]
    same = torch.equal(a["student_logp"], b["student_logp"])
    assert same, (
        "student logprobs differ between the offline-shaped and online-shaped process. The "
        "engine's presence changes the student's numerics, so the arms are NOT comparing "
        f"only rollout source. max|d|={float((a['student_logp'] - b['student_logp']).abs().max()):.3e}")
    assert torch.equal(a["loss"], b["loss"]), "losses differ despite identical logprobs"
