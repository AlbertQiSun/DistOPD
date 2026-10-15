"""`skipif not cuda.is_available()` cannot tell "no GPU" from "GPU that torch cannot drive".

The first is a laptop and a legitimate skip. The second is a broken node -- and on <node>
(driver CUDA 12.2, too old for this torch) the slow suite reported `1 skipped` for exactly
the gate we had queued for. A gate that goes quiet when the environment breaks is the same
failure shape as a gate that goes quiet when the code path moves: it keeps reporting green
while covering nothing.

So: skip when there is no GPU, FAIL when there is one and torch cannot use it.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.gpu_preflight import smi          # one copy of the nvidia-smi parsing


def require_gpu() -> None:
    """Call at the top of a test that needs CUDA."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch is not installed")

    if torch.version.cuda is None:            # a CPU-only build, not a broken node
        pytest.skip("torch is a CPU-only build")
    try:
        if torch.cuda.is_available():
            return
        why = "torch.cuda.is_available() is False"
    except RuntimeError as e:                   # the too-old-driver path raises here
        why = str(e).strip().splitlines()[0]

    gpus = smi()
    if not gpus:
        pytest.skip("no GPU on this machine")
    pytest.fail(
        f"this node HAS a GPU ({gpus[0][0]}, driver {gpus[0][1]}) but torch cannot use it "
        f"-- {why}. Skipping here would report green for a gate that never ran. Driver "
        "version is per-NODE: re-submit on another node of the same gpu type. "
        "(scripts/gpu_preflight.py reports the same thing before a job starts.)")
