"""Training leaves the batch-invariant overrides and provably comes back.

THE MEASUREMENT BEHIND THE DESIGN (bench --pin-batch-invariant, <node>, 2026-08-18): vLLM's
batch-invariant mode registers a direct CUDA kernel for aten::linear, which breaks autograd's
composite decomposition -- the first backward dies asking for `linear_backward`, an op with
no CUDA kernel. Training under the overrides is impossible, so "pin both arms" (the previous
design, one commit old) is dead, and the student trains VANILLA in both arms: offline never
enables the mode; online BRACKETS its training/diagnostic sections out of it.

What these tests hold:
  * the bracket is a NO-OP where the mode was never enabled -- the offline arm must not grow
    a vllm dependency or a different code path;
  * inside the bracket the overrides are gone AND torch.bmm (a Python monkeypatch _destroy
    cannot see) is the C binding again;
  * on exit the mode is re-enabled and VERIFIED -- a half-restored exit hands the next
    rollout a sampling configuration gate 3b never certified;
  * opd_train_step REFUSES to run under live overrides -- the guard that turns "a caller
    forgot the bracket" from a dispatcher riddle at backward into a named failure at entry;
  * the trainer actually wraps its training and diagnostic sections (producer side).

The fakes mirror vllm's real module shape (_batch_invariant_MODE / _batch_invariant_LIB /
enable_batch_invariant_mode / a Library with _destroy) because the bracket manipulates that
shape directly.
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rollout.engine import student_vanilla_math  # noqa: E402


class _FakeLib:
    def __init__(self, log):
        self._log = log

    def _destroy(self):
        self._log.append("destroy")


def _install_fake_bi(monkeypatch, *, mode: bool, reenable_works: bool = True):
    bi = types.ModuleType("vllm.model_executor.layers.batch_invariant")
    bi.log = []
    bi._batch_invariant_MODE = mode
    bi._batch_invariant_LIB = _FakeLib(bi.log) if mode else None

    def enable_batch_invariant_mode():
        bi.log.append("enable")
        if reenable_works:
            bi._batch_invariant_MODE = True
            bi._batch_invariant_LIB = _FakeLib(bi.log)
            torch.bmm = lambda a, b: None          # the monkeypatch enable() really does

    bi.enable_batch_invariant_mode = enable_batch_invariant_mode
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.batch_invariant", bi)
    return bi


@pytest.fixture(autouse=True)
def _restore_torch_bmm():
    orig = torch.bmm
    yield
    torch.bmm = orig


def test_noop_when_the_mode_was_never_enabled(monkeypatch):
    """The offline arm's path. No vllm modules at all."""
    for name in list(sys.modules):
        if name.startswith("vllm"):
            monkeypatch.delitem(sys.modules, name)
    with student_vanilla_math():
        pass                                        # must not raise, must not import vllm
    assert not any(n.startswith("vllm") for n in sys.modules), (
        "the bracket imported vllm into a process that never had it")


def test_noop_when_vllm_present_but_mode_off(monkeypatch):
    bi = _install_fake_bi(monkeypatch, mode=False)
    with student_vanilla_math():
        pass
    assert bi.log == [], "nothing to destroy, nothing to re-enable"


def test_bracket_destroys_then_reenables(monkeypatch):
    bi = _install_fake_bi(monkeypatch, mode=True)
    with student_vanilla_math():
        assert bi._batch_invariant_MODE is False, "overrides still live inside the bracket"
        assert bi.log == ["destroy"]
    assert bi._batch_invariant_MODE is True, "the mode did not come back"
    assert bi.log == ["destroy", "enable"]


def test_torch_bmm_is_the_C_binding_inside_the_bracket(monkeypatch):
    """enable() monkeypatches the PYTHON attribute torch.bmm; Library._destroy cannot see
    it. Inside the bracket a student calling torch.bmm must get vanilla."""
    bi = _install_fake_bi(monkeypatch, mode=True)
    torch.bmm = lambda a, b: None                   # simulate the live monkeypatch
    with student_vanilla_math():
        assert torch.bmm is torch._C._VariableFunctions.bmm
    assert bi._batch_invariant_MODE is True


def test_a_failed_reenable_raises_not_returns(monkeypatch):
    """Half-restored is the worst state: the next rollout samples under a configuration
    gate 3b never certified, silently."""
    _install_fake_bi(monkeypatch, mode=True, reenable_works=False)
    with pytest.raises(RuntimeError, match="never certified"):
        with student_vanilla_math():
            pass


def test_reenable_happens_even_when_the_body_raises(monkeypatch):
    bi = _install_fake_bi(monkeypatch, mode=True)
    with pytest.raises(ValueError):
        with student_vanilla_math():
            raise ValueError("training exploded")
    assert bi._batch_invariant_MODE is True, (
        "an exception in the training body left the overrides down for the next rollout")


def test_opd_train_step_refuses_under_live_overrides(monkeypatch):
    """The guard for a caller that forgot the bracket: fail at entry with the cause named,
    not at backward with 'aten::linear_backward has no CUDA kernel'."""
    from opd.step import opd_train_step
    from opd.trajectory import TokenType, Trajectory

    _install_fake_bi(monkeypatch, mode=True)
    ids = torch.tensor([1, 2, 3, 4])
    tt = torch.tensor([int(TokenType.PROMPT)] + [int(TokenType.ACTION)] * 3)
    with pytest.raises(RuntimeError, match="student_vanilla_math"):
        opd_train_step(Trajectory(ids, tt), torch.zeros(4), object())


# ---------------------------------------------------------------- producer side
def test_the_trainer_brackets_its_training_block():
    src = (ROOT / "opd" / "trainer.py").read_text(encoding="utf-8")
    step = src.split("def step(", 1)[1].split("def _log_diagnostics", 1)[0]
    assert "with student_vanilla_math():" in step, "the training block is not bracketed"
    inside = step.split("with student_vanilla_math():", 1)[1]
    head = inside.split("train_seconds = time.perf_counter()", 1)[0]
    assert "opd_train_step(" in head and ".backward()" in head and "optimizer.step()" in head, (
        "forward, backward and the optimizer must all be inside the bracket")


def test_the_trainer_brackets_the_diagnostics_too():
    """The diagnostics compute the curves the paper compares; run under different kernels in
    the two arms, the online-offline difference carries a kernel term."""
    src = (ROOT / "opd" / "trainer.py").read_text(encoding="utf-8")
    assert "with student_vanilla_math(), _student_device(self.student):" in src


def test_train_py_no_longer_pins_and_equalizes_cublas_workspace():
    src = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "pin_student_math" not in src, (
        "the pin is the design the linear_backward measurement killed; it must not survive "
        "alongside the bracket")
    assert 'os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG"' in src, (
        "the online process inherits a workspace config from vllm and cuBLAS picks "
        "algorithms by it -- the offline arm must pin the same value or student matmuls "
        "differ at the bit level between arms")
