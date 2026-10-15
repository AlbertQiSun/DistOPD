"""Gate 1: online and offline teacher backends must be identical."""
from __future__ import annotations

import pytest

from teacher.backend import BackendConfig, assert_same_backend, template_fingerprint


def test_identical_backends_pass():
    a = BackendConfig(engine="vllm", dtype="bfloat16", tokenizer_name="Qwen/Qwen3-4B")
    b = BackendConfig(engine="vllm", dtype="bfloat16", tokenizer_name="Qwen/Qwen3-4B")
    assert_same_backend(a, b)  # must not raise


@pytest.mark.parametrize("override", [
    {"engine": "hf"},                        # different engine
    {"dtype": "float32"},                    # different precision
    {"tokenizer_name": "Qwen/Qwen3-8B"},     # different tokenizer
])
def test_mismatched_backends_raise(override):
    base = dict(engine="vllm", dtype="bfloat16", tokenizer_name="Qwen/Qwen3-4B")
    a = BackendConfig(**base)
    b = BackendConfig(**{**base, **override})
    with pytest.raises(AssertionError):
        assert_same_backend(a, b)


def test_template_fingerprint_stable_and_included():
    assert template_fingerprint() == template_fingerprint()
    assert len(template_fingerprint()) == 16
    # fingerprint is part of the config, so template drift also trips the gate
    cfg = BackendConfig(engine="hf", dtype="float32", tokenizer_name="mock")
    assert cfg.template_fp == template_fingerprint()
