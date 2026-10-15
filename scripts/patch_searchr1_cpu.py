#!/usr/bin/env python3
"""Patch Search-R1's retrieval_server.py to run the E5 encoder on CPU.

WHY: Search-R1's load_model() calls model.cuda() UNCONDITIONALLY, and its Encoder.encode()
moves inputs with .cuda(). On our cheap 4090d retriever node the NVIDIA driver is CUDA 12.4
but the OPD env's torch is built for CUDA 13, so torch._C._cuda_init() raises "driver too
old" and the server dies on startup. The encoder is e5-base-v2 (110M) doing ONE short query
at a time -- CPU is more than fast enough -- and moving the device does NOT change the encode
math (same weights, same mean pooling, same "query:" prefix, same L2-normalize), so the
retrieval space stays identical to how the index was built. This keeps the structural
guarantee intact while removing the GPU-driver dependency entirely (cuda_init is never called).

The patch is three targeted, verified string replacements; it is IDEMPOTENT (re-running on an
already-patched file is a no-op) and FAILS LOUD if the upstream source no longer matches (so a
Search-R1 update can't silently un-patch us).

    python scripts/patch_searchr1_cpu.py /path/to/Search-R1/search_r1/search/retrieval_server.py
"""
from __future__ import annotations

import sys

# (old, new) pairs. `new` must be detectable to make re-runs idempotent.
_REPLACEMENTS = [
    ("model.cuda()", "model.cpu()  # patched: CPU encode (4090d driver != torch CUDA)"),
    ("model = model.half()", "model = model  # patched: keep fp32 on CPU"),
    ("inputs = {k: v.cuda() for k, v in inputs.items()}",
     "inputs = {k: v for k, v in inputs.items()}  # patched: keep inputs on CPU"),
]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_searchr1_cpu.py <retrieval_server.py>")
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        text = f.read()

    changed = False
    for old, new in _REPLACEMENTS:
        if new in text:
            print(f"[patch] already applied: {old!r}")
            continue
        n = text.count(old)
        if n == 0:
            raise SystemExit(
                f"FATAL: neither patched nor original form found for {old!r} in {path}. "
                "Search-R1 source changed -- re-inspect load_model/encode before trusting the retriever.")
        if n > 1:
            raise SystemExit(f"FATAL: {old!r} occurs {n}x (expected 1) in {path}; patch is ambiguous.")
        text = text.replace(old, new)
        changed = True
        print(f"[patch] applied: {old!r} -> CPU")

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[patch] wrote CPU-patched {path}")
    else:
        print(f"[patch] no change needed (already CPU-patched): {path}")


if __name__ == "__main__":
    main()
