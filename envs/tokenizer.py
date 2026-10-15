"""Deterministic word-level tokenizer for CPU dry-run (no model download).

NOT used in real runs -- the HPC path uses the Qwen3 HF tokenizer. But it honors
the same protocol constants from configs.template so tag boundaries behave the
same way structurally. Determinism is via hashlib (NOT Python's salted hash), so
ids are identical across processes and runs -- required for reproducible parity.
"""
from __future__ import annotations

import hashlib
import re

import torch

from configs import template as T


class MockTokenizer:
    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = vocab_size
        # protocol/scaffold tokens get fixed low ids
        self.specials = [
            T.IM_START, T.IM_END,
            T.THINK_OPEN, T.THINK_CLOSE,
            T.SEARCH_OPEN, T.SEARCH_CLOSE,
            T.INFO_OPEN, T.INFO_CLOSE,
            T.ANSWER_OPEN, T.ANSWER_CLOSE,
            "\n",
        ]
        self.special_ids = {tok: i for i, tok in enumerate(self.specials)}
        self._n_special = len(self.specials)
        # match specials first, then whitespace runs, word runs, or single symbols
        self._pat = re.compile(
            "(" + "|".join(re.escape(s) for s in self.specials) + r"|\s+|\w+|[^\w\s])"
        )

    def _tok_id(self, tok: str) -> int:
        if tok in self.special_ids:
            return self.special_ids[tok]
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        return self._n_special + (h % (self.vocab_size - self._n_special))

    def encode(self, text: str) -> list[int]:
        toks = [t for t in self._pat.findall(text) if t != ""]
        return [self._tok_id(t) for t in toks]

    def encode_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)
