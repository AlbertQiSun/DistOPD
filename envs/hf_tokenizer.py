"""Adapter exposing a HuggingFace tokenizer through the same interface the
rollout expects (encode_tensor + render_prompt_ids), honoring the frozen
template constants. Real runs use this; dry-run uses MockTokenizer.

Prompt is rendered ONCE via apply_chat_template with the frozen flags. All later
segments are encoded independently with add_special_tokens=False and concatenated
at the ID level -- never a re-render of the growing conversation (that would hit
the Qwen3 thinking-strip + boundary-merge hazards).
"""
from __future__ import annotations

import torch

from configs import template as T


class HFTokenizerAdapter:
    def __init__(self, hf_tokenizer) -> None:
        self.tok = hf_tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def encode_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tok.decode(ids)

    def render_prompt_ids(self, question: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": T.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        kwargs = dict(tokenize=True, add_generation_prompt=T.ADD_GENERATION_PROMPT)
        try:
            out = self.tok.apply_chat_template(messages, enable_thinking=T.ENABLE_THINKING, **kwargs)
        except TypeError:
            # older/newer templates may not accept enable_thinking kwarg
            out = self.tok.apply_chat_template(messages, **kwargs)
        return torch.tensor(self._extract_ids(out), dtype=torch.long)

    @staticmethod
    def _extract_ids(out) -> list[int]:
        # transformers 5.x returns a BatchEncoding when tokenize=True
        if hasattr(out, "input_ids"):
            out = out["input_ids"]
        if len(out) > 0 and isinstance(out[0], (list, tuple)):
            out = out[0]
        return list(out)
