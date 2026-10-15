"""Flat, token-id-level trajectory representation.

A trajectory is ONE contiguous token-id sequence `input_ids[0..N-1]` plus an
equal-length `token_type[0..N-1]`. `token_type` is the SINGLE SOURCE OF TRUTH
for which tokens are prompt / student-action / environment-observation. Every
mask used anywhere (student loss, teacher scoring online, teacher precompute
offline) is derived from this one array -- never recomputed independently.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import torch


class TokenType(enum.IntEnum):
    PROMPT = 0        # system/user + assistant header; not scored
    ACTION = 1        # student-generated tokens; scored (loss + teacher)
    OBSERVATION = 2   # env-injected retrieved docs; masked from loss, but conditioned on


class TerminationReason(str, enum.Enum):
    ANSWERED = "answered"                    # emitted <answer>...</answer>
    TRUNCATED_MAXLEN = "truncated_maxlen"    # hit max_response_length first
    TRUNCATED_MAXTURNS = "truncated_maxturns"  # hit max search turns first


@dataclass
class Trajectory:
    input_ids: torch.Tensor      # LongTensor [N]
    token_type: torch.Tensor     # LongTensor [N] in {TokenType}
    # confound #2 hook: recorded now, judged later. Distinguishes real divergence
    # from mere truncation. Structure reserved even before detection logic exists.
    terminated: Optional[TerminationReason] = None
    num_search_calls: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.input_ids.ndim == 1, "input_ids must be 1-D"
        assert self.input_ids.shape == self.token_type.shape, "type/id length mismatch"
        assert self.input_ids.dtype == torch.long, "input_ids must be long"
        assert self.token_type.dtype == torch.long, "token_type must be long"

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[0])

    def has_observation(self) -> bool:
        return bool((self.token_type == int(TokenType.OBSERVATION)).any())


class TrajectoryBuilder:
    """Accumulates (ids, type) segments and concatenates at the ID level.

    Enforces the point-1 invariant implicitly: observation segments are appended
    into the SAME sequence (never dropped), so any token generated after an
    observation is conditioned on it. Nothing here ever removes observation ids.
    """

    def __init__(self) -> None:
        self._id_segs: list[torch.Tensor] = []
        self._type_segs: list[torch.Tensor] = []

    def add(self, ids: torch.Tensor, ttype: TokenType) -> "TrajectoryBuilder":
        ids = ids.long().reshape(-1)
        self._id_segs.append(ids)
        self._type_segs.append(torch.full((ids.shape[0],), int(ttype), dtype=torch.long))
        return self

    def current_length(self) -> int:
        return sum(int(s.shape[0]) for s in self._id_segs)

    def current_ids(self) -> torch.Tensor:
        """The exact id prefix appended so far. Fed to the policy as the generation
        context so that what the student conditions on is bit-for-bit what later
        scoring sees -- guards the prefix off-by-one, a hazard subtler than
        detok->retok that only real (id-emitting) generation exposes."""
        if not self._id_segs:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(self._id_segs).long()

    def build(self, terminated: Optional[TerminationReason] = None,
              num_search_calls: int = 0, meta: Optional[dict] = None) -> Trajectory:
        input_ids = torch.cat(self._id_segs) if self._id_segs else torch.empty(0, dtype=torch.long)
        token_type = torch.cat(self._type_segs) if self._type_segs else torch.empty(0, dtype=torch.long)
        return Trajectory(
            input_ids=input_ids.long(),
            token_type=token_type.long(),
            terminated=terminated,
            num_search_calls=num_search_calls,
            meta=meta or {},
        )
