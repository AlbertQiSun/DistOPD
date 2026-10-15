"""Gate 2: multi-turn alignment on the REAL Qwen3 tokenizer.

Proves the "no detok->retok" rule actually holds under Qwen3's real merge
behavior, which MockTokenizer cannot exercise. If the tokenizer can't be
downloaded (offline), the test skips -- but where HF is reachable it MUST pass.
"""
from __future__ import annotations

import os

import pytest

transformers = pytest.importorskip("transformers")

import torch  # noqa: E402

from configs import template as T  # noqa: E402
from envs.hf_tokenizer import HFTokenizerAdapter  # noqa: E402
from envs.retriever import MockRetriever  # noqa: E402
from opd.masking import build_score_mask, scored_indices  # noqa: E402
from opd.trajectory import TokenType  # noqa: E402
from rollout.policy import MockPolicy  # noqa: E402
from rollout.rollout import generate_trajectory  # noqa: E402

# Offline HPC: export OPD_TEST_MODEL=/path/to/local/Qwen3-1.7B (default is the HF id).
MODEL = os.environ.get("OPD_TEST_MODEL", "Qwen/Qwen3-1.7B")
CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
]


@pytest.fixture(scope="module")
def qwen():
    try:
        hf = transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # offline / rate-limited
        pytest.skip(f"cannot load {MODEL} tokenizer: {e}")
    return HFTokenizerAdapter(hf)


def test_boundary_merge_hazard_is_real(qwen):
    """The </search> -> \\n\\n<information> boundary merges differently when joined
    vs encoded independently. This is exactly why we must never retokenize."""
    a = f"{T.THINK_OPEN}find it{T.THINK_CLOSE}{T.SEARCH_OPEN}capital of France{T.SEARCH_CLOSE}"
    b = f"{T.OBS_PREFIX}{T.INFO_OPEN}Paris is the capital.{T.INFO_CLOSE}{T.OBS_SUFFIX}"
    assert qwen.encode(a + b) != qwen.encode(a) + qwen.encode(b)


def _traj(qwen):
    retr = MockRetriever(CORPUS, seed=0)
    policy = MockPolicy(qwen, queries=["capital of France", "Eiffel Tower height"], answer="Paris")
    return generate_trajectory("What is the capital of France?", policy, retr, qwen,
                               max_turns=6, max_len=8192, top_k=1, seed=0)


def test_multiturn_trajectory_shape(qwen):
    traj = _traj(qwen)
    assert traj.num_search_calls >= 2          # at least two search turns
    assert traj.has_observation()
    # token_type spans correspond exactly to independently-encoded segments =>
    # scored positions are precisely the ACTION tokens.
    sm = build_score_mask(traj.token_type)
    idx = scored_indices(sm)
    assert int(sm[0].item()) == 0              # position 0 never scored
    assert bool((idx >= 1).all())              # no off-by-one into position 0
    assert bool((traj.token_type[idx] == int(TokenType.ACTION)).all())


class _AdversarialRetriever:
    """Returns one long real-Wikipedia-like passage: special chars, unicode, quotes,
    em-dashes, angle brackets -- the content MockRetriever never produces but the
    live Search-R1 retriever will."""

    def __init__(self, passage: str) -> None:
        self.passage = passage

    def retrieve(self, query: str, k: int = 1, step: int = 0):  # noqa: ARG002
        return [self.passage]


_ADV_PASSAGE = (
    "The Eiffel Tower (French: Tour Eiffel) is a wrought-iron lattice tower on the "
    "Champ de Mars in Paris. It is 330 m (1,083 ft) tall -- about the same as an "
    "81-storey building. Punctuation & symbols: <<angle>>, \"quotes\", em--dashes, "
    "and accents like e/u/o appear throughout. Numbers: 7,300 tonnes; 1889. " * 12
)


def _is_contiguous_sublist(whole: list[int], part: list[int]) -> bool:
    if not part:
        return False
    for i in range(len(whole) - len(part) + 1):
        if whole[i:i + len(part)] == part:
            return True
    return False


def test_adversarial_observation_keeps_boundary_clean(qwen):
    """Seam 2: with real long/special-char retrieved content injected, the
    ACTION/OBSERVATION boundary still tokenizes cleanly. The observation span,
    encoded INDEPENDENTLY, must appear verbatim and contiguous inside the frozen
    ids -- i.e. real doc content did not merge across a span boundary, and scored
    positions remain exactly the ACTION tokens."""
    retr = _AdversarialRetriever(_ADV_PASSAGE)
    policy = MockPolicy(qwen, queries=["capital of France"], answer="Paris")
    traj = generate_trajectory("What is the capital of France?", policy, retr, qwen,
                               max_turns=4, max_len=8192, top_k=1)
    assert traj.has_observation()

    sm = build_score_mask(traj.token_type)
    idx = scored_indices(sm)
    assert int(sm[0].item()) == 0
    assert bool((traj.token_type[idx] == int(TokenType.ACTION)).all())

    obs_text = T.OBS_PREFIX + T.INFO_OPEN + _ADV_PASSAGE + T.INFO_CLOSE + T.OBS_SUFFIX
    obs_ids = qwen.encode(obs_text)
    assert _is_contiguous_sublist(traj.input_ids.tolist(), obs_ids)


def test_frozen_ids_diverge_from_retokenization(qwen):
    """Regression guard: decoding the frozen assistant stream and re-encoding it
    yields a DIFFERENT id sequence (because of the boundary merge). Our pipeline
    keeps the frozen concat ids; any future code that retokenizes would corrupt
    the ACTION/OBSERVATION boundaries and flip this test."""
    traj = _traj(qwen)
    assistant_mask = traj.token_type != int(TokenType.PROMPT)
    assistant_ids = traj.input_ids[assistant_mask].tolist()
    retok = qwen.encode(qwen.decode(torch.tensor(assistant_ids)))
    assert retok != assistant_ids
