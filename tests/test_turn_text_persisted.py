"""The raw action text of every turn must survive to disk.

WHY THIS FILE EXISTS. `empty_query` is the largest drop reason on both new teachers
(Qwen3-32B 28%, Qwen3-30B-A3B-Thinking 50% on the 64-question smokes, 2026-08-21), and
until now it was UNREADABLE. rollout routes EVERY non-answer action into the search
branch, and `_extract_between` returns "" when the tags are missing, so one reason string
covers three failures that need opposite fixes:

  * a literal <search></search>            -> the model made a broken tool call
  * an <answer> the model never closed     -> a length / termination problem
  * prose with no tool call at all         -> the SYSTEM_PROMPT format does not fit

Only the turn's raw text distinguishes them, and recovering it after the fact costs a full
regeneration of the shard. So these are PRODUCER tests, in the sense the project conventions means:
delete `"text": action_text` in rollout.rollout, or `"text"` from datagen's turns_summary,
and something here goes red. A test that hand-built a turn dict and fed it to a reader
would stay green while the field silently stopped being written.
"""
from __future__ import annotations

import json
import os

import pytest
import torch

import configs.template as T
from envs.retriever import MockRetriever
from envs.tokenizer import MockTokenizer
from opd.trajectory import TerminationReason
from rollout.policy import ActionStep, MockPolicy, act_batch_via_act
from rollout.rollout import generate_trajectory
from sft.datagen import SFTGenConfig, classify_trajectory, generate_sft_dataset
from teacher.backend import BackendConfig

CORPUS = ["Paris is the capital of France.", "Berlin is the capital of Germany."]
MOCK_BACKEND = BackendConfig(engine="mock", dtype="mock", tokenizer_name="mock")


class ScriptedPolicy:
    """Emits verbatim texts, one per turn. Unlike MockPolicy it is not constrained to
    well-formed <search>/<answer> actions -- which is the whole point: the malformed
    action is the object under test."""

    def __init__(self, tokenizer, texts, forced_text=None):
        self.tok, self.texts = tokenizer, list(texts)
        self.forced_text = forced_text if forced_text is not None else f"unknown{T.ANSWER_CLOSE}"

    def act(self, prefix_ids, turn, forced=False):  # noqa: ARG002
        text = self.forced_text if forced else self.texts[min(turn, len(self.texts) - 1)]
        return ActionStep(ids=self.tok.encode_tensor(text).long(), text=text)

    def act_batch(self, requests):
        return act_batch_via_act(self, requests)

    def sync_from(self, student):  # noqa: ARG002
        return


def _run(policy, tok, **kw):
    return generate_trajectory("q?", policy, MockRetriever(CORPUS, seed=0), tok,
                               top_k=1, max_turns=4, max_len=4096, **kw)


def test_rollout_records_the_raw_action_text_of_a_normal_search_turn():
    tok = MockTokenizer(vocab_size=8192)
    traj = _run(MockPolicy(tok, queries=["capital of France"], answer="Paris"), tok)
    turns = traj.meta["turns"]
    assert len(turns) == 1
    # not just present -- it must be the WHOLE action, think channel included, because the
    # think block is where a format mismatch is visible.
    assert turns[0]["text"] == (f"{T.THINK_OPEN}I should look this up.{T.THINK_CLOSE}"
                                f"{T.SEARCH_OPEN}capital of France{T.SEARCH_CLOSE}")


def test_a_duplicate_turn_also_records_its_text():
    # Guard 2 short-circuits before retrieval and appends its own turn dict; that second
    # append is a separate line of code and can lose the field on its own.
    tok = MockTokenizer(vocab_size=8192)
    traj = _run(MockPolicy(tok, queries=["same", "same"], answer="Paris"), tok)
    turns = traj.meta["turns"]
    assert turns[1].get("duplicate") is True
    assert T.SEARCH_OPEN + "same" + T.SEARCH_CLOSE in turns[1]["text"]


def test_the_three_empty_query_causes_are_distinguishable_from_the_recorded_text():
    """The payoff. All three drop to reason == "empty_query"; only `text` separates them."""
    tok = MockTokenizer(vocab_size=8192)
    cases = {
        "broken tool call": f"{T.THINK_OPEN}hmm{T.THINK_CLOSE}{T.SEARCH_OPEN}{T.SEARCH_CLOSE}",
        "unclosed answer": f"{T.THINK_OPEN}hmm{T.THINK_CLOSE}{T.ANSWER_OPEN}Paris",
        "no tool call": "I think I already know this one, it is Paris.",
    }
    seen = {}
    for name, text in cases.items():
        traj = _run(ScriptedPolicy(tok, [text]), tok)
        v = classify_trajectory(traj)
        assert not v.keep and v.reason == "empty_query", (name, v)
        turns = traj.meta["turns"]
        assert turns[0]["query"] == ""          # the reason string cannot tell them apart ...
        seen[name] = turns[0]["text"]           # ... the recorded text can.
    assert seen["broken tool call"].endswith(T.SEARCH_OPEN + T.SEARCH_CLOSE)
    assert T.ANSWER_OPEN in seen["unclosed answer"] and T.ANSWER_CLOSE not in seen["unclosed answer"]
    assert T.SEARCH_OPEN not in seen["no tool call"] and T.ANSWER_OPEN not in seen["no tool call"]
    assert len(set(seen.values())) == 3


def test_manifest_persists_turn_text_for_dropped_rows(tmp_path):
    """generate_sft_dataset is the writer nothing else covered. The DROPPED row is the one
    that matters: it has no *.pt record, so the manifest is the only place its evidence can
    live, and an empty_query row is dropped by definition."""
    tok = MockTokenizer(vocab_size=8192)
    policy = ScriptedPolicy(tok, ["I already know this: Paris."])
    out = str(tmp_path / "gen")
    stats = generate_sft_dataset(
        questions=["capital of France?"], golds=["Paris"], policy=policy,
        retriever=MockRetriever(CORPUS, seed=0), tokenizer=tok,
        cfg=SFTGenConfig(max_turns=4, max_len=4096, top_k=1),
        gen_backend=MOCK_BACKEND, teacher_backend=MOCK_BACKEND, out_dir=out,
        tags=["1hop"])
    assert stats.dropped == {"empty_query": 1}
    rows = [json.loads(l) for l in open(os.path.join(out, "manifest.jsonl"), encoding="utf-8")]
    assert len(rows) == 1 and rows[0]["kept"] is False
    ts = rows[0]["turns_summary"]
    assert ts and ts[0]["q"] == ""
    assert ts[0]["text"] == "I already know this: Paris.", \
        "the manifest must carry the raw action text of a dropped turn -- without it an " \
        "empty_query drop can only be diagnosed by regenerating the shard"
    # and the DROPPED row must say how the trajectory ended. This one hit the forced seam and
    # answered there, so it is a recoverable trajectory with one malformed turn -- a fact that
    # is invisible if `terminated` is written only on the kept branch.
    assert rows[0]["terminated"] == TerminationReason.ANSWERED.value


def test_every_turn_records_the_exact_number_of_tokens_sampled():
    """PRODUCER test for n_tok. It must be len(sampled ids), NOT a re-encode of the decoded
    text: distinguishing an action that STOPPED at </search> from one that exhausted
    max_action_tokens is an exact-equality question against the cap, and a re-encode that is
    off by one turns a truncated action into a clean one. Measured 2026-08-21, 100% of
    Qwen3-30B-A3B-Thinking's empty_query drops sat exactly at 512 -- that finding is only
    readable because this number is exact."""
    tok = MockTokenizer(vocab_size=8192)
    policy = MockPolicy(tok, queries=["capital of France", "b"], answer="Paris")
    traj = _run(policy, tok)
    for t in traj.meta["turns"]:
        assert t["n_tok"] == len(tok.encode(t["text"])),             "n_tok must equal the token count of what was actually sampled"
        assert t["n_tok"] > 0
    assert traj.meta["final_action_n_tok"] == len(tok.encode(traj.meta["final_action_text"]))


def test_the_manifest_carries_the_token_counts_too(tmp_path):
    """The distribution is read from the manifest, and a dropped row -- which has no *.pt --
    is exactly the row whose length says whether the cap caused the drop."""
    tok = MockTokenizer(vocab_size=8192)
    out = str(tmp_path / "gen")
    generate_sft_dataset(
        questions=["capital of France?"], golds=["Paris"],
        policy=ScriptedPolicy(tok, ["I already know this: Paris."]),
        retriever=MockRetriever(CORPUS, seed=0), tokenizer=tok,
        cfg=SFTGenConfig(max_turns=4, max_len=4096, top_k=1),
        gen_backend=MOCK_BACKEND, teacher_backend=MOCK_BACKEND, out_dir=out)
    row = json.loads(open(os.path.join(out, "manifest.jsonl"), encoding="utf-8").readline())
    assert row["kept"] is False and row["reason"] == "empty_query"
    ts = row["turns_summary"][0]
    assert ts["n_tok"] == len(tok.encode(ts["text"]))
    assert row["final_action_n_tok"] is not None


def test_the_eval_writer_keeps_everything_the_rollout_computed(tmp_path):
    """PRODUCER-side audit of scripts/stage3_eval's per-sample writer.

    It kept the query STRING from each turn and dropped the rest of traj.meta, so the
    ACTION-span composition -- and therefore the decisive-token share the dilution analysis
    is built on -- could not be computed from an eval report at all. Same shape as every
    other persistence incident in the project conventions, and it had already cost one full GPU re-run
    (cover_em). This test names the fields so a future edit that drops one goes red.
    """
    import inspect

    import scripts.stage3_eval as S
    src = inspect.getsource(S)
    # the block that builds one sample row
    block = src[src.index("samples.append({"):src.index("return {\n        \"question\"")]
    for field in ("turns_detail", "final_action_text", "final_action_n_tok", "has_think",
                  "n_duplicate_queries", "num_search_calls", "n_tokens",
                  "forced_text", "prediction"):
        assert f'"{field}"' in block, f"the eval writer no longer persists {field}"
    # ... and turns_detail must carry the RAW text and the EXACT token count, not just the
    # query -- those two are what the composition and the cap check need.
    detail = block[block.index('"turns_detail"'):]
    for field in ("text", "n_tok", "docs", "dup"):
        assert f'"{field}"' in detail, f"turns_detail dropped {field}"


def test_a_precompute_dir_belonging_to_another_teacher_is_refused(tmp_path):
    """Three lines running at once all default to one shared 'precomputed/'. The second
    overwrites the first's traj_*.pt AND its manifest, so gate 1 -- which compares a run to
    the manifest it finds -- sees a cache perfectly consistent with whoever clobbered it. The
    confound is a WRITE collision; no fingerprint can catch it, so it is caught before the
    write."""
    from scripts.train import _resolve_precompute_dir

    d = tmp_path / "precomputed"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"backend": {"tokenizer_name": "/models/MiroThinker-1.7-mini"}}),
        encoding="utf-8")
    cfg = {"model": {"teacher": "/models/Qwen3-32B"}}
    with pytest.raises(SystemExit, match="was written for teacher"):
        _resolve_precompute_dir(str(d), cfg, dry_run=False)
    # the SAME teacher is fine -- resuming or extending its own cache is normal
    same = {"model": {"teacher": "/models/MiroThinker-1.7-mini"}}
    assert _resolve_precompute_dir(str(d), same, dry_run=False) == str(d)
    # an empty directory is fine, and dry_run never checks
    assert _resolve_precompute_dir(str(tmp_path / "fresh"), cfg, dry_run=False) is not None
    assert _resolve_precompute_dir(str(d), cfg, dry_run=True) == str(d)
