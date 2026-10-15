"""M-SFT Stage 2 acceptance: the action-only mask must SURVIVE the Stage-1 -> framework
round-trip, and the gate that checks it must BITE on the two ways a framework breaks it.

Stage 1 stores labels[t] == input_ids[t] on ACTION tokens and -100 on PROMPT+OBSERVATION.
The load-bearing invariant is that OBSERVATION (retrieved-doc) tokens are conditioned on but
NEVER supervised. A framework can break this by (1) re-tokenizing (input_ids drift) or (2)
re-masking by chat role (observation tokens inlined in the assistant turn get supervised).
These tests feed a Stage-1 record through disk (pack -> jsonl -> load) and assert:
  - faithful pretokenized round-trip PASSES and supervises exactly the ACTION tokens;
  - a role-remask round-trip is CAUGHT (obs tokens supervised);
  - a re-tokenization round-trip is CAUGHT (input_ids drift).
"""
from __future__ import annotations

import os
import tempfile

import torch

from opd.masking import build_score_mask
from opd.trajectory import TerminationReason, TokenType, Trajectory
from sft.datagen import build_sft_record
from sft.pack import (assert_mask_roundtrip, check_mask_roundtrip, load_packed_jsonl,
                      pack_record, write_packed_jsonl)

P, A, O = TokenType.PROMPT, TokenType.ACTION, TokenType.OBSERVATION


def _record_with_observation():
    """prompt, action(search), OBSERVATION docs, action(answer): has OBSERVATION tokens that
    MUST stay -100 through the round-trip."""
    types = [P, P, A, A, O, O, O, A, A]
    n = len(types)
    ids = torch.arange(10, 10 + n, dtype=torch.long)
    tt = torch.tensor([int(t) for t in types], dtype=torch.long)
    traj = Trajectory(ids, tt, terminated=TerminationReason.ANSWERED, num_search_calls=1,
                      meta={"question": "q", "final_answer": "Paris",
                            "turns": [{"turn": 0, "query": "capital of France",
                                       "docs": ["Paris ..."]}]})
    return build_sft_record(traj)


def _disk_roundtrip(rec):
    """Pack -> jsonl -> load, returning the pre-tokenized row a faithful framework ingests."""
    row = pack_record(rec, 0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "packed.jsonl")
        write_packed_jsonl([row], path)
        back = load_packed_jsonl(path)[0]
    return back


def test_faithful_pretokenized_roundtrip_preserves_action_only_mask():
    rec = _record_with_observation()
    back = _disk_roundtrip(rec)
    tt = rec["token_type"]
    rt_ids = torch.tensor(back["input_ids"], dtype=torch.long)
    rt_lab = torch.tensor(back["labels"], dtype=torch.long)
    res = assert_mask_roundtrip(tt, rec["input_ids"], rt_ids, rt_lab)  # must NOT raise
    assert res.ok and res.ids_drifted == 0
    # supervised set is EXACTLY the ACTION tokens; every OBSERVATION token stayed masked.
    assert res.n_supervised_rt == int(build_score_mask(tt).sum())
    assert res.n_observation > 0 and res.n_observation_supervised == 0


def test_gate_bites_on_role_remask_supervising_observations():
    # Failure mode (2): framework re-masks by chat role -> supervises the WHOLE assistant turn,
    # including the inlined OBSERVATION docs. The gate must catch supervised observation tokens.
    rec = _record_with_observation()
    tt = rec["token_type"]
    ids = rec["input_ids"]
    # "train on everything that isn't PROMPT" == the naive role rule (obs treated as assistant).
    bad_labels = torch.where(tt != int(TokenType.PROMPT), ids, torch.full_like(ids, -100))
    res = check_mask_roundtrip(tt, ids, ids, bad_labels)
    assert not res.ok
    assert res.n_observation_supervised == res.n_observation > 0
    try:
        assert_mask_roundtrip(tt, ids, ids, bad_labels)
    except AssertionError:
        pass
    else:
        raise AssertionError("gate must raise when observation tokens are supervised")


def test_gate_bites_on_retokenization_drift():
    # Failure mode (1): framework re-tokenized -> input_ids differ from the stored ids.
    rec = _record_with_observation()
    tt = rec["token_type"]
    ids = rec["input_ids"]
    drifted = ids.clone()
    drifted[3] = drifted[3] + 1000                      # one token changed by "re-tokenization"
    res = check_mask_roundtrip(tt, ids, drifted, rec["labels"])
    assert not res.ok and res.ids_drifted >= 1
    try:
        assert_mask_roundtrip(tt, ids, drifted, rec["labels"])
    except AssertionError:
        pass
    else:
        raise AssertionError("gate must raise when input_ids drift from re-tokenization")


def test_pack_record_is_pretokenized_not_text():
    # The row we hand the framework carries input_ids/labels DIRECTLY (no text field), so a
    # faithful framework has nothing to re-tokenize.
    rec = _record_with_observation()
    row = pack_record(rec, 7)
    assert "input_ids" in row and "labels" in row and "attention_mask" in row
    assert "text" not in row and "messages" not in row and "conversations" not in row
    assert row["_id"] == 7
    assert len(row["input_ids"]) == len(row["labels"]) == rec["input_ids"].shape[0]
