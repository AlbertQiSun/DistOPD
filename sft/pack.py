"""M-SFT Stage 2: package the Stage-1 action-only-masked records for the SFT trainer
(LlamaFactory) AND gate that the action-only mask SURVIVES the framework round-trip.

THE SEAM THIS GUARDS (why Stage 2 is a gate, not glue)
------------------------------------------------------
Stage 1 saved, per trajectory, `input_ids` + `labels` where labels[t] == input_ids[t] on
ACTION tokens and -100 everywhere else (PROMPT + OBSERVATION). The load-bearing invariant
is: OBSERVATION (retrieved-doc) tokens are conditioned on but NEVER supervised -- pi_ref
must learn to ACT, not to reproduce retrieved documents.

An SFT framework can silently BREAK that in two ways:
  (1) re-TOKENIZE from text -> different ids -> the stored per-token mask no longer aligns
      (detok->retok drift, the same hazard confound #1 fights);
  (2) re-COMPUTE the label mask from CHAT ROLES -> "mask the user turn, supervise the whole
      assistant turn" -> because observations are inlined into the assistant turn, the
      retrieved-doc tokens get SUPERVISED. pi_ref is then trained on the wrong tokens.
Either way the failure is SILENT: training runs, loss goes down, and pi_ref is subtly wrong.

So we do NOT hand the framework text and trust its role masking. We hand it PRE-TOKENIZED
input_ids + labels (this module's packed form) and then run check_mask_roundtrip against
what the framework will ACTUALLY train on, asserting bit-for-bit that:
  - input_ids are unchanged (no retok drift), and
  - the supervised set (labels != -100) equals EXACTLY the ACTION tokens
    (== build_score_mask(token_type)), so every OBSERVATION token is still -100.
This is the SFT analogue of test_through_disk_parity (which does the same for teacher logp)
and the same framework-seam discipline as gate 1 (assert_same_backend) and gate 3.

    packed = pack_records_from_manifest(stage1_out_dir)   # -> pre-tokenized rows (manifest-scoped)
    write_packed_jsonl(packed, packed.jsonl)              # feed THIS to LlamaFactory
    # after the framework tokenizes/collates, read back (input_ids, labels) and:
    assert_mask_roundtrip(orig_token_type, orig_input_ids, rt_input_ids, rt_labels)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from opd.masking import build_score_mask
from opd.trajectory import TokenType


# ---------------------------------------------------------------------------
# packing: Stage-1 .pt records -> pre-tokenized rows the framework ingests as-is
# ---------------------------------------------------------------------------
def pack_record(rec: dict, idx: int) -> dict:
    """One Stage-1 record -> a JSON-serializable, PRE-TOKENIZED training row.

    We emit input_ids + labels DIRECTLY (never text), so a faithful framework does not
    re-tokenize and does not re-mask. `_token_type` is carried for the round-trip gate ONLY
    (it is not a training field); a trainer ignores unknown keys or we strip it before feeding.
    """
    ids = rec["input_ids"].long().tolist()
    labels = rec["labels"].long().tolist()
    attn = rec["attention_mask"].long().tolist()
    tt = rec["token_type"].long().tolist()
    assert len(ids) == len(labels) == len(attn) == len(tt), "packed field length mismatch"
    return {
        "_id": idx,
        "input_ids": ids,
        "labels": labels,
        "attention_mask": attn,
        "_token_type": tt,     # gate-only provenance; NOT a training input
    }


def _manifest_kept_paths(stage1_out_dir: str) -> list[str]:
    """The .pt paths THIS run's manifest marks kept, in manifest (ascending-i) order.

    PROVENANCE, NOT PRESENCE. manifest.jsonl (rewritten fresh each generation run) is the
    AUTHORITATIVE record of what this run produced. We derive the kept set from it instead of
    glob'ing records/, because glob asks "what files are on disk" -- which silently includes a
    stale .pt left by a PRIOR run at an index this run structurally dropped (datagen's drop path
    does `continue` without deleting the old file). Reading the manifest asks "what did THIS run
    keep", so such a leftover is structurally unable to enter export/verify. The guarantee holds
    even if nobody `rm`'d records/ first.

    Sorted-glob order was ascending filename == ascending i; the manifest is appended in i order,
    so the kept rows are already in that same order -- byte-identical export ordering.
    """
    manifest_path = os.path.join(stage1_out_dir, "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"no manifest.jsonl under {stage1_out_dir} -- cannot establish which records THIS run "
            "kept (provenance guard). Re-run gen_sft_data.py, which writes the manifest.")
    records_dir = os.path.join(stage1_out_dir, "records")
    paths = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("kept"):
                continue
            # canonical filename from the manifest index (robust to shard-dir relocation).
            p = os.path.join(records_dir, f"sft_{int(row['i']):04d}.pt")
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"manifest marks i={row['i']} kept but {p} is missing -- the run's records and "
                    "its manifest disagree; do NOT export a partial shard, regenerate.")
            paths.append(p)
    return paths


def count_manifest_kept(stage1_out_dir: str) -> int:
    """Independent count of kept=True manifest rows -- the export's Layer-3 count backstop."""
    manifest_path = os.path.join(stage1_out_dir, "manifest.jsonl")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip() and json.loads(line).get("kept"))


def load_stage1_records_from_manifest(stage1_out_dir: str) -> list[dict]:
    """Raw Stage-1 .pt records for the manifest-kept set, in export order (used by the verifier)."""
    return [torch.load(p, weights_only=False) for p in _manifest_kept_paths(stage1_out_dir)]


def pack_records_from_manifest(stage1_out_dir: str) -> list[dict]:
    """Pack ONLY the manifest-kept records. Provenance-safe replacement for a glob over records/:
    a stale .pt from a prior/aborted run cannot enter the shard."""
    return [pack_record(rec, i)
            for i, rec in enumerate(load_stage1_records_from_manifest(stage1_out_dir))]


def write_packed_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_packed_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# the gate: assert the action-only mask survived the framework round-trip
# ---------------------------------------------------------------------------
@dataclass
class MaskRoundTripResult:
    ok: bool
    n_tokens: int
    n_action: int             # supervised tokens expected (from stored token_type)
    n_supervised_rt: int      # supervised tokens the framework actually produced
    n_observation: int        # observation tokens (must ALL be -100 after round-trip)
    n_observation_supervised: int   # observation tokens the framework WRONGLY supervised (must be 0)
    ids_drifted: int          # positions where round-tripped input_ids != stored (must be 0)
    reason: str = "ok"


def check_mask_roundtrip(orig_token_type: torch.Tensor, orig_input_ids: torch.Tensor,
                         rt_input_ids: torch.Tensor, rt_labels: torch.Tensor
                         ) -> MaskRoundTripResult:
    """Compare Stage-1 stored (token_type, input_ids) against what the framework will train
    on (rt_input_ids, rt_labels). NON-RAISING -- returns a structured result; use
    assert_mask_roundtrip to enforce.

    Checks, in order of what breaks pi_ref most severely:
      1. length preserved;
      2. input_ids bit-equal (no detok->retok drift);
      3. supervised set (rt_labels != -100) == ACTION tokens (build_score_mask), so
         OBSERVATION tokens are ALL still masked -- the load-bearing invariant;
      4. supervised labels carry the correct action token ids.
    """
    tt = orig_token_type.long()
    n = int(tt.shape[0])
    action_mask = build_score_mask(tt)                       # expected supervised set
    obs_mask = tt == int(TokenType.OBSERVATION)

    rt_ids = rt_input_ids.long()
    rt_lab = rt_labels.long()

    # 1. length
    if not (rt_ids.shape[0] == n and rt_lab.shape[0] == n):
        return MaskRoundTripResult(
            ok=False, n_tokens=n, n_action=int(action_mask.sum()), n_supervised_rt=-1,
            n_observation=int(obs_mask.sum()), n_observation_supervised=-1, ids_drifted=-1,
            reason=f"length changed: stored {n} vs round-trip ids {int(rt_ids.shape[0])} "
                   f"labels {int(rt_lab.shape[0])} (framework re-chunked/re-tokenized)")

    # 2. input_ids drift (re-tokenization)
    ids_drifted = int((rt_ids != orig_input_ids.long()).sum())

    # 3. supervised set == ACTION set
    supervised_rt = rt_lab != -100
    n_supervised_rt = int(supervised_rt.sum())
    obs_supervised = int((supervised_rt & obs_mask).sum())
    set_matches = bool(torch.equal(supervised_rt, action_mask))

    # 4. supervised label ids correct (only meaningful if ids didn't drift)
    labels_ok = True
    if ids_drifted == 0 and set_matches:
        labels_ok = bool(torch.equal(rt_lab[action_mask], orig_input_ids.long()[action_mask]))

    ok = (ids_drifted == 0) and set_matches and (obs_supervised == 0) and labels_ok
    reason = "ok"
    if not ok:
        if ids_drifted:
            reason = f"input_ids drifted at {ids_drifted} positions (re-tokenization)"
        elif obs_supervised:
            reason = (f"{obs_supervised} OBSERVATION tokens got SUPERVISED "
                      "(framework re-masked by chat role -- pi_ref would train on retrieved docs)")
        elif not set_matches:
            reason = (f"supervised set != ACTION set (expected {int(action_mask.sum())}, "
                      f"got {n_supervised_rt})")
        elif not labels_ok:
            reason = "supervised label ids != stored action token ids"

    return MaskRoundTripResult(
        ok=ok, n_tokens=n, n_action=int(action_mask.sum()), n_supervised_rt=n_supervised_rt,
        n_observation=int(obs_mask.sum()), n_observation_supervised=obs_supervised,
        ids_drifted=ids_drifted, reason=reason)


def assert_mask_roundtrip(orig_token_type: torch.Tensor, orig_input_ids: torch.Tensor,
                          rt_input_ids: torch.Tensor, rt_labels: torch.Tensor,
                          what: str = "") -> MaskRoundTripResult:
    """Raising wrapper: raise AssertionError if the action-only mask did not survive."""
    res = check_mask_roundtrip(orig_token_type, orig_input_ids, rt_input_ids, rt_labels)
    if not res.ok:
        raise AssertionError(
            f"SFT mask round-trip FAILED {what}: {res.reason}. "
            "The framework did not preserve the action-only mask; pi_ref would be trained on "
            "the wrong tokens. Feed PRE-TOKENIZED input_ids+labels (do not let it re-tokenize "
            "or re-mask by chat role).")
    return res
