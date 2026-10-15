"""M-SFT Stage 1: turn MiroThinker rollouts in the OPD env into an action-only-
supervised SFT dataset for Qwen3-4B-Base (-> pi_ref).

This is the data-generation half of the SFT prerequisite milestone. It reuses the
EXACT OPD rollout (rollout.generate_trajectory), so the SFT trajectories live in the
same env (same retriever, prompt template, tag protocol, max_turns/max_len/top_k) the
OPD run will later use. Nothing here scores teacher logprobs -- SFT supervises the
student on MiroThinker's ACTIONS; the teacher logp path is OPD's job.

TEACHER-CONSISTENCY RED LINE (machine-checked)
----------------------------------------------
    SFT-data generator == OPD teacher == MiroThinker
generate_sft_dataset asserts the generator's model identity equals the OPD teacher's
(assert_same_teacher_model) BEFORE producing a single record. If SFT data were made by
any other model, sigma_Delta > 0 and offline inherits the Theorem-3.11 irreducible bias
-- so this is a hard gate at the top of the function, not a comment.

ZERO QUALITY FILTERING -- WE TAKE THE TEACHER DISTRIBUTION AS-IS (deliberate)
----------------------------------------------------------------------------
SFT's ONLY job is to faithfully imitate pi_T. Any rule that drops a trajectory because
of its QUALITY trains pi_T(tau | quality-is-good) != pi_T(tau) -- which is itself
sigma_Delta, re-introducing the exact bias the teacher-consistency red line exists to
kill. So the drop predicate is constrained to be a function of the trajectory's
STRUCTURE / COMPLETENESS, NEVER its quality:

  we DROP only structurally-broken trajectories:
    - no_action      : no student ACTION span exists -> nothing to supervise
    - empty_query    : a turn produced NO usable query. rollout treats every non-answer
                       action as a search turn, so this covers a literal <search></search>,
                       an <answer> the model never closed, and free prose with no tool call.
                       manifest turns_summary[].text carries the raw action text to tell
                       these apart.
    - empty_answer   : terminated ANSWERED but the <answer></answer> span is empty
    - not_terminated : did NOT reach <answer> even after the forced-final-answer seam
                       (rollout Guard 1) -> no supervised terminal answer span. This is a
                       STRUCTURAL incompleteness, not a quality judgment: the environment
                       forced an answer turn and the model still produced no <answer>. Post-
                       Guard-1 it should be RARE; the stats.dropped["not_terminated"] count is
                       a MONITORING signal -- a high value means the forced-answer turn is
                       failing (or max_len is cutting trajectories before they can answer).

  we KEEP everything that is a well-formed trajectory, INCLUDING low-quality behavior:
    - WRONG final answer            (coherent search, incorrect synthesis -> hard positive;
                                     there is NO EM/gold-match gate anywhere below)
    - no_search (single trajectory) (MiroThinker judged it didn't need to search -> that is
                                     real teacher behavior; a HIGH BATCH RATE is a device
                                     red flag surfaced by scripts/check_sft_stats.py, but a
                                     single one is not dropped)
    - retrieval_empty               (query was well-formed but the retriever returned no docs
                                     -- that is the ENVIRONMENT's output, not a broken
                                     trajectory; the teacher answering under empty retrieval
                                     is behavior pi_ref should see. A high rate is a
                                     retriever-too-weak signal -> report, not a drop.)
    - repeated_query_loop           (repeating a query is "dumb", not "broken" -> keep)
    - high-hop spinning             (many searches that never surfaced gold evidence: this is
                                     genuine teacher behavior on a hard question, kept)
    - best-effort forced answer     (Guard 1 forced the model to answer at budget exhaustion;
                                     the answer may be a low-EM guess -> that is COHERENT
                                     terminal behavior, NOT quality damage. There is no EM gate
                                     -> kept. Only NON-termination (no <answer> at all) drops.)

DIAGNOSTICS ARE COMPUTED BUT NEVER GATE (productivity is report-only)
---------------------------------------------------------------------
We STILL compute the per-hop retrieval productivity signal (opd.diagnostics.
retrieval_quality -> productivity_summary) and write it to every manifest row and every
saved record's meta. It is a DIAGNOSTIC -- read by scripts/check_sft_stats.py to REPORT
MiroThinker's actual behavior distribution (hop dist / productive-vs-spinning ratio /
retrieval_empty rate / no_search rate) so the human can decide "questions too hard ->
change the question mix" vs "retriever too weak -> change index/top_k". It is
PHYSICALLY isolated from the drop path: classify_trajectory does not read it. The error
we removed was never "computing productivity" -- it was USING productivity to filter.

ACTION-ONLY SUPERVISION
-----------------------
labels[t] = input_ids[t] iff token_type[t] == ACTION, else -100 (ignored). This is the
SAME token set as opd.masking.build_score_mask, so SFT supervises exactly the tokens OPD
later scores: the student is taught to ACT (emit queries / think / answer), never to
reproduce retrieved observation docs.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import torch

import configs.template as T
from opd.diagnostics import retrieval_quality
from opd.masking import build_score_mask
from opd.trajectory import TerminationReason, TokenType, Trajectory
from rollout.rollout import generate_trajectories, generate_trajectory
from teacher.backend import BackendConfig, assert_same_teacher_model, template_fingerprint


@dataclass
class SFTGenConfig:
    """Rollout knobs for SFT data generation. Mirror the OPD run's env so the SFT
    trajectory distribution matches what OPD will see. (No quality-filter knob lives here
    by design -- SFT takes the teacher distribution as-is; see module docstring.)"""
    max_turns: int = 4
    max_len: int = 4096
    top_k: int = 3
    seed: int = 0


@dataclass
class Verdict:
    keep: bool
    reason: str = "kept"


def productivity_summary(traj: Trajectory, rq: dict = None) -> dict:
    """Per-hop retrieval productivity of a trajectory (retrieval axis, not correctness).

    DIAGNOSTIC ONLY -- read by the check script to report MiroThinker's behavior; NEVER
    consulted by classify_trajectory. productive := the search surfaced the gold EVIDENCE
    in retrieved docs at some hop (retrieval_quality.union_hit). productive_multihop :=
    productive AND hops>=2. When there is no gold or no search turn, union_hit is None ->
    productive/productive_multihop are None (unknown), never forced True/False.
    """
    if rq is None:
        rq = retrieval_quality(traj.meta)
    hops = int(traj.num_search_calls)
    union = rq.get("union_hit")
    productive = None if union is None else bool(union)
    return {
        "hops": hops,
        "union_hit": union,
        "per_turn_hit_rate": rq.get("per_turn_hit_rate"),
        "first_hit_turn": rq.get("first_hit_turn"),
        "productive": productive,
        "productive_multihop": None if productive is None else (productive and hops >= 2),
    }


def _all_retrieval_empty(traj: Trajectory) -> bool:
    """True iff the trajectory searched at least once but EVERY search turn came back with
    no docs. Report-only signal (retriever-too-weak); NOT a drop reason -- answering under
    empty retrieval is real teacher behavior."""
    turns = traj.meta.get("turns", []) or []
    return bool(turns) and all(len(t.get("docs", []) or []) == 0 for t in turns)


def classify_trajectory(traj: Trajectory) -> Verdict:
    """Drop ONLY structurally-broken trajectories.

    THE RULE (sharp): the drop predicate is a function of the trajectory's STRUCTURE /
    COMPLETENESS, never its QUALITY. Anything that would need to read "how good the search
    was" to decide is quality-filtering == sigma_Delta and is refused here (see module
    docstring). So there is NO EM/correctness gate, NO no_search gate, NO retrieval_empty
    gate, NO repeated-query gate, NO productivity/spinning gate. Only three structural
    defects drop a trajectory.
    """
    tt = traj.token_type
    # (0) no student ACTION span -- literally nothing to supervise. Structural.
    if not bool((tt == int(TokenType.ACTION)).any()):
        return Verdict(False, "no_action")

    turns = traj.meta.get("turns", []) or []
    # (1) malformed <search></search>: the teacher emitted an empty query string. This is a
    # generation-layer structural defect (a broken tool call), NOT low-quality search.
    if any(not (t.get("query", "") or "").strip() for t in turns):
        return Verdict(False, "empty_query")

    # (2) malformed answer: terminated ANSWERED but the <answer></answer> span is empty.
    if traj.terminated == TerminationReason.ANSWERED:
        if not (traj.meta.get("final_answer") or "").strip():
            return Verdict(False, "empty_answer")

    # (3) not terminated: no <answer> even after the forced-final-answer seam (rollout Guard 1).
    # Structural incompleteness -- there is no supervised terminal answer span to imitate. This
    # is NOT a quality gate (a low-EM best-effort forced answer terminates ANSWERED and is kept);
    # it only fires when the trajectory has no terminal answer AT ALL. Should be rare post-Guard-1;
    # stats.dropped["not_terminated"] is the monitoring signal (see module docstring).
    if traj.terminated != TerminationReason.ANSWERED:
        return Verdict(False, "not_terminated")

    return Verdict(True, "kept")


def build_sft_record(traj: Trajectory, prod: dict = None) -> dict:
    """One pre-tokenized, action-only-masked SFT example.

    labels are HF/LlamaFactory-native: -100 = ignored, token_id = supervised. The
    supervised set is exactly build_score_mask(token_type) == ACTION tokens, so this
    record teaches the SAME tokens OPD scores. Observation (retrieved-doc) tokens stay
    in input_ids (the model still conditions on them) but carry label -100.

    meta["productivity"] is a DIAGNOSTIC field only (populated for later OPD-run analysis
    of which trajectories pi_ref rolls out well/poorly); it plays NO role in whether this
    record was kept -- classify_trajectory never reads it.
    """
    ids = traj.input_ids.long()
    mask = build_score_mask(traj.token_type)                       # ACTION-only
    labels = torch.where(mask, ids, torch.full_like(ids, -100))
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": torch.ones_like(ids),
        "token_type": traj.token_type.long(),                      # kept for round-trip checks
        "meta": {
            "question": traj.meta.get("question"),
            "num_search_calls": int(traj.num_search_calls),
            "terminated": traj.terminated.value if traj.terminated else None,
            "n_action_tokens": int(mask.sum()),
            # diagnostic-only; never gates (see docstring).
            "productivity": prod if prod is not None else productivity_summary(traj),
        },
    }


@dataclass
class GenStats:
    kept: int = 0
    dropped: dict = field(default_factory=dict)   # reason -> count (structural drops only)
    hops: dict = field(default_factory=dict)      # num_search_calls -> count (kept, difficulty dist)
    productive_multihop: int = 0                   # kept trajectories that are productive AND hops>=2 (report)
    no_search: int = 0                             # generated trajectories with zero search (report / red-flag input)
    retrieval_empty: int = 0                       # generated trajectories whose every search returned no docs (report)

    def as_dict(self) -> dict:
        frac = (self.productive_multihop / self.kept) if self.kept else None
        return {"kept": self.kept, "dropped": self.dropped, "hops": self.hops,
                "productive_multihop": self.productive_multihop,
                "productive_multihop_frac": frac,
                "no_search": self.no_search, "retrieval_empty": self.retrieval_empty}


def _trajectory_stream(questions, golds, policy, retriever, tokenizer, cfg: SFTGenConfig, batch: int):
    """(i, question, trajectory) in question order. batch <= 1 is the original one-question-
    at-a-time path; batch > 1 drives rollout.generate_trajectories in gathered rounds -- the
    SAME coroutine per trajectory (rollout.rollout), so a trajectory's control flow does not
    depend on its batch-mates, only the engine's batching does. Added 2026-09-10 for the SeqKD
    baseline: the single-stream path decoded a 32B teacher at ~22 action tokens/s
    (runs/prod_dense_sft/stats.json), which priced 38.4k trajectories at ~240 card-hours."""
    if batch <= 1:
        for i, q in enumerate(questions):
            gold = golds[i % len(golds)] if golds else None
            yield i, q, generate_trajectory(q, policy, retriever, tokenizer, max_turns=cfg.max_turns,
                                            max_len=cfg.max_len, top_k=cfg.top_k, seed=cfg.seed, gold_answer=gold)
        return
    for start in range(0, len(questions), batch):
        qs = questions[start:start + batch]
        gs = [golds[j % len(golds)] if golds else None for j in range(start, start + len(qs))]
        trajs = generate_trajectories(qs, policy, retriever, tokenizer, max_turns=cfg.max_turns,
                                      max_len=cfg.max_len, top_k=cfg.top_k, seed=cfg.seed, golds=gs)
        for j, traj in enumerate(trajs):
            yield start + j, qs[j], traj


def generate_sft_dataset(*, questions, golds, policy, retriever, tokenizer,
                         cfg: SFTGenConfig, gen_backend: BackendConfig,
                         teacher_backend: BackendConfig, out_dir: str,
                         tags=None, engine_kwargs: dict = None, batch: int = 1) -> GenStats:
    """Generate -> drop-only-if-structurally-broken -> action-only-mask -> save. Returns GenStats.

    The FIRST thing this does is enforce teacher-consistency: the generator model must
    be the OPD teacher model (assert_same_teacher_model). This makes the red line
    machine-checked -- a mismatched generator raises before any data is written.

    `tags` (optional, parallel to questions) is the per-question DECLARED hop class /
    source label from load_mixed (1hop/2hop/multihop). It is provenance ONLY -- it never
    gates or filters -- and is written into every manifest row as `source_tag` so the
    health report can split natural_frac / answered% / think / productive_multihop PER
    SOURCE. The aggregate hides the real signal because the three sources have vastly
    different difficulty, so each layer is judged against its MEASURED baseline, not an
    invented threshold. MiroThinker is an exhaustive-confirmation teacher, so a low
    natural_frac is its real signature, NOT a defect; measured full-regen natural_frac
    (anti-confirmation prompt f9da237f476eb71e): 1hop ~39%, 2hop ~31%, multihop ~19%. The
    alarm is a layer materially BELOW its baseline (regression), which the aggregate masks.
    """
    # RED LINE: SFT generator must be the OPD teacher model (MiroThinker). Raises loudly.
    assert_same_teacher_model(teacher_backend, gen_backend)

    records_dir = os.path.join(out_dir, "records")
    os.makedirs(records_dir, exist_ok=True)

    stats = GenStats()
    manifest: list[dict] = []
    # Generation wall clock, started AFTER the engine is up (make_generator loads the weights
    # and runs torch.compile before this function is entered), so the number isolates sampling
    # from a ~2-minute fixed start-up cost that would otherwise dominate a 64-question run.
    # Without this, "how fast is this teacher" can only be reconstructed by subtracting log
    # timestamps by hand, and per-model throughput is a question the three-line comparison
    # actually asks: a dense 32B reads ~64 GB per decoded token where a 30B-A3B MoE reads
    # ~6 GB, and nothing on disk recorded which side of that a shard came from.
    t0 = time.monotonic()
    tok_total = act_total = 0
    for i, q, traj in _trajectory_stream(questions, golds, policy, retriever, tokenizer, cfg, batch):
        rq = retrieval_quality(traj.meta)               # per-hop retrieval signal (compute once)
        prod = productivity_summary(traj, rq)           # DIAGNOSTIC only (never gates)
        v = classify_trajectory(traj)                   # structural drop check ONLY
        retrieval_empty = _all_retrieval_empty(traj)
        # token-length signals over ALL generated trajectories (kept OR dropped): n_tokens is
        # the FULL trajectory length (prompt+action+observation), n_action_tokens is the
        # supervised ACTION span (== build_score_mask). These feed the health report's
        # teacher-complexity-vs-4B-student-capacity readout -- long MiroThinker trajectories
        # are a signal the 4B student may struggle to imitate, NOT a drop reason.
        n_tokens = int(traj.input_ids.shape[0])
        n_action_tokens = int(build_score_mask(traj.token_type).sum())
        # totals over ALL generated trajectories, kept or dropped -- the GPU produced the
        # dropped ones too, so throughput computed over kept-only would flatter a teacher
        # whose output the structural filter happens to like.
        tok_total += n_tokens
        act_total += n_action_tokens
        # tally report-only device/behavior signals over ALL generated trajectories.
        if prod["hops"] == 0:
            stats.no_search += 1
        if retrieval_empty:
            stats.retrieval_empty += 1
        # every manifest row carries the diagnostic fields (kept OR dropped) so the check
        # script can report behavior distributions over ALL generated trajectories.
        row = {"i": i, "question": q, "hops": prod["hops"],
               # declared source / hop class (1hop/2hop/multihop) from load_mixed -- provenance
               # ONLY (never gates), threaded through so the health report can split per source.
               "source_tag": (tags[i] if tags is not None and i < len(tags) else None),
               "productive": prod["productive"],
               "productive_multihop": prod["productive_multihop"],
               "per_turn_hit_rate": prod["per_turn_hit_rate"],
               "no_search": prod["hops"] == 0,
               "retrieval_empty": retrieval_empty,
               # forced-answer seam (Guard 1) + verbatim-repeat (Guard 2) diagnostics, over ALL
               # generated trajectories -> the smoke report reads the trigger/success/repeat rates.
               "forced_answer_fired": bool(traj.meta.get("forced_answer_fired")),
               "forced_answer_answered": bool(traj.meta.get("forced_answer_answered")),
               # did the teacher's native <think> channel fire in ANY action span (regression
               # guard for the empty-<think></think> suppression bug -> think-appearance rate).
               "has_think": bool(traj.meta.get("has_think")),
               "n_duplicate_queries": int(traj.meta.get("n_duplicate_queries") or 0),
               # FULL forced-turn text (incl. for DROPPED not_terminated rows) -> lets us see
               # whether a failed forced turn re-searched / got cut before <answer> / missed format.
               "forced_answer_text": traj.meta.get("forced_answer_text"),
               # the terminal answer, twice: `final_answer` is the extracted span (what a
               # scorer would read) and `final_action_text` is the whole action the model
               # wrote. Keeping only the first makes "how long are the answers" and "is there
               # deliberation inside the span" unanswerable without regenerating -- and those
               # are exactly the questions a prompt A/B asks. Present on DROPPED rows too,
               # which have no *.pt record at all.
               "final_answer": traj.meta.get("final_answer"),
               "final_action_text": traj.meta.get("final_action_text"),
               "final_action_n_tok": traj.meta.get("final_action_n_tok"),
               # per-turn shape: query + whether Guard 2 flagged it a verbatim repeat + #docs
               # retrieved (dup => 0; a real search that returned nothing => 0 but dup=False).
               # This exposes the UPSTREAM cause: a forced-turn failure whose 4 turns were mostly
               # duplicates arrived at the wall with ~no evidence (fix Guard 2), vs one that had
               # real evidence but still didn't answer (fix the forced turn).
               # `text` is the turn's RAW action text, persisted in full. Without it an
               # empty_query drop is unreadable: the reason covers a literal <search></search>,
               # an unclosed <answer>, and prose with no tool call at all (rollout routes every
               # non-answer action here), and those need opposite fixes -- prompt-format work
               # vs a length/termination problem. Recovering it after the fact costs a full
               # regeneration, so it is written at generation time.
               "turns_summary": [{"q": (t.get("query") or ""),
                                  "dup": bool(t.get("duplicate")),
                                  "ndocs": len(t.get("docs") or []),
                                  "text": (t.get("text") or ""),
                                  "n_tok": t.get("n_tok")}
                                 for t in (traj.meta.get("turns") or [])],
               # termination reason for EVERY row, kept or dropped. It used to be written only
               # on the kept branch, which made a dropped row unreadable in the one way that
               # matters: an empty_query row that nonetheless reached ANSWERED is a recoverable
               # trajectory (one malformed turn), while one that ended TRUNCATED is not.
               "terminated": traj.terminated.value if traj.terminated else None,
               "n_tokens": n_tokens,
               "n_action_tokens": n_action_tokens}
        if not v.keep:
            stats.dropped[v.reason] = stats.dropped.get(v.reason, 0) + 1
            row.update({"kept": False, "reason": v.reason})
            manifest.append(row)
            continue
        rec = build_sft_record(traj, prod)
        path = os.path.join(records_dir, f"sft_{i:04d}.pt")
        torch.save(rec, path)
        stats.kept += 1
        stats.hops[prod["hops"]] = stats.hops.get(prod["hops"], 0) + 1
        if prod["productive_multihop"]:
            stats.productive_multihop += 1
        row.update({"kept": True, "path": path})
        manifest.append(row)

    with open(os.path.join(out_dir, "manifest.jsonl"), "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        # persist the generation budget so the health report reads the ACTUAL max_len used
        # (single source of truth) -- otherwise check_sft_stats falls back to a CLI default
        # that silently drifts from rollout.max_len and mis-reports the near-cap fraction.
        out = stats.as_dict()
        out["gen_config"] = {"max_len": cfg.max_len, "max_turns": cfg.max_turns,
                             "top_k": cfg.top_k, "batch": int(batch)}
        # WHICH PROMPT MADE THIS SHARD. The fingerprint alone proves two runs differ; it
        # cannot say how, and a prompt A/B produces directories that are otherwise identical.
        # The full text is stored, not just the variant name: a variant name is a pointer into
        # a file that keeps changing, and the point of provenance is to survive that.
        # what the engine actually ran with. The generation stage now RESOLVES its
        # utilization from the checkpoint at runtime, so this is no longer derivable from the
        # config a reader has in front of them.
        out["engine"] = dict(engine_kwargs or {})
        elapsed = time.monotonic() - t0
        n_gen = len(questions)
        # ACTION tokens per second is the honest throughput figure: observation tokens are
        # appended by the environment, and with prefix caching on they are prefilled, not
        # decoded. Decode is ~200x slower per token than prefill, so action tokens are what
        # the wall clock is actually made of. All four raw numbers are stored so any other
        # rate can be recomputed without re-running anything.
        out["throughput"] = {
            "elapsed_s": round(elapsed, 2),
            "n_generated": n_gen,
            "n_tokens_total": tok_total,
            "n_action_tokens_total": act_total,
            "seconds_per_question": round(elapsed / n_gen, 3) if n_gen else None,
            "action_tokens_per_s": round(act_total / elapsed, 2) if elapsed > 0 else None,
        }
        out["template"] = {"fingerprint": template_fingerprint(),
                           "prompt_variant": T.PROMPT_VARIANT,
                           "system_prompt": T.SYSTEM_PROMPT,
                           "enable_thinking": T.ENABLE_THINKING}
        json.dump(out, f, indent=2, ensure_ascii=False)
    return stats
