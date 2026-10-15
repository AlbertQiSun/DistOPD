#!/usr/bin/env python3
"""M-SFT Stage 3: π_ref QUALITY ACCEPTANCE -- does the SFT checkpoint actually SEARCH?

Row 9 Stage 2 proved the mask survived and the checkpoint LOADS. Neither proves the
thing OPD's whole offline-advantage argument rests on: that π_ref is a COMPETENT
agentic searcher (emits sensible queries -> surfaces evidence -> synthesizes an answer).
A randomly-initialized 4B also "loads OK". If π_ref never learned to search, the states
it induces are dirty from step 0, the offline premise ("π_ref induces coherent state,
teacher calibrated on it") collapses, and we would misread "offline also fails" as an
off-dynamics result when the real cause is a soft π_ref foundation. So Stage 3 GATES OPD.

WHAT THIS RUNS
--------------
The SAME multi-turn rollout the trainer uses (rollout.generate_trajectory over the live
HTTP retriever), on OFFICIAL DEV questions, and scores three gates:

  GATE A  format legality      -- does it emit well-formed actions and terminate via
                                  <answer> (not run out of turns/length)?
  GATE B  retrieval effectiveness (BASELINE-INDEPENDENT) -- are the queries healthy and
                                  do the retrieved docs actually CONTAIN the gold? This
                                  gate alone answers "is π_ref's search REAL or a shell",
                                  regardless of any baseline dispute.
  GATE C  answer EM/F1 vs a FAIR baseline -- run the identical harness on the baseline
                                  model and compare. The baseline is Qwen3-4B-INSTRUCT,
                                  NOT the un-SFT'd Base: the raw Base doesn't know the
                                  <search>/<answer> protocol (that's what SFT teaches), so
                                  it collapses to ~0 on format AND EM, and "π_ref 40% vs
                                  Base 0%" cannot distinguish "learned to search" from
                                  "merely learned the tag format". Instruct follows the
                                  protocol (the frozen SYSTEM_PROMPT already describes it),
                                  so π_ref - Instruct is a FAIR agentic-capability delta.

CONTAMINATION GUARD (MANDATORY, not optional)
---------------------------------------------
The SFT trajectories were generated from the TRAIN splits of NQ/HotpotQA/MuSiQue (see
scripts/prepare_sft_data.py: it pulls train splits from FlashRAG). Evaluating on the
OFFICIAL DEV split is therefore disjoint BY CONSTRUCTION -- but FlashRAG is a second-hand
aggregator whose train/dev partition is NOT guaranteed to equal the original official one
(re-shuffles / de-dup / version mixing are possible). Project discipline turns "should be
clean" into "proven clean": we compute the normalized-question intersection of the eval
set with the exact SFT train file(s), DROP any overlap, and assert the remainder is
disjoint. A contaminated EM would falsely inflate Gate C and misjudge π_ref competent.

READING GATE B -- IT IS A LOWER BOUND, NOT A PRECISE SEARCH-ABILITY MEASUREMENT
------------------------------------------------------------------------------
hit@k depends not only on whether π_ref issues good queries, but ALSO on whether the gold
evidence is present in THIS Wikipedia dump and in a form the query can reach. Our dump
version need not match the Wikipedia that HotpotQA/NQ were annotated against (Wikipedia
drifts; wording/entities change). retrieval never being empty rules out a dead retriever,
NOT "gold is in the dump but in a form π_ref's query can't surface". So:
  * hit@k clearly > 0  => π_ref HAS basic search ability (competent enough to surface
    evidence)  -- a trustworthy LOWER BOUND.
  * hit@k not very high => NOT necessarily π_ref's fault; may be dump-version mismatch.
Do NOT fail π_ref as incompetent on a middling Gate B alone -- read it together with Gate A
(is it even searching well-formed?) and Gate C (does it answer?).

SHARED CORPUS IS A DELIBERATE, LOCKED CONTROLLED VARIABLE (do NOT "optimize" it away)
------------------------------------------------------------------------------------
All Wikipedia-based simple sets (NQ / HotpotQA / 2Wiki / MuSiQue) share ONE Wikipedia dump +
E5 index. Intentional: it is the Search-R1/FlashRAG setup we benchmark against, and it
removes a confound (otherwise "π_ref worse on HotpotQA than NQ" cannot separate task
difficulty from corpus quality) -- the study is about OPD dynamics, not the retriever. The
accepted cost is the dump-drift above (hence Gate B = lower bound). NEVER give different
Wikipedia sets different corpora.

SCOPE -- THIS IS STAGE-3 ACCEPTANCE, NOT THE FINAL-EVAL INFRASTRUCTURE
---------------------------------------------------------------------
FlashRAG dev + Wikipedia+E5 is the right, clean setup to ACCEPT π_ref and get the first
offline>online signal. It is NOT the final evaluation. BrowseComp-Plus / FRAMES are
long-horizon, not FlashRAG, not covered by a single-hop Wikipedia dump, and likely need web
retrieval -- a SEPARATE, still-TBD corpus/retriever. Stage 3 passing here must NOT create
the illusion that the final-eval infrastructure exists; that is handled on its own later.

SAMPLING
--------
Temp matches the OPD rollout (rollout.temperature, 0.8) -- Stage 3 must observe how π_ref
behaves AS IT WILL IN OPD, not a greedy argmax it never uses there. Each question is
sampled K times and metrics averaged, to denoise the stochastic policy.

    # real (HPC): one card for the model + the live retriever job. NO teacher needed.
    python scripts/stage3_eval.py --config configs/opd_qwen3.yaml \
        --model /path/to/pi_ref_ckpt --label pi_ref \
        --eval-data /path/to/hotpotqa_dev.jsonl --dataset-name hotpotqa \
        --sft-data /path/to/scratch/opd_sft_data/hotpotqa_with_gold.jsonl \
        --retriever-endpoint http://<host>:8000/retrieve --limit 200 --num-samples 4
    # ...then again with --model <Qwen3-4B-Instruct> --label instruct, same flags.
    python scripts/stage3_eval.py --compare runs/stage3/stage3_pi_ref.json \
                                            runs/stage3/stage3_instruct.json

    # local wiring smoke-test (no model/retriever/vllm):
    python scripts/stage3_eval.py --dry_run --label smoke
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import sys
from statistics import mean
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opd.config import load_config  # noqa: E402
from opd.diagnostics import (  # noqa: E402
    answer_em_f1,
    normalize_answer,
    query_health,
    retrieval_quality,
    termination_record,
)
from opd.trajectory import TerminationReason  # noqa: E402
from rollout.rollout import generate_trajectory  # noqa: E402

# deterministic mock corpus/questions for the CPU dry-run (mirrors scripts/train.py)
_DRY_CORPUS = [
    "Paris is the capital of France and home to the Eiffel Tower.",
    "The Eiffel Tower is 330 metres tall, located in Paris.",
    "Berlin is the capital of Germany.",
]
_DRY_QUESTIONS = [
    {"question": "What is the capital of France?", "gold_answers": ["Paris"]},
    {"question": "How tall is the Eiffel Tower?", "gold_answers": ["330 metres", "330 m"]},
]


# ---------------------------------------------------------------------------
# factories -- real vs dry-run mock (same swap pattern as scripts/train.py)
# ---------------------------------------------------------------------------
def _make_tokenizer(model_path: str, dry_run: bool):
    if dry_run:
        from envs.tokenizer import MockTokenizer
        return MockTokenizer(vocab_size=8192)
    from transformers import AutoTokenizer
    from envs.hf_tokenizer import HFTokenizerAdapter
    return HFTokenizerAdapter(AutoTokenizer.from_pretrained(model_path))


def _make_retriever(endpoint: Optional[str], dry_run: bool, seed: int):
    if dry_run:
        from envs.retriever import MockRetriever
        return MockRetriever(_DRY_CORPUS, seed=seed)
    if not endpoint or "REPLACE_ME" in endpoint:
        raise SystemExit(
            "FATAL: no live retriever endpoint. Pass --retriever-endpoint "
            "http://<retriever-host>:8000/retrieve (the running retriever job's host).")
    from envs.http_retriever import HTTPRetriever
    return HTTPRetriever(endpoint=endpoint)


def _make_policy(model_path: str, tokenizer, dry_run: bool, *, temperature: float,
                 top_p: float, dtype: str, seed: int, max_action_tokens: int,
                 gpu_memory_utilization: Optional[float] = None,
                 max_model_len: Optional[int] = None):
    """TEACHER-AS-POLICY is a supported use of --model, with one methodological caveat.

    Pointing --model at a TEACHER (MiroThinker-1.7-mini / Qwen3-30B-A3B-Thinking /
    Qwen3-32B) measures the teacher's DELIVERABLE performance under THIS interface --
    the <search>/<answer> protocol, this turn cap, this action-token cap, this retriever.
    None of the three teachers was trained on this interface (MiroThinker's home regime is
    MCP/XML tool calls with 200+ invocations; the Qwen teachers are generic thinkers), so
    the number is NOT the teacher's capability ceiling and must never be quoted as one.

    It is, however, exactly the right number for HEADROOM: distillation can only transfer
    behaviour expressible inside the interface, so "teacher-as-policy minus pi_ref" bounds
    what OPD could have gained. teacher ~= pi_ref means SFT already banked everything the
    interface admits; teacher >> pi_ref means transfer failed somewhere real.

    The tokenizer passed here is built from the SAME --model path (run_eval does this), so a
    teacher scores with its own template -- do not hand a teacher pi_ref's tokenizer.

    The two engine knobs exist because a 32B teacher on one 80 GiB card has no room for
    vLLM's defaults: default gpu_memory_utilization keeps too little KV for the model's
    native max_model_len and the engine can refuse to boot. They default to None = vLLM's
    own defaults, so every existing 4B call site is unchanged.
    """
    if dry_run:
        from rollout.policy import MockPolicy
        # scripted searcher: two searches then answers "Paris" (matches _DRY_CORPUS)
        return MockPolicy(tokenizer, queries=["capital of France", "Eiffel Tower height"],
                          answer="Paris")
    from vllm import LLM
    from rollout.vllm_policy import VLLMPolicy
    engine_kwargs = {}
    if gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = max_model_len
    llm = LLM(model=model_path, dtype=dtype, **engine_kwargs)
    return VLLMPolicy(llm, tokenizer, temperature=temperature, top_p=top_p, seed=seed,
                      max_action_tokens=max_action_tokens)


# ---------------------------------------------------------------------------
# contamination guard (MANDATORY on real runs)
# ---------------------------------------------------------------------------
def _sft_question_keys(sft_paths: list[str]) -> set[str]:
    """Normalized-question keys of every SFT TRAIN question (the generation pool).

    The *_with_gold.jsonl files carry only {question, golden_answers} (no stable id),
    so the intersection is on the NORMALIZED question string -- the same SQuAD/DPR
    normalizer used everywhere else, so punctuation/article/casing variants still match
    (conservative: borderline matches are treated as overlap and dropped, the safe
    direction for contamination)."""
    from dataio.dataset import load
    keys: set[str] = set()
    for p in sft_paths:
        for r in load(p):
            k = normalize_answer(r["question"])
            if k:
                keys.add(k)
    return keys


def _decontaminate(eval_rows: list[dict], sft_paths: list[str]) -> tuple[list[dict], int]:
    """Drop eval questions that appear in the SFT train pool; PROVE the remainder is
    disjoint. Returns (clean_rows, n_removed)."""
    sft_keys = _sft_question_keys(sft_paths)
    kept, removed = [], 0
    for r in eval_rows:
        if normalize_answer(r["question"]) in sft_keys:
            removed += 1
        else:
            kept.append(r)
    # the proof: after removal the eval set is provably disjoint from the SFT pool
    assert all(normalize_answer(r["question"]) not in sft_keys for r in kept), \
        "decontamination failed -- overlap remains after filtering"
    return kept, removed


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def _mean(xs: list) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def load_question_keys(path: str) -> set:
    """Normalized question keys from a selection file ({"questions": [...]} or a bare list)."""
    with open(path, encoding="utf-8") as f:
        wanted = json.load(f)
    if isinstance(wanted, dict):
        wanted = wanted.get("questions", [])
    return {normalize_answer(q) for q in wanted}


def require_matching_limit(path: str, limit: int) -> None:
    """The selection knows how long the prefix was when it was made; refuse a shorter one.

    Without this the failure is a wall of "requested questions are not in the first N rows"
    that names the symptom, not the fix. And a subset run that used a DIFFERENT --limit would
    be wrong even where the questions happen to match, because base_seed is seed + 1000*i
    over the prefix and a shorter prefix renumbers everything after the first gap.
    """
    with open(path, encoding="utf-8") as f:
        sel = json.load(f)
    need = sel.get("n_source_questions") if isinstance(sel, dict) else None
    if need and limit < need:
        raise SystemExit(
            f"FATAL: this selection came from a {need}-question run but --limit is {limit}. "
            f"The selected questions live past row {limit}, and even the ones that did match "
            f"would draw different samples, because the per-question seed is derived from "
            f"the index in the prefix. Re-run with OPD_EVAL_LIMIT={need}.")


def subset_todo(todo: list, keys: set, limit: int) -> list:
    """Filter [(original_index, row)] to `keys`, KEEPING the original index.

    The index is the whole point: the per-question base_seed is seed + 1000*i, so filtering
    and re-enumerating would draw different samples than the full run drew on the very same
    question -- and the first draws of a larger-K rerun would no longer reproduce the
    original ones. A requested question that is not present is fatal rather than dropped,
    because a silently shrunk question set changes the denominator of every rate reported.
    """
    kept = [(i, r) for i, r in todo if normalize_answer(r["question"]) in keys]
    missing = keys - {normalize_answer(r["question"]) for _i, r in kept}
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} of {len(keys)} requested questions are not in the first "
            f"{limit} decontaminated rows of this dataset. Raise --limit, or check that the "
            f"selection came from THIS dataset. Example: {sorted(missing)[0][:80]!r}")
    return kept


def _dump_trajectory(traj, out_dir: str, qi: int, s: int, backend, seed: int = -1) -> None:
    """Write ONE eval-time trajectory in the precompute cache's record shape.

    WHY THIS EXISTS. Every token-level measurement in this project scores checkpoints on
    pi_ref's TRAINING trajectories, while the eval that shows the damage happens on states
    each checkpoint walks into itself. Nothing could close that gap, because the report
    persists trajectory TEXT (turns_detail[].text, final_action_text, tail_text) and never
    the token ids -- so rebuilding a probe set would mean decode->re-encode, which is not
    guaranteed to reproduce the ids and would misalign every per-position log-prob and role
    label. Same under-persist shape as every earlier under-persist incident; the fix is to write the ids
    while they exist.

    Dumping from INSIDE the eval loop, rather than from a separate generator, is deliberate:
    the probe set is then produced by the same policy construction, the same retriever, the
    same decontaminated question order and the same per-sample seeds as the numbers it is
    meant to explain. A standalone generator would be an approximation of the eval states.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"input_ids": traj.input_ids.cpu(),
                "token_type": traj.token_type.cpu(),
                "terminated": getattr(traj.terminated, "value", traj.terminated),
                "num_search_calls": int(traj.num_search_calls),
                "meta": traj.meta,
                # The SEED, because base_seed is a function of the question's index in the
                # decontaminated prefix and nothing on disk records it. Without it a record
                # cannot be tied back to the eval draw that produced it, which is the whole
                # point of dumping from inside the eval loop.
                "seed": int(seed), "question_index": int(qi), "sample_index": int(s),
                "backend": backend},
               # `traj_` PREFIX IS LOAD-BEARING: cross_score_cache.cache_paths globs
               # 'traj_*.pt' when a directory has no manifest, so any other name makes the
               # whole probe set invisible to every reader in this codebase.
               os.path.join(out_dir, f"traj_q{qi:05d}_s{s}.pt"))


def _eval_question(q: str, gold: list[str], policy, retriever, tokenizer, *,
                   max_turns: int, max_len: int, top_k: int, num_samples: int,
                   base_seed: int, dump_dir: str = None, dump_qi: int = 0,
                   dump_backend=None) -> dict:
    """Roll out `num_samples` trajectories for ONE question and average the metrics.

    Also persists the per-sample RAW materials under "samples" -- the prediction TEXT plus
    each sample's em/f1/cover_em/answered/terminated -- NOT only the K-averaged means. This is
    deliberate: the prediction text is the load-bearing field the old schema dropped, and
    without it ANY new answer metric (cover_em here, or a future one) would need a full GPU
    re-run instead of a free offline recompute. Rule: persist raw materials.
    """
    em, f1, cover, answered = [], [], [], []
    union_hit, per_turn_hit, first_hit, n_searches = [], [], [], []
    degeneracy, query_len = [], []
    answered_via_answer, trunc_maxlen, trunc_maxturns, no_search = [], [], [], []
    samples = []
    for s in range(num_samples):
        seed = base_seed + s
        if hasattr(policy, "seed"):
            policy.seed = seed          # per-sample seed -> independent draws (denoise)
        traj = generate_trajectory(q, policy, retriever, tokenizer,
                                   max_turns=max_turns, max_len=max_len, top_k=top_k,
                                   seed=seed, gold_answer=gold)
        if dump_dir:
            _dump_trajectory(traj, dump_dir, dump_qi, s, dump_backend, seed)
        task = answer_em_f1(traj.meta.get("final_answer"), traj.meta.get("gold_answer"))
        retr = retrieval_quality(traj.meta)
        qh = query_health(traj.meta)
        term = termination_record(traj)

        em.append(task["em"]); f1.append(task["f1"]); answered.append(float(task["answered"]))
        cover.append(task["cover_em"])
        union_hit.append(None if retr["union_hit"] is None else float(retr["union_hit"]))
        per_turn_hit.append(retr["per_turn_hit_rate"])
        first_hit.append(None if retr["first_hit_turn"] < 0 else retr["first_hit_turn"])
        n_searches.append(int(traj.num_search_calls))
        degeneracy.append(qh["degeneracy_rate"]); query_len.append(qh["query_len_mean"])
        answered_via_answer.append(float(traj.terminated == TerminationReason.ANSWERED))
        trunc_maxlen.append(float(traj.terminated == TerminationReason.TRUNCATED_MAXLEN))
        trunc_maxturns.append(float(traj.terminated == TerminationReason.TRUNCATED_MAXTURNS))
        no_search.append(float(int(traj.num_search_calls) == 0))
        # RAW material for offline re-scoring: prediction TEXT is what the old schema dropped.
        # queries + tail_text joined it on 2026-08-20: the objective-artifact check
        # (Demystifying-OPD's RKL-repetition runaway produces the same truncation
        # signature as termination drift) needs to READ what a truncated trajectory's
        # final stretch actually says -- distinct well-formed searches vs a degenerate
        # loop -- and nothing persisted the trajectory text, so that check would have
        # cost a GPU re-run. Same under-persist shape as every earlier under-persist incident.
        tk = getattr(tokenizer, "tok", tokenizer)
        samples.append({
            "prediction": traj.meta.get("final_answer"),
            "em": task["em"], "f1": task["f1"], "cover_em": task["cover_em"],
            "answered": bool(task["answered"]),
            "terminated": getattr(traj.terminated, "name", str(traj.terminated)),
            "queries": [t.get("query", "") for t in (traj.meta.get("turns") or [])],
            "tail_text": (tk.decode(traj.input_ids[-768:].tolist(),
                                    skip_special_tokens=False)
                          if hasattr(tk, "decode") else None),
            # THE FORCED TURN, persisted. Guard 1 already prefills <answer> at max_turns
            # in every rollout, so a TRUNCATED_MAXTURNS outcome means forcing fired AND
            # the model still failed to close the span -- the "what does it do when
            # handed the open tag" experiment runs on every eval already. The rollout
            # computed forced_answer_text for exactly this question and the writer kept
            # only the termination reason, so classifying those failures (immediate EOS
            # vs prose that never closes vs another search) or salvage-scoring what was
            # written would have cost a full GPU re-run. Same shape as every earlier
            # persistence incident.
            "forced_fired": bool(traj.meta.get("forced_answer_fired")),
            "forced_answered": bool(traj.meta.get("forced_answer_answered")),
            "forced_text": traj.meta.get("forced_answer_text"),
            # --- SYMMETRY WITH THE SFT MANIFEST, added after an audit of what this writer
            # was still dropping from traj.meta. `queries` above kept ONLY the query string
            # from each turn, so everything else the rollout computed for this sample died
            # here: the raw action text (hence the ACTION-span composition, hence the
            # decisive-token share -- the quantity the whole dilution analysis needs), the
            # exact per-turn token count (hence whether max_action_tokens bit at eval time),
            # the retrieved documents (hence any retrieval metric anyone adds later), and
            # Guard 2's duplicate flag. `forced_text` covered the final action only when
            # forcing fired, so a NATURAL answer had no raw text at all and in-span
            # deliberation could not be measured on exactly the samples where it matters.
            #
            # Judge criterion: could a new metric be computed from this file
            # weeks from now without a GPU re-run? Before this block, no -- and that had
            # already cost one full re-run (cover_em). The cost of carrying it is a few MB.
            "turns_detail": [{"q": (t.get("query") or ""),
                              "dup": bool(t.get("duplicate")),
                              "ndocs": len(t.get("docs") or []),
                              "docs": list(t.get("docs") or []),
                              "text": t.get("text"),
                              "n_tok": t.get("n_tok")}
                             for t in (traj.meta.get("turns") or [])],
            "final_action_text": traj.meta.get("final_action_text"),
            "final_action_n_tok": traj.meta.get("final_action_n_tok"),
            "has_think": bool(traj.meta.get("has_think")),
            "n_duplicate_queries": int(traj.meta.get("n_duplicate_queries") or 0),
            "num_search_calls": int(traj.num_search_calls),
            "n_tokens": int(traj.input_ids.shape[0]),
        })
    return {
        "question": q, "gold": gold,
        "em": _mean(em), "f1": _mean(f1), "cover_em": _mean(cover), "answered": _mean(answered),
        "union_hit": _mean(union_hit), "per_turn_hit_rate": _mean(per_turn_hit),
        "first_hit_turn": _mean(first_hit), "n_searches": _mean(n_searches),
        "degeneracy_rate": _mean(degeneracy), "query_len_mean": _mean(query_len),
        "answered_via_answer": _mean(answered_via_answer),
        "trunc_maxlen": _mean(trunc_maxlen), "trunc_maxturns": _mean(trunc_maxturns),
        "no_search": _mean(no_search),
        "samples": samples,
    }


def _aggregate(rows: list[dict]) -> dict:
    """Roll per-question means up to the three gates."""
    gate_a = {   # format legality (generation-side structure)
        "answered_via_answer_rate": _mean([r["answered_via_answer"] for r in rows]),
        "truncated_maxlen_rate": _mean([r["trunc_maxlen"] for r in rows]),
        "truncated_maxturns_rate": _mean([r["trunc_maxturns"] for r in rows]),
        "no_search_rate": _mean([r["no_search"] for r in rows]),
        "query_degeneracy_rate": _mean([r["degeneracy_rate"] for r in rows]),
        "mean_num_searches": _mean([r["n_searches"] for r in rows]),
    }
    gate_b = {   # retrieval effectiveness -- BASELINE-INDEPENDENT search-quality backstop
        "union_hit_rate": _mean([r["union_hit"] for r in rows]),
        "per_turn_hit_rate": _mean([r["per_turn_hit_rate"] for r in rows]),
        "first_hit_turn_mean": _mean([r["first_hit_turn"] for r in rows]),
        "query_len_mean": _mean([r["query_len_mean"] for r in rows]),
    }
    gate_c = {   # answer performance -- compared against the Instruct baseline
        "em_mean": _mean([r["em"] for r in rows]),
        "cover_em_mean": _mean([r["cover_em"] for r in rows]),   # length-insensitive; strips style confound
        "f1_mean": _mean([r["f1"] for r in rows]),
        "answered_rate": _mean([r["answered"] for r in rows]),
    }
    return {"gate_A_format": gate_a, "gate_B_retrieval": gate_b, "gate_C_answer": gate_c}


def run_eval(args) -> None:
    cfg = load_config(args.config)
    roll = cfg["rollout"]
    max_turns, max_len, top_k = roll["max_turns"], roll["max_len"], roll["top_k"]
    # Sampled at the SAME action budget the SFT data was generated under. Indexed, not
    # .get(): a config without the key must fail here rather than fall back to a number
    # nobody chose -- see VLLMPolicy.__init__ for what that fallback cost.
    max_action_tokens = roll["max_action_tokens"]
    if args.max_turns is not None:
        # TERMINATION-CEILING PROBE, not a config drift: pi_ref averages 3.47 searches
        # under the training cap of 4, i.e. the cap is nearly saturated, so a policy that
        # merely searches a LITTLE more reads as catastrophic truncation. Overriding to a
        # larger budget at EVAL time distinguishes "truncation is a ceiling artifact"
        # (rates ~0, scores recover) from "the policy genuinely will not stop" (rates
        # persist at 2.5x budget). CAVEAT recorded in the report: the ckpts were TRAINED
        # at the config cap; this measures ceiling-sensitivity of the eval numbers, not
        # what training at the larger budget would produce.
        print(f"[stage3] max_turns OVERRIDDEN: {max_turns} -> {args.max_turns}")
        max_turns = args.max_turns
    if args.max_len is not None:
        # LONG-HORIZON budget : the whole-trajectory token
        # budget, recorded in the report's rollout block like max_turns; --max-model-len must
        # be at least this plus the prompt (Qwen3-4B: 40960 native).
        print(f"[stage3] max_len OVERRIDDEN: {max_len} -> {args.max_len}")
        max_len = int(args.max_len)
    top_p = roll.get("top_p", 1.0)
    temperature = args.temperature if args.temperature is not None else roll["temperature"]
    dtype = cfg["model"].get("dtype", "bfloat16")
    endpoint = args.retriever_endpoint or cfg.get("retriever", {}).get("endpoint")

    # ---- load eval questions (official DEV) ----
    if args.dry_run:
        eval_rows = [dict(r) for r in _DRY_QUESTIONS]
    else:
        if not args.eval_data:
            raise SystemExit("FATAL: --eval-data (official DEV jsonl) is required for a real run.")
        from dataio.dataset import load
        eval_rows = load(args.eval_data, name=args.dataset_name, limit=None)
        if not eval_rows:
            raise SystemExit(f"no usable rows in {args.eval_data}")
    n_loaded = len(eval_rows)

    # ---- CONTAMINATION GUARD (mandatory on real runs) ----
    n_removed = 0
    if args.dry_run and not args.sft_data:
        print("[stage3] dry-run: skipping contamination guard (no --sft-data given)")
    else:
        if not args.sft_data:
            raise SystemExit(
                "FATAL: --sft-data is MANDATORY -- the SFT TRAIN file(s) whose questions must "
                "not leak into the DEV eval set. Pass the *_with_gold.jsonl used for this "
                "dataset's SFT generation so we can PROVE the eval set is disjoint.")
        eval_rows, n_removed = _decontaminate(eval_rows, args.sft_data)
        tag = "CLEAN (train/dev disjoint as expected)" if n_removed == 0 else \
              f"WARNING: removed {n_removed} contaminated question(s) from eval"
        print(f"[stage3] contamination guard: {tag}; {len(eval_rows)} eval questions remain")

    # ---- subset AFTER decontamination so --limit counts clean questions ----
    if args.limit is not None:
        eval_rows = eval_rows[:args.limit]

    # ---- build the rollout stack for THIS model ----
    seed = cfg["run"]["seed"]
    tokenizer = _make_tokenizer(args.model, args.dry_run)
    retriever = _make_retriever(endpoint, args.dry_run, seed)
    policy = _make_policy(args.model, tokenizer, args.dry_run,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          max_model_len=args.max_model_len,
                          temperature=temperature, top_p=top_p, dtype=dtype, seed=seed,
                          max_action_tokens=max_action_tokens)

    print(f"[stage3] label={args.label} model={args.model} temp={temperature} "
          f"K={args.num_samples} | eval={len(eval_rows)} Q | max_turns={max_turns} "
          f"max_len={max_len} top_k={top_k} max_action_tokens={max_action_tokens}")

    # ---- INCREMENTAL PERSISTENCE + RESUME. Job 26845863 (dense offline, 2026-08-27) hit
    # its 9 h walltime around question ~900 of 1000 and left NOTHING: this writer only
    # produced the report at the end, so nine hours of completed rollouts were lost -- the
    # same under-persist shape as every earlier under-persist incident, in time instead of fields.
    # Every finished question is now appended to <report>.partial.jsonl immediately, and a
    # rerun skips the questions already there. The skip is SEED-SAFE: base_seed is derived
    # from the question's INDEX in the deterministic decontaminated prefix, which a resume
    # preserves by construction -- resumed and un-resumed runs are sample-identical.
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"stage3_{args.label}.json")
    partial_path = out_path + ".partial.jsonl"
    # SUBSET SELECTION, seed-preserving. The per-question base_seed is seed + 1000*i where i
    # is the index in the deterministic decontaminated prefix. Filtering eval_rows and
    # re-enumerating would renumber every survivor and silently draw DIFFERENT samples than
    # the full run did on the same question -- which is exactly the comparison a follow-up
    # subset run exists to make. So the original index travels with the row.
    todo = list(enumerate(eval_rows))
    subset_key = None
    if args.only_questions:
        require_matching_limit(args.only_questions, args.limit)
        todo = subset_todo(todo, load_question_keys(args.only_questions), args.limit)
        subset_key = f"{os.path.basename(args.only_questions)}:{len(todo)}"
        print(f"[stage3] SUBSET: {len(todo)} of {len(eval_rows)} questions, original "
              f"indices preserved (seeds unchanged)")

    # subset_key is IN the protocol: a partial file from the full 200-question run must not
    # be resumed into a subset run (or the reverse). Same clash guard, one more axis.
    protocol = {"label": args.label, "model": args.model, "num_samples": args.num_samples,
                "max_turns": max_turns, "temperature": temperature, "max_len": max_len,
                "subset": subset_key}
    done: dict[str, dict] = {}
    if os.path.exists(partial_path):
        with open(partial_path, encoding="utf-8") as pf:
            lines = [l for l in pf.read().splitlines() if l.strip()]
        if lines:
            head = json.loads(lines[0])
            prev = head.get("__protocol__")
            # A partial written before an axis existed cannot disagree about it. `subset`
            # was added on 2026-08-29; every file older than that is by construction a FULL
            # run, so the absent key means None and defaulting it is exact, not lenient.
            # Without this, adding the key would have rejected every partial already on
            # disk -- including the dense-offline one that the resume path exists to save.
            if prev is not None:
                prev = {"subset": None, **prev}
            if prev != protocol:
                raise SystemExit(
                    f"FATAL: {partial_path} was written under a different protocol "
                    f"(then: {prev} | now: {protocol}). Mixing them would put two eval "
                    "configurations in one report. Delete the partial file if the old "
                    "run is abandoned.")
            for line in lines[1:]:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue          # a killed job ends mid-line; everything before is good
                done[normalize_answer(rec["question"])] = rec
            print(f"[stage3] RESUMING: {len(done)} questions already in {partial_path}")
    pf = open(partial_path, "a", encoding="utf-8")
    if not done and os.path.getsize(partial_path) == 0:
        pf.write(json.dumps({"__protocol__": protocol}) + chr(10))
        pf.flush()

    # PROVENANCE FOR A DUMPED PROBE SET. Deliberately NOT a BackendConfig: these records
    # carry no teacher_logp, so stamping them with a precompute cache's backend would claim
    # a scoring provenance that does not exist and would let assert_same_backend pass on a
    # file that was never scored. What is true and worth carrying is which model walked
    # these states, under which template, on which machine.
    dump_prov = None
    if args.dump_trajectories:
        from teacher.backend import runtime_fingerprint, template_fingerprint
        dump_prov = {"kind": "eval_trajectories", "model": os.path.abspath(args.model),
                     "label": args.label, "template_fp": template_fingerprint(),
                     "num_samples": args.num_samples, "max_turns": max_turns,
                     "temperature": temperature, "seed": seed,
                     "runtime": runtime_fingerprint()}
        print(f"[stage3] dumping eval trajectories -> {args.dump_trajectories}")

    per_q = []
    n_skipped = 0
    for done_n, (i, r) in enumerate(todo):
        key = normalize_answer(r["question"])
        if key in done:
            per_q.append(done[key])
            n_skipped += 1
            continue
        rec = _eval_question(r["question"], r["gold_answers"], policy, retriever, tokenizer,
                             max_turns=max_turns, max_len=max_len, top_k=top_k,
                             num_samples=args.num_samples, base_seed=seed + 1000 * i,
                             dump_dir=args.dump_trajectories, dump_qi=i,
                             dump_backend=dump_prov)
        per_q.append(rec)
        pf.write(json.dumps(rec, ensure_ascii=False) + chr(10))
        pf.flush()
        if (done_n + 1) % 20 == 0 or (done_n + 1) == len(todo):
            print(f"  [{done_n + 1}/{len(todo)}] em={rec['em']} f1={rec['f1']} "
                  f"union_hit={rec['union_hit']}")
    pf.close()
    if args.dump_trajectories:
        # A manifest, so the probe set is read by cache_paths' PRIMARY path rather than its
        # glob fallback, and so the provenance travels with the data instead of living only
        # in this run's log.
        d = args.dump_trajectories
        names = sorted(x for x in os.listdir(d) if x.endswith(".pt"))
        with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as mf:
            json.dump({"paths": names, "provenance": dump_prov}, mf, indent=1)
        print(f"[stage3] wrote {d}/manifest.json ({len(names)} trajectories)")
    if n_skipped:
        print(f"[stage3] resume skipped {n_skipped} already-evaluated questions")

    report = {
        "label": args.label, "model": args.model,
        "eval_data": args.eval_data, "dataset_name": args.dataset_name,
        "n_questions_loaded": n_loaded, "n_contaminated_removed": n_removed,
        "n_questions_eval": len(eval_rows), "num_samples": args.num_samples,
        "temperature": temperature,
        "rollout": {"max_turns": max_turns, "max_len": max_len, "top_k": top_k,
                    "top_p": top_p, "max_action_tokens": max_action_tokens},
        "caveats": {
            "gate_B_is_lower_bound": (
                "hit@k depends on the Wikipedia dump version, not only on pi_ref's queries; "
                "clearly>0 proves basic search ability, but middling hit@k may be dump-drift, "
                "NOT pi_ref incompetence. Read with Gate A and Gate C."),
            "shared_corpus_is_deliberate": (
                "all Wikipedia sets share one dump+E5 (controlled variable vs Search-R1); "
                "never per-dataset corpora."),
            "scope": (
                "Stage-3 acceptance only. BrowseComp-Plus/FRAMES final eval is separate, "
                "TBD corpus/retriever -- not this Wikipedia+E5 setup."),
        },
        **_aggregate(per_q),
        "per_question": per_q,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # The report now holds everything the partial did; keeping the partial would let a later
    # run "resume" on top of a finished eval and silently skip all its work.
    if os.path.exists(partial_path):
        os.remove(partial_path)
    _print_gates(report)
    print(f"[stage3] wrote {out_path}")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x:6.3f}"


def _print_gates(rep: dict) -> None:
    print(f"\n===== Stage 3 gates: {rep['label']} "
          f"({rep['n_questions_eval']} Q x {rep['num_samples']} samples, "
          f"temp={rep['temperature']}) =====")
    a, b, c = rep["gate_A_format"], rep["gate_B_retrieval"], rep["gate_C_answer"]
    print("  GATE A  format legality")
    print(f"    answered_via_<answer>_rate : {_fmt(a['answered_via_answer_rate'])}")
    print(f"    truncated_maxlen_rate      : {_fmt(a['truncated_maxlen_rate'])}")
    print(f"    truncated_maxturns_rate    : {_fmt(a['truncated_maxturns_rate'])}")
    print(f"    no_search_rate             : {_fmt(a['no_search_rate'])}")
    print(f"    query_degeneracy_rate      : {_fmt(a['query_degeneracy_rate'])}")
    print(f"    mean_num_searches          : {_fmt(a['mean_num_searches'])}")
    print("  GATE B  retrieval effectiveness (baseline-independent; LOWER BOUND -- see below)")
    print(f"    union_hit_rate             : {_fmt(b['union_hit_rate'])}")
    print(f"    per_turn_hit_rate          : {_fmt(b['per_turn_hit_rate'])}")
    print(f"    first_hit_turn_mean        : {_fmt(b['first_hit_turn_mean'])}")
    print(f"    query_len_mean             : {_fmt(b['query_len_mean'])}")
    print("    NOTE: hit@k also depends on the Wikipedia dump version -- clearly>0 proves basic")
    print("          search ability; middling hit@k may be dump-drift, NOT pi_ref incompetence.")
    print("  GATE C  answer EM/F1 (compare vs Instruct baseline)")
    print(f"    em_mean                    : {_fmt(c['em_mean'])}")
    print(f"    cover_em_mean              : {_fmt(c.get('cover_em_mean'))}")
    print(f"    f1_mean                    : {_fmt(c['f1_mean'])}")
    print(f"    answered_rate              : {_fmt(c['answered_rate'])}")
    print("    NOTE: cover_em is length-INSENSITIVE (gold as substring). cover_em>>em means the")
    print("          answer is CORRECT but VERBOSE, not wrong -- read the em-vs-cover_em gap as")
    print("          answer STYLE, and the cover_em delta vs Instruct as capability.")


def run_compare(path_a: str, path_b: str) -> None:
    with open(path_a, encoding="utf-8") as f:
        ra = json.load(f)
    with open(path_b, encoding="utf-8") as f:
        rb = json.load(f)
    la, lb = ra["label"], rb["label"]
    print(f"\n===== Stage 3 comparison: {la}  vs  {lb} =====")

    def row(name, va, vb, delta=False):
        d = ""
        if delta and va is not None and vb is not None:
            d = f"   delta={va - vb:+.3f}"
        print(f"  {name:28s} {_fmt(va)} | {_fmt(vb)}{d}")

    print(f"  {'metric':28s} {la:>6s} | {lb:>6s}")
    print("  -- GATE A format --")
    for k in ("answered_via_answer_rate", "truncated_maxlen_rate", "truncated_maxturns_rate",
              "no_search_rate", "query_degeneracy_rate", "mean_num_searches"):
        row(k, ra["gate_A_format"][k], rb["gate_A_format"][k])
    print("  -- GATE B retrieval (baseline-independent) --")
    for k in ("union_hit_rate", "per_turn_hit_rate", "first_hit_turn_mean", "query_len_mean"):
        row(k, ra["gate_B_retrieval"][k], rb["gate_B_retrieval"][k])
    print("  -- GATE C answer (delta = capability lift; cover_em delta strips style confound) --")
    for k in ("em_mean", "cover_em_mean", "f1_mean", "answered_rate"):
        # .get: an OLD stage3_*.json (written before cover_em) has no cover_em_mean -> prints n/a
        row(k, ra["gate_C_answer"].get(k), rb["gate_C_answer"].get(k), delta=True)
    # DERIVED -- retrieval->answer CONVERSION = cover_em / union_hit: of the questions where gold
    # was retrieved AT ALL, what fraction actually surface it in the final answer. A model can
    # retrieve well yet SYNTHESIZE poorly (high union_hit, low conversion) -- that gap is a real
    # pi_ref trait (SFT 4B multi-hop synthesis ceiling), NOT a scoring artifact, and is worth
    # reporting. CAVEAT (do not over-read the raw ratio): union_hit is a cross-turn UNION, so a
    # model that searches MORE turns has more chances to hit and a mechanically inflated
    # denominator -> its conversion looks worse than it is. Read this next to per_turn_hit_rate /
    # first_hit_turn_mean (Gate B above) for a search-count-normalized view.
    def _conv(rep):
        uh = rep["gate_B_retrieval"].get("union_hit_rate")
        ce = rep["gate_C_answer"].get("cover_em_mean")
        return (ce / uh) if (uh and ce is not None) else None
    row("cover_em/union_hit (conv)", _conv(ra), _conv(rb), delta=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 π_ref quality acceptance (agentic search).")
    ap.add_argument("--compare", nargs=2, metavar=("REPORT_A", "REPORT_B"),
                    help="print a side-by-side gate table of two stage3_*.json reports and exit")
    ap.add_argument("--config", default="configs/opd_qwen3.yaml",
                    help="OPD config for rollout params (max_turns/max_len/top_k/temperature)")
    ap.add_argument("--model", default="MOCK",
                    help="model dir to evaluate (pi_ref_ckpt, or Qwen3-4B-Instruct baseline)")
    ap.add_argument("--label", default="pi_ref", help="report name -> stage3_<label>.json")
    ap.add_argument("--eval-data", default=None, help="official DEV jsonl (question + gold)")
    ap.add_argument("--dataset-name", default=None, help="dataset hint for gold parsing")
    ap.add_argument("--sft-data", nargs="+", default=None,
                    help="SFT TRAIN *_with_gold.jsonl file(s) for the contamination guard "
                         "(MANDATORY on real runs)")
    ap.add_argument("--retriever-endpoint", default=None,
                    help="live retriever host:port/retrieve (overrides config)")
    ap.add_argument("--limit", type=int, default=200, help="#DEV questions (after decontam)")
    ap.add_argument("--dump-trajectories", default=None,
                    help="directory to write each eval-time trajectory into, in the "
                         "precompute cache's record shape. The report persists trajectory "
                         "TEXT but never token ids, so without this a probe set on the "
                         "model's OWN states can only be rebuilt by decode->re-encode, "
                         "which is not id-exact and misaligns every per-position measure.")
    ap.add_argument("--only-questions", default=None,
                    help="JSON file of question strings: evaluate ONLY these, keeping each "
                         "question's ORIGINAL index in the decontaminated prefix so its "
                         "per-question base_seed is unchanged and the draws stay comparable "
                         "with the full run's.")
    ap.add_argument("--max-len", type=int, default=None,
                    help="override rollout.max_len (whole-trajectory token budget) for a long-horizon eval")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="override rollout.max_turns for THIS eval (termination-ceiling "
                         "probe; recorded in the report's rollout block)")
    ap.add_argument("--num-samples", type=int, default=4, help="rollouts per question (denoise)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override sampling temp (default = OPD rollout.temperature, 0.8)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="vLLM engine knob; needed for a 32B teacher on one 80 GiB card. "
                         "Default None = vLLM's default (existing 4B behaviour unchanged).")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="vLLM engine knob; cap the KV budget for big teachers. Set >= "
                         "rollout max_len or the engine truncates trajectories the 4B runs "
                         "kept.")
    ap.add_argument("--out-dir", default="runs/stage3")
    ap.add_argument("--dry_run", action="store_true",
                    help="CPU mock path (no model/retriever/vllm) -- wiring smoke-test only")
    args = ap.parse_args()

    if args.compare:
        run_compare(args.compare[0], args.compare[1])
        return
    run_eval(args)


if __name__ == "__main__":
    main()
