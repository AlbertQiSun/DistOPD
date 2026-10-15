"""Dataset loader -- the answer-recall@k correctness seam.

Real datasets (NQ / TriviaQA / HotpotQA / 2Wiki / Musique) name and shape the gold
answer differently, and the per-hop hit headline is only as correct as this parse.
So gold extraction is verified LOCALLY (no HPC needed): if the gold field is parsed
wrong, the mechanism headline is silently wrong.

Two things are pinned here to match prior work and avoid systematic curve shift:
  * gold is always a LIST OF ALIASES. Search-R1's unified processed format uses
    `golden_answers: [...]`; raw dataset formats are mapped to the same list shape.
  * hit semantics are OR over aliases (any alias, normalized, appearing in the
    passage counts) and use the SAME `normalize_answer` as task EM -- enforced in
    diagnostics.has_answer, which this loader feeds.

Each record is `{"question": str, "gold_answers": list[str]}`. train.py passes
`gold_answers` straight into the trajectory meta (`gold_answer`), which the
diagnostics answer-recall reads.
"""
from __future__ import annotations

import json
from typing import Optional

from opd.diagnostics import normalize_answer


def _clean(aliases) -> list[str]:
    """Coerce to a de-duplicated list of non-empty alias strings (order preserved)."""
    if aliases is None:
        return []
    if isinstance(aliases, str):
        aliases = [aliases]
    out, seen = [], set()
    for a in aliases:
        if a is None:
            continue
        s = str(a).strip()
        key = normalize_answer(s)
        if s and key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def extract_gold(row: dict, name: Optional[str] = None) -> list[str]:
    """Map a raw dataset row to the alias list. Prefers Search-R1's unified
    `golden_answers`; otherwise falls back to per-dataset raw fields."""
    if "golden_answers" in row:                       # Search-R1 unified format
        return _clean(row["golden_answers"])
    if "answers" in row:                              # NQ-style multi short-answer
        return _clean(row["answers"])

    ans = row.get("answer")
    if isinstance(ans, dict):                         # TriviaQA raw: {value, aliases}
        aliases = list(ans.get("aliases") or [])
        aliases += list(ans.get("normalized_aliases") or [])
        if ans.get("value"):
            aliases.append(ans["value"])
        return _clean(aliases)
    # HotpotQA / 2Wiki / Musique raw: single "answer" string
    return _clean(ans)


def load(path: str, name: Optional[str] = None, limit: Optional[int] = None,
         offset: int = 0) -> list[dict]:
    """Load a JSONL (or JSON array) file into [{question, gold_answers}].

    `name` is an optional dataset hint for the raw-format fallback; the unified
    `golden_answers` field, when present, makes it unnecessary. Rows without a
    question or without any gold alias are dropped (they cannot score answer-recall).

    `offset` skips the first `offset` VALID rows before collecting `limit` of them.
    The count is deliberately POST-FILTER -- the same valid-row counting `limit` uses --
    because that is the unit SFT consumed: `load(path, limit=N)` takes the first N VALID
    rows, so "skip the first N valid rows" is what makes a later slice disjoint from it.
    Counting raw file lines instead would drift by however many rows the filter dropped.

    Selection is deterministic first-N in file order (no shuffle), so a contiguous
    offset is disjoint from a smaller prefix BY CONSTRUCTION. That is what lets OPD
    train on an unused slice of the SFT train file instead of spending the dev split.
    """
    rows = _read(path)
    out: list[dict] = []
    skipped = 0
    for row in rows:
        q = (row.get("question") or "").strip()
        gold = extract_gold(row, name)
        if not q or not gold:
            continue
        if skipped < offset:                          # post-filter skip (see docstring)
            skipped += 1
            continue
        out.append({"question": q, "gold_answers": gold})
        if limit is not None and len(out) >= limit:
            break
    if offset and skipped < offset:
        raise ValueError(
            f"{path}: asked to skip {offset} valid rows but the file only has {skipped}. "
            "The offset is past the end -- the requested slice does not exist."
        )
    return out


def load_mixed(sources: list[dict], total: Optional[int] = None,
               shuffle: bool = True, seed: int = 0) -> list[dict]:
    """Load + tag + interleave several question sources into one difficulty-mixed set.

    Difficulty balance (e.g. ~30% 1-hop / ~50% 2-hop / ~20% multi-hop) is set by each
    source's own `limit` -- this loader does NOT invent difficulty, it composes files the
    caller already knows the hop class of. Each source is a dict:
        {"path": str, "name": str|None (opt), "limit": int|None (opt), "tag": str (opt)}
    The `tag` is provenance only (which file / declared hop class); the REAL difficulty of
    a generated trajectory is measured downstream by productive-hop count, never by the tag.

    NOTE this only shapes the INPUT question mix. It cannot by itself guarantee the OUTPUT
    (kept SFT trajectory) hop distribution -- MiroThinker may search fewer/more hops than a
    question's declared class, and spinning trajectories get dropped. So the tag here is the
    intent; scripts/check_sft_stats.py reports what was actually realized.
    """
    out: list[dict] = []
    for src in sources:
        rows = load(src["path"], name=src.get("name"), limit=src.get("limit"))
        tag = src.get("tag", src["path"])
        for r in rows:
            out.append({**r, "source": src["path"], "tag": tag})
    if shuffle:
        import random
        random.Random(seed).shuffle(out)
    if total is not None:
        out = out[:total]
    return out


def _read(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith("["):                      # JSON array
        return list(json.loads(stripped))
    rows = []
    for line in text.splitlines():                    # JSONL
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def preview(path: str, name: Optional[str] = None, n: int = 5) -> None:
    """Print the first `n` parsed rows so you can EYEBALL the gold ANSWER SHAPE before
    picking a headline metric. Short factual strings (e.g. TriviaQA, HotpotQA) score
    fine under EM; free-form / long answers (e.g. FRAMES) need F1 to be meaningful.
    Decide the primary metric PER DATASET from what you see here -- do not blanket-apply
    F1 to 'hard' sets.

        python -m dataio.dataset <path> [--name NAME] [--n 5]
    """
    rows = load(path, name=name, limit=n)
    print(f"{path}  (name={name!r})  showing {len(rows)} row(s):")
    for i, r in enumerate(rows):
        golds = r["gold_answers"]
        lens = [len(g.split()) for g in golds]
        shape = "short-fact" if (lens and max(lens) <= 3) else "free-form/long"
        print(f"  [{i}] Q: {r['question'][:90]}")
        print(f"      gold ({len(golds)} alias, max {max(lens) if lens else 0} tok, {shape}): {golds}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preview parsed gold answers for a dataset file.")
    ap.add_argument("path")
    ap.add_argument("--name", default=None)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    preview(args.path, name=args.name, n=args.n)
