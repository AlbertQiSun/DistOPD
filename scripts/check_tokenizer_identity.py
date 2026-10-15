#!/usr/bin/env python3
"""Teacher and student must tokenize IDENTICALLY. Checked against the staged checkpoints.

WHY THIS EXISTS. configs/opd_qwen3.yaml asserts that MiroThinker's tokenizer is "bit-identical
to the student's (vocab_size 151936; eos 151645; per-token ids equal)" and calls the property
load-bearing -- correctly, because OPD scores the student's exact ids position-by-position, so
a teacher that segments the same text differently misaligns every per-token advantage. Nothing
established the claim. It was a sentence, which is the situation the project's rule about
unverified equivalence claims exists to forbid.

It is not free of risk either. The two checkpoints came from different lineages (a 30B-A3B MoE
fine-tune vs a 4B base we SFT'd), they are loaded by whatever transformers version the channel
happens to resolve, and the failure is SILENT: no exception, no shape mismatch, just an
advantage attached to the wrong token. A gate that runs in seconds on CPU is the right price.

WHAT IT COMPARES, and why not simply the vocab. Equal vocab size and equal eos would pass with
two tokenizers that merge differently -- and merges are exactly what shifts boundaries. So the
probe set is the text this project actually feeds through: every frozen template marker (they
sit at the seams where a boundary shift would land), the system prompt, an observation block,
and the adversarial cases where a continuation can retro-actively re-segment what precedes it.

    python scripts/check_tokenizer_identity.py --teacher <path> --student <path>

Exit 0 = identical on every probe. Exit 1 = a divergence, with the first one printed in full.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import template as T  # noqa: E402


def probe_strings() -> list[str]:
    """The text whose segmentation this project depends on.

    Template markers first: they are where a merge difference actually bites, because the
    masking code locates action spans by them. Then the adversarial pairs -- a string and the
    same string plus one character -- which is how a boundary hazard shows up at all.
    """
    markers = [T.THINK_OPEN, T.THINK_CLOSE, T.SEARCH_OPEN, T.SEARCH_CLOSE,
               T.INFO_OPEN, T.INFO_CLOSE, T.ANSWER_OPEN, T.ANSWER_CLOSE,
               T.OBS_PREFIX, T.OBS_SUFFIX, T.IM_START, T.IM_END]
    out = list(markers) + [T.SYSTEM_PROMPT]
    # markers in context, not in isolation: a merge only happens next to something
    out += [f"{T.SEARCH_OPEN}capital of France{T.SEARCH_CLOSE}",
            f"{T.INFO_OPEN}Paris is the capital.{T.INFO_CLOSE}",
            f"{T.THINK_OPEN}I should search for this.{T.THINK_CLOSE}",
            f"{T.ANSWER_OPEN}Paris{T.ANSWER_CLOSE}",
            f"{T.OBS_PREFIX}Doc 1: Paris is in France.{T.OBS_SUFFIX}"]
    # adversarial: a continuation that can re-segment what precedes it
    for stem in ("Par", "search", "<sear", "answer", "  ", "\n\n"):
        out += [stem, stem + "is", stem + ">", stem + "\n"]
    out += ["Paris", " Paris", "Paris.", "北京是中国的首都", "𝕏 emoji 🇫🇷 test"]
    return out


# Fields the PIPELINE consumes, and therefore the only ones whose disagreement can break it.
# vocab_size decides the width of every logit row the teacher hands back, so a mismatch is a
# silent misalignment of the whole distribution.
FATAL_ATTRS = ("vocab_size",)
# Reported, never fatal. Nothing in rollout/ or teacher/ reads tokenizer.eos_token_id: the
# rollout appends the literal T.IM_END (rollout/rollout.py) and vLLM stops on
# stop=[T.SEARCH_CLOSE, T.ANSWER_CLOSE] (rollout/vllm_policy.py). Those are frozen template
# constants, and they are in the probe set above -- so their SEGMENTATION is what has to agree,
# and that is checked, rather than a metadata field neither arm looks at.
ADVISORY_ATTRS = ("eos_token_id", "bos_token_id", "pad_token_id")


def compare_tokenizers(teacher, student, probes) -> dict:
    """-> {"ok", "segmentation_ok", "n", "first_divergence", "attrs", "fatal_attrs"}.

    Takes OBJECTS, not paths, so the comparison is testable without two multi-gigabyte
    checkpoints on disk -- the CLI below is the only part that needs real files.

    TWO PROPERTIES, DEDUCED SEPARATELY, because they are not the same claim and the first
    version of this gate reported one and measured the other. It ANDed segmentation identity
    with equality of four special-token ids into a single `ok`, then printed the segmentation
    warning whenever `ok` was false. On the real pair that produced a FAIL whose message --
    "the teacher and the student segment text differently" -- was contradicted by the gate's
    own evidence: every probe matched, and the sole difference was eos_token_id (teacher 151645
    <|im_end|> from the Thinking lineage, student 151643 <|endoftext|> inherited from
    Qwen3-4B-Base). That is the shape of claim the project's conventions forbid: a conclusion written ahead of
    what the check can actually establish.

    Segmentation identity is load-bearing: OPD scores the student's exact ids
    position-by-position. Special-token METADATA is not, unless something reads it.
    """
    attrs = {}
    for name in FATAL_ATTRS + ADVISORY_ATTRS:
        a, b = getattr(teacher, name, None), getattr(student, name, None)
        attrs[name] = {"teacher": a, "student": b, "equal": a == b,
                       "fatal": name in FATAL_ATTRS}

    first = None
    for text in probes:
        ta = teacher.encode(text, add_special_tokens=False)
        sa = student.encode(text, add_special_tokens=False)
        if list(ta) != list(sa):
            first = {"text": text, "teacher_ids": list(ta), "student_ids": list(sa)}
            break

    bad_attrs = [n for n in FATAL_ATTRS if not attrs[n]["equal"]]
    return {"ok": first is None and not bad_attrs,
            "segmentation_ok": first is None,
            "n": len(probes), "first_divergence": first,
            "attrs": attrs, "fatal_attrs": bad_attrs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True, help="LOCAL teacher checkpoint path")
    ap.add_argument("--student", required=True, help="LOCAL student/pi_ref checkpoint path")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    import transformers

    print(f"[tok] transformers {transformers.__version__}")
    print(f"[tok] teacher: {args.teacher}")
    print(f"[tok] student: {args.student}")
    teacher = AutoTokenizer.from_pretrained(args.teacher)
    student = AutoTokenizer.from_pretrained(args.student)

    res = compare_tokenizers(teacher, student, probe_strings())
    for name, v in res["attrs"].items():
        if v["equal"]:
            flag = "ok  "
        elif v["fatal"]:
            flag = "FATAL"
        else:
            flag = "note"
        print(f"[tok] {flag} {name}: teacher={v['teacher']} student={v['student']}")

    advisory = [n for n, v in res["attrs"].items() if not v["equal"] and not v["fatal"]]
    if advisory:
        # Say WHY it is not fatal, by naming what would have to read it. A difference reported
        # without that is indistinguishable from one nobody thought about.
        print(f"[tok] the differing field(s) {advisory} are metadata this pipeline never "
              "reads: the rollout appends the literal T.IM_END (rollout/rollout.py) and vLLM "
              "stops on stop=[T.SEARCH_CLOSE, T.ANSWER_CLOSE] (rollout/vllm_policy.py). Those "
              "are frozen template constants and their segmentation IS checked above. A "
              "student SFT'd from Qwen3-4B-Base carrying eos 151643 <|endoftext|> while a "
              "Thinking-lineage teacher carries 151645 <|im_end|> is the expected shape of "
              "this pair, not a defect.")

    if res["ok"]:
        print(f"[tok] PASS: identical ids on all {res['n']} probes"
              + (f" (with {len(advisory)} advisory metadata difference(s) above)" if advisory
                 else " and on every consumed special-token id")
              + ". The pure-id contract's premise holds on this stack.")
        print("[tok] NOT PROVEN HERE: that the teacher's LOGPROBS align -- that is the "
              "determinism probe's job, and its verdict does not follow from this one.")
        return 0

    d = res["first_divergence"]
    if d:
        print(f"[tok] FAIL: divergent segmentation on {d['text']!r}", file=sys.stderr)
        print(f"[tok]   teacher: {d['teacher_ids']}", file=sys.stderr)
        print(f"[tok]   student: {d['student_ids']}", file=sys.stderr)
        print("[tok] The teacher scores the student's ids position-by-position, so every "
              "per-token advantage after this point is attached to the wrong token -- "
              "silently, with no exception and no shape mismatch. Do NOT run OPD on this pair.",
              file=sys.stderr)
    for name in res["fatal_attrs"]:
        v = res["attrs"][name]
        print(f"[tok] FAIL: {name} differs (teacher={v['teacher']} student={v['student']}). "
              "It sets the width of every logit row the teacher returns, so the whole "
              "distribution is misaligned rather than merely shifted.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
