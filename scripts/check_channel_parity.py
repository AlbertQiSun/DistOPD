#!/usr/bin/env python3
"""Did both arms of this comparison run in the SAME execution channel?

    python scripts/check_channel_parity.py runs/diag_online.jsonl runs/diag_offline.jsonl \
        --manifest precomputed/manifest.json

A "channel" is (GPU model, compute capability, torch build, vLLM build). The plan is to keep
more than one open so work can go wherever the queue is shorter, and the hard rule attached
to that plan is:

    ONE experiment's precompute and BOTH its training arms stay inside ONE channel.

Cross-channel is fine. Cross-ARM is not: an A800-scored teacher logprob and an H20-scored one
are not the same number, and the entire claim of this phase is that online and offline differ
in exactly one thing -- where the trajectory came from. A hardware split would be a second
difference, sitting in the same place as the effect, and it would look like a result.

NO OTHER CHECK CAN MAKE THIS ONE. Gate 1 fires inside a single process against a single
cache; it never sees the other arm. The two arms are separate SLURM jobs that share no
runtime state -- the first moment both are visible is when someone sits down to compare their
diagnostics, which is here. Run it BEFORE reading any online-vs-offline number.

Exit 0 = same channel throughout. Exit 1 = do not compare these two runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

FIELDS = ("gpu_name", "gpu_sm", "torch_version", "vllm_version")

# The files whose change means "different numerics": everything the TRAINING process
# imports or reads as configuration. Launcher-level differences (sbatch env vars, CLI
# shapes, site vocabulary) are deliberately NOT here -- not because they are harmless,
# but because every one of them is independently stamped per record and independently
# compared by this very script (runtime fingerprint, channel, cuda_alloc_conf, step
# shape in the rows themselves). code_version equivalence only has to cover what no
# other stamp sees: the repo's own executable code and configs.
NUMERICS_SURFACE = ("opd/", "teacher/", "rollout/", "envs/", "dataio/", "configs/")
NUMERICS_FILES = ("scripts/train.py",)


def _git_diff_files(a: str, b: str) -> list[str]:
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.check_output(["git", "diff", "--name-only", a, b],
                                  cwd=root, text=True, stderr=subprocess.PIPE)
    return [line.strip() for line in out.splitlines() if line.strip()]


def verify_code_version_equivalence(a: str, b: str):
    """(ok, touched_files, numerics_files_touched | reason).

    A special-case green for two code_versions is EARNED, never decreed: the gate runs
    the diff itself and accepts only if nothing on the numerics surface moved. No git,
    no commits, no verification -> unverifiable claim -> refused, same doctrine as an
    unstamped artifact."""
    try:
        files = _git_diff_files(a, b)
    except Exception as e:                 # noqa: BLE001 - no git IS the finding here
        return False, [], f"git could not verify {a}..{b} ({e}); an unverifiable equivalence claim is refused"
    touched = [f for f in files
               if f in NUMERICS_FILES or any(f.startswith(p) for p in NUMERICS_SURFACE)]
    return (not touched), files, touched


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def distinct_runtimes(path, key="runtime"):
    """Every distinct runtime stamp in a diag file, with how many rows carried it.

    Plural is not impossible and is worth surfacing: a resumed run whose second half landed
    on a different node would show two, and that is a split INSIDE one arm.
    """
    seen = {}
    missing = 0
    for rec in rows(path):
        rt = rec.get(key)
        if rt is None:
            missing += 1
            continue
        k = tuple(rt.get(f) for f in FIELDS)
        seen[k] = seen.get(k, 0) + 1
    return seen, missing


def fmt(key) -> str:
    return "  ".join(f"{f}={v!r}" for f, v in zip(FIELDS, key))


def main() -> int:
    ap = argparse.ArgumentParser(description="Both arms in one execution channel?")
    ap.add_argument("diags", nargs="+", help="runs/diag_*.jsonl, one per arm")
    ap.add_argument("--manifest", default=None,
                    help="precomputed/manifest.json -- adds the SAMPLER and TEACHER machines")
    ap.add_argument("--equate-code-versions", default=None, metavar="A,B",
                    help="Treat two code_version stamps as one -- ONLY after this script "
                         "verifies via `git diff A B` that nothing on the numerics surface "
                         "(opd/ teacher/ rollout/ envs/ dataio/ configs/ scripts/train.py) "
                         "changed between them. Launcher/analysis-file diffs are acceptable "
                         "because every launcher-level quantity is independently stamped and "
                         "compared. An unverifiable claim (no git, unknown commits) is refused.")
    args = ap.parse_args()

    problems, observed = [], {}

    equate = {}
    if args.equate_code_versions:
        try:
            va, vb = [s.strip() for s in args.equate_code_versions.split(",")]
        except ValueError:
            print("--equate-code-versions wants exactly two comma-separated hashes")
            return 1
        ok, files, touched = verify_code_version_equivalence(va, vb)
        print(f"=== --equate-code-versions {va},{vb} ===")
        if isinstance(touched, str):                    # git itself failed
            print(f"  REFUSED: {touched}")
            return 1
        for f in files:
            print(f"  diff touches: {f}")
        if not ok:
            print(f"  REFUSED: the diff moves the NUMERICS SURFACE ({', '.join(touched)}) -- "
                  "these are genuinely different numerics, not a launcher-level difference.")
            return 1
        print(f"  ACCEPTED: nothing on the numerics surface changed between {va} and {vb}; "
              "every launcher-level difference is covered by the per-row stamps compared "
              "below. The two stamps are treated as one version.")
        equate = {vb: va}

    code_versions = {}
    for path in args.diags:
        seen, missing = distinct_runtimes(path)
        print(f"\n=== {path} ===")
        if missing:
            print(f"  {missing} row(s) carry NO runtime stamp -- written before the field "
                  "existed. This file cannot prove which channel it ran in.")
            problems.append(f"{path}: {missing} unstamped rows")
        for key, n in seen.items():
            print(f"  {n:>6} rows  {fmt(key)}")
            observed.setdefault(key, []).append(f"{path}(train)")
        if len(seen) > 1:
            problems.append(f"{path}: {len(seen)} DIFFERENT runtimes inside ONE arm")

        cache, _ = distinct_runtimes(path, key="cache_sampler_runtime")
        for key, n in cache.items():
            print(f"  {n:>6} rows  [cache sampled on] {fmt(key)}")
            observed.setdefault(key, []).append(f"{path}(cache sampler)")

        # CODE VERSION is part of the channel since the chunked-lm_head change
        # (2026-08-19): the fused path's numbers differ from the full-logits era by
        # ~1 ulp (opd/masking.py), so arms on different commits trained under different
        # kernels. Same shape as the hardware split, one level up.
        vers = {rec.get("code_version") for rec in rows(path)}
        vers = {equate.get(v, v) for v in vers}          # verified-equivalent -> one token
        for v in sorted(vers, key=str):
            code_versions.setdefault(v, []).append(path)
        if None in vers:
            problems.append(f"{path}: rows without a code_version stamp (pre-stamp code)")
        print(f"  code_version(s): {sorted((str(v) for v in vers))}")

    if len(code_versions) > 1:
        problems.append(
            "code_version differs across arms: "
            + "; ".join(f"{v} <- {', '.join(sorted(set(ps)))}"
                        for v, ps in code_versions.items())
            + " -- the arms ran DIFFERENT numerics (chunked-lm_head era vs before, or "
              "any later numerics-touching commit). Do not compare.")

    if args.manifest:
        man = json.load(open(args.manifest, encoding="utf-8"))
        print(f"\n=== {args.manifest} ===")
        for label, block in (("sampler", man.get("sampler")),
                             ("teacher", man.get("backend"))):
            if not block:
                print(f"  {label}: NOT RECORDED -- a cache from before this field existed")
                problems.append(f"{args.manifest}: no {label} runtime")
                continue
            key = tuple(block.get(f) for f in FIELDS)
            if all(v is None for v in key):
                print(f"  {label}: all runtime fields empty -- pre-dates the field")
                problems.append(f"{args.manifest}: {label} runtime empty")
                continue
            print(f"  {label}: {fmt(key)}")
            observed.setdefault(key, []).append(f"{args.manifest}({label})")

    print("\n" + "=" * 78)
    if len(observed) > 1:
        print("FAIL: more than one execution channel is involved.\n")
        for key, where in observed.items():
            print(f"  {fmt(key)}\n      from: {', '.join(where)}")
        print("\nDo NOT compare these runs. An online-vs-offline difference measured across "
              "a hardware or stack boundary contains that boundary, and it sits in exactly "
              "the place the effect would.")
        return 1
    if problems:
        print("INCOMPLETE: one channel among what IS recorded, but the record has holes:\n")
        for p in problems:
            print(f"  - {p}")
        print("\nAn unstamped artifact is not evidence of a match. Regenerate it, or accept "
              "the comparison on grounds other than this check.")
        return 1
    if not observed:
        print("FAIL: nothing carried a runtime stamp at all.")
        return 1
    print(f"OK: one channel throughout -- {fmt(next(iter(observed)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
