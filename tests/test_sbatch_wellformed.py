"""The sbatch files must parse into the argv they look like they mean.

WHY THIS EXISTS. Editing these files programmatically kept turning an intended line
continuation into the two-character sequence backslash-n, which bash reads as an escaped
`n` -- so

    python scripts/train.py --mode offline \\n  --limit 64

becomes `python scripts/train.py --mode offline n --limit 64`, passing a stray positional
argument. `bash -n` accepts it (the syntax is valid), the file looks right at a glance, and
the failure surfaces as an argparse error after the queue wait. It happened three times in
one session, to three different files.

So this checks the two things `bash -n` cannot: that no line carries a literal backslash-n,
and that every train.py invocation parses into flags and values with no stray positionals.
"""
from __future__ import annotations

import pathlib
import re
import shlex

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SBATCH = sorted((ROOT / "slurm").glob("*.sbatch"))

# values that legitimately appear as positionals or flag arguments
ALLOWED = {"online", "offline", "runs", "precomputed"}


def test_there_are_sbatch_files_to_check():
    assert SBATCH, "no sbatch files found -- this test would be vacuous"


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_no_literal_backslash_n(path):
    """A line continuation that got written as the characters \\ and n."""
    offenders = [
        (i, line) for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        # a real continuation is a backslash at END of line; anything else followed by 'n'
        # mid-line is the corruption, unless it is inside an echo/printf format string
        if re.search(r"\\n(?!\")", line) and not re.match(r"\s*(echo|printf)\b", line.strip())
    ]
    assert not offenders, (
        f"{path.name} has a literal backslash-n where a line continuation was meant:\n" +
        "\n".join(f"  line {i}: {l.strip()[:100]}" for i, l in offenders))


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_train_py_invocations_have_no_stray_positionals(path):
    """Joins continuations and parses the command, which is what bash will actually do."""
    text = re.sub(r"\\\n\s*", " ", path.read_text(encoding="utf-8"))
    for line in text.splitlines():
        line = line.strip()
        if not re.match(r"(\w+=\S+\s+)*python\d?\s+scripts/train\.py\b", line):
            continue
        toks = shlex.split(line, comments=True)
        argv = toks[toks.index("scripts/train.py") + 1:]
        i, stray = 0, []
        while i < len(argv):
            t = argv[i]
            if t.startswith("--"):
                # flag; its value (if any) is the next token when that is not another flag
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    i += 2
                else:
                    i += 1
                continue
            stray.append(t)
            i += 1
        assert not stray, (
            f"{path.name}: stray positional argument(s) {stray} in\n  {line[:140]}\n"
            "A lone 'n' here is the backslash-n corruption; train.py takes no positionals.")


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_every_line_continuation_is_followed_by_something(path):
    """A trailing backslash on the last line of a block silently swallows the next command."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.rstrip().endswith("\\"):
            assert i + 1 < len(lines), f"{path.name}: file ends on a line continuation"
            assert lines[i + 1].strip(), (
                f"{path.name} line {i + 1}: continuation followed by a blank line")
