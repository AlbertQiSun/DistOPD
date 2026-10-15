"""Which pi_ref a job loads is a per-job input, and the job -- not the site -- must demand it.

THE SITUATION THIS ENCODES. Until the three-line comparison there was one pi_ref, so
slurm/env.sh could list $OPD_STUDENT_CKPT in the cross-site existence guard and be right.
Now there are three, all the same 4B student, structurally indistinguishable on disk, and the
site can only supply a default. A default is exactly how a run silently rolls out the
PREVIOUS round's checkpoint: it loads cleanly, every gate passes, and the only evidence is a
path in a log nobody printed.

That is not hypothetical. The first miro precompute smoke was submitted with
OPD_STUDENT_CKPT=.../pi_ref_miro and the site file replaced it with .../pi_ref_ckpt -- the
2026-07-23 checkpoint, trained under the PREVIOUS prompt (f9da237f476eb71e, v2_terse) while
the rest of the run was v3 (3165062cda7688de). Nothing could have caught it downstream: the
template fingerprint written into the manifest is computed from the CURRENT process's
SYSTEM_PROMPT, not from the checkpoint, so the artifact would have claimed v3 in good faith.

TWO SEPARATE CONCERNS, now separated:

  opd_require_paths          "are this SITE's staged assets on this filesystem?"  -- every job
  opd_require_student_ckpt   "did THIS job name the checkpoint it means?"         -- only jobs
                                                                                    that load one

Keeping the student in the first one forced the retriever and the teacher server -- which
never load a student -- to name a checkpoint they do not use, and the pressure that creates
is to name whichever path happens to exist. Hence the split, and hence
test_jobs_that_do_not_load_a_student_do_not_demand_one.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

LIB = "slurm/lib_site_guard.sh"

# Jobs whose whole purpose involves rolling out or training the student.
STUDENT_JOBS = ["slurm/precompute_teacher.sbatch", "slurm/train_offline.sbatch",
                "slurm/train_online.sbatch"]
# Jobs that never load one. stage3_eval is absent from BOTH lists on purpose: it evaluates a
# model named by --model / OPD_PI_REF and does its own `test -d` on it.
NO_STUDENT_JOBS = ["slurm/retriever_wiki18.sbatch", "slurm/teacher_server.sbatch"]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="the guard is shell")


def _guard(ckpt, unset=False):
    env = dict(os.environ)
    env.pop("OPD_STUDENT_CKPT", None)
    if not unset:
        env["OPD_STUDENT_CKPT"] = ckpt
    p = subprocess.run(["bash", "-c", f". {LIB}; opd_require_student_ckpt"],
                       capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


# --------------------------------------------------------------------------- behaviour
def test_an_existing_directory_passes_and_says_which_one(tmp_path):
    d = tmp_path / "pi_ref_miro"
    d.mkdir()
    rc, out, err = _guard(str(d))
    assert rc == 0, err
    assert str(d) in out, "the log must record WHICH checkpoint, not just that one was found"


def test_a_missing_directory_refuses(tmp_path):
    rc, _, err = _guard(str(tmp_path / "pi_ref_that_was_renamed_away"))
    assert rc != 0
    assert "not a directory" in err


def test_an_unset_variable_refuses(tmp_path):
    rc, _, err = _guard("", unset=True)
    assert rc != 0
    assert "<unset>" in err


def test_an_empty_variable_refuses():
    """Distinct from unset: `OPD_STUDENT_CKPT= bash ...` is a plausible typo, and `[ -d "" ]`
    is false, so this would refuse anyway -- but only if the emptiness is checked BEFORE the
    path is used to build a message."""
    rc, _, err = _guard("")
    assert rc != 0


def test_a_FILE_is_not_a_checkpoint(tmp_path):
    """-d, not -e. A stray file with the right name would satisfy an existence test and then
    fail deep inside vLLM's loader with a message about safetensors."""
    f = tmp_path / "pi_ref_miro"
    f.write_text("not a checkpoint", encoding="utf-8")
    rc, _, err = _guard(str(f))
    assert rc != 0


def test_the_refusal_tells_the_operator_what_to_type(tmp_path):
    rc, _, err = _guard(str(tmp_path / "gone"))
    assert "OPD_STUDENT_CKPT=" in err and "pi_ref_" in err
    assert "SUBMIT time" in err, "the fix is at submission, not in the config"


# --------------------------------------------------------------------------- placement
def test_the_site_guard_no_longer_demands_the_student():
    """If it did, renaming the stale default away would refuse the retriever and the teacher
    server too -- jobs that have no student and no business naming one."""
    code = "\n".join(l for l in open("slurm/env.sh", encoding="utf-8").read().splitlines()
                     if not l.lstrip().startswith("#"))
    line = [l for l in code.splitlines() if "opd_require_paths" in l]
    assert line, "env.sh no longer calls opd_require_paths at all"
    assert "OPD_STUDENT_CKPT" not in " ".join(line)


@pytest.mark.parametrize("path", STUDENT_JOBS)
def test_a_job_that_loads_the_student_demands_it(path):
    code = "\n".join(l for l in open(path, encoding="utf-8").read().splitlines()
                     if not l.lstrip().startswith("#"))
    assert "opd_require_student_ckpt" in code, (
        f"{path} loads the student but never checks which one it was given")


@pytest.mark.parametrize("path", NO_STUDENT_JOBS)
def test_jobs_that_do_not_load_a_student_do_not_demand_one(path):
    code = "\n".join(l for l in open(path, encoding="utf-8").read().splitlines()
                     if not l.lstrip().startswith("#"))
    assert "opd_require_student_ckpt" not in code, (
        f"{path} never loads a student; requiring one pushes the operator to name any path "
        "that exists, which is the habit this whole guard is trying to break")


@pytest.mark.parametrize("path", STUDENT_JOBS)
def test_the_demand_comes_before_the_expensive_part(path):
    """A guard that fires after the engine is up has spent the queue wait and the weight
    load already. It must precede the python entry point."""
    src = open(path, encoding="utf-8").read()
    guard = src.index("opd_require_student_ckpt")
    run = src.index("python scripts/train.py")
    assert guard < run, f"{path} checks the student AFTER launching training"


def test_the_shell_files_parse():
    for path in [LIB, "slurm/env.sh"] + STUDENT_JOBS + NO_STUDENT_JOBS:
        p = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert p.returncode == 0, f"{path}: {p.stderr}"
