"""Which teacher and which pi_ref a job runs must be selectable at submit time.

THE FAILURE THIS ENCODES. slurm/sites/*.sh assigned the two per-line selectors
unconditionally:

    export OPD_TEACHER_MODEL="$OPD_MODELS_DIR/MiroThinker-1.7-mini"
    export OPD_STUDENT_CKPT="$OPD_MODELS_DIR/pi_ref_ckpt"

Correct while one teacher and one pi_ref existed. The three-line comparison selects both by
environment variable at submit time, and `source slurm/env.sh` -- which every sbatch does
first -- overwrote whatever the operator exported.

WHY THAT IS THE DANGEROUS KIND. It is silent, and it defeats guards that look independent.
Every downstream check compares two values that BOTH derive from these variables: the served
teacher's --model against cfg["model"]["teacher"] (== ${OPD_TEACHER_MODEL}), and the derived
host file against the derived host file. Overwriting the variable moves both sides together,
so TeacherHTTPScorer._verify_backend passes, the endpoint derivation agrees with itself, and
a job submitted as the "moe" line runs MiroThinker end to end with every gate green. The job
NAME is not evidence of what ran.

The fix is the `${VAR:-default}` form OPD_CONDA_ENV two lines above already used: the site
supplies a default for a submission that names neither, not a value for every submission.

WHAT THESE TESTS DO. Source each site file in a subshell, with and without the variables
preset, and read back what survived. That is the same operation an sbatch performs, so the
test exercises the mechanism rather than a description of it.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess

import pytest

# Forward slashes even on Windows: these paths are handed to `bash -c ". <path>"`, and a
# backslash there is an ESCAPE, not a separator -- the source silently no-ops and every
# assertion about defaults reads an empty variable.
SITES = sorted(p.replace("\\", "/") for p in glob.glob("slurm/sites/*.sh"))
SELECTORS = ["OPD_TEACHER_MODEL", "OPD_STUDENT_CKPT"]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _source(site: str, **preset) -> dict:
    """Source a site file the way an sbatch does and report the selectors it left behind."""
    env = dict(os.environ)
    for k in SELECTORS:
        env.pop(k, None)
    env.update(preset)
    script = ". " + site + " >/dev/null 2>&1; " + "; ".join(
        f'echo "{k}=${k}"' for k in SELECTORS)
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    return dict(line.split("=", 1) for line in p.stdout.strip().splitlines())


assert SITES, "no site files found -- this test's premise is gone"


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("var", SELECTORS)
def test_an_exported_selector_survives_the_site_file(site, var):
    """THE regression. Without this, a `-p moe` submission runs miro and says PASS."""
    chosen = f"/chosen/by/the/operator/{var.lower()}"
    got = _source(site, **{var: chosen})
    assert got[var] == chosen, (
        f"{site} overwrote {var}. Use ${{{var}:-<default>}}: the site supplies a default for "
        "a submission that names neither, not a value for every submission.")


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("var", SELECTORS)
def test_an_unset_selector_still_gets_the_site_default(site, var):
    """The escape hatch must not become a requirement -- every pre-three-line command line
    omits both variables and must keep working."""
    got = _source(site)
    assert got[var], f"{site} left {var} empty when nothing was exported"
    assert got[var].startswith("/"), f"{site} {var} is not an absolute path: {got[var]!r}"


@pytest.mark.parametrize("site", SITES)
def test_the_two_selectors_are_independent(site):
    """Setting one must not disturb the other; a shared code path that resolved both from a
    single variable would pass every test above and still cross the lines."""
    got = _source(site, OPD_TEACHER_MODEL="/only/the/teacher")
    default = _source(site)
    assert got["OPD_TEACHER_MODEL"] == "/only/the/teacher"
    assert got["OPD_STUDENT_CKPT"] == default["OPD_STUDENT_CKPT"]


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("var", SELECTORS)
def test_the_assignment_is_written_in_the_overridable_form(site, var):
    """Belt to the braces above: read the source. A future edit that reverts to the bare
    form is caught here even if someone changes how the tests source the file."""
    src = open(site, encoding="utf-8").read()
    m = re.search(rf'^export {var}=(.+)$', src, re.M)
    assert m, f"{site} no longer assigns {var}"
    assert m.group(1).startswith(f'"${{{var}:-'), (
        f'{site} assigns {var} unconditionally: {m.group(1)}')


def test_env_sh_prints_the_line_it_resolved():
    """PERSIST rule, applied to the log. After the fact the log is the only record of which
    teacher and which pi_ref a job actually used -- the job name is not evidence.

    ANCHORED AT START OF LINE, not a substring search. The first version of this test used
    `'echo "line: teacher=' in src`, and commenting the echo out left that text in the file
    -- the mutation ran GREEN. A test whose trigger survives its own mutation is decoration.
    """
    src = open("slurm/env.sh", encoding="utf-8").read()
    for pat in (r'^echo "line: teacher=\$\(basename "\$OPD_TEACHER_MODEL"\)',
                r'^\s*echo "\s*teacher_path=\$OPD_TEACHER_MODEL"',
                r'^\s*echo "\s*student_path=\$OPD_STUDENT_CKPT"'):
        assert re.search(pat, src, re.M), f"env.sh no longer emits {pat!r} at line start"
