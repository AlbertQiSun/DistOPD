"""Load a run config, resolving ${SITE_VARIABLE} against the site layer.

WHY THIS EXISTS. `configs/opd_qwen3.yaml` carried six absolute the H20 cluster paths
(`/path/to/scratch/...`). The site layer (slurm/sites/*.sh) already moved the SHELL
side of the project between clusters, but nothing resolved the YAML side, so on the A100 cluster the
first thing to read `model.student` got a path that does not exist there.

Certify gate 6 is where that landed, and the shape of the failure is the reason this module
refuses rather than substitutes. transformers saw a non-existent path, decided it must be a
Hub repo id, and raised

    HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'

The gate then printed its configured meaning -- "the student does not fit this card at the
real trajectory lengths" -- which was FALSE. Nothing had been measured; the load never began.
A wrong verdict stated confidently is worse than a missing one, so:

  * an unset variable RAISES, naming the variable and the site files. It is never left in the
    string, because `${OPD_STUDENT_CKPT}/config.json` is exactly the kind of value that looks
    like a path to every check downstream and is not one.
  * a bare `$VAR` (no braces) is refused too. It is a typo for the supported form, and the
    alternative -- treating it as literal text -- reproduces the same class of silent
    not-a-path.

WHAT THIS DELIBERATELY DOES NOT DO: check that the resolved path exists. Configs name outputs
as well as inputs, and a loader that stats everything would refuse to write `runs/` on a fresh
checkout. Existence is the caller's business at the point of use -- see `require_input_asset`,
which is for the assets a run READS.
"""
from __future__ import annotations

import os
import pathlib
import re
from typing import Any

import yaml

# ${NAME} only. NAME is the shell's own identifier shape, so anything a site file can export
# can be referenced and nothing else can.
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# A `$` that starts something identifier-shaped but is NOT `${`. Caught, not passed through.
_BARE = re.compile(r"\$(?!\{)[A-Za-z_]")

SITES_HINT = ("slurm/sites/*.sh defines the site vocabulary; `. slurm/env.sh` exports it. "
              "Run through slurm/submit.sh, or source it, before reading a config.")


def expand(value: str, env: dict | None = None, *, where: str = "") -> str:
    """Substitute every ${VAR} in `value`, or raise naming what is missing."""
    env = os.environ if env is None else env
    at = f" (at {where})" if where else ""

    bare = _BARE.search(value)
    if bare:
        raise ValueError(
            f"config uses the unsupported bare-$ form{at}: {value!r}. Write ${{NAME}} with "
            f"braces -- the bare form would be left as literal text, producing a string that "
            f"looks like a path to everything downstream and resolves to nothing.")

    missing = [n for n in _VAR.findall(value) if not env.get(n)]
    if missing:
        raise KeyError(
            f"config references site variable(s) {missing}{at} but they are unset or empty. "
            f"{SITES_HINT}")
    return _VAR.sub(lambda m: env[m.group(1)], value)


def expand_tree(obj: Any, env: dict | None = None, *, where: str = "") -> Any:
    """Recursively expand every string in a loaded YAML document."""
    if isinstance(obj, str):
        return expand(obj, env, where=where)
    if isinstance(obj, dict):
        return {k: expand_tree(v, env, where=f"{where}.{k}" if where else str(k))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_tree(v, env, where=f"{where}[{i}]") for i, v in enumerate(obj)]
    return obj


def load_config(path: str | os.PathLike, env: dict | None = None) -> dict:
    """`yaml.safe_load` + site-variable resolution. The ONE way a run config is read."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(text) or {}
    return expand_tree(doc, env, where=os.fspath(path))


def require_input_asset(path: str, what: str) -> str:
    """Assert an asset a run READS is actually here, and say which site was consulted.

    The point is the message. Without it the first reader decides what a missing path means --
    transformers guesses "Hub repo id", pandas guesses "no such file", vLLM guesses something
    else again -- and none of those name the real cause, which is a config resolved against
    the wrong site or a transfer that never finished.
    """
    if os.path.exists(path):
        return path
    site = os.environ.get("OPD_SITE_NAME", "<no OPD_SITE_NAME in the environment>")
    raise FileNotFoundError(
        f"{what} does not exist at {path!r} (site={site}). Either the config resolved against "
        f"a different site than the one holding this asset, or the asset was never staged "
        f"here. {SITES_HINT}")
