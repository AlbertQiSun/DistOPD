#!/usr/bin/env bash
# Sourced by every sbatch template. Adjust module loads / venv path to your cluster.
# Kept in one place so all three jobs share an identical environment (gate-1 spirit:
# online and offline must run under the same stack).
# Strict mode ONLY for batch (sbatch) execution. In an interactive `srun --pty bash`
# session, `set -e` leaks into the caller's shell -- and if that shell has a stray
# PROMPT_COMMAND (e.g. VS Code shell integration's __vsc_prompt_cmd_original, which is
# undefined on the compute node), the failing prompt command trips errexit and kills the
# shell with exit 127. Skip strict mode when interactive so sourcing this file for manual
# test runs is safe; sbatch (non-interactive) still gets the strict behavior.
case $- in
  *i*) : ;;                 # interactive: do NOT enable errexit/nounset
  *)   set -euo pipefail ;; # batch: strict
esac

# module load cuda/12.4                 # REPLACE with your cluster's modules if needed

# --- SITE: where this cluster keeps things ------------------------------------------------
# A site says WHERE an asset lives; a channel says how much of a card the teacher may take.
# Different axes, different files, different safety rules -- see slurm/sites/README.md.
# Resolved from BASH_SOURCE so `source slurm/env.sh` works from any cwd.
_OPD_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OPD_SITE:=example}"               # default keeps every pre-migration command working
_OPD_SITE_FILE="$_OPD_SLURM_DIR/sites/${OPD_SITE}.sh"
if [ ! -f "$_OPD_SITE_FILE" ]; then
  echo "FATAL: unknown OPD_SITE=${OPD_SITE} -- have:" >&2
  ls "$_OPD_SLURM_DIR/sites/"*.sh 2>/dev/null | sed 's#.*/##;s#\.sh$##;s#^#  #' >&2
  return 1 2>/dev/null || exit 1
fi
. "$_OPD_SITE_FILE"
# Teacher endpoint derivation. Sourced AFTER the site file because it reads OPD_RUNSTATE_DIR
# and OPD_TEACHER_MODEL, both of which the site defines.
. "$_OPD_SLURM_DIR/lib_teacher_endpoint.sh"
echo "site: $OPD_SITE_NAME  (models=$OPD_MODELS_DIR  partition=$OPD_PARTITION  channel=$OPD_CHANNEL)"
# WHICH LINE. Three teachers and three pi_ref checkpoints share this repo, they are selected
# by environment variable, and a job's NAME is not evidence of what it ran. Printing the two
# resolved paths costs nothing and makes every log self-describing -- the log is the only
# place this is recoverable after the fact.
echo "line: teacher=$(basename "$OPD_TEACHER_MODEL")  student=$(basename "$OPD_STUDENT_CKPT")"
echo "      teacher_path=$OPD_TEACHER_MODEL"
echo "      student_path=$OPD_STUDENT_CKPT"

# THE CROSS-SITE GUARD: do this site's declared paths EXIST on the machine we are standing on?
#
# The hazard is a job that runs with the wrong site exported and reads paths that are not
# there -- or, worse, paths that are and hold something else.
#
# This was first written as `SLURM_JOB_PARTITION == OPD_PARTITION`, which was the wrong
# discriminator and inconveniently wrong in both directions. It REFUSES legitimate jobs: a
# CPU-only test run on cpu_short, an env build on <transfer-partition>, a GPU job on the A100 cluster's
# <short-partition>/<long-partition> (29 + 19 more A100 nodes than <a100-partition> alone). And it does not test
# the hazard: you can only submit to partitions the local cluster HAS, so a cross-site mistake
# shows up as a nonexistent partition long before it shows up as a mismatch -- while the real
# symptom, paths pointing nowhere, goes unexamined.
#
# Existence is the direct test. Neither cluster mounts the other's filesystem, so a swapped
# OPD_SITE fails here instead of at model load, and a site whose assets were never staged
# fails the same way for the same reason. The check itself lives in lib_site_guard.sh because
# slurm/setup_env.sh needs it too -- with a shorter list, since it runs before staging.
. "$_OPD_SLURM_DIR/lib_site_guard.sh"
# OPD_STUDENT_CKPT is deliberately NOT in this list -- it is a per-job input, not a
# staged site asset, and this guard runs in jobs that never load a student (the
# retriever, the teacher server). The jobs that DO load it call
# opd_require_student_ckpt; see lib_site_guard.sh.
opd_require_paths "$OPD_SITE_ROOT" "$OPD_MODELS_DIR" "$OPD_SFT_DATA_DIR" \
  || { return 1 2>/dev/null || exit 1; }
       # `return` when SOURCED: a bare `exit` kills the caller's login shell, which
       # is what a wrong OPD_SITE did here -- the default is h20-site, so sourcing
       # this on the A100 cluster without exporting OPD_SITE failed the path check and closed
       # the terminal. Same idiom as the unknown-site branch above: a wrong site must
       # be reportable, not fatal to the session.

# The partition is REPORTED, not enforced -- it belongs in the log next to the site so a
# surprising queue wait or a wrong-card result can be traced, but it is not evidence of a
# mistake. See above for why it was demoted from a guard.
[ -n "${SLURM_JOB_PARTITION:-}" ] && echo "partition: $SLURM_JOB_PARTITION (site default: $OPD_PARTITION)"

# conda env (same stack for retriever + online + offline -- gate-1 spirit)
source "$(conda info --base)/etc/profile.d/conda.sh"
# The site file supplies OPD_CONDA_ENV, and a caller may override it so a second hardware
# channel can exist beside the first (the A800's CUDA 12.2 driver cannot run the cu130 stack
# at all -- a MAJOR version gap). Which stack each channel gets, and why those exact versions,
# is in slurm/setup_env.sh; it is not repeated here, because two copies of a version table
# drift and the drift is what makes a channel behave differently for no recorded reason.
#
# ONE CHANNEL PER EXPERIMENT: a run's precompute and BOTH its arms must use the same env on
# the same hardware. See the A800 section of README.md for why a partial move is
# not allowed. The env name is recorded in every artifact through BackendConfig's runtime
# fields, so a cross-channel mix is refused rather than merely discouraged.
# WHAT ACTUALLY ACTIVATED, not what we asked for.
#
# `conda activate <missing>` prints EnvironmentNameNotFound and leaves the previous env in
# place -- and it does not trip `set -e`. So a wrong or absent env is silent here, and the run
# proceeds on whatever was already active (typically `base`). Observed: a job kept going on
# base's python and only failed later at `No module named pytest`, which sends you looking at
# the test runner instead of at the activate.
#
# CHECKING FOR A PACKAGE IS NOT ENOUGH. `python -c "import torch"` passes on any env that
# happens to have torch, and a shared base usually does -- so it certifies the wrong
# environment as the right one. The only thing that answers "is the requested env active" is
# CONDA_PREFIX. Factored into a function so tests/test_site_config.py can exercise the
# comparison without a conda installation.
opd_assert_env_active() {                # $1 = requested (name or path), $2 = CONDA_PREFIX
  local want="$1" prefix="$2"
  [ -n "$prefix" ] || { echo "FATAL: no conda environment is active" >&2; return 1; }
  case "$want" in
    /*) [ "$prefix" = "$want" ] || {
          echo "FATAL: asked for env '$want' but '$prefix' is active." >&2; return 1; } ;;
    *)  [ "$(basename "$prefix")" = "$want" ] || {
          echo "FATAL: asked for env '$want' but '$prefix' is active." >&2; return 1; } ;;
  esac
}

conda activate "${OPD_CONDA_ENV:-OPD}"
opd_assert_env_active "${OPD_CONDA_ENV:-OPD}" "${CONDA_PREFIX:-}" || {
  echo "       conda activate does NOT fail loudly -- it leaves the previous env in place." >&2
  # THE THIRD STALE-RESIDUE INCIDENT (after the site-selector overwrite and the pi_ref_moe
  # baseline): --export=ALL carries the submitting shell's OPD_* exports into the job, and
  # the site file's :- default DEFERS to them. Jobs 26866469/26866471 died in 3 s because a
  # login shell had once run `source slurm/env.sh` WITHOUT OPD_SITE (default: h20-site),
  # which exported OPD_CONDA_ENV=OPD -- the H20 cluster's env NAME -- and that residue outlived the
  # path-guard failure that followed. Name the suspect so the 3-second log convicts it.
  echo "       OPD_CONDA_ENV='${OPD_CONDA_ENV:-<unset>}' -- if you did not export this" >&2
  echo "       deliberately, your submitting shell carries another site's residue" >&2
  echo "       (\`source slurm/env.sh\` without OPD_SITE defaults to example and exports" >&2
  echo "       its vocabulary). Resubmit from a fresh login shell (exec bash -l)." >&2
  echo "       Build this site's env on the LOGIN node (compute nodes have no network):" >&2
  echo "         OPD_SITE=$OPD_SITE_NAME bash slurm/setup_env.sh" >&2
  return 1 2>/dev/null || exit 1        # sourced: report, do not kill the shell
}
echo "conda env: ${OPD_CONDA_ENV:-OPD}  (CONDA_PREFIX=$CONDA_PREFIX)"

# HF_HOME comes from the site file (always on scratch, never the home quota) -- weights are
# read from disk, never re-downloaded, and compute nodes are offline anyway.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# Allocator: expandable segments. The offline OOM report showed 4.36 GiB "reserved but
# unallocated" -- fragmentation, not demand. Trajectory lengths here span 708..13750 tokens,
# so every step asks the allocator for a different set of block sizes, which is precisely the
# pattern that strands memory in the caching allocator's fixed-size pools. Costs nothing.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# vLLM FlashInfer sampler OFF -- the cached flashinfer sampling.so links libcudart.so.12,
# but the OPD env is torch 2.11+cu130 / CUDA 13, so loading it crashes the EngineCore
# ("Failed to load .../flashinfer/.../sampling.so: libcudart.so.12: cannot open shared object
# file"). Native torch sampling is the fallback -- NO numeric impact (FlashInfer is a kernel
# optimization, not a different algorithm), so gate-1 parity is untouched. Baked in here so
# retriever + online + offline all inherit the same fix.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "python: $(command -v python)"
python --version

# --- service readiness gates -------------------------------------------------------------
# wait_for_teacher / wait_for_retriever / start_private_teacher / stop_private_teacher MOVED to
# slurm/lib_teacher_endpoint.sh (sourced just above). They are unchanged for every caller --
# sourcing this file still defines all four. The move exists so they can be EXERCISED by a
# test: tests/test_teacher_endpoint_derivation.py sources the lib in a clean shell, and it
# cannot source this file (conda activate, the site-path guard). While they lived here the
# poll loop was untestable, and it shipped a bug that cost 40 minutes of two A100s: the
# private teacher's process died in its first minute and wait_for_teacher went on polling to
# its 2400 s timeout, because nothing was watching the PID (job 27001269, 2026-09-03).
