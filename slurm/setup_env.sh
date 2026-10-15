#!/usr/bin/env bash
# Build this site's conda environment. WHERE it may run is a site fact -- see the guard below.
#
#   OPD_SITE=example bash slurm/setup_env.sh                 # login node (no network partition)
#   OPD_SITE=example  ... --wrap 'bash slurm/setup_env.sh'    # <transfer-partition> (the script prints it)
#
# EVERY FETCH HAPPENS WHERE THE SITE PUTS THE NETWORK, into a path on shared scratch that the
# GPU nodes can then read. On the H20 cluster that is the login node, because no batch partition has
# outbound network. On the A100 cluster it is <transfer-partition>, because the login node's reaper SIGKILLs long
# downloads -- it killed a 423 MB cublas wheel halfway through the first attempt.
#
# Re-running is safe and cheap: an existing env is continued rather than recreated, and pip's
# cache lives on scratch, so a killed run costs the one wheel it died on.
#
# ONE SCRIPT FOR BOTH STACKS, because two copies of an install lifecycle drift and the drift
# is invisible until a channel behaves differently for a reason nobody wrote down. The stack
# is chosen by DRIVER CUDA MAJOR VERSION, which is a property of the nodes, not a preference:
#
#   stack   driver     torch          vllm     chosen for
#   cu130   CUDA 13.0  2.11.0+cu130   0.25.1   H20 (<h20-partition>), A100 80GB (the A100 cluster <a100-partition>)
#   cu128   CUDA 12.2  2.9.0+cu128    0.11.2   A800 (<h20-partition> <node>)
#
# WHY THOSE EXACT vLLM VERSIONS -- read off the published wheels rather than guessed:
#
#   vllm      batch_invariant?   torch      compiled against
#   0.11.0    NO                 2.8.0      libcudart.so.12
#   0.11.1    yes                2.9.0      libcudart.so.12
#   0.11.2    yes                2.9.0      libcudart.so.12   <-- newest CUDA-12 build with it
#   0.21.0    yes                2.11.0     libcudart.so.13
#   0.25.1    yes                2.11.0     libcudart.so.13   <-- the cu130 stack
#
# 0.12.0 through 0.20.2 publish no wheel for this platform, so 0.11.2 is the newest release
# that both ships VLLM_BATCH_INVARIANT (a hard requirement -- without it batch composition
# changes sampled tokens 100% of the time at temperature 0) and links a CUDA 12 runtime. Its
# module path is IDENTICAL to 0.25.1's -- vllm/model_executor/layers/batch_invariant.py -- so
# scripts/policy_invariance_probe.py imports the same name either way.
#
# CUDA 13 -> 12.2 is a MAJOR version gap that minor-version compatibility does NOT bridge
# (measured on <node>: gpu_preflight --compute fails to initialise with the cu130 build). The
# cluster's forward-compatibility libraries only reach 12.6.
#
# NOTHING HERE IS CERTIFIED BY A SUCCESSFUL INSTALL. An import proves the wheels resolve; it
# says nothing about batch invariance, weight hot-update, or the numbers. That is what
# slurm/certify_channel.sbatch is for.

set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OPD_SITE:=example}"
_SITE_FILE="$_HERE/sites/${OPD_SITE}.sh"
[ -f "$_SITE_FILE" ] || { echo "FATAL: unknown OPD_SITE=$OPD_SITE" >&2; exit 1; }
# shellcheck source=/dev/null
. "$_SITE_FILE"

# --- IS THIS SITE'S TREE EVEN HERE ---------------------------------------------------------
# ROOT ONLY, not the assets: building the env before staging pi_ref is legitimate, and on a
# fresh cluster it is the normal order. Must come BEFORE the mkdir below -- without it, a
# wrong OPD_SITE walks into `mkdir -p /path/to/scratch/...` and reports "Permission denied" at the
# filesystem root, which names neither the site nor the mistake.
# shellcheck source=/dev/null
. "$_HERE/lib_site_guard.sh"
opd_require_paths "$OPD_SITE_ROOT" || exit 1

# --- WHERE THIS IS ALLOWED TO RUN, which is a SITE fact ------------------------------------
# the H20 cluster: no batch partition has outbound network, so the login node is the only option and
# its reaper has to be worked around (see download_wiki18.sh's wget retry loop).
# the A100 cluster: <transfer-partition> has network AND a real memory cgroup. The login node there killed a
# 423 MB cublas wheel mid-download -- "Killed", no traceback, halfway through torch.
#
# So the rule is not "login node" or "batch node", it is "wherever this site puts the
# network", and getting it wrong costs a 2 GB re-download either way.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [ -z "${OPD_NETWORK_PARTITION:-}" ]; then
    echo "REFUSING: inside a SLURM job, but site '$OPD_SITE_NAME' has no partition with" >&2
    echo "outbound network. Run this on the login node." >&2
    exit 1
  fi
  if [ "${SLURM_JOB_PARTITION:-}" != "$OPD_NETWORK_PARTITION" ]; then
    echo "REFUSING: this job is in partition '${SLURM_JOB_PARTITION:-?}', but only" >&2
    echo "'$OPD_NETWORK_PARTITION' has outbound network at site '$OPD_SITE_NAME'." >&2
    exit 1
  fi
elif [ -n "${OPD_NETWORK_PARTITION:-}" ]; then
  echo "REFUSING: site '$OPD_SITE_NAME' has a network partition, so this does NOT belong on" >&2
  echo "the login node -- its reaper SIGKILLs long downloads partway through (observed on a" >&2
  echo "423 MB cublas wheel). Submit it instead:" >&2
  echo >&2
  echo "  mkdir -p \"\$OPD_SITE_ROOT/logs\" && sbatch -p $OPD_NETWORK_PARTITION -J opd-setup-env \\" >&2
  echo "    -t 04:00:00 -c 8 --mem=32G -o \"$OPD_SITE_ROOT/logs/setup-env-%j.out\" \\" >&2
  echo "    --export=ALL,OPD_SITE=$OPD_SITE_NAME --wrap 'bash slurm/setup_env.sh'" >&2
  echo >&2
  echo "Override with OPD_ALLOW_LOGIN_INSTALL=1 if you know the reaper is not in play." >&2
  [ -n "${OPD_ALLOW_LOGIN_INSTALL:-}" ] || exit 1
fi

# Big wheels unpack in TMPDIR and cache in PIP_CACHE_DIR. Both default into $HOME or a small
# (sometimes tmpfs-backed) /tmp; torch's CUDA wheels alone are ~2 GB unpacked, which is a way
# to fill either. Put both on scratch, where the site already keeps everything else.
export TMPDIR="${TMPDIR_OVERRIDE:-$OPD_SITE_ROOT/cache/tmp}"
export PIP_CACHE_DIR="$OPD_SITE_ROOT/cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
# A resumed build re-uses the cache, so a killed run costs the wheel it died on, not all of them.
export PIP_RETRIES="${PIP_RETRIES:-10}"
export PIP_TIMEOUT="${PIP_TIMEOUT:-60}"

# The stack follows the site's channel unless overridden. a800 is the only cu128 channel.
case "${OPD_STACK:-$OPD_CHANNEL}" in
  a800) TORCH_VER=2.9.0;  TORCH_INDEX=https://download.pytorch.org/whl/cu128; VLLM_VER=0.11.2; CUDA_MAJOR=12 ;;
  *)    TORCH_VER=2.11.0; TORCH_INDEX=https://download.pytorch.org/whl/cu130; VLLM_VER=0.25.1; CUDA_MAJOR=13 ;;
esac

echo "[env] site   : $OPD_SITE_NAME"
echo "[env] target : $OPD_CONDA_ENV"
echo "[env] torch  : $TORCH_VER from $TORCH_INDEX"
echo "[env] vllm   : $VLLM_VER"
echo

source "$(conda info --base)/etc/profile.d/conda.sh"

# A PATH (the A100 cluster: on scratch) and a NAME (the H20 cluster: in the conda root) both work with
# `conda activate`; only `conda create` needs to know which it was given.
#
# IDEMPOTENT ON PURPOSE. A killed pip leaves a real env holding python and nothing else, and
# `conda create` over an existing prefix errors out -- so the obvious retry fails at step one
# and invites a `rm -rf` that throws away the interpreter and every wheel already installed.
# Re-running this script must just continue.
if conda env list | grep -qE "(^|[ /])${OPD_CONDA_ENV}\$"; then
  echo "[env] $OPD_CONDA_ENV already exists -- skipping create, continuing the install"
else
  case "$OPD_CONDA_ENV" in
    /*) conda create -y -p "$OPD_CONDA_ENV" python=3.10 ;;
    *)  conda create -y -n "$OPD_CONDA_ENV" python=3.10 ;;
  esac
fi
conda activate "$OPD_CONDA_ENV"
[ "${CONDA_PREFIX:-}" = "$OPD_CONDA_ENV" ] || [ "$(basename "${CONDA_PREFIX:-}")" = "$OPD_CONDA_ENV" ] || {
  echo "FATAL: asked for env '$OPD_CONDA_ENV' but '${CONDA_PREFIX:-<none>}' is active." >&2
  echo "       conda activate does not fail loudly; installing here would go into the WRONG" >&2
  echo "       env -- typically base, which is exactly what must not happen." >&2
  exit 1; }
echo "[env] installing into $CONDA_PREFIX"

# TORCH FIRST, from the pinned index. Then vLLM sees the requested version already satisfied
# (a local version like 2.11.0+cu130 satisfies ==2.11.0) and will not replace it with the
# PyPI default build.
pip install --index-url "$TORCH_INDEX" "torch==${TORCH_VER}"
python -c "import torch; assert torch.__version__.startswith('${TORCH_VER}'), torch.__version__; \
print('[env] torch installed:', torch.__version__, '| built for CUDA', torch.version.cuda)"

pip install "vllm==${VLLM_VER}"

# THE CHECK THAT MATTERS AFTER INSTALL: did vLLM quietly pull a different-CUDA torch back in?
CUDA_MAJOR="$CUDA_MAJOR" python - <<'PY'
import os
import torch, vllm
want = os.environ["CUDA_MAJOR"]
cuda = torch.version.cuda or ""
print(f"[env] torch {torch.__version__} (CUDA {cuda}) | vllm {vllm.__version__}")
assert cuda.startswith(want), (
    f"torch is built for CUDA {cuda}, not {want}.x -- installing vLLM replaced it with a "
    "different build. Reinstall torch from the pinned index and re-check.")
from vllm.model_executor.layers import batch_invariant           # noqa: F401
print("[env] vllm.model_executor.layers.batch_invariant: importable")
PY

# The training/eval stack, plus what Search-R1's retrieval_server.py needs on top of it.
# The retriever runs in THIS env: at a single site there is one stack, and giving the
# retriever its own would be a second environment to keep in step for no benefit.
pip install "transformers>=4.51" accelerate numpy pyyaml pytest matplotlib
pip install faiss-cpu datasets fastapi uvicorn pydantic

python - <<'PY'
import faiss, torch, transformers, datasets, numpy, fastapi, uvicorn, pydantic, yaml, vllm  # noqa: F401
print(f"[env] transformers {transformers.__version__} | numpy {numpy.__version__} | faiss ok")
PY

echo
echo "[env] BUILT at $OPD_CONDA_ENV"
echo "[env] Nothing is certified yet -- an import proves the install and says nothing about"
echo "[env] batch invariance, weight hot-update, or the numbers."
echo "[env] Next:  bash slurm/submit.sh 2 slurm/certify_channel.sbatch"
