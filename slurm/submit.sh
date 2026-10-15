#!/usr/bin/env bash
# sbatch with THIS SITE's partition, account and gres name filled in.
#
#   bash slurm/submit.sh <n_gpus> <sbatch-file> [extra sbatch args...]
#   OPD_SITE=example bash slurm/submit.sh 2 slurm/train_online.sbatch --time=36:00:00
#
# WHY A WRAPPER AND NOT #SBATCH LINES. Partition names, account names and gres TYPE strings
# are site vocabulary (`<h20-partition>` + `gpu:h20:2` vs `<a100-partition>` + `gpu:a100:2`); the GPU COUNT is a
# property of the job (the online arm needs the policy and the student on separate cards,
# whatever the cards are called). So the templates carry the count as a comment and nothing
# else, and every site's vocabulary lives in exactly one file.
#
# The templates USED to keep the H20 cluster's values as #SBATCH defaults, on the theory that a
# command line beats an #SBATCH line. It does -- but only when the command line SUPPLIES the
# option. the A100 cluster declares no account (AllowAccounts=ALL), so nothing was passed, the H20 cluster's
# `--account=<netid>` stayed in force, and every submission was rejected with "Invalid account
# or account/partition combination". A gres TYPE has the identical trap: `gpu:h20:2` survives
# into a cluster that has no h20. An #SBATCH line cannot be un-set, so it cannot hold anything
# site-specific. tests/test_site_config.py enforces that.
#
# The other hazard -- right partition, WRONG site exported, so the job reads paths that are not
# there -- is caught by slurm/env.sh, which checks that the site's declared paths exist on this
# machine before anything loads.

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: bash slurm/submit.sh <n_gpus> <sbatch-file> [extra sbatch args...]" >&2
  echo "  n_gpus 0 = no GPU request at all (e.g. a <transfer-partition> job)" >&2
  exit 2
fi

NGPU="$1"; SCRIPT="$2"; shift 2
case "$NGPU" in ''|*[!0-9]*) echo "FATAL: n_gpus must be an integer, got '$NGPU'" >&2; exit 2 ;; esac
[ -f "$SCRIPT" ] || { echo "FATAL: no such sbatch file: $SCRIPT" >&2; exit 2; }

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OPD_SITE:=example}"
_SITE_FILE="$_HERE/sites/${OPD_SITE}.sh"
[ -f "$_SITE_FILE" ] || { echo "FATAL: unknown OPD_SITE=$OPD_SITE" >&2; exit 1; }
# shellcheck source=/dev/null
. "$_SITE_FILE"

# PARTITION OVERRIDE. The site default is the GPU partition, because that is where nearly
# every job goes. The retriever is the exception that makes this necessary: its GPU is never
# used at all (patch_searchr1_cpu.py forces the E5 encoder and FAISS onto CPU), so on the A100 cluster
# it belongs on cpu_long -- 28 days, default QOS, no GPU quota -- while <a100-partition> is saturated
# and every GPU ask waits behind (Resources).
#
# Implemented by SUPPRESSING our default when the caller supplies one, rather than by passing
# -p twice and trusting sbatch to take the last. "Later wins" is probably true and was NOT
# verified here; a silent wrong partition is how a job lands on a node whose paths differ, and
# this repo does not ship unverified behaviour as a mechanism.
# All four spellings sbatch accepts. Each strips its OWN prefix: a single `${_a#*[=p]}` looks
# like it covers both forms and does not -- on `--partition=cpu_long` it stops at the `p` in
# "partition" and reports `artition=cpu_long`. Suppression would still be correct and the log
# line would be wrong, which is the kind of nearly-right that gets believed.
_want_partition=""
_take_next=""
for _a in "$@"; do
  if [ -n "$_take_next" ]; then _want_partition="$_a"; _take_next=""; continue; fi
  case "$_a" in
    -p|--partition)     _take_next=1 ;;
    --partition=*)      _want_partition="${_a#--partition=}" ;;
    -p*)                _want_partition="${_a#-p}" ;;
  esac
done
[ -n "$_take_next" ] && { echo "FATAL: trailing -p with no partition after it" >&2; exit 2; }

ARGS=()
if [ -n "$_want_partition" ]; then
  echo "[submit] partition overridden by the caller ($_want_partition); site default" \
       "$OPD_PARTITION not applied"
else
  ARGS+=(-p "$OPD_PARTITION")
fi
[ -n "${OPD_ACCOUNT:-}" ] && ARGS+=(--account="$OPD_ACCOUNT")
[ "$NGPU" -gt 0 ] && ARGS+=(--gres="gpu:${OPD_GRES_GPU}:${NGPU}")

# OPD_SITE must reach the job, or env.sh inside it defaults back to h20-site and the guard
# fires (or, on h20-site, silently does not).
ARGS+=(--export="ALL,OPD_SITE=${OPD_SITE}")

# EXTRA ARGS GO BEFORE THE SCRIPT. sbatch stops parsing its own options at the script name and
# hands everything after it to the batch script as positional arguments -- so `submit.sh 1 x.sbatch
# --time=01:00:00` placed after would SILENTLY not set the walltime, and walltime is the one
# knob that decides whether a job pending on (Priority) can backfill at all. None of the
# templates take positional arguments; they read the environment.
echo "[submit] site=$OPD_SITE_NAME  ${ARGS[*]} $*  $SCRIPT"

# Testability hook, not a convenience: tests/test_site_config.py runs this script for real and
# reads the line below, so what is asserted is what would actually be submitted. Checking the
# assembly by re-deriving it in Python would be the test inventing the very data it verifies.
if [ -n "${OPD_SUBMIT_DRY_RUN:-}" ]; then
  echo "WOULD RUN: sbatch ${ARGS[*]} $* $SCRIPT"
  exit 0
fi
exec sbatch "${ARGS[@]}" "$@" "$SCRIPT"
