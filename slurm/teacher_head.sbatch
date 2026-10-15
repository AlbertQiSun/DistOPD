#!/usr/bin/env bash
#SBATCH --job-name=teacher_head
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=runs/logs/teacher_head_%x.%j.out
#SBATCH --error=runs/logs/teacher_head_%x.%j.out
#
# delta-OPD step 2: the teacher HEAD (top-K_q + log q on S, residual) at every SOFT position
# of the line's frozen offline cache, from a PRIVATE served teacher on this job's one card --
# the same engine class and config that scored the cache's teacher_logp, so the alignment gate
# (lp_y here vs the record's teacher_logp, < 1e-3) is a same-backend identity check.
# Resumable parts under <cache>/teacher_head/ mirror ref_topk's; re-submit to continue.
#
#   OPD_SITE=example OPD_LINE=miro bash slurm/submit.sh 1 slurm/teacher_head.sbatch \
#       --exclude=<bad nodes> -J thead_miro
#   OPD_N=200 ...            first 200 trajectories (a smoke: measures the rate before the chain)
#   OPD_HEAD_KWIDE=256 ...   prompt_logprobs width the server returns (>= the K' lookups need)
#
# GPUS: 1
# The card holds the served teacher (MiroThinker / Qwen3-30B-A3B / Qwen3-32B in bf16); the client
# is CPU. OPD_TEACHER_UTIL per line = the runbook's certified values (0.80 / 0.80 / 0.858).
# Paths come from the SITE (slurm/sites/<site>.sh via slurm/env.sh), never from this file.
source slurm/env.sh
LINE=${OPD_LINE:?set OPD_LINE=miro|dense|moe}
M=$OPD_MODELS_DIR
case "$LINE" in
  miro)  TEACHER=${OPD_TEACHER_DIR:-$M/MiroThinker-1.7-mini};   UTIL_DEFAULT=0.80 ;;
  moe)   TEACHER=${OPD_TEACHER_DIR:-$M/Qwen3-30B-A3B-Thinking}; UTIL_DEFAULT=0.80 ;;
  dense) TEACHER=${OPD_TEACHER_DIR:-$M/Qwen3-32B};              UTIL_DEFAULT=0.858 ;;
  *) echo "FATAL: unknown line $LINE" >&2; exit 1 ;;
esac
CACHE=${OPD_CACHE:-precomputed_${LINE}_full}
PIREF=${OPD_PIREF_CKPT:-$M/pi_ref_$LINE}
KWIDE="${OPD_HEAD_KWIDE:-256}"
N_LIMIT="${OPD_N:-}"
IGNORE="${OPD_IGNORE_GATES:-}"
for p in "$CACHE/manifest.json" "$CACHE/ref_topk.npz" "$TEACHER" "$PIREF"; do
  test -e "$p" || { echo "FATAL: $p missing" >&2; exit 1; }
done
python scripts/gpu_preflight.py --device-tries "${OPD_DEVICE_TRIES:-20}" --device-delay "${OPD_DEVICE_DELAY:-15}"

# the served teacher on the one card, wide enough for /score_head, torn down on exit
trap stop_private_teacher EXIT INT TERM
export OPD_TEACHER_MODEL="$TEACHER"
export OPD_TEACHER_UTIL="${OPD_TEACHER_UTIL:-$UTIL_DEFAULT}"
export OPD_TEACHER_MAX_LOGPROBS="$KWIDE"
# OPD_HEAD_SKIP_TOKENIZER=1 (default): the engine skips its tokenizer, so the k_wide logprob
# entries per prompt position are not detokenized -- the 2026-09-05 smoke without it ran at
# ~6 s per trajectory (~64 h for the cache). Numerically inert; the alignment gate re-checks lp_y.
SKIP_TOK="${OPD_HEAD_SKIP_TOKENIZER:-1}"
if [ "$SKIP_TOK" = "0" ]; then SKIP_TOK=""; fi
export OPD_TEACHER_SKIP_TOKENIZER="$SKIP_TOK"
start_private_teacher 0 "$TEACHER" bfloat16
echo "[head] teacher=$TEACHER at $TEACHER_ENDPOINT (max_logprobs=$KWIDE, util=$OPD_TEACHER_UTIL, skip_tokenizer=${SKIP_TOK:-0})"

python -u scripts/teacher_head_dump.py --line "$LINE" --cache "$CACHE" --teacher "$TEACHER" \
    --teacher-endpoint "$TEACHER_ENDPOINT" --tokenizer "$PIREF" --k-wide "$KWIDE" \
    ${N_LIMIT:+--n "$N_LIMIT"} ${IGNORE:+--ignore-gates}
rc=$?
echo "[head] rc=$rc (0 = complete and gates passed; 1 = partial or a gate failed -- read $CACHE/teacher_head/gates.json)"
exit $rc
