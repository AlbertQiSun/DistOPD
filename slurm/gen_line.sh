#!/usr/bin/env bash
# One teacher line's SFT generation. Call it with the line name and the model directory:
#
#     bash slurm/gen_line.sh dense Qwen3-32B
#     bash slurm/gen_line.sh moe   Qwen3-30B-A3B-Thinking
#     bash slurm/gen_line.sh miro  MiroThinker-1.7-mini
#
# WHY THIS FILE EXISTS. The same thing was being expressed as an sbatch --wrap string, and
# the quoting bit three times. The last one cost three jobs: `\\\$(cat \\\${OPD_RUNSTATE_DIR}
# /...)` collapsed to a LITERAL ${OPD_RUNSTATE_DIR}, so cat failed, --retriever-endpoint got
# an empty string, and argparse killed the run in three seconds -- on all three lines, on
# three different nodes, which read convincingly like a cluster fault and was diagnosed as
# one twice before anybody read the traceback. A script has ONE level of quoting; a --wrap
# nested inside a for loop inside a shell has three, and the failure it produces looks
# nothing like its cause.
#
# Assumes slurm/env.sh has already been sourced (the caller does that, because OPD_SITE has
# to be set before it).
set -uo pipefail

LINE="${1:?usage: gen_line.sh <line-name> <model-dir-name>}"
MODEL_NAME="${2:?usage: gen_line.sh <line-name> <model-dir-name>}"

export OPD_TEACHER_MODEL="${OPD_MODELS_DIR}/${MODEL_NAME}"
OUT="runs/prod_${LINE}_sft"
HOST_FILE="${OPD_RUNSTATE_DIR}/opd_retriever_host.txt"

echo "[gen_line] line=${LINE} node=$(hostname -s) teacher=${OPD_TEACHER_MODEL}"

# Read the retriever endpoint HERE and refuse if it is empty. The failure above reached
# argparse with an empty string; a run that gets this far should already know it cannot work.
if [ ! -s "$HOST_FILE" ]; then
  echo "FATAL: $HOST_FILE is missing or empty -- is the retriever job running?" >&2
  exit 1
fi
ENDPOINT="$(cat "$HOST_FILE")"
echo "[gen_line] retriever=${ENDPOINT}"
case "$ENDPOINT" in
  http*://*/retrieve) ;;
  *) echo "FATAL: retriever endpoint does not look like a URL: '${ENDPOINT}'" >&2; exit 1 ;;
esac

# RETRY ONLY STARTUP FAILURES. vLLM's device init intermittently returns
# cudaErrorDevicesUnavailable when a fresh allocation lands on a GPU whose previous tenant has
# not finished releasing it (observed twice, 2026-08-22, on different nodes, each time within
# 30 s of job start). gen_sft_data has no resume -- gen_sft.sbatch does rm -rf records/ -- so a
# retry after generation has begun would silently redo hours of work. 300 s is well past
# engine start-up (weights ~70 s + compile ~40 s for the largest model here) and well before
# any meaningful number of questions is done.
STARTUP_GRACE=300
for TRY in 1 2 3; do
  T0=$SECONDS
  python scripts/gen_sft_data.py \
    --config configs/opd_qwen3.yaml \
    --out-dir "$OUT" \
    --retriever-endpoint "$ENDPOINT" \
    --expect-action-cap 2560 \
    --expect-prompt-variant v3_terse_shortans
  RC=$?
  [ $RC -eq 0 ] && break
  ELAPSED=$((SECONDS - T0))
  if [ "$ELAPSED" -gt "$STARTUP_GRACE" ]; then
    echo "[gen_line] FAILED rc=$RC after ${ELAPSED}s -- past start-up, NOT retrying" >&2
    exit "$RC"
  fi
  if [ "$TRY" -eq 3 ]; then
    echo "[gen_line] FAILED rc=$RC after ${ELAPSED}s on the last of 3 start-up attempts" >&2
    exit "$RC"
  fi
  echo "[gen_line] start-up failure rc=$RC after ${ELAPSED}s, retrying (${TRY}/3) in 90s" >&2
  sleep 90
done

# Only reached on success -- the loop exits non-zero otherwise, so the health report can
# never run against a shard that was never written (it did, and printed a FileNotFoundError
# traceback that read like a second, unrelated bug).
python scripts/check_sft_stats.py "$OUT"
