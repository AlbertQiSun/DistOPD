#!/usr/bin/env bash
#SBATCH --job-name=opd-teacher
#
# PARTITION / ACCOUNT / GRES ARE DELIBERATELY NOT HERE. They are site vocabulary, and
# slurm/submit.sh supplies all three:
#
#     OPD_SITE=<site> bash slurm/submit.sh <n_gpus> <this file>
#
# They cannot stay as #SBATCH lines. A command line beats an #SBATCH line only when it SUPPLIES
# the option -- so a site that declares NO account (the A100 cluster: AllowAccounts=ALL) leaves
# the H20 cluster's `--account=<netid>` in force and the submission is rejected outright. Same trap for
# a gres TYPE: `gpu:h20:2` survives into a cluster that has no h20. Observed 2026-08-16.
#SBATCH --nodes=1
# GPUS: 1                    # MiroThinker-1.7-mini is a 30B-A3B MoE (~30.5B total
#                                           # params) -- it needs its OWN big card. PIN the type
#                                           # (H20; a800 UNVERIFIED, see gpu_preflight) -- a plain gpu:1 risks a consumer
#                                           # 4090d whose CUDA 12.4 driver is too old for vLLM/torch.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00                     # long-lived: load MiroThinker ONCE, serve every job
#SBATCH --output=slurm/logs/teacher-%j.out
#SBATCH --error=slurm/logs/teacher-%j.out
#
# The MiroThinker teacher is a SEPARATE long-lived vLLM PROCESS, not part of any training
# job. Why a separate PROCESS (not in-process vLLM): the teacher must be numerically
# isolated from the policy's vLLM engine -- two live vLLM V1 engines in ONE process
# interfere via CUDA graphs / global state, so the only clean isolation is a separate
# process on its OWN card. Serving also makes online and offline hit the SAME backend by
# construction (gate 1): online calls /score_logprobs live per step; the decoupled offline
# precompute calls the SAME endpoint, stores to disk, then trains off disk.
#
# Handoff (mirrors retriever_wiki18.sbatch): this job writes its /score_logprobs URL to a scratch
# file; the training / precompute sbatch reads it and passes --teacher-endpoint, so nobody
# hand-edits the committed config (teacher.endpoint stays a REPLACE_ME placeholder).

set -euo pipefail
source slurm/env.sh                        # conda activate OPD  (see env.sh)

# --- staged artifacts (LOCAL paths; compute nodes are offline) ---
TEACHER=${OPD_TEACHER_MODEL}
DTYPE=bfloat16
# DERIVED per teacher (slurm/lib_teacher_endpoint.sh), not a fixed 8100: two teacher jobs can
# land on the same 4-GPU node, and a bind collision there costs the queue wait plus the weight
# load before it is discovered.
PORT=$(opd_teacher_port) || exit 1

# deps sanity: the served teacher needs vLLM in THIS env (fail here, not mid-serve).
python -c "import vllm" \
  || { echo "ERROR: env missing vllm (the served teacher requires it)"; exit 1; }

# --- publish this node's /score_logprobs URL so training/precompute jobs can find it ---
# PER-LINE. Each teacher line serves a DIFFERENT model, and one shared host file means
# the last server to start owns it. A client pointed at the wrong server DOES fail loudly
# -- HTTPScorer._verify_backend compares tokenizer_name over /health, and tokenizer_name
# is not an ENGINE_RUNTIME_FIELD so it is really compared -- but failing at hour three of
# a queued job is not the same as not colliding.
HOST_FILE=$(opd_teacher_host_file) || exit 1
mkdir -p "$(dirname "$HOST_FILE")"
echo "http://$(hostname -f):${PORT}/score_logprobs" > "$HOST_FILE"
echo "[teacher] model=${TEACHER}"
echo "[teacher] endpoint published to $HOST_FILE : $(cat "$HOST_FILE")"

# --- launch the served vLLM teacher (slurm/teacher_server.py) -------------------------
# Blocks and serves POST /score_logprobs until the job is cancelled. The server FORCES the
# determinism pins (VLLM_BATCH_INVARIANT=1, enforce_eager, prefix-caching OFF = gate 3a),
# feeds input_ids VERBATIM (pure-id contract -- never detok->retok), and publishes its
# backend fingerprint on GET /health for the client's cross-service gate-1 check.
# RETRY ONLY STARTUP FAILURES -- the same rule, and the same reason, as slurm/gen_line.sh.
# vLLM's init_device intermittently returns cudaErrorDevicesUnavailable ("CUDA-capable
# device(s) is/are busy or unavailable") when a fresh allocation lands on a GPU whose previous
# tenant has not finished releasing it. It fires inside MemorySnapshot, BEFORE any weights are
# read, so it always shows up within a minute of job start. Observed 2026-08-22 during
# generation (twice) and 2026-08-24 on this script (twice) -- the second time on the very card
# the previous failed attempt had just died on, which is the mechanism in one sentence.
#
# 300 s is well past this failure and well before the server is useful: Qwen3-32B needs ~2 min
# to load and then runs a batch-equivalence self-check. So a death inside the grace period is
# a start-up problem and worth retrying; a death after it is a real fault and must surface.
#
# WHY THIS MATTERS MORE HERE THAN ANYWHERE ELSE: two arms sit in wait_for_teacher with a hard
# 1800 s timeout. A teacher that dies at 50 s and is not retried does not just lose its own
# job -- it kills both arms of that line half an hour later, for a transient that would have
# cleared on the next attempt.
STARTUP_GRACE=300
for TRY in 1 2 3; do
  T0=$SECONDS
  CUDA_VISIBLE_DEVICES=0 python slurm/teacher_server.py \
    --model "$TEACHER" \
    --dtype "$DTYPE" \
    --host 0.0.0.0 --port "$PORT"
  RC=$?
  ELAPSED=$((SECONDS - T0))
  # rc=0 means the server was cancelled or exited cleanly -- not something to restart.
  [ $RC -eq 0 ] && break
  if [ "$ELAPSED" -gt "$STARTUP_GRACE" ]; then
    echo "[teacher] FAILED rc=$RC after ${ELAPSED}s -- past start-up, NOT retrying" >&2
    exit "$RC"
  fi
  if [ "$TRY" -eq 3 ]; then
    echo "[teacher] FAILED rc=$RC after ${ELAPSED}s on the last of 3 start-up attempts" >&2
    exit "$RC"
  fi
  echo "[teacher] start-up failure rc=$RC after ${ELAPSED}s, retrying (${TRY}/3) in 60s" >&2
  sleep 60
  # Republish: a retry keeps the same node and port, but the file may have been overwritten
  # by another line's server in the meantime, and the waiting arms read it on every poll.
  echo "http://$(hostname -f):${PORT}/score_logprobs" > "$HOST_FILE"
done
