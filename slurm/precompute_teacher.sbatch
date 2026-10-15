#!/usr/bin/env bash
#SBATCH --job-name=opd-precompute
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
# GPUS: 1                 # student sampling (Qwen3-4B) card. The 30B MoE teacher is
#                                        # NOT here -- it is the SEPARATE served process
#                                        # (teacher_server.sbatch); this job only CALLS it over HTTP.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00                  # 38,400 trajectories at the measured 1.55 s each
#                                        # (batch-invariant ON) = 16.5h, plus margin.
#                                        # measured, not guessed: 1.348 s/trajectory of
#                                        # training (weighted over the real length
#                                        # distribution) + 1.55 s/trajectory of rollout, from
#                                        # scripts/bench_train_step.py on an H20.
#                                        #
#                                        # --time is the MEASURED NEED PLUS MARGIN, not the
#                                        # 48h ceiling. A request longer than the work needs
#                                        # cannot backfill: on this partition other users hold
#                                        # 3-5 day allocations, so the scheduler only starts a
#                                        # lower-priority job it can prove finishes before the
#                                        # next reservation. That is what kept the retriever
#                                        # pending at 48h with an idle node available.
#                                        # Sharding by step range makes this several SHORT
#                                        # jobs instead, which is strictly better here.
#SBATCH --output=slurm/logs/precompute-%j.out
#
# OFFLINE data prep, DECOUPLED: sample trajectories from pi_ref ONCE and score the SERVED
# teacher over each FULL trajectory, caching teacher_logp + a manifest to precomputed/.
# This process only ROLLS OUT and CALLS the teacher service (teacher_server.sbatch must be
# up); the 30B MoE teacher itself lives on its own card. The manifest records the backend
# fingerprint, which gate 1 (assert_same_backend) checks when the offline training run reads
# the cache back. Online never runs this -- it scores the same service live per step.
#
# You (the researcher) submit this; it is never launched automatically.

set -euo pipefail
source slurm/env.sh

# ---- node-level preflight: prove the pipeline wires up on this hardware (CPU/mock) ----
python scripts/dry_run_smoke.py

# ---- node-level GPU preflight: this node's DRIVER, not just its gpu type ----
# `--gres=gpu:h20:N` pins hardware uniformity, not a working driver: <node> has an
# A800 whose CUDA 12.2 driver this torch cannot initialise. Fail in seconds here
# rather than after the queue wait, model load and endpoint handshakes.
# The retry budget is a knob because the transient it waits out is not ours. SLURM hands out
# the lowest-index free GPU first, so "our device 0" is systematically the card most recently
# released on that node -- which is why six consecutive cudaErrorDevicesUnavailable failures
# were ALL on device 0 and never on device 1. The default 6 x 10 s was not enough on the
# churned partitions (measured: six failures, none recovering inside 50 s), while later jobs
# on the same nodes ran fine, so the card does free -- just later. Raise it when submitting
# into a pool someone else is cycling; a genuinely dead card then costs the longer wait
# instead of one minute, which is the trade.
python scripts/gpu_preflight.py --device-tries "${OPD_DEVICE_TRIES:-6}"                                 --device-delay "${OPD_DEVICE_DELAY:-10}"

# The student is what gets rolled out here; name it or refuse.
opd_require_student_ckpt || exit 1

# ---- resolve the endpoints published by retriever_wiki18.sbatch + teacher_server.sbatch ----
# Offline still rolls out through the retriever (it just scores the teacher once, to disk),
# and it scores against the SAME served teacher online will use -> gate 1 holds by construction.
# BLOCK until both actually answer: the host files are written at service job START but the
# services need minutes to load (71 GB index / 30B MoE). Submitting this right behind them
# otherwise dies on ConnectionRefused against a not-yet-listening (or previous) node.
wait_for_retriever
wait_for_teacher
RETRIEVER=$(cat ${OPD_RUNSTATE_DIR}/opd_retriever_host.txt)
TEACHER=$(cat "$(opd_teacher_host_file)")
echo "[precompute] retriever: $RETRIEVER"
echo "[precompute] teacher  : $TEACHER"

# ---- SHAPE. Defaults are the 64-question smoke; the real run overrides at submit time ----
#
#   OPD_LIMIT=38400 OPD_STEPS=150 bash slurm/submit.sh 1 slurm/precompute_teacher.sbatch --time=20:00:00
#
# It writes exactly LIMIT trajectories -- train.py's precompute branch is
# `precompute_offline(...)` over `questions`, and it prints len(questions). NOT
# steps x global_batch: those govern how many times TRAINING revisits the cache, and offline
# re-reads the same cached trajectory on each revisit. So LIMIT alone sets this job's cost.
#
# EPOCHS IS DERIVED, not a third knob. train.py asserts
# steps x global_batch == limit x epochs and refuses otherwise -- this job passed --limit 64
# while inheriting steps=150 from the config, which is 38400 draws over 64 questions = 600
# epochs against a declared 1, and it died there after waiting hours for a GPU.
#
# Deriving does make that assertion true by construction. What still carries the signal is the
# line train.py prints: at epochs != 1 it says the run is NOT apples-to-apples, and the real
# run (LIMIT 38400, STEPS 150) derives exactly 1.0. Same convention as selfcontained.sbatch.
LIMIT="${OPD_LIMIT:-64}"
STEPS="${OPD_STEPS:-40}"
EPOCHS=$(python -c "import yaml; c=yaml.safe_load(open('configs/opd_qwen3.yaml')); print(${STEPS}*c['optim']['global_batch']/${LIMIT})")
echo "[precompute] shape: limit=$LIMIT steps=$STEPS -> epochs=$EPOCHS"

# ---- REAL precompute (data.path set in configs/opd_qwen3.yaml) ----
# Rolls out LIMIT pi_ref trajectories through the retriever, scores each FULL trajectory
# against the SERVED teacher, and writes precomputed/traj_*.pt + manifest.json, then STOPS (no
# training). The offline training job then reads that manifest read-only; gate 1
# (assert_same_backend) re-checks the manifest's backend fp at that read.
python scripts/train.py --config configs/opd_qwen3.yaml --mode offline --precompute-only \
  --retriever-endpoint "$RETRIEVER" --teacher-endpoint "$TEACHER" \
  --channel "$OPD_CHANNEL" \
  --precompute-dir "${OPD_PRECOMPUTE_DIR:-precomputed}" \
  --limit "$LIMIT" --steps "$STEPS" --epochs "$EPOCHS"
# The fingerprint moved with the prompt: f9da237f476eb71e was v2_terse, the current default
# v3_terse_shortans is 3165062cda7688de (tests/test_prompt_variants.py holds the literal).
# Checking against the old one would read as "the precompute is broken" when it is correct.
echo "[precompute] DONE -- wrote traj_*.pt + manifest.json in ${OPD_PRECOMPUTE_DIR:-precomputed}."
echo "[precompute] Verify the manifest's template fingerprint == 3165062cda7688de and its"
echo "[precompute] teacher == ${OPD_TEACHER_MODEL} before Step 3."
