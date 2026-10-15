#!/usr/bin/env bash
#SBATCH --job-name=opd-certify
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
# GPUS: 2                                 # card 0: the served teacher.
#                                         # card 1: the policy + the student.
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/certify-%j.out
#
# EVERY GATE FOR ONE CHANNEL, IN ONE ALLOCATION.
#
#   OPD_SITE=example bash slurm/submit.sh 2 slurm/certify_channel.sbatch
#
# card 0: the served teacher.   card 1: the policy + the student.
#
# Splitting these into eight submissions would mean eight queue waits, and a free-card window
# can close at any time. They also have to run in ORDER: each gate makes the next one
# meaningful, and the early ones are the cheap refutations.
#
# The channel is certified only if ALL of them pass. A failure at the determinism probe or at
# sync_from voids the channel outright -- it does not degrade it to "offline only", because one
# experiment's precompute and BOTH its arms must live on the same channel (README.md).
#
# ONE SCRIPT FOR EVERY CHANNEL. It replaced certify_a800.sbatch, which was a copy with two
# words changed; the copy would have drifted, and a certification harness that differs per
# channel for undocumented reasons certifies nothing comparable. What legitimately differs is
# the RISK PROFILE, and that is stated per gate rather than duplicated per file:
#
#   a800  torch 2.9.0+cu128 / vLLM 0.11.2. A DOWNGRADE from the reference stack, so gate 5
#         (sync_from on a live V1 engine) is the highest-risk gate and a plausible failure.
#   a100  torch 2.11.0+cu130 / vLLM 0.25.1 -- IDENTICAL to the H20 channel, because the A100 cluster's
#         driver is 580.126.20 = CUDA 13.0. Only the architecture differs (sm_90 -> sm_80), so
#         a gate-5 failure here would be a genuine surprise rather than an expected cost.
#
# PREREQUISITE: the env exists (slurm/setup_env.sh) and the assets are staged.

set -uo pipefail                          # NOT -e: a failing gate must be reported, not silent

source slurm/env.sh                       # site paths, conda, and the cross-site guard

# RE-ASSERT IT, because slurm/env.sh just took it away. env.sh does `set -euo pipefail` for
# non-interactive shells -- correct for the ten other templates, which all want strict mode and
# set it themselves -- but it is sourced AFTER the line above, so this script's stated intent
# has never once been in force.
#
# It went unnoticed because run_gate calls each gate as `if "$@"; then`, and a command in an
# `if` condition is exempt from errexit. Every gate therefore behaved exactly as designed, by
# accident. The first failing command OUTSIDE an `if` was `ls traj_*.pt` on a directory with no
# matches -- `ls` exits 2, pipefail carries the 2 out of the pipeline, and errexit killed a run
# that had just passed five gates, one line before it would have recorded a SKIP and continued.
#
# The self-check below is the part that matters: a comment claiming the option is off is the
# same class of unverified assertion that produced this bug.
set +e
case $- in
  *e*) echo "FATAL: errexit is still on after slurm/env.sh. Gate failures would abort this" >&2
       echo "       script instead of being reported, and the summary would never print." >&2
       exit 1 ;;
esac

CHANNEL="${OPD_CHANNEL:?}"
TEACHER_MODEL="$OPD_TEACHER_MODEL"
STUDENT_CKPT="$OPD_STUDENT_CKPT"
# DELIBERATELY EMPTY BY DEFAULT. 0.72 was tuned for the H20's 95 GiB; on an 80 GiB card that is
# 57.0 GiB, less than the teacher's ~61 GiB of bf16 weights, so it does not even load. Gate 0.5
# SEARCHES for the value this card needs and gates 1+ refuse to start the real teacher without
# it -- a channel certified at one utilization and run at another has certified nothing.
TEACHER_UTIL="${OPD_TEACHER_UTIL:-}"

PASSED=0
run_gate() {                              # run_gate <name> <what a failure means> <cmd...>
  local name="$1" meaning="$2"; shift 2
  echo
  echo "================================================================"
  echo "[gate] $name"
  echo "================================================================"
  if "$@"; then
    echo "[gate] PASS: $name"
    PASSED=$((PASSED + 1))
    python - "$CHANNEL:$name" <<'PY'
import sys
sys.path.insert(0, ".")
from opd.gate_log import append_gate_run
from teacher.backend import runtime_fingerprint
append_gate_run(sys.argv[1], "PASS", details=runtime_fingerprint())
PY
    return 0
  fi
  echo "[gate] FAIL: $name" >&2
  echo "[gate] MEANING: $meaning" >&2
  python - "$CHANNEL:$name" <<'PY'
import sys
sys.path.insert(0, ".")
from opd.gate_log import append_gate_run
from teacher.backend import runtime_fingerprint
append_gate_run(sys.argv[1], "FAIL", details=runtime_fingerprint())
PY
  echo
  echo "[certify] STOPPING at $name -- $PASSED gate(s) passed before it. Gates below this one" >&2
  echo "[certify] are not meaningful until it is resolved." >&2
  exit 1
}

SKIPPED=0
skip_gate() {                             # skip_gate <name> <why it cannot run yet>
  local name="$1" why="$2"
  echo
  echo "================================================================"
  echo "[gate] $name -- SKIPPED"
  echo "================================================================"
  echo "[gate] $why"
  SKIPPED=$((SKIPPED + 1))
  # RECORDED, and recorded as SKIP -- never as PASS, and never merely absent. A gate that
  # silently did not run is indistinguishable from one that passed when someone reads the log
  # later, and this project's worst failures have all been something absent that looked
  # present. The final summary refuses to say "certified" while this counter is non-zero.
  python - "$CHANNEL:$name" "$why" <<'PY'
import sys
sys.path.insert(0, ".")
from opd.gate_log import append_gate_run
from teacher.backend import runtime_fingerprint
append_gate_run(sys.argv[1], "SKIP", details={**runtime_fingerprint(), "why": sys.argv[2]})
PY
}

echo "[certify] channel=$CHANNEL  site=$OPD_SITE_NAME  node=$(hostname -s)  util=${TEACHER_UTIL:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------------------------------
run_gate "0-preflight" \
  "this node cannot run the env's kernels at all; nothing else can be trusted" \
  python scripts/gpu_preflight.py --compute

# ---------------------------------------------------------------------------------------
# CHEAPEST REFUTATION OF ALL, and it needs no GPU: seconds, on the tokenizers alone. The whole
# pure-id contract rests on the teacher and the student segmenting text identically, and until
# now that was a sentence in configs/opd_qwen3.yaml with nothing behind it. Its failure mode is
# silent -- no exception, no shape mismatch, just every per-token advantage after the first
# divergence attached to the wrong token. Worth re-running per stack: the transformers version
# is resolved at install time and is NOT in the BackendConfig fingerprint.
run_gate "0.1-tokenizer-identity" \
  "the teacher and the student segment text differently on this stack, so scoring the student's ids with the teacher misaligns every per-token advantage. The channel cannot host the experiment until the pair agrees" \
  python scripts/check_tokenizer_identity.py \
    --teacher "$TEACHER_MODEL" --student "$STUDENT_CKPT"

# ---------------------------------------------------------------------------------------
# GATE 0.5 EARLY, because it is minutes and it is the next cheapest refutation: at the H20's
# 0.72 the teacher does not even load on an 80 GiB card.
run_gate "0.5-teacher-loads-and-scores" \
  "the 30B MoE teacher does not fit this card at any workable utilization -- the channel cannot host the teacher, so it cannot host either arm" \
  env CUDA_VISIBLE_DEVICES=0 python scripts/probe_teacher_fits.py \
    --model "$TEACHER_MODEL" --dtype bfloat16

# Everything below starts the REAL teacher and must use the utilization gate 0.5 just
# certified. Refusing rather than defaulting keeps the certified configuration and the run
# configuration the same object.
if [ -z "$TEACHER_UTIL" ]; then
  echo
  echo "[certify] Gate 0.5 passed. Re-submit with the utilization it reported:" >&2
  echo "[certify]   OPD_SITE=$OPD_SITE_NAME OPD_TEACHER_UTIL=<value> \\" >&2
  echo "[certify]     bash slurm/submit.sh 2 slurm/certify_channel.sbatch" >&2
  echo "[certify] Gates 0..0.5 re-run in minutes; gates 1+ need the certified value." >&2
  exit 1
fi

# ---------------------------------------------------------------------------------------
# The one gate whose failure has NO code fix. An H20 result does not transfer: the architecture
# changes even when the stack does not, and batch-invariance is a kernel property.
run_gate "1-teacher-determinism" \
  "prompt_logprobs is not batch-invariant on this backend. There is no code fix -- gate 3a's bit-identity claim is void and the channel is dead for this experiment" \
  env CUDA_VISIBLE_DEVICES=0 python scripts/determinism_probe.py \
    --model "$TEACHER_MODEL" --dtype bfloat16 \
    --gpu-memory-utilization "$TEACHER_UTIL"

# ---------------------------------------------------------------------------------------
# Starting the server IS gate 2: it refuses to serve unless scoring a sequence alone and
# co-batched agree, and it probes the worst case at startup.
trap stop_private_teacher EXIT INT TERM
run_gate "2-teacher-server-batch-equivalence" \
  "TWO DIFFERENT FAILURES SHARE THIS GATE -- read the [teacher] line above before concluding anything. (a) 'the PROCESS EXITED during startup': the server never served, so NOTHING was measured about batching; the usual cause is an engine configuration this card cannot hold, and the traceback is in THIS log. (b) the server started and refused: it scores a sequence differently alone vs co-batched, so batched precompute would not equal live online scoring" \
  start_private_teacher 0 "$TEACHER_MODEL" bfloat16

# ---------------------------------------------------------------------------------------
# Two PROCESSES on purpose: VLLM_BATCH_INVARIANT is read at import, and two live V1 engines in
# one process corrupt each other. The second invocation picks up the first's JSON and prints
# the comparison.
# THE ONE DEPENDENCY THAT RUNS BACKWARDS THROUGH THIS SCRIPT. policy_invariance_probe.py
# refuses synthetic prefixes -- the batch layouts it tests have to match production tiling, and
# a made-up prefix distribution would certify a shape the arms never see. So it needs
# precomputed/traj_*.pt, which exists only after a precompute, which needs the teacher these
# gates just certified AND a live retriever. Certifying 3a/3b therefore cannot happen in the
# same allocation as certifying everything they depend on.
#
# Stopping here would be wrong twice over: it burns the rest of a 2-GPU allocation on gates
# 4-6 that are ready to run, and it reports a MISSING INPUT as a batch-invariance failure --
# which is what the previous message ("the probe could not run with the flag off") did.
PRECOMPUTE_DIR="${OPD_PRECOMPUTE_DIR:-precomputed}"
# Counted with a nullglob array rather than `ls | wc -l`: "no matches" is the EXPECTED case
# here, and `ls` reports it as exit 2, which pipefail carries out of the pipeline. There is no
# exit status to trip over this way, whatever shell options are in force.
shopt -s nullglob
_traj=("$PRECOMPUTE_DIR"/traj_*.pt)
shopt -u nullglob
N_TRAJ=${#_traj[@]}
if [ "$N_TRAJ" -eq 0 ]; then
  skip_gate "3a-policy-invariance-OFF" \
    "no traj_*.pt in $PRECOMPUTE_DIR. The probe needs REAL prefixes and refuses synthetic ones. Run a small precompute on this channel first (it needs gates 0.5-2, which have just passed, plus the retriever), then re-run this script -- 3a/3b will execute and the rest will re-run in minutes."
  skip_gate "3b-policy-invariance-ON" \
    "same dependency as 3a. Until BOTH run, it is unknown whether VLLM_BATCH_INVARIANT=1 makes sampled tokens independent of batch composition on this architecture -- on sm_90 the flag was the difference between a 100% flip rate and none."
else
  echo "[certify] $N_TRAJ real trajectories in $PRECOMPUTE_DIR -- 3a/3b can run"
  run_gate "3a-policy-invariance-OFF" \
    "the probe failed with the flag OFF. This is the CONTROL arm: it is expected to show flips, so a failure here means the probe itself could not complete, not that invariance is broken. Read its traceback" \
    env CUDA_VISIBLE_DEVICES=1 python scripts/policy_invariance_probe.py \
      --model "$STUDENT_CKPT" --batch-invariant off --cost-questions 0 \
      --precomputed "$PRECOMPUTE_DIR"

  run_gate "3b-policy-invariance-ON" \
    "sampled tokens still depend on batch composition with VLLM_BATCH_INVARIANT=1 on this architecture -- reproducibility is unattainable on this channel" \
    env CUDA_VISIBLE_DEVICES=1 python scripts/policy_invariance_probe.py \
      --model "$STUDENT_CKPT" --batch-invariant on --cost-questions 0 \
      --precomputed "$PRECOMPUTE_DIR"
fi

# ---------------------------------------------------------------------------------------
# ONE PROCESS PER TEST. Each of these builds a vLLM engine, and since the test helper started
# building it the way production does (VLLM_ENABLE_V1_MULTIPROCESSING=0), the EngineCore lives
# in the pytest process and never gives its KV pool back -- measured at exactly 35% free on two
# nodes, with and without vLLM's own destroy_* helpers. A process boundary returns it
# unconditionally, and it is the boundary the real arms already have: precompute, the teacher
# server and each training job are separate processes with one engine each.
run_gate "4-slow-suite" \
  "one of the GPU gates fails on this stack -- read which; the verbatim-ids and batched-pairing gates protect the cache, the parity gates protect the teacher numbers" \
  env CUDA_VISIBLE_DEVICES=1 bash scripts/run_slow_isolated.sh

run_gate "5-sync-from-weight-hot-update" \
  "VLLMPolicy.sync_from cannot rewrite the live sampler's weights on this vLLM. The ONLINE arm cannot be on-policy here, and since one experiment's precompute and BOTH arms must share a channel, that voids the whole channel -- it does NOT reduce to running offline here and online elsewhere" \
  env CUDA_VISIBLE_DEVICES=1 python -m pytest -q -m slow tests/test_vllm_weight_hot_update.py

# ---------------------------------------------------------------------------------------
# The meaning below states a CONDITION, not a cause. It used to assert "the student does not
# fit this card", and printed that verdict when the bench had died before loading anything --
# a config resolved against the wrong site, so transformers read the path as a Hub repo id.
# Nothing had been measured. Read the traceback above this line before believing any cause.
run_gate "6-memory-at-the-longest-trajectory" \
  "IF the bench reported an OOM: the student does not fit this card at the real trajectory lengths, the H20 ceiling (16169 tokens) does not hold here, and rollout.max_len must be re-derived for this channel. IF it failed before the first measurement, the memory question is still UNANSWERED -- do not record a ceiling from this run" \
  env CUDA_VISIBLE_DEVICES=1 python scripts/bench_train_step.py \
    --config configs/opd_qwen3.yaml --repeats 2 --find-max-length \
    --out "runs/bench_${CHANNEL}.json"

# ---------------------------------------------------------------------------------------
echo
if [ "$SKIPPED" -gt 0 ]; then
  echo "================================================================"
  echo "[certify] PARTIAL: $PASSED passed, $SKIPPED SKIPPED. The $CHANNEL channel is NOT certified."
  echo "================================================================"
  echo "[certify] A skipped gate is not a passed one. The skips are in runs/gate_log.jsonl with"
  echo "[certify] verdict SKIP and the reason, so a later reader cannot mistake absence for"
  echo "[certify] success -- which is the shape of nearly every failure this project has had."
  echo "[certify] Everything that COULD run did, so this allocation was not wasted: re-running"
  echo "[certify] after the dependency is satisfied re-does the cheap gates in minutes."
  exit 1
fi
echo "================================================================"
echo "[certify] ALL $PASSED GATES PASSED -- the $CHANNEL channel is certified."
echo "================================================================"
echo "[certify] Verdicts are in runs/gate_log.jsonl, stamped with this node's fingerprint."
echo "[certify] STILL REQUIRED before running an experiment here:"
echo "[certify]   * the teacher's engine settings differ from other channels' (utilization"
echo "[certify]     $TEACHER_UTIL), so the BackendConfig fingerprint differs and gate 1 will"
echo "[certify]     refuse a cache scored elsewhere. Regenerate the cache on THIS channel;"
echo "[certify]     do not copy one across."
echo "[certify]   * re-read runs/bench_${CHANNEL}.json: the per-trajectory cost and the"
echo "[certify]     trainable length ceiling are ${CHANNEL} numbers now, and the schedule must"
echo "[certify]     be rebuilt from them rather than from the H20 measurements."
echo "[certify]   * write the measured utilization and max_len into"
echo "[certify]     configs/channels/${CHANNEL}.yaml -- they are PROVISIONAL there until this"
echo "[certify]     run replaces them, and a provisional value that never got replaced is how"
echo "[certify]     a run dies hours in."
