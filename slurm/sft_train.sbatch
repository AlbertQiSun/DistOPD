#!/usr/bin/env bash
#SBATCH --job-name=opd-sft-pi-ref
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
# GPUS: 6                  # pi_ref = full SFT of Qwen3-4B-Base. H20 is a dead end here:
#                                        # only ~2 H20 nodes exist, one is long-occupied, and the other
#                                        # (<node>) repeatedly hands out a faulty/under-counted card
#                                        # (NVML fault -> torch sees <4) -- not luck, a sick node. A800
#                                        # is also uncontendable (never 4 free on one node). L20 (Ada,
#                                        # 48 GB) is schedulable. Capped at 6 (not 8) because our
#                                        # retriever+teacher servers eat part of the per-user GRES quota
#                                        # (QOSMaxGRESPerUser) -- 6 is what's borrowable now, and it fits:
#                                        # 6 L20 = 4B full-FT via DeepSpeed ZeRO-2, the 64 GB opt/grad
#                                        # state SHARDED across the 6 ranks (~17 GB + acts ~27 GB/card).
#                                        # ZeRO-2 touches only the optimizer/grad wrap, NOT the data
#                                        # path -- both mask gates still hold. PIN the type.
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/sft-pi-ref-%j.out
#
# M-SFT Stage 2: SFT Qwen3-4B-Base on the Stage-1 MiroThinker trajectories -> pi_ref.
#
# THE STAGE-2 RED LINE: the action-only mask (OBSERVATION/retrieved-doc tokens are conditioned
# on but NEVER supervised) must survive LlamaFactory. LlamaFactory has NO dataset_info route that
# ingests raw input_ids/labels -- every converter tokenizes from TEXT (ids drift + role re-mask).
# The ONLY contract that loads our exact ids/labels verbatim is `tokenized_path`: get_dataset()
# load_from_disk()s it and returns early, skipping all tokenization/preprocessing/packing. So we
# PRE-BUILD that arrow dataset, train off it, then GATE the mask against what LlamaFactory loaded.
# No pi_ref checkpoint is trusted until scripts/verify_llamafactory_mask.py prints VERDICT: PASS.

set -euo pipefail

# ---- site paths ONLY. Deliberately not slurm/env.sh: that activates the OPD env, which would
# ---- shadow LlamaFactory's. The site file is pure `export`s -- no conda, no guards.
# ---- (Path is repo-relative, not $0-relative: sbatch runs the SPOOL copy of this file but
# ---- with cwd = the submit directory, so $0 does not locate the repo.)
. "slurm/sites/${OPD_SITE:-example}.sh"

# ---- LlamaFactory env (its OWN stack, NOT the OPD env). Set this to your install. ----
# LlamaFactory 0.9.4.dev0 (datasets 3.6.0, torch 2.8.0+cu128, deepspeed 0.16.9). A NAME works
# where the env lives under conda's envs dir; a PREFIX PATH works everywhere and is what a
# scratch-hosted conda needs, so both are accepted -- conda activate takes either.
LF_ENV="${OPD_LF_ENV:-train}"
# Accept a conda env (name or prefix) OR a plain venv. The venv path exists because the A100 cluster's
# conda lost its channel configuration when it was moved to scratch, and building this stack
# needs nothing from conda beyond a Python 3.10 -- `OPD/bin/python -m venv` gives one with an
# isolated site-packages, so LlamaFactory's torch cannot be shadowed by the OPD env's. Detected
# by what is actually in the directory rather than by a flag, so neither form has to be declared.
if [ -f "${LF_ENV}/bin/activate" ] && [ ! -d "${LF_ENV}/conda-meta" ]; then
  echo "[sft] activating VENV ${LF_ENV}"
  # shellcheck disable=SC1090
  . "${LF_ENV}/bin/activate"
else
  echo "[sft] activating CONDA env ${LF_ENV}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$LF_ENV"
fi
# PYTHONPATH / PYTHONHOME must not survive into this env. Neither the repo nor the site file
# sets them, but sbatch inherits the submitting shell, and a value pointing at the OPD env
# makes THIS interpreter load the OTHER one's stdlib -- observed 2026-08-22 on the login node,
# where pip from envs/train imported platform.py from envs/OPD and died parsing its own
# version string (3.10.21 vs 3.10.20). The whole reason this sbatch avoids slurm/env.sh is to
# keep the two stacks apart; leaving these set would reopen the same door.
unset PYTHONPATH PYTHONHOME

# CUDA_HOME, because DeepSpeed demands one before it will even import. Its op_builder runs an
# is_compatible() probe at import time (deepspeed/git_version_info.py -> fp_quantizer.py ->
# installed_cuda_version()), which shells out to $CUDA_HOME/bin/nvcc and raises
# MissingCUDAException when the variable is unset. ZeRO-3 compiles nothing here -- the probe is
# what needs it -- but it kills every rank before training starts, after the export and the
# tokenised dataset have already been built.
#
# Two sources, in order: whatever the caller or a cluster module already set, then the
# nvidia-cuda-nvcc wheel inside this env, which is self-contained and needs no module system.
if { [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; } && [ -n "${OPD_CUDA_HOME:-}" ] && [ -x "${OPD_CUDA_HOME}/bin/nvcc" ]; then
  export CUDA_HOME="$OPD_CUDA_HOME"          # the site's toolkit (the RFT SFT died without it, 2026-09-10)
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  # SEARCH rather than guess a layout. nvidia-cuda-nvcc-cu12 12.8.93 does NOT put nvcc at
  # nvidia/cuda_nvcc/bin/nvcc, which is where the first version of this looked, and a fixed
  # path that is wrong fails exactly like the package being absent. Whatever the wheel does
  # with its directories, DeepSpeed only ever wants a root whose bin/nvcc is executable.
  _envroot="$(python -c 'import sys; print(sys.prefix)')"
  _nvcc="$(find "$_envroot" -type f -name nvcc -perm -u+x 2>/dev/null | head -1)"
  if [ -n "$_nvcc" ]; then
    export CUDA_HOME="$(dirname "$(dirname "$_nvcc")")"
  fi
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "FATAL: DeepSpeed needs CUDA_HOME with bin/nvcc and none was found." >&2
  echo "  Either load the cluster toolkit (module avail cuda) before submitting, or install" >&2
  echo "  the self-contained wheel into ${LF_ENV}:" >&2
  echo "    python -m pip install 'nvidia-cuda-nvcc-cu12==12.8.*'" >&2
  exit 1
fi
echo "[sft] CUDA_HOME=$CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | tail -1))"
python - <<'PYCHK'
import sys, platform
assert platform.__file__.startswith(sys.prefix), (
    f"stdlib leak: sys.prefix={sys.prefix} but platform.py is {platform.__file__}. "
    "Something outside this env is on the interpreter's path.")
print(f"[sft] interpreter clean: {sys.prefix}")
PYCHK
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# expandable_segments defrags the allocator so the big per-step logits spike (seq_len x 151936
# vocab, ~5-10 GB on a 16k trajectory) doesn't fail on fragmentation. Belt-and-suspenders with
# ZeRO-3 (which is what actually frees the room by sharding params).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "python: $(command -v python)"; python --version
python -c "import llamafactory, datasets, torch; print('llamafactory', llamafactory.__version__, '| datasets', datasets.__version__, '| torch', torch.__version__)"

# ---- DEEPSPEED VERSION GATE (fail EARLY on login, before export+torchrun). LlamaFactory 0.9.4.dev0
#      HARD-pins deepspeed>=0.10.0,<=0.16.9 as a MANDATORY check (DISABLE_VERSION_CHECK can't bypass
#      mandatory). A newer deepspeed (e.g. 0.19.2) aborts every rank AFTER a wasted export+launch.
#      Assert the range here so a bad env fails in seconds. Fix: pip install 'deepspeed==0.16.9'. ----
python - <<'PY'
from importlib.metadata import version
from packaging.version import Version
ds = version("deepspeed")
assert Version("0.10.0") <= Version(ds) <= Version("0.16.9"), (
    f"LlamaFactory 0.9.4.dev0 hard-pins deepspeed>=0.10.0,<=0.16.9 (MANDATORY) but found {ds}. "
    "Fix: conda activate train && pip install 'deepspeed==0.16.9'.")
print(f"[sft] deepspeed {ds} within LlamaFactory's [0.10.0, 0.16.9] pin")
PY

# ---- GPU PREFLIGHT: fail EARLY + LOUD on a bad node. A faulty card (NVML can't init / a GPU
#      handle is Unknown) otherwise crashes deep in model load with a cryptic "invalid device
#      ordinal". Assert torch sees all 6 requested L20s and can query each one; if not, this node
#      is unhealthy -- resubmit (SLURM may give healthy cards) or --exclude the bad node. ----
# RANKS AND ACCUMULATION ARE NODE-TOPOLOGY BOUND, so both are overridable and the effective
# batch is asserted rather than assumed.
#
#   the H20 cluster: 6 L20 on one node -> 1 x 5 x 6 = 30
#   the A100 cluster:  a100 nodes hold FOUR GPUs, so 6 cannot be scheduled on one node at all
#             (--gres=gpu:a100:6 never starts) -> 1 x 8 x 4 = 32
#
# 32 was the target all along; 30 was the accommodation 6 ranks forced. Aggregate memory is
# larger here too (4 x 80 GiB against 6 x 48), so ZeRO-3 has more room per rank, not less.
# This DOES change the optimisation -- a different effective batch is a different run -- so
# it is stated here and recorded, not absorbed.
NPROC="${OPD_SFT_NPROC:-6}"
GRAD_ACCUM="${OPD_SFT_GRAD_ACCUM:-5}"
EFF_BATCH=$(( 1 * GRAD_ACCUM * NPROC ))
echo "[sft] ranks=$NPROC grad_accum=$GRAD_ACCUM -> effective batch $EFF_BATCH"
if [ "$EFF_BATCH" -lt 28 ] || [ "$EFF_BATCH" -gt 36 ]; then
  echo "FATAL: effective batch $EFF_BATCH is outside [28,36]; the target is 32. Set
  OPD_SFT_GRAD_ACCUM so that grad_accum * ranks lands there." >&2; exit 1
fi

nvidia-smi || echo "[sft] WARN: nvidia-smi/NVML failed on this node (bad driver state?)"
python - <<'PY'
import os
import torch
n = torch.cuda.device_count()
print(f"[sft] torch sees {n} CUDA device(s)")
# The number to demand is the number of ranks torchrun will launch, not a constant. It was
# hardcoded at 6 for the H20 cluster's L20 node and refused a correctly-allocated 4-GPU A100 node.
want = int(os.environ.get("OPD_SFT_NPROC", "6"))
assert n >= want, (f"need {want} GPUs but torch sees {n} -- either this node's GPUs are "
                   "unhealthy (NVML/driver fault, which is what this check exists for) or "
                   "fewer cards were allocated than OPD_SFT_NPROC asks torchrun to use. "
                   "Resubmit, or add '--exclude=<node>'.")
# RETRY EACH DEVICE BEFORE CONDEMNING THE NODE. Observed 2026-08-22: cuda:0 answered
# (free=78GiB/79GiB) and cuda:1 raised cudaErrorDevicesUnavailable, on a node whose nvidia-smi
# showed all four cards idle, 0 MiB used, no processes, compute mode Default. The same error
# class hit the generation jobs three times that day and cleared on retry every time -- the
# devices are not ready the instant the allocation is handed over. A preflight that condemns
# the node on the first failure turns a 30-second wait into a lost queue slot.
#
# This does NOT weaken the check it was written for. A card whose NVML handle is genuinely
# Unknown still fails all the attempts and still fails the job here, before model load, which
# is the whole point.
import time
for i in range(n):
    for attempt in range(1, 7):
        try:
            free, total = torch.cuda.mem_get_info(i)
            if attempt > 1:
                print(f"  cuda:{i} became available on attempt {attempt}")
            break
        except Exception as exc:
            if attempt == 6:
                raise AssertionError(
                    f"cuda:{i} still unavailable after 6 attempts over ~50s on "
                    f"{os.environ.get('SLURMD_NODENAME', 'this node')}: {exc}. nvidia-smi above "
                    "shows what the driver thinks. If it shows the card idle, this node is "
                    "sick -- resubmit or --exclude it.") from exc
            print(f"  cuda:{i} not ready (attempt {attempt}/6): {type(exc).__name__}; "
                  "waiting 10s", flush=True)
            time.sleep(10)
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)}  free={free>>30}GiB/{total>>30}GiB")
print(f"[sft] GPU preflight OK: {n} card(s) healthy and queryable, {want} needed.")
PY

# PER-LINE. Three teacher lines produce three shards; the default is the historical
# single-line path so nothing that used to work changes.
STAGE1="${OPD_SFT_OUT:-runs/sft_shard1}"
SHARD=$STAGE1/llamafactory_shard
TOKENIZED=$SHARD/tokenized
CUTOFF=16384                              # MUST match rollout.max_len (Stage-1) and the export cutoff
BASE=${OPD_SFT_BASE:-${OPD_MODELS_DIR}/Qwen3-4B-Base}   # continued SFT (RFT / SeqKD) starts from pi_ref
MAXSTEPS="${OPD_SFT_MAX_STEPS:-0}"                    # 0 = the epochs rule; else scripts/sft_budget.py's number
CKPT_NAME="${OPD_SFT_CKPT_NAME:-pi_ref_ckpt}"
SAVESTEPS="${OPD_SFT_SAVE_STEPS:-500}"                # intermediate checkpoints every N steps (SeqKD: one per epoch)
echo "[sft] base=$BASE max_steps=$MAXSTEPS ckpt=$CKPT_NAME stage1=$STAGE1"
DS_CONFIG=configs/ds_z3_config.json       # ZeRO-3: shard params TOO (not just opt/grad). ZeRO-2 fit
#                                        # the resident state (~12.5 GB/card) but OOM'd at step 81 on a
#                                        # long trajectory: the LM-head logits spike (16k x 151936 vocab
#                                        # ~= 5-10 GB fwd+bwd) topped 44 GB. ZeRO-3 frees the 8 GB param
#                                        # replica (-> ~1.3 GB/card) so the spike fits the longest sample.
#                                        # Saves a normal consolidated bf16 ckpt (gather_16bit on save).

# ---- STEP 1: (re)export the pre-tokenized shard + the corrected tokenized_path yaml. Idempotent;
#      regenerates pretokenized.jsonl + qwen3_4b_sft.yaml from the source records (single source
#      of truth). Refuses any record > CUTOFF (would clip ACTION tokens). --deepspeed emits the
#      ZeRO-2 `deepspeed:` line; --grad-accum keeps effective batch at 32 on 8 ranks. ----
python -m sft.export_llamafactory $STAGE1 --out $SHARD --cutoff-len $CUTOFF --base-model $BASE \
  --grad-accum $GRAD_ACCUM --deepspeed $DS_CONFIG --max-steps "$MAXSTEPS" --ckpt-name "$CKPT_NAME" --save-steps "$SAVESTEPS"

# ---- STEP 2: pre-build the tokenized_path arrow dataset LlamaFactory will load VERBATIM.
#      Built HERE (LlamaFactory env) so the datasets version that writes save_to_disk == the one
#      that reads load_from_disk at train time (a login-node preflight build in the OPD env can
#      leave a cross-version arrow whose internal state schema this env won't load). Wipe any stale
#      dir first so save_to_disk never collides with a pre-existing/foreign arrow. ----
rm -rf "$TOKENIZED"
python -m sft.build_tokenized_dataset $SHARD/pretokenized.jsonl --out $TOKENIZED
test -d "$TOKENIZED" || { echo "FATAL: $TOKENIZED not built -- LlamaFactory would fall back to "\
  "re-tokenizing the dataset_info stub and corrupt the mask. Aborting."; exit 1; }
echo "[sft] tokenized_path ready: $TOKENIZED"

# ---- STEP 3: full SFT under 6-way DeepSpeed ZeRO-3. FORCE_TORCHRUN spawns one process per card;
#      effective batch = per_device_train_batch_size * gradient_accumulation_steps * 6 (set in the
#      yaml). tee the run log so STEP 3b can PROVE which data path LlamaFactory actually took. ----
TRAIN_LOG=$SHARD/train_${SLURM_JOB_ID:-manual}.log
FORCE_TORCHRUN=1 NPROC_PER_NODE=$NPROC \
  llamafactory-cli train $SHARD/qwen3_4b_sft.yaml 2>&1 | tee "$TRAIN_LOG"

# ---- STEP 3b: BEHAVIORAL backstop (closes the version-drift hole). STEP 4 below verifies the
#      arrow WE built -- but if a LlamaFactory version silently ignored tokenized_path and fell
#      back to re-tokenizing dataset_info, STEP 4 would still PASS (it checks the arrow, not what
#      the trainer internally consumed). So ASSERT from the run log that LlamaFactory took the
#      load-from-disk path (logs "Loaded tokenized dataset from ...") and did NOT re-tokenize
#      (would log "Running tokenizer on dataset" / "Saving dataset to disk"). ----
if ! grep -Eq "Loaded tokenized dataset from|Loading dataset from disk will ignore" "$TRAIN_LOG"; then
  echo "FATAL: no load-from-disk marker in the train log -- LlamaFactory did NOT use tokenized_path."
  echo "  It fell back to re-tokenizing the dataset_info stub -> id drift + role re-mask -> the"
  echo "  action-only mask is CORRUPT and pi_ref is INVALID. Check your LlamaFactory version's"
  echo "  tokenized_path contract; do NOT trust $SHARD/pi_ref_ckpt."
  exit 1
fi
if grep -Eq "Running tokenizer on dataset|Converting format of dataset|Tokenized dataset is saved at" "$TRAIN_LOG"; then
  echo "FATAL: the train log shows LlamaFactory RE-TOKENIZED/SAVED the dataset (it did not load the"
  echo "  pre-built arrow verbatim). The mask cannot be trusted; pi_ref is INVALID. Aborting."
  exit 1
fi
echo "[sft] behavioral backstop OK: LlamaFactory loaded tokenized_path verbatim (no re-tokenize)."

# ---- STEP 3c: ASSERT TRAINING ACTUALLY RAN. STEP 3b + STEP 4 both verify only the DATA path
#      (which data LlamaFactory loaded, that the mask survived) -- NEITHER verifies the optimizer
#      stepped. A stale FINAL checkpoint in output_dir made a run auto-resume to a 0-step no-op
#      (train_runtime ~0, loss 0, "Will skip the first N epochs") while BOTH gates passed and
#      pi_ref stayed a prior run's weights of unverified provenance. overwrite_output_dir=true (in
#      the exported yaml) prevents the silent resume; THIS is the behavioral proof the optimizer
#      actually stepped -- a real 4B/357-step train logs periodic "{'loss': ...}" and a multi-minute
#      runtime; a no-op logs neither. Closes the gap that let the no-op pass every structural gate. ----
if grep -Eq "Will skip the first [0-9]+ epochs" "$TRAIN_LOG"; then
  echo "FATAL: LlamaFactory RESUMED to an already-complete checkpoint and SKIPPED training (0"
  echo "  optimization steps). pi_ref would be a PRIOR run's weights, not a fresh train on this"
  echo "  shard. Remove $SHARD/pi_ref_ckpt (unverified provenance) and re-run -- the yaml now sets"
  echo "  overwrite_output_dir so a fresh run will not auto-resume. Aborting."
  exit 1
fi
if ! grep -Eq "'loss': [0-9]" "$TRAIN_LOG"; then
  echo "FATAL: no intermediate training-loss line in the train log -- the optimizer never stepped"
  echo "  (logging_steps=10 should have emitted periodic loss over 357 steps). Do NOT trust"
  echo "  pi_ref; investigate before Stage 3. Aborting."
  exit 1
fi
echo "[sft] training-happened backstop OK: optimizer stepped (periodic loss logged, no full-skip)."

# ---- STEP 4: THE MASK GATE. Now that STEP 3b proved the trainer consumed exactly $TOKENIZED,
#      verify that arrow preserves the action-only mask per record vs the Stage-1 records.
#      VERDICT: PASS => every OBSERVATION token stayed masked; supervised set == ACTION. STEP 3b
#      + STEP 4 together => LlamaFactory trained on verified-correct data. Only then is the pi_ref
#      checkpoint ($SHARD/pi_ref_ckpt) trustworthy. FAIL => do NOT use it. ----
python scripts/verify_llamafactory_mask.py $STAGE1 --processed $TOKENIZED --cutoff-len $CUTOFF

echo "STEP 1-4 done. checkpoint -> $SHARD/$CKPT_NAME . Both gates (3b behavioral + 4 mask)"
echo "must be green -- read the VERDICT above before Stage 3."
