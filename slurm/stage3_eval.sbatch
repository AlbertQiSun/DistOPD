#!/usr/bin/env bash
#SBATCH --job-name=opd-stage3-eval
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
# GPUS: 1                    # INFERENCE only: vLLM loads Qwen3-4B (~8 GB bf16) + KV
#                                           # cache. One card is plenty. H20 is fine here (pure
#                                           # decode, no training). The retriever is a SEPARATE job
#                                           # on a CHEAP 4090d (slurm/retriever_wiki18.sbatch) -- do NOT put
#                                           # the FAISS server on an H20; its bottleneck is 512G host
#                                           # RAM, not GPU. No teacher needed for Stage 3.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/stage3-%j.out
#SBATCH --error=slurm/logs/stage3-%j.out
#
# M-SFT Stage 3: pi_ref QUALITY ACCEPTANCE -- the MANDATORY gate before OPD/Row 10.
# "checkpoint loads OK / params=4022468096" only proves a valid 4B, NOT that pi_ref learned
# to SEARCH (a random 4B also loads). If pi_ref is a soft foundation, the states it induces
# are dirty from step 0, the OFFLINE premise collapses, and we would misread "offline also
# fails" as off-dynamics when the real cause is pi_ref. So this GATES OPD; fail -> fix pi_ref,
# do NOT enter Row 10.
#
# WHAT THIS RUNS (scripts/stage3_eval.py -- SAME rollout stack the trainer uses):
#   {hotpotqa, nq, musique} DEV  x  {pi_ref, Qwen3-4B-Instruct baseline},  200 Q x K=4 @ temp 0.8,
#   scoring three gates: A=format legality, B=retrieval effectiveness (baseline-independent
#   LOWER BOUND), C=answer EM/F1 vs the FAIR Instruct baseline. Then a side-by-side --compare.
#
# PREREQUISITE: the retriever server (slurm/retriever_wiki18.sbatch) MUST already be up -- it
# publishes its endpoint to $HOST_FILE below. This job reads that file, verifies the server
# responds, and ONLY THEN spends the GPU. Submit the retriever first and wait for its index
# to finish loading (minutes).

set -euo pipefail
source slurm/env.sh                          # conda activate OPD (same stack as the trainer)

# ---- paths (all pre-staged on scratch; eyeball before submit) ----------------------------
CONFIG=configs/opd_qwen3.yaml
PI_REF=${OPD_PI_REF:-${OPD_STUDENT_CKPT}}
BASELINE=${OPD_MODELS_DIR}/Qwen3-4B          # NO suffix = Instruct/post-trained
DATA_DIR=${OPD_SFT_DATA_DIR}
OUT_DIR=${OPD_STAGE3_OUT:-runs/stage3}
LIMIT=200
K=4

HOST_FILE=${OPD_RUNSTATE_DIR}/opd_retriever_host.txt

# ---- RETRIEVER PREFLIGHT: fail EARLY + LOUD if the server isn't up (don't waste the GPU) --
test -f "$HOST_FILE" || { echo "FATAL: $HOST_FILE missing -- start slurm/retriever_wiki18.sbatch first "\
  "and wait for it to publish its endpoint."; exit 1; }
ENDPOINT=$(cat "$HOST_FILE")
echo "[stage3] retriever endpoint = $ENDPOINT"
# a real POST proves the index finished loading (not just that the port is open)
curl -sf -X POST "$ENDPOINT" -H 'Content-Type: application/json' \
  -d '{"queries":["capital of France"],"topk":3,"return_scores":true}' >/dev/null \
  || { echo "FATAL: retriever at $ENDPOINT did not answer a test /retrieve POST -- is the index "\
       "still loading, or the job dead? Check slurm/logs/retriever-*.out."; exit 1; }
echo "[stage3] retriever OK (answered a live /retrieve POST)"

# ---- GPU preflight: torch must see the 1 requested H20 ------------------------------------
nvidia-smi || echo "[stage3] WARN: nvidia-smi/NVML failed on this node"
python - <<'PY'
import torch
n = torch.cuda.device_count()
print(f"[stage3] torch sees {n} CUDA device(s)")
assert n >= 1, "no CUDA device visible -- bad node, resubmit"
free, total = torch.cuda.mem_get_info(0)
print(f"  cuda:0 {torch.cuda.get_device_name(0)}  free={free>>30}GiB/{total>>30}GiB")
PY

# ---- checkpoint sanity: pi_ref must exist (Stage 2 must have produced it) -----------------
test -d "$PI_REF"   || { echo "FATAL: pi_ref checkpoint $PI_REF not found -- run Stage 2 (sft_train.sbatch) first."; exit 1; }
test -d "$BASELINE" || { echo "FATAL: baseline $BASELINE not found (Qwen3-4B Instruct, NOT -Base)."; exit 1; }

# ---- run: {dataset} x {model}. Each eval decontaminates DEV against ALL THREE SFT TRAIN pools
#      (pi_ref was trained on the nq+hotpotqa+musique MIXED shard, so a dev question that leaked
#      into ANY pool was seen in training; matching-pool-only would miss cross-pool leaks and,
#      given pi_ref's memorization, inflate Gate A/C). Then rolls out 200 Q x K=4 @ temp 0.8. NO
#      teacher. ----------------------------------------------------------------------------------
run_one() {   # $1=dataset  $2=model_path  $3=label
  local ds=$1 model=$2 label=$3
  echo "===== stage3 eval: dataset=$ds label=$label ====="
  python scripts/stage3_eval.py \
    --config "$CONFIG" \
    --model "$model" --label "${ds}_${label}" \
    --eval-data   "$DATA_DIR/${ds}_dev_with_gold.jsonl" --dataset-name "$ds" \
    --sft-data    "$DATA_DIR/nq_with_gold.jsonl" "$DATA_DIR/hotpotqa_with_gold.jsonl" "$DATA_DIR/musique_with_gold.jsonl" \
    --retriever-endpoint "$ENDPOINT" \
    --limit "$LIMIT" --num-samples "$K" --out-dir "$OUT_DIR"
}

for DS in hotpotqa nq musique; do
  run_one "$DS" "$PI_REF"   pi_ref
  run_one "$DS" "$BASELINE" instruct
  echo "===== stage3 compare: $DS (pi_ref vs instruct) ====="
  python scripts/stage3_eval.py --compare \
    "$OUT_DIR/stage3_${DS}_pi_ref.json" \
    "$OUT_DIR/stage3_${DS}_instruct.json"
done

echo "[stage3] DONE. Reports in $OUT_DIR/stage3_*.json ; read the gate tables above."
echo "         pi_ref enters Row 10 ONLY if the three gates say it is a competent searcher."
