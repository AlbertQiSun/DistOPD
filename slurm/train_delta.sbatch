#!/usr/bin/env bash
#SBATCH --job-name=train_delta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/logs/delta-%j.out
#SBATCH --error=slurm/logs/delta-%j.out
#
# DistOPD training (the "delta" loss, opd/delta_loss.py). This is the offline arm with only
# the loss swapped: same rollout cache, same config (configs/opd_qwen3.yaml), same served-teacher
# gate 1 at the disk boundary as train_offline.sbatch. It additionally reads two caches written
# beside the precompute, <cache>/ref_topk (scripts/ref_topk_dump.py) and <cache>/teacher_head
# (scripts/teacher_head_dump.py). Both must be complete or the trainer refuses to start.
#
# Example (MiroThinker teacher, beta = 0.5):
#   OPD_LINE=miro OPD_BETA=0.5 OPD_SITE=example bash slurm/submit.sh 1 slurm/train_delta.sbatch -J distopd_miro
# The line's teacher server (slurm/teacher_server.sbatch) must be up, as for train_offline.sbatch.
#
# Outputs: runs/opd_<line>_delta_b<tag>/{ckpt_offline, diag_offline.jsonl}. Evaluate the
# checkpoint with slurm/stage3_eval.sbatch.
#
# GPUS: 1
# Paths come from the SITE (slurm/sites/<site>.sh via slurm/env.sh), never from this file.
source slurm/env.sh
LINE=${OPD_LINE:?set OPD_LINE=miro|dense|moe}
BETA="${OPD_BETA:-1.0}"
# OPD_LOSS=opd runs the LIGHTNING loss through this same file (the same-step control for the
# delta gate: OPD_LOSS=opd OPD_STEPS=30 OPD_SAVE_EVERY=0 OPD_OUT_DIR=runs/opd_<line>_lightning30).
LOSS="${OPD_LOSS:-delta}"
DATA_SEED="${OPD_DATA_SEED:-}"          # seed-2 arm: permutes the cache order (+ run.seed)
BETA_END="${OPD_BETA_END:-}"            # annealing arm: beta -> BETA_END linearly over the run
ABL="${OPD_ABLATION:-none}"             # opd.delta_loss.ABLATIONS: none | teacher_head | no_bucket | no_anchor (M3) | supp1 | supp2 | supp4 (batch 3) | rho50..rho99 (the rho sweep)
BTAG=$(python -c "b=float('$BETA'); print(('%g' % b).replace('.', ''))")     # 1.0 -> 1, 0.5 -> 05
M=$OPD_MODELS_DIR
case "$LINE" in
  miro)  TEACHER=${OPD_TEACHER_DIR:-$M/MiroThinker-1.7-mini} ;;
  moe)   TEACHER=${OPD_TEACHER_DIR:-$M/Qwen3-30B-A3B-Thinking} ;;
  dense) TEACHER=${OPD_TEACHER_DIR:-$M/Qwen3-32B} ;;
  *) echo "FATAL: unknown line $LINE" >&2; exit 1 ;;
esac
export OPD_TEACHER_MODEL="$TEACHER"
export OPD_STUDENT_CKPT="${OPD_PIREF_CKPT:-$M/pi_ref_$LINE}"        # the config's model.student
CACHE=${OPD_PRECOMPUTE_DIR:-precomputed_${LINE}_full}
OUT=${OPD_OUT_DIR:-runs/opd_${LINE}_delta_b${BTAG}$( [ "$ABL" != none ] && echo "_$ABL" )}

python scripts/dry_run_smoke.py
python scripts/gpu_preflight.py --device-tries "${OPD_DEVICE_TRIES:-6}" --device-delay "${OPD_DEVICE_DELAY:-10}"
opd_require_student_ckpt || exit 1
wait_for_teacher
test -e "$CACHE/manifest.json" || { echo "FATAL: $CACHE/manifest.json missing" >&2; exit 1; }
if [ "$LOSS" = "delta" ]; then
  for p in "$CACHE/ref_topk.npz" "$CACHE/teacher_head.npz"; do
    test -e "$p" || { echo "FATAL: $p missing -- the delta arm needs the precompute AND both step-1/step-2 caches complete" >&2; exit 1; }
  done
  python - <<PY || exit 1
import numpy as np
for n in ("ref_topk", "teacher_head"):
    z = np.load("$CACHE/%s.npz" % n, allow_pickle=True)
    ok = bool(z["usable"]) if "usable" in z.files else bool(z["gates_passed"])
    print("[delta] %s: usable=%s" % (n, ok))
    assert ok, "%s gates failed and not adjudicated" % n
PY
fi
RETRIEVER=$(cat ${OPD_RUNSTATE_DIR}/opd_retriever_host.txt 2>/dev/null || echo "http://unused")
TEACHER_EP=$(cat "$(opd_teacher_host_file)")
echo "[delta] line=$LINE loss=$LOSS beta=$BETA ablation=$ABL cache=$CACHE out=$OUT teacher=$TEACHER_EP"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LIMIT="${OPD_LIMIT:-38400}"
STEPS="${OPD_STEPS:-150}"
EPOCHS=$(python -c "import yaml; c=yaml.safe_load(open('configs/opd_qwen3.yaml')); print(${STEPS}*c['optim']['global_batch']/${LIMIT})")
SAVE_EVERY="${OPD_SAVE_EVERY:-25}"
echo "[delta] shape: limit=$LIMIT steps=$STEPS -> epochs=$EPOCHS (must be 1.0 to match the offline arm)"
python scripts/train.py --config configs/opd_qwen3.yaml --mode offline --out-dir "$OUT" \
  --retriever-endpoint "$RETRIEVER" --teacher-endpoint "$TEACHER_EP" \
  --channel "$OPD_CHANNEL" \
  --precompute-dir "$CACHE" \
  --loss "$LOSS" --beta "$BETA" --beta-end "${BETA_END:-nan}" --data-seed "${DATA_SEED:-0}" --delta-ablation "$ABL" \
  --save-every "$SAVE_EVERY" --keep-steps "${OPD_KEEP_STEPS:-30}" \
  --limit "$LIMIT" --steps "$STEPS" --epochs "$EPOCHS"
echo "[delta] DONE -- $OUT/diag_offline.jsonl (each record carries a delta block)"
