#!/usr/bin/env bash
#SBATCH --job-name=ref_topk
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/logs/ref_topk_%x.%j.out
#SBATCH --error=runs/logs/ref_topk_%x.%j.out
#
# delta-OPD step 1: pi_ref's top-K' at every action position of the line's frozen offline
# rollout cache (precomputed_<line>_full, 38,400 trajectories), written in RESUMABLE parts of
# 1000 trajectories to <cache>/ref_topk/part_*.npz; part 0 is gated (log-sum-exp, alignment
# against the tail_slope dump) before the rest is spent. Re-submitting the same job continues
# from the first missing part; the run that completes the last part writes <cache>/ref_topk.npz
# and ref_topk_report.txt.
#
#   OPD_SITE=example OPD_LINE=miro bash slurm/submit.sh 1 slurm/ref_topk_dump.sbatch \
#       --exclude=<bad nodes> -J reftopk_miro
#   OPD_REF_TOPK=64 ...          K' (default 32, step 0's verdict)
#   OPD_N=200 ...                first 200 trajectories only (a smoke)
#
# GPUS: 1
# WALLTIME: 4 h is a backfill-sized slice, not the whole job. Measured 2026-09-05 (progress
# lines of the first, non-resumable attempt): miro ~2.9 traj/s, dense ~2.3, moe ~1.25 -> the
# full cache is ~3.7 h / ~4.5 h / ~8.6 h of GPU time. Chain slices with --dependency=afterany
# (see the launch message) and let each continue where the last stopped.
# ONE card: pi_ref (4B, bf16) prefill only -- no teacher, no sampling, NO RETRIEVER.
# Paths come from the SITE (slurm/sites/<site>.sh via slurm/env.sh), never from this file.
source slurm/env.sh
LINE=${OPD_LINE:?set OPD_LINE=miro|dense|moe}
M=$OPD_MODELS_DIR
CACHE=${OPD_CACHE:-precomputed_${LINE}_full}
PIREF=${OPD_PIREF_CKPT:-$M/pi_ref_$LINE}
K=${OPD_REF_TOPK:-32}
ALIGN=${OPD_ALIGN_NPZ:-none}   # optional: an npz of reference top-K rows to cross-check against
N_LIMIT="${OPD_N:-}"
# OPD_DIAG=1: do not dump; re-score the positions where part 0 disagreed with the tail_slope
# dump (both chunk sizes, twice each, this node) and exit. OPD_IGNORE_GATES=1: continue a dump
# whose gates.json says failed -- only after that failure has been adjudicated.
DIAG="${OPD_DIAG:-}"; IGNORE="${OPD_IGNORE_GATES:-}"
# OPD_ALIGN_NPZ=none (default): skip the optional alignment cross-check
for p in "$CACHE/manifest.json" "$PIREF" $( [ "$ALIGN" != none ] && echo "$ALIGN" ); do
  test -e "$p" || { echo "FATAL: $p missing" >&2; exit 1; }
done
python scripts/gpu_preflight.py --device-tries "${OPD_DEVICE_TRIES:-20}" --device-delay "${OPD_DEVICE_DELAY:-15}"

python -u scripts/ref_topk_dump.py --line "$LINE" --cache "$CACHE" --pi-ref "$PIREF" --k "$K" --align "$ALIGN" \
    ${N_LIMIT:+--n "$N_LIMIT"} ${DIAG:+--diagnose-align} ${IGNORE:+--ignore-gates}
rc=$?
echo "[reftopk] rc=$rc (0 = complete and gates passed; 1 = partial or a gate failed -- read $CACHE/ref_topk/gates.json)"
exit $rc
