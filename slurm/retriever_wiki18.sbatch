#!/usr/bin/env bash
#SBATCH --job-name=opd-retriever-wiki18
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
# GPUS: 1                        # ANY card. On the H20 cluster the cards the training needs
#                                           # are kept free by an EXCLUDE LIST, which is site
#                                           # vocabulary and therefore lives in the site file, not
#                                           # here -- an `#SBATCH --exclude=<node>,...` line cannot
#                                           # be un-set from the command line, so it would be an
#                                           # invalid node name on any other cluster. Pass it at
#                                           # submit time where the site defines one:
#                                           #   bash slurm/submit.sh 1 slurm/retriever_wiki18.sbatch \
#                                           #        --exclude="$OPD_RETRIEVER_EXCLUDE"
#                                           # the A100 cluster's <a100-partition> partition has one card type, so
#                                           # its list is empty and nothing is passed.
#                                           #
#                                           # This partition REJECTS a submission with no --gres at all
#                                           # ("You must always specify a GPU card useage when requesting
#                                           # gpu partition"), so a bare gpu:1 is the minimum legal ask.
#                                           # The old comment here warned against bare gpu:1 because it
#                                           # "could land on an H20/A800" -- true then, impossible now:
#                                           # --exclude rules those four nodes out by construction, which
#                                           # is a stronger guarantee than pinning to one card type.
#                                           #
#                                           # The GPU is NEVER USED:
#                                           # patch_searchr1_cpu.py forces the E5 encoder onto CPU
#                                           # (model.cpu(), fp32, inputs stay on CPU) and HNSW/FAISS runs
#                                           # on CPU too (no --faiss_gpu), so this process never calls
#                                           # cuda_init. Card TYPE cannot affect retrieval numerics --
#                                           # same weights, pooling, "query:" prefix and L2-normalize,
#                                           # all in CPU fp32.
#                                           #
#                                           # It used to request gpu:l20:1 to keep off the H20/A800 the
#                                           # teacher and both training arms need. That intent is right;
#                                           # the mechanism was self-defeating -- asking for a card it
#                                           # never touches pinned it to ONE node (<node>) and left it
#                                           # PENDING(Priority) while <node> sat completely idle with 64
#                                           # free CPUs and 1.3 TB free memory. Eligible now on
#                                           # <node>/190/196/197/agpu7/agpu8 instead of one node.
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G                          # the 76 GB HNSW index loads into host RAM -- the memory hog.
#                                           # Was 512G (=524288 MB), which SILENTLY EXCLUDED every node
#                                           # with ~515 GB (<node>/187/188, incl. an IDLE one) by ~9 GB
#                                           # and left this job queueing on (Resources). 256G still gives
#                                           # 3.4x the index, and fits EVERY node in the partition.
#SBATCH --time=08:00:00                     # OVERRIDE AT SUBMIT for a long campaign:
#                                           #   sbatch --time=36:00:00 slurm/retriever_wiki18.sbatch
#                                           # (the ceiling is 48h; ask for what the campaign
#                                           # needs, since a longer request backfills worse)
#                                           #
#                                           # Was 48:00:00, and that was the real reason this job sat
#                                           # PENDING with resources available. Its reason code was
#                                           # (Priority), not (Resources): the node had room, but a
#                                           # lower-priority job can only start via BACKFILL, and
#                                           # backfill starts a job only if it can prove it finishes
#                                           # before the higher-priority reservation. A 48h job can
#                                           # essentially never prove that on a partition where everyone
#                                           # holds 3-5 day allocations. 8h loads the index once and
#                                           # still covers a full sharded work session.
#SBATCH --output=slurm/logs/retriever-wiki18-%j.out
#SBATCH --error=slurm/logs/retriever-wiki18-%j.out
#
# CANONICAL (and ONLY) RETRIEVER after the 2026-07 migration. It REPLACED the now-removed
# slurm/retriever.sbatch, which served the misaligned self-built e5-large-v2 index and, sharing
# this same host file, was a silent footgun (no fingerprint gate guards the retriever). This one
# serves the community-standard
# wiki-18 + e5-base-v2 environment (download it first with scripts/download_wiki18.sh).
#
# STRUCTURAL GUARANTEE, not measured-acceptable: query-time encoding MUST match how the index
# was built. So we run Search-R1's OWN retrieval_server.py verbatim (cloned by the download
# script) rather than re-wrapping E5 ourselves -- their server hard-wires the parts that make
# an E5 index work: mean pooling, L2-normalized embeddings (=> inner-product/cosine on the
# IndexFlatIP-family HNSW), and the E5 "query: " prefix prepended INSIDE the server (keyed on
# --retriever_name "e5"). The client (envs/http_retriever.py) sends the RAW question and must
# NOT prepend "query: " itself -- doing so would double-prefix. topk in the request overrides
# the --topk default per call.
#
# Handoff unchanged: this job writes its hostname:8000 to the shared scratch file; stage3_eval
# and the gold-retrievability diagnostic read that file, so nothing hand-edits committed config.
# NOTE: Search-R1's server HARD-CODES host 0.0.0.0 and port 8000 (no CLI flag) -- so the port
# below is fixed to 8000 to match; changing it would require editing their source.

set -euo pipefail
source slurm/env.sh                         # conda activate OPD  (see env.sh)

# --- staged artifacts (defaults match scripts/download_wiki18.sh) --------------------------
ENCODER_DIR=${OPD_MODELS_DIR}
DATA_DIR=${OPD_WIKI18_DIR}
ENCODER="$ENCODER_DIR/e5-base-v2"           # 768-dim -- MUST match the index (NOT e5-large-v2)
# ANOTHER CORPUS on the same server code (BrowseComp-Plus):
# OPD_RETR_INDEX / OPD_RETR_CORPUS name its files, OPD_RETR_HOST_FILE its host file
# (opd_retriever_host_bcplus.txt), so the wiki-18 endpoint is never overwritten. The port is
# the server's fixed 8000, so the two servers must sit on different nodes (--exclude the other's).
INDEX_FILE="${OPD_RETR_INDEX:-$DATA_DIR/e5_HNSW64.index}"
CORPUS_FILE="${OPD_RETR_CORPUS:-$DATA_DIR/wiki-18.jsonl}"
SERVER="$DATA_DIR/Search-R1/search_r1/search/retrieval_server.py"
PORT=8000                                   # hard-coded in Search-R1's server -- do not change

# --- fail-fast: prove every artifact exists BEFORE queuing a 76 GB index load --------------
for f in "$ENCODER/config.json" "$INDEX_FILE" "$CORPUS_FILE" "$SERVER"; do
  [[ -e "$f" ]] || { echo "FATAL: missing $f -- run scripts/download_wiki18.sh first"; exit 1; }
done
# Search-R1's server needs fastapi/uvicorn/datasets/pydantic ON TOP OF torch/transformers/faiss.
python -c "import faiss, torch, transformers, datasets, numpy, fastapi, uvicorn, pydantic" \
  || { echo "ERROR: env missing a Search-R1 server dep (pip install fastapi uvicorn datasets pydantic faiss-cpu)"; exit 1; }

# compute nodes are offline -- force local encoder resolution (retriever_model is a local path)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Search-R1's load_model() hard-calls model.cuda(); this 4090d node's driver (CUDA 12.4) is
# older than the OPD torch build (CUDA 13), so cuda_init() dies on startup. Patch the encoder
# to CPU (identical encode math -- same weights/pooling/prefix/normalize -- just a device
# change; e5-base-v2 on one query at a time is plenty fast on CPU). Idempotent + fails loud.
python scripts/patch_searchr1_cpu.py "$SERVER"

# --- publish this node's endpoint so training/eval jobs can find the server ----------------
HOST_FILE=${OPD_RETR_HOST_FILE:-${OPD_RUNSTATE_DIR}/opd_retriever_host.txt}
echo "http://$(hostname -f):${PORT}/retrieve" > "$HOST_FILE"
echo "[retriever] endpoint published to $HOST_FILE : $(cat "$HOST_FILE")"
echo "[retriever] encoder=$ENCODER"
echo "[retriever] index=$INDEX_FILE  corpus=$CORPUS_FILE"
echo "[retriever] server=$SERVER (commit $(cat "$DATA_DIR/Search-R1.commit.txt" 2>/dev/null || echo '?'))"

# --- launch Search-R1's retriever verbatim (HNSW = CPU, so NO --faiss_gpu) -----------------
# Blocks and serves POST /retrieve on :8000 until the job is cancelled. Loads the index once.
python "$SERVER" \
  --index_path     "$INDEX_FILE" \
  --corpus_path    "$CORPUS_FILE" \
  --retriever_name e5 \
  --retriever_model "$ENCODER" \
  --topk 3
