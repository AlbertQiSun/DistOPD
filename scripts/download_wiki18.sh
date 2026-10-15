#!/usr/bin/env bash
# Download the Search-R1 community-standard retrieval environment (wiki-18 + e5-base-v2).
# RUN ON A LOGIN NODE (compute nodes are offline). Idempotent: re-running skips finished
# steps. NON-DESTRUCTIVE: the 75 GB index part files are kept after concatenation; this
# script only PRINTS the rm command to reclaim that space -- it never deletes on its own.
#
# WHY THIS EXACT SETUP (see memory/reference_retrieval_convention.md): the self-built index
# was triple-misaligned with the community standard (2017 dump vs DPR 2018-12-20; 512-word
# vs 100-word passages; e5-large-v2/1024-dim vs e5-base-v2/768-dim) AND had a token/word
# truncation bug. We migrate UNCONDITIONALLY to the setup 4/6 static-corpus agentic-search
# RL papers use (Search-R1, ReSearch, StepSearch, s3): the Dec-2018 English Wikipedia dump
# in 100-word passages (21M), e5-base-v2 dense, served by Search-R1's OWN retrieval_server.py
# (borrow logic not deps -- their query-time encode/pool/prefix must match how the index was
# built, so we run THEIR code, not a re-implementation).
#
# TWO INDEPENDENT DESTINATIONS (by request):
#   --encoder-dir  the e5-base-v2 weights go here (a model cache dir shared with other jobs)
#   --data-dir     the corpus + FAISS index + Search-R1 server code go here (the big data)
#
# Downloads:
#   1. intfloat/e5-base-v2                       (encoder, 768-dim)      -> ENCODER_DIR/e5-base-v2
#   2. PeterJinGo/wiki-18-corpus                 (wiki-18.jsonl.gz 5 GB) -> DATA_DIR, gunzip'd
#   3. PeterJinGo/wiki-18-e5-index-HNSW64        (part_aa 43G + part_ab 33G) -> DATA_DIR, cat'd
#   4. PeterGriffinJin/Search-R1 (git clone)     (retrieval_server.py)   -> DATA_DIR/Search-R1
#
# Disk: ~76 GB parts + 76 GB concatenated index + ~18 GB corpus jsonl = plan for ~180 GB
# on DATA_DIR during the run (drops to ~110 GB once you rm the parts).
#
#   bash scripts/download_wiki18.sh \
#       --encoder-dir ${OPD_MODELS_DIR} \
#       --data-dir    ${OPD_WIKI18_DIR}
set -euo pipefail

# Site paths only -- NOT slurm/env.sh: this runs on the LOGIN node before the env may even
# exist, so activating a conda env here would be a chicken-and-egg. Resolved from $0 rather
# than cwd because this one is invoked directly, not through sbatch.
. "$(cd "$(dirname "$0")/.." && pwd)/slurm/sites/${OPD_SITE:-example}.sh"

ENCODER_DIR="$OPD_MODELS_DIR"
DATA_DIR="$OPD_WIKI18_DIR"
SEARCHR1_URL=https://github.com/PeterGriffinJin/Search-R1.git

while [[ $# -gt 0 ]]; do
  case "$1" in
    --encoder-dir) ENCODER_DIR="$2"; shift 2 ;;
    --data-dir)    DATA_DIR="$2";    shift 2 ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# HF_HOME comes from the site file -- always on scratch, never the home quota, even when
# --local-dir is used.

# Use the ACTIVE (conda OPD) python's huggingface_hub via snapshot_download -- do NOT shell out
# to ~/.local/bin/huggingface-cli, whose shebang may point at a python that lacks the module
# (that is exactly the ModuleNotFoundError you can hit). snapshot_download is a long-stable API.
python -c "import huggingface_hub" 2>/dev/null \
  || { echo "FATAL: huggingface_hub not importable by '$(command -v python)'. Activate the OPD env (conda activate OPD), or: pip install -U huggingface_hub"; exit 1; }
# Opt into the fast transfer backend ONLY if installed AND it has not been explicitly set.
# Set HF_HUB_ENABLE_HF_TRANSFER=0 before running to DISABLE it -- hf_transfer's many parallel
# buffers can trip a login node's per-process memory cap and get the 43 GB part SIGKILL'd.
if [[ -z "${HF_HUB_ENABLE_HF_TRANSFER:-}" ]] && python -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi
command -v git >/dev/null || { echo "FATAL: git not found"; exit 1; }

# hf_snapshot <repo_id> <model|dataset> <local_dir>  -- resumable, skips already-downloaded files.
# Login nodes often SIGKILL big/long downloads (memory/CPU reaper). snapshot_download resumes
# from the on-disk cache, so we RETRY: each fresh process starts with reset memory and continues
# from where the last one died -> a 75 GB pull completes across several attempts.
hf_snapshot() {
  local repo="$1" rtype="$2" dest="$3" tries=0 max="${HF_MAX_TRIES:-12}"
  while true; do
    if python - "$repo" "$rtype" "$dest" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rtype, dest = sys.argv[1], sys.argv[2], sys.argv[3]
p = snapshot_download(repo_id=repo, repo_type=rtype, local_dir=dest, max_workers=4)
print(f"[hf] downloaded {repo} ({rtype}) -> {p}")
PY
    then
      return 0
    fi
    tries=$((tries + 1))
    if [[ "$tries" -ge "$max" ]]; then
      echo "FATAL: '$repo' still failing after $max attempts -- see message above" >&2
      return 1
    fi
    echo "[hf] '$repo' interrupted (SIGKILL/OOM/throttle) -- resume attempt $tries/$max in 10s..." >&2
    sleep 10
  done
}

mkdir -p "$ENCODER_DIR" "$DATA_DIR"
echo "[dl] ENCODER_DIR=$ENCODER_DIR"
echo "[dl] DATA_DIR   =$DATA_DIR"

# --- 1. encoder: e5-base-v2 (768-dim; MUST match the index -- NOT e5-large-v2) -------------
ENC_OUT="$ENCODER_DIR/e5-base-v2"
if [[ -f "$ENC_OUT/config.json" && ( -f "$ENC_OUT/pytorch_model.bin" || -f "$ENC_OUT/model.safetensors" ) ]]; then
  echo "[dl] e5-base-v2 already present at $ENC_OUT -- skip"
else
  echo "[dl] downloading intfloat/e5-base-v2 -> $ENC_OUT"
  hf_snapshot intfloat/e5-base-v2 model "$ENC_OUT"
fi

# --- 2. corpus: wiki-18.jsonl.gz -> wiki-18.jsonl ------------------------------------------
CORPUS_JSONL="$DATA_DIR/wiki-18.jsonl"
if [[ -f "$CORPUS_JSONL" ]]; then
  echo "[dl] corpus already decompressed at $CORPUS_JSONL -- skip"
else
  echo "[dl] downloading PeterJinGo/wiki-18-corpus -> $DATA_DIR"
  hf_snapshot PeterJinGo/wiki-18-corpus dataset "$DATA_DIR"
  GZ="$DATA_DIR/wiki-18.jsonl.gz"
  [[ -f "$GZ" ]] || { echo "FATAL: expected $GZ after download; got:"; ls -la "$DATA_DIR"; exit 1; }
  echo "[dl] gunzip $GZ (keep the .gz) -> $CORPUS_JSONL"
  gunzip -k -f "$GZ"
  [[ -f "$CORPUS_JSONL" ]] || { echo "FATAL: gunzip produced no $CORPUS_JSONL"; exit 1; }
fi
echo "[dl] corpus lines: $(wc -l < "$CORPUS_JSONL")   (expect ~21,015,324)"

# --- 3. index: part_aa + part_ab -> e5_HNSW64.index ---------------------------------------
INDEX_FILE="$DATA_DIR/e5_HNSW64.index"
PA="$DATA_DIR/part_aa"; PB="$DATA_DIR/part_ab"
if [[ -f "$INDEX_FILE" ]]; then
  echo "[dl] index already assembled at $INDEX_FILE -- skip"
else
  if [[ -f "$PA" && -f "$PB" ]]; then
    # supports the wget fallback: fetch the parts by ANY means, then re-run to concatenate.
    echo "[dl] part_aa & part_ab already present -- skip download, concatenate only"
  else
    # The 43+33 GB parts via python/snapshot_download get SIGKILL'd by the login-node reaper
    # AND do not resume (restart from zero). wget -c is the fix: a few-MB-RSS process (dodges
    # the memory OOM) that RESUMES from the on-disk partial, so progress accumulates across
    # kills instead of restarting. The retry loop makes it unattended -- each round continues
    # part_aa/part_ab from wherever the last SIGKILL left them until wget reports "fully
    # retrieved" (exit 0). Set HF_TOKEN to lift the anonymous rate limit.
    command -v wget >/dev/null || { echo "FATAL: wget not found (needed for the resumable index download)"; exit 1; }
    AUTH=()
    [[ -n "${HF_TOKEN:-}" ]] && AUTH=(--header="Authorization: Bearer ${HF_TOKEN}")
    BASE="https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index-HNSW64/resolve/main"
    for part in part_aa part_ab; do
      echo "[dl] wget -c $part -> $DATA_DIR (resumable; survives repeated login-node SIGKILL)"
      tries=0
      until wget -c "${AUTH[@]}" -P "$DATA_DIR" "$BASE/$part"; do
        tries=$((tries + 1))
        if [[ "$tries" -ge "${DL_MAX_TRIES:-300}" ]]; then
          echo "FATAL: $part still incomplete after $tries wget attempts" >&2; exit 1
        fi
        echo "[dl] $part interrupted (SIGKILL/throttle) -- wget -c resume $tries in 5s (progress is kept)..." >&2
        sleep 5
      done
    done
    [[ -f "$PA" && -f "$PB" ]] || { echo "FATAL: expected part_aa & part_ab; got:"; ls -la "$DATA_DIR"; exit 1; }
  fi
  echo "[dl] concatenating part_aa + part_ab -> $INDEX_FILE (this rewrites ~76 GB, be patient)"
  cat "$PA" "$PB" > "$INDEX_FILE"
  SUM=$(( $(stat -c%s "$PA") + $(stat -c%s "$PB") ))
  GOT=$(stat -c%s "$INDEX_FILE")
  [[ "$SUM" -eq "$GOT" ]] || { echo "FATAL: size mismatch parts=$SUM index=$GOT -- concat failed"; exit 1; }
  echo "[dl] index assembled OK ($GOT bytes == sum of parts)"
  echo "[dl] to reclaim ~76 GB once the retriever has loaded the index successfully, run:"
  echo "        rm -f '$PA' '$PB'"
fi

# --- 4. Search-R1 server (borrow logic not deps: run THEIR retrieval_server.py verbatim) ---
SR1_DIR="$DATA_DIR/Search-R1"
SERVER="$SR1_DIR/search_r1/search/retrieval_server.py"
if [[ -f "$SERVER" ]]; then
  echo "[dl] Search-R1 already cloned at $SR1_DIR -- skip (pinned commit in Search-R1.commit.txt)"
else
  echo "[dl] cloning Search-R1 -> $SR1_DIR"
  git clone --depth 1 "$SEARCHR1_URL" "$SR1_DIR"
  ( cd "$SR1_DIR" && git rev-parse HEAD ) > "$DATA_DIR/Search-R1.commit.txt"
  echo "[dl] pinned Search-R1 commit -> $DATA_DIR/Search-R1.commit.txt ($(cat "$DATA_DIR/Search-R1.commit.txt"))"
  [[ -f "$SERVER" ]] || { echo "FATAL: $SERVER not found after clone (repo layout changed?)"; exit 1; }
fi

echo
echo "[dl] DONE. Artifacts:"
echo "     encoder : $ENC_OUT"
echo "     corpus  : $CORPUS_JSONL"
echo "     index   : $INDEX_FILE"
echo "     server  : $SERVER"
echo "[dl] next: submit slurm/retriever_wiki18.sbatch (it reads these same default paths)."
