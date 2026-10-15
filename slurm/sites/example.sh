# Example site file. Copy it to slurm/sites/<your_site>.sh, fill in every value, then
#   export OPD_SITE=<your_site>
#   source slurm/env.sh
# Every variable below is required. slurm/lib_site_guard.sh refuses to run a job whose
# declared paths do not exist on the current machine.

export OPD_SITE_NAME=example                       # must equal this file's name without .sh

# Root of everything this project writes. Every path below lives under it.
export OPD_SITE_ROOT=/path/to/scratch

# Conda env name or absolute path (conda activate accepts either).
export OPD_CONDA_ENV="${OPD_CONDA_ENV:-OPD}"

export OPD_MODELS_DIR="$OPD_SITE_ROOT/models"            # teacher, pi_ref, retrieval encoder
export OPD_SFT_DATA_DIR="$OPD_SITE_ROOT/data/opd_sft_data" # question / gold JSONL files
export OPD_WIKI18_DIR="$OPD_SITE_ROOT/data/wiki18_index"   # corpus + FAISS index + Search-R1 clone
export OPD_BCPLUS_DIR="$OPD_SITE_ROOT/data/bcplus"         # optional, long-horizon evals only
export OPD_RUNSTATE_DIR="$OPD_SITE_ROOT/logs"              # host files published by long-lived services

# Per-line selectors. Override at submit time to switch teacher / reference student.
export OPD_TEACHER_MODEL="${OPD_TEACHER_MODEL:-$OPD_MODELS_DIR/MiroThinker-1.7-mini}"
export OPD_STUDENT_CKPT="${OPD_STUDENT_CKPT:-$OPD_MODELS_DIR/pi_ref_ckpt}"

# Tool caches (HF, vLLM compile cache, Triton JIT). Keep them off a quota'd home directory.
export OPD_CACHE_ROOT="${OPD_CACHE_ROOT:-$OPD_SITE_ROOT/cache}"
export HF_HOME="$OPD_CACHE_ROOT/hf"
export VLLM_CACHE_ROOT="$OPD_CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$OPD_CACHE_ROOT/triton"

# CUDA toolkit DeepSpeed compiles against (slurm/sft_train.sbatch). Empty = search the env for nvcc.
export OPD_CUDA_HOME="${OPD_CUDA_HOME:-}"

# Scheduler vocabulary.
export OPD_PARTITION=gpu                 # comma-separated list is fine
export OPD_GRES_GPU=a100                 # the gres name used in --gres=gpu:<name>:<n>
export OPD_ACCOUNT=""                    # empty = submit.sh omits --account
export OPD_CHANNEL=a100                  # one of configs/channels/*.yaml
export OPD_NETWORK_PARTITION=""          # partition with outbound network, empty = login node
export OPD_RETRIEVER_EXCLUDE=""          # nodes the retriever job must avoid, comma-separated
