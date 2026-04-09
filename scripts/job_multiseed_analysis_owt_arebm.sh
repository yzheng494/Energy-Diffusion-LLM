#!/usr/bin/env bash
# Multi-seed inference analysis for AREBM on OpenWebText.
#
# Key knobs:
#   NUM_SEEDS    - number of random seeds (8 or 16 recommended)
#   QUANT_TYPES  - comma-separated list: none,fp4,fp8,int8,nf4
#                  use '[none]' for no quantization (default)
#                  use '[none,fp4,nf4]' to compare multiple quant types
#   PPL_LOG_EVERY - score perplexity every N denoising steps
#   USE_REAL_QUANT - true = replace nn.Linear (real), false = simulated
#   SEED          - random seed passed into Hydra (default: 1)
#
# Usage:
#   bash scripts/job_multiseed_analysis_owt_arebm.sh
#
#   Override via env vars:
#   NUM_SEEDS=16 QUANT_TYPES='[none,nf4]' bash scripts/job_multiseed_analysis_owt_arebm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

exp=arebm_owt_ckpt_quant_multiseed_analysis

# Configurable defaults (override via env var)
NUM_SEEDS="${NUM_SEEDS:-8}"
QUANT_TYPES="${QUANT_TYPES:-[none]}"
PPL_LOG_EVERY="${PPL_LOG_EVERY:-10}"
USE_REAL_QUANT="${USE_REAL_QUANT:-true}"
SEED="${SEED:-1}"

export HYDRA_FULL_ERROR=1

echo "================================================="
echo "Multi-seed analysis: exp=${exp}"
echo "  num_seeds=${NUM_SEEDS}"
echo "  quant_types=${QUANT_TYPES}"
echo "  ppl_log_every=${PPL_LOG_EVERY}"
echo "  use_real_quant=${USE_REAL_QUANT}"
echo "  seed=${SEED}"
echo "================================================="

python -u -m multiseed_analysis \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=subs \
    ebm_backbone=ar \
    model.length=1024 \
    T=0 \
    seed=${SEED} \
    hydra.run.dir=outputs/${exp} \
    +wandb.offline=true \
    "+num_seeds=${NUM_SEEDS}" \
    "+quant_types=${QUANT_TYPES}" \
    "+ppl_log_every=${PPL_LOG_EVERY}" \
    "+use_real_quant=${USE_REAL_QUANT}"
