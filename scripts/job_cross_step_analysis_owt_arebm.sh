#!/usr/bin/env bash
# Cross-step inference analysis for AREBM on OpenWebText.
#
# Runs both quant_analysis and multiseed_analysis with the cross_step
# inference strategy.  Quantization is optional via QUANT_TYPES.
#
# Key knobs (override via env vars):
#   INFERENCE_STRATEGY - standard (default) or cross_step
#   QUANT_TYPES        - comma-separated: none,fp4,fp8,int8,nf4
#                        use '[none]' for no quantization
#   NUM_SEEDS          - number of random seeds for multiseed run (default 8)
#   PPL_LOG_EVERY      - score perplexity every N steps (default 10)
#   USE_REAL_QUANT     - true = replace nn.Linear, false = simulated (default true)
#   SEED               - random seed passed into Hydra (default: 1)
#
# Usage:
#   bash scripts/job_cross_step_analysis_owt_arebm.sh
#
#   Cross-step + nf4 quantization:
#   INFERENCE_STRATEGY=cross_step QUANT_TYPES='[none,nf4]' \
#     bash scripts/job_cross_step_analysis_owt_arebm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

INFERENCE_STRATEGY="${INFERENCE_STRATEGY:-cross_step}"
QUANT_TYPES="${QUANT_TYPES:-[none]}"
NUM_SEEDS="${NUM_SEEDS:-8}"
PPL_LOG_EVERY="${PPL_LOG_EVERY:-10}"
USE_REAL_QUANT="${USE_REAL_QUANT:-true}"
SEED="${SEED:-1}"

exp=arebm_owt_cross_step_${INFERENCE_STRATEGY}

export HYDRA_FULL_ERROR=1

echo "================================================="
echo "Cross-step analysis: exp=${exp}"
echo "  inference_strategy=${INFERENCE_STRATEGY}"
echo "  quant_types=${QUANT_TYPES}"
echo "  num_seeds=${NUM_SEEDS}"
echo "  ppl_log_every=${PPL_LOG_EVERY}"
echo "  use_real_quant=${USE_REAL_QUANT}"
echo "  seed=${SEED}"
echo "================================================="

# --- quant_analysis: trajectory divergence vs reference ---
echo ""
echo "=== quant_analysis (strategy=${INFERENCE_STRATEGY}) ==="
python -u -m quant_analysis \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=subs \
    ebm_backbone=ar \
    model.length=1024 \
    T=0 \
    seed=${SEED} \
    hydra.run.dir=outputs/${exp}/quant \
    +wandb.offline=true \
    "+inference_strategy=${INFERENCE_STRATEGY}"

# --- multiseed_analysis: best-of-N ppl with optional quantization ---
echo ""
echo "=== multiseed_analysis (strategy=${INFERENCE_STRATEGY}) ==="
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
    hydra.run.dir=outputs/${exp}/multiseed \
    +wandb.offline=true \
    "+inference_strategy=${INFERENCE_STRATEGY}" \
    "+num_seeds=${NUM_SEEDS}" \
    "+quant_types=${QUANT_TYPES}" \
    "+ppl_log_every=${PPL_LOG_EVERY}" \
    "+use_real_quant=${USE_REAL_QUANT}"
