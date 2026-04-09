#!/usr/bin/env bash
# Cross-step quantization analysis for AREBM on OpenWebText using REAL quantization.
#
# This launches `quant_analysis_real.py`, which replaces nn.Linear modules
# inside model.ebm with true quantized modules (fp4/fp8/int8/nf4) and compares
# them against the default trajectory.
#
# Key knobs (override via env vars):
#   INFERENCE_STRATEGY - standard or cross_step (default: cross_step)
#   EXP_NAME           - Hydra output folder suffix
#   SEED               - random seed passed into Hydra (default: 1)
#
# Usage:
#   bash scripts/job_cross_step_real_quant_owt_arebm.sh
#
#   Standard strategy with real quantization:
#   INFERENCE_STRATEGY=standard \
#     bash scripts/job_cross_step_real_quant_owt_arebm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

INFERENCE_STRATEGY="${INFERENCE_STRATEGY:-cross_step}"
EXP_NAME="${EXP_NAME:-arebm_owt_cross_step_real_quant_${INFERENCE_STRATEGY}}"
SEED="${SEED:-1}"

export HYDRA_FULL_ERROR=1

echo "================================================="
echo "Cross-step real-quant analysis: exp=${EXP_NAME}"
echo "  inference_strategy=${INFERENCE_STRATEGY}"
echo "  seed=${SEED}"
echo "================================================="

python -u -m quant_analysis_real \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=subs \
    ebm_backbone=ar \
    model.length=1024 \
    T=0 \
    seed=${SEED} \
    hydra.run.dir=outputs/${EXP_NAME} \
    +wandb.offline=true \
    "+inference_strategy=${INFERENCE_STRATEGY}"
