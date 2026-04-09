# cd to project root (the directory containing this scripts/ folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

exp=arebm_owt_ckpt_real

export HYDRA_FULL_ERROR=1

python -u -m quant_analysis_real \
    loader.batch_size=1 \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    model=small \
    parameterization=subs \
    ebm_backbone=ar \
    model.length=1024 \
    T=0 \
    hydra.run.dir=outputs/$exp \
    +wandb.offline=true
