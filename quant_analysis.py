"""
Quantization analysis for EBM inference.

Runs 5 experiments:
  1. Default (no quantization)
  2. fp4  - bitsandbytes 4-bit FloatingPoint
  3. fp8  - simulated FP8 weight-only (torch.float8_e4m3fn cast)
  4. int8 - bitsandbytes 8-bit LLM.int8()
  5. nf4  - bitsandbytes 4-bit NormalFloat

All experiments start from the same x0. After the default trajectory is
recorded, each quantized model runs from the same x0 with the same RNG
seed so that stochasticity is controlled. At every denoising step we record:
  - Normalised L2 divergence  ||x_q - x_ref||_2 / ||x_ref||_2
  - Token-level accuracy  mean(x_q == x_ref)
A two-panel figure is saved to outputs/quant_analysis/quant_comparison.png.
"""

import os

import hydra
import lightning as L
import matplotlib
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# causal_conv1d has an ABI mismatch with PyTorch 2.8; stub it out since
# DiMamba is never used in this analysis (ebm_backbone=ar only).
import sys
from unittest.mock import MagicMock
for _mod in ("causal_conv1d", "causal_conv1d_cuda", "mamba_ssm",
             "mamba_ssm.ops.selective_scan_interface",
             "mamba_ssm.ops.triton.selective_state_update"):
    sys.modules.setdefault(_mod, MagicMock())

import dataloader
import diffusion as diff_module
from diffusion import EBM, _sample_categorical

# ---------------------------------------------------------------------------
# Hydra resolver registration (mirrors main.py)
# ---------------------------------------------------------------------------
for _name, _fn in [
    ("cwd", os.getcwd),
    ("device_count", torch.cuda.device_count),
    ("eval", eval),
    ("div_up", lambda x, y: (x + y - 1) // y),
]:
    try:
        omegaconf.OmegaConf.register_new_resolver(_name, _fn)
    except omegaconf.exceptions.OmegaConfException:
        pass  # already registered


# ---------------------------------------------------------------------------
# Quantisation helper
# ---------------------------------------------------------------------------

def quantize_model(model: nn.Module, quant_type: str) -> nn.Module:
    """Simulate weight-only quantization on all nn.Linear layers in-place.

    Instead of replacing modules (which disrupts dtype flow and breaks
    FlashAttention), we quantize each weight to the target format and
    immediately dequantize it back to its original dtype.  This correctly
    captures precision loss from quantization while leaving the module
    structure — and every intermediate tensor dtype — completely unchanged.

    Supported quant_type values:
      - 'nf4': bitsandbytes 4-bit NormalFloat  (quantize → dequantize)
      - 'fp4': bitsandbytes 4-bit FloatingPoint (quantize → dequantize)
      - 'int8': round weights to int8 range     (quantize → dequantize)
      - 'fp8': cast weights through float8_e4m3fn (quantize → dequantize)
    """
    import bitsandbytes.functional as bnb_F

    total_err, total_numel = 0.0, 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.data
        if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue

        orig_dtype = w.dtype
        w_orig = w.clone()

        if quant_type in ("nf4", "fp4"):
            # bnb quantize_4bit requires float16 on CUDA; float32 input may
            # silently no-op or return float16-precision output in some versions.
            w_fp16 = w.to(torch.float16)
            q_w, state = bnb_F.quantize_4bit(
                w_fp16, quant_type=quant_type, compress_statistics=False
            )
            w_dq = bnb_F.dequantize_4bit(q_w, state, quant_type=quant_type)
            module.weight.data = w_dq.to(orig_dtype)

        elif quant_type == "int8":
            w_f32 = w.to(torch.float32)
            scale = w_f32.abs().max() / 127.0
            w_int8 = (w_f32 / scale).round().clamp(-128, 127).to(torch.int8)
            module.weight.data = (w_int8.to(torch.float32) * scale).to(orig_dtype)

        elif quant_type == "fp8":
            w_f32 = w.to(torch.float32)
            w_fp8 = w_f32.to(torch.float8_e4m3fn)
            module.weight.data = w_fp8.to(orig_dtype)

        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")

        total_err += (module.weight.data.float() - w_orig.float()).norm().item()
        total_numel += w.numel()

    mean_err = total_err / max(total_numel, 1)
    print(f"[quantize_model] {quant_type}: mean weight perturbation = {mean_err:.6e} "
          f"over {total_numel} params")
    return model


# ---------------------------------------------------------------------------
# Trajectory runner  (replicates EBM._sample step-by-step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_trajectory(model: EBM, x0: torch.Tensor, num_steps: int, eps: float = 1e-5):
    """Run the full EBM denoising trajectory starting from x0.

    Returns a list of (batch, seq_len) LongTensors, one per step
    (including the initial x0 and the final noise-removal step if enabled).
    """
    x = x0.clone().to(model.device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    trajectory = [x.clone().cpu()]
    # best-guess clean sequences: argmax(p_x0) at each step, scored for ppl.
    # Step 0 has no p_x0 yet (all masks), so we skip it (None placeholder).
    x0_preds = [None]

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)

        p_x0, x_next = model._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)

        if p_x0_cache is None:
            if (t[0] > model.config.sampling.is_start
                    or t[0] < model.config.sampling.is_end):
                # Outside importance-sampling window — cache p_x0 directly.
                p_x0_cache = p_x0
            else:
                # Energy-based importance sampling (mirrors EBM._sample).
                k = model.config.sampling.is_size
                x0_samples = _sample_categorical(p_x0, num_samples=k)
                energy = model.ebm_forward(
                    x.repeat(k, 1),
                    t.repeat(k, 1),
                    x0=x0_samples,
                    log_p_x0=p_x0.repeat(k, 1, 1),
                    attention_mask=torch.ones_like(x0_samples),
                )
                energy = energy.view(x.shape[0], k)
                energy = energy - energy.max(dim=-1, keepdim=True)[0]
                importance_weights = torch.softmax(
                    energy / model.config.sampling.is_temp, dim=-1
                )
                x0_index = torch.multinomial(importance_weights, 1).view(x.shape[0])
                x0_samples = x0_samples.view(x.shape[0], k, -1)
                x0_selected = x0_samples[torch.arange(x.shape[0]), x0_index]
                p_x0_cache = F.one_hot(
                    x0_selected, num_classes=model.vocab_size
                ).float()
                _, x_next = model._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)

        # Record the model's best guess of the clean sequence at this step.
        # We clamp to vocab_size-1 to exclude the mask token index from argmax.
        current_p_x0 = p_x0_cache if p_x0_cache is not None else p_x0
        x0_pred = current_p_x0[..., :model.mask_index].argmax(dim=-1)
        x0_preds.append(x0_pred.clone().cpu())

        if not torch.allclose(x_next, x) or model.time_conditioning:
            p_x0_cache = None
        x = x_next
        trajectory.append(x.clone().cpu())

    if model.config.sampling.noise_removal:
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=model.device)
        unet_conditioning = model.noise(t)[0]
        x = model.forward(x, unet_conditioning).argmax(dim=-1)
        trajectory.append(x.clone().cpu())
        x0_preds.append(x.clone().cpu())  # final = noise-removed sequence

    return trajectory, x0_preds  # x0_preds[0] is None (step 0 has no prediction)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(traj_ref, traj_q):
    """Per-step normalised L2 divergence and token-agreement vs reference."""
    n = min(len(traj_ref), len(traj_q))
    l2_divs, accs = [], []
    for xr, xq in zip(traj_ref[:n], traj_q[:n]):
        xr_f = xr.float()
        xq_f = xq.float()
        l2 = torch.norm(xq_f - xr_f) / (torch.norm(xr_f) + 1e-8)
        acc = (xq == xr).float().mean()
        l2_divs.append(l2.item())
        accs.append(acc.item())
    return l2_divs, accs


def compute_ppl_trajectory(model, x0_preds, tokenizer, log_every: int = 10):
    """Score each step's best-guess clean sequence with the model's GPT-2 eval.

    x0_preds: list returned by run_trajectory — x0_preds[0] is None (skipped).
    log_every: only score every N steps to save time (GPT-2 forward is costly).

    Returns:
        step_indices: list of step indices that were scored
        ppls: corresponding generative perplexity values
    """
    step_indices, ppls = [], []
    for i, x0_pred in enumerate(x0_preds):
        if x0_pred is None or i % log_every != 0:
            continue
        tokens = x0_pred.to(model.device)
        text_samples = tokenizer.batch_decode(tokens)
        model.gen_ppl_metric.reset()
        model.compute_generative_perplexity(text_samples)
        ppl = model.gen_ppl_metric.compute().item()
        step_indices.append(i)
        ppls.append(ppl)
    return step_indices, ppls


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, out_dir: str):
    """
    results: dict mapping quant_type (+ 'default') ->
        {'l2': [...], 'acc': [...], 'ppl_steps': [...], 'ppl': [...]}
    """
    os.makedirs(out_dir, exist_ok=True)

    colors = {
        "default": "black",
        "fp4": "tab:blue", "fp8": "tab:orange",
        "int8": "tab:green", "nf4": "tab:red",
    }
    linestyles = {
        "default": "-",
        "fp4": "-", "fp8": "--", "int8": "-.", "nf4": ":",
    }
    quant_types = [k for k in results if k != "default"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    fig.suptitle("Quantization Effect on EBM Inference Trajectory", fontsize=13)
    ax_l2, ax_acc, ax_ppl = axes

    # --- L2 and token-agreement plots (share denoising step x-axis) ---
    for qt in quant_types:
        metrics = results[qt]
        steps = np.arange(len(metrics["l2"]))
        ax_l2.plot(steps, metrics["l2"],
                   label=qt, color=colors[qt], linestyle=linestyles[qt], linewidth=1.8)
        ax_acc.plot(steps, metrics["acc"],
                    label=qt, color=colors[qt], linestyle=linestyles[qt], linewidth=1.8)

    ax_l2.set_xlabel("Denoising step")
    ax_l2.set_ylabel("Normalised L2 Divergence\n$\\|x_q - x_{ref}\\|_2 / \\|x_{ref}\\|_2$")
    ax_l2.legend(loc="upper right")
    ax_l2.grid(True, alpha=0.3)
    ax_l2.set_title("Divergence from default trajectory")

    ax_acc.set_xlabel("Denoising step")
    ax_acc.set_ylabel("Token Agreement\n(fraction matching default)")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title("Token-level agreement with default trajectory")
    ax_acc.set_ylim(-0.05, 1.05)

    # --- Generative perplexity over steps (sampled every 10 steps) ---
    for key in ["default"] + quant_types:
        m = results[key]
        if not m.get("ppl"):
            continue
        ax_ppl.plot(m["ppl_steps"], m["ppl"],
                    label=key, color=colors[key], linestyle=linestyles[key],
                    linewidth=1.8, marker="o", markersize=3)

    ax_ppl.set_xlabel("Denoising step")
    ax_ppl.set_ylabel("Generative Perplexity (GPT-2)\nof argmax(p_x0)")
    ax_ppl.legend(loc="upper right")
    ax_ppl.grid(True, alpha=0.3)
    ax_ppl.set_title("Generative perplexity of model's best-guess clean sequence per step")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "quant_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Model loading  (mirrors main.py)
# ---------------------------------------------------------------------------

def _load_model(config, tokenizer):
    if config.ebm_backbone == "ar":
        return diff_module.EBM(config, tokenizer=tokenizer).to("cuda")
    return diff_module.EBM.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    L.seed_everything(config.seed)

    # PyTorch 2.6+ defaults weights_only=True, but Lightning checkpoints
    # contain numpy types that aren't allowlisted. Patch torch.load globally
    # so all checkpoint loads in this process use weights_only=False.
    _orig_torch_load = torch.load
    def _patched_torch_load(f, map_location=None, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(f, map_location=map_location, **kwargs)
    torch.load = _patched_torch_load

    import utils
    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    logger.info("Loading model…")
    model = _load_model(config, tokenizer)
    model.eval()
    if model.ema:
        import itertools
        model.ema.store(
            itertools.chain(model.backbone.parameters(), model.noise.parameters())
        )
        model.ema.copy_to(
            itertools.chain(model.backbone.parameters(), model.noise.parameters())
        )

    # ------------------------------------------------------------------ #
    # Shared starting point x0  (all-mask prior sample)                   #
    # ------------------------------------------------------------------ #
    batch_size = config.loader.eval_batch_size
    x0 = model._sample_prior(batch_size, config.model.length).to(model.device)
    num_steps = config.sampling.steps
    rng_seed = config.seed

    # ------------------------------------------------------------------ #
    # Experiment 1: default (no quantization)                             #
    # ------------------------------------------------------------------ #
    logger.info("Running default (unquantized) trajectory…")
    torch.manual_seed(rng_seed)
    traj_default, x0_preds_default = run_trajectory(model, x0, num_steps)
    logger.info(f"  trajectory length: {len(traj_default)} steps")

    # Generative perplexity of the default trajectory (every 10 steps)
    logger.info("  Scoring default trajectory perplexity…")
    ppl_steps_default, ppls_default = compute_ppl_trajectory(
        model, x0_preds_default, tokenizer, log_every=10
    )
    logger.info(f"  Default final ppl: {ppls_default[-1]:.2f}")

    # ------------------------------------------------------------------ #
    # Experiments 2-5: quantized variants                                 #
    # ------------------------------------------------------------------ #
    quant_types = ["fp4", "fp8", "int8", "nf4"]
    results = {}

    for qt in quant_types:
        logger.info(f"Running {qt} trajectory…")
        # Reload from checkpoint instead of deepcopy — deepcopy fails on
        # non-leaf tensors (e.g. weight_norm / autocast internals).
        model_q = _load_model(config, tokenizer)
        model_q.eval()
        if model_q.ema:
            import itertools as _it
            model_q.ema.store(
                _it.chain(model_q.backbone.parameters(), model_q.noise.parameters())
            )
            model_q.ema.copy_to(
                _it.chain(model_q.backbone.parameters(), model_q.noise.parameters())
            )
        # Only quantize model.ebm (the trained EBM component).
        # model.backbone is a frozen HF diffusion model using FlashAttention,
        # which requires fp16/bf16 and breaks under 4bit/8bit quantization.
        quantize_model(model_q.ebm, qt)

        torch.manual_seed(rng_seed)  # same RNG state as default run
        traj_q, x0_preds_q = run_trajectory(model_q, x0, num_steps)

        l2_divs, accs = compute_metrics(traj_default, traj_q)

        logger.info(f"  Scoring {qt} trajectory perplexity…")
        ppl_steps_q, ppls_q = compute_ppl_trajectory(
            model_q, x0_preds_q, tokenizer, log_every=10
        )

        results[qt] = {
            "l2": l2_divs,
            "acc": accs,
            "ppl_steps": ppl_steps_q,
            "ppl": ppls_q,
        }
        logger.info(
            f"  {qt}: final L2={l2_divs[-1]:.4f}, final acc={accs[-1]:.4f}, "
            f"final ppl={ppls_q[-1]:.2f}"
        )
        del model_q
        torch.cuda.empty_cache()

    # Store default ppl alongside quantized results for plotting
    results["default"] = {
        "l2": [0.0] * len(traj_default),
        "acc": [1.0] * len(traj_default),
        "ppl_steps": ppl_steps_default,
        "ppl": ppls_default,
    }

    # ------------------------------------------------------------------ #
    # Restore EMA weights if used                                         #
    # ------------------------------------------------------------------ #
    if model.ema:
        import itertools
        model.ema.restore(
            itertools.chain(model.backbone.parameters(), model.noise.parameters())
        )

    # ------------------------------------------------------------------ #
    # Save numerical results and figure                                   #
    # ------------------------------------------------------------------ #
    # hydra.run.dir is the experiment output dir (set via hydra.run.dir= on CLI).
    # os.getcwd() resolves to that dir after hydra changes working directory.
    out_dir = os.path.join(os.getcwd(), "quant_analysis")
    np.save(
        os.path.join(os.makedirs(out_dir, exist_ok=True) or out_dir, "results.npy"),
        results,
    )
    plot_results(results, out_dir)

    # Print summary table
    header = f"{'Step':>6} | " + " | ".join(f"{qt:>12}" for qt in quant_types)
    logger.info("=== Normalised L2 divergence per step ===")
    logger.info(header)
    n_steps = len(next(iter(results.values()))["l2"])
    log_every = max(1, n_steps // 10)
    for s in list(range(0, n_steps, log_every)) + [n_steps - 1]:
        row = f"{s:>6} | " + " | ".join(
            f"{results[qt]['l2'][s]:>12.4f}" for qt in quant_types
        )
        logger.info(row)


if __name__ == "__main__":
    main()
