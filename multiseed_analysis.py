"""
Multi-seed inference analysis for EBM with optional quantization.

Runs `num_seeds` different random seeds and at each tracked denoising step
selects the best sample (lowest generative perplexity) across all seeds.
Compares this "best-of-N" oracle ppl curve against the single-seed baseline.
Supports combining with quantization so you can assess whether multi-seeding
recovers quality lost to quantization.

Hydra overrides (all prefixed with '+'):
  +num_seeds=8          number of random seeds to run (default 8)
  +quant_types=[none]   comma-separated list from {none,fp4,fp8,int8,nf4}
                        each entry produces its own figure (default [none])
  +ppl_log_every=10     score perplexity every N denoising steps (default 10)
  +use_real_quant=true  use module-replacement quantization (default true)
                        false = simulated (quantize-then-dequantize) quant

Example:
  python -m multiseed_analysis \\
      loader.eval_batch_size=1 data=openwebtext-split model=small \\
      parameterization=subs ebm_backbone=ar model.length=1024 T=0 \\
      +num_seeds=8 '+quant_types=[none,fp4,nf4]' +ppl_log_every=10 \\
      hydra.run.dir=outputs/multiseed +wandb.offline=true
"""

import os
import sys
from unittest.mock import MagicMock

# causal_conv1d ABI mismatch with PyTorch 2.8; stub it out.
for _mod in (
    "causal_conv1d", "causal_conv1d_cuda", "mamba_ssm",
    "mamba_ssm.ops.selective_scan_interface",
    "mamba_ssm.ops.triton.selective_state_update",
):
    sys.modules.setdefault(_mod, MagicMock())

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

import dataloader
import diffusion as diff_module
from diffusion import EBM, _sample_categorical

# ---------------------------------------------------------------------------
# Hydra resolver registration
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
        pass


# ---------------------------------------------------------------------------
# Quantization helpers (simulated — quantize then dequantize in-place)
# ---------------------------------------------------------------------------

def quantize_model_simulated(model: nn.Module, quant_type: str) -> nn.Module:
    """Simulate weight-only quantization: quantize then immediately dequantize."""
    import bitsandbytes.functional as bnb_F

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.data
        if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            continue

        if quant_type in ("nf4", "fp4"):
            w_f32 = w.to(torch.float32)
            q_w, state = bnb_F.quantize_4bit(
                w_f32, quant_type=quant_type, compress_statistics=False
            )
            w_dq = bnb_F.dequantize_4bit(q_w, state, quant_type=quant_type)
            module.weight.data = w_dq.to(torch.float32)
        elif quant_type == "int8":
            w_f32 = w.to(torch.float32)
            scale = w_f32.abs().max() / 127.0
            w_int8 = (w_f32 / scale).round().clamp(-128, 127).to(torch.int8)
            module.weight.data = (w_int8.to(torch.float32) * scale).to(torch.float32)
        elif quant_type == "fp8":
            w_f32 = w.to(torch.float32)
            module.weight.data = w_f32.to(torch.float8_e4m3fn).to(torch.float32)
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")

    return model


# ---------------------------------------------------------------------------
# Real quantization helpers (module replacement)
# ---------------------------------------------------------------------------

class FP8Linear(nn.Module):
    """Stores weight in float8_e4m3fn; dequantizes to bf16 at forward time."""

    def __init__(self, weight_fp8: torch.Tensor, bias, compute_dtype=torch.bfloat16):
        super().__init__()
        self.register_buffer("weight_fp8", weight_fp8)
        self.compute_dtype = compute_dtype
        if bias is not None:
            self.bias = nn.Parameter(bias.data.clone())
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        return self.weight_fp8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.compute_dtype
        w = self.weight_fp8.to(dtype)
        out = F.linear(x.to(dtype), w)
        if self.bias is not None:
            out = out + self.bias.to(dtype)
        return out


class _BF16Adapter(nn.Module):
    """Forces input to bfloat16 before the wrapped bnb module."""

    def __init__(self, inner: nn.Module, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.inner = inner
        self.dtype = dtype

    @property
    def weight(self):
        return self.inner.weight

    @property
    def bias(self):
        return getattr(self.inner, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x.to(self.dtype))


def _make_quantized_linear(m: nn.Linear, quant_type: str) -> nn.Module:
    import bitsandbytes as bnb

    if quant_type in ("nf4", "fp4"):
        inner = bnb.nn.Linear4bit(
            m.in_features, m.out_features,
            bias=m.bias is not None,
            quant_type=quant_type,
            compute_dtype=torch.bfloat16,
        )
        inner.weight = bnb.nn.Params4bit(
            data=m.weight.data.to(torch.float32),
            requires_grad=False,
            quant_type=quant_type,
        )
        if m.bias is not None:
            inner.bias = nn.Parameter(m.bias.data.clone())
        return _BF16Adapter(inner, dtype=torch.bfloat16)

    elif quant_type == "int8":
        inner = bnb.nn.Linear8bitLt(
            m.in_features, m.out_features,
            bias=m.bias is not None,
            has_fp16_weights=False,
            threshold=6.0,
        )
        inner.weight = bnb.nn.Int8Params(
            data=m.weight.data,
            has_fp16_weights=False,
            requires_grad=False,
        )
        if m.bias is not None:
            inner.bias = nn.Parameter(m.bias.data.clone())
        return _BF16Adapter(inner, dtype=torch.bfloat16)

    elif quant_type == "fp8":
        w_fp8 = m.weight.data.to(torch.float32).to(torch.float8_e4m3fn)
        return FP8Linear(w_fp8, m.bias, compute_dtype=torch.bfloat16)

    else:
        raise ValueError(f"Unknown quant_type: {quant_type!r}")


def _replace_linear_recursive(model: nn.Module, quant_type: str) -> None:
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, _make_quantized_linear(child, quant_type))
        else:
            _replace_linear_recursive(child, quant_type)


def quantize_model_real(model: nn.Module, quant_type: str) -> nn.Module:
    """Replace all nn.Linear modules with true quantized counterparts in-place."""
    _replace_linear_recursive(model, quant_type)
    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(config, tokenizer):
    if config.ebm_backbone == "ar":
        return diff_module.EBM(config, tokenizer=tokenizer).to("cuda")
    return diff_module.EBM.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config,
    )


def _setup_model(config, tokenizer, quant_type: str, use_real_quant: bool) -> EBM:
    """Load, eval-mode, EMA-copy, and optionally quantize a fresh model."""
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
    if quant_type != "none":
        if use_real_quant:
            quantize_model_real(model.ebm, quant_type)
            model.ebm.to(model.device)
        else:
            quantize_model_simulated(model.ebm, quant_type)
    return model


# ---------------------------------------------------------------------------
# Trajectory runner  (replicates EBM._sample step-by-step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_trajectory(model: EBM, x0: torch.Tensor, num_steps: int, eps: float = 1e-5):
    """Run the full EBM denoising trajectory.

    Returns:
        trajectory:  list of (batch, seq_len) LongTensors per step.
        x0_preds:    list of argmax(p_x0) per step; index 0 is None.
    """
    x = x0.clone().to(model.device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    trajectory = [x.clone().cpu()]
    x0_preds = [None]

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
        p_x0, x_next = model._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)

        if p_x0_cache is None:
            if (t[0] > model.config.sampling.is_start
                    or t[0] < model.config.sampling.is_end):
                p_x0_cache = p_x0
            else:
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
        x0_preds.append(x.clone().cpu())

    return trajectory, x0_preds


@torch.no_grad()
def run_trajectory_cross_step(
    model: EBM, x0: torch.Tensor, num_steps: int, eps: float = 1e-5
):
    """Cross-step EBM denoising trajectory.

    Normal strategy uses g[t-1] (last IS result) for sample_update.
    Cross-step uses g[t-2] (two steps earlier) instead:

        for t in 0..1 (warm-up):          # initialise g_buf normally
            x_next = sample_update(x, g[t-1])
            p_x0   = fwd(x)
            g[t]   = bwd(p_x0)            # IS
        for t >= 2:
            x_next = sample_update(x, g[t-2])
            p_x0   = fwd(x)               # always fresh
            g[t]   = bwd(p_x0)            # IS

    Quantized models are supported transparently: model.ebm_forward runs on
    whatever precision model.ebm was set to before calling this function.

    Returns:
        trajectory: list of (batch, seq_len) LongTensors, one per step.
        x0_preds:   list of argmax(p_x0) per step; index 0 is None.
    """
    x = x0.clone().to(model.device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps

    trajectory = [x.clone().cpu()]
    x0_preds = [None]

    # g_buf[0] = g[t-2], g_buf[1] = g[t-1]
    g_buf = [None, None]

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)

        # --- sample_update(x, g[t-2 or t-1]) ---
        # Warm-up (i < 2): use g[t-1]; cross-step (i >= 2): use g[t-2].
        p_x0_for_update = g_buf[0] if i >= 2 else g_buf[1]

        if p_x0_for_update is None:
            # First warm-up step: _ddpm_caching_update will run forward internally;
            # reuse the returned p_x0 to avoid a redundant forward pass.
            p_x0_fresh, x_next = model._ddpm_caching_update(
                x, t, dt, p_x0=None
            )
        else:
            # Use the stored gradient for sample_update (skips internal forward).
            _, x_next = model._ddpm_caching_update(x, t, dt, p_x0=p_x0_for_update)
            # fwd: always run a fresh forward on current x to compute g[t].
            sigma_t, _ = model.noise(t)
            p_x0_fresh = model.forward(x, sigma_t).exp()

        # --- bwd: IS on p_x0_fresh → g[t] ---
        if (
            t[0] > model.config.sampling.is_start
            or t[0] < model.config.sampling.is_end
        ):
            # Outside IS window — use raw diffusion p_x0 as gradient.
            g_t = p_x0_fresh
        else:
            k = model.config.sampling.is_size
            x0_samples = _sample_categorical(p_x0_fresh, num_samples=k)
            energy = model.ebm_forward(
                x.repeat(k, 1),
                t.repeat(k, 1),
                x0=x0_samples,
                log_p_x0=p_x0_fresh.repeat(k, 1, 1),
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
            g_t = F.one_hot(x0_selected, num_classes=model.vocab_size).float()

        # Shift buffer: g_buf[0]=g[t-2] ← old g[t-1]; g_buf[1]=g[t-1] ← g[t]
        g_buf[0] = g_buf[1]
        g_buf[1] = g_t

        x0_pred = g_t[..., :model.mask_index].argmax(dim=-1)
        x0_preds.append(x0_pred.clone().cpu())

        x = x_next
        trajectory.append(x.clone().cpu())

    if model.config.sampling.noise_removal:
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=model.device)
        unet_conditioning = model.noise(t)[0]
        x = model.forward(x, unet_conditioning).argmax(dim=-1)
        trajectory.append(x.clone().cpu())
        x0_preds.append(x.clone().cpu())

    return trajectory, x0_preds


# ---------------------------------------------------------------------------
# Perplexity scoring
# ---------------------------------------------------------------------------

def score_ppl(model: EBM, x0_pred: torch.Tensor, tokenizer) -> float:
    """Return GPT-2 generative perplexity for a single step prediction tensor."""
    tokens = x0_pred.to(model.device)
    text_samples = tokenizer.batch_decode(tokens)
    model.gen_ppl_metric.reset()
    success = model.compute_generative_perplexity(text_samples)
    if not success:
        return float('nan')
    return model.gen_ppl_metric.compute().item()


def score_multiseed(
    model: EBM,
    all_x0_preds: list,   # list of num_seeds x0_preds lists
    tokenizer,
    log_every: int = 10,
) -> tuple:
    """Score all seeds at each logged step and return single-seed vs best-of-N ppls.

    Args:
        model: EBM model used for scoring (must have gen_ppl_metric).
        all_x0_preds: list of length num_seeds, each entry is the x0_preds list
                      returned by run_trajectory (x0_preds[0] is None).
        tokenizer: tokenizer for decoding.
        log_every: score perplexity every this many steps.

    Returns:
        step_indices:  list of step indices that were scored.
        ppl_single:    perplexity of seed-0 at each scored step.
        ppl_best_of_n: best (min) perplexity across all seeds at each step.
        ppl_all_seeds: list of per-seed ppl lists, shape [num_seeds, len(step_indices)].
    """
    num_seeds = len(all_x0_preds)
    num_steps = len(all_x0_preds[0])

    step_indices = []
    ppl_single = []
    ppl_best_of_n = []
    ppl_all_seeds = [[] for _ in range(num_seeds)]

    for i in range(num_steps):
        if all_x0_preds[0][i] is None or i % log_every != 0:
            continue

        step_ppls = []
        for s_idx in range(num_seeds):
            ppl = score_ppl(model, all_x0_preds[s_idx][i], tokenizer)
            step_ppls.append(ppl)
            ppl_all_seeds[s_idx].append(ppl)

        step_indices.append(i)
        ppl_single.append(step_ppls[0])  # seed-0 = single-seed baseline
        ppl_best_of_n.append(min(step_ppls))

    return step_indices, ppl_single, ppl_best_of_n, ppl_all_seeds


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_multiseed_comparison(
    step_indices: list,
    ppl_single: list,
    ppl_best_of_n: list,
    ppl_all_seeds: list,
    quant_type: str,
    num_seeds: int,
    out_dir: str,
):
    """One figure per quant_type: single-seed ppl vs best-of-N ppl over steps."""
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Draw individual seed curves (light, for context)
    for s_idx, seed_ppls in enumerate(ppl_all_seeds):
        if s_idx == 0:
            continue  # drawn separately as the baseline
        ax.plot(
            step_indices, seed_ppls,
            color="tab:blue", alpha=0.18, linewidth=0.8,
        )

    # Single-seed baseline (seed 0)
    ax.plot(
        step_indices, ppl_single,
        color="black", linewidth=2.0, linestyle="-",
        label="Single seed (seed 0)",
        marker="o", markersize=3,
    )

    # Best-of-N oracle
    ax.plot(
        step_indices, ppl_best_of_n,
        color="tab:red", linewidth=2.0, linestyle="--",
        label=f"Best of {num_seeds} seeds (oracle)",
        marker="s", markersize=3,
    )

    title_quant = f"({quant_type} quantization)" if quant_type != "none" else "(no quantization)"
    ax.set_title(
        f"Multi-seed inference: single seed vs best-of-{num_seeds} {title_quant}",
        fontsize=12,
    )
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Generative Perplexity (GPT-2 large)\nof argmax($p_{x_0}$)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add annotation for final improvement
    if ppl_single and ppl_best_of_n:
        final_single = ppl_single[-1]
        final_best = ppl_best_of_n[-1]
        pct = 100.0 * (final_single - final_best) / final_single
        ax.annotate(
            f"Final: {final_single:.1f} → {final_best:.1f} ({pct:+.1f}%)",
            xy=(step_indices[-1], final_best),
            xytext=(-120, 20),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
            color="tab:red",
        )

    plt.tight_layout()
    fname = f"multiseed_{quant_type}_n{num_seeds}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {out_path}")
    return out_path


def plot_quant_comparison(
    quant_results: dict,
    num_seeds: int,
    out_dir: str,
):
    """Summary figure: best-of-N ppl for each quant type on one axes.

    quant_results: dict of quant_type -> {'step_indices', 'ppl_best', 'ppl_single'}
    """
    os.makedirs(out_dir, exist_ok=True)
    colors = {
        "none": "black",
        "fp4": "tab:blue", "fp8": "tab:orange",
        "int8": "tab:green", "nf4": "tab:red",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    ax_single, ax_best = axes

    for qt, res in quant_results.items():
        c = colors.get(qt, "gray")
        steps = res["step_indices"]
        ax_single.plot(steps, res["ppl_single"], color=c, linewidth=1.8,
                       label=qt if qt != "none" else "none (default)")
        ax_best.plot(steps, res["ppl_best"], color=c, linewidth=1.8,
                     label=qt if qt != "none" else "none (default)")

    for ax, title in zip(
        [ax_single, ax_best],
        ["Single seed", f"Best of {num_seeds} seeds (oracle)"],
    ):
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("Generative Perplexity (GPT-2 large)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Quantization × Multi-seed ({num_seeds} seeds): ppl comparison",
        fontsize=13,
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"quant_multiseed_summary_n{num_seeds}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Summary figure saved to {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    L.seed_everything(config.seed)

    # Patch torch.load for Lightning checkpoint compatibility.
    _orig_torch_load = torch.load
    def _patched_torch_load(f, map_location=None, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(f, map_location=map_location, **kwargs)
    torch.load = _patched_torch_load

    import utils
    logger = utils.get_logger(__name__)

    # ------------------------------------------------------------------ #
    # Parse extra CLI overrides with defaults                              #
    # ------------------------------------------------------------------ #
    num_seeds = int(getattr(config, "num_seeds", 8))
    raw_quant = getattr(config, "quant_types", ["none"])
    if isinstance(raw_quant, str):
        quant_types = [q.strip() for q in raw_quant.strip("[]").split(",")]
    else:
        quant_types = list(raw_quant)
    ppl_log_every = int(getattr(config, "ppl_log_every", 10))
    use_real_quant = str(getattr(config, "use_real_quant", "true")).lower() != "false"

    # +inference_strategy=standard (default) or cross_step
    inference_strategy = str(getattr(config, "inference_strategy", "standard"))
    if inference_strategy == "cross_step":
        _run_traj = run_trajectory_cross_step
    elif inference_strategy == "standard":
        _run_traj = run_trajectory
    else:
        raise ValueError(
            f"Unknown inference_strategy={inference_strategy!r}. "
            "Choose 'standard' or 'cross_step'."
        )

    logger.info(
        f"Multi-seed analysis: num_seeds={num_seeds}, quant_types={quant_types}, "
        f"ppl_log_every={ppl_log_every}, use_real_quant={use_real_quant}, "
        f"inference_strategy={inference_strategy}"
    )
    logger.info(f"Seed: {config.seed}")

    tokenizer = dataloader.get_tokenizer(config)
    base_seed = config.seed
    num_steps = config.sampling.steps

    out_dir = os.path.join(os.getcwd(), "multiseed_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Shared starting point (all-mask prior)                              #
    # We load a temporary model just to sample x0, then discard it.       #
    # ------------------------------------------------------------------ #
    logger.info("Sampling shared x0 (all-mask prior)…")
    _tmp_model = _load_model(config, tokenizer)
    _tmp_model.eval()
    batch_size = config.loader.eval_batch_size
    x0 = _tmp_model._sample_prior(batch_size, config.model.length).to(_tmp_model.device)
    del _tmp_model
    torch.cuda.empty_cache()
    logger.info(f"Using shared RNG seed: {base_seed}")

    # ------------------------------------------------------------------ #
    # Run each quantization variant                                       #
    # ------------------------------------------------------------------ #
    quant_results = {}

    for quant_type in quant_types:
        qt_label = quant_type if quant_type != "none" else "none (unquantized)"
        logger.info(f"\n{'='*60}")
        logger.info(f"Quantization: {qt_label} | num_seeds={num_seeds}")
        logger.info(f"{'='*60}")

        # Load a fresh model for this quant type.
        logger.info(f"  Loading model for quant_type={quant_type}…")
        model = _setup_model(config, tokenizer, quant_type, use_real_quant)

        # ---- Run num_seeds trajectories ---- #
        all_x0_preds = []
        for seed_i in range(num_seeds):
            rng = base_seed + seed_i
            logger.info(f"  Seed {seed_i+1}/{num_seeds} (rng={rng})…")
            torch.manual_seed(rng)
            _, x0_preds_i = _run_traj(model, x0, num_steps)
            all_x0_preds.append(x0_preds_i)

        # ---- Score all seeds at every ppl_log_every steps ---- #
        logger.info(f"  Scoring perplexity (log_every={ppl_log_every})…")
        step_indices, ppl_single, ppl_best, ppl_all_seeds = score_multiseed(
            model, all_x0_preds, tokenizer, log_every=ppl_log_every
        )

        if step_indices:
            logger.info(
                f"  Single-seed final ppl:   {ppl_single[-1]:.2f}\n"
                f"  Best-of-{num_seeds} final ppl: {ppl_best[-1]:.2f}"
            )

        # ---- Individual figure for this quant type ---- #
        plot_multiseed_comparison(
            step_indices, ppl_single, ppl_best, ppl_all_seeds,
            quant_type=quant_type,
            num_seeds=num_seeds,
            out_dir=out_dir,
        )

        quant_results[quant_type] = {
            "step_indices": step_indices,
            "ppl_single": ppl_single,
            "ppl_best": ppl_best,
            "ppl_all_seeds": ppl_all_seeds,
        }

        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Summary figure (all quant types on one canvas)                      #
    # ------------------------------------------------------------------ #
    if len(quant_types) > 1:
        plot_quant_comparison(quant_results, num_seeds, out_dir)

    # ------------------------------------------------------------------ #
    # Save numerical results                                               #
    # ------------------------------------------------------------------ #
    payload = {
        "results": quant_results,
        "x0": x0.detach().cpu(),
        "metadata": {
            "seed": int(base_seed),
            "inference_strategy": inference_strategy,
            "num_seeds": int(num_seeds),
            "x0_shape": tuple(x0.shape),
        },
    }
    np.save(os.path.join(out_dir, "results.npy"), payload)
    logger.info("Saved results.npy with keys: results, x0, metadata")
    logger.info(f"\nAll results saved to {out_dir}")

    # ---- Print summary table ---- #
    logger.info("\n=== Final perplexity summary ===")
    logger.info(f"{'Quant':>8} | {'Single seed':>12} | {'Best-of-'+str(num_seeds):>14} | {'Improvement':>12}")
    logger.info("-" * 56)
    for qt, res in quant_results.items():
        if not res["ppl_single"]:
            continue
        s = res["ppl_single"][-1]
        b = res["ppl_best"][-1]
        pct = 100.0 * (s - b) / s
        logger.info(f"{qt:>8} | {s:>12.2f} | {b:>14.2f} | {pct:>+11.1f}%")


if __name__ == "__main__":
    main()
