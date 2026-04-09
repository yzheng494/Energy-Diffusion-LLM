"""
Quantization analysis for EBM inference — TRUE quantization edition.

Unlike quant_analysis.py (which simulates quantization by quantizing then
immediately dequantizing weights back to float32), this script performs REAL
quantization: nn.Linear modules inside model.ebm are replaced with lower-
precision counterparts that store compressed weights and dequantize on-the-fly
at each forward pass.  This gives genuine memory savings and realistic
compute-time behaviour.

Supported quant_type values:
  - 'nf4':  bitsandbytes Linear4bit (NormalFloat 4-bit)  — bnb.nn.Linear4bit
  - 'fp4':  bitsandbytes Linear4bit (FloatingPoint 4-bit) — bnb.nn.Linear4bit
  - 'int8': bitsandbytes Linear8bitLt (LLM.int8())        — bnb.nn.Linear8bitLt
  - 'fp8':  custom FP8Linear (torch.float8_e4m3fn storage) — dequant at forward

model.backbone (frozen HF diffusion model) is intentionally NOT quantized
because it contains FlashAttention kernels that are sensitive to weight dtype.
model.ebm uses the same FlashAttention kernels, but all bnb quantized modules
produce bf16 outputs (matching the autocast context in AR.forward), so they
remain compatible.

Runs 5 experiments (default + 4 quantized), all starting from the same x0
with the same RNG seed.  At every denoising step records:
  - Normalised L2 divergence  ||x_q - x_ref||_2 / ||x_ref||_2
  - Token-level accuracy  mean(x_q == x_ref)
Generative perplexity (GPT-2 large) of argmax(p_x0) is scored every 10 steps.
A three-panel figure is saved to <hydra_run_dir>/quant_analysis_real/.
"""

import os
import sys
from unittest.mock import MagicMock

# causal_conv1d has an ABI mismatch with PyTorch 2.8; stub it out since
# DiMamba is never used in this analysis (ebm_backbone=ar only).
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
# Hydra resolver registration  (mirrors main.py)
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
# Custom FP8 linear module
# ---------------------------------------------------------------------------

class FP8Linear(nn.Module):
    """nn.Linear replacement that stores the weight tensor in float8_e4m3fn.

    On each forward call the weight is dequantized to the input tensor's dtype
    (or compute_dtype if provided) before the matrix multiply.  This gives
    genuine fp8 weight compression with realistic per-step dequant overhead.

    Bias (if present) is kept in float32 and cast at forward time.
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        bias: "nn.Parameter | None",
        compute_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        # Store compressed weight as a non-trainable buffer.
        self.register_buffer("weight_fp8", weight_fp8)
        self.compute_dtype = compute_dtype
        if bias is not None:
            self.bias = nn.Parameter(bias.data.clone())
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        """Expose .weight for any code that inspects module.weight."""
        return self.weight_fp8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.compute_dtype if self.compute_dtype is not None else x.dtype
        w = self.weight_fp8.to(dtype)
        out = F.linear(x.to(dtype), w)
        if self.bias is not None:
            out = out + self.bias.to(dtype)
        return out


# ---------------------------------------------------------------------------
# Module-replacement quantization
# ---------------------------------------------------------------------------

class _BF16Adapter(nn.Module):
    """Forces the input into bfloat16 before the inner module.

    bitsandbytes (Linear4bit / Linear8bitLt) bypasses PyTorch's torch.autocast
    and outputs in whatever dtype the input tensor has.  The LayerNorm in the
    AR model explicitly disables autocast and returns float32, so without this
    adapter the qkv projection would produce float32 qkv tensors — which
    FlashAttention rejects (it only accepts fp16 / bf16).

    Wrapping every replaced bnb module with this adapter keeps the dtype flow
    identical to the unquantized model (autocast bfloat16 throughout).
    """

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


def _make_quantized_linear(
    m: nn.Linear, quant_type: str
) -> nn.Module:
    """Return a quantized replacement for a single nn.Linear layer."""
    import bitsandbytes as bnb

    if quant_type in ("nf4", "fp4"):
        # Linear4bit stores the weight as packed uint8 (two 4-bit values per byte)
        # and dequantizes to compute_dtype on each forward call.
        inner = bnb.nn.Linear4bit(
            m.in_features,
            m.out_features,
            bias=m.bias is not None,
            quant_type=quant_type,
            compute_dtype=torch.bfloat16,
        )
        # Replace the weight with a Params4bit wrapping the original data.
        inner.weight = bnb.nn.Params4bit(
            data=m.weight.data.to(torch.float32),
            requires_grad=False,
            quant_type=quant_type,
        )
        if m.bias is not None:
            inner.bias = nn.Parameter(m.bias.data.clone())

    elif quant_type == "int8":
        # Linear8bitLt uses LLM.int8() mixed-precision: outlier columns in fp16,
        # remaining columns in int8.  has_fp16_weights=False triggers quantization.
        inner = bnb.nn.Linear8bitLt(
            m.in_features,
            m.out_features,
            bias=m.bias is not None,
            has_fp16_weights=False,
            threshold=6.0,  # LLM.int8 standard threshold for outlier detection
        )
        inner.weight = bnb.nn.Int8Params(
            data=m.weight.data,
            has_fp16_weights=False,
            requires_grad=False,
        )
        if m.bias is not None:
            inner.bias = nn.Parameter(m.bias.data.clone())

    elif quant_type == "fp8":
        # FP8Linear already casts input to bf16 in its forward — no adapter needed.
        w_fp8 = m.weight.data.to(torch.float32).to(torch.float8_e4m3fn)
        return FP8Linear(w_fp8, m.bias, compute_dtype=torch.bfloat16)

    else:
        raise ValueError(f"Unknown quant_type: {quant_type!r}")

    # Wrap bnb modules so input is forced to bf16 before the bnb forward.
    # bnb bypasses torch.autocast and returns in the input's dtype; without this
    # the LayerNorm float32 output would propagate and break FlashAttention.
    return _BF16Adapter(inner, dtype=torch.bfloat16)


def _replace_linear_recursive(model: nn.Module, quant_type: str) -> None:
    """Recursively walk model and replace every nn.Linear in-place."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            new_child = _make_quantized_linear(child, quant_type)
            setattr(model, name, new_child)
        else:
            _replace_linear_recursive(child, quant_type)


def quantize_model_real(model: nn.Module, quant_type: str) -> nn.Module:
    """Replace all nn.Linear modules with true quantized counterparts.

    The module is modified in-place.  Returns the same model object for
    convenience.  Prints a mean weight perturbation diagnostic (requires one
    forward to actually quantize 4-bit/8-bit bnb layers; for those we compute
    perturbation from the float32 reference weight before replacement).
    """
    total_err, total_numel = 0.0, 0

    # Collect original weights BEFORE replacement so we can measure perturbation.
    orig_weights: dict[str, torch.Tensor] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            w = m.weight.data
            if w.dtype in (torch.float16, torch.bfloat16, torch.float32):
                orig_weights[name] = w.float().clone()

    _replace_linear_recursive(model, quant_type)

    # Measure perturbation: compare dequantized weight vs original.
    # For bnb 4-bit/8-bit the weight is not yet quantized until the first
    # forward pass, so we approximate by quantizing to float32 here.
    import bitsandbytes.functional as bnb_F

    # Unwrap _BF16Adapter to get the underlying bnb/FP8 module for inspection.
    def _unwrap(m):
        return m.inner if isinstance(m, _BF16Adapter) else m

    import bitsandbytes as bnb
    for name, m in model.named_modules():
        if name not in orig_weights:
            continue
        w_orig = orig_weights[name]
        core = _unwrap(m)

        if quant_type in ("nf4", "fp4"):
            if isinstance(core, bnb.nn.Linear4bit):
                w_f32 = core.weight.data
                if w_f32 is None or w_f32.dtype == torch.uint8:
                    continue
                q_w, state = bnb_F.quantize_4bit(
                    w_f32.to(torch.float32), quant_type=quant_type,
                    compress_statistics=False
                )
                w_dq = bnb_F.dequantize_4bit(q_w, state, quant_type=quant_type)
                total_err += (w_dq.float() - w_orig).norm().item()
                total_numel += w_orig.numel()

        elif quant_type == "int8":
            # Approximate: per-tensor int8 round-trip.
            scale = w_orig.abs().max() / 127.0
            w_int8 = (w_orig / scale).round().clamp(-128, 127).to(torch.int8)
            w_dq = w_int8.float() * scale
            total_err += (w_dq - w_orig).norm().item()
            total_numel += w_orig.numel()

        elif quant_type == "fp8":
            if isinstance(core, FP8Linear):
                w_dq = core.weight_fp8.float()
                total_err += (w_dq - w_orig).norm().item()
                total_numel += w_orig.numel()

    mean_err = total_err / max(total_numel, 1)
    print(
        f"[quantize_model_real] {quant_type}: mean weight perturbation = "
        f"{mean_err:.6e} over {total_numel} params"
    )
    return model


# ---------------------------------------------------------------------------
# Trajectory runner  (replicates EBM._sample step-by-step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_trajectory(model: EBM, x0: torch.Tensor, num_steps: int, eps: float = 1e-5):
    """Run the full EBM denoising trajectory starting from x0.

    Returns:
        trajectory:  list of (batch, seq_len) LongTensors, one per step.
        x0_preds:    list of argmax(p_x0) per step; x0_preds[0] is None
                     (step 0 has all-mask input, no clean prediction yet).
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
    Cross-step uses g[t-2] (two steps earlier) instead.

    Returns:
        trajectory: list of (batch, seq_len) LongTensors, one per step.
        x0_preds:   list of argmax(p_x0) per step; index 0 is None.
    """
    x = x0.clone().to(model.device)
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps

    trajectory = [x.clone().cpu()]
    x0_preds = [None]

    g_buf = [None, None]

    for i in range(num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=model.device)
        p_x0_for_update = g_buf[0] if i >= 2 else g_buf[1]

        if p_x0_for_update is None:
            p_x0_fresh, x_next = model._ddpm_caching_update(
                x, t, dt, p_x0=None
            )
        else:
            _, x_next = model._ddpm_caching_update(x, t, dt, p_x0=p_x0_for_update)
            sigma_t, _ = model.noise(t)
            p_x0_fresh = model.forward(x, sigma_t).exp()

        if (
            t[0] > model.config.sampling.is_start
            or t[0] < model.config.sampling.is_end
        ):
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
    """Score each step's best-guess clean sequence with GPT-2 generative ppl.

    x0_preds: list returned by run_trajectory — x0_preds[0] is None (skipped).

    Returns:
        step_indices: list of step indices that were scored.
        ppls: corresponding generative perplexity values.
    """
    step_indices, ppls = [], []
    for i, x0_pred in enumerate(x0_preds):
        if x0_pred is None or i % log_every != 0:
            continue
        tokens = x0_pred.to(model.device)
        text_samples = tokenizer.batch_decode(tokens)
        model.gen_ppl_metric.reset()
        success = model.compute_generative_perplexity(text_samples)
        ppl = model.gen_ppl_metric.compute().item() if success else float('nan')
        step_indices.append(i)
        ppls.append(ppl)
    return step_indices, ppls


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, out_dir: str):
    """
    results: dict mapping quant_type (including 'default') ->
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
    fig.suptitle(
        "Quantization Effect on EBM Inference Trajectory\n(True quantization — bnb module replacement)",
        fontsize=13,
    )
    ax_l2, ax_acc, ax_ppl = axes

    for qt in quant_types:
        m = results[qt]
        steps = np.arange(len(m["l2"]))
        ax_l2.plot(steps, m["l2"], label=qt, color=colors[qt],
                   linestyle=linestyles[qt], linewidth=1.8)
        ax_acc.plot(steps, m["acc"], label=qt, color=colors[qt],
                    linestyle=linestyles[qt], linewidth=1.8)

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

    for key in ["default"] + quant_types:
        m = results[key]
        if not m.get("ppl"):
            continue
        ax_ppl.plot(m["ppl_steps"], m["ppl"], label=key, color=colors[key],
                    linestyle=linestyles[key], linewidth=1.8, marker="o", markersize=3)

    ax_ppl.set_xlabel("Denoising step")
    ax_ppl.set_ylabel("Generative Perplexity (GPT-2)\nof argmax(p_x0)")
    ax_ppl.legend(loc="upper right")
    ax_ppl.grid(True, alpha=0.3)
    ax_ppl.set_title("Generative perplexity of model's best-guess clean sequence per step")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "quant_comparison_real.png")
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

    # PyTorch 2.6+ defaults weights_only=True, but Lightning checkpoints contain
    # numpy types that aren't allowlisted.  Patch torch.load globally.
    _orig_torch_load = torch.load
    def _patched_torch_load(f, map_location=None, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(f, map_location=map_location, **kwargs)
    torch.load = _patched_torch_load

    import utils
    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

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
    logger.info(f"Inference strategy: {inference_strategy}")
    logger.info(f"Seed: {config.seed}")

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
    # Shared starting point (all-mask prior sample)                        #
    # ------------------------------------------------------------------ #
    batch_size = config.loader.eval_batch_size
    x0 = model._sample_prior(batch_size, config.model.length).to(model.device)
    num_steps = config.sampling.steps
    rng_seed = config.seed
    logger.info(f"Using shared RNG seed: {rng_seed}")

    # ------------------------------------------------------------------ #
    # Experiment 1: default (no quantization)                              #
    # ------------------------------------------------------------------ #
    logger.info("Running default (unquantized) trajectory…")
    torch.manual_seed(rng_seed)
    traj_default, x0_preds_default = _run_traj(model, x0, num_steps)
    logger.info(f"  trajectory length: {len(traj_default)} steps")

    logger.info("  Scoring default trajectory perplexity…")
    ppl_steps_default, ppls_default = compute_ppl_trajectory(
        model, x0_preds_default, tokenizer, log_every=10
    )
    logger.info(f"  Default final ppl: {ppls_default[-1]:.2f}")

    # ------------------------------------------------------------------ #
    # Experiments 2-5: quantized variants                                  #
    # ------------------------------------------------------------------ #
    quant_types = ["fp4", "fp8", "int8", "nf4"]
    results = {}

    for qt in quant_types:
        logger.info(f"Running {qt} trajectory (true quantization)…")
        # Reload fresh from checkpoint — deepcopy fails on non-leaf tensors.
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

        # Replace nn.Linear modules in model.ebm with quantized counterparts.
        # model.backbone (frozen HF backbone) is left intact — its FlashAttention
        # layers require a specific dtype flow that bnb modules cannot guarantee.
        quantize_model_real(model_q.ebm, qt)
        # Move any newly created buffers/parameters to the right device.
        model_q.ebm.to(model_q.device)

        torch.manual_seed(rng_seed)
        traj_q, x0_preds_q = _run_traj(model_q, x0, num_steps)
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

    results["default"] = {
        "l2": [0.0] * len(traj_default),
        "acc": [1.0] * len(traj_default),
        "ppl_steps": ppl_steps_default,
        "ppl": ppls_default,
    }

    if model.ema:
        import itertools
        model.ema.restore(
            itertools.chain(model.backbone.parameters(), model.noise.parameters())
        )

    # ------------------------------------------------------------------ #
    # Save results and figure                                              #
    # ------------------------------------------------------------------ #
    out_dir = os.path.join(os.getcwd(), "quant_analysis_real")
    payload = {
        "results": results,
        "x0": x0.detach().cpu(),
        "metadata": {
            "seed": int(rng_seed),
            "inference_strategy": inference_strategy,
            "x0_shape": tuple(x0.shape),
        },
    }
    np.save(
        os.path.join(os.makedirs(out_dir, exist_ok=True) or out_dir, "results.npy"),
        payload,
    )
    logger.info("Saved results.npy with keys: results, x0, metadata")
    plot_results(results, out_dir)

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
