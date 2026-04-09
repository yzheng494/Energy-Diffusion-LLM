"""
Compare model perplexity between default (standard) inference and cross-step
inference for each quantization type.

Sources:
  outputs/arebm_owt_ckpt_real/quant_analysis_real/results.npy
  outputs/arebm_owt_cross_step_real_quant_cross_step/quant_analysis_real/results.npy
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

default_npy = np.load(
    "outputs/arebm_owt_ckpt_real/quant_analysis_real/results.npy",
    allow_pickle=True,
).item()
cross_npy = np.load(
    "outputs/arebm_owt_cross_step_real_quant_cross_step/quant_analysis_real/results.npy",
    allow_pickle=True,
).item()

# Normalise to the same inner dict shape
default_results = default_npy  # keys: fp4, fp8, int8, nf4, default
cross_results = cross_npy["results"]  # same keys

quant_types = ["default", "fp4", "fp8", "int8", "nf4"]

colors = {
    "default": "black",
    "fp4": "tab:blue",
    "fp8": "tab:orange",
    "int8": "tab:green",
    "nf4": "tab:red",
}

# ---------------------------------------------------------------------------
# Figure 1: one subplot per quantization type
#           each shows standard vs cross-step PPL across denoising steps
# ---------------------------------------------------------------------------

fig1, axes = plt.subplots(1, len(quant_types), figsize=(18, 4), sharey=False)
fig1.suptitle(
    "Perplexity: Standard vs Cross-Step Inference\n(one panel per quantization type)",
    fontsize=13,
)

for ax, qt in zip(axes, quant_types):
    d_steps = np.array(default_results[qt]["ppl_steps"])
    d_ppl   = np.array(default_results[qt]["ppl"])
    c_steps = np.array(cross_results[qt]["ppl_steps"])
    c_ppl   = np.array(cross_results[qt]["ppl"])

    ax.plot(d_steps, d_ppl, color=colors[qt], linestyle="-",
            linewidth=2, marker="o", markersize=4, label="standard")
    ax.plot(c_steps, c_ppl, color=colors[qt], linestyle="--",
            linewidth=2, marker="s", markersize=4, label="cross-step")

    ax.set_title(qt, fontsize=12)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Perplexity (GPT-2)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = "outputs/ppl_standard_vs_cross_step_per_quant.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# Figure 2: two subplots side-by-side (standard | cross-step)
#           all quantization types overlaid in each
# ---------------------------------------------------------------------------

fig2, (ax_std, ax_cross) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig2.suptitle(
    "Model Perplexity Across Denoising Steps\n(all quantization types)",
    fontsize=13,
)

linestyles = {
    "default": "-",
    "fp4": "-",
    "fp8": "--",
    "int8": "-.",
    "nf4": ":",
}

for qt in quant_types:
    d_steps = np.array(default_results[qt]["ppl_steps"])
    d_ppl   = np.array(default_results[qt]["ppl"])
    c_steps = np.array(cross_results[qt]["ppl_steps"])
    c_ppl   = np.array(cross_results[qt]["ppl"])

    ax_std.plot(d_steps, d_ppl, color=colors[qt], linestyle=linestyles[qt],
                linewidth=2, marker="o", markersize=3, label=qt)
    ax_cross.plot(c_steps, c_ppl, color=colors[qt], linestyle=linestyles[qt],
                  linewidth=2, marker="o", markersize=3, label=qt)

for ax, title in [(ax_std, "Standard Inference"), (ax_cross, "Cross-Step Inference")]:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Perplexity (GPT-2)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = "outputs/ppl_all_quants_standard_vs_cross_step.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")

# ---------------------------------------------------------------------------
# Figure 3: grouped bar chart at the final step for each quantization type
# ---------------------------------------------------------------------------

final_standard = []
final_cross    = []
labels         = []

for qt in quant_types:
    labels.append(qt)
    final_standard.append(float(np.array(default_results[qt]["ppl"])[-1]))
    final_cross.append(float(np.array(cross_results[qt]["ppl"])[-1]))

x = np.arange(len(labels))
w = 0.35

fig3, ax3 = plt.subplots(figsize=(9, 5))
bars_std   = ax3.bar(x - w / 2, final_standard, w, label="Standard",
                     color=[colors[q] for q in labels], alpha=0.8, edgecolor="black", linewidth=0.6)
bars_cross = ax3.bar(x + w / 2, final_cross,    w, label="Cross-step",
                     color=[colors[q] for q in labels], alpha=0.4, edgecolor="black", linewidth=0.6,
                     hatch="//")

ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=11)
ax3.set_ylabel("Final Perplexity (GPT-2)", fontsize=11)
ax3.set_title(
    "Final Step Perplexity — Standard vs Cross-Step Inference\nper Quantization Type",
    fontsize=12,
)
ax3.legend(fontsize=11)
ax3.grid(True, axis="y", alpha=0.3)

for bar in bars_std:
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{bar.get_height():.1f}",
        ha="center", va="bottom", fontsize=8,
    )
for bar in bars_cross:
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{bar.get_height():.1f}",
        ha="center", va="bottom", fontsize=8,
    )

plt.tight_layout()
out3 = "outputs/ppl_final_step_bar_standard_vs_cross_step.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {out3}")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

print("\n=== Final-step perplexity summary ===")
print(f"{'Quant':>10} | {'Standard':>10} | {'Cross-step':>10} | {'Delta':>10}")
print("-" * 48)
for qt, s, c in zip(labels, final_standard, final_cross):
    print(f"{qt:>10} | {s:>10.2f} | {c:>10.2f} | {c - s:>+10.2f}")
