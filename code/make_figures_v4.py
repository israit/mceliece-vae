"""
Figures G and H (v4); figures A-F are produced by make_figures_v3_1.py.

Fig G  Conceptual diagram of the VAE-based McEliece protocol (who holds what, what is sent, where the
       hash is checked), requested by reviewers (Round 1 R4-5; Round 2 Comment 10).
Fig H  Key generation: (a) share of each step in McEliece key generation, (b) time to generate one S or P
       with a CSPRNG vs the VAE, (c) diversity (min-entropy) of generated S and P.

Usage: python make_figures_v4.py --keygen ../results/keygen_SP_v1 --out ../figures
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

VERSION = "v4"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#898781"
DOUBLE = 7.16

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, out, name):
    base = os.path.join(out, name)
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print("  wrote", base + ".{png,tiff,pdf}")


# ----------------------------------------------------------------------------- Fig G
def box(ax, x, y, w, h, text, fc, ec=INK2, fs=7.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, text=None, dx=0.0, dy=0.012, style="-|>", ls="-", color=INK2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=0.8, color=color, linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=6.6, color=INK2)


def fig_conceptual(out):
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lanes = [(0.0, 0.40, "Key owner (receiver)"), (0.40, 0.22, "Public channel"), (0.62, 0.38, "Sender")]
    for x, w, title in lanes:
        ax.add_patch(FancyBboxPatch((x + 0.005, 0.02), w - 0.01, 0.92, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc="#f9f9f7", ec=GRID, lw=0.8))
        ax.text(x + w / 2, 0.965, title, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    lt, c1, c2, opt = "#e8f1fc", "#fdeee7", "#e6f6ef", "#fff4d6"
    fs = 6.4
    # key owner lane
    box(ax, 0.025, 0.78, 0.235, 0.12, "Key generation\nGoppa code G (L, g)\nS invertible, P permutation", lt, fs=fs)
    box(ax, 0.275, 0.78, 0.11, 0.12, "Optional:\nVAE-generated\nS and P\n(Sec. 4.5)", opt, ec=YELLOW, fs=6.0)
    box(ax, 0.025, 0.60, 0.36, 0.11, "Public key K_pub = S·G·P  (or systematic T)\nPrivate key (S, G, P)", lt, fs=fs)
    box(ax, 0.025, 0.42, 0.17, 0.11, "Encoder mean\nZ = μ(K_pub)", c1, fs=fs)
    box(ax, 0.215, 0.42, 0.17, 0.11, "Hash\nh = SHA-256(K_pub)", c1, fs=fs)
    box(ax, 0.025, 0.07, 0.36, 0.13, "Decryption with (S, G, P)\nc·P⁻¹ → Goppa decoding → ·S⁻¹\n(no VAE)", lt, fs=fs)
    arrow(ax, (0.275, 0.84), (0.26, 0.84), color=YELLOW, ls=(0, (2, 2)))
    arrow(ax, (0.14, 0.78), (0.14, 0.71))
    arrow(ax, (0.11, 0.60), (0.11, 0.53))
    arrow(ax, (0.30, 0.60), (0.30, 0.53))
    # channel lane
    box(ax, 0.42, 0.42, 0.18, 0.11, "(Z, h)\nauthenticated", c1, fs=fs)
    box(ax, 0.42, 0.24, 0.18, 0.11, "Fallback:\nfull K_pub", "#f2f2f2", fs=fs)
    box(ax, 0.42, 0.09, 0.18, 0.09, "Ciphertext c", "#f2f2f2", fs=fs)
    arrow(ax, (0.385, 0.475), (0.42, 0.475))
    arrow(ax, (0.385, 0.62), (0.45, 0.35), ls=(0, (2, 2)))
    # sender lane
    box(ax, 0.645, 0.78, 0.33, 0.12, "Shared VAE decoder D\n(public, trained on public keys only)", c2, fs=fs)
    box(ax, 0.645, 0.60, 0.33, 0.11, "Reconstruct  K̂ = round(D(Z))", c2, fs=fs)
    box(ax, 0.645, 0.42, 0.33, 0.11, "Verify  SHA-256(K̂) = h ?", opt, ec=YELLOW, fs=fs)
    box(ax, 0.645, 0.07, 0.33, 0.13, "Encryption  c = m·K + e,  wt(e) = t\nK = K̂ if verified, else K_pub", lt, fs=fs)
    arrow(ax, (0.60, 0.50), (0.645, 0.64))
    arrow(ax, (0.81, 0.78), (0.81, 0.71))
    arrow(ax, (0.81, 0.60), (0.81, 0.53))
    arrow(ax, (0.88, 0.42), (0.88, 0.20), "yes", dx=0.025, dy=0.0)
    arrow(ax, (0.70, 0.42), (0.60, 0.32), "no", dx=0.0, dy=0.01)
    arrow(ax, (0.60, 0.27), (0.70, 0.20), ls=(0, (2, 2)))
    arrow(ax, (0.645, 0.135), (0.60, 0.135))
    arrow(ax, (0.42, 0.135), (0.385, 0.135))
    save(fig, out, "Fig_G_conceptual")


# ----------------------------------------------------------------------------- Fig H
def fig_keygen(res, out):
    data = json.load(open(os.path.join(res, "results.json"), encoding="utf-8"))
    prof = [r for r in data["keygen_profile"] if r["step"] != "TOTAL"]
    sets = []
    for r in prof:
        if r["parameter_set"] not in sets:
            sets.append(r["parameter_set"])
    groups = {"Goppa polynomial": ["goppa_polynomial"], "Gaussian elimination": ["gaussian_elimination"],
              "S (generate + invert)": ["S_generation", "S_inverse"], "Public key (S·G·P or systematic)": ["public_key_SGP", "systematic_public_key"],
              "P generation": ["P_generation"], "Other": ["field_and_support", "parity_check_matrix"]}
    colors = [BLUE, ORANGE, YELLOW, AQUA, MAGENTA, "#c3c2b7"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.9), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    left = [0.0] * len(sets)
    for (gname, steps), col in zip(groups.items(), colors):
        vals = [sum(r["share_percent_mean"] for r in prof if r["parameter_set"] == s and r["step"] in steps) for s in sets]
        ax.barh(range(len(sets)), vals, left=left, color=col, edgecolor="white", linewidth=0.6, label=gname, height=0.6)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(range(len(sets)), sets)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of key-generation time (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) Key-generation cost", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=6.8)

    base = data["csprng_baseline"]
    vae = data["vae_SP"]
    lat = sorted({r["latent_dim"] for r in vae})
    labels = ["Uniform RNG"] + [f"VAE d={l}" for l in lat]
    tS = [base["csprng_S_time_ms"]] + [next(r["vae_ms_per_valid_S"] for r in vae if r["latent_dim"] == l and r["component"] == "S") for l in lat]
    tP = [base["csprng_P_time_ms"]] + [next(r["vae_ms_per_projected_P"] for r in vae if r["latent_dim"] == l and r["component"] == "P") for l in lat]
    ax = axes[1]
    x = range(len(labels))
    w = 0.38
    ax.bar([i - w / 2 for i in x], tS, w * 0.92, color=BLUE, label="one valid S")
    ax.bar([i + w / 2 for i in x], tP, w * 0.92, color=ORANGE, hatch="///", edgecolor="white", linewidth=0.6, label="one P")
    ax.set_yscale("log")
    ax.set_xticks(list(x), labels, rotation=30, ha="right")
    ax.set_ylabel("ms per matrix (log scale)")
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) Generation time", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)

    ax = axes[2]
    hS = [base["csprng_S"]["min_entropy_bits_estimate"]] + [next(r.get("valid_min_entropy_bits_estimate", 0) for r in vae if r["latent_dim"] == l and r["component"] == "S") for l in lat]
    hP = [base["csprng_P"]["min_entropy_bits_estimate"]] + [next(r["projected_min_entropy_bits_estimate"] for r in vae if r["latent_dim"] == l and r["component"] == "P") for l in lat]
    ax.bar([i - w / 2 for i in x], hS, w * 0.92, color=BLUE, label="S (valid)")
    ax.bar([i + w / 2 for i in x], hP, w * 0.92, color=ORANGE, hatch="///", edgecolor="white", linewidth=0.6, label="P (projected)")
    ax.axhline(base["csprng_S"]["max_entropy_bits_possible_with_n_samples"], color=INK2, lw=0.8, ls=":")
    ax.set_xticks(list(x), labels, rotation=30, ha="right")
    ax.set_ylabel("Min-entropy estimate (bits)")
    ax.set_ylim(0, 11.5)
    ax.grid(axis="x", visible=False)
    ax.set_title("(c) Diversity of samples", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    fig.tight_layout()
    save(fig, out, "Fig_H_keygen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keygen", default="results_keygen_SP_v1")
    ap.add_argument("--out", default="figures_v4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"make_figures {VERSION}")
    fig_conceptual(args.out)
    fig_keygen(args.keygen, args.out)


if __name__ == "__main__":
    main()
