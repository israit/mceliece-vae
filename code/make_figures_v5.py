"""
Figures for the v5 revision: updated Fig G (conceptual) and Fig H (key generation, now with the
Goppa polynomial), and the new Fig J (VAE as a detector of weak key generators).
Figures A-F come from make_figures_v3_1.py.

Fig G  Conceptual diagram: the three uses of the VAE (generation of S, P and g; public-key encoding;
       randomness test of a key generator).
Fig H  Key generation: (a) cost profile, (b) time per S or P, (c) diversity of S and P,
       (d) irreducible fraction of Goppa candidates, (e) time per irreducible Goppa polynomial.
Fig J  Detection: VAE reconstruction of unseen keys from correct and weakened key generators.

Usage:
  python make_figures_v5.py --keygen ../results/keygen_SP_v1
                            --goppa ../results/goppa_detect_v1 --out ../figures
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_figures_v4 as f4
from make_figures_v4 import AQUA, AXIS, BLUE, DOUBLE, GRID, INK, INK2, MAGENTA, ORANGE, YELLOW, arrow, box, save

VERSION = "v5"


# ----------------------------------------------------------------------------- Fig G
def fig_conceptual(out):
    """Fig G of v4 with the generation box covering g and an extra box for the randomness test."""
    fig, ax = plt.subplots(figsize=(DOUBLE, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    from matplotlib.patches import FancyBboxPatch
    lanes = [(0.0, 0.40, "Key owner (receiver)"), (0.40, 0.22, "Public channel"), (0.62, 0.38, "Sender")]
    for x, w, title in lanes:
        ax.add_patch(FancyBboxPatch((x + 0.005, 0.02), w - 0.01, 0.92, boxstyle="round,pad=0,rounding_size=0.01",
                                    fc="#f9f9f7", ec=GRID, lw=0.8))
        ax.text(x + w / 2, 0.965, title, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    lt, c1, c2, opt = "#e8f1fc", "#fdeee7", "#e6f6ef", "#fff4d6"
    fs = 6.4
    box(ax, 0.025, 0.78, 0.235, 0.12, "Key generation\nGoppa code G (L, g)\nS invertible, P permutation", lt, fs=fs)
    box(ax, 0.275, 0.78, 0.11, 0.12, "Optional:\nVAE-generated\nS, P or g\n(Sec. 4.5)", opt, ec=YELLOW, fs=6.0)
    box(ax, 0.025, 0.60, 0.36, 0.11, "Public key K_pub = S·G·P  (or systematic T)\nPrivate key (S, G, P)", lt, fs=fs)
    box(ax, 0.025, 0.42, 0.17, 0.11, "Encoder mean\nZ = μ(K_pub)", c1, fs=fs)
    box(ax, 0.215, 0.42, 0.17, 0.11, "Hash\nh = SHA-256(K_pub)", c1, fs=fs)
    box(ax, 0.025, 0.235, 0.30, 0.12, "Offline check of the key generator:\nVAE randomness test on many\npublic keys (Sec. 4.7)", opt, ec=YELLOW, fs=6.0)
    box(ax, 0.025, 0.07, 0.36, 0.13, "Decryption with (S, G, P)\nc·P⁻¹ → Goppa decoding → ·S⁻¹\n(no VAE)", lt, fs=fs)
    arrow(ax, (0.275, 0.84), (0.26, 0.84), color=YELLOW, ls=(0, (2, 2)))
    arrow(ax, (0.14, 0.78), (0.14, 0.71))
    arrow(ax, (0.11, 0.60), (0.11, 0.53))
    arrow(ax, (0.30, 0.60), (0.30, 0.53))
    box(ax, 0.42, 0.42, 0.18, 0.11, "(Z, h)\nauthenticated", c1, fs=fs)
    box(ax, 0.42, 0.24, 0.18, 0.11, "Fallback:\nfull K_pub", "#f2f2f2", fs=fs)
    box(ax, 0.42, 0.09, 0.18, 0.09, "Ciphertext c", "#f2f2f2", fs=fs)
    arrow(ax, (0.385, 0.475), (0.42, 0.475))
    arrow(ax, (0.385, 0.62), (0.45, 0.35), ls=(0, (2, 2)))
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
def fig_keygen(keygen_res, goppa_res, out):
    """Fig H of v4 (panels a-c) plus two Goppa panels (d, e)."""
    data = json.load(open(os.path.join(keygen_res, "results.json"), encoding="utf-8"))
    gop = json.load(open(os.path.join(goppa_res, "results.json"), encoding="utf-8"))["goppa"]
    fig = plt.figure(figsize=(DOUBLE, 5.9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=1.05, wspace=1.4)
    axes = [fig.add_subplot(gs[0, 0:3]), fig.add_subplot(gs[0, 3:6]), None, fig.add_subplot(gs[1, 0:3]),
            fig.add_subplot(gs[1, 3:6])]

    # (a) cost profile
    prof = [r for r in data["keygen_profile"] if r["step"] != "TOTAL"]
    sets = []
    for r in prof:
        if r["parameter_set"] not in sets:
            sets.append(r["parameter_set"])
    groups = {"Goppa polynomial": ["goppa_polynomial"], "Gaussian elimination": ["gaussian_elimination"],
              "S (generate + invert)": ["S_generation", "S_inverse"],
              "Public key (S·G·P or systematic)": ["public_key_SGP", "systematic_public_key"],
              "P generation": ["P_generation"], "Other": ["field_and_support", "parity_check_matrix"]}
    colors = [BLUE, ORANGE, YELLOW, AQUA, MAGENTA, "#c3c2b7"]
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.30), ncol=2, fontsize=6.6)

    # (b) S and P: time and min-entropy side by side as a small table-like bar chart
    base = data["csprng_baseline"]
    vae = data["vae_SP"]
    lat = sorted({r["latent_dim"] for r in vae})
    labels = ["Uniform RNG"] + [f"VAE d={l}" for l in lat]
    tS = [base["csprng_S_time_ms"]] + [next(r["vae_ms_per_valid_S"] for r in vae if r["latent_dim"] == l and r["component"] == "S") for l in lat]
    tP = [base["csprng_P_time_ms"]] + [next(r["vae_ms_per_projected_P"] for r in vae if r["latent_dim"] == l and r["component"] == "P") for l in lat]
    hS = [base["csprng_S"]["min_entropy_bits_estimate"]] + [next(r.get("valid_min_entropy_bits_estimate", 0) for r in vae if r["latent_dim"] == l and r["component"] == "S") for l in lat]
    ax = axes[1]
    x = range(len(labels))
    w = 0.38
    ax.bar([i - w / 2 for i in x], tS, w * 0.92, color=BLUE, label="ms per valid S")
    ax.bar([i + w / 2 for i in x], tP, w * 0.92, color=ORANGE, hatch="///", edgecolor="white", linewidth=0.6, label="ms per P")
    for i, h in enumerate(hS):
        ax.text(i - w / 2, tS[i] * 1.25, f"H={h:.1f}", ha="center", va="bottom", fontsize=6.0, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(top=max(tS + tP) * 12)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("ms per matrix (log scale)")
    ax.grid(axis="x", visible=False)
    ax.set_title("(b) S and P: time and min-entropy H of S (bits)", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2)

    # (c) Goppa: irreducible fraction relative to uniform sampling
    base_rows = {r["parameter_set"]: r for r in gop if r["mode"] == "baseline"}
    vrows = [r for r in gop if r["mode"] != "baseline"]
    gsets = list(base_rows)
    gl = sorted({r["latent_dim"] for r in vrows})
    series = [("Root filter (no learning)", None, None, "#c3c2b7", "")]
    pal = {(gl[0], "threshold"): (BLUE, ""), (gl[0], "bernoulli"): (BLUE, "///"),
           (gl[-1], "threshold"): (ORANGE, ""), (gl[-1], "bernoulli"): (ORANGE, "///")}
    for l in gl:
        for mode in ("threshold", "bernoulli"):
            c, h = pal.get((l, mode), (AQUA, ""))
            series.append((f"VAE d={l}, {mode}", l, mode, c, h))
    ax = axes[3]
    nb = len(series)
    w = 0.8 / nb
    for j, (name, l, mode, c, h) in enumerate(series):
        vals = []
        for s in gsets:
            b = base_rows[s]
            if l is None:
                vals.append(b["baseline_root_filter_hit_rate"] / b["baseline_uniform_hit_rate"])
            else:
                vals.append(next(r["hit_ratio_vs_uniform"] for r in vrows if r["parameter_set"] == s and r["latent_dim"] == l and r["mode"] == mode))
        ax.bar([i - 0.4 + w * (j + 0.5) for i in range(len(gsets))], vals, w * 0.92, color=c, hatch=h,
               edgecolor="white", linewidth=0.5, label=name)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(range(len(gsets)), [f"{s.replace('mceliece', '')}\n(t={base_rows[s]['t']})" for s in gsets])
    ax.set_ylabel("Irreducible fraction / uniform")
    ax.grid(axis="x", visible=False)
    ax.set_title("(c) Goppa candidates: hit rate vs uniform", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6.4)

    # (d) Goppa: time per irreducible polynomial
    ax = axes[4]
    series_t = [("Uniform sampling", None, None, INK2, ""), ("Root filter", "f", None, "#c3c2b7", "")]
    series_t += [(f"VAE d={l}, bernoulli", l, "bernoulli", pal.get((l, "bernoulli"), (AQUA, ""))[0], "///") for l in gl]
    nb = len(series_t)
    w = 0.8 / nb
    for j, (name, l, mode, c, h) in enumerate(series_t):
        vals = []
        for s in gsets:
            b = base_rows[s]
            if l is None:
                vals.append(b["baseline_uniform_ms_per_irreducible"])
            elif l == "f":
                vals.append(b["baseline_filter_ms_per_irreducible"])
            else:
                vals.append(next(r["vae_ms_per_irreducible"] for r in vrows if r["parameter_set"] == s and r["latent_dim"] == l and r["mode"] == mode))
        ax.bar([i - 0.4 + w * (j + 0.5) for i in range(len(gsets))], vals, w * 0.92, color=c, hatch=h,
               edgecolor="white", linewidth=0.5, label=name)
    ax.set_yscale("log")
    ax.set_xticks(range(len(gsets)), [f"{s.replace('mceliece', '')}\n(t={base_rows[s]['t']})" for s in gsets])
    ax.set_ylabel("ms per irreducible g (log scale)")
    ax.grid(axis="x", visible=False)
    ax.set_title("(d) Goppa: time per irreducible polynomial", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6.4)
    for a in axes:
        if a is not None:
            a.tick_params(axis="x", labelsize=6.6)
    save(fig, out, "Fig_H_keygen")


# ----------------------------------------------------------------------------- Fig J
def fig_detection(goppa_res, out):
    rows = json.load(open(os.path.join(goppa_res, "results.json"), encoding="utf-8"))["detection"]
    labels = [r["label"] for r in rows]
    y = list(range(len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 0.25 * len(rows) + 1.3), sharey=True,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.axvline(50, color=INK2, lw=0.8, ls=":")
    for i, r in enumerate(rows):
        a, b = 100 * r["majority_bit_accuracy"], 100 * r["vae_bit_accuracy"]
        ax.plot([a, b], [i, i], color=GRID, lw=1.6, zorder=1)
    ax.scatter([100 * r["majority_bit_accuracy"] for r in rows], y, s=16, color="#c3c2b7", label="Majority-bit predictor", zorder=2)
    ax.scatter([100 * r["vae_bit_accuracy"] for r in rows], y, s=18, color=BLUE, label="VAE reconstruction", zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Bit accuracy on unseen keys (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(a) Bit accuracy (50% = chance)", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.6)

    ax = axes[1]
    allx = [100 * r["vae_exact_recovery"] for r in rows]
    nov = [100 * (r["vae_exact_recovery_novel_keys"] if r["vae_exact_recovery_novel_keys"] == r["vae_exact_recovery_novel_keys"] else 0) for r in rows]
    h = 0.38
    ax.barh([i - h / 2 for i in y], allx, h * 0.92, color=BLUE, label="all test keys")
    ax.barh([i + h / 2 for i in y], nov, h * 0.92, color=ORANGE, hatch="///", edgecolor="white", linewidth=0.5,
            label="keys not in training set")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Exact recovery of unseen keys (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) Exact recovery", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.6)
    fig.tight_layout()
    save(fig, out, "Fig_J_detection")


# ----------------------------------------------------------------------------- Fig K
def fig_goppa_vs_random(dist_res, out):
    rows = json.load(open(os.path.join(dist_res, "results.json"), encoding="utf-8"))["distinguish"]
    labels = [f"({r['n']}, {r['m']}, {r['t']})" for r in rows]
    series = [("Row + pair rule (no learning)", "rule_row_and_pair_weight", INK2, "o", "-"),
              ("Row-weight rule (no learning)", "rule_row_weight", "#c3c2b7", "s", "-"),
              ("VAE reconstruction (Goppa only)", "vae_recon", BLUE, "D", "-"),
              ("Supervised MLP", "mlp", ORANGE, "^", "--"),
              ("Logistic regression on VAE latent", "vae_latent_logreg", AQUA, "v", ":")]
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, 2.6))
    x = range(len(rows))
    for name, key, col, mk, ls in series:
        ax.plot(list(x), [r[key + "_auc"] for r in rows], marker=mk, ms=4, lw=1.2, ls=ls, color=col, label=name)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(list(x), labels)
    ax.set_xlabel("Code parameters (n, m, t)")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.45, 1.02)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.8)
    fig.tight_layout()
    save(fig, out, "Fig_K_goppa_vs_random")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keygen", default="results_keygen_SP_v1")
    ap.add_argument("--goppa", default="results_goppa_detect_v1")
    ap.add_argument("--dist", default=None, help="results_distinguish_health_v1 (Fig K)")
    ap.add_argument("--out", default="figures_v5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"make_figures {VERSION}")
    fig_conceptual(args.out)
    fig_keygen(args.keygen, args.goppa, args.out)
    fig_detection(args.goppa, args.out)
    if args.dist:
        fig_goppa_vs_random(args.dist, args.out)


if __name__ == "__main__":
    main()
