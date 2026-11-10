"""
Publication-quality figures from the saved results of mceliece_vae_revised_v2.py
(no experiment is re-run; every value is read from the results folders).

Each figure is written three times:
  *.png   600 dpi raster
  *.tiff  600 dpi, LZW-compressed (the format most journals ask for)
  *.pdf   vector, fonts embedded (TrueType)

Sizes follow common journal widths: 7.16 in (double column) and 3.5 in (single column),
with 8 pt text at final printed size.

Usage (Colab):
    !unzip -q results_v2.zip -d results_v2
    !cp -r results/* results_v2/results/          # add the v3 bit-flip results
    !python make_figures_v3_1.py --results results_v2/results --out figures_v3_1
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VERSION = "v3.1"   # v3.1: Fig F (bit-flip sensitivity), Fig D label placed after the error bar

# validated categorical order (blue, orange, aqua) + neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#898781"
DOUBLE, SINGLE = 7.16, 3.5

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "lines.markersize": 4.5, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(fig, out, name):
    base = os.path.join(out, name)
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print("  wrote", base + ".{png,tiff,pdf}")


def summary_series(rows, split, key):
    sel = sorted((r for r in rows if r["split"] == split), key=lambda r: int(r["latent_dim"]))
    lat = [int(r["latent_dim"]) for r in sel]
    mean = [float(r[f"{key}_mean"]) * 100 for r in sel]
    sd = [float(r[f"{key}_sd"]) * 100 for r in sel]
    return lat, mean, sd


def latent_axis(ax, lat):
    ax.set_xscale("log", base=2)
    ax.set_xticks(lat, [str(l) for l in lat])
    ax.minorticks_off()
    ax.set_xlabel("Latent dimension d")


def fig_generalisation(res, out):
    rows = read_csv(os.path.join(res, "toy_sgp", "table_generalisation_summary.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.3))
    panels = [("exact_recovery_rate", "Exact key recovery (%)", (-3, 103)),
              ("bit_accuracy_mean", "Bit accuracy (%)", None),
              ("dfr_without_check", "Decryption failure rate (%)", (-3, 103))]
    for ax, (key, label, ylim) in zip(axes, panels):
        for split, color, marker, ls in (("train", BLUE, "o", "-"), ("test", ORANGE, "s", "--")):
            lat, mean, sd = summary_series(rows, split, key)
            ax.errorbar(lat, mean, yerr=sd, color=color, marker=marker, ls=ls, capsize=2,
                        label="Training keys" if split == "train" else "Unseen test keys")
        latent_axis(ax, lat)
        ax.set_ylabel(label)
        if ylim:
            ax.set_ylim(*ylim)
    axes[1].axhline(50, color=INK2, lw=0.8, ls=":")
    axes[1].annotate("chance (50%)", xy=(lat[-1], 50), xytext=(0, -9), textcoords="offset points",
                     ha="right", fontsize=7, color=INK2)
    for ax, tag in zip(axes, "abc"):
        ax.set_title(f"({tag})", loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, out, "Fig_A_generalisation")


def fig_compression(res, out):
    rows = read_csv(os.path.join(res, "toy_sgp", "table_compression_ratios.csv"))
    lat = [int(r["latent_dim"]) for r in rows]
    series = [("naive_ratio_key_over_latent", "Key / latent (naive)", BLUE, ""),
              ("ratio_with_hash_and_fallback", "Shared decoder, with hash check and fallback", ORANGE, "///"),
              ("ratio_including_decoder_one_key", "Per-key decoder included", AQUA, "...")]
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.6))
    w = 0.26
    for j, (key, label, color, hatch) in enumerate(series):
        xs = [i + (j - 1) * w for i in range(len(lat))]
        ax.bar(xs, [float(r[key]) for r in rows], w * 0.92, color=color, edgecolor="white",
               linewidth=0.6, hatch=hatch, label=label)
    ax.axhline(1, color=INK, lw=0.9)
    ax.text(len(lat) - 0.5, 1.15, "ρ = 1 (no compression)", ha="right", va="bottom", fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(lat)), [f"d = {l}" for l in lat])
    ax.set_ylabel("Compression ratio ρ (log scale)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    save(fig, out, "Fig_B_compression_ratios")


def fig_ablation(res, out):
    runs = [("toy_sgp", "VAE (β = 1, ReLU)", BLUE, "o", "-"),
            ("toy_sgp_beta0", "No KL term (β = 0)", ORANGE, "s", "--"),
            ("toy_sgp_gelu", "VAE with GELU", AQUA, "^", "-.")]
    runs = [r for r in runs if os.path.exists(os.path.join(res, r[0], "table_generalisation_summary.csv"))]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.4), sharey=True)
    for ax, split, title in ((axes[0], "train", "(a) Training keys"), (axes[1], "test", "(b) Unseen test keys")):
        for folder, label, color, marker, ls in runs:
            rows = read_csv(os.path.join(res, folder, "table_generalisation_summary.csv"))
            lat, mean, sd = summary_series(rows, split, "bit_accuracy_mean")
            ax.errorbar(lat, mean, yerr=sd, color=color, marker=marker, ls=ls, capsize=2, label=label)
        ax.axhline(50, color=INK2, lw=0.8, ls=":")
        latent_axis(ax, lat)
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].set_ylabel("Bit accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, out, "Fig_C_ablation_bit_accuracy")


def fig_timing(res, out):
    toy = read_json(os.path.join(res, "toy_sgp", "results.json"))["timing"]
    big_path = os.path.join(res, "mceliece348864_systematic", "results.json")
    panels = [("(a) Toy parameters (n = 64)", toy,
               [("encrypt_ms", "Encryption"), ("encrypt_with_vae_inference_ms", "Encryption with VAE\ndecoder + hash check"),
                ("decrypt_ms", "Decryption")])]
    if os.path.exists(big_path):
        big = read_json(big_path)["timing"]
        panels.append(("(b) mceliece348864", big,
                       [("keygen_ms", "Key generation"), ("encrypt_ms", "Encryption"), ("decrypt_ms", "Decryption")]))
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE, 2.2))
    axes = axes if len(panels) > 1 else [axes]
    for ax, (title, t, items) in zip(axes, panels):
        items = [(k, lab) for k, lab in items if k in t]
        ys = range(len(items))
        means = [t[k]["mean"] for k, _ in items]
        ax.barh(list(ys), means, xerr=[t[k]["ci95"] for k, _ in items], height=0.55, color=BLUE,
                error_kw={"ecolor": INK, "capsize": 2, "lw": 0.8})
        for y, (k, _), m in zip(ys, items, means):
            ax.text((m + t[k]["ci95"]) * 1.3, y, f"{m:,.2f} ms (n = {t[k]['n']})", va="center", fontsize=7, color=INK)
        ax.set_yticks(list(ys), [lab for _, lab in items])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(min(means) / 4, max(m + t[k]["ci95"] for (k, _), m in zip(items, means)) * 60)
        ax.set_xlabel("Time (ms, mean ± 95% CI, log scale)")
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, out, "Fig_D_timing")


def fig_compressors(res, out):
    sets = [("toy_sgp", "Toy key (2,560 bits)", BLUE, ""),
            ("mceliece348864_systematic", "mceliece348864 key (2,088,960 bits)", ORANGE, "///")]
    sets = [s for s in sets if os.path.exists(os.path.join(res, s[0], "results.json"))]
    names = ["gzip-9", "bzip2-9", "lzma-9"]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    w = 0.38
    for j, (folder, label, color, hatch) in enumerate(sets):
        b = read_json(os.path.join(res, folder, "results.json"))["baseline_compressors"]
        xs = [i + (j - (len(sets) - 1) / 2) * w for i in range(len(names))]
        ax.bar(xs, [b[n]["mean"] for n in names], w * 0.92, yerr=[b[n]["std"] for n in names], color=color,
               edgecolor="white", linewidth=0.6, hatch=hatch, label=label, error_kw={"capsize": 2, "lw": 0.8})
    ax.axhline(1, color=INK, lw=0.9)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Compression ratio\n(original / compressed bytes)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.3))
    fig.tight_layout()
    save(fig, out, "Fig_E_generic_compressors")


def fig_bitflip(res, out):
    """Decryption failure rate when the sender encrypts under a key with j wrong bits."""
    found = []
    for folder in sorted(os.listdir(res)):
        path = os.path.join(res, folder, "results.json")
        if os.path.exists(path):
            data = read_json(path)
            if data.get("bitflip_sensitivity"):
                found.append((data["config"]["preset"], data["code"]["public_key_bits"], data["bitflip_sensitivity"]))
    if not found:
        print("  Fig F skipped: no results with bitflip_sensitivity (run mceliece_vae_revised_v3.py)")
        return
    styles = [(BLUE, "o", "-"), (ORANGE, "s", "--"), (AQUA, "^", "-.")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for (preset, bits, rows), (color, marker, ls) in zip(found, styles):
        rows = sorted(rows, key=lambda r: r["flipped_bits"])
        xs = range(len(rows))
        ax.errorbar(xs, [r["dfr"] * 100 for r in rows], yerr=[r["dfr_ci95"] * 100 for r in rows], color=color,
                    marker=marker, ls=ls, capsize=2, label=f"{preset} ({bits:,} key bits)")
        ax.set_xticks(list(xs), [str(r["flipped_bits"]) for r in rows])
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Wrong bits in the public key used for encryption")
    ax.set_ylabel("Decryption failure rate (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=1)
    fig.tight_layout()
    save(fig, out, "Fig_F_bitflip_sensitivity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures_v3_1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"make_figures {VERSION}: reading {os.path.abspath(args.results)}")
    for fn in (fig_generalisation, fig_compression, fig_ablation, fig_timing, fig_compressors, fig_bitflip):
        fn(args.results, args.out)
    print(f"done -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
