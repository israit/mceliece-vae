"""
Control experiment (reviewer comment 2 on MJCSC-D-26-00004R4): do the autoencoders exceed 50% bit accuracy on
UNIFORMLY RANDOM binary matrices of the size of the toy public keys (40 x 64 = 2,560 bits)?

Random matrices have no per-position regularities, so the majority-bit baseline is 50% in expectation. If a model
still reconstructs unseen random matrices above 50%, the excess must come from information about the test matrix
itself that the encoder passes through the latent vector, not from structure of the keys.

The models, training sizes and training settings are those of the paper:
  plain autoencoder (beta = 0), d = 16, 32, 64, 128, 50 epochs, seeds 0, 1, 2   (Section 5.5, Fig. 6)
  VAE (beta = 1),  d = 32, 50 epochs, seeds 0, 1, 2                            (main detector, Table XVI)
  VAE (beta = 0.1), d = 32, 200 epochs, seeds 0, 1, 2                          (stronger detector, Table XVI)
1,000 training matrices (10% used for validation, as in the paper) and 200 test matrices per seed; the
reconstruction is deterministic (decoder of the latent mean, rounded), as in the paper.

Usage:  python control_random_matrices.py --out ../results/control_random_matrices
"""
import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                   # noqa: E402
import mceliece_vae_revised_v4 as mv                                 # noqa: E402

VERSION = "control_random_matrices_v1"
K, N = 40, 64                                                         # toy public key: k x n = 40 x 64


def run(cfg, seed, n_train, n_test, val_split):
    import tensorflow as tf
    rng = np.random.default_rng([seed, 2026])
    X = rng.integers(0, 2, size=(n_train + n_test, K * N)).astype(np.float32)
    n_val = int(round(n_train * val_split))
    Xtr, Xval, Xte = X[:n_train - n_val], X[n_train - n_val:n_train], X[n_train:]
    majority = (Xtr.mean(0) > 0.5).astype(np.float32)
    maj = float((Xte == majority).mean())
    tf.keras.utils.set_random_seed(seed)
    vae = mv.build_vae(Xtr.shape[1], cfg["d"], [256, 128], cfg["beta"], 1e-3)
    t0 = time.perf_counter()
    vae.fit(Xtr, epochs=cfg["epochs"], batch_size=64, verbose=0, validation_data=(Xval, Xval))
    secs = time.perf_counter() - t0
    out = {}
    for split, M in (("train", Xtr), ("test", Xte)):
        R = mv.vae_reconstruct(vae, mv.vae_compress(vae, M))
        out[split] = float((R == M).mean())
        out[split + "_exact"] = float(np.all(R == M, axis=1).mean())
    z, p = mv.z_test_vs(out["test"], Xte.size)
    _, p_maj = mv.two_proportion_test(out["test"], maj, Xte.size)
    return {"model": cfg["name"], "beta": cfg["beta"], "latent_dim": cfg["d"], "epochs": cfg["epochs"], "seed": seed,
            "test_bit_accuracy": out["test"], "train_bit_accuracy": out["train"],
            "test_exact_recovery": out["test_exact"], "majority_test_bit_accuracy": maj,
            "z_vs_half": z, "p_vs_half": p, "p_vs_majority": p_maj, "n_test_bits": int(Xte.size),
            "seconds": round(secs, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--quick", action="store_true", help="short smoke test")
    ap.add_argument("--out", default="results_control_random_matrices")
    args = ap.parse_args()
    cfgs = ([dict(name="AE beta=0", beta=0.0, d=d, epochs=50) for d in (16, 32, 64, 128)]
            + [dict(name="VAE beta=1 (main detector)", beta=1.0, d=32, epochs=50),
               dict(name="VAE beta=0.1 (stronger detector)", beta=0.1, d=32, epochs=200)])
    if args.quick:
        cfgs = [dict(c, epochs=2) for c in cfgs[:2]]
        args.seeds = args.seeds[:1]
    os.makedirs(args.out, exist_ok=True)
    rows = []
    print(f"{VERSION}: {K} x {N} uniformly random binary matrices, {args.n_train} training / {args.n_test} test per seed")
    for cfg in cfgs:
        for seed in args.seeds:
            r = run(cfg, seed, args.n_train, args.n_test, args.val_split)
            rows.append(r)
            print(f"  {r['model']:34s} d={r['latent_dim']:3d} seed={seed}  test bit-acc={r['test_bit_accuracy']*100:6.2f}%  "
                  f"(p vs 50% = {r['p_vs_half']:.2g}; majority {r['majority_test_bit_accuracy']*100:.2f}%, "
                  f"p vs majority = {r['p_vs_majority']:.2g})  train {r['train_bit_accuracy']*100:.2f}%  "
                  f"exact {r['test_exact_recovery']*100:.1f}%  [{r['seconds']} s]", flush=True)
    with open(os.path.join(args.out, "table_control.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"script_version": VERSION, "config": vars(args), "environment": mv.environment_info(),
                   "rows": rows}, f, indent=1)
    print("Summary (mean over seeds):")
    for cfg in cfgs:
        rs = [r for r in rows if r["model"] == cfg["name"] and r["latent_dim"] == cfg["d"]]
        acc = np.array([r["test_bit_accuracy"] for r in rs]) * 100
        print(f"  {cfg['name']:34s} d={cfg['d']:3d}: test bit-acc {acc.mean():.2f}% (runs {', '.join(f'{a:.2f}' for a in acc)}); "
              f"max p vs 50% = {max(r['p_vs_half'] for r in rs):.2g}")
    print("All results written to", args.out)


if __name__ == "__main__":
    main()
