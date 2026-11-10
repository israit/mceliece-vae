"""
Diagnostic for reviewer comment C5: which training keys are reconstructed exactly at d = 16, and what they share.

Rebuilds the toy training set of mceliece_vae_revised_v4.py (same seeds), checks for repeated keys, then trains the
three models of Table XI (VAE beta=1 ReLU, beta=0, GELU) at d = 16 for VAE seeds 0, 1, 2 and records the indices of the
exactly reconstructed training keys.

Usage: python diag_c5_exact_train_keys.py --out ../04_النتائج/diag_c5
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mceliece_vae_revised_v4 as m   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diag_c5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    P = m.PRESETS["toy"]
    pool = m.generate_keys(P["n_train"], P, "sgp", np.random.SeedSequence([0, 1]))
    n_val = int(round(0.1 * P["n_train"]))
    train = pool[:P["n_train"] - n_val]
    X = np.stack([k.public.flatten() for k in train]).astype(np.float32)

    hashes = [k.public_hash() for k in train]
    counts = collections.Counter(hashes)
    repeated = {h: c for h, c in counts.items() if c > 1}
    groups = [[i for i, h in enumerate(hashes) if h == g] for g in repeated]
    print(f"training keys: {len(train)}; distinct: {len(counts)}; repeated keys: {len(repeated)} groups {groups}")

    import tensorflow as tf
    found = {}
    for name, beta, act in (("VAE beta=1 ReLU", 1.0, "relu"), ("beta=0", 0.0, "relu"), ("VAE GELU", 1.0, "gelu")):
        for seed in args.seeds:
            tf.keras.utils.set_random_seed(seed)
            vae = m.build_vae(X.shape[1], 16, P["hidden"], beta, 1e-3, act)
            vae.fit(X, epochs=50, batch_size=64, verbose=0)
            Kh = m.vae_reconstruct(vae, m.vae_compress(vae, X)).reshape(len(train), -1)
            ham = (Kh != X.astype(Kh.dtype)).sum(1)
            exact = [int(i) for i in np.where(ham == 0)[0]]
            found[f"{name}, seed {seed}"] = {"exact_indices": exact,
                                             "hamming_of_others_mean": float(ham[ham > 0].mean())}
            print(f"{name:16s} seed {seed}: exact {exact}")

    # what the exactly reconstructed keys share
    all_exact = sorted({i for v in found.values() for i in v["exact_indices"]})
    in_groups = {i: next((g for g in groups if i in g), None) for i in all_exact}
    out = {"n_train": len(train), "n_distinct": len(counts), "repeated_groups": groups,
           "runs": found, "exact_keys_union": all_exact, "exact_key_in_repeated_group": in_groups}
    json.dump(out, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    print("exact keys (union):", all_exact)
    print("group membership:", in_groups)


if __name__ == "__main__":
    main()
