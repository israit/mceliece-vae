"""
Follow-up to diag_c5_exact_train_keys.py: the exactly reconstructed training keys depend only on the training seed
(the same 4 keys for all three models), and 900 = 14 x 64 + 4. Hypothesis: they are the keys of the last, partial
mini-batch, which the final optimiser step fits while all other outputs stay near 0.5.
Test: vary the batch size so that the last batch has 0, 4 or 20 keys, and count exact reconstructions.

Usage: python diag_c5_batch_check.py --out ../04_النتائج/diag_c5
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mceliece_vae_revised_v4 as m   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diag_c5")
    args = ap.parse_args()
    P = m.PRESETS["toy"]
    pool = m.generate_keys(P["n_train"], P, "sgp", np.random.SeedSequence([0, 1]))
    train = pool[:900]
    X = np.stack([k.public.flatten() for k in train]).astype(np.float32)
    import tensorflow as tf
    rows = []
    for batch in (60, 64, 88):                  # last batch: 0, 4, 20 keys
        last = len(train) % batch
        for seed in (0, 1, 2):
            tf.keras.utils.set_random_seed(seed)
            vae = m.build_vae(X.shape[1], 16, P["hidden"], 1.0, 1e-3, "relu")
            vae.fit(X, epochs=50, batch_size=batch, verbose=0)
            Kh = m.vae_reconstruct(vae, m.vae_compress(vae, X)).reshape(len(train), -1)
            ham = (Kh != X.astype(Kh.dtype)).sum(1)
            # how far from 0.5 the decoder outputs are for a typical key
            probs = vae.decoder(m.vae_compress(vae, X), training=False).numpy()
            rows.append({"batch": batch, "last_batch_size": last, "seed": seed, "exact": int((ham == 0).sum()),
                         "median_abs_output_minus_half": float(np.median(np.abs(probs - 0.5)))})
            print(rows[-1])
    path = os.path.join(args.out, "batch_check.json")
    json.dump(rows, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
