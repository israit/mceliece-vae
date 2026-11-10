"""
Two follow-up experiments (v3 revision).

Part C  Can a VAE see the structure of Goppa codes?  Goppa codes vs random linear codes in systematic form
        [I_k | A] (the input representation of DeepDistinguisher, Malhou et al. 2025), for (n, m, t) =
        (32, 5, 2), (64, 6, 2), (64, 6, 3) and (64, 6, 4) (the last one = our toy keys). Classifiers on the
        k x (n - k) block A:
          - minimum-distance rules (no learning): every row of [I|A] has weight >= 2t + 1, and every sum of two
            rows has weight >= 2t + 1 (Goppa codes have minimum distance >= 2t + 1);
          - VAE trained on Goppa keys only (unsupervised, one-class): score = reconstruction error;
          - logistic regression on the VAE latent mean (supervised);
          - a supervised MLP on the raw bits (neural baseline).
        Metrics: test accuracy (threshold chosen on a validation set) and ROC AUC.

Part H  Key-generator health check: false-alarm rate of the four tests of Part D (VAE reconstruction,
        majority bit, duplicates, pooled rank) on the correct generator, over independent seeds.

Reuses mceliece_vae_revised_v4.py and vae_goppa_detect_v1.py (same folder).

Usage:
    python vae_distinguish_health_v1.py --out ../results/distinguish_health_v1
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

import mceliece_vae_revised_v4 as mv
import vae_goppa_detect_v1 as gd

VERSION = "distinguish_health_v1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# Part C: Goppa vs random codes
# =============================================================================


def _goppa_blocks(args):
    """Worker: A blocks of systematic generator matrices [I|A] of random binary Goppa codes."""
    m, n, t, count, seed = args
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < count:
        mc = mv.McEliece(m, n, t, "systematic", rng)
        if mc.k == n - m * t:                 # keep full-dimension codes only (same k as the random codes)
            out.append(mc.T.astype(np.uint8))
    return out


def random_blocks(k, n, count, rng):
    """A blocks of systematic generator matrices of uniformly random [n, k] binary codes."""
    out = []
    while len(out) < count:
        M = rng.integers(0, 2, size=(k, n), dtype=np.uint8)
        R, piv = mv.gf2_rref(M)
        if len(piv) < k:
            continue
        free = [c for c in range(n) if c not in set(piv)]
        out.append(R[:k][:, list(piv) + free][:, k:].copy())
    return out


def gf2_weights(A):
    """Row weights of [I|A] and weights of all sums of two rows of [I|A]."""
    w = A.sum(1).astype(int) + 1
    X = A.astype(np.int16)
    pair = (X[:, None, :] ^ X[None, :, :]).sum(2) + 2
    iu = np.triu_indices(A.shape[0], 1)
    return w, pair[iu]


def auc(scores_pos, scores_neg):
    """ROC AUC by the Mann-Whitney statistic (ties count one half)."""
    s = np.concatenate([scores_pos, scores_neg])
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):                         # average ranks of ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = len(scores_pos), len(scores_neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def threshold_accuracy(val_pos, val_neg, test_pos, test_neg):
    """Accuracy on the test set with the threshold (score >= thr -> Goppa) that is best on the validation set."""
    cand = np.unique(np.concatenate([val_pos, val_neg]))
    best, thr = -1, cand[0]
    for c in cand:
        a = ((val_pos >= c).sum() + (val_neg < c).sum()) / (len(val_pos) + len(val_neg))
        if a > best:
            best, thr = a, c
    return float(((test_pos >= thr).sum() + (test_neg < thr).sum()) / (len(test_pos) + len(test_neg)))


def logistic(Xtr, ytr, Xte, l2=1e-3, iters=3000, lr=0.1):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    B = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        w -= lr * (A.T @ (p - ytr) / len(A) + l2 * w)
    return B @ w


def run_distinguish(args, rng):
    import tensorflow as tf
    from tensorflow import keras
    rows = []
    for (n, m, t) in args.sets:
        k = n - m * t
        t0 = time.perf_counter()
        ntot = args.n_train + args.n_val + args.n_test
        per = int(np.ceil(ntot / args.workers))
        G = [a for r in gd.parallel(_goppa_blocks, [(m, n, t, per, int(s)) for s in rng.integers(0, 2**63, size=args.workers)],
                                    args.workers) for a in r][:ntot]
        Rn = random_blocks(k, n, ntot, rng)
        Gf = np.stack([a.flatten() for a in G]).astype(np.float32)
        Rf = np.stack([a.flatten() for a in Rn]).astype(np.float32)
        a, b = args.n_train, args.n_train + args.n_val
        Gtr, Gva, Gte = Gf[:a], Gf[a:b], Gf[b:]
        Rtr, Rva, Rte = Rf[:a], Rf[a:b], Rf[b:]
        print(f"[C] (n, m, t) = ({n}, {m}, {t}), k = {k}, A = {k}x{n - k} = {k * (n - k)} bits; "
              f"{ntot} Goppa + {ntot} random codes in {time.perf_counter() - t0:.0f} s")
        res = {"n": n, "m": m, "t": t, "k": k, "A_bits": k * (n - k), "n_train_per_class": a,
               "n_val_per_class": args.n_val, "n_test_per_class": args.n_test}

        # minimum-distance rules
        def rule_scores(F, both):
            out = []
            for x in F:
                w, pw = gf2_weights(x.reshape(k, n - k).astype(np.uint8))
                s = w.min() - (2 * t + 1)
                if both:
                    s = min(s, pw.min() - (2 * t + 1))
                out.append(s)
            return np.array(out, dtype=float)
        for name, both in (("rule_row_weight", False), ("rule_row_and_pair_weight", True)):
            sv, rv, st, rt = (rule_scores(F, both) for F in (Gva, Rva, Gte, Rte))
            res[name + "_auc"] = auc(st, rt)
            res[name + "_acc"] = threshold_accuracy(sv, rv, st, rt)
            res[name + "_goppa_pass"] = float((st >= 0).mean())
            res[name + "_random_pass"] = float((rt >= 0).mean())

        # VAE trained on Goppa keys only; score = minus reconstruction BCE
        tf.keras.utils.set_random_seed(args.seed)
        vae = mv.build_vae(Gtr.shape[1], args.latent, [256, 128], 1.0, 1e-3)
        vae.fit(Gtr, epochs=args.epochs, batch_size=64, verbose=0)

        def recon_score(F):
            P = np.clip(vae.decoder(vae.encoder(F, training=False)[0], training=False).numpy(), 1e-7, 1 - 1e-7)
            return (F * np.log(P) + (1 - F) * np.log(1 - P)).sum(1)
        sv, rv, st, rt = (recon_score(F) for F in (Gva, Rva, Gte, Rte))
        res["vae_recon_auc"] = auc(st, rt)
        res["vae_recon_acc"] = threshold_accuracy(sv, rv, st, rt)
        res["vae_recon_bitacc_goppa"] = float(((vae.decoder(vae.encoder(Gte)[0]).numpy() > 0.5) == Gte).mean())

        # logistic regression on the VAE latent mean (supervised)
        Z = lambda F: vae.encoder(F, training=False)[0].numpy()
        Xtr = np.vstack([Z(Gtr), Z(Rtr)])
        ytr = np.r_[np.ones(len(Gtr)), np.zeros(len(Rtr))]
        sv = logistic(Xtr, ytr, np.vstack([Z(Gva), Z(Rva)]))
        st = logistic(Xtr, ytr, np.vstack([Z(Gte), Z(Rte)]))
        nv, nt = len(Gva), len(Gte)
        res["vae_latent_logreg_auc"] = auc(st[:nt], st[nt:])
        res["vae_latent_logreg_acc"] = threshold_accuracy(sv[:nv], sv[nv:], st[:nt], st[nt:])

        # supervised MLP on the raw bits
        tf.keras.utils.set_random_seed(args.seed)
        clf = keras.Sequential([keras.Input(shape=(Gtr.shape[1],)), keras.layers.Dense(256, activation="relu"),
                                keras.layers.Dense(128, activation="relu"), keras.layers.Dense(1, activation="sigmoid")])
        clf.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
        clf.fit(np.vstack([Gtr, Rtr]), ytr, epochs=args.epochs, batch_size=64, verbose=0, shuffle=True)
        sv = clf.predict(np.vstack([Gva, Rva]), verbose=0)[:, 0]
        st = clf.predict(np.vstack([Gte, Rte]), verbose=0)[:, 0]
        res["mlp_auc"] = auc(st[:nt], st[nt:])
        res["mlp_acc"] = threshold_accuracy(sv[:nv], sv[nv:], st[:nt], st[nt:])
        res["seconds"] = time.perf_counter() - t0
        rows.append(res)
        print("    " + ", ".join(f"{kk}={vv:.3f}" for kk, vv in res.items() if kk.endswith(("_auc", "_acc", "_pass"))))
    return rows


# =============================================================================
# Part H: false alarms of the health check on the correct generator
# =============================================================================


def run_health(args, rng):
    m, n, t = 6, 64, 4
    total = args.det_train + args.det_test
    rows = []
    for s in range(args.health_runs):
        t0 = time.perf_counter()
        per = int(np.ceil(total / args.workers))
        jobs = [("correct", None, per, int(x), m, n, t, {}, w * per) for w, x in enumerate(rng.integers(0, 2**63, size=args.workers))]
        X = np.concatenate(gd.parallel(gd._family_keys, jobs, args.workers))[:total]
        Xtr, Xte = X[:args.det_train], X[args.det_train:]
        vae, _ = gd.train_vae(Xtr, 32, [256, 128], 1.0, 1e-3, args.epochs, 64, 100 + s)
        res = gd.detection_tests(Xtr, Xte, n, mv.vae_reconstruct(vae, mv.vae_compress(vae, Xte)))
        _, p_maj = mv.z_test_vs(res["majority_bit_accuracy"], Xte.size, 0.5)
        flags = {"vae": bool((res["p_vae_vs_half"] < 1e-3 and res["vae_bit_accuracy"] > 0.5) or res["vae_exact_recovery"] > 0),
                 "majority": bool(p_maj < 1e-3 and res["majority_bit_accuracy"] > 0.5),
                 "duplicate": bool(res["test_keys_duplicated_in_training"] > 0),
                 "rank": bool(res["pooled_row_rank"] < n)}
        row = {"run": s, **{k_: res[k_] for k_ in ("vae_bit_accuracy", "majority_bit_accuracy", "p_vae_vs_half",
                                                     "vae_exact_recovery", "test_keys_duplicated_in_training", "pooled_row_rank")},
               "p_majority_vs_half": p_maj, **{f"flag_{k_}": v for k_, v in flags.items()},
               "flag_any": any(flags.values()), "seconds": time.perf_counter() - t0}
        rows.append(row)
        print(f"[H] run {s}: VAE {row['vae_bit_accuracy']:.4f} (p {row['p_vae_vs_half']:.2g}), majority "
              f"{row['majority_bit_accuracy']:.4f} (p {p_maj:.2g}), dup {row['test_keys_duplicated_in_training']:.3f}, "
              f"rank {row['pooled_row_rank']}; any flag = {row['flag_any']}  [{row['seconds']:.0f} s]")
    return rows


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument("--skip-h", action="store_true")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--det-train", type=int, default=1000)
    ap.add_argument("--det-test", type=int, default=200)
    ap.add_argument("--health-runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results_distinguish_health_v1")
    args = ap.parse_args()
    args.sets = [(32, 5, 2), (64, 6, 2), (64, 6, 3), (64, 6, 4)]
    if args.quick:
        args.sets = [(32, 5, 2)]
        args.n_train, args.n_val, args.n_test, args.epochs, args.health_runs = 300, 100, 200, 5, 1
        args.det_train, args.det_test = 200, 50
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = {"script_version": VERSION, "config": {k_: v for k_, v in vars(args).items()},
               "environment": mv.environment_info()}
    t0 = time.perf_counter()
    if not args.skip_c:
        results["distinguish"] = run_distinguish(args, rng)
        write_csv(os.path.join(args.out, "table_distinguish.csv"), results["distinguish"])
    if not args.skip_h:
        results["health_correct_generator"] = run_health(args, rng)
        write_csv(os.path.join(args.out, "table_health_correct.csv"), results["health_correct_generator"])
    results["total_seconds"] = time.perf_counter() - t0
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Total {results['total_seconds'] / 60:.1f} min. Results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
