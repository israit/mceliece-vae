"""
VAE-based generation of the McEliece key components S and P (the original thesis idea).

Question: can a Variational Autoencoder, trained on valid scrambling matrices S (random invertible
k x k over GF(2)) and permutation matrices P (n x n), generate new S and P for McEliece key
generation, and does this make key generation faster or keep it secure?

Part A  Key-generation profile: time of every step of McEliece key generation (Goppa polynomial,
        parity-check matrix, Gaussian elimination, S, S^-1, P, public key), so the share of S and P
        in the total cost is measured, not assumed.
Part B  VAE generation of S and P: validity, diversity (unique outputs, min-entropy estimate),
        memorisation of training matrices, bias against a CSPRNG baseline, use of the generated
        S and P in real McEliece keys (encrypt/decrypt self-test), and generation time.

Reuses the McEliece and VAE code of mceliece_vae_revised_v4.py (same folder).

Usage:
    python vae_keygen_SP_v1.py --out ../results/keygen_SP_v1
"""

import argparse
import csv
import json
import os
import sys
import time
from math import erfc, log2, sqrt

import numpy as np

import mceliece_vae_revised_v4 as mv

VERSION = "keygen_SP_v1"


# =============================================================================
# Part A: profile of McEliece key generation
# =============================================================================


def profile_keygen(m, n, t, form, rng):
    """Re-runs the steps of mv.McEliece.__init__ with a timer around each step."""
    T = {}
    t0 = time.perf_counter()
    F = mv.GF2m(m)
    L = rng.permutation(F.q)[:n].astype(np.int64)
    T["field_and_support"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    attempts = 0
    while True:
        attempts += 1
        g = np.append(rng.integers(0, F.q, size=t), 1).astype(np.int64)
        if mv.is_irreducible(F, g):
            break
    T["goppa_polynomial"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ginv = F.inv(mv.peval(F, g, L))
    Hb = np.zeros((m * t, n), dtype=np.uint8)
    for j in range(t):
        row = F.mul(F.pow(L, j), ginv)
        for b in range(m):
            Hb[j * m + b] = (row >> b) & 1
    T["parity_check_matrix"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    R, piv = mv.gf2_rref(Hb)
    free = [c for c in range(n) if c not in set(piv)]
    k = len(free)
    B = np.zeros((k, n), dtype=np.uint8)
    B[np.arange(k), free] = 1
    if piv:
        B[:, piv] = R[:len(piv), free].T
    T["gaussian_elimination"] = time.perf_counter() - t0

    if form == "systematic":
        t0 = time.perf_counter()
        perm = free + list(piv)
        _ = B[:, perm][:, k:].copy()
        T["systematic_public_key"] = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        S = mv.random_invertible_gf2(k, rng)
        T["S_generation"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = mv.gf2_inv(S)
        T["S_inverse"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = rng.permutation(n)
        T["P_generation"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = mv.gf2_matmul(S, B)[:, p]
        T["public_key_SGP"] = time.perf_counter() - t0
    return {k_: v * 1e3 for k_, v in T.items()}, attempts


def run_profile(sets, runs, seed):
    rows = []
    for name, m, n, t, form, r in sets:
        rng = np.random.default_rng([seed, m, n, t])
        samples, attempts = [], []
        for _ in range(min(runs, r)):
            T, a = profile_keygen(m, n, t, form, rng)
            samples.append(T)
            attempts.append(a)
        steps = list(samples[0].keys())
        total = [sum(s.values()) for s in samples]
        for st in steps:
            vals = [s[st] for s in samples]
            share = [s[st] / tot * 100 for s, tot in zip(samples, total)]
            rows.append({"parameter_set": name, "form": form, "step": st,
                         "time_ms_mean": float(np.mean(vals)), "time_ms_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                         "share_percent_mean": float(np.mean(share)), "runs": len(vals)})
        rows.append({"parameter_set": name, "form": form, "step": "TOTAL",
                     "time_ms_mean": float(np.mean(total)), "time_ms_sd": float(np.std(total, ddof=1)) if len(total) > 1 else 0.0,
                     "share_percent_mean": 100.0, "runs": len(total),
                     "goppa_attempts_mean": float(np.mean(attempts))})
        print(f"  {name:16s} total {np.mean(total):10.1f} ms  (n={len(total)})")
        for r_ in rows:
            if r_["parameter_set"] == name and r_["step"] != "TOTAL":
                print(f"      {r_['step']:24s} {r_['time_ms_mean']:10.2f} ms  {r_['share_percent_mean']:6.2f} %")
    return rows


# =============================================================================
# Part B: VAE generation of S and P
# =============================================================================


def random_permutation_matrix(n, rng):
    P = np.zeros((n, n), dtype=np.uint8)
    P[np.arange(n), rng.permutation(n)] = 1
    return P


def nearest_permutation(scores):
    """Greedy projection of an n x n score matrix onto a permutation matrix
    (largest scores first, each row and column used once)."""
    n = scores.shape[0]
    order = np.argsort(-scores, axis=None)
    rows_used, cols_used = np.zeros(n, bool), np.zeros(n, bool)
    P = np.zeros((n, n), dtype=np.uint8)
    for idx in order:
        i, j = divmod(int(idx), n)
        if not rows_used[i] and not cols_used[j]:
            P[i, j] = 1
            rows_used[i] = cols_used[j] = True
            if rows_used.all():
                break
    return P


def chi2_uniform_p(counts, expected):
    """Wilson-Hilferty normal approximation of the chi-square upper-tail p-value."""
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    dof = counts.size - 1
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / sqrt(2 / (9 * dof))
    return chi2, dof, 0.5 * erfc(z / sqrt(2))


def diversity(samples, train_hashes):
    keys = [s.tobytes() for s in samples]
    uniq, counts = np.unique(np.array(keys, dtype=object), return_counts=True)
    n = len(keys)
    return {"n_samples": n, "unique_fraction": len(uniq) / n,
            "min_entropy_bits_estimate": float(-log2(counts.max() / n)),
            "max_entropy_bits_possible_with_n_samples": float(log2(n)),
            "memorised_fraction": float(np.mean([k_ in train_hashes for k_ in keys]))}


def bit_bias(samples):
    X = np.stack([s.flatten() for s in samples]).astype(np.float64)
    return float(np.mean(np.abs(X.mean(0) - 0.5)))


def sample_prior(vae, latent, count, rng):
    z = rng.standard_normal((count, latent)).astype(np.float32)
    return vae.decoder(z, training=False).numpy()


def run_vae_SP(args, rng):
    import tensorflow as tf

    F_m, n, t = args.m, args.n, args.t
    base = mv.McEliece(F_m, n, t, "sgp", np.random.default_rng([args.seed, 99]))
    k = base.k
    print(f"[B] VAE generation of S ({k}x{k}) and P ({n}x{n}); {args.n_train} training matrices each")

    S_train = [mv.random_invertible_gf2(k, rng) for _ in range(args.n_train)]
    P_train = [random_permutation_matrix(n, rng) for _ in range(args.n_train)]
    S_hash, P_hash = {s.tobytes() for s in S_train}, {p.tobytes() for p in P_train}

    # CSPRNG baselines
    raw_inv = float(np.mean([mv.gf2_rank(rng.integers(0, 2, (k, k), dtype=np.uint8)) == k for _ in range(args.n_gen)]))
    S_ref = [mv.random_invertible_gf2(k, rng) for _ in range(args.n_gen)]
    P_ref = [random_permutation_matrix(n, rng) for _ in range(args.n_gen)]
    baseline = {
        "uniform_random_kxk_invertible_fraction": raw_inv,
        "csprng_S": {**diversity(S_ref, S_hash), "bit_bias": bit_bias(S_ref)},
        "csprng_P": {**diversity(P_ref, P_hash),
                     "chi2_position_uniformity_p": chi2_uniform_p(np.stack(P_ref).sum(0).astype(float), args.n_gen / n)[2]},
    }
    t0 = time.perf_counter()
    for _ in range(200):
        mv.random_invertible_gf2(k, rng)
    baseline["csprng_S_time_ms"] = (time.perf_counter() - t0) / 200 * 1e3
    t0 = time.perf_counter()
    for _ in range(200):
        random_permutation_matrix(n, rng)
    baseline["csprng_P_time_ms"] = (time.perf_counter() - t0) / 200 * 1e3
    print(f"    CSPRNG: S {baseline['csprng_S_time_ms']:.3f} ms, P {baseline['csprng_P_time_ms']:.3f} ms per matrix; "
          f"uniform k x k invertible {raw_inv:.3f}")

    rows = []
    for lat in args.latents:
        for kind, train, size in (("S", S_train, k * k), ("P", P_train, n * n)):
            tf.keras.utils.set_random_seed(args.seed)
            vae = mv.build_vae(size, lat, args.hidden, args.beta, args.lr)
            X = np.stack([a.flatten() for a in train]).astype(np.float32)
            hist = vae.fit(X, epochs=args.epochs, batch_size=args.batch, verbose=0)
            scores = sample_prior(vae, lat, args.n_gen, rng)
            gen = (scores > 0.5).astype(np.uint8)

            t0 = time.perf_counter()
            for _ in range(50):
                vae.decoder(rng.standard_normal((1, lat)).astype(np.float32), training=False).numpy()
            dec_ms = (time.perf_counter() - t0) / 50 * 1e3

            row = {"latent_dim": lat, "component": kind, "train_size": len(train), "epochs": args.epochs,
                   "final_loss": float(hist.history["loss"][-1]), "decoder_ms_per_sample": dec_ms,
                   "decoder_params": vae.decoder.count_params()}
            if kind == "S":
                mats = [g_.reshape(k, k) for g_ in gen]
                valid = [mv.gf2_rank(M) == k for M in mats]
                good = [M for M, v in zip(mats, valid) if v]
                row.update({"valid_fraction_raw": float(np.mean(valid)),
                            **({f"valid_{kk}": vv for kk, vv in diversity(good, S_hash).items()} if good else {}),
                            "bit_bias": bit_bias(mats)})
                # time to obtain one valid S from the VAE (decode + rank test, rejection sampling)
                t0, got, tries = time.perf_counter(), 0, 0
                while got < 20 and tries < 2000:
                    tries += 1
                    M = (sample_prior(vae, lat, 1, rng)[0] > 0.5).astype(np.uint8).reshape(k, k)
                    got += int(mv.gf2_rank(M) == k)
                row["vae_ms_per_valid_S"] = (time.perf_counter() - t0) / max(got, 1) * 1e3 if got else float("nan")
            else:
                mats = [g_.reshape(n, n) for g_ in gen]
                raw_valid = [bool((M.sum(0) == 1).all() and (M.sum(1) == 1).all()) for M in mats]
                proj = [nearest_permutation(s.reshape(n, n)) for s in scores]
                chi2, dof, p_uni = chi2_uniform_p(np.stack(proj).sum(0).astype(float), args.n_gen / n)
                row.update({"valid_fraction_raw": float(np.mean(raw_valid)),
                            **{f"projected_{kk}": vv for kk, vv in diversity(proj, P_hash).items()},
                            "projected_chi2_position_uniformity_p": p_uni})
                t0 = time.perf_counter()
                for _ in range(20):
                    nearest_permutation(sample_prior(vae, lat, 1, rng)[0].reshape(n, n))
                row["vae_ms_per_projected_P"] = (time.perf_counter() - t0) / 20 * 1e3
            rows.append(row)
            print(f"    z={lat:4d} {kind}: " + ", ".join(f"{kk}={vv:.4g}" if isinstance(vv, float) else f"{kk}={vv}"
                                                    for kk, vv in row.items() if kk not in ("latent_dim", "component", "train_size", "epochs")))
    return baseline, rows


def mceliece_with_given_SP(base, S, perm):
    """Builds the public key S*G*P of an existing code with externally supplied S and P."""
    base.S, base.S_inv, base.p = S, mv.gf2_inv(S), perm
    base.public = mv.gf2_matmul(S, base.G)[:, perm]
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--t", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-gen", type=int, default=1000)
    ap.add_argument("--latents", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile-runs", type=int, default=20)
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument("--out", default="results_keygen_SP_v1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = {"script_version": VERSION, "config": vars(args), "environment": mv.environment_info()}
    print(f"{VERSION}")

    print("[A] Key-generation profile (time per step)")
    sets = [("toy", 6, 64, 4, "sgp", args.profile_runs),
            ("McEliece 1978", 10, 1024, 50, "sgp", 3),
            ("mceliece348864", 12, 3488, 64, "systematic", 2)]
    results["keygen_profile"] = run_profile(sets, args.profile_runs, args.seed)

    if not args.skip_vae:
        baseline, rows = run_vae_SP(args, rng)
        results["csprng_baseline"] = baseline
        results["vae_SP"] = rows

        # end-to-end: McEliece keys built with VAE-generated S and P (latent = first value)
        import tensorflow as tf
        k = mv.McEliece(args.m, args.n, args.t, "sgp", np.random.default_rng([args.seed, 99])).k
        lat = args.latents[0]
        tf.keras.utils.set_random_seed(args.seed)
        S_train = [mv.random_invertible_gf2(k, rng) for _ in range(args.n_train)]
        P_train = [random_permutation_matrix(args.n, rng) for _ in range(args.n_train)]
        vae_S = mv.build_vae(k * k, lat, args.hidden, args.beta, args.lr)
        vae_S.fit(np.stack([a.flatten() for a in S_train]).astype(np.float32), epochs=args.epochs, batch_size=args.batch, verbose=0)
        vae_P = mv.build_vae(args.n * args.n, lat, args.hidden, args.beta, args.lr)
        vae_P.fit(np.stack([a.flatten() for a in P_train]).astype(np.float32), epochs=args.epochs, batch_size=args.batch, verbose=0)
        ok_keys, ok_msgs, tried = 0, 0, 0
        for i in range(20):
            S = None
            for _ in range(500):
                M = (sample_prior(vae_S, lat, 1, rng)[0] > 0.5).astype(np.uint8).reshape(k, k)
                if mv.gf2_rank(M) == k:
                    S = M
                    break
            if S is None:
                continue
            Pm = nearest_permutation(sample_prior(vae_P, lat, 1, rng)[0].reshape(args.n, args.n))
            perm = np.argmax(Pm, axis=1)
            mc = mceliece_with_given_SP(mv.McEliece(args.m, args.n, args.t, "sgp", np.random.default_rng([args.seed, 500 + i])), S, perm)
            good = mv.self_test(mc, rng, 50)
            tried += 1
            ok_keys += int(good == 50)
            ok_msgs += good
        results["end_to_end_vae_SP_keys"] = {"latent_dim": lat, "keys_built": tried, "keys_fully_correct": ok_keys,
                                             "messages_correct": ok_msgs, "messages_total": tried * 50}
        print(f"[C] McEliece keys with VAE-generated S and P: {ok_keys}/{tried} keys correct "
              f"({ok_msgs}/{tried * 50} messages decrypted)")

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    for name in ("keygen_profile", "vae_SP"):
        rows = results.get(name)
        if rows:
            keys = sorted({kk for r in rows for kk in r}, key=lambda x: list(rows[0]).index(x) if x in rows[0] else 99)
            with open(os.path.join(args.out, f"table_{name}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    print(f"All results written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
