"""
Two further uses of a VAE in McEliece key generation (follow-up to vae_keygen_SP_v1.py).

Part G  VAE-guided Goppa polynomial generation.
        Generating the irreducible Goppa polynomial g(x) is the most expensive step of our key
        generation (48-75 %), because random monic polynomials of degree t are tested until one is
        irreducible (about t tries). Can a VAE trained on irreducible polynomials propose candidates
        that are irreducible more often, so that fewer tests are needed? Measured: hit rate against
        uniform sampling and against a cheap non-learned filter (no root in GF(2^m)), time per
        irreducible polynomial, diversity (unique outputs, min-entropy, memorised training polynomials),
        coefficient bias, and correctness of McEliece keys built with the generated polynomials.

Part D  VAE as a detector of weak key generators (positive control).
        The same VAE that fails on correct keys is trained on public keys from deliberately weakened
        generators: small seed space (b-bit seeds), a shared Goppa code and P with biased or uniform S,
        and the VAE-generated components of Parts G and B. If it reconstructs unseen keys of a weak
        generator but not of the correct one, its failure on correct keys reflects their randomness,
        not a weak model, and the VAE serves as a black-box test of key-generator randomness.
        Compared with three simple tests: duplicate keys, majority-bit predictor, pooled GF(2) rank.

Reuses the McEliece and VAE code of mceliece_vae_revised_v4.py (same folder).

Usage:
    python vae_goppa_detect_v1.py --out ../results/goppa_detect_v1
    python vae_goppa_detect_v1.py --quick --out smoke        # small smoke test
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from math import log2

import numpy as np

import mceliece_vae_revised_v4 as mv

VERSION = "goppa_detect_v1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# Helpers
# =============================================================================


def poly_to_bits(g, m):
    """Monic g of degree t -> t*m bits (coefficients g_0 .. g_{t-1}, bit b of each)."""
    c = np.asarray(g[:-1], dtype=np.int64)
    return ((c[:, None] >> np.arange(m)) & 1).astype(np.uint8).flatten()


def bits_to_poly(bits, m, t):
    c = (bits.reshape(t, m).astype(np.int64) << np.arange(m)).sum(1)
    return np.append(c, 1).astype(np.int64)


def has_root(F, g):
    """True if g has a root in GF(2^m) (then g is reducible). Cheap, vectorised."""
    return bool((mv.peval(F, g, np.arange(F.q, dtype=np.int64)) == 0).any())


def _irreducible_batch(args):
    """Worker: returns (list of irreducible polys, number of tests) using its own seed."""
    m, t, count, seed = args
    F = mv.GF2m(m)
    rng = np.random.default_rng(seed)
    out, tests = [], 0
    while len(out) < count:
        g = np.append(rng.integers(0, F.q, size=t), 1).astype(np.int64)
        tests += 1
        if mv.is_irreducible(F, g):
            out.append(g)
    return out, tests


def _test_batch(args):
    """Worker: irreducibility of a list of candidate polynomials."""
    m, polys = args
    F = mv.GF2m(m)
    return [bool(mv.is_irreducible(F, np.asarray(g, dtype=np.int64))) for g in polys]


def _uniform_hits(args):
    """Worker: hit rate of uniform sampling with and without the root filter."""
    m, t, count, seed = args
    F = mv.GF2m(m)
    rng = np.random.default_rng(seed)
    irr = filt_pass = filt_irr = 0
    for _ in range(count):
        g = np.append(rng.integers(0, F.q, size=t), 1).astype(np.int64)
        ok = mv.is_irreducible(F, g)
        irr += ok
        if not has_root(F, g):
            filt_pass += 1
            filt_irr += ok
    return irr, filt_pass, filt_irr


def parallel(fn, jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs))


def diversity(samples, train_set):
    keys = [s.tobytes() for s in samples]
    if not keys:
        return {"n": 0, "unique_fraction": float("nan"), "min_entropy_bits": float("nan"),
                "max_entropy_bits": float("nan"), "memorised_fraction": float("nan")}
    _, counts = np.unique(np.array(keys, dtype=object), return_counts=True)
    n = len(keys)
    return {"n": n, "unique_fraction": len(counts) / n,
            "min_entropy_bits": float(-log2(counts.max() / n)), "max_entropy_bits": float(log2(n)),
            "memorised_fraction": float(np.mean([k_ in train_set for k_ in keys]))}


def build_key(m, n, t, rng, g=None, L=None, S=None, perm=None):
    """McEliece key (S*G*P form) with optionally supplied g, support L, S and permutation P.
    Same construction as mv.McEliece.__init__; missing components are drawn from rng."""
    mc = object.__new__(mv.McEliece)
    mc.F = F = mv.GF2m(m)
    mc.m, mc.n, mc.t, mc.form = m, n, t, "sgp"
    L = rng.permutation(F.q)[:n].astype(np.int64) if L is None else np.asarray(L, dtype=np.int64)
    if g is None:
        while True:
            g = np.append(rng.integers(0, F.q, size=t), 1).astype(np.int64)
            if mv.is_irreducible(F, g):
                break
    mc.g = np.asarray(g, dtype=np.int64)
    ginv = F.inv(mv.peval(F, mc.g, L))
    Hb = np.zeros((m * t, n), dtype=np.uint8)
    for j in range(t):
        row = F.mul(F.pow(L, j), ginv)
        for b in range(m):
            Hb[j * m + b] = (row >> b) & 1
    R, piv = mv.gf2_rref(Hb)
    free = [c for c in range(n) if c not in set(piv)]
    k = len(free)
    B = np.zeros((k, n), dtype=np.uint8)
    B[np.arange(k), free] = 1
    if piv:
        B[:, piv] = R[:len(piv), free].T
    mc.k, mc.info_set = k, np.array(free)
    mc.S = mv.random_invertible_gf2(k, rng) if S is None else S
    mc.S_inv = mv.gf2_inv(mc.S)
    mc.p = rng.permutation(n) if perm is None else np.asarray(perm)
    mc.public = mv.gf2_matmul(mc.S, B)[:, mc.p]
    mc.L, mc.G = L, B
    mc._precompute_decoder()
    return mc


def biased_invertible(k, p, rng):
    """S = Lo * Up with unit-triangular factors whose free entries are Bernoulli(p): always
    invertible; p = 0.5 gives a near-uniform S, small p gives an S close to the identity."""
    Lo = np.tril((rng.random((k, k)) < p).astype(np.uint8), -1) | np.eye(k, dtype=np.uint8)
    Up = np.triu((rng.random((k, k)) < p).astype(np.uint8), 1) | np.eye(k, dtype=np.uint8)
    return mv.gf2_matmul(Lo, Up)


def train_vae(X, latent, hidden, beta, lr, epochs, batch, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    vae = mv.build_vae(X.shape[1], latent, hidden, beta, lr)
    hist = vae.fit(X.astype(np.float32), epochs=epochs, batch_size=batch, verbose=0)
    return vae, float(hist.history["loss"][-1])


def sample_prior(vae, latent, count, rng, mode):
    z = rng.standard_normal((count, latent)).astype(np.float32)
    prob = vae.decoder(z, training=False).numpy()
    if mode == "threshold":
        return (prob > 0.5).astype(np.uint8)
    return (rng.random(prob.shape) < prob).astype(np.uint8)   # Bernoulli sampling


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows:
        keys += [k_ for k_ in r if k_ not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# Part G: VAE-guided Goppa polynomial generation
# =============================================================================


def run_goppa(args, rng):
    rows, models = [], {}
    for spec in args.goppa_sets:
        name, m, t, n_train, n_gen, n_time = spec
        F = mv.GF2m(m)
        print(f"[G] {name}: m={m}, t={t}, {t * m}-bit polynomials, {n_train} training polynomials")

        # training data: irreducible polynomials from uniform sampling (parallel)
        t0 = time.perf_counter()
        per = int(np.ceil(n_train / args.workers))
        seeds = rng.integers(0, 2**63, size=args.workers)
        res = parallel(_irreducible_batch, [(m, t, per, int(s)) for s in seeds], args.workers)
        train = [g for r in res for g in r[0]][:n_train]
        train_tests = sum(r[1] for r in res)
        X = np.stack([poly_to_bits(g, m) for g in train])
        train_set = {x.tobytes() for x in X}
        print(f"    training data: {len(train)} polynomials in {time.perf_counter() - t0:.1f} s "
              f"({train_tests} tests, hit {len(train) * 1.0 / train_tests:.4f})")

        # baselines: uniform sampling and uniform + root filter (parallel, large sample)
        n_base = args.base_samples.get(name, 4000)
        per = int(np.ceil(n_base / args.workers))
        res = parallel(_uniform_hits, [(m, t, per, int(s)) for s in rng.integers(0, 2**63, size=args.workers)], args.workers)
        N = per * args.workers
        irr, fpass, firr = (sum(r[i] for r in res) for i in range(3))
        base = {"uniform_hit_rate": irr / N, "root_filter_pass_rate": fpass / N,
                "root_filter_hit_rate": firr / max(fpass, 1), "n_samples": N}
        # uniform bits: bias and diversity of uniform irreducible polynomials (reference)
        Xu = np.stack([poly_to_bits(g, m) for g in parallel(
            _irreducible_batch, [(m, t, max(1, n_gen // args.workers), int(s))
                                 for s in rng.integers(0, 2**63, size=args.workers)], args.workers)[0][0]])
        print(f"    uniform hit {base['uniform_hit_rate']:.4f} (1/t = {1 / t:.4f}); "
              f"root filter: pass {base['root_filter_pass_rate']:.3f}, hit {base['root_filter_hit_rate']:.4f}")

        # single-thread time per irreducible polynomial, uniform and uniform + root filter
        def time_uniform(filter_roots):
            t0, got, tests = time.perf_counter(), 0, 0
            while got < n_time:
                g = np.append(rng.integers(0, F.q, size=t), 1).astype(np.int64)
                if filter_roots and has_root(F, g):
                    continue
                tests += 1
                got += int(mv.is_irreducible(F, g))
            return (time.perf_counter() - t0) / n_time * 1e3, tests / n_time
        base["uniform_ms_per_irreducible"], base["uniform_tests_per_irreducible"] = time_uniform(False)
        base["filter_ms_per_irreducible"], base["filter_tests_per_irreducible"] = time_uniform(True)
        print(f"    time per irreducible: uniform {base['uniform_ms_per_irreducible']:.2f} ms, "
              f"root filter {base['filter_ms_per_irreducible']:.2f} ms")

        for lat in args.goppa_latents:
            hidden = [256, 128] if t * m <= 256 else [512, 256]
            vae, loss = train_vae(X, lat, hidden, args.beta, args.lr, args.epochs, args.batch, args.seed)
            models[(name, lat)] = vae
            for mode in ("threshold", "bernoulli"):
                cand = sample_prior(vae, lat, n_gen, rng, mode)
                polys = [bits_to_poly(b, m, t) for b in cand]
                chunks = np.array_split(np.arange(n_gen), args.workers)
                ok = np.array([v for r in parallel(_test_batch, [(m, [polys[i] for i in c]) for c in chunks], args.workers)
                               for v in r])
                hit = float(ok.mean())
                good = cand[ok]
                z, p = mv.two_proportion_test(hit, base["uniform_hit_rate"], n_gen)
                div = diversity(list(good), train_set)
                cand_div = diversity(list(cand), train_set)
                # single-thread time per irreducible polynomial from the VAE: candidates are decoded in
                # one batch (amortised decoder cost per candidate) and then tested one by one
                B = int(max(64, 3 * n_time / max(hit, 1e-3)))
                t0 = time.perf_counter()
                batch = sample_prior(vae, lat, B, rng, mode)
                dec_ms = (time.perf_counter() - t0) / B * 1e3
                t0, got, tests = time.perf_counter(), 0, 0
                while got < n_time and tests < B:
                    got += int(mv.is_irreducible(F, bits_to_poly(batch[tests], m, t)))
                    tests += 1
                ms = ((time.perf_counter() - t0) * 1e3 + tests * dec_ms) / got if got else float("nan")
                row = {"parameter_set": name, "m": m, "t": t, "poly_bits": t * m, "latent_dim": lat, "mode": mode,
                       "n_train": len(train), "final_loss": loss, "decoder_params": vae.decoder.count_params(),
                       "n_candidates": n_gen, "vae_hit_rate": hit,
                       "uniform_hit_rate": base["uniform_hit_rate"], "root_filter_hit_rate": base["root_filter_hit_rate"],
                       "hit_ratio_vs_uniform": hit / base["uniform_hit_rate"], "z_vs_uniform": z, "p_vs_uniform": p,
                       "candidates_unique_fraction": cand_div["unique_fraction"],
                       "irreducible_unique_fraction": div["unique_fraction"],
                       "irreducible_min_entropy_bits": div["min_entropy_bits"],
                       "max_entropy_bits": div["max_entropy_bits"],
                       "irreducible_memorised_fraction": div["memorised_fraction"],
                       "bit_bias_vae": float(np.mean(np.abs(cand.mean(0) - 0.5))),
                       "bit_bias_uniform_irreducible": float(np.mean(np.abs(Xu.mean(0) - 0.5))),
                       "vae_ms_per_irreducible": ms, "vae_tests_per_irreducible": tests / max(got, 1),
                       "uniform_ms_per_irreducible": base["uniform_ms_per_irreducible"],
                       "filter_ms_per_irreducible": base["filter_ms_per_irreducible"],
                       "speedup_vs_uniform": base["uniform_ms_per_irreducible"] / ms if got else float("nan")}
                rows.append(row)
                print(f"    d={lat:3d} {mode:9s}: hit {hit:.4f} (x{row['hit_ratio_vs_uniform']:.2f}, p={p:.2g}), "
                      f"unique {div['unique_fraction']:.3f}, H_min {div['min_entropy_bits']:.2f}/{div['max_entropy_bits']:.2f}, "
                      f"memorised {div['memorised_fraction']:.3f}, bias {row['bit_bias_vae']:.3f} "
                      f"(uniform {row['bit_bias_uniform_irreducible']:.3f}), {ms:.2f} ms/irreducible")
        rows.append({"parameter_set": name, "m": m, "t": t, "poly_bits": t * m, "latent_dim": 0, "mode": "baseline",
                     **{f"baseline_{k_}": v for k_, v in base.items()}})
    return rows, models


def goppa_end_to_end(args, models, rng):
    """McEliece keys whose Goppa polynomial comes from the VAE (Bernoulli mode), other parts random."""
    out = []
    for name, m, n, t in args.e2e_sets:
        lat = args.goppa_latents[0]
        vae = models.get((name, lat))
        if vae is None:
            continue
        F = mv.GF2m(m)
        keys_ok = msgs_ok = built = 0
        for i in range(args.e2e_keys):
            g = None
            for _ in range(500):
                cand = bits_to_poly(sample_prior(vae, lat, 1, rng, "bernoulli")[0], m, t)
                if mv.is_irreducible(F, cand):
                    g = cand
                    break
            if g is None:
                continue
            mc = build_key(m, n, t, rng, g=g)
            good = mv.self_test(mc, rng, args.e2e_msgs)
            built += 1
            keys_ok += int(good == args.e2e_msgs)
            msgs_ok += good
        out.append({"parameter_set": name, "m": m, "n": n, "t": t, "latent_dim": lat, "keys_built": built,
                    "keys_fully_correct": keys_ok, "messages_correct": msgs_ok,
                    "messages_total": built * args.e2e_msgs})
        print(f"[G] end-to-end {name}: {keys_ok}/{built} keys correct, {msgs_ok}/{built * args.e2e_msgs} messages")
    return out


# =============================================================================
# Part D: VAE as a detector of weak key generators
# =============================================================================


def _family_keys(args):
    """Worker: public keys (flattened) of one generator family. Returns (keys, ranks-ready array)."""
    family, param, count, seed, m, n, t, extra, offset = args
    rng = np.random.default_rng(seed)
    base_rng = np.random.default_rng(extra.get("shared_seed", 12345))
    out = []
    if family in ("shared_code_biased_S", "shared_code_uniform_S", "shared_code_vae_S"):
        F = mv.GF2m(m)
        L0 = base_rng.permutation(F.q)[:n].astype(np.int64)
        base = build_key(m, n, t, base_rng, L=L0)
        g0, p0, k = base.g, base.p, base.k
    for i in range(count):
        if family == "correct":
            mc = build_key(m, n, t, rng)
        elif family == "seed_bits":
            mc = build_key(m, n, t, np.random.default_rng(int(rng.integers(0, 2 ** param))))
        elif family == "shared_code_uniform_S":
            mc = build_key(m, n, t, rng, g=g0, L=L0, S=mv.random_invertible_gf2(k, rng), perm=p0)
        elif family == "shared_code_biased_S":
            mc = build_key(m, n, t, rng, g=g0, L=L0, S=biased_invertible(k, param, rng), perm=p0)
        elif family == "shared_code_vae_S":
            mc = build_key(m, n, t, rng, g=g0, L=L0, S=extra["S_list"][(offset + i) % len(extra["S_list"])], perm=p0)
        elif family == "random_code_vae_S":
            mc = build_key(m, n, t, rng, S=extra["S_list"][(offset + i) % len(extra["S_list"])])
        elif family == "vae_goppa":
            mc = build_key(m, n, t, rng, g=extra["g_list"][(offset + i) % len(extra["g_list"])])
        else:
            raise ValueError(family)
        out.append(mc.public.flatten())
    return np.stack(out)


def detection_tests(Xtr, Xte, n, vae_pred):
    """VAE reconstruction of unseen keys against three simple tests."""
    tr_set = {x.tobytes() for x in Xtr}
    dup = np.array([x.tobytes() in tr_set for x in Xte])
    majority = (Xtr.mean(0) > 0.5).astype(np.uint8)
    maj_acc = float((majority[None] == Xte).mean())
    vae_acc = float((vae_pred == Xte).mean())
    exact = np.all(vae_pred == Xte, axis=1)
    nbits = Xte.size
    z, p = mv.two_proportion_test(vae_acc, maj_acc, nbits)
    z5, p5 = mv.z_test_vs(vae_acc, nbits, 0.5)
    rows_all = Xte.reshape(-1, n)                    # all rows of all test keys
    rank = mv.gf2_rank(rows_all[: min(len(rows_all), 4 * n)])
    return {"vae_bit_accuracy": vae_acc, "majority_bit_accuracy": maj_acc,
            "vae_minus_majority": vae_acc - maj_acc, "z_vae_vs_majority": z, "p_vae_vs_majority": p,
            "p_vae_vs_half": p5,
            "vae_exact_recovery": float(exact.mean()),
            "vae_exact_recovery_novel_keys": float(exact[~dup].mean()) if (~dup).any() else float("nan"),
            "test_keys_duplicated_in_training": float(dup.mean()),
            "distinct_test_keys_fraction": len({x.tobytes() for x in Xte}) / len(Xte),
            "pooled_row_rank": int(rank), "row_rank_full": int(n)}


def run_detection(args, rng, g_lists, S_lists):
    m, n, t = 6, 64, 4
    families = [("correct", None, "Correct key generation")]
    families += [("seed_bits", b, f"{b}-bit seed") for b in args.seed_bits]
    families += [("shared_code_biased_S", p, f"Shared code and P, S with p={p}") for p in args.bias_levels]
    families += [("shared_code_uniform_S", None, "Shared code and P, uniform S")]
    extra_common = {}
    if S_lists.get("S"):
        families += [("shared_code_vae_S", None, "Shared code and P, VAE-generated S"),
                     ("random_code_vae_S", None, "Random code, VAE-generated S")]
        extra_common["S_list"] = S_lists["S"]
    if g_lists.get("toy"):
        families += [("vae_goppa", None, "VAE-generated Goppa polynomial")]
        extra_common["g_list"] = g_lists["toy"]

    total = args.det_train + args.det_test
    rows = []
    if args.det_families:
        families = [f_ for f_ in families if f_[0] in args.det_families]
    for fam, param, label in families:
        t0 = time.perf_counter()
        per = int(np.ceil(total / args.workers))
        jobs = [(fam, param, per, int(s), m, n, t, extra_common, w * per)
                for w, s in enumerate(rng.integers(0, 2**63, size=args.workers))]
        X = np.concatenate(parallel(_family_keys, jobs, args.workers))[:total]
        Xtr, Xte = X[:args.det_train], X[args.det_train:]
        vae, loss = train_vae(Xtr, args.det_latent, [256, 128],
                              args.beta if args.det_beta is None else args.det_beta, args.lr,
                              args.epochs if args.det_epochs is None else args.det_epochs, args.batch, args.seed)
        pred = mv.vae_reconstruct(vae, mv.vae_compress(vae, Xte))
        res = detection_tests(Xtr, Xte, n, pred)
        # a test flags the generator as weak if it predicts unseen keys better than chance (p < 0.001)
        _, p_maj = mv.z_test_vs(res["majority_bit_accuracy"], Xte.size, 0.5)
        flag = bool((res["p_vae_vs_half"] < 1e-3 and res["vae_bit_accuracy"] > 0.5) or res["vae_exact_recovery"] > 0)
        row = {"family": fam, "parameter": "" if param is None else param, "label": label,
               "n_train": len(Xtr), "n_test": len(Xte), "latent_dim": args.det_latent, "final_loss": loss,
               **res, "vae_flags_weak": flag,
               "majority_flags_weak": bool(p_maj < 1e-3 and res["majority_bit_accuracy"] > 0.5),
               "duplicate_flags_weak": bool(res["test_keys_duplicated_in_training"] > 0),
               "rank_flags_weak": bool(res["pooled_row_rank"] < n),
               "seconds": time.perf_counter() - t0}
        rows.append(row)
        print(f"[D] {label:40s} VAE {res['vae_bit_accuracy']:.4f} vs majority {res['majority_bit_accuracy']:.4f} "
              f"(p={res['p_vae_vs_majority']:.2g}); exact {res['vae_exact_recovery']:.3f} "
              f"(novel {res['vae_exact_recovery_novel_keys']:.3f}); dup {res['test_keys_duplicated_in_training']:.3f}; "
              f"rank {res['pooled_row_rank']}/{n}; flag={flag}  [{row['seconds']:.0f} s]")
    return rows


def vae_S_list(args, rng, k, count):
    """S matrices from a VAE trained on uniform invertible S (as in vae_keygen_SP_v1, z=16), rank-checked."""
    S_train = [mv.random_invertible_gf2(k, rng) for _ in range(2000 if not args.quick else 300)]
    vae, _ = train_vae(np.stack([s.flatten() for s in S_train]), 16, [256, 128], args.beta, args.lr,
                       args.epochs, args.batch, args.seed)
    out, tries = [], 0
    while len(out) < count and tries < 50 * count:
        tries += 1
        M = sample_prior(vae, 16, 1, rng, "threshold")[0].reshape(k, k)
        if mv.gf2_rank(M) == k:
            out.append(M)
    return out


# =============================================================================
# Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small smoke test")
    ap.add_argument("--skip-goppa", action="store_true")
    ap.add_argument("--skip-detect", action="store_true")
    ap.add_argument("--goppa-latents", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--e2e-keys", type=int, default=20)
    ap.add_argument("--e2e-msgs", type=int, default=50)
    ap.add_argument("--det-train", type=int, default=1000)
    ap.add_argument("--det-test", type=int, default=200)
    ap.add_argument("--det-latent", type=int, default=32)
    ap.add_argument("--det-beta", type=float, default=None, help="KL weight of the detector (default: --beta)")
    ap.add_argument("--det-epochs", type=int, default=None, help="epochs of the detector (default: --epochs)")
    ap.add_argument("--det-families", nargs="*", default=None, help="restrict Part D to these families")
    ap.add_argument("--seed-bits", type=int, nargs="+", default=[8, 10, 12, 16])
    ap.add_argument("--bias-levels", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--out", default="results_goppa_detect_v1")
    args = ap.parse_args()

    # (name, m, t, n_train, n_candidates, polynomials timed single-thread)
    if args.quick:
        args.goppa_sets = [("toy", 6, 4, 400, 400, 20), ("small", 8, 8, 300, 300, 10)]
        args.base_samples = {"toy": 2000, "small": 1000}
        args.e2e_sets = [("toy", 6, 64, 4)]
        args.e2e_keys, args.e2e_msgs, args.epochs = 3, 10, 5
        args.det_train, args.det_test, args.seed_bits, args.bias_levels = 200, 50, [8], [0.05]
    else:
        args.goppa_sets = [("toy", 6, 4, 5000, 4000, 200),
                           ("small", 8, 8, 5000, 4000, 100),
                           ("medium", 10, 20, 2000, 2000, 30),
                           ("mceliece348864", 12, 64, 1000, 1000, 10)]
        args.base_samples = {"toy": 40000, "small": 20000, "medium": 8000, "mceliece348864": 4000}
        args.e2e_sets = [("toy", 6, 64, 4), ("small", 8, 256, 8)]

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = {"script_version": VERSION, "config": {k_: v for k_, v in vars(args).items()},
               "environment": mv.environment_info()}
    print(f"{VERSION}  workers={args.workers}")
    t_start = time.perf_counter()

    g_lists = {}
    if not args.skip_goppa:
        rows, models = run_goppa(args, rng)
        results["goppa"] = rows
        write_csv(os.path.join(args.out, "table_goppa.csv"), rows)
        results["goppa_end_to_end"] = goppa_end_to_end(args, models, rng)
        write_csv(os.path.join(args.out, "table_goppa_end_to_end.csv"), results["goppa_end_to_end"])
        # irreducible VAE polynomials of the toy model, for the detection experiment
        vae = models.get(("toy", args.goppa_latents[0]))
        if vae is not None:
            F = mv.GF2m(6)
            cand = sample_prior(vae, args.goppa_latents[0], 6000, rng, "bernoulli")
            g_lists["toy"] = [g for g in (bits_to_poly(b, 6, 4) for b in cand) if mv.is_irreducible(F, g)]
            print(f"[G] {len(g_lists['toy'])} irreducible VAE polynomials kept for Part D "
                  f"({len({g.tobytes() for g in g_lists['toy']})} distinct)")
        with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=float)

    if not args.skip_detect:
        S_lists = {"S": vae_S_list(args, rng, 40, args.det_train + args.det_test)}
        print(f"[D] {len(S_lists['S'])} VAE-generated S matrices ({len({s.tobytes() for s in S_lists['S']})} distinct)")
        rows = run_detection(args, rng, g_lists, S_lists)
        results["detection"] = rows
        write_csv(os.path.join(args.out, "table_detection.csv"), rows)

    results["total_seconds"] = time.perf_counter() - t_start
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Total {results['total_seconds'] / 60:.1f} min. All results written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
