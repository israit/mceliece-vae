"""
McEliece public-key compression with a Variational Autoencoder (VAE)
--------------------------------------------------------------------
Revised, reproducible experiment for manuscript MJCSC-D-26-00004 (R2).

What this script does (each item is mapped to a reviewer comment in README_revision.md):

  * A real McEliece cryptosystem over binary Goppa codes (GF(2^m) arithmetic,
    irreducible Goppa polynomial, algebraic decoding of t errors).  Two public-key
    forms are supported and stated explicitly:
        --form sgp         original McEliece   K_pub = S*G*P          (k x n)
        --form systematic  Classic-McEliece-like systematic form [I_k | T],
                           public key = T                          (k x (n-k))
  * All GF(2) linear algebra is done mod 2 (no floating-point inverse/rank).
  * Correctness is checked by a self-test: decrypt(encrypt(m)) == m.
  * VAE trained ONCE on a set of training keys and evaluated on DISJOINT,
    freshly generated test keys (generalisation regime), and separately the
    per-key "repeats" regime (memorisation regime).
  * Correctness metric = EXACT key equality and decryption-failure rate (DFR),
    not bit accuracy.  A SHA-256 hash of K_pub is sent with Z; the sender
    verifies it and falls back to the full key on mismatch.
  * Honest compression ratios that include latent precision (float32), the hash,
    the fallback rate and the decoder size.
  * Real measurements of gzip / bzip2 / LZMA on the packed public key.
  * Timing with N repetitions: mean, standard deviation and 95% CI; VAE inference
    is counted inside the encryption cost; decryption is shown to be unchanged.
  * Explicitly specified leakage probe (attack model, features, training data).
  * Seeds, configuration and library versions are written to the output folder.

Usage examples
    python mceliece_vae_revised.py --preset toy
    python mceliece_vae_revised.py --preset small --form sgp
    python mceliece_vae_revised.py --preset mceliece348864 --skip-vae
"""

import argparse
import bz2
import csv
import gzip
import hashlib
import json
import lzma
import os
import platform
import sys
import time
from math import erfc, lgamma, log, log2, sqrt

import numpy as np

VERSION = "v4"   # v1: first Colab run; v2: validation split, training logs, run summary, ablation options;
                 # v3: bit-flip sensitivity test (reviewer comment 2);
                 # v4: public-key form sizes, majority-bit baseline, lossless VAE + residual scheme, p-values

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")      # reproducible results on GPU as well

# =============================================================================
# 1. Finite field GF(2^m)
# =============================================================================


def _find_primitive_poly(m):
    """Smallest primitive polynomial of degree m (x is a generator of GF(2^m)*)."""
    q = 1 << m
    for poly in range(q + 1, 2 * q, 2):
        x, order = 1, 0
        while True:
            x <<= 1
            if x & q:
                x ^= poly
            order += 1
            if x == 1 or order > q:
                break
        if x == 1 and order == q - 1:
            return poly
    raise ValueError(f"no primitive polynomial found for m={m}")


class GF2m:
    """GF(2^m) with exp/log tables; all operations are vectorised over numpy arrays."""

    def __init__(self, m):
        self.m = m
        self.q = 1 << m
        self.poly = _find_primitive_poly(m)
        self.exp = np.zeros(2 * self.q, dtype=np.int64)
        self.log = np.zeros(self.q, dtype=np.int64)
        x = 1
        for i in range(self.q - 1):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & self.q:
                x ^= self.poly
        self.exp[self.q - 1:2 * (self.q - 1)] = self.exp[:self.q - 1]

    def mul(self, a, b):
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        r = self.exp[self.log[a] + self.log[b]]
        return np.where((a == 0) | (b == 0), 0, r)

    def inv(self, a):
        a = np.asarray(a, dtype=np.int64)
        if np.any(a == 0):
            raise ZeroDivisionError("inverse of 0 in GF(2^m)")
        return self.exp[(self.q - 1 - self.log[a]) % (self.q - 1)]

    def pow(self, a, e):
        a = np.asarray(a, dtype=np.int64)
        if e == 0:
            return np.ones_like(a)
        r = self.exp[(self.log[a] * e) % (self.q - 1)]
        return np.where(a == 0, 0, r)


# =============================================================================
# 2. Polynomials over GF(2^m)  (numpy int64 arrays, lowest degree first)
# =============================================================================

_ZERO = np.zeros(1, dtype=np.int64)


def pdeg(p):
    nz = np.nonzero(p)[0]
    return int(nz[-1]) if len(nz) else -1


def ptrim(p):
    d = pdeg(p)
    return p[:d + 1].copy() if d >= 0 else _ZERO.copy()


def padd(a, b):
    r = np.zeros(max(len(a), len(b)), dtype=np.int64)
    r[:len(a)] ^= a
    r[:len(b)] ^= b
    return ptrim(r)


def pmul(F, a, b):
    a, b = ptrim(a), ptrim(b)
    r = np.zeros(len(a) + len(b) - 1, dtype=np.int64)
    for i, c in enumerate(a):
        if c:
            r[i:i + len(b)] ^= F.mul(c, b)
    return ptrim(r)


def pdivmod(F, a, b):
    a, b = ptrim(a), ptrim(b)
    db, da = pdeg(b), pdeg(a)
    if db < 0:
        raise ZeroDivisionError("polynomial division by zero")
    if da < db:
        return _ZERO.copy(), a
    inv_lead = F.inv(b[db])
    q = np.zeros(da - db + 1, dtype=np.int64)
    for i in range(da, db - 1, -1):
        c = a[i]
        if c:
            f = F.mul(c, inv_lead)
            q[i - db] = f
            a[i - db:i + 1] ^= F.mul(f, b)
    rem = ptrim(a[:db]) if db > 0 else _ZERO.copy()
    return ptrim(q), rem


def pmod(F, a, b):
    return pdivmod(F, a, b)[1]


def pgcd(F, a, b):
    a, b = ptrim(a), ptrim(b)
    while pdeg(b) >= 0:
        a, b = b, pmod(F, a, b)
    return a


def psquare(F, a):
    """Squaring in characteristic 2: (sum a_i x^i)^2 = sum a_i^2 x^(2i)."""
    a = ptrim(a)
    r = np.zeros(2 * len(a) - 1, dtype=np.int64)
    r[0::2] = F.mul(a, a)
    return ptrim(r)


def peval(F, p, xs):
    xs = np.asarray(xs, dtype=np.int64)
    r = np.zeros_like(xs)
    for c in p[::-1]:
        r = F.mul(r, xs) ^ c
    return r


def is_irreducible(F, g):
    """Ben-Or test over GF(q), q = 2^m: g has no irreducible factor of degree <= deg(g)/2."""
    t = pdeg(g)
    x = np.array([0, 1], dtype=np.int64)
    h = x.copy()
    for _ in range(1, t // 2 + 1):
        for _ in range(F.m):            # h <- h^q  (Frobenius = m squarings)
            h = pmod(F, psquare(F, h), g)
        if pdeg(pgcd(F, g, padd(h, x))) > 0:
            return False
    return True


# =============================================================================
# 3. Linear algebra over GF(2)
# =============================================================================


def gf2_rref(M):
    """Reduced row-echelon form over GF(2). Returns (R, pivot_columns)."""
    A = (np.asarray(M, dtype=np.uint8) & 1).copy()
    rows, cols = A.shape
    pivots, r = [], 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0:
            continue
        p = r + nz[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        mask = A[:, c].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        pivots.append(c)
        r += 1
    return A, pivots


def gf2_rank(M):
    return len(gf2_rref(M)[1])


def gf2_inv(M):
    k = M.shape[0]
    R, piv = gf2_rref(np.hstack([M, np.eye(k, dtype=np.uint8)]))
    if piv[:k] != list(range(k)):
        raise ValueError("matrix is singular over GF(2)")
    return R[:k, k:].copy()


def gf2_matmul(A, B):
    return ((A.astype(np.int32) @ B.astype(np.int32)) & 1).astype(np.uint8)


def random_invertible_gf2(k, rng):
    while True:
        S = rng.integers(0, 2, size=(k, k), dtype=np.uint8)
        if gf2_rank(S) == k:
            return S


# =============================================================================
# 4. Binary Goppa code and McEliece
# =============================================================================


class McEliece:
    """McEliece over a binary Goppa code Gamma(L, g), deg g = t, |L| = n, code dimension k."""

    def __init__(self, m, n, t, form, rng):
        if form not in ("sgp", "systematic"):
            raise ValueError("form must be 'sgp' or 'systematic'")
        self.F = F = GF2m(m)
        self.m, self.n, self.t, self.form = m, n, t, form
        if n > F.q:
            raise ValueError(f"n={n} exceeds field size 2^{m}={F.q}")

        # private: support L (n distinct field elements) and irreducible Goppa polynomial g
        L = rng.permutation(F.q)[:n].astype(np.int64)
        while True:
            g = np.append(rng.integers(0, F.q, size=t), 1).astype(np.int64)
            if is_irreducible(F, g):
                break
        self.g = g

        # parity-check matrix over GF(2^m): H[j, i] = L_i^j / g(L_i), expanded to m*t binary rows
        ginv = F.inv(peval(F, g, L))
        Hb = np.zeros((m * t, n), dtype=np.uint8)
        for j in range(t):
            row = F.mul(F.pow(L, j), ginv)
            for b in range(m):
                Hb[j * m + b] = (row >> b) & 1

        # generator matrix = basis of the GF(2) null space of Hb
        R, piv = gf2_rref(Hb)
        free = [c for c in range(n) if c not in set(piv)]
        k = len(free)
        B = np.zeros((k, n), dtype=np.uint8)
        B[np.arange(k), free] = 1
        if piv:
            B[:, piv] = R[:len(piv), free].T
        self.k = k

        if form == "systematic":
            perm = free + list(piv)           # columns reordered so B = [I_k | T]
            L = L[perm]
            B = B[:, perm]
            self.info_set = np.arange(k)
            self.T = B[:, k:].copy()
            self.public = self.T
        else:
            self.info_set = np.array(free)
            self.S = random_invertible_gf2(k, rng)
            self.S_inv = gf2_inv(self.S)
            self.p = rng.permutation(n)       # column permutation P
            self.public = gf2_matmul(self.S, B)[:, self.p]
        self.L, self.G = L, B
        self._precompute_decoder()

    # ---------------------------------------------------------------- decoding
    def _precompute_decoder(self):
        """Rows of 1/(x - L_i) mod g(x)^2 (binary Goppa: Gamma(L, g) = Gamma(L, g^2))."""
        F, t = self.F, self.t
        G2 = pmul(F, self.g, self.g)
        a = self.L
        Q = np.zeros((self.n, 2 * t), dtype=np.int64)
        Q[:, 2 * t - 1] = G2[2 * t]
        for j in range(2 * t - 1, 0, -1):
            Q[:, j - 1] = G2[j] ^ F.mul(a, Q[:, j])
        Q = F.mul(Q, F.inv(peval(F, G2, a))[:, None])
        self.G2, self.inv_rows = G2, Q

    def decode(self, y):
        """Correct up to t errors in y (codeword of Gamma(L, g) + e). Returns codeword or None."""
        F, t = self.F, self.t
        S = np.bitwise_xor.reduce(self.inv_rows[y.astype(bool)], axis=0) if y.any() else None
        if S is None or not S.any():
            return y.copy()
        r0, r1 = self.G2, ptrim(S)
        u0, u1 = _ZERO.copy(), np.array([1], dtype=np.int64)
        while pdeg(r1) >= t:                  # Sugiyama / extended Euclid on the key equation
            q, r = pdivmod(F, r0, r1)
            r0, r1 = r1, r
            u0, u1 = u1, padd(u0, pmul(F, q, u1))
        sigma = u1
        roots = np.nonzero(peval(F, sigma, self.L) == 0)[0]
        if len(roots) != pdeg(sigma) or len(roots) > t:
            return None                       # decoding failure
        x = y.copy()
        x[roots] ^= 1
        return x

    # ------------------------------------------------------- encrypt / decrypt
    def encrypt(self, msg, rng, public=None):
        """c = m * K_pub + e, wt(e) = t.  `public` may be a reconstructed (VAE) key."""
        K = self.public if public is None else public
        if self.form == "systematic":
            cw = np.concatenate([msg.astype(np.uint8), gf2_matmul(msg[None], K)[0]])
        else:
            cw = gf2_matmul(msg[None], K)[0]
        e = np.zeros(self.n, dtype=np.uint8)
        e[rng.choice(self.n, self.t, replace=False)] = 1
        return cw ^ e

    def decrypt(self, c):
        if self.form == "systematic":
            x = self.decode(c)
            return None if x is None else x[:self.k]
        y = np.zeros(self.n, dtype=np.uint8)
        y[self.p] = c                         # undo the column permutation P
        x = self.decode(y)
        if x is None:
            return None
        mS = x[self.info_set]                 # B is the identity on the information set
        return gf2_matmul(mS[None], self.S_inv)[0]

    # ------------------------------------------------------------ bookkeeping
    @property
    def public_bits(self):
        return int(self.public.size)

    def public_hash(self, K=None):
        K = self.public if K is None else K
        return hashlib.sha256(np.packbits(K.astype(np.uint8)).tobytes()).digest()

    def private_target_bits(self, max_bits=512):
        """Bits of the private key used as targets in the leakage probe."""
        parts = []
        if self.form == "sgp":
            parts.append(self.S.flatten())
        parts.append(((self.g[:, None] >> np.arange(self.m)) & 1).flatten().astype(np.uint8))
        parts.append(((self.L[:, None] >> np.arange(self.m)) & 1).flatten().astype(np.uint8))
        return np.concatenate(parts)[:max_bits]


def bitflip_experiment(mc, rng, levels, n_msgs):
    """Encrypt under the true public key with j random bits flipped, decrypt with the true private key.
    Separates decoding failures (detected) from wrong plaintexts returned silently (undetected)."""
    rows = []
    for j in levels:
        detected = silent = 0
        for _ in range(n_msgs):
            K = mc.public.copy()
            if j:
                K.flat[rng.choice(K.size, j, replace=False)] ^= 1
            msg = rng.integers(0, 2, size=mc.k, dtype=np.uint8)
            d = mc.decrypt(mc.encrypt(msg, rng, public=K))
            if d is None:
                detected += 1
            elif not np.array_equal(d, msg):
                silent += 1
        p = (detected + silent) / n_msgs
        rows.append({"flipped_bits": j, "n_messages": n_msgs, "dfr": p,
                     "dfr_ci95": 1.96 * np.sqrt(p * (1 - p) / n_msgs),
                     "decoding_failure_detected": detected / n_msgs,
                     "wrong_plaintext_undetected": silent / n_msgs})
    return rows


def self_test(mc, rng, trials):
    ok = 0
    for _ in range(trials):
        msg = rng.integers(0, 2, size=mc.k, dtype=np.uint8)
        d = mc.decrypt(mc.encrypt(msg, rng))
        ok += int(d is not None and np.array_equal(d, msg))
    return ok


# =============================================================================
# 5. VAE (TensorFlow / Keras)
# =============================================================================


def build_vae(input_dim, latent_dim, hidden, beta, lr, activation="relu"):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    enc_in = keras.Input(shape=(input_dim,))
    h = enc_in
    for u in hidden:
        h = layers.Dense(u, activation=activation)(h)
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
    encoder = keras.Model(enc_in, [z_mean, z_log_var], name="encoder")

    dec_in = keras.Input(shape=(latent_dim,))
    h = dec_in
    for u in reversed(hidden):
        h = layers.Dense(u, activation=activation)(h)
    dec_out = layers.Dense(input_dim, activation="sigmoid")(h)
    decoder = keras.Model(dec_in, dec_out, name="decoder")

    def losses(x, zm, zlv, xr):
        xr = tf.clip_by_value(xr, 1e-7, 1 - 1e-7)
        rec = -tf.reduce_mean(tf.reduce_sum(x * tf.math.log(xr) + (1 - x) * tf.math.log(1 - xr), axis=1))
        kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + zlv - tf.square(zm) - tf.exp(zlv), axis=1))
        return rec + beta * kl, rec, kl

    class VAE(keras.Model):
        """Loss = BCE (summed over bits) + beta * KL(q(z|x) || N(0, I))."""

        def __init__(self):
            super().__init__()
            self.encoder, self.decoder, self.beta = encoder, decoder, beta

        def call(self, x):
            zm, _ = self.encoder(x)
            return self.decoder(zm)

        def train_step(self, data):
            x = data[0] if isinstance(data, tuple) else data
            with tf.GradientTape() as tape:
                zm, zlv = self.encoder(x, training=True)
                z = zm + tf.exp(0.5 * zlv) * tf.random.normal(tf.shape(zm))
                loss, rec, kl = losses(x, zm, zlv, self.decoder(z, training=True))
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {"loss": loss, "reconstruction": rec, "kl": kl}

        def test_step(self, data):
            x = data[0] if isinstance(data, tuple) else data
            zm, zlv = self.encoder(x, training=False)
            loss, rec, kl = losses(x, zm, zlv, self.decoder(zm, training=False))
            return {"loss": loss, "reconstruction": rec, "kl": kl}

    vae = VAE()
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    vae(np.zeros((1, input_dim), dtype=np.float32))
    return vae


def vae_compress(vae, K_flat):
    """Deployment encoder: Z = mu(K_pub)  (deterministic, so Z is a function of K_pub only)."""
    return vae.encoder(K_flat.astype(np.float32), training=False)[0].numpy().astype(np.float32)


def vae_reconstruct(vae, Z):
    return (vae.decoder(Z, training=False).numpy() > 0.5).astype(np.uint8)


def decoder_param_count(input_dim, latent_dim, hidden):
    dims = [latent_dim] + list(reversed(hidden)) + [input_dim]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


# =============================================================================
# 6. Metrics and statistics
# =============================================================================


def mean_std_ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {"mean": float(x.mean()), "std": sd, "ci95": 1.96 * sd / np.sqrt(len(x)), "n": int(len(x))}


def log2_binom(n, w):
    return (lgamma(n + 1) - lgamma(w + 1) - lgamma(n - w + 1)) / log(2)


def residual_cost_bits(delta):
    """Bits needed to send the correction mask Delta = K xor K_hat losslessly:
    'enum' = ideal enumerative code (log2 C(N, w) + log2(N + 1)), 'lzma' = an actual LZMA-9 stream."""
    n, w = int(delta.size), int(delta.sum())
    enum = log2_binom(n, w) + log2(n + 1)
    lz = len(lzma.compress(np.packbits(delta.astype(np.uint8)).tobytes(), preset=9)) * 8
    return w, enum, lz


def z_test_vs(acc, n, p0=0.5):
    """Two-sided normal-approximation test of a bit accuracy against p0 over n bits."""
    z = (acc - p0) / sqrt(p0 * (1 - p0) / n)
    return z, erfc(abs(z) / sqrt(2))


def two_proportion_test(a1, a2, n):
    """Two-sided test that two accuracies measured on n bits each are equal."""
    p = (a1 + a2) / 2
    se = sqrt(2 * p * (1 - p) / n) if 0 < p < 1 else 1e-12
    z = (a1 - a2) / se
    return z, erfc(abs(z) / sqrt(2))


def public_key_form_sizes():
    """Public-key size of the same code in the original S*G*P form (k x n) and in systematic form
    (k x (n - k)), assuming k = n - m*t (full-rank parity-check matrix)."""
    sets = [("toy (this work)", 6, 64, 4), ("McEliece 1978", 10, 1024, 50),
            ("mceliece348864", 12, 3488, 64), ("mceliece460896", 13, 4608, 96)]
    rows = []
    for name, m, n, t in sets:
        k = n - m * t
        sgp, sysb = k * n, k * (n - k)
        rows.append({"parameter_set": name, "m": m, "n": n, "t": t, "k": k,
                     "sgp_bits": sgp, "sgp_bytes": sgp / 8, "sgp_KiB": sgp / 8 / 1024,
                     "systematic_bits": sysb, "systematic_bytes": sysb / 8, "systematic_KiB": sysb / 8 / 1024,
                     "systematic_ratio": sgp / sysb})
    return rows


def evaluate_reconstruction(keys, K_hat, n_msgs, rng):
    """Per-key exact recovery, bit accuracy, Hamming distance, DFR, hash-check outcome,
    and the cost of the lossless VAE + residual scheme."""
    rows = []
    for mc, Kh in zip(keys, K_hat):
        Kh = Kh.reshape(mc.public.shape)
        ham = int(np.count_nonzero(Kh != mc.public))
        _, res_enum, res_lzma = residual_cost_bits(Kh ^ mc.public)
        fails = 0
        for _ in range(n_msgs):
            msg = rng.integers(0, 2, size=mc.k, dtype=np.uint8)
            d = mc.decrypt(mc.encrypt(msg, rng, public=Kh))
            fails += int(d is None or not np.array_equal(d, msg))
        rows.append({
            "hamming": ham,
            "bit_acc": 1 - ham / mc.public_bits,
            "exact": int(ham == 0),
            "dfr_without_check": fails / n_msgs,
            "hash_ok": int(mc.public_hash(Kh) == mc.public_hash()),
            "residual_bits_enum": res_enum,
            "residual_bits_lzma": res_lzma,
        })
    return rows


def summarise_rows(rows):
    return {
        "residual_bits_enum": mean_std_ci([r["residual_bits_enum"] for r in rows]),
        "residual_bits_lzma": mean_std_ci([r["residual_bits_lzma"] for r in rows]),
        "exact_recovery_rate": float(np.mean([r["exact"] for r in rows])),
        "bit_accuracy": mean_std_ci([r["bit_acc"] for r in rows]),
        "hamming_bits": mean_std_ci([r["hamming"] for r in rows]),
        "dfr_without_check": float(np.mean([r["dfr_without_check"] for r in rows])),
        "fallback_rate": float(1 - np.mean([r["hash_ok"] for r in rows])),
        "dfr_with_hash_check_and_fallback": 0.0,   # mismatch => full key is fetched
    }


def compression_ratios(key_bits, latent_dim, decoder_bits, fallback_rate, latent_bits_per_dim=32, hash_bits=256):
    latent_bits = latent_dim * latent_bits_per_dim
    sent = latent_bits + hash_bits + fallback_rate * key_bits
    return {
        "key_bits": key_bits,
        "latent_bits": latent_bits,
        "naive_ratio_key_over_latent": key_bits / latent_bits,
        "ratio_with_hash_and_fallback": key_bits / sent,
        "ratio_including_decoder_one_key": key_bits / (sent + decoder_bits),
        "decoder_bits": decoder_bits,
    }


def logistic_probe(Xtr, Ytr, Xte, Yte, l2=1e-2, iters=2000, lr=0.1):
    """Attack model: L2-regularised logistic regression, one output per private-key bit,
    features = latent vector Z (public information), trained on training keys
    (whose private keys the attacker is assumed to know), tested on unseen keys."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    Bt = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    W = np.zeros((A.shape[1], Ytr.shape[1]))
    for _ in range(iters):
        P = 1 / (1 + np.exp(-np.clip(A @ W, -30, 30)))
        W -= lr * (A.T @ (P - Ytr) / len(A) + l2 * W)
    pred = (Bt @ W > 0).astype(np.uint8)
    majority = (Ytr.mean(0) > 0.5).astype(np.uint8)
    acc = float((pred == Yte).mean())
    n_obs = Yte.size
    return {
        "attack_model": "logistic regression (L2=%.0e), features=Z, targets=private-key bits" % l2,
        "n_train_keys": int(len(Xtr)), "n_test_keys": int(len(Xte)), "n_target_bits": int(Ytr.shape[1]),
        "bit_accuracy": acc,
        "bit_accuracy_ci95": 1.96 * np.sqrt(acc * (1 - acc) / n_obs),
        "majority_baseline_accuracy": float((majority[None] == Yte).mean()),
        "exact_private_key_recovery_rate": float(np.mean(np.all(pred == Yte, axis=1))),
    }


# =============================================================================
# 7. Experiments
# =============================================================================

PRESETS = {
    # m, n, t, default form, hidden layers, n_train, n_test
    "toy": dict(m=6, n=64, t=4, form="sgp", hidden=[256, 128], n_train=1000, n_test=200),
    "small": dict(m=8, n=256, t=8, form="sgp", hidden=[512, 256], n_train=800, n_test=200),
    "mceliece348864": dict(m=12, n=3488, t=64, form="systematic", hidden=[512, 256], n_train=0, n_test=0),
}


def generate_keys(count, P, form, seed_seq):
    keys = []
    for child in seed_seq.spawn(count):
        keys.append(McEliece(P["m"], P["n"], P["t"], form, np.random.default_rng(child)))
    return keys


def baseline_compressors(keys):
    out = {}
    for name, fn in (("gzip-9", lambda b: gzip.compress(b, 9)),
                     ("bzip2-9", lambda b: bz2.compress(b, 9)),
                     ("lzma-9", lambda b: lzma.compress(b, preset=9))):
        ratios = []
        for mc in keys:
            raw = np.packbits(mc.public).tobytes()
            ratios.append(len(raw) / len(fn(raw)))
        out[name] = mean_std_ci(ratios)
    return out


def timing_experiment(mc, rng, runs, vae=None, Z=None):
    enc, enc_vae, dec = [], [], []
    for _ in range(runs):
        msg = rng.integers(0, 2, size=mc.k, dtype=np.uint8)
        t0 = time.perf_counter()
        c = mc.encrypt(msg, rng)
        enc.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        mc.decrypt(c)
        dec.append((time.perf_counter() - t0) * 1e3)
        if vae is not None:
            t0 = time.perf_counter()          # sender: reconstruct, verify hash, encrypt
            Kh = vae_reconstruct(vae, Z).reshape(mc.public.shape)
            K_use = Kh if mc.public_hash(Kh) == mc.public_hash() else mc.public
            mc.encrypt(msg, rng, public=K_use)
            enc_vae.append((time.perf_counter() - t0) * 1e3)
    res = {"encrypt_ms": mean_std_ci(enc), "decrypt_ms": mean_std_ci(dec)}
    if vae is not None:
        res["encrypt_with_vae_inference_ms"] = mean_std_ci(enc_vae)
        res["note"] = ("decryption uses only (S, G, P) / (g, L); it is identical with or without the VAE. "
                       "encrypt_with_vae_inference_ms = decoder inference + SHA-256 check + encryption; "
                       "the network time of downloading the full key after a failed check is not included")
    return res


def environment_info():
    info = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "numpy": np.__version__}
    try:
        import tensorflow as tf
        info["tensorflow"] = tf.__version__
    except ImportError:
        info["tensorflow"] = None
    return info


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_figures(out, gen_table, ratio_table, baselines, timing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if gen_table:
        lat = sorted({r["latent_dim"] for r in gen_table})
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for split, mk in (("train", "o-"), ("test", "s--")):
            def per_latent(key):
                return [np.mean([r[key] for r in gen_table if r["latent_dim"] == l and r["split"] == split]) * 100 for l in lat]
            ax[0].plot(lat, per_latent("exact_recovery_rate"), mk, label=f"{split} keys")
            ax[1].plot(lat, per_latent("bit_accuracy_mean"), mk, label=f"{split} keys")
            ax[2].plot(lat, per_latent("dfr_without_check"), mk, label=f"{split} keys")
        ax[0].set(xlabel="Latent dimension (float32 values)", ylabel="Exact key recovery (%)", ylim=(-5, 105))
        ax[1].set(xlabel="Latent dimension (float32 values)", ylabel="Bit accuracy (%)")
        ax[1].axhline(50, color="k", lw=.8, ls=":")
        ax[2].set(xlabel="Latent dimension (float32 values)", ylabel="Decryption failure rate without hash check (%)", ylim=(-5, 105))
        for a in ax:
            a.set_xscale("log", base=2)
            a.legend()
            a.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_generalisation.png"), dpi=300)
        plt.close(fig)

    if ratio_table:
        lab = [f"z={r['latent_dim']}" for r in ratio_table]
        x = np.arange(len(lab))
        w = 0.27
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x - w, [r["naive_ratio_key_over_latent"] for r in ratio_table], w, label="key / latent (naive)")
        ax.bar(x, [r["ratio_with_hash_and_fallback"] for r in ratio_table], w, label="with hash + fallback")
        ax.bar(x + w, [r["ratio_including_decoder_one_key"] for r in ratio_table], w, label="including decoder")
        ax.axhline(1, color="k", lw=.8)
        ax.set_yscale("log")
        ax.set_xticks(x, lab)
        ax.set(ylabel="Compression ratio (bits / bits, log scale)", xlabel="Configuration")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_compression_ratios.png"), dpi=300)
        plt.close(fig)

    if baselines:
        names = list(baselines)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(names, [baselines[k]["mean"] for k in names], yerr=[baselines[k]["std"] for k in names], capsize=4)
        ax.axhline(1, color="k", lw=.8)
        ax.set(ylabel="Compression ratio (original bytes / compressed bytes)", title="Generic compressors on the public key")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_baseline_compressors.png"), dpi=300)
        plt.close(fig)

    if timing:
        keys = [k for k in ("keygen_ms", "encrypt_ms", "encrypt_with_vae_inference_ms", "decrypt_ms") if k in timing]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(keys, [timing[k]["mean"] for k in keys], yerr=[timing[k]["ci95"] for k in keys], capsize=4)
        ax.set_yscale("log")
        ax.set(ylabel="Time (ms, mean ± 95% CI, log scale)")
        plt.setp(ax.get_xticklabels(), rotation=15)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_timing.png"), dpi=300)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="toy")
    ap.add_argument("--form", choices=["sgp", "systematic"], default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--latents", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--hidden", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight; 0 = plain autoencoder (ablation)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--activation", default="relu", help="hidden-layer activation, e.g. relu or gelu (ablation)")
    ap.add_argument("--val-split", type=float, default=0.1, help="fraction of training keys used for validation")
    ap.add_argument("--tag", default="", help="suffix of the output folder, e.g. 'beta0' for an ablation run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="independent runs")
    ap.add_argument("--n-msgs", type=int, default=20, help="messages per key for the DFR")
    ap.add_argument("--perkey-keys", type=int, default=3)
    ap.add_argument("--perkey-repeats", type=int, default=256)
    ap.add_argument("--perkey-epochs", type=int, default=50)
    ap.add_argument("--timing-runs", type=int, default=50)
    ap.add_argument("--keygen-runs", type=int, default=10)
    ap.add_argument("--selftest", type=int, default=100)
    ap.add_argument("--bitflip-levels", type=int, nargs="*", default=[0, 1, 2, 4, 8, 16, 32, 64],
                    help="numbers of public-key bits to flip (empty = skip the test)")
    ap.add_argument("--bitflip-msgs", type=int, default=200, help="messages per bit-flip level")
    ap.add_argument("--skip-vae", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    P = dict(PRESETS[args.preset])
    form = args.form or P["form"]
    hidden = args.hidden or P["hidden"]
    n_train = P["n_train"] if args.n_train is None else args.n_train
    n_test = P["n_test"] if args.n_test is None else args.n_test
    out = os.path.join(args.out, f"{args.preset}_{form}" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out, exist_ok=True)
    master = np.random.SeedSequence(args.seeds[0])
    rng = np.random.default_rng(master.spawn(1)[0])
    results = {"script_version": VERSION,
               "config": {**vars(args), "form": form, "hidden": hidden, "n_train": n_train, "n_test": n_test,
                          "m": P["m"], "n": P["n"], "t": P["t"]},
               "environment": environment_info()}
    print(f"mceliece_vae_revised {VERSION}")

    # ---- key generation timing + correctness self-test -----------------------------
    print(f"[1] Key generation  m={P['m']} n={P['n']} t={P['t']} form={form}")
    kg = []
    for i in range(args.keygen_runs):
        t0 = time.perf_counter()
        mc = McEliece(P["m"], P["n"], P["t"], form, np.random.default_rng(10_000 + i))
        kg.append((time.perf_counter() - t0) * 1e3)
    ok = self_test(mc, rng, args.selftest)
    results["code"] = {"k": mc.k, "public_key_shape": list(mc.public.shape), "public_key_bits": mc.public_bits,
                       "public_key_bytes": mc.public_bits / 8, "public_key_KiB": mc.public_bits / 8 / 1024,
                       "selftest_success": f"{ok}/{args.selftest}"}
    print(f"    k={mc.k}, public key {mc.public.shape} = {mc.public_bits:,} bits = {mc.public_bits / 8:,.0f} bytes")
    print(f"    self-test decrypt(encrypt(m)) == m : {ok}/{args.selftest}")
    if ok != args.selftest:
        sys.exit("self-test failed: the McEliece implementation is not correct")

    timing = {"keygen_ms": mean_std_ci(kg)}
    results["timing"] = timing

    # ---- sensitivity of decryption to wrong bits in the public key (reviewer comment 2) ----
    if args.bitflip_levels:
        print(f"[1b] Bit-flip sensitivity: encrypt with j flipped public-key bits, {args.bitflip_msgs} messages per level")
        results["bitflip_sensitivity"] = bitflip_experiment(mc, rng, args.bitflip_levels, args.bitflip_msgs)
        for r in results["bitflip_sensitivity"]:
            print(f"    flipped={r['flipped_bits']:3d}  DFR={r['dfr']:.3f} ± {r['dfr_ci95']:.3f}  "
                  f"(detected {r['decoding_failure_detected']:.3f}, wrong plaintext {r['wrong_plaintext_undetected']:.3f})")

    # ---- public-key size in S*G*P form versus systematic form (reviewer comment 4) ----
    results["public_key_forms"] = public_key_form_sizes()
    print("[1c] Public-key size: S*G*P form (k x n) versus systematic form (k x (n-k))")
    for r in results["public_key_forms"]:
        print(f"    {r['parameter_set']:16s} k={r['k']:5d}  SGP {r['sgp_bits']:>11,} bits ({r['sgp_KiB']:9.2f} KiB)  "
              f"systematic {r['systematic_bits']:>10,} bits ({r['systematic_KiB']:8.2f} KiB)  ratio {r['systematic_ratio']:.2f}x")

    # ---- analytical decoder size for this key size ----------------------------------
    results["decoder_size_analytical"] = [
        {"latent_dim": l, "params": decoder_param_count(mc.public_bits, l, hidden),
         "MiB_float32": decoder_param_count(mc.public_bits, l, hidden) * 4 / 2**20,
         "public_key_KiB": mc.public_bits / 8 / 1024} for l in args.latents]

    gen_table, ratio_table, perkey_table, training_logs = [], [], [], []
    vae_for_timing = Z_for_timing = None
    if not args.skip_vae and n_train > 0:
        import tensorflow as tf
        print(f"[2] Generating {n_train} training and {n_test} test keys (disjoint seeds)")
        ss_train, ss_test = np.random.SeedSequence([args.seeds[0], 1]), np.random.SeedSequence([args.seeds[0], 2])
        pool_keys = generate_keys(n_train, P, form, ss_train)
        test_keys = generate_keys(n_test, P, form, ss_test)
        n_val = int(round(args.val_split * n_train))
        train_keys, val_keys = pool_keys[:n_train - n_val], pool_keys[n_train - n_val:]
        hashes = [{k.public_hash() for k in s} for s in (train_keys, val_keys, test_keys)]
        overlap = len(hashes[0] & hashes[2]) + len(hashes[1] & hashes[2]) + len(hashes[0] & hashes[1])
        results["data"] = {"n_train": len(train_keys), "n_val": len(val_keys), "n_test": n_test,
                           "overlapping_keys_between_splits": overlap,
                           "input_representation": "public-key matrix flattened row-major, bits as float32 {0,1}",
                           "latent_precision": "float32 (32 bits per latent dimension)"}
        print(f"    train={len(train_keys)}  validation={len(val_keys)}  test={n_test}  overlapping keys={overlap}")
        Xtr = np.stack([k.public.flatten() for k in train_keys]).astype(np.float32)
        Xval = np.stack([k.public.flatten() for k in val_keys]).astype(np.float32) if val_keys else None
        Xte = np.stack([k.public.flatten() for k in test_keys]).astype(np.float32)

        # baseline that needs no model: predict every bit as its majority value in the training keys
        majority = (Xtr.mean(0) > 0.5).astype(np.float32)
        maj_test, maj_train = float((Xte == majority).mean()), float((Xtr == majority).mean())
        n_test_bits = Xte.size
        z_maj, p_maj = z_test_vs(maj_test, n_test_bits)
        results["majority_bit_baseline"] = {"train_bit_accuracy": maj_train, "test_bit_accuracy": maj_test,
                                            "test_z_vs_chance": z_maj, "test_p_vs_chance": p_maj,
                                            "n_test_bits": n_test_bits}
        print(f"[2b] Majority-bit baseline (no model): train bit-acc={maj_train:.4f}  test bit-acc={maj_test:.4f}  "
              f"(p vs 0.5 = {p_maj:.3g})")

        print("[3] Generalisation regime: one VAE trained once, evaluated on unseen keys")
        for seed in args.seeds:
            for lat in args.latents:
                tf.keras.utils.set_random_seed(seed)
                vae = build_vae(Xtr.shape[1], lat, hidden, args.beta, args.lr, args.activation)
                hist = vae.fit(Xtr, epochs=args.epochs, batch_size=args.batch, verbose=0,
                               validation_data=(Xval, Xval) if Xval is not None else None)
                training_logs.append({"seed": seed, "latent_dim": lat,
                                      "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}})
                dec_bits = vae.decoder.count_params() * 32
                for split, keys, X in (("train", train_keys, Xtr), ("test", test_keys, Xte)):
                    Z = vae_compress(vae, X)
                    rows = evaluate_reconstruction(keys, vae_reconstruct(vae, Z), args.n_msgs, rng)
                    s = summarise_rows(rows)
                    acc, nbits = s["bit_accuracy"]["mean"], X.size
                    _, p_chance = z_test_vs(acc, nbits)
                    _, p_major = two_proportion_test(acc, maj_test if split == "test" else maj_train, nbits)
                    side = lat * 32 + 256
                    gen_table.append({"seed": seed, "latent_dim": lat, "split": split,
                                      "p_bit_acc_vs_chance": p_chance,
                                      "p_bit_acc_vs_majority_baseline": p_major,
                                      "residual_bits_enum_mean": s["residual_bits_enum"]["mean"],
                                      "residual_bits_lzma_mean": s["residual_bits_lzma"]["mean"],
                                      "rho_lossless_vae_plus_residual_enum": mc.public_bits / (side + s["residual_bits_enum"]["mean"]),
                                      "rho_lossless_vae_plus_residual_lzma": mc.public_bits / (side + s["residual_bits_lzma"]["mean"]),
                                      "exact_recovery_rate": s["exact_recovery_rate"],
                                      "bit_accuracy_mean": s["bit_accuracy"]["mean"],
                                      "hamming_mean": s["hamming_bits"]["mean"],
                                      "hamming_std": s["hamming_bits"]["std"],
                                      "dfr_without_check": s["dfr_without_check"],
                                      "fallback_rate": s["fallback_rate"],
                                      "final_train_loss": float(hist.history["loss"][-1]),
                                      "final_val_loss": float(hist.history["val_loss"][-1]) if "val_loss" in hist.history else float("nan")})
                    if split == "test" and seed == args.seeds[0]:
                        ratio_table.append({"latent_dim": lat, **compression_ratios(
                            mc.public_bits, lat, dec_bits, s["fallback_rate"])})
                te = gen_table[-1]
                print(f"    seed={seed} z={lat:4d}  test bit-acc={te['bit_accuracy_mean']:.4f} "
                      f"(p vs 0.5={te['p_bit_acc_vs_chance']:.3g}, p vs majority={te['p_bit_acc_vs_majority_baseline']:.3g})  "
                      f"lossless VAE+residual rho={te['rho_lossless_vae_plus_residual_enum']:.3f} (ideal) / "
                      f"{te['rho_lossless_vae_plus_residual_lzma']:.3f} (lzma)")
                print(f"    seed={seed} z={lat:4d}  train exact={gen_table[-2]['exact_recovery_rate']:.3f}  "
                      f"test exact={gen_table[-1]['exact_recovery_rate']:.3f}  "
                      f"test bit-acc={gen_table[-1]['bit_accuracy_mean']:.4f}  "
                      f"test DFR={gen_table[-1]['dfr_without_check']:.3f}")
                if seed == args.seeds[0] and lat == args.latents[0]:
                    vae_for_timing, Z_for_timing = vae, vae_compress(vae, Xte[:1])
                    Ztr, Zte = vae_compress(vae, Xtr), vae_compress(vae, Xte)
                    Ytr = np.stack([k.private_target_bits() for k in train_keys])
                    Yte = np.stack([k.private_target_bits() for k in test_keys])
                    results["leakage_probe"] = logistic_probe(Ztr, Ytr, Zte, Yte)

        print("[4] Per-key regime ('repeats'): one VAE trained per key, decoder must be shipped")
        for i, mc_k in enumerate(test_keys[:args.perkey_keys]):
            for lat in args.latents:
                tf.keras.utils.set_random_seed(args.seeds[0] + i)
                vae = build_vae(mc_k.public_bits, lat, hidden, args.beta, args.lr, args.activation)
                X1 = np.repeat(mc_k.public.flatten()[None].astype(np.float32), args.perkey_repeats, axis=0)
                vae.fit(X1, epochs=args.perkey_epochs, batch_size=args.batch, verbose=0)
                r = evaluate_reconstruction([mc_k], vae_reconstruct(vae, vae_compress(vae, X1[:1])), args.n_msgs, rng)[0]
                dec_bits = vae.decoder.count_params() * 32
                perkey_table.append({"key": i, "latent_dim": lat, "repeats": args.perkey_repeats,
                                     "epochs": args.perkey_epochs, "exact": r["exact"], "hamming": r["hamming"],
                                     "dfr_without_check": r["dfr_without_check"],
                                     "decoder_params": dec_bits // 32,
                                     "honest_ratio_key_over_latent_plus_decoder":
                                         mc_k.public_bits / (lat * 32 + 256 + dec_bits)})
            print(f"    key {i}: exact={[r['exact'] for r in perkey_table if r['key'] == i]}  "
                  f"honest ratio={perkey_table[-1]['honest_ratio_key_over_latent_plus_decoder']:.2e}")

    # ---- baselines and timing ------------------------------------------------------
    print("[5] Generic compressors on the packed public key")
    base_keys = [mc] + [McEliece(P["m"], P["n"], P["t"], form, np.random.default_rng(20_000 + i))
                        for i in range(4 if P["n"] > 1024 else 19)]
    results["baseline_compressors"] = baseline_compressors(base_keys)
    for k, v in results["baseline_compressors"].items():
        print(f"    {k:8s} ratio = {v['mean']:.4f} ± {v['std']:.4f}")

    print(f"[6] Timing over {args.timing_runs} runs")
    if vae_for_timing is not None:
        mc_t = test_keys[0]
        vae_for_timing(np.zeros((1, mc_t.public_bits), dtype=np.float32))   # warm-up
    else:
        mc_t = mc
    timing.update(timing_experiment(mc_t, rng, args.timing_runs, vae_for_timing, Z_for_timing))
    for k, v in timing.items():
        if isinstance(v, dict):
            print(f"    {k:32s} {v['mean']:10.3f} ± {v['std']:.3f} ms  (95% CI ± {v['ci95']:.3f}, n={v['n']})")

    # ---- save everything -----------------------------------------------------------
    gen_summary = []
    for lat in sorted({r["latent_dim"] for r in gen_table}):
        for split in ("train", "test"):
            sel = [r for r in gen_table if r["latent_dim"] == lat and r["split"] == split]
            row = {"latent_dim": lat, "split": split, "n_runs": len(sel)}
            for key in ("exact_recovery_rate", "bit_accuracy_mean", "hamming_mean", "dfr_without_check", "fallback_rate",
                        "rho_lossless_vae_plus_residual_enum", "rho_lossless_vae_plus_residual_lzma"):
                st = mean_std_ci([r[key] for r in sel])
                row[f"{key}_mean"], row[f"{key}_sd"] = st["mean"], st["std"]
            gen_summary.append(row)
    results["generalisation"] = gen_table
    results["generalisation_summary_over_runs"] = gen_summary
    results["training_logs"] = training_logs
    results["compression_ratios"] = ratio_table
    results["per_key"] = perkey_table
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    write_csv(os.path.join(out, "table_generalisation.csv"), gen_table)
    write_csv(os.path.join(out, "table_generalisation_summary.csv"), gen_summary)
    write_csv(os.path.join(out, "table_compression_ratios.csv"), ratio_table)
    write_csv(os.path.join(out, "table_per_key.csv"), perkey_table)
    write_csv(os.path.join(out, "table_decoder_size_analytical.csv"), results["decoder_size_analytical"])
    write_csv(os.path.join(out, "table_bitflip_sensitivity.csv"), results.get("bitflip_sensitivity", []))
    write_csv(os.path.join(out, "table_public_key_forms.csv"), results["public_key_forms"])
    try:
        make_figures(out, gen_table, ratio_table, results["baseline_compressors"], timing)
    except ImportError:
        print("    matplotlib not installed: figures skipped")
    if "leakage_probe" in results:
        lp = results["leakage_probe"]
        print(f"[7] Leakage probe: bit accuracy {lp['bit_accuracy']:.4f} ± {lp['bit_accuracy_ci95']:.4f} "
              f"(majority baseline {lp['majority_baseline_accuracy']:.4f}), "
              f"exact private-key recovery {lp['exact_private_key_recovery_rate']:.3f}")
    print(f"\nAll results written to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
