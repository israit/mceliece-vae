# Code and data: Variational Autoencoders for McEliece key generation and public-key compression

This repository contains the code, raw results and run logs for the experiments in the paper
*Post-Quantum Cryptography Enhancement: Applying Variational AutoEncoders (VAE) to McEliece Key Generation
and Public-Key Compression for Secure Communication*. Every number reported in the paper is read from the
files in `results/` or `logs/`, and every file in `results/` can be regenerated with the commands below.

The experiments study a Variational Autoencoder (VAE) in three roles:

1. **Key generation**: generating the secret key components of McEliece (the scrambling matrix S, the
   permutation P and the Goppa polynomial g), with a time profile of each key-generation step.
2. **Public-key compression**: encoding the public key into a latent vector before transmission, evaluated on
   training keys and on disjoint, unseen test keys, with exact recovery, decryption failure rate (DFR) and an
   effective compression ratio that counts the latent vector, a SHA-256 verification hash, the fallback
   traffic and the decoder.
3. **Key-generator testing**: using the VAE as a randomness test on public keys from correct and deliberately
   weakened generators, combined with duplicate, majority-bit and GF(2)-rank tests, and checking whether a
   VAE recognizes the structure of Goppa codes.

McEliece is implemented from scratch over binary Goppa codes (GF(2^m) arithmetic, irreducible Goppa
polynomial, algebraic decoding of t errors with the extended Euclidean (Sugiyama) algorithm on g²). All GF(2) linear algebra is done modulo 2.
Every run starts with an encryption/decryption self-test.

## Repository layout

```
code/
  mceliece_vae_revised_v4.py     McEliece + VAE public-key compression (main experiment)
  vae_keygen_SP_v1.py            key-generation profile; VAE generation of S and P
  vae_goppa_detect_v1.py         VAE-guided Goppa polynomial generation; VAE as a weak-generator detector
  vae_distinguish_health_v1.py   Goppa vs random codes; false-alarm rate of the key-generator health check
  make_figures_v3_1.py           Figures A-F (from results/compression_v2_v3)
  make_figures_v4.py             helper module imported by make_figures_v5.py
  make_figures_v5.py             Figures G, H, J and K
  legacy/                        earlier versions that produced results/compression_v2_v3
    mceliece_vae_revised_v2.py   v2: generalisation, ablations (beta = 0, GELU), mceliece348864
    mceliece_vae_revised_v3.py   v3: v2 + bit-flip sensitivity test
    make_figures_v3.py           superseded by make_figures_v3_1.py
results/
  compression_v2_v3/             runs of the v2 and v3 scripts (Google Colab)
  compression_v4/                runs of the v4 script (local machine)
  keygen_SP_v1/
  goppa_detect_v1/
  goppa_detect_v1_strong/
  distinguish_health_v1/
logs/                            console output of every run (*_raw.txt) and short summaries (*_summary.txt)
requirements.txt
LICENSE                          MIT license for the code
```

Each result folder holds a `results.json` file with the full configuration (all command-line arguments,
including seeds), the software environment and every measured value, plus CSV tables and quick-look PNG plots.

## Requirements

- Python 3.12 or 3.13 (tested with 3.12.10 and 3.13.16)
- NumPy 2.x (tested with 2.1.3 and 2.5.4)
- TensorFlow 2.21.0 (needed only for the VAE parts; runs with `--skip-vae` do not import it)
- Matplotlib 3.x (only for the figure scripts)

```
pip install -r requirements.txt
```

All experiments run on a CPU; no GPU is needed.

## Quick check (about one minute, no TensorFlow)

```
cd code
python mceliece_vae_revised_v4.py --preset toy --skip-vae --seeds 0 --perkey-keys 0 \
       --timing-runs 3 --keygen-runs 2 --selftest 20 --bitflip-levels 0 1 --bitflip-msgs 20 --out ../check
```

This builds toy McEliece keys, runs the self-test, and prints the public-key sizes in the S·G·P and systematic
forms (ratios 2.67×, 2.05×, 4.54× and 3.69× for the four parameter sets) and the gzip, bzip2 and LZMA ratios
on the packed key (0.93, 0.66 and 0.84 for the toy key).

## Parameter sets

| Preset | (m, n, t) | k | Public key |
|---|---|---|---|
| `toy` | (6, 64, 4) | 40 | S·G·P: 40 × 64 = 2,560 bits |
| `small` | (8, 256, 8) | 192 | S·G·P: 192 × 256 bits |
| `mceliece348864` | (12, 3488, 64) | 2720 | systematic T: 2720 × 768 = 2,088,960 bits |

## Reproducing the results

Run all commands from the `code/` folder. Each command writes to the folder named in the last column.

### Public-key compression (v4, local machine)

| Run | Command | Output |
|---|---|---|
| Toy, S·G·P form | `python mceliece_vae_revised_v4.py --preset toy --seeds 0 --perkey-keys 0 --bitflip-levels --tag v4 --out ../results/compression_v4` | `compression_v4/toy_sgp_v4` |
| Toy, systematic form | same, plus `--form systematic` | `compression_v4/toy_systematic_v4` |
| Larger key (n = 256) | `python mceliece_vae_revised_v4.py --preset small --seeds 0 --perkey-keys 1 --bitflip-levels --tag v4 --out ../results/compression_v4` | `compression_v4/small_sgp_v4` |
| Deeper model | `python mceliece_vae_revised_v4.py --preset toy --hidden 1024 512 256 128 --seeds 0 --perkey-keys 0 --bitflip-levels --tag deep_v4 --out ../results/compression_v4` | `compression_v4/toy_sgp_deep_v4` |
| More training keys | `python mceliece_vae_revised_v4.py --preset toy --n-train 5000 --seeds 0 --perkey-keys 0 --bitflip-levels --tag n5000_v4 --out ../results/compression_v4` | `compression_v4/toy_sgp_n5000_v4` |

Each v4 run reports: public-key sizes in both forms, the majority-bit baseline, VAE bit accuracy and exact
recovery on training and unseen test keys with p-values against the baseline, the lossless VAE + residual
scheme, decoder size, effective compression ratios, generic compressors, the leakage probe and timing.

### Public-key compression (v2 and v3, Google Colab)

| Run | Command | Output |
|---|---|---|
| Toy, 3 seeds | `python legacy/mceliece_vae_revised_v2.py --preset toy --out ../results/compression_v2_v3` | `compression_v2_v3/toy_sgp` |
| Ablation, no KL term | same, plus `--beta 0 --tag beta0` | `compression_v2_v3/toy_sgp_beta0` |
| Ablation, GELU | same, plus `--activation gelu --tag gelu` | `compression_v2_v3/toy_sgp_gelu` |
| mceliece348864 (sizes, compressors, timing) | `python legacy/mceliece_vae_revised_v2.py --preset mceliece348864 --skip-vae --keygen-runs 3 --out ../results/compression_v2_v3` | `compression_v2_v3/mceliece348864_systematic` |
| Bit-flip sensitivity, toy | `python legacy/mceliece_vae_revised_v3.py --preset toy --skip-vae --tag bitflip --out ../results/compression_v2_v3` | `compression_v2_v3/toy_sgp_bitflip` |
| Bit-flip sensitivity, mceliece348864 | `python legacy/mceliece_vae_revised_v3.py --preset mceliece348864 --skip-vae --keygen-runs 3 --tag bitflip --out ../results/compression_v2_v3` | `compression_v2_v3/mceliece348864_systematic_bitflip` |

The per-key (memorization) regime is in `table_per_key.csv` of `toy_sgp` and of `small_sgp_v4`.

### Key generation, Goppa polynomials and key-generator testing (local machine)

| Run | Command | Output | Time |
|---|---|---|---|
| Key-generation profile; VAE generation of S and P | `python vae_keygen_SP_v1.py --out ../results/keygen_SP_v1` | `keygen_SP_v1` | |
| VAE-guided Goppa polynomials; weak-generator detection | `python vae_goppa_detect_v1.py --out ../results/goppa_detect_v1` | `goppa_detect_v1` | about 63 min |
| Stronger detector | `python vae_goppa_detect_v1.py --skip-goppa --det-beta 0.1 --det-epochs 200 --det-families correct seed_bits shared_code_biased_S shared_code_uniform_S shared_code_vae_S random_code_vae_S --seed-bits 8 10 --bias-levels 0.02 0.1 --out ../results/goppa_detect_v1_strong` | `goppa_detect_v1_strong` | about 25 min |
| Goppa vs random codes; health-check false alarms | `python vae_distinguish_health_v1.py --out ../results/distinguish_health_v1` | `distinguish_health_v1` | about 15 min |

Times are for a 12-thread Intel CPU; the Goppa and detection scripts run in parallel (`--workers`, default:
number of CPUs minus one). Add `--quick` to `vae_goppa_detect_v1.py` or `vae_distinguish_health_v1.py`
for a short smoke test.

### Figures

Figures are drawn only from the saved results; no experiment is re-run.

```
python make_figures_v3_1.py --results ../results/compression_v2_v3 --out ../figures
python make_figures_v5.py --keygen ../results/keygen_SP_v1 --goppa ../results/goppa_detect_v1 \
                          --dist ../results/distinguish_health_v1 --out ../figures
```

| File | Content |
|---|---|
| `Fig_A_generalisation` | exact recovery, bit accuracy and decryption failure rate on training vs unseen test keys, by latent size |
| `Fig_B_compression_ratios` | effective compression ratios |
| `Fig_C_ablation_bit_accuracy` | VAE vs no KL term (β = 0) vs GELU |
| `Fig_D_timing` | key generation, encryption (with and without VAE) and decryption times |
| `Fig_E_generic_compressors` | gzip, bzip2 and LZMA on the packed public key |
| `Fig_F_bitflip_sensitivity` | decryption failures caused by wrong key bits |
| `Fig_G_conceptual` | conceptual diagram of the three uses of the VAE |
| `Fig_H_keygen` | key-generation cost profile; generation of S, P and g |
| `Fig_J_detection` | VAE reconstruction of keys from correct and weakened generators |
| `Fig_K_goppa_vs_random` | Goppa codes vs random codes (ROC AUC) |

Each figure is written as PNG (600 dpi), TIFF (600 dpi, LZW) and PDF.

## Notes on reproducibility

- **Seeds.** All random choices use NumPy generators and `tf.keras.utils.set_random_seed`, seeded from the
  `--seed`/`--seeds` arguments recorded in each `results.json`. `TF_DETERMINISTIC_OPS=1` is set.
- **Colab vs local.** The v4 script was run on Google Colab (log: `logs/v4_colab.txt`; its result files are
  not included) and again locally (`results/compression_v4`). All deterministic values (key sizes, majority-bit
  baseline, ratios of the lossless scheme, compressor ratios, exact recovery, DFR, leakage probe) were
  identical; VAE bit accuracies differed by at most 0.06 percentage points (floating-point differences).
- **Timing.** McEliece is implemented in Python/NumPy without optimization, so absolute times are only
  meaningful relative to each other on the same machine; they vary between machines by a factor of about
  1.5–3.
- **Randomness baseline.** Where the output says "CSPRNG", the baseline is NumPy's PCG64 generator, which is
  uniform but not cryptographically secure.
- **Paths.** Local absolute paths written by the scripts into `results.json` and the logs were replaced by
  the relative paths of this repository. No values were changed.

## License

- **Code** (`code/`): MIT License, see [`LICENSE`](LICENSE).
- **Results and logs** (`results/`, `logs/`): [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You may use, modify and share the code and data, provided that you credit the authors and cite the paper.

## Citation

If you use this code or data, please cite the paper (citation details will be added after publication).
