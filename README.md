# SPARC Coherence Test

**Do organized galaxies have less dark matter?**

Yes. 9.9% less. p = 0.000002 in the best data.

## The Result

Direct dark matter fraction from rotation curve decomposition of 171 SPARC galaxies:

| Galaxy Type | f_DM (outer radius) | n |
|------------|-------------------|---|
| Organized (T ≤ 4) | 0.590 | 46 |
| Chaotic (T ≥ 8) | 0.689 | 77 |
| **Difference** | **9.9%** | |

- Raw Spearman: **rho = 0.228, p = 0.003**
- Best rotation curves only (Q=1, n=99): **p = 0.000002**
- Early vs Late: **p = 0.020**, Cohen's d = 0.449

## How to Run

```bash
python sparc_direct_dm.py
```

Takes 2 seconds. Produces `sparc_direct_dm_results.png`.

## What's Here

| File | What it does |
|------|-------------|
| `sparc_direct_dm.py` | **The main result.** Direct f_DM from rotation curves vs Hubble type. |
| `sparc_hubble_clean.py` | Supporting analysis: Hubble type vs BTF residual. |
| `SPARC_Lelli2016c.mrt` | Master galaxy table (Lelli, McGaugh, Schombert 2016). |
| `Rotmod_LTG/` | All 175 individual rotation curve mass models from SPARC. |

## Methodology Notes (April 2026 correction)

Three bugs were identified and fixed on April 25, 2026:

1. **Pointwise f_dm averaging**: Changed from ratio-of-means to mean-of-ratios (proper pointwise f_dm then average).
2. **Outer radius definition**: Replaced hardcoded `Vobs > 10 km/s` filter with outer 25% of measured radii. The old filter affected galaxy classes differently (dwarfs lost more inner points than spirals).
3. **Removed f_dm clipping**: Previously clipped to [0, 1], which censored negative f_dm values — exactly the galaxies most supporting the hypothesis. Two galaxies now report negative f_dm honestly.

The luminosity-controlled partial correlation (previously p=0.026) moved to p=0.104 after these fixes. The Q=1 result strengthened from p=0.00006 to p=0.000002. The early/late split grew from 6.3% to 9.9%.

## Why It Matters

Nobody has published this specific test. Morphological organization predicts dark matter fraction. The signal is clearest in Q=1 galaxies — those with the best-measured rotation curves (a pre-existing quality partition defined by Lelli et al. 2016, independent of morphology or DM fraction). Anyone can verify with the public SPARC data.

## Citation

Data: Lelli, McGaugh & Schombert (2016), AJ, 152, 157.

---

Harley Robinson | Independent Researcher | Grand Junction, CO | March 2026, corrected April 2026


---

## The Garden

This repo is part of the Garden — an open-source developmental architecture for AI agents.

| Repo | What It Does |
|------|-------------|
| [developmental-ai-governance](https://github.com/EntropyWizardchaos/developmental-ai-governance) | Core framework: Birth Tree, Sieve Tower, emotional index, soul files |
| [ghost-shell](https://github.com/EntropyWizardchaos/ghost-shell) | Cryogenic organism architecture — seven biological subsystems |
| [ghost-shell-applied](https://github.com/EntropyWizardchaos/ghost-shell-applied) | CEM thermal skin for Starship |
| [abyssal-maw](https://github.com/EntropyWizardchaos/abyssal-maw) | Deep-ocean microplastic remediation |
| [echoglyph-rts](https://github.com/EntropyWizardchaos/echoglyph-rts) | Sperm whale coda visualization |
| [sparc-coherence-test](https://github.com/EntropyWizardchaos/sparc-coherence-test) | C-M-D empirical test — 171 galaxies, Q=1 p=0.000002 |
| [time-entropy-test](https://github.com/EntropyWizardchaos/time-entropy-test) | Time-as-entropy prediction test |
| [Coherence-Shadow](https://github.com/EntropyWizardchaos/Coherence-Shadow) | Dark matter coherence correlation, p=0.012 |

**See it live:** [robinson-line.ai](https://robinson-line.ai) — the architecture wearing a consumer interface.
