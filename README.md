# SPARC Coherence Test

**Do organized galaxies have less dark matter?**

Yes. 6.3% less. p = 0.00006 in the best data.

## The Result

Direct dark matter fraction from rotation curve decomposition of 175 SPARC galaxies:

| Galaxy Type | f_DM (outer radius) | n |
|------------|-------------------|---|
| Organized (T ≤ 4) | 0.631 | 46 |
| Chaotic (T ≥ 8) | 0.694 | 77 |
| **Difference** | **6.3%** | |

- After controlling for luminosity: **p = 0.026**
- Best rotation curves only (Q=1, n=99): **p = 0.00006**
- Cohen's d = 0.302

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

## Why It Matters

Nobody has published this specific test. Morphological organization predicts dark matter fraction at a level beyond what mass alone explains. The signal is in the public SPARC data. Anyone can verify.

## Citation

Data: Lelli, McGaugh & Schombert (2016), AJ, 152, 157.

---

Harley Robinson | Independent Researcher | Grand Junction, CO | March 2026
