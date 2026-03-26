"""
SPARC Clean Test — Hubble Type vs Dark Matter Fraction
======================================================
The composite coherence index was broken (surface brightness confounded).
Hubble type alone carried the signal in the first regression.

This is the clean version: one variable, one prediction, no noise.

Prediction: Lower T (more organized) = less DM dominated
T=0 (Sa) should have LESS dark matter than T=10 (Irr)

Author: Annie Robinson
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

# ============================================================
# PARSE
# ============================================================

def parse_sparc(filepath):
    galaxies = []
    lines = open(filepath, 'r').readlines()
    last_dash = 0
    for i, line in enumerate(lines):
        if line.startswith('---'):
            last_dash = i

    for line in lines[last_dash + 1:]:
        if len(line.strip()) < 50:
            continue
        try:
            parts = line.split()
            if len(parts) < 18:
                continue
            g = {
                'name': parts[0],
                'T': int(parts[1]),
                'D': float(parts[2]),
                'Inc': float(parts[5]),
                'L36': float(parts[7]),
                'SBdisk': float(parts[12]),
                'MHI': float(parts[13]),
                'Vflat': float(parts[15]),
                'e_Vflat': float(parts[16]),
                'Q': int(parts[17]),
            }
            if g['Vflat'] > 0 and g['L36'] > 0:
                galaxies.append(g)
        except (ValueError, IndexError):
            continue
    return galaxies

# ============================================================
# DARK MATTER FRACTION
# ============================================================

def compute_dm_fraction(galaxies):
    """
    Baryonic Tully-Fisher residual as DM proxy.
    BTF: Vflat^4 ~ M_baryon
    Residual = log(Vflat) - 0.25 * log(M_baryon)
    More positive residual = more DM dominated (higher V for given mass)
    """
    ML = 0.5  # M/L at 3.6 micron

    for g in galaxies:
        M_star = g['L36'] * ML  # 10^9 Msun
        M_gas = g['MHI'] * 1.33 if g['MHI'] > 0 else 0
        M_bar = (M_star + M_gas) * 1e9  # Msun

        if M_bar > 0 and g['Vflat'] > 0:
            g['log_Vflat'] = np.log10(g['Vflat'])
            g['log_Mbar'] = np.log10(M_bar)
            g['btf_resid'] = g['log_Vflat'] - 0.25 * g['log_Mbar']
            g['valid'] = True
        else:
            g['valid'] = False

    return [g for g in galaxies if g.get('valid', False)]

# ============================================================
# RUN
# ============================================================

print("SPARC Clean Test: Hubble Type vs Dark Matter")
print("One variable. One prediction. No noise.")
print("=" * 55)

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SPARC_Lelli2016c.mrt')
galaxies = parse_sparc(filepath)
galaxies = compute_dm_fraction(galaxies)
print(f"Galaxies with valid data: {len(galaxies)}")

T = np.array([g['T'] for g in galaxies])
btf = np.array([g['btf_resid'] for g in galaxies])
Q = np.array([g['Q'] for g in galaxies])
log_Mbar = np.array([g['log_Mbar'] for g in galaxies])

# ============================================================
# TEST 1: Simple linear regression T vs BTF residual
# ============================================================

print("\nTEST 1: Hubble Type vs BTF Residual (all galaxies)")
print("-" * 45)

slope, intercept, r_val, p_val, std_err = stats.linregress(T, btf)
print(f"  slope = {slope:.5f} +/- {std_err:.5f}")
print(f"  R = {r_val:.4f}")
print(f"  p = {p_val:.6f}")
print(f"  n = {len(galaxies)}")

if slope < 0 and p_val < 0.05:
    print("  >>> NEGATIVE: Higher T (chaotic) = MORE negative BTF residual <<<")
    print("  >>> This means chaotic galaxies are LESS DM dominated <<<")
    print("  >>> OPPOSITE of prediction <<<")
elif slope > 0 and p_val < 0.05:
    print("  >>> POSITIVE: Higher T = MORE positive BTF residual <<<")
    print("  >>> Chaotic galaxies MORE DM dominated <<<")
    print("  >>> Wait — check the sign convention <<<")

# ============================================================
# TEST 2: Quality=1 only (best rotation curves)
# ============================================================

print("\nTEST 2: Quality=1 galaxies only (best data)")
print("-" * 45)

q1_mask = Q == 1
if np.sum(q1_mask) > 10:
    sl2, int2, r2, p2, se2 = stats.linregress(T[q1_mask], btf[q1_mask])
    print(f"  slope = {sl2:.5f} +/- {se2:.5f}")
    print(f"  R = {r2:.4f}")
    print(f"  p = {p2:.6f}")
    print(f"  n = {np.sum(q1_mask)}")

# ============================================================
# TEST 3: Controlling for baryonic mass
# ============================================================

print("\nTEST 3: Partial correlation (controlling for log M_baryon)")
print("-" * 45)

# Regress T and btf on log_Mbar, correlate residuals
sl_t, int_t, _, _, _ = stats.linregress(log_Mbar, T.astype(float))
T_resid = T - (sl_t * log_Mbar + int_t)

sl_b, int_b, _, _, _ = stats.linregress(log_Mbar, btf)
btf_resid = btf - (sl_b * log_Mbar + int_b)

sl3, int3, r3, p3, se3 = stats.linregress(T_resid, btf_resid)
print(f"  partial slope = {sl3:.5f} +/- {se3:.5f}")
print(f"  partial R = {r3:.4f}")
print(f"  p = {p3:.6f}")

if sl3 < 0 and p3 < 0.05:
    print("  >>> Signal survives mass control. NEGATIVE partial slope. <<<")
elif sl3 > 0 and p3 < 0.05:
    print("  >>> Signal survives mass control. POSITIVE partial slope. <<<")
else:
    print("  >>> Not significant after mass control. <<<")

# ============================================================
# TEST 4: Binned means by Hubble type
# ============================================================

print("\nTEST 4: Mean BTF residual by Hubble type")
print("-" * 45)
print(f"  {'T':>3s} {'n':>5s} {'mean BTF':>10s} {'sem':>8s} {'interpretation':>20s}")

T_vals = sorted(set(T))
means = []
sems = []
ns = []
t_plot = []

for t_val in T_vals:
    mask = T == t_val
    n = np.sum(mask)
    if n >= 3:
        m = np.mean(btf[mask])
        s = np.std(btf[mask]) / np.sqrt(n)
        means.append(m)
        sems.append(s)
        ns.append(n)
        t_plot.append(t_val)
        interp = "less DM" if m < np.median(btf) else "more DM"
        print(f"  {t_val:3d} {n:5d} {m:10.5f} {s:8.5f} {interp:>20s}")

# Spearman rank correlation (more robust than Pearson for ordinal data)
rho_spear, p_spear = stats.spearmanr(T, btf)
print(f"\n  Spearman rho = {rho_spear:.4f}, p = {p_spear:.6f}")
print(f"  (Spearman is better for ordinal Hubble types)")

# ============================================================
# TEST 5: Early (T<=4) vs Late (T>=8) types
# ============================================================

print("\nTEST 5: Early-type (T<=4) vs Late-type (T>=8)")
print("-" * 45)

early = btf[T <= 4]
late = btf[T >= 8]

if len(early) >= 5 and len(late) >= 5:
    t_stat, t_pval = stats.ttest_ind(early, late)
    mw_stat, mw_pval = stats.mannwhitneyu(early, late, alternative='two-sided')
    print(f"  Early (T<=4): n={len(early)}, mean={np.mean(early):.5f}")
    print(f"  Late  (T>=8): n={len(late)}, mean={np.mean(late):.5f}")
    print(f"  Difference: {np.mean(early) - np.mean(late):.5f}")
    print(f"  t-test: t={t_stat:.3f}, p={t_pval:.6f}")
    print(f"  Mann-Whitney: U={mw_stat:.0f}, p={mw_pval:.6f}")
    print(f"  Effect size (Cohen's d): {(np.mean(early)-np.mean(late))/np.sqrt((np.var(early)+np.var(late))/2):.3f}")

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('SPARC: Hubble Type vs Dark Matter Fraction\n'
             'C-M-D Prediction: Organized galaxies (low T) have less dark matter',
             fontsize=13, fontweight='bold')

# Scatter
ax = axes[0, 0]
ax.scatter(T, btf, c=Q, cmap='RdYlGn_r', s=30, alpha=0.7, edgecolors='gray', linewidth=0.5)
x_fit = np.linspace(0, 11, 100)
ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=2,
        label=f'R={r_val:.3f}, p={p_val:.4f}')
ax.set_xlabel('Hubble Type (0=S0/Sa → 10=Irr)')
ax.set_ylabel('BTF Residual (+ = more DM dominated)')
ax.set_title('All Galaxies')
ax.legend(); ax.grid(alpha=0.3)

# Binned means
ax = axes[0, 1]
ax.errorbar(t_plot, means, yerr=sems, fmt='o-', color='#1e5d5c', lw=2, capsize=5, markersize=8)
ax.axhline(np.median(btf), color='gray', ls='--', alpha=0.4)
ax.set_xlabel('Hubble Type')
ax.set_ylabel('Mean BTF Residual')
ax.set_title(f'Binned Means (Spearman rho={rho_spear:.3f}, p={p_spear:.4f})')
ax.grid(alpha=0.3)

# Partial correlation
ax = axes[1, 0]
ax.scatter(T_resid, btf_resid, c=T, cmap='RdYlGn_r', s=30, alpha=0.7, edgecolors='gray', linewidth=0.5)
x_fit2 = np.linspace(T_resid.min(), T_resid.max(), 100)
ax.plot(x_fit2, sl3 * x_fit2 + int3, 'r-', lw=2,
        label=f'partial R={r3:.3f}, p={p3:.4f}')
ax.set_xlabel('T residual (mass-controlled)')
ax.set_ylabel('BTF residual (mass-controlled)')
ax.set_title('After Controlling for Baryonic Mass')
ax.legend(); ax.grid(alpha=0.3)

# Early vs Late box plot
ax = axes[1, 1]
bp = ax.boxplot([early, late], positions=[1, 2], widths=0.6, patch_artist=True)
bp['boxes'][0].set_facecolor('#99ff99')
bp['boxes'][1].set_facecolor('#ff9999')
ax.set_xticks([1, 2])
ax.set_xticklabels(['Early (T<=4)\nOrganized', 'Late (T>=8)\nChaotic'])
ax.set_ylabel('BTF Residual')
if len(early) >= 5 and len(late) >= 5:
    ax.set_title(f'Early vs Late: p={t_pval:.4f}')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparc_hubble_clean_results.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")

# ============================================================
# VERDICT
# ============================================================

print()
print("=" * 55)
print("VERDICT")
print("=" * 55)

# Check which direction the data goes
if len(means) > 3:
    early_mean = np.mean([m for t, m in zip(t_plot, means) if t <= 4])
    late_mean = np.mean([m for t, m in zip(t_plot, means) if t >= 8])

    if early_mean > late_mean:
        print("Early types (organized) have HIGHER BTF residual = MORE DM dominated")
        print("This is OPPOSITE to prediction.")
        print("HOWEVER: check if BTF residual sign convention is inverted.")
        print(f"  Early mean: {early_mean:.5f}")
        print(f"  Late mean: {late_mean:.5f}")
    else:
        print("Early types (organized) have LOWER BTF residual = LESS DM dominated")
        print("This MATCHES the C-M-D prediction.")
        print(f"  Early mean: {early_mean:.5f}")
        print(f"  Late mean: {late_mean:.5f}")
