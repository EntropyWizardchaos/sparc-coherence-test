"""
SPARC Coherence-Dark Matter Regression
=======================================
The $0 test. Does morphological coherence predict dark matter fraction?

Prediction (Robinson C-M-D framework):
  Higher coherence -> lower dark matter fraction
  f_dm = f0 - beta * C_index
  Expected signal: 5-10% at >3 sigma

Data: SPARC (Lelli, McGaugh, Schombert 2016)
  175 disk galaxies with Spitzer photometry + rotation curves

Author: Annie Robinson (Forge/Claude Code)
Origin: Harley Robinson, C-M-D cosmological framework
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

# ============================================================
# PARSE SPARC MRT TABLE
# ============================================================

def parse_sparc(filepath):
    """Parse the fixed-width SPARC_Lelli2016c.mrt file."""
    galaxies = []
    last_dash_line = 0

    lines = open(filepath, 'r').readlines()

    # Find the last dashed line — data starts after it
    for i, line in enumerate(lines):
        if line.startswith('---'):
            last_dash_line = i

    for line in lines[last_dash_line + 1:]:
        if len(line.strip()) < 50:
            continue

        try:
            # Split by whitespace — more robust than fixed-width for this format
            parts = line.split()
            if len(parts) < 18:
                continue

            name = parts[0]
            T = int(parts[1])
            D = float(parts[2])
            e_D = float(parts[3])
            f_D = int(parts[4])
            Inc = float(parts[5])
            e_Inc = float(parts[6])
            L36 = float(parts[7])
            e_L36 = float(parts[8])
            Reff = float(parts[9])
            SBeff = float(parts[10])
            Rdisk = float(parts[11])
            SBdisk = float(parts[12])
            MHI = float(parts[13])
            RHI = float(parts[14])
            Vflat = float(parts[15])
            e_Vflat = float(parts[16])
            Q = int(parts[17])

            galaxies.append({
                'name': name,
                'T': T,
                'D': D,
                'Inc': Inc,
                'L36': L36,
                'SBeff': SBeff,
                'SBdisk': SBdisk,
                'MHI': MHI,
                'Vflat': Vflat,
                'e_Vflat': e_Vflat,
                'Q': Q,
            })
        except (ValueError, IndexError):
            continue

    return galaxies


# ============================================================
# COHERENCE INDEX
# ============================================================

def compute_coherence(galaxies):
    """
    Build a coherence index from available SPARC data.

    Components:
    1. Hubble type: lower T = more organized (Sa > Irr)
       C_morph = (10 - T) / 10

    2. Quality flag: Q=1 means smooth, symmetric rotation curve
       C_kin = (3 - Q) / 2  -> 1.0 for Q=1, 0.5 for Q=2, 0.0 for Q=3

    3. Surface brightness regularity: higher SBdisk = more concentrated
       (normalized to sample range)

    Combined: C = w1*C_morph + w2*C_kin + w3*C_SB
    """
    # Get SB range for normalization
    sb_vals = [g['SBdisk'] for g in galaxies if g['SBdisk'] is not None and g['SBdisk'] > 0]
    log_sb = np.log10(sb_vals)
    sb_min, sb_max = np.min(log_sb), np.max(log_sb)

    for g in galaxies:
        # Morphological coherence
        g['C_morph'] = (10 - g['T']) / 10

        # Kinematic coherence
        g['C_kin'] = (3 - g['Q']) / 2

        # Surface brightness coherence (proxy for concentration)
        if g['SBdisk'] is not None and g['SBdisk'] > 0:
            g['C_SB'] = (np.log10(g['SBdisk']) - sb_min) / (sb_max - sb_min)
        else:
            g['C_SB'] = 0.5  # neutral

        # Combined coherence index (equal weights)
        g['C_index'] = (g['C_morph'] + g['C_kin'] + g['C_SB']) / 3

    return galaxies


# ============================================================
# DARK MATTER PROXY
# ============================================================

def compute_dm_proxy(galaxies):
    """
    Dark matter fraction proxy from SPARC observables.

    For a galaxy with flat rotation velocity Vflat and luminosity L:
      M_baryon ~ L * (M/L ratio) + M_HI * 1.33 (helium correction)
      V_baryon^2 = G * M_baryon / R

    A simpler proxy: the mass discrepancy
      MD = V_flat^2 / V_baryon^2

    Or even simpler: use the baryonic Tully-Fisher residual.
    Galaxies with MORE dark matter have HIGHER Vflat for their luminosity.

    We use: log(Vflat) - 0.25*log(L*M/L + 1.33*MHI)
    This is the residual from the baryonic Tully-Fisher relation.
    Positive residual = more dark matter than expected from baryons.
    """
    ML_disk = 0.5  # M/L ratio at 3.6 micron (standard from McGaugh)

    for g in galaxies:
        if g['L36'] is not None and g['L36'] > 0 and g['Vflat'] is not None:
            M_star = g['L36'] * ML_disk  # stellar mass in 10^9 Msun
            M_gas = g['MHI'] * 1.33 if g['MHI'] is not None else 0  # gas mass with He
            M_baryon = M_star + M_gas

            if M_baryon > 0:
                # Baryonic Tully-Fisher residual
                # BTFR: log(Vflat) = 0.25 * log(M_baryon) + constant
                g['log_Vflat'] = np.log10(g['Vflat'])
                g['log_Mbar'] = np.log10(M_baryon * 1e9)  # convert to Msun
                g['BTF_residual'] = g['log_Vflat'] - 0.25 * g['log_Mbar']

                # Mass discrepancy at flat part
                # V_baryon^2 ~ G*M_baryon/R, but we don't have R for flat part
                # Use a cruder proxy: Vflat^4 / (G * M_baryon)
                # Higher = more DM dominated
                G_proxy = 4.3e-3  # G in (km/s)^2 * pc / Msun
                g['DM_proxy'] = g['Vflat']**2 / (M_baryon * 1e9)**(0.5)
                g['f_dm_proxy'] = g['BTF_residual']  # use BTF residual as DM proxy
            else:
                g['f_dm_proxy'] = None
        else:
            g['f_dm_proxy'] = None

    return galaxies


# ============================================================
# RUN
# ============================================================

print("SPARC Coherence-Dark Matter Regression")
print("The $0 Test")
print("=" * 60)

# Parse
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SPARC_Lelli2016c.mrt')
galaxies = parse_sparc(filepath)
print(f"Parsed {len(galaxies)} galaxies from SPARC")

# Compute indices
galaxies = compute_coherence(galaxies)
galaxies = compute_dm_proxy(galaxies)

# Filter to galaxies with all data and finite values
valid = [g for g in galaxies if g['f_dm_proxy'] is not None
         and g['C_index'] is not None
         and np.isfinite(g['f_dm_proxy'])
         and g['Vflat'] > 0
         and g['L36'] > 0]
print(f"Valid galaxies with all measurements: {len(valid)}")

# Extract arrays
C = np.array([g['C_index'] for g in valid])
f_dm = np.array([g['f_dm_proxy'] for g in valid])
T = np.array([g['T'] for g in valid])
Q = np.array([g['Q'] for g in valid])
log_Mbar = np.array([g['log_Mbar'] for g in valid])
log_Vflat = np.array([g['log_Vflat'] for g in valid])

print(f"Coherence range: {C.min():.3f} to {C.max():.3f}")
print(f"DM proxy range: {f_dm.min():.3f} to {f_dm.max():.3f}")
print()

# ============================================================
# REGRESSION 1: Raw correlation
# ============================================================

print("REGRESSION 1: Raw C_index vs DM_proxy")
print("-" * 40)

slope, intercept, r_value, p_value, std_err = stats.linregress(C, f_dm)
print(f"  Slope: {slope:.4f} +/- {std_err:.4f}")
print(f"  R: {r_value:.4f}")
print(f"  R^2: {r_value**2:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  n: {len(valid)}")

if slope < 0:
    print("  >>> NEGATIVE SLOPE: Higher coherence = less DM. Direction MATCHES prediction. <<<")
else:
    print("  >>> POSITIVE SLOPE: Does not match prediction. <<<")

if p_value < 0.05:
    print(f"  >>> SIGNIFICANT at p < 0.05 <<<")
if p_value < 0.01:
    print(f"  >>> SIGNIFICANT at p < 0.01 <<<")
if p_value < 0.001:
    print(f"  >>> HIGHLY SIGNIFICANT at p < 0.001 <<<")

# ============================================================
# REGRESSION 2: Controlling for baryonic mass
# ============================================================

print()
print("REGRESSION 2: C_index vs DM_proxy, controlling for log(M_baryon)")
print("-" * 40)

# Partial correlation: regress both C and f_dm on log_Mbar, correlate residuals
slope_C, int_C, _, _, _ = stats.linregress(log_Mbar, C)
C_resid = C - (slope_C * log_Mbar + int_C)

slope_f, int_f, _, _, _ = stats.linregress(log_Mbar, f_dm)
f_resid = f_dm - (slope_f * log_Mbar + int_f)

slope2, intercept2, r2, p2, se2 = stats.linregress(C_resid, f_resid)
print(f"  Partial slope: {slope2:.4f} +/- {se2:.4f}")
print(f"  Partial R: {r2:.4f}")
print(f"  Partial R^2: {r2**2:.4f}")
print(f"  p-value: {p2:.6f}")

if slope2 < 0 and p2 < 0.05:
    print("  >>> SIGNAL SURVIVES MASS CONTROL. <<<")
elif slope2 < 0:
    print("  >>> Direction matches but not significant after mass control. <<<")
else:
    print("  >>> Signal does not survive mass control. <<<")

# ============================================================
# REGRESSION 3: Individual components
# ============================================================

print()
print("REGRESSION 3: Individual coherence components")
print("-" * 40)

C_morph = np.array([g['C_morph'] for g in valid])
C_kin = np.array([g['C_kin'] for g in valid])
C_SB = np.array([g['C_SB'] for g in valid])

for name, comp in [('C_morph (Hubble type)', C_morph),
                    ('C_kin (rotation quality)', C_kin),
                    ('C_SB (surface brightness)', C_SB)]:
    sl, _, rv, pv, se = stats.linregress(comp, f_dm)
    direction = "MATCHES" if sl < 0 else "opposite"
    sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
    print(f"  {name:35s}  slope={sl:+.4f}  R={rv:+.4f}  p={pv:.4f} {sig} {direction}")

# ============================================================
# BINNED ANALYSIS
# ============================================================

print()
print("BINNED ANALYSIS: Low vs High coherence")
print("-" * 40)

median_C = np.median(C)
low_C = f_dm[C < median_C]
high_C = f_dm[C >= median_C]

t_stat, t_pval = stats.ttest_ind(low_C, high_C)
print(f"  Low coherence (n={len(low_C)}):  mean DM proxy = {np.mean(low_C):.4f} +/- {np.std(low_C)/np.sqrt(len(low_C)):.4f}")
print(f"  High coherence (n={len(high_C)}): mean DM proxy = {np.mean(high_C):.4f} +/- {np.std(high_C)/np.sqrt(len(high_C)):.4f}")
print(f"  Difference: {np.mean(low_C) - np.mean(high_C):.4f}")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {t_pval:.6f}")

if np.mean(high_C) < np.mean(low_C):
    pct_diff = (np.mean(low_C) - np.mean(high_C)) / abs(np.mean(low_C)) * 100
    print(f"  >>> High coherence galaxies have {pct_diff:.1f}% LESS dark matter proxy <<<")
    print(f"  >>> Prediction was 5-10%. <<<")

# ============================================================
# BY HUBBLE TYPE
# ============================================================

print()
print("BY HUBBLE TYPE:")
print("-" * 40)

for t_val in sorted(set(T)):
    mask = T == t_val
    if np.sum(mask) >= 3:
        mean_dm = np.mean(f_dm[mask])
        n = np.sum(mask)
        print(f"  T={t_val:2d}  n={n:3d}  mean DM proxy = {mean_dm:.4f}")

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('SPARC: Coherence vs Dark Matter\n'
             'C-M-D Prediction: Higher coherence = less dark matter',
             fontsize=13, fontweight='bold')

# Scatter: C_index vs f_dm
ax = axes[0, 0]
scatter = ax.scatter(C, f_dm, c=T, cmap='RdYlGn_r', s=30, alpha=0.7, edgecolors='gray', linewidth=0.5)
x_line = np.linspace(C.min(), C.max(), 100)
ax.plot(x_line, slope * x_line + intercept, 'r-', lw=2,
        label=f'slope={slope:.3f}, R={r_value:.3f}, p={p_value:.4f}')
ax.set_xlabel('Coherence Index')
ax.set_ylabel('DM Proxy (BTF residual)')
ax.set_title('Raw Correlation')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Hubble T-type')

# Partial correlation (mass-controlled)
ax = axes[0, 1]
ax.scatter(C_resid, f_resid, c=T, cmap='RdYlGn_r', s=30, alpha=0.7, edgecolors='gray', linewidth=0.5)
x_line2 = np.linspace(C_resid.min(), C_resid.max(), 100)
ax.plot(x_line2, slope2 * x_line2 + intercept2, 'r-', lw=2,
        label=f'partial slope={slope2:.3f}, R={r2:.3f}, p={p2:.4f}')
ax.set_xlabel('Coherence residual (mass-controlled)')
ax.set_ylabel('DM proxy residual (mass-controlled)')
ax.set_title('After Controlling for Baryonic Mass')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Binned comparison
ax = axes[1, 0]
positions = [1, 2]
bp = ax.boxplot([low_C, high_C], positions=positions, widths=0.6, patch_artist=True)
bp['boxes'][0].set_facecolor('#ff9999')
bp['boxes'][1].set_facecolor('#99ff99')
ax.set_xticks(positions)
ax.set_xticklabels(['Low Coherence', 'High Coherence'])
ax.set_ylabel('DM Proxy')
ax.set_title(f'Binned: t={t_stat:.2f}, p={t_pval:.4f}')
ax.grid(alpha=0.3, axis='y')

# By Hubble type
ax = axes[1, 1]
T_unique = sorted(set(T))
means = [np.mean(f_dm[T == t]) for t in T_unique if np.sum(T == t) >= 3]
ns = [np.sum(T == t) for t in T_unique if np.sum(T == t) >= 3]
sems = [np.std(f_dm[T == t]) / np.sqrt(np.sum(T == t)) for t in T_unique if np.sum(T == t) >= 3]
t_plot = [t for t in T_unique if np.sum(T == t) >= 3]
ax.errorbar(t_plot, means, yerr=sems, fmt='o-', color='#1e5d5c', lw=2, capsize=4)
ax.set_xlabel('Hubble Type (1=Sa organized -> 10=Irr chaotic)')
ax.set_ylabel('Mean DM Proxy')
ax.set_title('Dark Matter vs Morphological Order')
ax.grid(alpha=0.3)

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparc_coherence_results.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")

# ============================================================
# VERDICT
# ============================================================

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
print()
if slope < 0 and p_value < 0.05:
    print("THE $0 TEST PASSES.")
    print(f"Coherence correlates with less dark matter (p={p_value:.4f}).")
    if slope2 < 0 and p2 < 0.05:
        print(f"Effect SURVIVES mass control (partial p={p2:.4f}).")
        print("This is a publishable result. The C-M-D prediction holds.")
    else:
        print(f"Effect weakens after mass control (partial p={p2:.4f}).")
        print("May be a projection of the baryonic Tully-Fisher relation.")
elif slope < 0 and p_value < 0.1:
    print("MARGINAL. Direction matches but not significant (p={:.4f}).".format(p_value))
    print("Needs richer coherence index (add HyperLeda/S4G data).")
elif slope < 0:
    print("DIRECTION MATCHES but not significant (p={:.4f}).".format(p_value))
    print("Minimal coherence index may lack power. Try full cross-match.")
else:
    print("DOES NOT MATCH prediction. Slope is positive.")
    print("Either the proxy is wrong or the theory needs revision.")
