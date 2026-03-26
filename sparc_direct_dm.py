"""
SPARC Direct Dark Matter Fraction
==================================
Compute f_DM directly from rotation curve decomposition at outer radii.
No proxy. No BTF residual. The actual dark matter fraction.

f_DM(r) = 1 - V_baryon^2 / V_obs^2
where V_baryon^2 = V_gas^2 + (M/L_disk)^2 * V_disk^2 + V_bul^2

At the outermost measured radius, DM dominates. This is the clean test.

Author: Annie Robinson (night shift solo build)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob

# ============================================================
# PARSE MASTER TABLE FOR HUBBLE TYPES
# ============================================================

def parse_master(filepath):
    """Get Hubble type and quality for each galaxy."""
    info = {}
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
            name = parts[0]
            T = int(parts[1])
            Q = int(parts[17])
            L36 = float(parts[7])
            Vflat = float(parts[15])
            info[name] = {'T': T, 'Q': Q, 'L36': L36, 'Vflat': Vflat}
        except:
            continue
    return info

# ============================================================
# COMPUTE f_DM FROM ROTATION CURVES
# ============================================================

def compute_fdm_outer(rotmod_dir, galaxy_info, ML_disk=0.5):
    """
    For each galaxy, read the rotation curve mass model.
    Compute f_DM at the outermost 3 radial points (averaged).

    V_baryon^2 = V_gas^2 + ML_disk * V_disk^2 + V_bul^2
    f_DM = 1 - V_baryon^2 / V_obs^2
    """
    results = []

    for filepath in glob.glob(os.path.join(rotmod_dir, '*_rotmod.dat')):
        basename = os.path.basename(filepath)
        galaxy_name = basename.replace('_rotmod.dat', '')

        # Match to master table
        if galaxy_name not in galaxy_info:
            # Try variations
            matched = False
            for key in galaxy_info:
                if key.replace(' ', '') == galaxy_name or key.replace('-', '') == galaxy_name:
                    galaxy_name_matched = key
                    matched = True
                    break
            if not matched:
                continue
        else:
            galaxy_name_matched = galaxy_name

        try:
            data = np.loadtxt(filepath, comments='#')
            if len(data) < 5:
                continue

            Rad = data[:, 0]
            Vobs = data[:, 1]
            errV = data[:, 2]
            Vgas = data[:, 3]
            Vdisk = data[:, 4]
            Vbul = data[:, 5] if data.shape[1] > 5 else np.zeros_like(Vobs)

            # V_baryon^2 with M/L scaling
            # Vdisk in the file is for M/L = 1. Scale by sqrt(ML_disk)
            V_bar_sq = Vgas**2 + ML_disk * Vdisk**2 + Vbul**2

            # f_DM at each radius
            valid = Vobs > 10  # exclude very inner points with tiny velocities
            if np.sum(valid) < 3:
                continue

            Vobs_v = Vobs[valid]
            V_bar_v = V_bar_sq[valid]

            # Average over outermost 3 points
            n_outer = min(3, len(Vobs_v))
            Vobs_outer = Vobs_v[-n_outer:]
            Vbar_outer = V_bar_v[-n_outer:]

            f_dm_outer = 1.0 - np.mean(Vbar_outer) / np.mean(Vobs_outer**2)
            f_dm_outer = np.clip(f_dm_outer, 0, 1)

            # Also compute at half-radius
            mid = len(Vobs_v) // 2
            f_dm_mid = 1.0 - V_bar_v[mid] / Vobs_v[mid]**2
            f_dm_mid = np.clip(f_dm_mid, 0, 1)

            info = galaxy_info[galaxy_name_matched]
            results.append({
                'name': galaxy_name,
                'T': info['T'],
                'Q': info['Q'],
                'L36': info['L36'],
                'f_dm_outer': f_dm_outer,
                'f_dm_mid': f_dm_mid,
                'R_last': Rad[valid][-1],
                'Vobs_last': Vobs_v[-1],
            })

        except Exception as e:
            continue

    return results


# ============================================================
# RUN
# ============================================================

print("SPARC Direct Dark Matter Fraction Test")
print("No proxy. Actual f_DM from rotation curve decomposition.")
print("=" * 60)

master_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SPARC_Lelli2016c.mrt')
rotmod_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Rotmod_LTG')

galaxy_info = parse_master(master_path)
print(f"Master table: {len(galaxy_info)} galaxies")

results = compute_fdm_outer(rotmod_dir, galaxy_info)
print(f"Galaxies with valid f_DM: {len(results)}")

T = np.array([r['T'] for r in results])
f_dm = np.array([r['f_dm_outer'] for r in results])
Q = np.array([r['Q'] for r in results])
L36 = np.array([r['L36'] for r in results])
log_L = np.log10(np.maximum(L36, 1e-3))

print(f"f_DM range: {f_dm.min():.3f} to {f_dm.max():.3f}")
print(f"f_DM mean: {f_dm.mean():.3f}")
print()

# ============================================================
# TEST 1: T vs f_DM (outer)
# ============================================================

print("TEST 1: Hubble Type vs f_DM (outer radius)")
print("-" * 45)

slope, intercept, r_val, p_val, std_err = stats.linregress(T, f_dm)
rho, p_spear = stats.spearmanr(T, f_dm)

print(f"  Pearson: slope={slope:.5f}, R={r_val:.4f}, p={p_val:.6f}")
print(f"  Spearman: rho={rho:.4f}, p={p_spear:.6f}")

if slope > 0 and p_val < 0.05:
    print("  >>> POSITIVE: Chaotic galaxies have MORE dark matter <<<")
    print("  >>> MATCHES C-M-D prediction: coherence reduces DM <<<")
elif slope < 0 and p_val < 0.05:
    print("  >>> NEGATIVE: Organized galaxies have MORE dark matter <<<")
    print("  >>> OPPOSITE to prediction <<<")
else:
    print(f"  >>> Not significant (p={p_val:.4f}) <<<")

# ============================================================
# TEST 2: Partial correlation controlling for luminosity
# ============================================================

print("\nTEST 2: Controlling for luminosity")
print("-" * 45)

sl_t, int_t, _, _, _ = stats.linregress(log_L, T.astype(float))
T_resid = T - (sl_t * log_L + int_t)

sl_f, int_f, _, _, _ = stats.linregress(log_L, f_dm)
f_resid = f_dm - (sl_f * log_L + int_f)

sl2, int2, r2, p2, se2 = stats.linregress(T_resid, f_resid)
print(f"  Partial: slope={sl2:.5f}, R={r2:.4f}, p={p2:.6f}")

# ============================================================
# TEST 3: Binned by Hubble type
# ============================================================

print("\nTEST 3: Mean f_DM by Hubble type")
print("-" * 45)
print(f"  {'T':>3s} {'n':>5s} {'f_DM':>8s} {'sem':>8s}")

T_vals = sorted(set(T))
t_plot, means, sems = [], [], []
for t_val in T_vals:
    mask = T == t_val
    n = np.sum(mask)
    if n >= 3:
        m = np.mean(f_dm[mask])
        s = np.std(f_dm[mask]) / np.sqrt(n)
        t_plot.append(t_val)
        means.append(m)
        sems.append(s)
        print(f"  {t_val:3d} {n:5d} {m:8.4f} {s:8.4f}")

# ============================================================
# TEST 4: Early vs Late
# ============================================================

print("\nTEST 4: Early (T<=4) vs Late (T>=8)")
print("-" * 45)

early = f_dm[T <= 4]
late = f_dm[T >= 8]

if len(early) >= 5 and len(late) >= 5:
    t_stat, t_pval = stats.ttest_ind(early, late)
    d = (np.mean(late) - np.mean(early)) / np.sqrt((np.var(early) + np.var(late)) / 2)
    print(f"  Early (T<=4): n={len(early)}, f_DM={np.mean(early):.4f}")
    print(f"  Late  (T>=8): n={len(late)}, f_DM={np.mean(late):.4f}")
    print(f"  Difference: {np.mean(late) - np.mean(early):.4f}")
    print(f"  t={t_stat:.3f}, p={t_pval:.6f}")
    print(f"  Cohen's d = {d:.3f}")

    if np.mean(late) > np.mean(early) and t_pval < 0.05:
        print("  >>> CHAOTIC GALAXIES HAVE MORE DARK MATTER <<<")
        print("  >>> C-M-D PREDICTION CONFIRMED <<<")

# ============================================================
# TEST 5: Quality=1 only
# ============================================================

print("\nTEST 5: Quality=1 only")
print("-" * 45)

q1 = Q == 1
if np.sum(q1) > 20:
    sl5, _, r5, p5, _ = stats.linregress(T[q1], f_dm[q1])
    rho5, ps5 = stats.spearmanr(T[q1], f_dm[q1])
    print(f"  n={np.sum(q1)}")
    print(f"  Pearson: R={r5:.4f}, p={p5:.6f}")
    print(f"  Spearman: rho={rho5:.4f}, p={ps5:.6f}")

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('SPARC: Hubble Type vs ACTUAL Dark Matter Fraction\n'
             'Direct f_DM from rotation curve decomposition at outer radius',
             fontsize=13, fontweight='bold')

ax = axes[0, 0]
ax.scatter(T, f_dm, c=log_L, cmap='viridis', s=30, alpha=0.7, edgecolors='gray', linewidth=0.5)
x_fit = np.linspace(0, 11, 100)
ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=2,
        label=f'R={r_val:.3f}, p={p_val:.4f}')
ax.set_xlabel('Hubble Type (0=S0 → 10=Irr)')
ax.set_ylabel('f_DM at outer radius')
ax.set_title('Direct Dark Matter Fraction vs Morphology')
ax.legend(); ax.grid(alpha=0.3)
plt.colorbar(ax.collections[0], ax=ax, label='log(L_3.6)')

ax = axes[0, 1]
ax.errorbar(t_plot, means, yerr=sems, fmt='o-', color='#1e5d5c', lw=2, capsize=5, markersize=8)
ax.set_xlabel('Hubble Type')
ax.set_ylabel('Mean f_DM (outer)')
ax.set_title(f'Binned (Spearman rho={rho:.3f}, p={p_spear:.4f})')
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.scatter(T_resid, f_resid, c=T, cmap='RdYlGn_r', s=30, alpha=0.7)
x_fit2 = np.linspace(T_resid.min(), T_resid.max(), 100)
ax.plot(x_fit2, sl2 * x_fit2 + int2, 'r-', lw=2,
        label=f'partial R={r2:.3f}, p={p2:.4f}')
ax.set_xlabel('T residual (luminosity-controlled)')
ax.set_ylabel('f_DM residual (luminosity-controlled)')
ax.set_title('After Controlling for Luminosity')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 1]
if len(early) >= 5 and len(late) >= 5:
    bp = ax.boxplot([early, late], positions=[1, 2], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('#99ff99')
    bp['boxes'][1].set_facecolor('#ff9999')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Early (T<=4)\nOrganized', 'Late (T>=8)\nChaotic'])
    ax.set_ylabel('f_DM (outer)')
    ax.set_title(f'Early vs Late: p={t_pval:.4f}, d={d:.2f}')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparc_direct_dm_results.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")

# ============================================================
# VERDICT
# ============================================================

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)

if slope > 0 and p_val < 0.05:
    print("DIRECT f_DM CONFIRMS: organized galaxies have LESS dark matter.")
    print("The C-M-D prediction holds with the proper dark matter measure.")
    print("The BTF residual was confounded by baryonic efficiency.")
elif slope < 0 and p_val < 0.05:
    print("DIRECT f_DM SHOWS: organized galaxies have MORE dark matter.")
    print("This persists even with the clean measure.")
    print("The prediction may need revision.")
else:
    print(f"NOT SIGNIFICANT with direct f_DM measure (p={p_val:.4f}).")
    print("The signal may be in the baryonic efficiency, not the DM fraction.")
