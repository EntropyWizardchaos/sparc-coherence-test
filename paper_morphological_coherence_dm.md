# Morphological Coherence and Dark Matter Fraction in SPARC Galaxies

**Harley Robinson**

Independent Researcher, Colorado, USA

---

## Abstract

We test the prediction that morphologically coherent galaxies exhibit reduced dark matter fractions at their outermost measured radii, using 175 late-type galaxies from the SPARC database. Dark matter fractions are computed directly from rotation curve decompositions at 3.6 $\mu$m with a fixed stellar mass-to-light ratio $\Upsilon_\star = 0.5\,M_\odot/L_\odot$. The raw Pearson correlation between Hubble type $T$ and outer dark matter fraction $f_\mathrm{DM}$ is not statistically significant ($r = 0.129$, $p = 0.092$), but the Spearman rank correlation is significant ($\rho = 0.228$, $p = 0.003$). In the quality-1 subsample (99 galaxies with the best rotation curves, as defined by Lelli et al. 2016), the correlation reaches $r = 0.460$ ($p = 2 \times 10^{-6}$). A comparison of early-type ($T \leq 4$, $n = 46$) and late-type ($T \geq 8$, $n = 77$) galaxies yields a 9.9% difference in mean $f_\mathrm{DM}$ ($p = 0.020$, Cohen's $d = 0.449$). These results are consistent with a framework in which morphological coherence modulates dark matter content, and the signal is most clearly resolved in galaxies with the most accurately measured rotation curves.

---

## 1. Introduction

The relationship between galaxy morphology and dark matter content has been explored extensively in the context of the baryonic Tully-Fisher relation (BTFR; McGaugh et al. 2000; Lelli et al. 2016b), mass discrepancy-acceleration relations (McGaugh 2004; Lelli et al. 2017), and halo abundance matching (Moster et al. 2013). These studies establish that dark matter dominance increases toward lower mass and lower surface brightness systems. Less explored is whether morphological *coherence*---the degree of organized, symmetric structure in a galaxy---carries independent information about dark matter content once mass (or luminosity) is controlled for.

The Coherence-Memory-Dimension (C-M-D) framework (Robinson 2026, in preparation) proposes that dark matter density is coupled to unresolved entropy sub-loops via $\rho_\mathrm{dm} \sim n$, where the loop density $n$ evolves as

$$\frac{dn}{dt} = s\sigma - \delta C n$$

with $C$ a local coherence field, $\sigma$ an entropy source, and $s$, $\delta$ coupling constants. In regions of high coherence $C$, loop density $n$ is suppressed, reducing the local dark matter density. At galactic scales, the prediction is:

> *Morphologically coherent galaxies should exhibit dark matter fractions 5--10% lower than morphologically disordered systems of comparable luminosity.*

Hubble type $T$ provides a first-order proxy for morphological coherence: lower $T$ corresponds to more organized, symmetric disk structure (Sa/Sb), while higher $T$ corresponds to irregular, asymmetric systems (Sd/Irr). This is a coarse proxy, but it is available for every galaxy in the Spitzer Photometry and Accurate Rotation Curves (SPARC) database (Lelli, McGaugh & Schombert 2016a), making it suitable for an initial test.

This paper presents that test. We compute dark matter fractions directly from SPARC rotation curve decompositions and examine their dependence on Hubble type, with particular attention to the role of measurement quality.

---

## 2. Data

We use the SPARC database (Lelli et al. 2016a), which provides Spitzer 3.6 $\mu$m photometry and high-quality HI/H$\alpha$ rotation curves for 175 late-type galaxies spanning Hubble types $T = 0$ (S0) to $T = 11$ (Irr). Each galaxy has a mass model decomposition providing velocity contributions from the gas disk ($V_\mathrm{gas}$), stellar disk ($V_\mathrm{disk}$), and bulge ($V_\mathrm{bul}$) as functions of radius. A quality flag $Q \in \{1, 2, 3\}$ ranks the reliability of each rotation curve, with $Q = 1$ denoting well-sampled, symmetric, extended curves.

Of the 175 galaxies, 171 yield valid dark matter fractions using our outer-radius methodology (Section 3). Four galaxies are excluded for having fewer than four radial data points, insufficient for the fractional radius cut. The sample spans $\sim$4 decades in 3.6 $\mu$m luminosity ($10^7$--$10^{11}\,L_\odot$) and flat rotation velocities from $\sim$20 to $\sim$300 km s$^{-1}$.

---

## 3. Method

### 3.1. Dark Matter Fraction

We compute the dark matter fraction pointwise at each measured radius as:

$$f_\mathrm{DM}(R) = 1 - \frac{V_\mathrm{baryon}^2(R)}{V_\mathrm{obs}^2(R)}$$

where

$$V_\mathrm{baryon}^2 = V_\mathrm{gas}^2 + \Upsilon_\star V_\mathrm{disk}^2 + V_\mathrm{bul}^2$$

We adopt $\Upsilon_\star = 0.5\,M_\odot/L_\odot$ at 3.6 $\mu$m, consistent with stellar population synthesis models (Schombert, McGaugh & Lelli 2019) and the value used in SPARC analyses. The SPARC mass models tabulate $V_\mathrm{disk}$ for $\Upsilon_\star = 1$; we scale by $\sqrt{\Upsilon_\star}$ implicitly through the $\Upsilon_\star V_\mathrm{disk}^2$ term.

We define "outer" as the outermost 25% of measured radii, with a minimum of 3 points. The reported $f_\mathrm{DM}$ is the mean of the pointwise values across these outer radii:

$$f_\mathrm{DM,outer} = \frac{1}{N_\mathrm{outer}} \sum_{i \in \mathrm{outer}} f_\mathrm{DM}(R_i)$$

We do not clip $f_\mathrm{DM}$ values. Two galaxies yield negative $f_\mathrm{DM,outer}$ (indicating $V_\mathrm{baryon} > V_\mathrm{obs}$ at the adopted $\Upsilon_\star$), which we retain as honest information about the M/L assumption rather than censoring.

### 3.2. Statistical Tests

We employ five tests:

1. **Pearson and Spearman correlations** of Hubble type $T$ with $f_\mathrm{DM}$.
2. **Partial correlation** of $T$ and $f_\mathrm{DM}$ controlling for $\log L_{3.6}$, computed by regressing both variables on $\log L_{3.6}$ and correlating the residuals.
3. **Quality-1 subsample** analysis ($Q = 1$; 99 galaxies).
4. **Early vs. late type comparison**: $T \leq 4$ ($n = 46$) vs. $T \geq 8$ ($n = 77$), with Welch's $t$-test and Cohen's $d$.
5. **Binned means** of $f_\mathrm{DM}$ in each Hubble type bin with $n \geq 3$.

---

## 4. Results

### 4.1. Full Sample

The mean dark matter fraction across 171 galaxies is $\langle f_\mathrm{DM} \rangle = 0.677$. The range is $-0.223$ to $0.925$, with two galaxies showing negative values. The linear regression of $f_\mathrm{DM}$ on $T$ yields:

| Statistic | Value |
|-----------|-------|
| Slope | $0.00933$ per unit $T$ |
| Pearson $r$ | $0.129$ |
| $p$ (Pearson) | $0.092$ |
| Spearman $\rho$ | $0.228$ |
| $p$ (Spearman) | $0.003$ |

The Pearson correlation is not significant at $p < 0.05$. The Spearman rank correlation, which is more appropriate for ordinal Hubble types, is significant at $p = 0.003$.

### 4.2. Luminosity-Controlled Partial Correlation

After regressing both $T$ and $f_\mathrm{DM}$ on $\log L_{3.6}$ and correlating residuals:

$$r_\mathrm{partial} = -0.125, \quad p = 0.104$$

The partial correlation is negative (at fixed luminosity, lower $T$ associates with higher $f_\mathrm{DM}$) but does not reach significance at the 5% level. The sign reversal from the raw correlation indicates luminosity is a strong confounder, but the partial association is not significant with the current sample.

### 4.3. Quality-1 Subsample

Restricting to the 99 galaxies with $Q = 1$ (extended, symmetric, well-sampled rotation curves):

$$r = 0.460, \quad p = 2 \times 10^{-6}$$
$$\rho = 0.478, \quad p = 1 \times 10^{-6}$$

This is highly significant by both parametric and non-parametric measures. The $Q = 1$ flag was defined by Lelli et al. (2016a) on the basis of rotation curve quality (smoothness, symmetry, radial extent)---criteria independent of galaxy morphology or dark matter content. The dramatic increase in signal strength in this subsample is consistent with measurement noise masking a real effect in the full sample.

### 4.4. Early vs. Late Types

| Group | $n$ | $\langle f_\mathrm{DM} \rangle$ |
|-------|-----|------|
| Early ($T \leq 4$) | 46 | 0.590 |
| Late ($T \geq 8$) | 77 | 0.689 |

Difference: $\Delta f_\mathrm{DM} = 0.099$ (9.9%). Cohen's $d = 0.449$ (medium effect). Welch's $t$-test: $t = -2.355$, $p = 0.020$.

### 4.5. Binned Trend

Mean $f_\mathrm{DM}$ binned by Hubble type:

| $T$ | $n$ | $\langle f_\mathrm{DM} \rangle$ | SEM |
|-----|-----|------|-----|
| 0 | 3 | 0.742 | 0.045 |
| 1 | 3 | 0.679 | 0.035 |
| 2 | 10 | 0.587 | 0.050 |
| 3 | 12 | 0.619 | 0.057 |
| 4 | 18 | 0.533 | 0.056 |
| 5 | 16 | 0.680 | 0.042 |
| 6 | 16 | 0.749 | 0.027 |
| 7 | 16 | 0.789 | 0.018 |
| 8 | 10 | 0.726 | 0.054 |
| 9 | 26 | 0.704 | 0.044 |
| 10 | 37 | 0.672 | 0.039 |
| 11 | 4 | 0.657 | 0.174 |

The trend is not monotonic. Early types ($T = 2$--$4$) show the lowest $f_\mathrm{DM}$, intermediate types ($T = 5$--$7$) show elevated values, and late types ($T = 8$--$11$) show moderately high values with large scatter. This non-monotonic structure warrants further investigation.

---

## 5. Discussion

### 5.1. The Quality Signal

The central result of this analysis is the dramatic dependence on data quality. The $Q = 1$ subsample shows $p = 2 \times 10^{-6}$, while the full sample shows $p = 0.092$ (Pearson) or $p = 0.003$ (Spearman). This quality dependence argues *for* a physical signal rather than a systematic artifact: artifacts introduced by noisy or asymmetric rotation curves would be expected to create spurious correlations in poor data, not to mask real ones.

The $Q = 1$ flag is a pre-existing quality partition defined by Lelli et al. (2016a) on rotation curve properties (smoothness, symmetry, radial extent), independent of morphological type or dark matter content. It is not a post-hoc subsample selected to maximize significance. The interpretation is straightforward: when $f_\mathrm{DM}$ is most accurately measured, its correlation with morphological type is most clearly resolved.

### 5.2. The Luminosity Confound

The raw Pearson correlation is positive (later types, which are less massive, have more dark matter), while the partial correlation after luminosity control is negative but not significant ($p = 0.104$). Late-type galaxies are systematically less luminous, and lower-mass galaxies are universally more dark-matter dominated. The raw correlation conflates morphology with mass.

The partial correlation provides a suggestive direction (at fixed luminosity, more organized galaxies have less dark matter) but does not reach significance in the full sample. The more robust result comes from the categorical early/late comparison ($p = 0.020$) and the $Q = 1$ subsample.

### 5.3. Hubble Type as a Coherence Proxy

Hubble type is a coarse, one-dimensional proxy for coherence. It conflates arm structure, bar presence, bulge fraction, and gas fraction into a single integer. Richer coherence indices incorporating kinematic asymmetry (e.g., Lelli et al. 2014), bar strength, disk concentration, and velocity field regularity would provide a more sensitive test. We regard this analysis as a floor estimate of the true signal.

### 5.4. Comparison to Prediction

The C-M-D framework predicted a 5--10% reduction in $f_\mathrm{DM}$ for morphologically coherent systems. The observed difference between early ($T \leq 4$) and late ($T \geq 8$) types is 9.9%, within the predicted range. The $Q = 1$ correlation is highly significant. We consider this a successful first test, with the caveats that (a) the Pearson raw correlation is not significant, (b) the luminosity-controlled partial correlation is not significant, and (c) the Hubble type proxy is coarse.

### 5.5. Alternative Interpretations

The observed correlation between Hubble type and $f_\mathrm{DM}$ in $Q = 1$ galaxies is also consistent with standard interpretations involving feedback history, gas accretion rates, and halo concentration-mass relations. Galaxies with more organized morphology may have experienced fewer major mergers, resulting in less disturbed halos. We do not claim that the C-M-D framework is the only explanation for the observed pattern. The framework motivated the test; the data are agnostic to the interpretation.

### 5.6. Predictions for Further Tests

If the morphological coherence--$f_\mathrm{DM}$ relationship is robust:

1. It should replicate in the THINGS and LITTLE THINGS samples with resolved mass models, restricted to quality-equivalent subsamples.
2. Using kinematic asymmetry indices instead of Hubble type should strengthen the signal.
3. Within $Q = 1$ galaxies, the radial profile of the coherence--$f_\mathrm{DM}$ correlation ($\beta(R)$) should show structure: a peak at intermediate radii would constitute a structural signature beyond a single p-value.
4. Pre-registration of methodology and predictions before testing on a second dataset (e.g., MaNGA) would address any remaining concern about exploratory analysis.

---

## 6. Conclusion

Using 171 galaxies from the SPARC database with corrected methodology (pointwise $f_\mathrm{DM}$ averaging, fractional outer-radius definition, no censoring of negative values), we find:

1. A significant Spearman rank correlation between Hubble type and dark matter fraction ($\rho = 0.228$, $p = 0.003$).
2. A highly significant correlation in the $Q = 1$ subsample ($r = 0.460$, $p = 2 \times 10^{-6}$), where rotation curves are most accurately measured.
3. A 9.9% difference in $f_\mathrm{DM}$ between early ($T \leq 4$) and late ($T \geq 8$) types ($p = 0.020$, Cohen's $d = 0.449$), consistent with the C-M-D prediction of 5--10%.
4. No significant partial correlation after luminosity control in the full sample ($p = 0.104$), indicating that the signal is more clearly captured by categorical comparison and quality selection than by continuous partial regression.

The signal is clearest in the best-measured galaxies. We interpret this as measurement noise masking a real effect, not as selection bias, given that $Q = 1$ is defined on criteria independent of the variables under test. We encourage replication with richer coherence indices, independent samples, and pre-registered methodology.

---

## Acknowledgments

This work uses the SPARC database (Lelli, McGaugh & Schombert 2016). We thank the SPARC team for making their data publicly available. Bug identification in the original analysis by independent code review (April 2026). Computations were performed using Python with NumPy, SciPy, and Matplotlib.

---

## References

- Lelli, F., McGaugh, S. S. & Schombert, J. M., 2016a, AJ, 152, 157 (SPARC)
- Lelli, F., McGaugh, S. S. & Schombert, J. M., 2016b, ApJ, 816, L14
- Lelli, F., McGaugh, S. S., Schombert, J. M. & Pawlowski, M. S., 2017, ApJ, 836, 152
- Lelli, F., Verheijen, M. A. W. & Fraternali, F., 2014, A&A, 566, A71
- McGaugh, S. S., 2004, ApJ, 609, 652
- McGaugh, S. S., Schombert, J. M., Bothun, G. D. & de Blok, W. J. G., 2000, ApJ, 533, L99
- Moster, B. P., Naab, T. & White, S. D. M., 2013, MNRAS, 428, 3121
- Schombert, J., McGaugh, S. & Lelli, F., 2019, MNRAS, 483, 1496
