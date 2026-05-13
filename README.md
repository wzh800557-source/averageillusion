# The Averaging Illusion

**Population-averaging kernels manufacture apparent tensions across astrophysics**

Wang & Shan (2026)

## Overview

Analysis code for "Population-averaging kernels manufacture apparent tensions across astrophysics." We prove a kernel evolution theorem and detect the effect at 3.6σ in the SNIa distance ladder.

### Key Results

| Test | Result | Significance |
|------|--------|-------------|
| Pantheon+ mass split | ΔΩm = −0.051 ± 0.015 | 3.3σ |
| DES-SN5YR mass split | ΔΩm = −0.047 ± 0.037 | 1.3σ |
| Combined | ΔΩm = −0.050 ± 0.014 | 3.6σ |
| γ₁z single-likelihood | γ₁ = +0.076 ± 0.031 | 2.4σ |
| Overlap (339 SNe removed) | ΔΩm = −0.051 (unchanged) | — |
| Reweighted z-distribution | ΔΩm = −0.049 (unchanged) | — |
| THESAN galaxy bias | 13–21% at z = 7–9 | — |
| THESAN fesc suppression | 93% (median → lum-weighted) | — |
| Kernel artifact αz | +1.35 (68% of observed) | — |

## Repository Structure

```
pantheon/
  pantheon_split.py         # Host mass split (full STAT+SYS covariance)
  gamma1z_and_overlap.py    # γ₁z single-likelihood + overlap test
  robustness_suite.py       # Reweighting, threshold scan, cross-covariance

des/
  des_fullcov.py            # Full precision matrix analysis
  des_diagonal.py           # Diagonal-error analysis

thesan/
  thesan_bias.py            # Galaxy bias mismatch (3 kernel prescriptions)

reionization/
  reion_ode.py              # Tanh-calibrated reionization histories
  reion_ode_full.py         # Full ODE with proper σ(M)
  zeff_bias.py              # ⟨z⟩_K bias calculation (0.7% correction)

figures/
  make_all_figures.py       # All paper figures
  figS_toy.py               # Controlled toy example (Fig S1)

data/
  test1_overlap.json        # Overlap test results
  test2_gamma1z.json        # γ₁z test results
  tier12_robustness.json    # Robustness suite results
  des_sn5yr_mass_split.json # DES results
```

## Quick Start

```bash
pip install numpy scipy matplotlib

# Pantheon+ mass split
git clone https://github.com/PantheonPlusSH0ES/DataRelease.git pantheonplus_data
python pantheon/pantheon_split.py

# DES-SN5YR analysis
git clone --depth 1 https://github.com/des-science/DES-SN5YR.git des_data
python des/des_fullcov.py

# γ₁z test + overlap analysis
python pantheon/gamma1z_and_overlap.py

# Full robustness suite
python pantheon/robustness_suite.py

# Figures
python figures/figS_toy.py
```

## Data Sources

- **Pantheon+**: [Brout et al. (2022)](https://github.com/PantheonPlusSH0ES/DataRelease)
- **DES-SN5YR**: [DES Collaboration (2024)](https://github.com/des-science/DES-SN5YR)
- **THESAN**: [Kannan et al. (2022)](https://www.thesan-project.com/)


```

## License

MIT License
