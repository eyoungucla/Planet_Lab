"""
neptune_mcmc.py
===============
MCMC search over (surface_P_GPa, bulk_wt_H2_percent) to fit Neptune's
observed J2 and normalised moment of inertia C/MR², using emcee.

The forward model is Planet_LAB() from Planet_Lab_v33_fnc.py.
All other model parameters are held fixed at the values below —
edit the FIXED PARAMETERS block to match your preferred defaults.

Observational targets
---------------------
  Neptune mass          : 1.02413e26 kg  (fixed model input)
  Neptune 1-bar radius  : 24764 km       (constraint via J2 calibration)
  J2                    : 3341.43 ± 4.5  × 10⁻⁶
  J4                    : -33.40  ± 2.9  × 10⁻⁶
  C/MR²                 : 0.2315  (uncertainty assumed: see SIGMA_C below)

Usage
-----
  python neptune_mcmc.py                   # run MCMC
  python neptune_mcmc.py --plot            # load chains and plot only
  python neptune_mcmc.py --resume          # continue from saved checkpoint

Output files
------------
  neptune_mcmc_chain.h5      — emcee backend (checkpoint + full chain)
  neptune_mcmc_corner.png/pdf
  neptune_mcmc_chains.png/pdf
  neptune_mcmc_results.txt
"""

import argparse
import sys
import os
import time
import warnings
import logging
from pathlib import Path

import numpy as np
import emcee
import corner
import matplotlib
matplotlib.use('Agg')  # headless — remove if running interactively
import matplotlib.pyplot as plt

# ── ApJ-style defaults ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 11,
    'axes.linewidth': 0.8, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'lines.linewidth': 1.2, 'legend.fontsize': 9,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# ── Import your model ─────────────────────────────────────────────────────────
from Planet_Lab_v33_fnc import Planet_LAB

# ── Import J2 tools from calculate_J2.py ─────────────────────────────────────
# (expects calculate_J2.py to be in the same directory or on PYTHONPATH)
import calculate_J2 as J2mod

# ── J2/J4 calculation method ─────────────────────────────────────────────────
# 'empirical' : fast empirical calibration via calculate_J2.py (default,
#               recommended for production MCMC — ~0.001 s per call)
# 'cms'       : full CMS via CMSPlanet/cms.py (~30-60 s per call, no numba).
#               Use only for short validation runs (few walkers, few steps).
#               With numba installed this drops to ~2-5 s per call.
J2_METHOD = 'cms'   # change to 'cms' for CMS validation runs

# CMS settings (only used when J2_METHOD == 'cms')
CMS_N_LAYERS = 800    # CMS shells
CMS_XLAYERS  = -1     # -1 = full calculation (required for sharp profiles)

# ── Suppress noisy output from Planet_LAB during MCMC ────────────────────────
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# =============================================================================
#  OBSERVATIONAL TARGETS  (Neptune)
#  See Table 1 from Morf and Helled (2025) for values.
#
# Source: Morf & Helled (A&A, 2025), Table 1, which tabulates values from
# Wang et al. (A&A, 2023) at a conventional reference radius R_ref = 25225 km:
#
#   J2 = 3401.655 ± 3.994  × 10⁻⁶   (uncertainty is Monte Carlo, Wang+2023)
#   J4 =  -33.294 ± 10.000 × 10⁻⁶
#
# The gravitational harmonics are defined via the external potential:
#
#   V = (GM/r) [ 1 - J2*(R/r)^2 * P2(cosθ) - J4*(R/r)^4 * P4(cosθ) - ... ]
#
# where R is the reference radius. Because V is a physical observable,
# the J values must be renormalized when changing reference radius:
#
#   J2(R_new) = J2(R_old) * (R_old / R_new)^2
#   J4(R_new) = J4(R_old) * (R_old / R_new)^4
#
# We renormalize from R_ref = 25225 km to the physical 1-bar equatorial
# radius R_eq = 24766 km (Morf & Helled 2025, Table 1; Lindal 1992),
# which is the radius convention used throughout this model.
# The ratio (25225/24766)^2 = 1.03741, so J2 increases by ~3.7%.
#
#   J2 = 3528.912 ± 4.143  × 10⁻⁶   (renormalized to R_eq = 24766 km)
#   J4 =  -35.832 ± 10.762 × 10⁻⁶
#
# References:
#   Morf & Helled, A&A (2025)
#   Wang et al., A&A 671, A70 (2023)
#   Lindal, AJ 103, 967 (1992)        [1-bar equatorial radius]
# =============================================================================
M_NEPTUNE_KG   = 1.02413e26        # kg  — fixed model input

J2_OBS         = 3528.912e-6 -0.87e-6  # correct for winds, Paula
J2_SIGMA       = 4.143e-6            # observational 1-σ  (overridden below by SIGMA_MODE)

J4_OBS         = -35.832e-6 + 3.88e-06  # corrected for winds
J4_SIGMA       = 10.762e-6            # observational 1-σ  (overridden below by SIGMA_MODE)

C_OBS          = 0.241            # C/MR²  (from Voyager 2 + ToF, Nettelmann + 2013)
# Assumed uncertainty on C/MR²: observational precision is ~0.5% but
# model-to-observation mapping adds ~2–3% systematic.  We use 2% here.
# Increase SIGMA_C to broaden the C/MR² constraint if desired.
SIGMA_C        = 0.241 * 0.02     # 2% of observed value

# ── 1-bar radius (equatorial — see notes block below) ─────────────────────
# The chi^2 target for R is now R_eq.  Internally, the structure integration
# produces R_1bar (volumetric in physical content); the cost function then
# applies a first-order bulge correction to get the model's implied R_eq,
# and that is what enters chi^2.  σ_R is broadened from the Lindal ±4 km
# observational value to the R_vol-flavored value (19 km), reflecting that
# the underlying degree of freedom being fit is the spherical model radius;
# we report R_eq for direct comparison with the literature.
R_OBS          = 24764e3          # m   — Neptune R_eq  (Lindal 1992)
SIGMA_R        = 19e3             # m   — adopted: R_vol observational σ;
                                  #       prop. errors in Req and Rpol
                                  #   cite: +/- 15km for Req and 30 km for polar R

R_VOL_OBS      = 24622e3          # m   — Neptune R_vol, reported only
SIGMA_R_VOL    = 19e3             # m   — Lindal 1992 (NSSDC)

# Legacy alias — some downstream code may still reference RP_NEPTUNE_M
RP_NEPTUNE_M   = R_OBS

# =============================================================================
#  1-BAR ATMOSPHERIC TEMPERATURE CONSTRAINT  (optional)
# =============================================================================
# Optionally include the observed 1-bar temperature as an additional Gaussian
# term in the likelihood. The model T at 1 bar is interpolated from the returned
# atmosphere profile (P_planet_Pa, T_planet_K). This lets the fit balance the
# atmospheric thermal structure against J2/J4/C/R, rather than leaving the 1-bar
# temperature as a pure output.
#
# USE_T1BAR : master switch. Set False to recover the original gravity+radius fit
#             exactly (the term is then omitted from the likelihood AND from the
#             reported degrees of freedom). Toggling this is the clean A/B test
#             for "does constraining the atmosphere degrade the gravity fit?".
# T1BAR_OBS  : observed 1-bar temperature (K). Lindal (1992) Voyager 2 Neptune.
# SIGMA_T1BAR: 1-sigma uncertainty (K). Lindal quotes ±2 K; we broaden to 10 K
#             to absorb the He/H2 retrieval systematic.  See methods notes.
USE_T1BAR      = True
T1BAR_OBS      = 72.0    # K   — Neptune 1-bar temperature (Lindal 1992)
SIGMA_T1BAR    = 5.0    # K   — broadened from ±2 K obs; absorbs He/H2 systematic
P1BAR_PA       = 1.0e5   # Pa  — the pressure level at which to evaluate T

# =============================================================================
#  INTRINSIC LUMINOSITY (Lint) CONSTRAINT
# =============================================================================
# Optionally include the observed Neptune intrinsic luminosity as a Gaussian
# term.  Lint is returned by Planet_LAB as res.Lint (Watts).  Adding this
# constraint and freeing Ra (envelope Rayleigh number, see FIXED block) keeps
# the degrees of freedom the same (+1 observable, +1 parameter).
#
# USE_LINT  : master switch.  When True the term is added to the cost function
#             AND Ra is sampled (the 3rd theta dimension, in linear units).
# LINT_OBS  : 3.3e15 W is the canonical Neptune intrinsic luminosity (Pearl &
#             Sromovsky 1991, Pearl & Conrath 1991).
# SIGMA_LINT: 0.35e15 W (~10% systematic; covers radiometric calibration drift).
USE_LINT       = True
LINT_OBS       = 3.3e15   # W   — Neptune intrinsic luminosity (Pearl+ 1991)
SIGMA_LINT     = 0.35e15  # W   — 1-sigma (~ 10% systematic uncertainty)

# =============================================================================
#  MELT H2 COMPOSITION PENALTY
# =============================================================================
# Penalize solutions where the melt H2 weight fraction exceeds the crest of
# the binodal (~5.38 wt%).  We use a one-sided Gaussian penalty that activates
# only when massfracH2_core (in wt%) exceeds MELT_H2_MAX_WTP.
#
# MELT_H2_MAX_WTP : soft upper limit on melt H2 content in wt%
#                   Set to 6.0 to be slightly liberal relative to crest at 5.38 wt%
# SIGMA_MELT_H2   : penalty width in wt%.
#                   1.0 wt% gives strong penalty by ~7-8 wt%.
#                   Increase to 2.0 for a softer transition.
MELT_H2_MAX_WTP = 12.0  # wt% — soft ceiling on melt H2 content, ~ 0.87 by mole
SIGMA_MELT_H2   = 1.0   # wt% — penalty width

# =============================================================================
#  NOTES ON THE CHOICE OF CONSTANTS (rationale and known subtleties)
# =============================================================================
# R_eq vs R_vol.  Planet_LAB integrates hydrostatic equilibrium on spherical
#   shells, producing R_1bar that is volumetric in physical content.  CMS
#   then receives a0 = R_1bar and returns J_n normalized to that radius.
#   The cost function applies a first-order oblate bulge correction inside
#   compute_harmonics caller (see run_planet_and_J2): f ≈ (3 J2 + q)/2 and
#   R_eq_model = R_1bar (1 + f/3).  Both J_pred (renormalized via factor
#   (R_1bar/R_eq_model)^n) and R_eq_model are then compared to the published
#   wind-corrected J2_OBS / J4_OBS (at R_eq) and R_eq_obs.  σ_R is broadened
#   from the Lindal observational ±4 km to the R_vol-flavored 19 km,
#   reflecting that the model's degree of freedom is the volumetric R; the
#   broader σ_R absorbs the spherical-model systematic.
#
# SIGMA_T1BAR = 10 K.  Lindal (1992) Voyager radio occultation gives Neptune's
#   1-bar T at ±2 K *conditional on the assumed He/H2 ratio*, which sets the
#   mean molecular weight in the refractivity inversion.  Realistic He/H2
#   uncertainty propagates to several K, so we broaden to 10 K.  See methods
#   text; use SIGMA_T1BAR = 2 K only if you also pin He/H2.
#
# SIGMA_C = 2% of C_OBS.  This is a *modeling-spread* uncertainty (Movshovitz
#   & Fortney 2022 range of acceptable interior models), not an observational
#   value — C/MR² is not directly measured for the ice giants.  Treat the
#   constraint as a self-consistency target rather than a measurement.


# =============================================================================
#  SIGMA MODE FOR J2/J4 COST FUNCTION
# =============================================================================
# SIGMA_MODE : 'obs' — use observational measurement uncertainty (J2_err, J4_err).
#                      Justified when the model fits J2 to <1%; the calibration
#                      formula is not the limiting factor.  Default and recommended
#                      for production MCMC runs.
#              'cal' — use LOO-weighted in-sample calibration residual.
#                      More conservative; use to assess sensitivity of results.
SIGMA_MODE = 'obs'

_sig_J2_cal, _sig_J4_cal, _frac_J2, _frac_J4 = J2mod.calibration_sigma(
    'Neptune', sigma_mode=SIGMA_MODE)

# J2_SIGMA_OBS / J4_SIGMA_OBS sourced from SS dict via calibration_sigma('obs')
# to keep a single source of truth in calculate_J2.py
_obs_sig      = J2mod.calibration_sigma('Neptune', sigma_mode='obs')
J2_SIGMA_OBS  = _obs_sig[0]   # = SS['Neptune']['J2_err'] = 4.14e-6
J4_SIGMA_OBS  = _obs_sig[1]   # = SS['Neptune']['J4_err'] = 10.762e-6
J2_SIGMA      = _sig_J2_cal
J4_SIGMA      = _sig_J4_cal



# =============================================================================
#  CMS J4 SIGMA OVERRIDE
# =============================================================================
# When J2_METHOD == 'cms', the CMS J4 is a physically meaningful independent
# prediction (unlike the empirical method where J4 is derived from J2).
# Setting CMS_J4_SIGMA_OVERRIDE tightens the J4 constraint to give it more
# weight in the likelihood.  Set to None to use the same J4_SIGMA as the
# empirical method.  A value of 3.5e-6 (10% of J4_obs) gives J4 roughly
# equal weight to J2 in the CMS likelihood. 4.0e-6 accounts for dynamic uncertainties.
CMS_J4_SIGMA_OVERRIDE = 4.0e-6   # set to None to disable

# Effective J4 sigma used in cost function — pre-resolved here so it is
# available to the startup print, _WORKER_CFG, and write_results consistently.
_j4_sigma_effective = (CMS_J4_SIGMA_OVERRIDE
                       if J2_METHOD == 'cms' and CMS_J4_SIGMA_OVERRIDE is not None
                       else J4_SIGMA)

# ── Startup sigma report ──────────────────────────────────────────────────────
print(f"Sigma mode: {SIGMA_MODE!r}  |  J2 method: {J2_METHOD!r}")
print(f"  J2: obs={J2_SIGMA_OBS*1e6:.3f}e-6  "
      f"cal={_sig_J2_cal*1e6:.3f}e-6  "
      f"using={J2_SIGMA*1e6:.3f}e-6  "
      f"({_frac_J2*100:.3f}%)")
print(f"  J4: obs={J4_SIGMA_OBS*1e6:.3f}e-6  "
      f"cal={_sig_J4_cal*1e6:.3f}e-6  "
      f"using={J4_SIGMA*1e6:.3f}e-6  "
      f"({_frac_J4*100:.3f}%)")
if J2_METHOD == 'cms' and CMS_J4_SIGMA_OVERRIDE is not None:
    print(f"  J4 CMS override ACTIVE: sigma overridden to "
          f"{CMS_J4_SIGMA_OVERRIDE*1e6:.3f}e-6  "
          f"(was {J4_SIGMA*1e6:.3f}e-6)")
    print(f"  >>> EFFECTIVE J4 sigma in cost function: "
          f"{_j4_sigma_effective*1e6:.3f}e-6 <<<")
else:
    print(f"  J4 CMS override: not active")
    print(f"  >>> EFFECTIVE J4 sigma in cost function: "
          f"{_j4_sigma_effective*1e6:.3f}e-6 <<<")

# =============================================================================
#  FREE PARAMETERS  —  prior bounds
# =============================================================================
#  theta = [surface_P_GPa, bulk_wt_H2_percent]
P_MIN,  P_MAX  = 3.0,  12.0     # GPa — binodal surface pressure, 10.3 max
H2_MIN, H2_MAX = 8.0,  17.0    # wt%  — bulk H2 content
# Ra (envelope Rayleigh number) sampled in LINEAR space because factors of
# two matter physically. Tight range around the canonical 1e12; widen if
# the chain pegs at a boundary.
RA_MIN, RA_MAX = 5.0e11, 5.0e12

# =============================================================================
#  FIXED PARAMETERS  (edit to match your preferred run settings)
# =============================================================================
FIXED = dict(
    Mp_target_kg          = M_NEPTUNE_KG,
    cmf_core_metal        = 0.33,
    n_radial_steps        = 120000,
    RB_over_rmax          = 13.0,
    plot_radius_limit_RE  = 4.0,
    Teq_K                 = 47.0,         # Neptune equilibrium T  (K)
    mw_main_amu           = 2.3,
    mw_silicate_amu       = 100.0,
    gamma_gas             = 1.4,
    r_main_m              = 289e-12,
    rho0_silicate_gcc     = 0.0,
    metal_mass_deficit    = 0.01,
    structure_layers      = [1],       # One phase
    DeltaTrad_K           = 1.0,
    # Ra (envelope Rayleigh number) is NOT here — it has been promoted to a
    # search parameter (theta[2] = Ra, in linear units).  See RA_MIN/MAX below.
    tau_critical_chord    = 1.0,
    alpha_smooth_gradT    = 1.0,
    Inhibit               = True,
    write_files           = False,        # suppress all file I/O during MCMC
    output_dir            = ".",
)

# =============================================================================
#  EMCEE SETTINGS
# =============================================================================
N_DIM       = 3           # 3 free parameters: surface_P_GPa, bulk_H2_wt%, Ra
N_WALKERS   = 24           # 12 must be even and >= 2*N_DIM; 16 is a good start
N_STEPS     = 80         # 40 steps per walker (increase for production runs)
N_BURN      = 5         # burn-in steps to discard when making plots
CHECKPOINT  = 'neptune_mcmc_chain.h5'
THIN        = 1           # thinning factor for chain storage
# Number of parallel processes — set to the number of physical cores you want
# to use.  None = use all available cores.
import os as _os
N_CORES     = 12        # e.g. 4, 8, or None for all cores
# Stretch move scale parameter.  Target acceptance fraction ~0.2-0.5.
# If acceptance is too high (>0.5), increase a.  Too low (<0.2), decrease a.
STRETCH_A   = 2.5  # Broader search volume, set to 4.0

# =============================================================================
#  BEST GUESS for search space variables
# =============================================================================
P0_INIT       = 5.28    # GPa  (between old gravity peak ~6.8 and T1bar-likely ~5.0-5.5)
H2_INIT       = 12.77   # wt%  (slightly below old gravity-only value 13.4)
RA_INIT       = 1.0e12 # canonical Ra value (the previous FIXED choice)

# =============================================================================
#  GRAVITY HARMONICS CALCULATOR
#  Single function used by both run_planet() and log_prob() so the method
#  switch is applied consistently in both places.
# =============================================================================

def compute_harmonics(r_int, rho_int, R_1bar, planet='Neptune',
                      method=None, cms_n_layers=None, cms_xlayers=None):
    """
    Compute J2, J4, C/MR², and q_rot from a density profile.

    Parameters
    ----------
    r_int, rho_int : ndarray
        Radial profile trimmed to the 1-bar radius, ascending r.
    R_1bar : float
        1-bar equatorial radius in metres.
    planet : str
        'Neptune' or 'Uranus'.
    method : str or None
        'empirical' or 'cms'.  None falls back to the module-level J2_METHOD.
    cms_n_layers, cms_xlayers : int or None
        Override module-level CMS settings.

    Returns
    -------
    J2_pred, J4_pred, C_pred, q_rot  — all floats, or raises on failure.
    """
    import math
    if method is None:
        method = J2_METHOD
    if cms_n_layers is None:
        cms_n_layers = CMS_N_LAYERS
    if cms_xlayers is None:
        cms_xlayers = CMS_XLAYERS

    M, I   = J2mod.integrate_mass_moi(r_int, rho_int)
    C_pred = I / (M * R_1bar**2)
    Omega  = 2.0 * math.pi / J2mod.SS[planet]['P_rot']
    q_rot  = Omega**2 * R_1bar**3 / (J2mod.G * M)

    if method == 'empirical':
        coef = J2mod.build_calibration()
        J2_pred, J4_pred = J2mod.predict_J2_J4(C_pred, q_rot, coef)

    elif method == 'cms':
        # Import here so the module loads fine even if CMSPlanet is absent
        # when running in empirical mode.
        import warnings
        try:
            from run_cms_single import run_cms, PLANET_DATA
        except ImportError:
            raise ImportError(
                "run_cms_single.py not found — needed for J2_METHOD='cms'. "
                "Place it in the same directory as neptune_mcmc.py."
            )
        # run_cms expects centre-out arrays (ascending r), which is what we have.
        # Suppress the bracket-failure warnings during MCMC to keep output clean.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cms_res = run_cms(
                planet, r_int, rho_int,
                R_1bar   = R_1bar,
                N_layers = cms_n_layers,
                xlayers  = cms_xlayers,
                verbose  = False,
            )
        J2_pred = cms_res['J2']
        J4_pred = cms_res['J4']

    else:
        raise ValueError(f"J2_METHOD must be 'empirical' or 'cms', got {method!r}")

    return J2_pred, J4_pred, C_pred, q_rot


# =============================================================================
#  FORWARD MODEL CALL
# =============================================================================

def run_planet(surface_P_GPa, bulk_wt_H2_percent, Ra=1.0e12, walker_id=0):
    """
    Call Planet_LAB and return
        (J2_pred, J4_pred, C_pred, R_eq_model, Lint_pred, T1bar_pred).
    Returns six Nones on any exception.  T1bar_pred is the model temperature
    interpolated at the 1-bar pressure level using the atmosphere profile;
    it is np.nan if the profile is unavailable or 1 bar lies outside the
    profile range.  All stdout/stderr from Planet_LAB is suppressed.
    """
    import io, contextlib
    suffix = f"_mcmc_w{walker_id:03d}"
    _sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
            res = Planet_LAB(
                bulk_wt_H2_percent = bulk_wt_H2_percent,
                surface_P_GPa      = surface_P_GPa,
                Ra                 = Ra,
                output_suffix      = suffix,
                **FIXED
            )
    except Exception:
        return None, None, None, None, None, None

    if res is None:
        return None, None, None, None, None, None

    # Extract profiles
    profiles  = res.profiles or {}
    r_planet  = profiles.get("r_planet_m")
    rho_planet= profiles.get("rho_planet_kgm3")
    P_planet  = profiles.get("P_planet_Pa")
    T_planet  = profiles.get("T_planet_K")    # for diagnostic T_1bar report

    if r_planet is None or rho_planet is None or P_planet is None:
        return None, None, None, None, None, None
    if len(r_planet) < 10:
        return None, None, None, None, None, None

    # Convert arrays
    r_planet   = np.asarray(r_planet,   dtype=float)
    rho_planet = np.asarray(rho_planet, dtype=float)
    P_planet   = np.asarray(P_planet,   dtype=float)
    P_bar      = P_planet / 1e5

    # Ensure ascending r
    if r_planet[0] > r_planet[-1]:
        r_planet   = r_planet[::-1]
        rho_planet = rho_planet[::-1]
        P_bar      = P_bar[::-1]

    # Find 1-bar radius and truncate
    try:
        R_1bar, i_1bar = J2mod.find_1bar_radius(r_planet, P_bar)
    except Exception:
        return None, None, None, None, None, None

    r_int   = r_planet[:i_1bar + 1]
    rho_int = rho_planet[:i_1bar + 1]

    if len(r_int) < 5:
        return None, None, None, None, None, None

    # Integrate mass and MOI
    M, I   = J2mod.integrate_mass_moi(r_int, rho_int)
    C_pred = I / (M * R_1bar**2)

    # J2, J4 via selected method
    try:
        J2_pred, J4_pred, C_pred, q_rot = compute_harmonics(
            r_int, rho_int, R_1bar, planet='Neptune')
    except Exception:
        return None, None, None, None, None, None

    # ── Reference-radius renormalization (R_1bar → R_eq_model) ────────────────
    # The structure integration is spherical, so R_1bar is volumetric in
    # physical content.  CMS receives a0 = R_1bar and therefore returns J_n
    # normalized to that radius.  We convert both R and J_n to the implied
    # equatorial radius using the first-order oblate-spheroid bulge formula
    # f ≈ (3 J2 + q_rot)/2, R_eq ≈ R_1bar (1 + f/3).  After this block,
    # everything is at R_eq:  J_pred are at R_eq_model and can be compared
    # directly to J2_OBS / J4_OBS (which are at published R_eq); R_eq_model
    # is what χ²_R compares against R_OBS (= R_eq published).  R_1bar is
    # retained as an internal scratch quantity (= R_vol of the model).
    f_model    = 0.5 * (3.0 * J2_pred + q_rot)
    R_eq_model = R_1bar * (1.0 + f_model / 3.0)
    J2_pred    = J2_pred * (R_1bar / R_eq_model)**2
    J4_pred    = J4_pred * (R_1bar / R_eq_model)**4

    # Extract Lint (W) — diagnostic post-run reporting
    Lint_pred = getattr(res, 'Lint', None)

    # Interpolate model T at 1 bar from the atmosphere profile.  Mirrors the
    # extraction used inside log_prob (lines ~716-731) so the reporter prints
    # the same T_1bar value that entered the cost function.  Falls back to nan
    # if the profile is missing or 1 bar lies outside the profile range; in
    # that case the chain itself would have rejected this point as -inf.
    T1bar_pred = float('nan')
    if T_planet is not None and P_planet is not None:
        T_arr = np.asarray(T_planet, dtype=float)
        P_arr = np.asarray(P_planet, dtype=float)
        if T_arr.shape == P_arr.shape and T_arr.size >= 2:
            order  = np.argsort(P_arr)
            P_sort = P_arr[order]
            T_sort = T_arr[order]
            if P_sort[0] <= P1BAR_PA <= P_sort[-1]:
                T1bar_pred = float(np.interp(P1BAR_PA, P_sort, T_sort))

    return J2_pred, J4_pred, C_pred, R_eq_model, Lint_pred, T1bar_pred


# =============================================================================
#  LOG-PROBABILITY  (self-contained for clean pickling across spawn workers)
# =============================================================================
# All constants needed by workers are bundled into a single dict that is
# passed as a module-level variable.  This avoids pickle failures that cause
# emcee to silently fall back to serial execution with spawn multiprocessing.

_WORKER_CFG = dict(
    P_MIN=P_MIN, P_MAX=P_MAX, H2_MIN=H2_MIN, H2_MAX=H2_MAX,
    RA_MIN=RA_MIN, RA_MAX=RA_MAX,
    J2_OBS=J2_OBS, J2_SIGMA=J2_SIGMA,
    J4_OBS=J4_OBS, J4_SIGMA=J4_SIGMA,
    J4_SIGMA_EFFECTIVE=_j4_sigma_effective,   # pre-resolved override — used in cost fn
    C_OBS=C_OBS, SIGMA_C=SIGMA_C,
    R_OBS=R_OBS, SIGMA_R=SIGMA_R,
    MELT_H2_MAX_WTP=MELT_H2_MAX_WTP,
    SIGMA_MELT_H2=SIGMA_MELT_H2,
    USE_T1BAR=USE_T1BAR,
    T1BAR_OBS=T1BAR_OBS,
    SIGMA_T1BAR=SIGMA_T1BAR,
    USE_LINT=USE_LINT,
    LINT_OBS=LINT_OBS,
    SIGMA_LINT=SIGMA_LINT,
    P1BAR_PA=P1BAR_PA,
    FIXED=FIXED,
    J2_METHOD=J2_METHOD,
    CMS_N_LAYERS=CMS_N_LAYERS,
    CMS_XLAYERS=CMS_XLAYERS,
    CMS_J4_SIGMA_OVERRIDE=CMS_J4_SIGMA_OVERRIDE,
)


def log_prob(theta):
    """
    Fully self-contained log-probability function.
    All imports are local so this function pickles cleanly under spawn.
    """
    import io, contextlib, math
    import numpy as np

    cfg     = _WORKER_CFG
    P_GPa, H2_pct, Ra = theta

    # ── Prior ────────────────────────────────────────────────────────────────
    if not (cfg['P_MIN'] < P_GPa < cfg['P_MAX'] and
            cfg['H2_MIN'] < H2_pct < cfg['H2_MAX'] and
            cfg['RA_MIN'] < Ra < cfg['RA_MAX']):
        return -np.inf

    # ── Forward model ────────────────────────────────────────────────────────
    from Planet_Lab_v33_fnc import Planet_LAB
    import calculate_J2 as J2mod

    _sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
            res = Planet_LAB(
                bulk_wt_H2_percent = H2_pct,
                surface_P_GPa      = P_GPa,
                Ra                 = Ra,
                output_suffix      = '_mcmc',
                **cfg['FIXED']
            )
    except Exception:
        return -np.inf

    if res is None:
        return -np.inf

    # Extract massfracH2_core — melt H2 weight fraction (0–1 scale)
    massfracH2_core = getattr(res, 'massfracH2_core', None)
    if massfracH2_core is None or not np.isfinite(massfracH2_core):
        return -np.inf

    # Extract Lint (intrinsic luminosity, W).  May be None if the model
    # didn't reach atmosphere integration; treat that as a model failure.
    Lint_pred = getattr(res, 'Lint', None)
    if cfg.get('USE_LINT', False):
        if Lint_pred is None or not np.isfinite(Lint_pred) or Lint_pred <= 0.0:
            return -np.inf

    profiles   = res.profiles or {}
    r_planet   = profiles.get("r_planet_m")
    rho_planet = profiles.get("rho_planet_kgm3")
    P_planet   = profiles.get("P_planet_Pa")
    T_planet   = profiles.get("T_planet_K")   # for the optional 1-bar T constraint

    if r_planet is None or rho_planet is None or P_planet is None:
        return -np.inf
    if len(r_planet) < 10:
        return -np.inf

    r_planet   = np.asarray(r_planet,   dtype=float)
    rho_planet = np.asarray(rho_planet, dtype=float)
    P_bar      = np.asarray(P_planet,   dtype=float) / 1e5

    if r_planet[0] > r_planet[-1]:
        r_planet   = r_planet[::-1]
        rho_planet = rho_planet[::-1]
        P_bar      = P_bar[::-1]

    try:
        R_1bar, i_1bar = J2mod.find_1bar_radius(r_planet, P_bar)
    except Exception:
        return -np.inf

    r_int   = r_planet[:i_1bar + 1]
    rho_int = rho_planet[:i_1bar + 1]

    if len(r_int) < 5:
        return -np.inf

    # ── J2, J4, C/MR², and R ─────────────────────────────────────────────────
    try:
        import warnings
        # Import method settings from parent module scope via cfg
        _method      = cfg.get('J2_METHOD',      'empirical')
        _cms_n       = cfg.get('CMS_N_LAYERS',   256)
        _cms_xl      = cfg.get('CMS_XLAYERS',    -1)

        M, I   = J2mod.integrate_mass_moi(r_int, rho_int)
        C_pred = I / (M * R_1bar**2)
        Omega  = 2.0 * math.pi / J2mod.SS['Neptune']['P_rot']
        q_rot  = Omega**2 * R_1bar**3 / (J2mod.G * M)

        if _method == 'empirical':
            coef = J2mod.build_calibration()
            J2_pred, J4_pred = J2mod.predict_J2_J4(C_pred, q_rot, coef)
        elif _method == 'cms':
            from run_cms_single import run_cms
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cms_res = run_cms(
                    'Neptune', r_int, rho_int,
                    R_1bar   = R_1bar,
                    N_layers = _cms_n,
                    xlayers  = _cms_xl,
                    verbose  = False,
                )
            J2_pred = cms_res['J2']
            J4_pred = cms_res['J4']
        else:
            return -np.inf
    except Exception:
        return -np.inf

    # ── Reference-radius renormalization (R_1bar → R_eq_model) ────────────────
    # The structure integration is spherical, so R_1bar is volumetric in
    # physical content.  CMS (or the empirical calibration) returned J_n at
    # reference radius R_1bar.  We convert both R and J_n to the implied
    # equatorial radius using the first-order oblate-spheroid bulge formula
    # f ≈ (3 J2 + q_rot)/2, R_eq ≈ R_1bar (1 + f/3).  After this block:
    #   R_eq_model is what χ²_R compares to R_OBS (= published R_eq)
    #   J2_pred, J4_pred are renormalized to R_eq_model (matching published)
    # R_1bar is kept as an internal diagnostic (= R_vol of the model).
    f_model    = 0.5 * (3.0 * J2_pred + q_rot)
    R_eq_model = R_1bar * (1.0 + f_model / 3.0)
    J2_pred    = J2_pred * (R_1bar / R_eq_model)**2
    J4_pred    = J4_pred * (R_1bar / R_eq_model)**4

    if not (np.isfinite(J2_pred) and np.isfinite(J4_pred) and
            np.isfinite(C_pred)  and np.isfinite(R_eq_model)):
        return -np.inf

    # ── Likelihood ───────────────────────────────────────────────────────────
    # Use the pre-resolved effective J4 sigma (accounts for CMS override).
    # This is computed once at startup and stored in _WORKER_CFG so all
    # workers use exactly the same value that was reported at startup.
    _j4_sigma = cfg['J4_SIGMA_EFFECTIVE']

    chi2_J2 = ((J2_pred    - cfg['J2_OBS']) / cfg['J2_SIGMA'])**2
    chi2_J4 = ((J4_pred    - cfg['J4_OBS']) / _j4_sigma)**2
    chi2_C  = ((C_pred     - cfg['C_OBS'])  / cfg['SIGMA_C'])**2
    chi2_R  = ((R_eq_model - cfg['R_OBS'])  / cfg['SIGMA_R'])**2

    # ── Melt H2 composition penalty ──────────────────────────────────────────
    # One-sided Gaussian penalty: activates only when melt H2 exceeds
    # MELT_H2_MAX_WTP (in wt%).  massfracH2_core is on 0-1 scale so multiply by 100.
    melt_H2_wtp     = massfracH2_core * 100.0
    _melt_H2_max    = cfg.get('MELT_H2_MAX_WTP', 6.0)
    _melt_H2_sigma  = cfg.get('SIGMA_MELT_H2',   1.0)
    excess          = melt_H2_wtp - _melt_H2_max
    chi2_meltH2     = (max(0.0, excess) / _melt_H2_sigma)**2

    # ── Optional 1-bar atmospheric temperature constraint ─────────────────────
    # Interpolate the model temperature at 1 bar from the returned atmosphere
    # profile (T_planet_K vs P_planet_Pa) and add a Gaussian likelihood term.
    # Disabled cleanly when USE_T1BAR is False (term omitted from likelihood and
    # from the reported dof).
    chi2_T1bar = 0.0
    T_1bar     = np.nan
    if cfg.get('USE_T1BAR', False):
        if T_planet is None:
            return -np.inf
        T_arr = np.asarray(T_planet,  dtype=float)
        P_arr = np.asarray(P_planet,  dtype=float)   # Pa, original order
        if T_arr.shape != P_arr.shape or T_arr.size < 2:
            return -np.inf
        # interpolate T at the 1-bar level; np.interp needs ascending P
        order  = np.argsort(P_arr)
        P_sort = P_arr[order]
        T_sort = T_arr[order]
        P1     = float(cfg.get('P1BAR_PA', 1.0e5))
        # require the 1-bar level to be inside the profile (no extrapolation)
        if not (P_sort[0] <= P1 <= P_sort[-1]):
            return -np.inf
        T_1bar = float(np.interp(P1, P_sort, T_sort))
        if not np.isfinite(T_1bar):
            return -np.inf
        chi2_T1bar = ((T_1bar - cfg['T1BAR_OBS']) / cfg['SIGMA_T1BAR'])**2

    # ── Optional Lint (intrinsic luminosity) constraint ──────────────────────
    # Gaussian penalty against the observed Neptune intrinsic luminosity.
    # Disabled cleanly when USE_LINT is False.
    chi2_Lint = 0.0
    if cfg.get('USE_LINT', False):
        chi2_Lint = ((Lint_pred - cfg['LINT_OBS']) / cfg['SIGMA_LINT'])**2

    R_EARTH = 6371000.0    # m — Earth radius for diagnostic print
    _t1_str = (f"T1bar={T_1bar:.1f}K (chi2_T={chi2_T1bar:.2f}) | "
               if cfg.get('USE_T1BAR', False) else "")
    _lint_str = (f"Lint={Lint_pred:.2e}W (chi2_L={chi2_Lint:.2f}) | "
                 if cfg.get('USE_LINT', False) and Lint_pred is not None
                 else "")
    print(f"  [worker] P={P_GPa:.2f} H2={H2_pct:.2f} Ra={Ra:.3e} | "
          f"J2={J2_pred*1e6:.1f}e-6 J4={J4_pred*1e6:.2f}e-6 "
          f"C={C_pred:.4f} R={R_eq_model/R_EARTH:.4f}Earth | "
          f"meltH2={melt_H2_wtp:.2f}wt% (chi2_melt={chi2_meltH2:.2f}) | "
          f"{_t1_str}"
          f"{_lint_str}"
          f"lnp={-0.5*(chi2_J2+chi2_J4+chi2_C+chi2_R+chi2_meltH2+chi2_T1bar+chi2_Lint):.2f}")

    return -0.5 * (chi2_J2 + chi2_J4 + chi2_C + chi2_R + chi2_meltH2 + chi2_T1bar + chi2_Lint)


# =============================================================================
#  INITIALISE WALKERS
# =============================================================================

def initial_positions(n_walkers, rng=None, P0=P0_INIT, H2_0=H2_INIT,
                      Ra_0=RA_INIT):
    """
    Start walkers in a small ball near a reasonable starting point.
    Centre is set near your best run so far: P~6.5 GPa, H2~3 wt%.
    Edit P0, H2_0 to your current best-fit values.

    Ra is sampled in LINEAR space (factors of two matter physically); ball
    width is 3e11 around the canonical 1e12, so the initial ensemble spans
    roughly [4e11, 1.6e12] -- well within the [5e11, 5e12] prior.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Small fractional scatter (5%) around centre
    # Additive scatter in the natural units of each parameter, NOT fractional.
    # The previous Neptune T1bar-aware MCMC showed the chain trying to migrate
    # from P~6.8 (gravity-only peak) to P~5.0-5.5 (T1bar-favored region), and
    # spending many steps doing so with a tight initial ball. Centering the
    # ball below the old peak and giving it ~1 GPa of width lets the ensemble
    # immediately span both possibilities and use DEMove's ridge-aligned
    # proposals efficiently from step 1.
    P_ball  = P0   + 1.0 * rng.standard_normal(n_walkers)   # GPa, sigma=1.0
    H2_ball = H2_0 + 0.7 * rng.standard_normal(n_walkers)   # wt%, sigma=0.7
    Ra_ball = Ra_0 + 3.0e11 * rng.standard_normal(n_walkers)  # sigma 3e11

    # Clip to prior bounds
    P_ball  = np.clip(P_ball,  P_MIN  + 0.01,    P_MAX  - 0.01)
    H2_ball = np.clip(H2_ball, H2_MIN + 0.01,    H2_MAX - 0.01)
    Ra_ball = np.clip(Ra_ball, RA_MIN + 1.0e9,   RA_MAX - 1.0e9)

    return np.column_stack([P_ball, H2_ball, Ra_ball])


# =============================================================================
#  PLOTTING HELPERS
# =============================================================================

LABELS = [r'$P_{\rm binodal}$ (GPa)', r'Bulk H$_2$ (wt\%)', r'$\mathrm{Ra}$']

def plot_chains(sampler, burn=N_BURN, savepath='neptune_mcmc_chains'):
    chain = sampler.get_chain()             # (steps, walkers, ndim)
    fig, axes = plt.subplots(N_DIM, 1, figsize=(7, 2.0 * N_DIM), sharex=True)
    # axes is an array when N_DIM>1; ensure iterable
    if N_DIM == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], color='C0', alpha=0.3, lw=0.5)
        ax.axvline(burn, color='C3', lw=1.0, ls='--', label='burn-in' if i==0 else None)
        ax.set_ylabel(LABELS[i])
    axes[-1].set_xlabel('Step')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('.png', '.pdf'):
        fig.savefig(savepath + ext)
    plt.close(fig)
    print(f"  Chain plot → {savepath}.png/.pdf")


def plot_corner(flat_samples, savepath='neptune_mcmc_corner'):
    # Check each parameter has dynamic range (needed for corner to work)
    stds = np.std(flat_samples, axis=0)
    dead = np.where(stds == 0)[0]
    if len(dead) > 0:
        names = [LABELS[i] for i in dead]
        print(f"  Corner plot skipped: parameter(s) {names} have no variance.")
        print(f"  This is normal for very short chains — run more steps.")
        return

    # Compute explicit ranges with a small buffer to avoid edge issues
    ranges = [(col.min() - 0.1*s, col.max() + 0.1*s)
              for col, s in zip(flat_samples.T, stds)]

    truths = np.median(flat_samples, axis=0)
    fig = corner.corner(
        flat_samples,
        labels=LABELS,
        truths=truths,
        truth_color='C3',
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={'fontsize': 10},
        label_kwargs={'fontsize': 11},
        quantiles=[0.16, 0.50, 0.84],
        range=ranges,
    )
    for ext in ('.png', '.pdf'):
        fig.savefig(savepath + ext)
    plt.close(fig)
    print(f"  Corner plot → {savepath}.png/.pdf")


def plot_predictions(flat_samples, savepath='neptune_mcmc_predictions',
                     n_samples=200):
    """
    Draw n_samples from the posterior, re-run the model for each, and plot
    the distribution of predicted J2 and C/MR² against observed values.
    """
    rng  = np.random.default_rng(0)
    idx  = rng.choice(len(flat_samples), size=min(n_samples, len(flat_samples)),
                      replace=False)
    J2s, Cs = [], []
    print(f"  Computing {len(idx)} posterior predictive samples...")
    for k, i in enumerate(idx):
        # Handle both 2D and 3D samples robustly (older chains may be 2D)
        row = flat_samples[i]
        if len(row) >= 3:
            P_GPa, H2, Ra = row[0], row[1], row[2]
        else:
            P_GPa, H2 = row[0], row[1]
            Ra = 1.0e12
        J2p, J4p, Cp, _, _ = run_planet(P_GPa, H2, Ra=Ra)
        if J2p is not None and np.isfinite(J2p) and np.isfinite(Cp):
            J2s.append(J2p * 1e6)
            Cs.append(Cp)
        if (k+1) % 20 == 0:
            print(f"    {k+1}/{len(idx)}")

    J2s, Cs = np.array(J2s), np.array(Cs)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    axes[0].hist(J2s, bins=20, color='C0', alpha=0.7, density=True)
    axes[0].axvline(J2_OBS*1e6, color='C3', lw=1.5, ls='--',
                    label=f'Neptune obs = {J2_OBS*1e6:.1f}')
    axes[0].axvspan((J2_OBS - J2_SIGMA)*1e6, (J2_OBS + J2_SIGMA)*1e6,
                    alpha=0.15, color='C3', label=r'±1σ obs')
    axes[0].set_xlabel(r'$J_2 \times 10^6$')
    axes[0].set_ylabel('Posterior density')
    axes[0].legend()

    axes[1].hist(Cs, bins=20, color='C0', alpha=0.7, density=True)
    axes[1].axvline(C_OBS, color='C3', lw=1.5, ls='--',
                    label=f'Neptune obs = {C_OBS:.4f}')
    axes[1].axvspan(C_OBS - SIGMA_C, C_OBS + SIGMA_C,
                    alpha=0.15, color='C3', label=r'±1σ assumed')
    axes[1].set_xlabel(r'$C/MR^2$')
    axes[1].legend()

    fig.tight_layout()
    for ext in ('.png', '.pdf'):
        fig.savefig(savepath + ext)
    plt.close(fig)
    print(f"  Prediction plot → {savepath}.png/.pdf")


# =============================================================================
#  RESULTS SUMMARY
# =============================================================================

def write_results(flat_samples, sampler=None, burn=0,
                  savepath='neptune_mcmc_results.txt'):
    # ── Posterior median and credible intervals ───────────────────────────────
    P_med,  P_lo,  P_hi  = np.percentile(flat_samples[:, 0], [50, 16, 84])
    H2_med, H2_lo, H2_hi = np.percentile(flat_samples[:, 1], [50, 16, 84])
    # 3rd dimension (Ra) — only present when N_DIM >= 3
    if flat_samples.shape[1] >= 3:
        Ra_med, Ra_lo, Ra_hi = np.percentile(flat_samples[:, 2], [50, 16, 84])
    else:
        Ra_med = Ra_lo = Ra_hi = None

    # ── Maximum likelihood solution ───────────────────────────────────────────
    if sampler is not None:
        log_probs = sampler.get_log_prob(discard=burn, flat=True, thin=THIN)
        best_idx  = np.argmax(log_probs)
        best_P    = flat_samples[best_idx, 0]
        best_H2   = flat_samples[best_idx, 1]
        if flat_samples.shape[1] >= 3:
            best_Ra      = flat_samples[best_idx, 2]
        else:
            best_Ra      = 1.0e12
        best_lnp  = log_probs[best_idx]
        # n_obs / n_free bookkeeping
        n_obs, n_free = 4, N_DIM    # J2, J4, C, R always present
        if USE_T1BAR:
            n_obs += 1              # + T at 1 bar
        if USE_LINT:
            n_obs += 1              # + Lint
        chi2_total   = -2.0 * best_lnp
        chi2_reduced = chi2_total / max(1, n_obs - n_free)

        # Re-run forward model at best-fit point to get predicted observables
        print(f"\n  Re-running forward model at best-fit point "
              f"(P={best_P:.4f} GPa, H2={best_H2:.4f} wt%, "
              f"Ra={best_Ra:.3e}) ...")
        (J2_best, J4_best, C_best, R_best, Lint_best,
         T1bar_best) = run_planet(best_P, best_H2, Ra=best_Ra)

        if J2_best is not None:
            R_EARTH = 6371000.0
            # Use CMS J4 sigma override if active, otherwise use J4_SIGMA
            _j4_sigma_eff = (CMS_J4_SIGMA_OVERRIDE
                             if J2_METHOD == 'cms'
                             and CMS_J4_SIGMA_OVERRIDE is not None
                             else J4_SIGMA)
            obs_lines = [
                "",
                "Best-fit predicted observables vs Neptune:",
                f"  {'':4} {'Predicted':>12} {'Observed':>12} {'Residual':>10} {'sigma':>8}",
                f"  {'J2':4} {J2_best*1e6:>11.2f}e-6 {J2_OBS*1e6:>11.2f}e-6 "
                f"{(J2_best-J2_OBS)*1e6:>+9.2f}e-6 "
                f"{(J2_best-J2_OBS)/J2_SIGMA:>+7.2f}σ",
                f"  {'J4':4} {J4_best*1e6:>11.2f}e-6 {J4_OBS*1e6:>11.2f}e-6 "
                f"{(J4_best-J4_OBS)*1e6:>+9.2f}e-6 "
                f"{(J4_best-J4_OBS)/_j4_sigma_eff:>+7.2f}σ",
                f"  {'C':4} {C_best:>12.4f} {C_OBS:>12.4f} "
                f"{(C_best-C_OBS):>+10.4f} "
                f"{(C_best-C_OBS)/SIGMA_C:>+7.2f}σ",
                f"  {'R':4} {R_best/1e3:>11.1f} km {R_OBS/1e3:>9.1f} km "
                f"{(R_best-R_OBS)/1e3:>+9.1f} km "
                f"{(R_best-R_OBS)/SIGMA_R:>+7.2f}σ",
            ]
            if USE_LINT and Lint_best is not None and np.isfinite(Lint_best):
                obs_lines.append(
                    f"  {'Lint':4} {Lint_best:>11.3e} W  {LINT_OBS:>11.3e} W  "
                    f"{(Lint_best-LINT_OBS):>+9.3e} W "
                    f"{(Lint_best-LINT_OBS)/SIGMA_LINT:>+7.2f}σ"
                )
            if USE_T1BAR and T1bar_best is not None and np.isfinite(T1bar_best):
                obs_lines.append(
                    f"  {'T1b':4} {T1bar_best:>11.2f} K  {T1BAR_OBS:>11.2f} K  "
                    f"{(T1bar_best-T1BAR_OBS):>+9.2f} K "
                    f"{(T1bar_best-T1BAR_OBS)/SIGMA_T1BAR:>+7.2f}σ"
                )
        else:
            obs_lines = ["", "  (forward model failed at best-fit point)"]

        best_lines = [
            "",
            "Maximum likelihood solution (best ln-p):",
            f"  surface_P_GPa   = {best_P:.4f}  GPa",
            f"  bulk_H2_wt_pct  = {best_H2:.4f}  wt%",
            f"  Ra              = {best_Ra:.3e}",
            f"  ln-p            = {best_lnp:.4f}",
            f"  chi2_total      = {chi2_total:.4f}",
            f"  chi2_reduced    = {chi2_reduced:.4f}  (dof = {n_obs - n_free})",
        ] + obs_lines
    else:
        best_lines = [
            "",
            "(sampler not available — max likelihood solution not computed)",
        ]

    # Build the posterior median lines.  The Ra line is only added when the
    # chain is 3D (N_DIM>=3) so the function remains usable with older 2D
    # chain files saved before Ra was promoted.
    median_lines = [
        f"  surface_P_GPa   = {P_med:.3f}  +{P_hi-P_med:.3f} / -{P_med-P_lo:.3f}",
        f"  bulk_H2_wt_pct  = {H2_med:.3f}  +{H2_hi-H2_med:.3f} / -{H2_med-H2_lo:.3f}",
    ]
    if Ra_med is not None:
        median_lines.append(
            f"  Ra              = {Ra_med:.3e}  "
            f"+{Ra_hi-Ra_med:.3e} / -{Ra_med-Ra_lo:.3e}"
        )

    obs_summary_lines = [
        "",
        "Observational targets and uncertainties used in cost function:",
        f"  sigma_mode = {SIGMA_MODE!r}",
        f"  J2    = {J2_OBS*1e6:.2f} × 10⁻⁶",
        f"    obs sigma    = {J2_SIGMA_OBS*1e6:.3f}e-6",
        f"    cal sys      = {_sig_J2_cal*1e6:.3f}e-6  ({_frac_J2*100:.3f}%)",
        f"    using sigma  = {J2_SIGMA*1e6:.3f}e-6",
        f"  J4    = {J4_OBS*1e6:.3f} × 10⁻⁶",
        f"    obs sigma    = {J4_SIGMA_OBS*1e6:.3f}e-6",
        f"    cal sys      = {_sig_J4_cal*1e6:.3f}e-6  ({_frac_J4*100:.3f}%)",
        f"    using sigma  = {J4_SIGMA*1e6:.3f}e-6  (pre-override)",
        f"    CMS override = "
        + (f"{CMS_J4_SIGMA_OVERRIDE*1e6:.3f}e-6  (active)"
           if J2_METHOD == 'cms' and CMS_J4_SIGMA_OVERRIDE is not None
           else "not active"),
        f"    effective sigma in cost function = "
        + f"{_j4_sigma_effective*1e6:.3f}e-6",
        f"  C/MR² = {C_OBS:.4f}  ± {SIGMA_C:.4f}  (assumed 2%)",
        f"  R_eq  = {R_OBS/1e3:.1f} ± {SIGMA_R/1e3:.1f}  km  (Lindal 1992; σ broadened)",
    ]
    if USE_T1BAR:
        obs_summary_lines.append(
            f"  T_1bar = {T1BAR_OBS:.2f} ± {SIGMA_T1BAR:.2f}  K  (Lindal 1992)"
        )
    if USE_LINT:
        obs_summary_lines.append(
            f"  Lint   = {LINT_OBS:.3e} ± {SIGMA_LINT:.3e}  W  (Pearl+ 1991)"
        )

    lines = [
        "Neptune MCMC Results",
        "=" * 50,
        "",
        "Posterior median and 1-sigma credible intervals:",
    ] + median_lines + best_lines + obs_summary_lines + [
        f"",
        f"Melt H2 composition penalty:",
        f"  ceiling  = {MELT_H2_MAX_WTP:.2f} wt%  (binodal crest ~ 5.38 wt%)",
        f"  sigma    = {SIGMA_MELT_H2:.2f} wt%  (one-sided Gaussian penalty)",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(savepath, 'w') as f:
        f.write(text + "\n")
    print(f"  Results → {savepath}")


# =============================================================================
#  WORKER INITIALISER
# =============================================================================

def _worker_init():
    """
    Called once in each worker process at pool startup.
    Pre-imports heavy modules so they are cached for subsequent log_prob calls.
    Sets matplotlib to non-interactive backend to avoid display deadlocks.
    """
    import os
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg')
    from Planet_Lab_v33_fnc import Planet_LAB  # noqa: F401
    import calculate_J2  # noqa: F401
    if J2_METHOD == 'cms':
        import run_cms_single  # noqa: F401


# =============================================================================
#  MAIN
# =============================================================================

def run_mcmc(resume=False):
    backend = emcee.backends.HDFBackend(CHECKPOINT)

    if resume and Path(CHECKPOINT).exists():
        print(f"Resuming from {CHECKPOINT}  "
              f"({backend.iteration} steps already done)")
        # Explicitly read last walker positions from checkpoint
        p0 = backend.get_last_sample().coords
        print(f"  Resuming from last positions, shape = {p0.shape}")
    else:
        if Path(CHECKPOINT).exists():
            os.remove(CHECKPOINT)
        backend.reset(N_WALKERS, N_DIM)
        p0 = initial_positions(N_WALKERS)
        print(f"Starting fresh MCMC: {N_WALKERS} walkers × {N_STEPS} steps")
        print(f"Free parameters: surface_P_GPa ∈ [{P_MIN}, {P_MAX}] GPa  |  "
              f"bulk_H2 ∈ [{H2_MIN}, {H2_MAX}] wt%  |  "
              f"Ra ∈ [{RA_MIN:.2e}, {RA_MAX:.2e}]")
        print(f"Atmospheric constraints: "
              f"T1bar {'ACTIVE ('+str(T1BAR_OBS)+'±'+str(SIGMA_T1BAR)+' K)' if USE_T1BAR else 'OFF'}, "
              f"Lint {'ACTIVE ('+f'{LINT_OBS:.2e}±{SIGMA_LINT:.2e} W'+')' if USE_LINT else 'OFF'}")
        print(f"Gravity/shape sigmas in likelihood: "
              f"J2={J2_SIGMA*1e6:.3f}e-6, "
              f"J4={_j4_sigma_effective*1e6:.3f}e-6"
              + (" [CMS override]" if (J2_METHOD == 'cms' and CMS_J4_SIGMA_OVERRIDE is not None) else "")
              + f", C={SIGMA_C:.4f}, R_eq={SIGMA_R/1e3:.1f} km "
              f"({SIGMA_R/R_OBS*100:.3f}% of R_OBS), "
              f"melt_H2={SIGMA_MELT_H2:.1f} wt%")

    # ── Parallel sampler using spawn-safe multiprocessing ────────────────────
    # We use 'spawn' (not 'fork') so each worker gets a clean Python process
    # with no inherited GUI/display state from the parent.
    # Planet_Lab_v33_fnc.py does not import PyQt5, so it is spawn-safe.
    import multiprocessing as _mp
    n_cores = N_CORES or _os.cpu_count()
    print(f"Using {n_cores} parallel processes")
    ctx = _mp.get_context('spawn')
    with ctx.Pool(processes=n_cores, initializer=_worker_init) as pool:
        sampler = emcee.EnsembleSampler(
            N_WALKERS, N_DIM, log_prob,
            backend=backend,
            pool=pool,
            # DEMove + DESnookerMove is the standard recipe for anisotropic /
            # narrow-valley posteriors. DEMove proposes moves along the
            # geometry of other walkers in the ensemble (so proposals
            # naturally align with the posterior ridge), DESnookerMove adds
            # occasional longer-range jumps to escape modes. The 0.8 / 0.2
            # weighting is the recommendation from emcee's documentation
            # and ter Braak's original DE-MCMC literature. Expect acceptance
            # fraction to settle in the 0.10-0.20 range, which is healthy
            # for DE moves (the 0.2-0.5 guideline is StretchMove-specific).
            #
            # gamma0 controls the DEMove proposal scale.  Default is
            # 2.38/sqrt(2*ndim) ~ 1.19 for 2 parameters (ter Braak's
            # optimal-mixing value for Gaussian posteriors).  Reduce this
            # if proposals are overshooting / hitting prior bounds.
            # Try 0.6 for half-size, 0.4 for ~third-size proposals.
            moves=[(emcee.moves.DEMove(gamma0=0.6),        0.8),
                   (emcee.moves.DESnookerMove(),           0.2)],
        )

        print("\nRunning sampler  (this will take a while)...")
        t0 = time.time()

        for sample in sampler.sample(p0, iterations=N_STEPS, thin_by=THIN,
                                     progress=False, skip_initial_state_check=True):
            step = sampler.iteration
            if step % 1 == 0:   # change to % 10 for production runs
                lp      = sampler.get_last_sample().log_prob
                acc     = np.mean(sampler.acceptance_fraction)
                best    = np.max(lp)
                elapsed = (time.time() - t0) / 60.0
                # best walker parameter values for quick inspection
                best_idx = np.argmax(lp)
                best_p   = sampler.get_last_sample().coords[best_idx]
                print(f"  step {step:4d} | accept={acc:.3f} | "
                      f"best ln-p={best:.2f} | "
                      f"P={best_p[0]:.2f} GPa  H2={best_p[1]:.2f} wt% | "
                      f"{elapsed:.1f} min elapsed")

        print(f"\nFinished in {(time.time()-t0)/60:.1f} min")
    return sampler


def make_plots(sampler_or_path=CHECKPOINT):
    if isinstance(sampler_or_path, str):
        backend  = emcee.backends.HDFBackend(sampler_or_path, read_only=True)
        # Read walker count and ndim directly from the saved chain so that
        # --plot works regardless of what N_WALKERS is set to in the script.
        _nwalkers, _ndim = backend.shape
        sampler  = emcee.EnsembleSampler(_nwalkers, _ndim, log_prob,
                                         backend=backend)
    else:
        sampler = sampler_or_path

    n_done = sampler.iteration
    burn   = min(N_BURN, n_done // 2)
    print(f"\nChain length: {n_done} steps  |  using burn-in = {burn}")

    flat = sampler.get_chain(discard=burn, flat=True, thin=THIN)
    print(f"Flat samples: {len(flat)}")

    plot_chains(sampler, burn=burn)
    plot_corner(flat)
    write_results(flat, sampler=sampler, burn=burn)

    # Auto-convergence check
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"\nAutocorrelation times: "
              f"P_GPa = {tau[0]:.1f} steps,  H2 = {tau[1]:.1f} steps")
        print(f"Effective samples: "
              f"P_GPa ≈ {len(flat)/tau[0]:.0f},  H2 ≈ {len(flat)/tau[1]:.0f}")
        if n_done < 50 * np.max(tau):
            print(f"WARNING: chain may not be converged "
                  f"(need ~{int(50*np.max(tau))} steps, have {n_done})")
    except emcee.autocorr.AutocorrError:
        print("Autocorrelation time could not be estimated "
              "(chain likely too short)")


# =============================================================================
#  ENTRY POINT
# =============================================================================

# ── Tee: mirror stdout to log file ───────────────────────────────────────────
class _Tee:
    """Mirrors all stdout to both terminal and a log file simultaneously."""
    def __init__(self, path, mode='w'):
        self._terminal = sys.stdout
        self._file     = open(path, mode, buffering=1)   # line-buffered
    def write(self, msg):
        self._terminal.write(msg)
        self._file.write(msg)
    def flush(self):
        self._terminal.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._terminal
        self._file.close()

# ── Spawn guard ──────────────────────────────────────────────────────────────
# Required on all platforms when using 'spawn' multiprocessing so that worker
# processes don't re-execute the top-level module code on import.
if __name__ == '__main__':
    import os as _os_main
    _os_main.environ['MPLBACKEND'] = 'Agg'   # must be set before matplotlib import
    import matplotlib as _mpl_main
    _mpl_main.use('Agg')
    import multiprocessing as _mp_guard
    _mp_guard.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(
        description='Neptune MCMC: fit J2 and C/MR² via Planet_LAB')
    parser.add_argument('--plot',   action='store_true',
                        help='Load existing chain and plot only (no new runs)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume sampling from saved checkpoint')
    parser.add_argument('--steps',  type=int, default=N_STEPS,
                        help=f'Number of MCMC steps (default {N_STEPS})')
    parser.add_argument('--walkers',type=int, default=N_WALKERS,
                        help=f'Number of walkers (default {N_WALKERS})')
    parser.add_argument('--burn',   type=int, default=N_BURN,
                        help=f'Burn-in steps to discard (default {N_BURN})')
    args = parser.parse_args()

    N_STEPS   = args.steps
    N_WALKERS = args.walkers
    N_BURN    = args.burn

    # ── Start logging to mcmc_output_log.txt ─────────────────────────────────
    # Fresh run or --plot: overwrite.  --resume: append so history is preserved.
    LOG_FILE = 'mcmc_output_log.txt'
    _log_mode = 'a' if args.resume else 'w'
    _tee = _Tee(LOG_FILE, mode=_log_mode)
    sys.stdout = _tee
    if args.resume:
        import datetime
        print(f"\n{'='*62}")
        print(f"  RESUMED  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*62}\n")


    if args.plot:
        make_plots(CHECKPOINT)
    else:
        sampler = run_mcmc(resume=args.resume)
        make_plots(sampler)

    _tee.close()
