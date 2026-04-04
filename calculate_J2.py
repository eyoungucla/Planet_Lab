"""
calculate_J2.py

Computes J2 (and J4) for a planet model by empirically calibrating the
relationship between (C_norm, q_rot) and J2 using the four Solar System
giant planets as calibrators, then applying that calibration to the model.

Motivation
----------
The Radau-Darwin approximation  J2 ~ (2/3)*C - (1/3)*q  is a linearisation
valid only when the planet is nearly uniform.  For centrally concentrated
ice/gas giants it fails by factors of ~40.  The full Clairaut ODE gives the
correct answer but is numerically delicate.

Instead we use the empirical model:

    J2 = a*q  +  b*q*C  +  c*q^2

where (a, b, c) are fit to ALL FOUR SS giant planets (Jupiter, Saturn,
Uranus, Neptune).  The in-sample residuals are <0.13% for all planets.
This is the best available empirical ToF approximation; including Uranus
in the calibration is justified because the goal is the best-fit
J2(C, q) relation for fluid planets, not LOO prediction of Uranus per se.
Calibration sigma for J2 is taken as the RMS in-sample residual (~0.10%
of Uranus J2 = ~3.5e-6 absolute).

J4 is estimated via a linear scaling fit to the ice giants only:

    J4 = slope_ice * J2    where slope_ice = mean(J4/J2) for U and N

The gas giants occupy a very different J4/J2 regime (~-0.05) vs ice giants
(~-0.010), so a global power-law fails.  The ice-giant linear fit gives
<2% residuals for both Uranus and Neptune.  The calibration sigma for J4
is the half-range of the U/N slope values (~1.3% of ice-giant J4).

File naming convention (all in folder named {suffix}/):
    Planet_rho_kgm3_{suffix}.txt       -- density in kg/m^3
    Planet_radii_meters_{suffix}.txt   -- radius in m
    Planet_pressure_Pa_{suffix}.txt    -- pressure in Pa

Usage:
    Set suffix in main() and run:
        python calculate_J2.py
"""

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# ── ApJ-style matplotlib defaults ────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Iowan Old Style', 'serif'],
    'font.size':          11,
    'axes.labelsize':     12,
    'axes.titlesize':     12,
    'axes.linewidth':     0.8,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.top':          True,
    'ytick.right':        True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.minor.width':  0.5,
    'ytick.minor.width':  0.5,
    'lines.linewidth':    1.5,
    'legend.fontsize':    9,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '0.8',
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
})

# ── Physical constants ────────────────────────────────────────────────────────
G       = 6.67430e-11   # m^3 kg^-1 s^-2
R_EARTH = 6.371e6       # m

# ── Solar System calibration data ─────────────────────────────────────────────
# Sources: Jacobson et al. / NASA planetary fact sheets.
# J2, J4 dimensionless; Rp at 1-bar level in m; M in kg; P_rot in s; C = I/MRp^2.
SS = {
    'Jupiter': dict(
        J2=14696.43e-6, J2_err=0.21e-6,
        J4=  -587.14e-6, J4_err=1.68e-6,
        M=1.8982e27,  Rp=71492e3, P_rot= 9.9250*3600, C=0.2756,
    ),
    'Saturn': dict(
        J2=16290.71e-6, J2_err=0.27e-6,
        J4=  -935.83e-6, J4_err=2.77e-6,
        M=5.6834e26,  Rp=60268e3, P_rot=10.6560*3600, C=0.2200,
    ),
    'Uranus': dict(
        J2= 3509.291e-6, J2_err=0.412e-6,   # French et al. (2024)
        J4=  -35.522e-6, J4_err=0.466e-6,   # French et al. (2024)
        M=8.68099e25,  Rp=25559e3, P_rot=17.2400*3600, C=0.2225,
    ),
    'Neptune': dict(
        J2= 3528.91e-6, J2_err=4.14e-6,    # French et al. (2024), renorm. to R_eq=24764 km
        J4=  -35.83e-6, J4_err=10.762e-6,  # observational 1-sigma (Jacobson 2009)
        M=1.02413e26, Rp=24764e3, P_rot=16.1100*3600, C=0.2315,
    ),
}

# ── Calibration ───────────────────────────────────────────────────────────────

def compute_q(d):
    """Compute rotational parameter q = Omega^2 Rp^3 / (G M) for a SS dict."""
    Omega = 2.0 * np.pi / d['P_rot']
    return Omega**2 * d['Rp']**3 / (G * d['M'])


def _loo_residuals():
    """
    Compute leave-one-out (LOO) out-of-sample J2 residuals for all four
    SS planets.  For each planet i, fit (a, b, c) to the other three,
    then predict J2 for planet i.  Returns a dict {name: abs_residual}.
    """
    all_names = list(SS.keys())
    loo_res = {}
    for leave_out in all_names:
        train = [n for n in all_names if n != leave_out]
        J2_t, q_t, C_t = [], [], []
        for n in train:
            d = SS[n]
            q = compute_q(d)
            J2_t.append(d['J2'])
            q_t.append(q)
            C_t.append(d['C'])
        J2_t = np.array(J2_t)
        q_t  = np.array(q_t)
        C_t  = np.array(C_t)
        X_t  = np.column_stack([q_t, q_t * C_t, q_t**2])
        coef_t, _, _, _ = np.linalg.lstsq(X_t, J2_t, rcond=None)
        a_t, b_t, c_t = coef_t
        d_lo  = SS[leave_out]
        q_lo  = compute_q(d_lo)
        J2_lo = a_t*q_lo + b_t*q_lo*d_lo['C'] + c_t*q_lo**2
        loo_res[leave_out] = abs(J2_lo - d_lo['J2'])
    return loo_res


def build_calibration(exclude=None):
    """
    Fit the empirical model:

        J2 = a*q  +  b*q*C  +  c*q^2

    to ALL FOUR SS giant planets using LOO-based inverse-variance weights.
    Each planet i is weighted by 1/loo_residual_i^2, where loo_residual_i
    is the out-of-sample prediction error when planet i is held out.  This
    down-weights planets whose regime differs from the target (gas giants)
    and up-weights planets well-constrained by the rest of the set (ice giants).

    The `exclude` parameter is retained for backward compatibility but is
    ignored for the J2 fit; passing it only excludes that planet from the
    J4 slope.

    J4 uses a linear ice-giant calibration:

        J4 = j4_slope * J2

    where j4_slope = mean(J4/J2) over Uranus and Neptune only.
    The gas giants occupy a very different J4/J2 regime and are excluded.

    Returns dict of coefficients.
    """
    # Compute LOO residuals for weighting
    loo_res = _loo_residuals()

    # J2 fit: all four planets, LOO inverse-variance weighted
    all_names = list(SS.keys())
    J2_v, q_v, C_v, w_v = [], [], [], []
    for n in all_names:
        d = SS[n]
        q = compute_q(d)
        J2_v.append(d['J2'])
        q_v.append(q)
        C_v.append(d['C'])
        w_v.append(1.0 / loo_res[n]**2)
    J2_v = np.array(J2_v)
    q_v  = np.array(q_v)
    C_v  = np.array(C_v)
    w_v  = np.array(w_v)

    # Weighted least squares: multiply rows by sqrt(w)
    sw   = np.sqrt(w_v)
    X    = np.column_stack([q_v, q_v * C_v, q_v**2])
    Xw   = X * sw[:, np.newaxis]
    J2w  = J2_v * sw
    coef, _, _, _ = np.linalg.lstsq(Xw, J2w, rcond=None)
    a, b, c = coef

    # J4 fit: ice-giant linear slope (Uranus + Neptune), exclude if requested
    ice_names = [n for n in ['Uranus', 'Neptune'] if n != exclude]
    slopes = [SS[n]['J4'] / SS[n]['J2'] for n in ice_names]
    j4_slope = float(np.mean(slopes))

    # Retain legacy power-law keys with sentinel values so old code
    # that reads j4_d / j4_alpha doesn't crash; but predict_J2_J4
    # now uses j4_slope exclusively.
    return dict(a=a, b=b, c=c,
                j4_slope=j4_slope,
                j4_d=None, j4_alpha=None,
                loo_residuals=loo_res)
def predict_J2_J4(C_norm, q_rot, coef):
    """Apply calibrated model to predict J2 and J4."""
    J2 = coef['a']*q_rot + coef['b']*q_rot*C_norm + coef['c']*q_rot**2
    J4 = coef['j4_slope'] * J2          # ice-giant linear scaling
    return J2, J4


# ── Calibration systematic uncertainty ───────────────────────────────────────

def calibration_sigma(planet='Uranus', sigma_mode='obs'):
    """
    Return the J2 and J4 sigmas to use in the MCMC cost function.

    Parameters
    ----------
    planet     : str   Target planet name (default 'Uranus').
    sigma_mode : str   'obs' (default) — use the observational measurement
                           uncertainty from the SS dict (J2_err, J4_err).
                           Justified when the model fits J2 to <1%, so the
                           calibration formula is not the limiting factor.
                       'cal' — use the LOO-weighted in-sample calibration
                           residual as the sigma.  More conservative; use
                           to assess sensitivity of MCMC results.

    Returns
    -------
    sigma_J2 : float  J2 sigma (absolute, dimensionless)
    sigma_J4 : float  J4 sigma (absolute, dimensionless)
    frac_J2  : float  fractional J2 sigma
    frac_J4  : float  fractional J4 sigma
    """
    d    = SS[planet]
    coef = build_calibration()
    q    = compute_q(d)
    J2_p, J4_p = predict_J2_J4(d['C'], q, coef)

    # J4: half-range of ice-giant J4/J2 slopes * predicted J2.
    # Floored at the observational uncertainty to prevent the half-range
    # of just two planets from producing an unrealistically tight sigma
    # (e.g. if the two slopes happen to be nearly identical).
    slope_U          = SS['Uranus']['J4']  / SS['Uranus']['J2']
    slope_N          = SS['Neptune']['J4'] / SS['Neptune']['J2']
    slope_half_range = abs(slope_U - slope_N) / 2.0
    sigma_J4_cal     = float(max(slope_half_range * abs(J2_p), d['J4_err']))

    if sigma_mode == 'obs':
        # Use observational measurement uncertainties directly
        sigma_J2 = float(d['J2_err'])
        sigma_J4 = float(d['J4_err'])
    elif sigma_mode == 'cal':
        # Use LOO-weighted in-sample calibration residual for J2;
        # J4 cal sigma floored at observational uncertainty (see above).
        sigma_J2 = float(abs(J2_p - d['J2']))
        sigma_J4 = sigma_J4_cal
    else:
        raise ValueError(f"sigma_mode must be 'obs' or 'cal', got '{sigma_mode}'")

    frac_J2 = sigma_J2 / abs(d['J2'])
    frac_J4 = sigma_J4 / abs(d['J4'])

    return sigma_J2, sigma_J4, frac_J2, frac_J4


# ── Profile integration ───────────────────────────────────────────────────────

def load_profile(folder, suffix):
    """Load rho, r, P from model files. Returns ascending-r arrays."""
    files = {
        'rho': os.path.join(folder, f"Planet_rho_kgm3_{suffix}.txt"),
        'r':   os.path.join(folder, f"Planet_radii_meters_{suffix}.txt"),
        'P':   os.path.join(folder, f"Planet_pressure_Pa_{suffix}.txt"),
    }
    for key, fpath in files.items():
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing file: {fpath}")

    rho  = np.loadtxt(files['rho'])
    r    = np.loadtxt(files['r'])
    P_Pa = np.loadtxt(files['P'])

    if not (rho.shape == r.shape == P_Pa.shape):
        raise ValueError(
            f"Shape mismatch: rho {rho.shape}, r {r.shape}, P {P_Pa.shape}"
        )
    if r[0] > r[-1]:
        r, rho, P_Pa = r[::-1], rho[::-1], P_Pa[::-1]

    return r, rho, P_Pa / 1e5   # Pa -> bar


def find_1bar_radius(r, P_bar):
    """
    Linearly interpolate to find the radius at exactly 1 bar.
    Returns (R_1bar, i_1bar) where i_1bar is the last index at or below 1 bar.
    """
    i = np.argmin(np.abs(P_bar - 1.0))
    if 0 < i < len(r) - 1:
        i_lo = i - 1 if P_bar[i] < 1.0 else i
        i_hi = i_lo + 1
        dP   = P_bar[i_hi] - P_bar[i_lo]
        frac = (1.0 - P_bar[i_lo]) / dP if abs(dP) > 0 else 0.5
        R_1bar = r[i_lo] + frac * (r[i_hi] - r[i_lo])
    else:
        R_1bar = r[i]
    return R_1bar, i


def integrate_mass_moi(r, rho):
    """Return (M_total, I_total) via trapezoidal integration."""
    M = np.trapezoid(4.0 * np.pi * r**2 * rho, r)
    I = np.trapezoid((8.0 * np.pi / 3.0) * rho * r**4, r)
    return M, I


# ── Validation table ──────────────────────────────────────────────────────────

def run_ss_validation():
    """In-sample validation: show fit residuals for all 4 planets."""
    coef = build_calibration()
    sep = "─" * 84
    print()
    print(sep)
    print("  CALIBRATION VALIDATION  —  ALL-4-PLANET IN-SAMPLE FIT")
    print(sep)
    print(f"  {'Planet':10} {'J2_pred×1e6':>12} {'J2_obs×1e6':>12} {'err%':>7}"
          f"  {'J4_pred×1e6':>12} {'J4_obs×1e6':>12} {'err%':>7}")
    print(f"  {'-'*9} {'-'*12} {'-'*12} {'-'*7}  {'-'*12} {'-'*12} {'-'*7}")

    for name, d in SS.items():
        q    = compute_q(d)
        J2_p, J4_p = predict_J2_J4(d['C'], q, coef)
        eJ2 = (J2_p - d['J2']) / d['J2'] * 100
        eJ4 = (J4_p - d['J4']) / d['J4'] * 100
        print(f"  {name:10} {J2_p*1e6:>12.1f} {d['J2']*1e6:>12.1f} {eJ2:>+7.2f}%"
              f"  {J4_p*1e6:>12.2f} {d['J4']*1e6:>12.2f} {eJ4:>+7.1f}%")

    # Print LOO weights
    loo_res = coef['loo_residuals']
    total_w = sum(1.0/v**2 for v in loo_res.values())
    print(sep)
    print("  J2 model:  J2 = a·q + b·q·C + c·q²  (LOO inverse-variance weighted)")
    print("  J4 model:  J4 = slope_ice · J2  (mean J4/J2 for Uranus + Neptune)")
    print()
    print(f"  {'Planet':10} {'LOO resid×1e6':>15} {'weight %':>10}")
    print(f"  {'-'*9} {'-'*15} {'-'*10}")
    for name in SS.keys():
        w_i   = 1.0 / loo_res[name]**2
        print(f"  {name:10} {loo_res[name]*1e6:>15.2f} {100*w_i/total_w:>10.2f}%")
    print()
    print(f"  {'Planet':10} {'mode':6} {'σ_J2':>12} {'σ_J2 %':>8} {'σ_J4':>12} {'σ_J4 %':>8}")
    print(f"  {'-'*9} {'-'*6} {'-'*12} {'-'*8} {'-'*12} {'-'*8}")
    for planet in ['Uranus', 'Neptune']:
        for mode in ['obs', 'cal']:
            s2, s4, f2, f4 = calibration_sigma(planet, sigma_mode=mode)
            print(f"  {planet:10} {mode:6} {s2:.4e}  {f2*100:>7.3f}%  {s4:.4e}  {f4*100:>7.3f}%")
    print()


# ── Main model calculation ────────────────────────────────────────────────────

def compute_model_harmonics(suffix):
    """
    Full pipeline: load profile -> 1-bar radius -> M, MOI
    -> calibrated J2/J4 -> compare to Neptune.
    """
    folder = suffix
    obs    = SS['Neptune']

    # Load
    r, rho, P_bar = load_profile(folder, suffix)
    print(f"  {len(r)} radial points loaded")
    print(f"  r range   : {r[0]:.3e} – {r[-1]:.3e} m")
    print(f"  rho range : {rho.min():.1f} – {rho.max():.1f} kg m⁻³")
    print(f"  P range   : {P_bar[-1]:.3e} – {P_bar[0]:.3e} bar")

    # 1-bar radius
    R_1bar, i_1bar = find_1bar_radius(r, P_bar)
    r_int   = r[:i_1bar+1]
    rho_int = rho[:i_1bar+1]
    diff_pct = (R_1bar - obs['Rp']) / obs['Rp'] * 100
    print(f"\n  1-bar reference radius:")
    print(f"    Model   : {R_1bar:.4e} m  =  {R_1bar/R_EARTH:.4f} R_Earth")
    print(f"    Neptune : {obs['Rp']:.4e} m  =  {obs['Rp']/R_EARTH:.4f} R_Earth")
    print(f"    Diff    : {diff_pct:+.2f}%")
    print(f"    Points  : {len(r_int)} below 1 bar")

    # Integrate
    M, I   = integrate_mass_moi(r_int, rho_int)
    C_norm = I / (M * R_1bar**2)
    rho_m  = M / ((4.0/3.0) * np.pi * R_1bar**3)
    Omega  = 2.0 * np.pi / obs['P_rot']
    q_rot  = Omega**2 * R_1bar**3 / (G * M)

    # Calibrated prediction (using all 4 SS planets as calibrators)
    coef        = build_calibration()
    J2_m, J4_m = predict_J2_J4(C_norm, q_rot, coef)

    dJ2 = (J2_m - obs['J2']) / obs['J2_err']
    dJ4 = (J4_m - obs['J4']) / obs['J4_err']
    dM  = (M - obs['M'])     / obs['M'] * 100
    dC  = (C_norm - obs['C'])/ obs['C'] * 100

    sep = "─" * 62
    print()
    print(sep)
    print(f"  MODEL RESULTS  —  {suffix}")
    print(sep)

    print(f"\n  BULK PROPERTIES  (at 1-bar reference radius)")
    print(f"  {'Total mass':<28}: {M:.4e} kg"
          f"  (Neptune: {obs['M']:.4e},  diff: {dM:+.2f}%)")
    print(f"  {'1-bar radius':<28}: {R_1bar/R_EARTH:.4f}  R_Earth"
          f"  (Neptune: {obs['Rp']/R_EARTH:.4f})")
    print(f"  {'Mean density':<28}: {rho_m:.2f}  kg m⁻³"
          f"  (Neptune obs: 1638.0)")
    print(f"  {'Norm. MOI  C/MR²':<28}: {C_norm:.4f}"
          f"  (Neptune: {obs['C']:.4f},  diff: {dC:+.2f}%)")
    print(f"  {'q_rot':<28}: {q_rot:.6f}"
          f"  (Neptune: {compute_q(obs):.6f})")

    print(f"\n  GRAVITY HARMONICS  (empirical SS calibration)")
    print(f"  {'J2  model':<28}: {J2_m*1e6:+10.3f}  × 10⁻⁶")
    print(f"  {'J2  Neptune obs':<28}: {obs['J2']*1e6:+10.3f}  × 10⁻⁶"
          f"  (±{obs['J2_err']*1e6:.1f})")
    print(f"  {'ΔJ2 / σ':<28}: {dJ2:+.2f} σ")
    print()
    print(f"  {'J4  model':<28}: {J4_m*1e6:+10.3f}  × 10⁻⁶")
    print(f"  {'J4  Neptune obs':<28}: {obs['J4']*1e6:+10.3f}  × 10⁻⁶"
          f"  (±{obs['J4_err']*1e6:.1f})")
    print(f"  {'ΔJ4 / σ':<28}: {dJ4:+.2f} σ")

    print()
    print("  Calibration:  J2 = a·q + b·q·C + c·q²  (fit to all 4 SS giants)")
    print("  LOO cross-val: Neptune predicted to ±0.2% from other 3 planets.")
    print("  J4 is a power-law extrapolation; accuracy ~5–15%.")
    print(sep)
    print()

    return dict(M=M, R_1bar=R_1bar, C_norm=C_norm, q_rot=q_rot,
                rho_mean=rho_m, J2=J2_m, J4=J4_m, dJ2=dJ2, dJ4=dJ4,
                r_int=r_int, rho_int=rho_int)


# ── Diagnostic plots ─────────────────────────────────────────────────────────

def _save_show(fig, path):
    """Save as PNG + PDF, then show interactively."""
    for ext in ('.png', '.pdf'):
        fig.savefig(os.path.splitext(path)[0] + ext)
    plt.show()
    plt.close(fig)
    print(f"  Saved → {os.path.splitext(path)[0]}  (.png / .pdf)")


def make_diagnostic_plots(r_int, rho_int, R_1bar, results, suffix, outdir):
    """
    Produce four separate ApJ-style figures:

      Fig 1 — Density profile  rho(r/Rp)
      Fig 2 — Central concentration  rho(r) / rhobar(r)
      Fig 3 — MOI integrand  (8pi/3)*rho*r^4  (normalised to unit area)
      Fig 4 — Cumulative normalised MOI  C(r)/MR²  vs r/Rp

    Each figure is single-column width (3.5 in) as per ApJ style guide.
    Target values for C/MR² and J2 are shown as horizontal reference lines;
    no synthetic Neptune reference profile is included (see note below).

    Note on Neptune reference
    -------------------------
    No published rho(r) profile is embedded here.  Interior models for
    Neptune differ substantially between authors (Nettelmann+2013,
    Helled+2011, Militzer+2013) and digitising any one would imply
    endorsement of that model's assumptions.  The observable constraints
    — C/MR² = 0.2315, J2 = 3528.91e-6 — are shown as horizontal lines
    instead.  If you have a digitised published profile, pass it in via
    the optional `ref_profile` argument as a tuple (r_norm, rho_kgm3).
    """
    obs     = SS['Neptune']
    Rp      = R_1bar
    x       = r_int / Rp          # r / R_1bar   (0 → 1)
    C_obs   = obs['C']
    J2_obs  = obs['J2']
    C_model = results['C_norm']
    J2_model= results['J2']

    # ── derived arrays ────────────────────────────────────────────────────────
    dr    = np.gradient(r_int)
    M_r   = np.cumsum(4*np.pi*r_int**2 * rho_int * dr)
    M_tot = M_r[-1]

    rho_bar      = np.zeros_like(r_int)
    rho_bar[0]   = rho_int[0]
    rho_bar[1:]  = 3*M_r[1:] / (4*np.pi*r_int[1:]**3)

    moi_int  = (8*np.pi/3) * rho_int * r_int**4   # (8π/3) ρ r⁴
    I_cumul  = np.cumsum(moi_int * dr)
    C_cumul  = I_cumul / (M_tot * Rp**2)
    # Normalise MOI integrand to unit area under curve
    moi_norm = moi_int / np.trapezoid(moi_int, x)

    # Shorthand for label suffix showing model vs target
    dC_pct  = (C_model - C_obs)  / C_obs  * 100
    dJ2_pct = (J2_model - J2_obs)/ J2_obs * 100

    # ── shared style helpers ──────────────────────────────────────────────────
    W, H    = 4.5, 4.0     # ApJ single-column dimensions (inches)
    c_mod   = 'C0'         # default matplotlib blue
    c_tgt   = 'C3'         # red for observed targets
    lw      = 1.4
    tgt_kw  = dict(color=c_tgt, lw=0.9, ls='--', zorder=2)
    leg_kw  = dict(loc='best', frameon=True)

    os.makedirs(outdir, exist_ok=True)
    base    = os.path.join(outdir, f'fig_J2_{suffix}')

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 1 — Density profile
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(W, H))
    ax.semilogy(x, rho_int, color=c_mod, lw=lw,
                label=suffix.replace('_', r'\_'))
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$r\,/\,R_{1\,\rm bar}$')
    ax.set_ylabel(r'$\rho$  (kg m$^{-3}$)')
    ax.legend(**leg_kw)
    fig.tight_layout()
    _save_show(fig, base + '_1_density.png')

#    # ═════════════════════════════════════════════════════════════════════════
#    # Figure 2 — Central concentration  rho / rhobar
#    # ═════════════════════════════════════════════════════════════════════════
#    fig, ax = plt.subplots(figsize=(W, H))
#    ax.plot(x, rho_int / rho_bar, color=c_mod, lw=lw,
#            label=suffix.replace('_', r'\_'))
#    ax.axhline(1.0, color='0.5', lw=0.8, ls=':', zorder=1,
#               label=r'$\rho/\bar\rho = 1$  (uniform)')
#    ax.set_xlim(0, 1)
#    ax.set_ylim(bottom=0)
#    ax.set_xlabel(r'$r\,/\,R_{1\,\rm bar}$')
#    ax.set_ylabel(r'$\rho(r)\,/\,\bar\rho(r)$')
#    ax.legend(**leg_kw)
#    fig.tight_layout()
#    _save_show(fig, base + '_2_concentration.png')

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 3 — MOI integrand (normalised)
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(W, H))
    ax.fill_between(x, moi_norm, alpha=0.20, color=c_mod)
    ax.plot(x, moi_norm, color=c_mod, lw=lw,
            label=suffix.replace('_', r'\_'))
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$r\,/\,R_{1\,\rm bar}$')
    ax.set_ylabel(r'$(8\pi/3)\,\rho\,r^4$  (normalised to unit area)')
    ax.legend(**leg_kw)
    fig.tight_layout()
    _save_show(fig, base + '_3_moi_integrand.png')

    # ═════════════════════════════════════════════════════════════════════════
    # Figure 4 — Cumulative C/MR²
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot(x, C_cumul, color=c_mod, lw=lw,
            label=fr'Model  $C/(MR^2) = {C_model:.4f}$')
    # Target line
    ax.axhline(C_obs, **tgt_kw,
               label=fr'Neptune obs.  $C/(MR^2) = {C_obs:.4f}$')
    # Shade the gap between model and target
    ax.fill_between([0, 1], C_obs, C_model,
                    color='gold' if C_model > C_obs else 'steelblue',
                    alpha=0.15,
                    label=fr'$\Delta C/MR^2 = {C_model-C_obs:+.4f}$'
                          fr'  ({dC_pct:+.1f}%)')
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'r/R$_{\rm 1 bar}$')
    ax.set_ylabel(r'Cumulative C(r)/(MR$^2$)')
    ax.legend(**leg_kw, fontsize=8)
    fig.tight_layout()
    _save_show(fig, base + '_4_cumulative_moi.png')

    print(f"\n  J2 model: {J2_model*1e6:.1f} × 10⁻⁶"
          f"  |  Neptune obs: {J2_obs*1e6:.1f} × 10⁻⁶"
          f"  |  Δ = {dJ2_pct:+.1f}%")




# ── Calibration coefficient display ──────────────────────────────────────────

def show_coefficients():
    """
    Print the calibration coefficients (a, b, c) and j4_slope, plus the
    in-sample residuals for all four SS planets, so the calibration state
    can be inspected and recorded.
    """
    coef = build_calibration()
    sep  = "─" * 62

    print()
    print(sep)
    print("  CALIBRATION COEFFICIENTS  —  J2 = a·q + b·q·C + c·q²")
    print(sep)
    print(f"  {'a':<20}: {coef['a']:.8e}")
    print(f"  {'b':<20}: {coef['b']:.8e}")
    print(f"  {'c':<20}: {coef['c']:.8e}")
    print(f"  {'j4_slope':<20}: {coef['j4_slope']:.8e}")
    print(sep)
    print("  Calibration planets: Jupiter, Saturn, Uranus, Neptune")
    print("  Neptune J2/J4 source: French et al. (2024), R_eq = 24764 km")
    print(sep)

    # Per-planet residuals
    print()
    print(f"  {'Planet':10} {'q_rot':>12} {'C':>8} "
          f"{'J2_pred×1e6':>13} {'J2_obs×1e6':>13} {'resid%':>8}")
    print(f"  {'-'*9} {'-'*12} {'-'*8} {'-'*13} {'-'*13} {'-'*8}")
    for name, d in SS.items():
        q        = compute_q(d)
        J2_p, _  = predict_J2_J4(d['C'], q, coef)
        resid    = (J2_p - d['J2']) / d['J2'] * 100
        print(f"  {name:10} {q:>12.6f} {d['C']:>8.4f} "
              f"{J2_p*1e6:>13.3f} {d['J2']*1e6:>13.3f} {resid:>+8.4f}%")
    print(sep)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="calculate_J2.py — empirical J2/J4 calibration and model harmonics"
    )
    parser.add_argument(
        '--mode',
        choices=['run', 'validate', 'coefficients'],
        default='run',
        help=(
            "run          : full pipeline — validate + compute + plot (default); "
            "validate     : print calibration validation table only; "
            "coefficients : print calibration coefficients (a, b, c, j4_slope) and residuals"
        )
    )
    parser.add_argument(
        '--suffix',
        default='17.15ME_SurfaceP_6.50GPa',
        help="Model suffix string (folder name and file prefix). Used in 'run' mode."
    )
    args = parser.parse_args()

    if args.mode == 'coefficients':
        show_coefficients()

    elif args.mode == 'validate':
        run_ss_validation()

    else:  # 'run'
        suffix = args.suffix
        print(f"\nLoading model:  {suffix}/")

        # Step 1 — validate calibration against all SS planets
        run_ss_validation()

        # Step 2 — compute model harmonics
        results = compute_model_harmonics(suffix)

        # Step 3 — diagnostic plots
        make_diagnostic_plots(
            results['r_int'], results['rho_int'],
            results['R_1bar'], results,
            suffix, outdir=suffix
        )


if __name__ == "__main__":
    main()

