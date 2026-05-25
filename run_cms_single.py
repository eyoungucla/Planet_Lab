"""
run_cms_single.py
=================
Run a single Planet_LAB model and compute J2, J4, C/MR^2 using both
the existing empirical calibration (calculate_J2.py) and the full CMS
method (CMSPlanet / cms.py), then print a side-by-side comparison.

Usage
-----
    python run_cms_single.py --planet Neptune --suffix <run_suffix>
    python run_cms_single.py --planet Uranus  --suffix <run_suffix>

The --suffix argument must match the folder name and file prefix used
by calculate_J2.py, e.g.:

    python run_cms_single.py --planet Neptune --suffix 17.15ME_SurfaceP_6.50GPa

Files expected in <suffix>/ directory
--------------------------------------
    Planet_rho_kgm3_<suffix>.txt
    Planet_radii_meters_<suffix>.txt
    Planet_pressure_Pa_<suffix>.txt

Alternatively, pass --profile_dir to specify a different directory.

How it works
------------
1.  Loads the rho(r) profile from the Planet_LAB output files using
    the existing load_profile() from calculate_J2.py.
2.  Trims to the 1-bar radius (same as current pipeline).
3.  Feeds the profile to CMSPlanet to compute J2, J4 via full CMS.
4.  Also runs the existing empirical calibration for comparison.
5.  Computes C/MR^2 from the spherical density profile (same formula
    as in calculate_J2.py; CMSPlanet.NMoI() is not yet implemented).
6.  Prints a side-by-side comparison table.

Array orientation note
----------------------
Planet_LAB outputs arrays indexed from the CENTRE outward (r[0] = 0,
r[-1] = surface).  CMSPlanet/cms.py requires arrays indexed from the
SURFACE inward (zvec[0] = outer radius).  This script handles the flip
automatically — you do not need to pre-reverse your arrays.
"""

import argparse
import sys
import os
import numpy as np

# ── Import local modules ──────────────────────────────────────────────────────
# These must be in the same directory or on PYTHONPATH.
try:
    import calculate_J2 as J2mod
except ImportError:
    sys.exit("ERROR: calculate_J2.py not found. Place it in the same directory.")

try:
    from CMSPlanet import CMSPlanet
except ImportError:
    sys.exit("ERROR: CMSPlanet.py not found. Place it in the same directory.")

# ── Physical constants (match calculate_J2.py) ───────────────────────────────
G       = 6.67430e-11   # m^3 kg^-1 s^-2
R_EARTH = 6.371e6       # m

# ── Planet observational data (from calculate_J2.py SS dict) ─────────────────
PLANET_DATA = {
    'Neptune': dict(
        M       = 1.02413e26,   # kg
        Rp      = 24764e3,      # m  (1-bar equatorial radius)
        P_rot   = 16.1100*3600, # s
        C       = 0.2410,
        J2_obs  = 3528.91e-6,
        J2_err  = 4.14e-6,
        J4_obs  = -35.83e-6,
        J4_err  = 10.762e-6,
    ),
    'Uranus': dict(
        M       = 8.68099e25,
        Rp      = 25559e3,
        P_rot   = 17.2400*3600,
        C       = 0.2225,
        J2_obs  = 3509.291e-6,
        J2_err  = 0.412e-6,
        J4_obs  = -35.522e-6,
        J4_err  = 0.466e-6,
    ),
}


# ── CMS observable struct (mimics the _default_planet class) ─────────────────

def _make_obs(planet_name, R_1bar=None, M_model=None, P_rot=None):
    """Build an obs struct for CMSPlanet from the planet data dict.

    If R_1bar and M_model are supplied they override the reference planet
    values, ensuring q_rot is computed from the actual model geometry rather
    than the nominal observed radius.

    If P_rot (rotation period in seconds) is supplied, it is used directly and
    no PLANET_DATA entry is required (planet_name then acts only as a label).
    Otherwise P_rot is taken from PLANET_DATA[planet_name].
    """
    d = PLANET_DATA.get(planet_name) if P_rot is None else None
    if P_rot is None and d is None:
        raise KeyError(
            f"run_cms: planet_name '{planet_name}' is not in PLANET_DATA and "
            f"no P_rot was supplied. Pass P_rot=<seconds> for a generic model."
        )

    class obs:
        pass

    obs.pname = planet_name
    obs.M     = M_model if M_model is not None else d['M']
    obs.a0    = R_1bar  if R_1bar  is not None else d['Rp']
    obs.s0    = obs.a0
    obs.P0    = 1e5           # surface pressure in Pa (1 bar)
    obs.P     = P_rot if P_rot is not None else d['P_rot']  # rotation period (s)
    return obs


# ── C/MR^2 from spherical profile ────────────────────────────────────────────

def compute_CMR2(r, rho, R_ref):
    """
    Compute the normalised axial moment of inertia C/MR^2 from a
    spherical density profile by trapezoidal integration.

    Uses the same formula as calculate_J2.integrate_mass_moi().
    """
    M, I = J2mod.integrate_mass_moi(r, rho)
    return I / (M * R_ref**2), M


# ── Main CMS run ─────────────────────────────────────────────────────────────

# ── Profile conditioning switches ────────────────────────────────────────────
# IRON_ATMOSPHERE : apply ln(rho) moving-average ironing to the outer
#                  atmosphere (above the transition zone) to remove staircase
#                  artefacts from the Planet_LAB grid.  Set True for production.
# IRON_WINDOW     : moving-average window width in index units for outer tail
#                   ironing.  301 matches the standalone smoothing script.
# PAD_KM_CORNER   : half-width of the tanh transition zone around the phase
#                   boundary in km.  200 km is the original broad window.
#                   Reducing to ~20 km narrows the corner smoothing to match
#                   the standalone script which uses a tight PCHIP corner fix.
IRON_ATMOSPHERE = False
IRON_WINDOW     = 301
PAD_KM_CORNER   = 20.0   # km — tune: 10-50 km for tight corner smoothing

def run_cms(planet_name, r_m, rho_kgm3, R_1bar=None, N_layers=512, xlayers=64,
            n_trans_override=None, sigma_atm=20, verbose=True, P_rot=None):
    """
    Feed a rho(r) profile to CMSPlanet and return J2, J4, C/MR^2.

    Parameters
    ----------
    planet_name      : str      'Neptune' or 'Uranus', or any label if P_rot
                                is supplied (e.g. 'model' for a generic planet)
    r_m              : ndarray  Radii in metres, ascending from centre to surface
    rho_kgm3         : ndarray  Densities in kg/m^3, same shape as r_m
    R_1bar           : float    Actual 1-bar radius from the model (m).
                                If None, uses the reference planet radius.
                                Should always be passed for consistent q_rot.
    N_layers         : int      Number of CMS shells (default 512)
    xlayers          : int      Skip-n-spline layers for speed (default 64)
    n_trans_override : int      If set, forces the number of transition-zone
                                shells to this value regardless of N_layers.
                                Use to test whether reducing transition-zone
                                resolution reduces Gibbs-type ringing at the
                                density discontinuity.  None = use default
                                fraction (20% of N_layers).
    sigma_atm        : float    Gaussian smoothing sigma (in index units) applied
                                to the atmosphere region above the transition zone.
                                Default 20.  Larger values more aggressively suppress
                                numerical zigzag artefacts in the low-density
                                envelope.  The core and transition zone always use
                                sigma=3.  Set to 3 to match the old uniform behaviour.
    verbose          : bool
    P_rot            : float    Rotation period in seconds. If supplied, used
                                directly and planet_name need not be in
                                PLANET_DATA. If None, taken from PLANET_DATA.

    Returns
    -------
    dict with keys J2, J4, C_over_MR2, M_kg, q
    """
    d    = PLANET_DATA.get(planet_name)
    if d is None and P_rot is None:
        raise KeyError(
            f"run_cms: planet_name '{planet_name}' is not in PLANET_DATA and "
            f"no P_rot was supplied. Pass P_rot=<seconds> for a generic model."
        )

    # ── Integrate mass from the full-resolution spherical profile ─────────────
    # Do this BEFORE any subsampling or smoothing so C/MR2 and M use the
    # original high-resolution data.
    M_model, I_model = J2mod.integrate_mass_moi(r_m, rho_kgm3)
    R_ref = R_1bar if R_1bar is not None else (d['Rp'] if d is not None else None)
    if R_ref is None:
        raise ValueError("run_cms: R_1bar must be supplied for a generic model "
                         "(no PLANET_DATA entry to fall back on).")
    C_MR2 = I_model / (M_model * R_ref**2)

    # ── Strip zero-radius centre point if present ─────────────────────────────
    start = 1 if r_m[0] == 0.0 else 0
    r_use   = r_m[start:].copy()
    rho_use = rho_kgm3[start:].copy()

    # ── Truncate at a minimum density floor ───────────────────────────────────
    # Shells with rho < rho_floor are essentially massless and contribute
    # negligibly to J2, but their near-zero density causes _eq52 to lose
    # its bracket (the function becomes very flat near zeta=1 for massless
    # shells).  We cut them off.  The 1-bar radius is preserved because
    # C/MR2 and M are already computed from the full profile above.
    # 0.5 kg/m^3 ~ density of H2 gas at ~0.3 bar / 50 K, well into the
    # upper atmosphere where the gravity contribution is negligible.
    rho_floor = 2.0   # kg/m^3
    floor_mask = rho_use >= rho_floor
    if np.any(~floor_mask):
        last_valid = int(np.where(floor_mask)[0][-1])
        r_use   = r_use[:last_valid+1]
        rho_use = rho_use[:last_valid+1]
        if verbose:
            print(f"  Density floor {rho_floor} kg/m³: "
                  f"trimmed to r = {r_use[-1]/1e3:.1f} km  "
                  f"({len(r_use)} pts retained)")

    # ── Remove duplicate or near-duplicate radial points ─────────────────────
    # Duplicate radii (zero-width shells) at the phase boundary cause the CMS
    # shape iteration to become ill-conditioned as N_layers increases, because
    # more shells straddle the pathological point and the quadrupole terms
    # accumulate errors.  We remove any point whose radius is within dr_tol
    # of the previous point, keeping the first of each pair.
    # dr_tol = 1 m catches exact duplicates; increase to 100 m to also catch
    # near-duplicates from floating-point noise at the phase boundary.
    dr_tol  = 1.0   # metres
    dr      = np.diff(r_use)
    keep    = np.concatenate(([True], dr > dr_tol))
    n_dupes = int(np.sum(~keep))
    if n_dupes > 0:
        r_use   = r_use[keep]
        rho_use = rho_use[keep]
        if verbose:
            print(f"  Duplicate removal (dr_tol={dr_tol:.0f} m): "
                  f"removed {n_dupes} point(s), {len(r_use)} pts remaining")
    elif verbose:
        print(f"  Duplicate removal: no duplicates found "
              f"(dr_tol={dr_tol:.0f} m)")

    # ── Find the sharp core-atmosphere boundary ───────────────────────────────
    # Use the single largest density DROP (not log-normalised, not spread out).
    # This targets the magma ocean surface specifically.
    drho = np.diff(rho_use)          # negative where density drops outward
    i_jump = int(np.argmin(drho))    # index of steepest drop
    rho_core = float(rho_use[i_jump])
    rho_atm  = float(rho_use[i_jump + 1])
    r_jump   = float(0.5 * (r_use[i_jump] + r_use[i_jump + 1]))

    # Transition zone: expand ±PAD_KM_CORNER km around the main jump.
    # Narrower pad (10-50 km) gives a tight corner smooth matching the
    # standalone ironing script; wider pad (200 km) smooths more broadly.
    pad_km   = PAD_KM_CORNER * 1e3   # convert km to metres
    i_lo = max(0,            np.searchsorted(r_use, r_jump - pad_km) - 1)
    i_hi = min(len(r_use)-1, np.searchsorted(r_use, r_jump + pad_km))
    r_trans_lo = r_use[i_lo]
    r_trans_hi = r_use[i_hi]

    if verbose:
        print(f"  Main density jump at r = {r_jump/1e3:.1f} km: "
              f"rho {rho_core:.1f} → {rho_atm:.1f} kg/m³")
        print(f"  Transition zone (±{PAD_KM_CORNER:.0f} km): "
              f"{r_trans_lo/1e3:.1f} – {r_trans_hi/1e3:.1f} km  "
              f"({i_hi - i_lo} pts)")

    # ── Targeted smoothing ────────────────────────────────────────────────────
    # Stage 1: light global Gaussian (sigma=3) for general noise
    # Stage 2: stronger Gaussian (sigma=sigma_atm) applied only to the
    #          atmosphere region above the transition zone, where numerical
    #          zigzag artefacts from the atmospheric model cause CMS ringing.
    #          The core and transition zone always use sigma=3.
    # Stage 3: tanh replacement in the transition zone.
    # Stage 4: enforce strict monotonicity.
    from scipy.ndimage import gaussian_filter1d

    # Stage 1: light global smooth
    rho_smooth = gaussian_filter1d(rho_use.astype(float), sigma=3)

    # Stage 2: stronger smooth in atmosphere only (r > r_trans_hi)
    if sigma_atm > 3:
        atm_smooth_mask = r_use > r_trans_hi
        if np.any(atm_smooth_mask):
            rho_atm_seg = rho_smooth[atm_smooth_mask].copy()
            rho_atm_seg = gaussian_filter1d(rho_atm_seg.astype(float),
                                            sigma=sigma_atm)
            rho_smooth[atm_smooth_mask] = rho_atm_seg

    # Stage 3: ln(rho) moving-average ironing of outer atmosphere (optional)
    # Removes staircase density artefacts from the Planet_LAB atmospheric grid
    # using a moving average in ln(rho) space, matching the standalone script.
    if IRON_ATMOSPHERE:
        atm_iron_mask = r_use > r_trans_hi
        n_atm_pts = int(np.sum(atm_iron_mask))
        if n_atm_pts > IRON_WINDOW:
            rho_atm_seg  = rho_smooth[atm_iron_mask].copy()
            ln_rho_atm   = np.log(np.maximum(rho_atm_seg, 1e-35))
            # Moving average with reflect padding — matches standalone script
            from scipy.ndimage import uniform_filter1d
            ln_ironed = uniform_filter1d(ln_rho_atm, size=IRON_WINDOW, mode='reflect')
            # Blend over first IRON_WINDOW//2 points to avoid hard edge
            blend_n = min(IRON_WINDOW // 2, n_atm_pts)
            w = np.linspace(0.0, 1.0, blend_n)
            ln_ironed[:blend_n] = ((1.0 - w) * ln_rho_atm[:blend_n]
                                   + w * ln_ironed[:blend_n])
            rho_smooth[atm_iron_mask] = np.exp(ln_ironed)
        if verbose:
            print(f"  Atmosphere ironing: ln(rho) moving avg "
                  f"(window={IRON_WINDOW}) applied")

    # Stage 4: replace transition zone with tanh profile
    rho_lo   = float(rho_smooth[i_lo])
    rho_hi   = float(rho_smooth[i_hi])
    width    = max(r_trans_hi - r_trans_lo, 50e3) / 4.0
    tanh_profile = rho_hi + (rho_lo - rho_hi) * 0.5 * (
        1.0 - np.tanh((r_use - r_jump) / width)
    )
    window = (r_use >= r_trans_lo) & (r_use <= r_trans_hi)
    rho_smooth[window] = tanh_profile[window]

    # Stage 5: enforce strict monotonicity
    for i in range(1, len(rho_smooth)):
        if rho_smooth[i] > rho_smooth[i-1]:
            rho_smooth[i] = rho_smooth[i-1]

    if verbose:
        n_changed = int(np.sum(np.abs(rho_smooth - rho_use) > 1.0))
        iron_str = f" + ln(rho) ironing(w={IRON_WINDOW})" if IRON_ATMOSPHERE else ""
        print(f"  Smoothing: Gaussian(sigma=3) global + "
              f"Gaussian(sigma={sigma_atm}) atmosphere"
              f"{iron_str} + tanh(±{PAD_KM_CORNER:.0f}km) transition")
        print(f"  {n_changed} of {len(rho_use)} points changed by >1 kg/m³")

    # ── Mass-conserving rebin: 40% core / 20% transition / 40% atmosphere ────
    # The previous implementation selected representative grid points by index,
    # which did not conserve mass or the radial moments that control J2/J4.
    # Here each rebinned shell is assigned a volume-averaged density so that the
    # mass of every rebinned region is preserved by construction.
    if len(r_use) > N_layers:
        trans_mask = window
        core_mask  = r_use < r_trans_lo
        atm_mask   = r_use > r_trans_hi

        # Count available points in each region
        n_core_avail  = int(np.sum(core_mask))
        n_trans_avail = int(np.sum(trans_mask))
        n_atm_avail   = int(np.sum(atm_mask))

        # Requested fractions: 40% core / 20% transition / 40% atmosphere
        def _rebin_global_mass_conserving(r_profile, rho_profile, n_shells):
            """Return shell outer radii and volume-averaged densities.

            This version deliberately rebins the *entire* profile at once,
            using equal-volume shells in r^3.  That makes N_layers a true
            global discretization parameter and avoids changing the planet via
            region-dependent shell quotas.
            """
            r_profile = np.asarray(r_profile, dtype=float)
            rho_profile = np.asarray(rho_profile, dtype=float)
            if r_profile.size == 0 or n_shells <= 0:
                return np.array([], dtype=float), np.array([], dtype=float)
            if r_profile.size == 1 or n_shells == 1:
                return np.array([float(r_profile[-1])], dtype=float), np.array([float(rho_profile[-1])], dtype=float)

            # Inner boundary at the center.  The first shell then represents
            # the full central volume out to its outer edge.
            r0 = 0.0
            r1 = float(r_profile[-1])
            if not (r1 > r0):
                return np.array([r1], dtype=float), np.array([float(rho_profile[-1])], dtype=float)

            vol_edges = np.linspace(r0**3, r1**3, int(n_shells) + 1)
            edges = np.cbrt(vol_edges)
            edges[0] = r0
            edges[-1] = r1

            radii_out = np.empty(int(n_shells), dtype=float)
            rho_out = np.empty(int(n_shells), dtype=float)

            for k in range(int(n_shells)):
                rin = float(edges[k])
                rout = float(edges[k + 1])

                inside = (r_profile > rin) & (r_profile < rout)
                r_seg = np.concatenate(([rin], r_profile[inside], [rout]))
                rho_seg = np.interp(r_seg, r_profile, rho_profile)

                mass_integral = np.trapezoid(rho_seg * r_seg**2, r_seg)
                vol_integral = (rout**3 - rin**3) / 3.0
                rho_bar = mass_integral / vol_integral if vol_integral > 0 else float(rho_seg[-1])

                radii_out[k] = rout
                rho_out[k] = rho_bar

            return radii_out, rho_out

        if verbose:
            print(
                f"  Global equal-volume rebin: requested {N_layers} shells; "
                f"input regions available = core {n_core_avail}, transition {n_trans_avail}, atmosphere {n_atm_avail}"
            )
            if n_trans_override is not None:
                print("  Note: n_trans_override is ignored in the global-rebin version.")

        r_cms, d_cms = _rebin_global_mass_conserving(r_use, rho_smooth, int(N_layers))

        # Guard against duplicate shell boundaries at region joins.
        if r_cms.size > 1:
            keep = np.concatenate(([True], np.diff(r_cms) > 0.0))
            r_cms = r_cms[keep]
            d_cms = d_cms[keep]
    else:
        r_cms = r_use.copy()
        d_cms = rho_smooth.copy()

    # ── Ensure strictly positive radii ───────────────────────────────────────
    r_cms = np.maximum(r_cms, 1.0)

    # ── Flip: centre-out → surface-in ────────────────────────────────────────
    r_cms = r_cms[::-1]
    d_cms = d_cms[::-1]

    if verbose:
        n_c = int(np.sum(r_cms[::-1] < r_trans_lo))
        n_t = int(np.sum((r_cms[::-1] >= r_trans_lo) & (r_cms[::-1] <= r_trans_hi)))
        n_a = int(np.sum(r_cms[::-1] > r_trans_hi))
        print(f"  CMS shells: {len(r_cms)} total "
              f"({n_c} core / {n_t} transition / {n_a} atmosphere)")
        print(f"  rho range: {d_cms[-1]:.0f} – {d_cms[0]:.3f} kg/m³")

    # ── Build CMSPlanet with actual model R_1bar and M ────────────────────────
    # This ensures q_rot = Omega^2 * R_1bar^3 / (G*M) is consistent with
    # the empirical calibration which also uses the model R_1bar.
    obs = _make_obs(planet_name, R_1bar=R_ref, M_model=M_model, P_rot=P_rot)
    cp  = CMSPlanet(obs, xlayers=xlayers, verbosity=2 if verbose else 0)
    cp.ai   = r_cms
    cp.rhoi = d_cms

    # ── Run CMS iteration ─────────────────────────────────────────────────────
    if verbose:
        print(f"  Running CMS (xlayers={xlayers}, dJtol=1e-7)...")
    cp.relax_to_HE(fixmass=True)

    # ── Extract results ───────────────────────────────────────────────────────
    J2_cms = float(cp.Js[1])
    J4_cms = float(cp.Js[2])
    J0_cms = float(cp.Js[0])   # sanity check — should be ~-1

    # q consistent with the actual model R_1bar and M
    q = cp._qrot()

    # C/MR^2 already computed above from full-resolution spherical profile
    # Extract zetas and lambdas from CMS output for shape diagnostics
    # out.zetas  : (N_shells, nangles) shape functions — 1 = spherical, <1 = oblate
    # out.lambdas: normalised shell radii (surface-in, normalised to outer radius)
    zetas_out   = cp.CMS.zetas    # (N_shells, 48)
    lambdas_out = cp.CMS.lambdas  # (N_shells,) normalised, surface-in

    return dict(
        J2             = J2_cms,
        J4             = J4_cms,
        J0             = J0_cms,
        C_over_MR2     = C_MR2,
        M_kg           = M_model,
        q              = q,
        n_shells       = len(r_cms),
        r_cms_surfin   = r_cms,       # surface-in, for plotting
        rho_cms_surfin = d_cms,       # surface-in, for plotting
        zetas          = zetas_out,   # shape functions (N_shells, nangles)
        lambdas        = lambdas_out, # normalised shell radii, surface-in
        R_ref          = R_ref,       # reference radius (m) for denormalising
    )


# ── Empirical calibration run (existing method) ───────────────────────────────

def run_empirical(planet_name, r_m, rho_kgm3, R_1bar):
    """Run the existing empirical J2/J4 calibration from calculate_J2.py."""
    d     = PLANET_DATA[planet_name]
    Omega = 2.0 * np.pi / d['P_rot']

    M, I   = J2mod.integrate_mass_moi(r_m, rho_kgm3)
    C_norm = I / (M * R_1bar**2)
    q_rot  = Omega**2 * R_1bar**3 / (G * M)

    coef        = J2mod.build_calibration()
    J2_emp, J4_emp = J2mod.predict_J2_J4(C_norm, q_rot, coef)

    return dict(
        J2         = J2_emp,
        J4         = J4_emp,
        C_over_MR2 = C_norm,
        M_kg       = M,
        q          = q_rot,
    )


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(planet_name, emp, cms_res, suffix):
    """Print a side-by-side comparison of empirical vs CMS results."""
    d   = PLANET_DATA[planet_name]
    sep = "─" * 74

    print()
    print(sep)
    print(f"  {planet_name.upper()} — CMS vs EMPIRICAL COMPARISON   [{suffix}]")
    print(sep)
    print(f"  {'Quantity':<28} {'Empirical':>14} {'CMS':>14} {'Observed':>14}")
    print(f"  {'-'*27} {'-'*14} {'-'*14} {'-'*14}")

    def row(label, e_val, c_val, o_val, fmt='.4f', scale=1.0):
        e_s = f"{e_val*scale:{fmt}}" if e_val is not None else "      —"
        c_s = f"{c_val*scale:{fmt}}" if c_val is not None else "      —"
        o_s = f"{o_val*scale:{fmt}}" if o_val is not None else "      —"
        print(f"  {label:<28} {e_s:>14} {c_s:>14} {o_s:>14}")

    row('M  (10^25 kg)',
        emp['M_kg'], cms_res['M_kg'], d['M'],
        fmt='.4f', scale=1e-25)
    row('C/MR²',
        emp['C_over_MR2'], cms_res['C_over_MR2'], d['C'],
        fmt='.5f')
    row('q_rot',
        emp['q'], cms_res['q'], None,
        fmt='.6f')
    row('J2  × 10⁶',
        emp['J2'], cms_res['J2'], d['J2_obs'],
        fmt='.3f', scale=1e6)
    row('J4  × 10⁶',
        emp['J4'], cms_res['J4'], d['J4_obs'],
        fmt='.3f', scale=1e6)

    # Residuals in sigma
    print()
    print(f"  {'Residuals (Δ/σ_obs)':<28} {'Empirical':>14} {'CMS':>14}")
    print(f"  {'-'*27} {'-'*14} {'-'*14}")

    def resid_row(label, e_val, c_val, obs_val, obs_err):
        e_r = (e_val - obs_val) / obs_err if e_val is not None else None
        c_r = (c_val - obs_val) / obs_err if c_val is not None else None
        e_s = f"{e_r:+.3f} σ" if e_r is not None else "      —"
        c_s = f"{c_r:+.3f} σ" if c_r is not None else "      —"
        print(f"  {label:<28} {e_s:>14} {c_s:>14}")

    resid_row('ΔJ2 / σ_obs', emp['J2'], cms_res['J2'],
              d['J2_obs'], d['J2_err'])
    resid_row('ΔJ4 / σ_obs', emp['J4'], cms_res['J4'],
              d['J4_obs'], d['J4_err'])

    # CMS-only diagnostics
    print()
    print(f"  CMS diagnostics:")
    print(f"    J0 (should be ~-1.0) : {cms_res['J0']:.6f}")
    print(f"    N shells used        : {cms_res['n_shells']}")
    print(sep)
    print()


def write_results(planet_name, emp, cms_res, suffix, outfile):
    """Write comparison results to a text file."""
    import datetime
    d   = PLANET_DATA[planet_name]
    sep = "─" * 74
    lines = []
    lines.append(f"run_cms_single.py  —  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Planet : {planet_name}    Suffix : {suffix}")
    lines.append(sep)
    lines.append(f"  {'Quantity':<28} {'Empirical':>14} {'CMS':>14} {'Observed':>14}")
    lines.append(f"  {'-'*27} {'-'*14} {'-'*14} {'-'*14}")

    def row(label, e_val, c_val, o_val, fmt='.4f', scale=1.0):
        e_s = f"{e_val*scale:{fmt}}" if e_val is not None else "      —"
        c_s = f"{c_val*scale:{fmt}}" if c_val is not None else "      —"
        o_s = f"{o_val*scale:{fmt}}" if o_val is not None else "      —"
        lines.append(f"  {label:<28} {e_s:>14} {c_s:>14} {o_s:>14}")

    row('M  (10^25 kg)',  emp['M_kg'],       cms_res['M_kg'],       d['M'],      '.4f', 1e-25)
    row('C/MR²',         emp['C_over_MR2'], cms_res['C_over_MR2'], d['C'],      '.5f')
    row('q_rot',         emp['q'],          cms_res['q'],          None,         '.6f')
    row('J2  x 1e6',     emp['J2'],         cms_res['J2'],         d['J2_obs'], '.3f', 1e6)
    row('J4  x 1e6',     emp['J4'],         cms_res['J4'],         d['J4_obs'], '.3f', 1e6)
    lines.append("")
    lines.append(f"  {'Residuals (D/sigma_obs)':<28} {'Empirical':>14} {'CMS':>14}")
    lines.append(f"  {'-'*27} {'-'*14} {'-'*14}")
    for label, e_v, c_v, obs_v, obs_e in [
        ('DJ2 / sigma_obs', emp['J2'], cms_res['J2'], d['J2_obs'], d['J2_err']),
        ('DJ4 / sigma_obs', emp['J4'], cms_res['J4'], d['J4_obs'], d['J4_err']),
    ]:
        e_r = (e_v - obs_v) / obs_e
        c_r = (c_v - obs_v) / obs_e
        lines.append(f"  {label:<28} {e_r:>+13.3f}s {c_r:>+13.3f}s")
    lines.append("")
    lines.append(f"  CMS diagnostics:")
    lines.append(f"    J0 (should be -1.0) : {cms_res['J0']:.6f}")
    lines.append(f"    N shells used       : {cms_res['n_shells']}")
    lines.append(sep)

    text = "\n".join(lines)
    with open(outfile, 'w') as f:
        f.write(text + "\n")
    print(f"  Results written to: {outfile}")


def plot_density(r_orig, rho_orig, r_cms_flipped, rho_cms_flipped,
                 R_1bar, suffix, outfile):
    """
    Plot original vs smoothed/subsampled density profile so the
    smoothing can be visually inspected.

    r_cms_flipped / rho_cms_flipped are the arrays as passed to CMS
    (surface-in order), so we flip them back for plotting.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r_cms   = r_cms_flipped[::-1]    # back to centre-out
    rho_cms = rho_cms_flipped[::-1]

    R_E = R_1bar   # normalise to 1-bar radius

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f'{suffix} — density profile (original vs CMS input)', fontsize=10)

    for ax, yscale in zip(axes, ['linear', 'log']):
        ax.plot(r_orig / R_E,   rho_orig,   lw=0.8, color='0.6',
                label='Original (full res)', zorder=1)
        ax.plot(r_cms  / R_E,   rho_cms,    lw=1.5, color='C0',
                marker='o', ms=2, label=f'CMS input ({len(r_cms)} shells)', zorder=2)
        ax.axvline(1.0, color='C3', lw=0.8, ls='--', label='1-bar radius')
        ax.set_xlabel(r'$r\,/\,R_{\rm 1bar}$')
        ax.set_ylabel(r'$\rho$  (kg m$^{-3}$)')
        ax.set_yscale(yscale)
        ax.set_xlim(0, 1.02)
        if yscale == 'linear':
            ax.set_title('Linear scale')
        else:
            ax.set_title('Log scale')
            ax.set_ylim(bottom=0.1)
        ax.legend(fontsize=8)

    fig.tight_layout()
    for ext in ('.png', '.pdf'):
        fig.savefig(outfile.replace('.png', ext), dpi=150)
    print(f"  Density plot saved to: {outfile}  (.png / .pdf)")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────


def plot_cumulative_moments(r_int, rho_int, R_1bar, r_cms_in, rho_cms_in,
                            cms_res, emp, planet_name, suffix, outfile,
                            plot_cms=True):
    """
    Plot cumulative J2 and J4 integrand contributions as a function of
    normalised radius r/R_1bar, for both the full spherical profile
    and the CMS-subsampled profile.

    The cumulative J2 integrand is proportional to rho(r)*r^4 integrated
    from the centre outward; J4 uses rho(r)*r^6.  Both are normalised to
    [0,1] so the curves show fractional contribution as a function of radius.
    The phase boundary is marked.  Where the CMS and empirical curves diverge
    shows where the CMS shell shapes depart from the spherical profile.

    Parameters
    ----------
    r_int      : full-resolution radius array (centre-out, m)
    rho_int    : full-resolution density array (kg/m3)
    R_1bar     : 1-bar radius (m)
    r_cms_in   : CMS shell radii (surface-in order, as stored in cms_res)
    rho_cms_in : CMS shell densities (surface-in order)
    cms_res    : dict returned by run_cms()
    emp        : dict returned by run_empirical()
    planet_name: str
    suffix     : str
    outfile    : str  base filename without extension
    plot_cms   : bool whether to overlay CMS profile integrand (default True)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family":  "sans-serif",
        "font.size":    11,
        "axes.labelsize": 12,
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True,        "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    # Full-resolution spherical integrands
    x_full  = r_int / R_1bar
    dr_full = np.gradient(r_int)
    dJ2_full = rho_int * r_int**4 * np.abs(dr_full)
    dJ4_full = rho_int * r_int**6 * np.abs(dr_full)
    cum_J2_full = np.cumsum(dJ2_full)
    cum_J4_full = np.cumsum(dJ4_full)
    if cum_J2_full[-1] > 0: cum_J2_full /= cum_J2_full[-1]
    if cum_J4_full[-1] > 0: cum_J4_full /= cum_J4_full[-1]

    # CMS shell integrands (flip to centre-out)
    r_cms   = r_cms_in[::-1]
    rho_cms = rho_cms_in[::-1]
    x_cms   = r_cms / R_1bar
    dr_cms  = np.gradient(r_cms)
    dJ2_cms = rho_cms * r_cms**4 * np.abs(dr_cms)
    dJ4_cms = rho_cms * r_cms**6 * np.abs(dr_cms)
    cum_J2_cms = np.cumsum(dJ2_cms)
    cum_J4_cms = np.cumsum(dJ4_cms)
    if cum_J2_cms[-1] > 0: cum_J2_cms /= cum_J2_cms[-1]
    if cum_J4_cms[-1] > 0: cum_J4_cms /= cum_J4_cms[-1]

    # Phase boundary location
    drho   = np.diff(rho_int)
    i_jump = int(np.argmin(drho))
    r_jump = 0.5 * (r_int[i_jump] + r_int[i_jump + 1])
    x_jump = r_jump / R_1bar

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"{planet_name} — cumulative J2 and J4 integrands  [{suffix}]",
        fontsize=10)

    panels = [
        (axes[0],
         "Cumulative J2 integrand (rho r^4 dr)",
         cum_J2_full, cum_J2_cms, emp["J2"]*1e6,
         PLANET_DATA[planet_name]["J2_obs"]*1e6),
        (axes[1],
         "Cumulative J4 integrand (rho r^6 dr)",
         cum_J4_full, cum_J4_cms, emp["J4"]*1e6,
         PLANET_DATA[planet_name]["J4_obs"]*1e6),
    ]

    for ax, ylabel, cf, cc, jv, jo in panels:
        ax.plot(x_full, cf, color="royalblue", lw=1.5,
                label=f"Full profile (empirical) J={jv:.1f}, obs={jo:.1f}")
        if plot_cms:
            ax.plot(x_cms, cc, color="darkorange", lw=1.5, ls="--",
                    label=f"CMS input ({len(r_cms)} shells)")
        ax.axvline(x_jump, color="crimson", lw=1.0, ls=":",
                   label=f"Phase boundary r/R={x_jump:.3f}")
        ax.axvline(1.0, color="0.4", lw=0.8, ls="--", label="1-bar radius")
        ax.set_xlabel("r / R_1bar")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(outfile + ext, dpi=150)
    print(f"  Cumulative moment plot -> {outfile}.png/.pdf")
    plt.close(fig)


def plot_zetas(cms_res, planet_name, suffix, outfile):
    """
    Plot CMS shell shape functions (zetas) as a function of normalised radius.

    zeta(r, theta) is the shape function of each shell — how oblate it is.
    zeta = 1 means perfectly spherical; zeta < 1 means oblate.
    For a well-converged solution, zetas should vary smoothly with radius.
    Wild oscillations near the phase boundary indicate Gibbs-type ringing
    or ill-conditioning in the shape iteration.

    Plots three quantities vs r/R_1bar:
      1. zeta at equator  (cos_theta ~ 0, most oblate)
      2. zeta at pole     (cos_theta ~ 1, least oblate)
      3. zeta_pole - zeta_equator  (oblateness proxy)
    All in surface-in order flipped to centre-out for display.
    The phase boundary is marked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.labelsize": 12, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    zetas   = cms_res["zetas"]      # (N_shells, nangles), surface-in
    lambdas = cms_res["lambdas"]    # normalised radii, surface-in
    R_ref   = cms_res["R_ref"]
    r_cms   = cms_res["r_cms_surfin"]
    rho_cms = cms_res["rho_cms_surfin"]

    # Flip to centre-out for plotting
    zetas_co   = zetas[::-1, :]
    lambdas_co = lambdas[::-1]      # now centre-out, normalised
    r_co       = r_cms[::-1]        # metres, centre-out
    x          = r_co / R_ref       # r / R_1bar

    # Find equator and pole angle indices
    # mus = cos(theta): equator = mu~0, pole = mu~1
    # cms.py uses gauleg(0,1,48) so mus span [0,1]
    # equator is smallest mu (index 0), pole is largest mu (index -1)
    nangles = zetas_co.shape[1]
    i_eq   = 0           # smallest mu ~ equator
    i_pole = nangles - 1 # largest mu ~ pole

    z_eq   = zetas_co[:, i_eq]
    z_pole = zetas_co[:, i_pole]
    z_diff = z_pole - z_eq   # oblateness: positive means pole more spherical

    # Find phase boundary
    rho_co = rho_cms[::-1]
    drho   = np.diff(rho_co)
    i_jump = int(np.argmin(drho))
    x_jump = 0.5 * (x[i_jump] + x[i_jump + 1])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"{planet_name} — CMS shell shape functions (zetas)  [{suffix}]\n"
        f"N={len(x)} shells  |  J2={cms_res['J2']*1e6:.1f}e-6  "
        f"J4={cms_res['J4']*1e6:.2f}e-6",
        fontsize=9)

    def vline(ax):
        ax.axvline(x_jump, color="crimson", lw=1.0, ls=":",
                   label=f"Phase boundary r/R={x_jump:.3f}")
        ax.axvline(1.0, color="0.4", lw=0.8, ls="--", label="1-bar radius")

    # Panel 1: zeta at equator
    axes[0].plot(x, z_eq, color="royalblue", lw=0.8)
    axes[0].axhline(1.0, color="0.6", lw=0.6, ls="--")
    vline(axes[0])
    axes[0].set_xlabel("r / R_1bar")
    axes[0].set_ylabel("zeta (equator)")
    axes[0].set_title("Shape at equator (cos theta ~ 0)")
    axes[0].legend(fontsize=7)

    # Panel 2: zeta at pole
    axes[1].plot(x, z_pole, color="darkorange", lw=0.8)
    axes[1].axhline(1.0, color="0.6", lw=0.6, ls="--")
    vline(axes[1])
    axes[1].set_xlabel("r / R_1bar")
    axes[1].set_ylabel("zeta (pole)")
    axes[1].set_title("Shape at pole (cos theta ~ 1)")
    axes[1].legend(fontsize=7)

    # Panel 3: oblateness proxy
    axes[2].plot(x, z_diff, color="green", lw=0.8)
    axes[2].axhline(0.0, color="0.6", lw=0.6, ls="--")
    vline(axes[2])
    axes[2].set_xlabel("r / R_1bar")
    axes[2].set_ylabel("zeta_pole - zeta_equator")
    axes[2].set_title("Oblateness proxy (should be smooth)")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(outfile + ext, dpi=150)
    print(f"  Zeta shape plot -> {outfile}.png/.pdf")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description='run_cms_single.py — compare CMS vs empirical J2/J4 for a single Planet_LAB run'
    )
    parser.add_argument('--planet',
        choices=['Neptune', 'Uranus'], default='Neptune',
        help='Planet name (default: Neptune)')
    parser.add_argument('--suffix',
        default='17.15ME_SurfaceP_6.50GPa',
        help='Model suffix — folder name and file prefix (default: Neptune example)')
    parser.add_argument('--profile_dir',
        default=None,
        help='Directory containing profile files (default: same as --suffix)')
    parser.add_argument('--N_layers',
        type=int, default=256,
        help='Number of CMS shells (default: 256; fewer + xlayers=-1 is more stable)')
    parser.add_argument('--xlayers',
        type=int, default=-1,
        help='CMS skip-n-spline layers (-1 = full calculation, default; '
             'skip-n-spline spreads errors from clamped layers)')
    parser.add_argument('--n_trans',
        type=int, default=None,
        help='Override number of transition-zone CMS shells (default: 20%% of '
             'N_layers).  Use small values (e.g. 10, 20, 30) to test whether '
             'reducing resolution at the discontinuity reduces Gibbs ringing.')
    parser.add_argument('--sigma_atm',
        type=float, default=20,
        help='Gaussian smoothing sigma for the atmosphere region above the '
             'transition zone (default: 20).  Larger values more aggressively '
             'suppress numerical zigzag artefacts in the low-density envelope. '
             'Set to 3 to match the old uniform behaviour.')
    parser.add_argument('--plot_moments',
        action='store_true', default=False,
        help='Plot cumulative J2 and J4 integrands as a function of '
             'normalised radius, showing where each shell contributes to '
             'the gravitational moments.')
    parser.add_argument('--plot_zetas',
        action='store_true', default=False,
        help='Plot CMS shell shape functions (zetas) vs normalised radius. '
             'Oscillations near the phase boundary diagnose Gibbs ringing '
             'or ill-conditioning in the CMS shape iteration.')
    args = parser.parse_args()

    planet   = args.planet
    suffix   = args.suffix
    folder   = args.profile_dir if args.profile_dir else suffix

    print(f"\nLoading {planet} profile from: {folder}/")

    # ── Load profile ──────────────────────────────────────────────────────────
    try:
        r_m, rho_kgm3, P_bar = J2mod.load_profile(folder, suffix)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")

    print(f"  {len(r_m)} radial points loaded")
    print(f"  r   : {r_m[0]:.3e} – {r_m[-1]:.3e} m")
    print(f"  rho : {rho_kgm3.min():.1f} – {rho_kgm3.max():.1f} kg/m^3")

    # ── Find 1-bar radius and trim ────────────────────────────────────────────
    R_1bar, i_1bar = J2mod.find_1bar_radius(r_m, P_bar)
    r_int   = r_m[:i_1bar+1]
    rho_int = rho_kgm3[:i_1bar+1]
    print(f"  1-bar radius : {R_1bar/1e3:.2f} km  ({len(r_int)} points below 1 bar)")

    # ── Empirical calibration (existing method) ───────────────────────────────
    print(f"\nRunning empirical calibration (calculate_J2.py)...")
    emp = run_empirical(planet, r_int, rho_int, R_1bar)

    # ── CMS (new method) ──────────────────────────────────────────────────────
    print(f"\nRunning CMS (CMSPlanet + cms.py)...")
    cms_res = run_cms(
        planet, r_int, rho_int,
        R_1bar=R_1bar,
        N_layers=args.N_layers,
        xlayers=args.xlayers,
        n_trans_override=args.n_trans,
        sigma_atm=args.sigma_atm,
        verbose=True,
    )

    # ── Side-by-side comparison ───────────────────────────────────────────────
    print_comparison(planet, emp, cms_res, suffix)

    # ── Output files ──────────────────────────────────────────────────────────
    outbase = f"cms_results_{planet}_{suffix}"
    write_results(planet, emp, cms_res, suffix, outbase + '.txt')
    plot_density(
        r_int, rho_int,
        cms_res['r_cms_surfin'], cms_res['rho_cms_surfin'],
        R_1bar, suffix, outbase + '_density.png'
    )
    if args.plot_moments:
        plot_cumulative_moments(
            r_int, rho_int, R_1bar,
            cms_res['r_cms_surfin'], cms_res['rho_cms_surfin'],
            cms_res, emp, planet, suffix,
            outbase + '_cumulative_moments',
            plot_cms=True,
        )
    if args.plot_zetas:
        plot_zetas(cms_res, planet, suffix, outbase + '_zetas')


if __name__ == '__main__':
    main()
