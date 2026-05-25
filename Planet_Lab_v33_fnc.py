# Standard library
import os
import sys
import time
import re
import warnings
from math import sqrt, log as ln, log10 as log, isnan
from random import uniform, randint
from statistics import mean, stdev
from scipy.signal import savgol_filter
from multiprocessing.pool import Pool
from multiprocessing import get_start_method, get_context
from pathlib import Path
# NumPy
import numpy as np

# Definitions from legacy
from math import *
from math import log as ln
from math import log10 as log

# SciPy
from scipy import optimize, integrate
from scipy.optimize import (
    minimize, fminbound, root, least_squares, minimize_scalar, root_scalar
)
from scipy.integrate import quad, solve_ivp
from scipy.special import erf

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge
from matplotlib.ticker import FuncFormatter

# Colorama
from colorama import Fore, Back, Style


from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class PlanetLABResults:
    # ---------------- Core ----------------
    Mc: float                          # kg
    r_core_m: float                    # r4[-1] (m)
    Rcmb_m: float                      # Rcmb4 (m)
    vol_core_m3: float
    bulk_rho_core: float               # kg/m^3
    E_core_thermal_J: float
    E_core_gravPE_J: float
    T_central_K: float

    # -------------- Atmosphere ------------
    T_contact_K: float
    P_surface_bar: float
    P_surface_GPa: float
    P_weight_bar: float                # P_s_atm/1e5
    Matm_kg: float
    Mp_kg: float
    Mp_Earth: float
    Matm_over_Mp: float
    massfrac_atm: float
    massfrac_cond: float
    massfracH2_atm: float
    massfracH2_cond: float
    massfracH2_core: float
    massfracH2_total: float
    massfracH2_gas: float
    RB_km: float
    RB_over_Rc: float
    i_rcb: int
    i_chord: int
    i_chord_p: int
    tau_rcb: float
    rcb_height_km: float
    Rrcb_over_Rc: float
    Trcb_K: float
    P_rcb_bar: float
    Rrcb_cross_over: float
    Teq_K: float
    Trad_K: float
    R1bar_over_Rc: float
    T_mass_weighted_K: float
    T_surface_K: float
    Tint: float
    Lint: float
    Ra: float
    delta_bl: float
    ledoux_top_height_m: float         # height of Ledoux layer top (m)

    # --------------- Planet ----------------
    r_chord_Earth: float
    r_1bar_Earth: float
    r_core_Earth: float
    density_atm_chord: float           # kg/m^3
    vol_atm_chord_m3: float
    density_atm_1bar: float            # kg/m^3
    vol_atm_1bar_m3: float
    vol_planet_m3: float
    vol_frac_core: float
    vol_frac_atm: float
    bulk_rho_planet: float             # kg/m^3
    dH_solvus_J: float
    E_thermal_planet_J: float
    E_gravPE_planet_J: float
    E_total_planet_J: float
    kMI: float

    # -------------- Escape etc. ------------
    Cs_m_per_s: float
    lambda_surface: float
    lambda_rcb: float

    # ----------- Full profiles -------------
    profiles: Dict[str, np.ndarray] = field(default_factory=dict)

    # ----------- Optional: filenames written
    files_written: Optional[list[str]] = None
    output_dir: Optional[str] = None

    def summary(self) -> str:
        """Compact human-readable summary replacing long print blocks."""
        s = []
        s.append("--------------------------------------------------------------------------")
        s.append("CORE STRUCTURE:")
        s.append(f"  Mass of core = {self.Mc/5.972e24:.5f} M_Earth")
        s.append(f"  Radius of core = {self.r_core_Earth:.5f} R_Earth")
        s.append(f"  Metal-core radius = {self.Rcmb_m/6.371e6:.3f} R_Earth")
        s.append(f"  Mass frac H2 (core) = {self.massfracH2_core:10.3f}")
        s.append(f"  Core volume = {self.vol_core_m3:10.3e} m^3")
        s.append(f"  Bulk core density = {self.bulk_rho_core:10.3f} kg/m^3")
        s.append(f"  Core thermal E = {self.E_core_thermal_J:10.5e} J")
        s.append(f"  Core grav PE = {self.E_core_gravPE_J:10.5e} J")
        s.append(f"  Central temperature = {self.T_central_K:.4f} K")
        s.append("")
        s.append("ATMOSPHERE STRUCTURE:")
        s.append(f"  Surface T = {self.T_contact_K:10.3f} K")
        s.append(f"  P_surface = {self.P_surface_bar:10.3e} bar ({self.P_surface_GPa:10.3e} GPa)")
        s.append(f"  P_at_base(weight,g0) = {self.P_weight_bar:10.3e} bar")
        s.append(f"  M_atm = {self.Matm_kg:10.4e} kg; M_p = {self.Mp_kg:10.3e} kg = {self.Mp_Earth:10.3f} M_Earth")
        s.append(f"  Matm/Mp = {self.Matm_over_Mp:10.4e}")
        s.append(f"  Gas mass % of planet = {self.massfrac_atm*100:8.3f} %")
        s.append(f"  Condensate mass % (returned to core) = {self.massfrac_cond*100:8.3f} %")
        s.append(f"  x_H2 gas/cond/core/total = {self.massfracH2_atm:7.4f}, {self.massfracH2_cond:7.4f}, {self.massfracH2_core:7.4f}, {self.massfracH2_total:7.4f}")
        s.append(f"  H2 gas mass frac of planet = {self.massfracH2_gas:7.4f}")
        s.append(f"  RB = {self.RB_km:10.4e} km; RB/Rc = {self.RB_over_Rc:10.4f}")
        s.append(f"  Ra envelope = {self.Ra:10.2e}")
        s.append(f"  delta_boundary_layer (m) = {self.delta_bl:10.3e}")
        s.append(f"  Ledoux layer top height (m) = {self.ledoux_top_height_m:10.3e}")
        s.append(f"  tau_rcb = {self.tau_rcb:10.2e}")
        s.append(f"  Lint = {self.Lint:10.3e}")
        s.append(f"  Height rcb = {self.rcb_height_km:10.3e} km; Rrcb/Rc = {self.Rrcb_over_Rc:.4f}")
        s.append(f"  Trcb = {self.Trcb_K:10.3f} K; P_rcb = {self.P_rcb_bar:.4e} bar; Rrcb(cross-over)/Rc = {self.Rrcb_cross_over:.4f}")
        s.append(f"  Teq = {self.Teq_K:10.3f} K; Trad = {self.Trad_K:10.3f} K; R(1 bar)/Rc = {self.R1bar_over_Rc:8.3f}")
        s.append(f"  Mass-weighted T = {self.T_mass_weighted_K:10.1f} K; Surface T = {self.T_surface_K:10.1f} K")
        s.append(f"  Teq = {self.Teq_K:10.3f} K; Trad = {self.Trad_K:10.3f} K; Tint r[0] = {self.Tint:10.3f} K")
        s.append("")
        s.append("PLANET STRUCTURE:")
        s.append(f"  M_p = {self.Mp_Earth:8.3f} M_Earth")
        s.append(f"  R_planet (chord tau) = {self.r_chord_Earth:10.3f} R_Earth")
        s.append(f"  R_planet (1 bar)    = {self.r_1bar_Earth:10.3f} R_Earth")
        s.append(f"  R_core              = {self.r_core_Earth:10.3f} R_Earth")
        s.append(f"  <rho_atm> (R_chord) = {self.density_atm_chord:10.3f} kg/m^3; Vol_atm(chord) = {self.vol_atm_chord_m3:10.3e} m^3")
        s.append(f"  <rho_atm> (1 bar)   = {self.density_atm_1bar:10.3f} kg/m^3; Vol_atm(1bar)  = {self.vol_atm_1bar_m3:10.3e} m^3")
        s.append(f"  Vol fractions: core = {self.vol_frac_core:10.3f}, atm = {self.vol_frac_atm:10.3f}")
        s.append(f"  Bulk rho (chord) = {self.bulk_rho_planet:10.3f} kg/m^3")
        s.append(f"  Latent heat (atm) = {self.dH_solvus_J:.3e} J")
        s.append(f"  E_thermal = {self.E_thermal_planet_J:10.5e} J; E_gravPE = {self.E_gravPE_planet_J:10.5e} J; E_total = {self.E_total_planet_J:10.6e} J")
        s.append(f"  k (MoI to 1 bar) = {self.kMI:8.3f}")
        s.append("")
        s.append("ESCAPE FACTORS:")
        s.append(f"  Cs = {self.Cs_m_per_s:10.3e} m/s")
        s.append(f"  lambda_surface = {self.lambda_surface}")
        s.append(f"  lambda_rcb = {self.lambda_rcb}")
        return "\n".join(s)

print('  ')
print('  ')
print ('  +-----------------------------------------+')
print ('  |               PLANET LAB                |')
print ('  +-----------------------------------------+')
print('')


TABLE_FILE = "TABLE_H_Trho_v1.txt"

# ---- module-scope cache ----
_H2_LOADED = False

logTK = logPGPa = logrho = grad_ad = None
T_row = P_row = rho_row = None
tkey_row = unique_tkey = slice_start = slice_end = None


def _resolve_table_path(table_file: str) -> str:
    if os.path.isabs(table_file):
        return table_file
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, table_file)


def load_H2_table(force_reload: bool = False) -> None:
    global _H2_LOADED
    global logTK, logPGPa, logrho, grad_ad
    global T_row, P_row, rho_row
    global tkey_row, unique_tkey, slice_start, slice_end

    if _H2_LOADED and (not force_reload):
        return

    path = _resolve_table_path(TABLE_FILE)
    values = np.genfromtxt(path, dtype=float)

    if values is None or values.size == 0:
        raise RuntimeError(f"H2 EOS table read failed or empty: {path}")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] < 4:
        raise RuntimeError(f"H2 EOS table has too few columns: {path} (shape={values.shape})")

    logTK   = values[:, 0].astype(np.float64)
    logPGPa = values[:, 1].astype(np.float64)
    logrho  = values[:, 2].astype(np.float64)
    grad_ad = values[:, -1].astype(np.float64)

    # Precompute SI row quantities
    T_row   = np.power(10.0, logTK)                   # K
    P_row   = np.power(10.0, logPGPa) * 1.0e9         # Pa
    rho_row = 1000.0 * np.power(10.0, logrho)         # kg/m^3

    # Integer key for rounded 0.1-dex logT
    tkey_row = np.rint(logTK * 10.0).astype(np.int32)

    # Build contiguous slices by tkey (assumes table grouped by logT)
    unique_tkey, first_idx = np.unique(tkey_row, return_index=True)
    order = np.argsort(first_idx)
    unique_tkey = unique_tkey[order].astype(np.int32)
    first_idx = first_idx[order].astype(np.int64)

    slice_start = first_idx
    slice_end = np.empty_like(slice_start)
    slice_end[:-1] = slice_start[1:]
    slice_end[-1] = len(tkey_row)

    # ---- validate for exactly current failure mode ----
    if unique_tkey.ndim != 1 or unique_tkey.size < 2:
        raise RuntimeError(f"Bad unique_tkey: ndim={getattr(unique_tkey,'ndim',None)} size={getattr(unique_tkey,'size',None)}")
    if slice_start.ndim != 1 or slice_end.ndim != 1:
        raise RuntimeError("Bad slice indices arrays (not 1-D).")

    _H2_LOADED = True


def _nearest_tkey_index(tkey: int) -> int:
    # assumes load_H2_table already ran
    k = int(np.searchsorted(unique_tkey, tkey))
    if k <= 0:
        return 0
    if k >= unique_tkey.size:
        return unique_tkey.size - 1
    return k if abs(int(unique_tkey[k]) - tkey) < abs(int(unique_tkey[k-1]) - tkey) else (k - 1)


def _bracket_tkey_indices(logT: float):
    """
    Return (k_lo, k_hi, w_hi) where k_lo and k_hi are indices into unique_tkey
    that bracket logT, and w_hi is the linear interpolation weight for k_hi.

    If logT is outside the table range, clamps to the nearest end and returns
    w_hi = 0 (pure k_lo) or w_hi = 1 (pure k_hi = k_lo+1 at upper end).
    This replaces the old nearest-neighbor _nearest_tkey_index snap, which
    caused step-discontinuities in Z and hence jagged density profiles.
    """
    # unique_tkey stores rounded 0.1-dex logT keys as integers (logT*10)
    tkey_float = logT * 10.0          # continuous version of tkey

    k = int(np.searchsorted(unique_tkey, tkey_float, side='right'))

    # clamp to valid bracket
    if k <= 0:
        return 0, 0, 0.0
    if k >= unique_tkey.size:
        n = unique_tkey.size - 1
        return n, n, 0.0

    k_lo = k - 1
    k_hi = k

    tkey_lo = float(unique_tkey[k_lo])
    tkey_hi = float(unique_tkey[k_hi])
    span = tkey_hi - tkey_lo
    if span <= 0.0:
        return k_lo, k_lo, 0.0

    w_hi = (tkey_float - tkey_lo) / span
    w_hi = float(np.clip(w_hi, 0.0, 1.0))
    return k_lo, k_hi, w_hi


def _interp_Z_at_slice(k: int, P_target: float) -> float:
    """
    For isotherm slice k, look up Z = (molar_mass * P) / (R * T * rho)
    at P_target by linear interpolation in P within the slice.
    Returns Z, which is the compressibility factor for H2.
    """
    molar_mass = 0.002
    Rgas = 8.3145

    i1 = int(slice_start[k])
    i2 = int(slice_end[k])

    P_sl  = P_row[i1:i2]
    T_sl  = T_row[i1:i2]
    rho_sl = rho_row[i1:i2]

    j_local = int(np.searchsorted(P_sl, float(P_target), side='left'))

    # clamp: if P_target is beyond table, use edge value (no extrapolation in P)
    if j_local <= 0:
        j = i1
        P_ = float(P_sl[0]);  T_ = float(T_sl[0]);  rho_ = float(rho_sl[0])
    elif j_local >= len(P_sl):
        j = i2 - 1
        P_ = float(P_sl[-1]); T_ = float(T_sl[-1]); rho_ = float(rho_sl[-1])
    else:
        # linear interpolation between j_local-1 and j_local in P
        jlo = j_local - 1
        jhi = j_local
        Plo = float(P_sl[jlo]); Phi = float(P_sl[jhi])
        dP  = Phi - Plo
        if dP <= 0.0:
            P_ = float(P_sl[jlo]); T_ = float(T_sl[jlo]); rho_ = float(rho_sl[jlo])
        else:
            wp = (float(P_target) - Plo) / dP
            wp = float(np.clip(wp, 0.0, 1.0))
            P_   = Plo + wp * (Phi - Plo)
            T_   = float(T_sl[jlo]) + wp * (float(T_sl[jhi]) - float(T_sl[jlo]))
            rho_ = float(rho_sl[jlo]) + wp * (float(rho_sl[jhi]) - float(rho_sl[jlo]))

    Z = (molar_mass * P_) / (Rgas * T_ * max(rho_, 1e-30))
    if (not np.isfinite(Z)) or (Z <= 0.0):
        return 1.0
    return float(Z)


def H2_density(T_target: float, P_target: float) -> float:
    """
    Return the H2 compressibility factor Z = (mu * P) / (R * T * rho).

    Uses bilinear interpolation in (logT, P) between the two bracketing
    isotherms in the Chabrier EOS table, replacing the old nearest-neighbour
    isotherm snap that caused step-discontinuities in rho vs r.
    """
    if not _H2_LOADED:
        load_H2_table()

    logT = np.log10(float(T_target))
    k_lo, k_hi, w_hi = _bracket_tkey_indices(logT)

    Z_lo = _interp_Z_at_slice(k_lo, P_target)
    if k_hi == k_lo:
        return Z_lo

    Z_hi = _interp_Z_at_slice(k_hi, P_target)
    return float((1.0 - w_hi) * Z_lo + w_hi * Z_hi)


def _interp_grad_at_slice(k: int, P_target: float) -> float:
    """
    For isotherm slice k, return grad_ad at P_target by linear interpolation in P.
    """
    i1 = int(slice_start[k])
    i2 = int(slice_end[k])

    P_sl    = P_row[i1:i2]
    grad_sl = grad_ad[i1:i2]

    j_local = int(np.searchsorted(P_sl, float(P_target), side='left'))

    if j_local <= 0:
        g = float(grad_sl[0])
    elif j_local >= len(P_sl):
        g = float(grad_sl[-1])
    else:
        jlo = j_local - 1
        jhi = j_local
        Plo = float(P_sl[jlo]); Phi = float(P_sl[jhi])
        dP  = Phi - Plo
        if dP <= 0.0:
            g = float(grad_sl[jlo])
        else:
            wp = float(np.clip((float(P_target) - Plo) / dP, 0.0, 1.0))
            g  = float(grad_sl[jlo]) + wp * (float(grad_sl[jhi]) - float(grad_sl[jlo]))

    return g if np.isfinite(g) else 0.3


def grad_H2_Chabrier(T_target: float, P_target: float) -> float:
    """
    Return the adiabatic gradient grad_ad from the Chabrier H2 EOS table,
    using bilinear interpolation in (logT, P) between bracketing isotherms.
    Replaces the old nearest-neighbour isotherm snap for consistency with
    the updated H2_density interpolation.
    """
    if not _H2_LOADED:
        load_H2_table()

    logT = np.log10(float(T_target))
    k_lo, k_hi, w_hi = _bracket_tkey_indices(logT)

    g_lo = _interp_grad_at_slice(k_lo, P_target)
    if k_hi == k_lo:
        return g_lo

    g_hi = _interp_grad_at_slice(k_hi, P_target)
    return float((1.0 - w_hi) * g_lo + w_hi * g_hi)


def rho_gas(P: float, T: float, mw: float) -> float:
    Rgas = 8.3145
    rho_ideal = float(mw) * float(P) / (Rgas * float(T))
    z = 1.0
    if float(P) > 0.05 * 1.0e9:
        z = H2_density(float(T), float(P))
    return float(rho_ideal / max(z, 1e-30))

#-----------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------

def Planet_LAB(
    bulk_wt_H2_percent=3.0,
    Mp_target_kg=6.0 * 5.972e24,    # Earth masses → kg
    Teq_K=1000.0,
    DeltaTrad_K=1.0,
    mw_main_amu=2.3,
    mw_silicate_amu=100.0,
    r_main_m=289.0e-12,             # pm → m
    gamma_gas=1.4,
    gamma_vis_th=1.0,               # T_eddington: kappa_vis / kappa_th
    surface_P_GPa=4.7,
    dTcdP_deg_per_GPa=-600.0,
    tau_critical_chord=1.0,
    Pnebula_Pa=1.0e-19 * 1.0e5,     # bar → Pa
    alpha_smooth_gradT=1.00,
    Inhibit=False,        # "ON" default → True
    n_radial_steps=120000,
    RB_over_rmax=13.0,   # Default goes with 120000 steps
    plot_radius_limit_RE=4.0,
    rho0_silicate_gcc=0.0,
    metal_mass_deficit=0.01,
    structure_layers=[1],           # silicate-only planet
    cmf_core_metal=0.33,
    output_suffix="_default",
    dr_logistic_slope=14,
    Ra = 1.0e12, # envelope Rayleigh number
    i_rcb = None,
    write_files=False,
    output_dir="."
):

    # CONSTANTS USED THROUGHOUT IN SI UNITS
    G=6.67e-11 # m^3 kg^-1 s^-2
    nucleon_mass = 1.6726e-27  # kg/nucleon
    kb = 1.38065e-23 # J/K
    Rgas=8.3144598
    pi = 3.14159265
    Av_number=6.02214e23 # particles per mole
    sigma=5.670374419e-8 # W/(m^2 K^4)
    L_solar=3.846e26 # Watts
    Hmultiplier=1.0
    Tmultiplier=1.0
    Mearth=5.972e24 # kg
    #tau_critical=0.667 # optical depth beyond which convection dominates over radiation
    

    # ==== legacy-name normalization (first lines of Planet_LAB) ====
    # safe default for mutable argument
    if structure_layers is None:
        structure_layers = [1]

    bulk              = bulk_wt_H2_percent
    Mp_target         = Mp_target_kg
    Mp_target_earth   = Mp_target / Mearth
    Teq               = Teq_K
    Trad              = Teq_K + DeltaTrad_K
    m_main            = mw_main_amu
    m_heavy           = mw_silicate_amu
    r_main            = r_main_m
    target            = surface_P_GPa
    dTcdP             = dTcdP_deg_per_GPa
    tau_critical      = tau_critical_chord
    Pnebula           = Pnebula_Pa
    nr                = int(n_radial_steps)
    rad_lim           = plot_radius_limit_RE
    rho_0_in          = rho0_silicate_gcc
    xrho              = metal_mass_deficit
    structure         = structure_layers[:] if isinstance(structure_layers, list) else [structure_layers]
    cmf               = cmf_core_metal
    cmf_mix           = cmf_core_metal
    suffix            = output_suffix
    sharp             = int(dr_logistic_slope)
    # ===============================================================

    # ---------------- Output directory / suffix logic ----------------
    # Prefer GUI-provided suffix
    if output_suffix is not None and str(output_suffix).strip() not in ["", "_default"]:
        suffix = str(output_suffix)
        if not suffix.startswith("_"):
            suffix = "_" + suffix
    else:
        suffix = f"_{Mp_target_kg/5.972e24:.2f}ME_SurfaceP_{surface_P_GPa:.2f}GPa"

    out_base = Path(output_dir).expanduser().resolve() if output_dir is not None else Path(".").resolve()
    out_dir = out_base / suffix.lstrip("_")

    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------

    def open_with_suffix(file_name: str, mode: str):
        """
        Open a file inside output_dir/<suffix>/ with name <base><suffix><ext>
        """
        base, ext = os.path.splitext(file_name)
        return open(out_dir / f"{base}{suffix}{ext}", mode)



    # MODIFY UNITS OF INPUT MOLECULAR WEIGHTS
    mw=m_main/1000.0 # kg per mole
    m_main=m_main*nucleon_mass # kg per single molecule of main gas, H2
    mw_h=m_heavy/1000.0
    m_heavy=m_heavy*nucleon_mass # kg per single molecule of the second species, usually MgSiO3 in this app

    # ADDITIONAL MOLECULAR WEIGHT TERMS USED BY SEVERAL FUNCTIONS, mw, mw_h = input converted to kg/mole
    MWH2=mw # kg/mole
    MWMgSiO3=mw_h # kg/mole
    MW1=mw*1000.0 # amu for light main gas
    MW2=mw_h*1000.0 # amu for heavy


    # INITIAL NOMINAL Radius of planet core from mass, refined later
    # R_s is surface radius, or radius of the core

    R_earth_meters=6371.0*1000.0
    R_s=6371.0*1000.0*(Mp_target_earth-(bulk/100)*Mp_target_earth)**(0.25) # meters
    print("")
    print('FIRST ITERATION PLANET DESCRIPTION:')
    print(" Nominal initial radius = ", "{:.5e}".format(R_s/1000.0)," km")
    print(" Target mass of planet = ", Mp_target," kg")

    # Report equilibrium temperature Teq in K
    print(' Equilibrium T = %10.2f K' %Teq)


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #                                          CORE FUNCTIONS

# -------------- FUNCTIONS FOR MgSiO3 melt EOS ------------- #
     

    # -------------- MgSiO3 melt Adiabatic T ------------- #

    def T_S_MgSiO3(eta,T0):
        """
        Adiabatic temperature for MgSiO3 melt after deKoker and Stixrude (2009) 
        as fit by Wolf (2018). This is for the reference adiabat. 
        Subscript zero implies reference T and Gruneisen parameter at that T.
        """
        gamma_o=0.282 # at 3000 K reference T
        gamma_o_prime=-1.35 # at 3000 K reference T
        a1=6.0*gamma_o
        a2=-12.0*gamma_o+36.0*gamma_o**(2.0)-18.0*gamma_o_prime
        f=0.5*((eta)**(2.0/3.0)-1.0) # should be f = 1/2 [(vo/v)^(2/3)-1]
        arg=1.0+a1*f+0.5*a2*f**(2.0)
        T_S=T0*sqrt(arg)
        
        return T_S

    # -------------------------------------------------------- #
    
    def potential_T_from_surface(Tsurf, eta, gamma0=0.282, gamma0_prime=-1.35):
        """
        New: convert boundary temperature at finite compression (eta) to potential T0 (reference at eta=1).
        Uses the same square-root Padé form as T_S_MgSiO3.
        """
        a1 = 6.0 * gamma0
        a2 = -12.0 * gamma0 + 36.0 * gamma0**2 - 18.0 * gamma0_prime
        f  = 0.5 * (eta**(2.0/3.0) - 1.0)
        scale = (1.0 + a1*f + 0.5*a2*f*f)**0.5
        return Tsurf / max(scale, 1e-12)

    # -------------------------------------------------------- #

    def compute_T0_potential_from_surface(Psurf, rho_0, Tsurf, wtfracH2=0.0):
        """
        New: convert a finite-pressure boundary temperature (Tsurf at Psurf)
        into the potential temperature T0 consistent with the adiabat.
        Uses the same EOS and rho_0 as the structure solver.
        Needed to get eta from Tsurf and Psurf for use to get T0.
        wtfracH2: weight fraction of H2 in the melt (default 0 = pure MgSiO3).
        """
        # density at the boundary using current (provisional) Tsurf
        rho_surf_SI = EOS_MgSiO3(Psurf, rho_0, Tsurf, wtfracH2)  # kg/m^3
        eta_surf     = rho_surf_SI / (rho_0 * 1000.0)  # rho_0 input is g/cc
        return potential_T_from_surface(Tsurf, eta_surf)


        
    # -------------- MgSiO3 melt-H2 mixture rho_0 ------------ #
    def rho_0_mix(wtfracH2,MW_sil,MW_H2,rho0sil,rho0H2):
        """
        Returns density at surface of supercritical MgSiO3-H2
        mixture, rho_0 in g/cm^3, based on: 
        wtfracH2 = weight fraction of mixture that is H2;
        Mw_sil, Mw_H2 = molecular weights of endmembers in g/mol;
        rho0sil = density at surface for pure MgSiO3 in g/cm^3;
        rho0H2  = corresponding density of H2 in g/cm^3;
        xMgSiO3, xH2 = mole fractions for mixture.  
        """
        
        wtfracMgSiO3=1-wtfracH2
        xMgSiO3=(wtfracMgSiO3/MW_sil)/((wtfracMgSiO3/MW_sil)+wtfracH2/MW_H2)
        xH2=1-xMgSiO3
        
        n_over_V_H2=rho0H2/MW_H2
        n_over_V_sil=rho0sil/MW_sil
        n_over_V_mix=xH2*n_over_V_H2 + xMgSiO3*n_over_V_sil
        
        MW_mix=xMgSiO3*MW_sil+xH2*MW_H2
        
        rho_mix = MW_mix*n_over_V_mix
        
        return rho_mix

    #-------------------------------------------------------- #
        

    # -------------- FUNCTION FOR MgSiO3 melt EOS ------------- #
     
    def func_MgSiO3(eta, P, rho_0, T0, wtfracH2=0.0):
        """
        P - Vinet EOS function to be minimized in order to solve for MgSiO3
        MELT DENSITY upon adiabatic compression.
        wtfracH2: weight fraction of H2 in the melt. Controls kprime via a
        linear interpolation between pure MgSiO3 (wtfracH2=0, kprime=7.42)
        and 4 wt% H2 (wtfracH2=0.04, kprime=4.47), extrapolating linearly
        beyond that range.
        [rest of docstring unchanged]
        """
        
        P_GPa=P/1.0e9
        # CONSTANTS
        m=3.0/5.0
        kb=8.61733e-5 # eV/K per atom
        cm3_per_A3=1.0e-24
        J_per_eV=1.60218e-19
        Avagadros_no=6.023e23 # atoms per mole
        J_per_mole_per_eV=Avagadros_no*J_per_eV # Use to convert eV/atom to J/mole atoms
        eVcm3_to_GPa=1.60217e-22 # Convert P in eV/cm^3 to GPa
        Vo=14.74*cm3_per_A3 # units = cm^3/atom, where cm^3/angstroms^3 = 1.0e-24
        k= 9.77 # GPa

        # kprime: linear in wtfracH2, anchored at two points:
        #   wtfracH2 = 0.00 -> kprime = 7.42  (pure MgSiO3)
        #   wtfracH2 = 0.04 -> kprime = 4.47  (4 wt% H2)
        kprime_pure  = 7.42
        kprime_4pct  = 4.47
        kprime = kprime_pure + (kprime_4pct - kprime_pure) * (wtfracH2 / 0.04)
        #kprime = max(kprime, kprime_4pct)   # floor at 4 wt% anchor: no extrapolation below 4.47
        #kprime = 7.42
        
        #CURRENT COMPRESSIBILITY FACTOR
        rho = eta*rho_0 # eta is 1/the volumetric compression factor V/Vo, or Vo/V
        
        #THERMAL CONTRIBUTIONS TO PRESSURE
        Tref=3000.0 # reference T for reference adiabat parameters, 3000
        T0S= T_S_MgSiO3(eta,Tref) # Defines reference adiabatic T profile, T_0S(P)
        T_S= T_S_MgSiO3(eta,T0) # Defines T profile along our adiabat, T_S(P)
        
        E0=-6.850 # eV/atom
        # Thermal coefficients for volume compression
        bcoeff=np.zeros(5)
        bcoeff[0]= 1.118 # eV/atom
        bcoeff[1]= -0.05
        bcoeff[2]= 2.1
        bcoeff[3]= 12.9
        bcoeff[4]= 15.5
        b=0.0 # eV/atom
        for i in range(5):
            b=b+bcoeff[i]*(1/eta-1)**float(i)
            
        V=Vo/eta # to convert to cm^3/mole use V*Avagadros_no*1.0e-24*5 where 5 is atoms pfu
        bprime=0.0 #  eV/cm^3, pressure
        for i in range(5):
            rn=float(i)
            bprime=bprime+bcoeff[i]*(rn/V)*(1/eta-1)**(rn-1.0)
        
        # Gruneisen parameter along reference adiabat
        gamma_o=0.282 # Gruneisen parameter at 3000 K
        gamma_o_prime=-1.35 # T derivative of Gruneisen parameter at 3000 K
        a1=6.0*gamma_o
        a2=-12.0*gamma_o+36.0*gamma_o**(2.0)-18.0*gamma_o_prime
        f=0.5*((eta)**(2.0/3.0)-1.0) # should be f = 1/2 [(vo/v)^(2/3)-1]
        global gamma0S # gamma0S is global for reporting, comment out for actual use
        gamma0S=((2.0*f+1)*(a1+a2*f))/(6.0*(1.0+a1*f+0.5*a2*f**2))
        
        # Dimensionless thermal deviation from the zero-P temperature
        fT=(T_S/T0)**m -1.0
        
        # T-derivative of thermal deviation, e.g., Eqn B.1 Wolf (2018), K^-1
        def fTprime_fnc(T,T0,m):
            fTp=(m/T0)*(T/T0)**(m-1)
            return fTp
        # Derivative at T_S(P), our adiabat
        fTprimeT=fTprime_fnc(T_S,T0,m)
        # Derivative at zero-P T, T0
        fTprimeT0=fTprime_fnc(T0,T0,m)
        # Derivative at corresponding reference adiabatic T, T0S(P)
        fTprimeT0S=fTprime_fnc(T0S,T0,m)

        # dP due to thermal energy, fT of T_o is 0, at constant V, P will rise
        dP_E=-bprime*fT
        dP_E_GPa=dP_E*eVcm3_to_GPa
        
        # dP due to deviations along the isentrope,
        # Cv is the per atom isochoric heat capacity along the reference adiabat
        N=5.0 #atoms per formula unit
        Cv=b*fTprimeT0S+(3.0/2.0)*N*kb # eV/(K atom)
        dP_S=((bprime/(m-1.0))*(T_S*(fTprimeT-fTprimeT0S)-T0*(fTprimeT0-fTprimeT0S))
              +gamma0S*Cv*(T_S-T0)/V)
        dP_S_GPa=dP_S*eVcm3_to_GPa
        
        # Return function to mimimize in order to find eta = rho/rho_o at this pressure
        func = (P_GPa - 3.0 * k * eta**(2.0/3.0)*(1.0-eta**(-1.0/3.0))
               *exp((3.0/2.0)*(kprime-1.0)*(1.0-eta**(-1.0/3.0)))-dP_E_GPa-dP_S_GPa)
               
        return func
    # -------------------------------------------------------- #



    # ---------- MgSiO3 melt density calculation ------------ #

    def EOS_MgSiO3(P, rho_0, T0, wtfracH2=0.0):
        """
        Returns density of MgSiO3 melt in kg / m^3.
        wtfracH2: weight fraction of H2 in the melt (default 0 = pure MgSiO3).
        kprime varies linearly from 7.42 (pure) to 4.47 (4 wt% H2).
        """
        
        eta = optimize.root_scalar(func_MgSiO3, bracket=[0.000001, 5000.0],
                                   args=(P, rho_0, T0, wtfracH2), method='brentq')
        rho = rho_0 * eta.root
        
        return rho*1000 # return density in SI units, kg/m^3

    # -------------------------------------------------------- #


    # ---------------- FUNCTIONS FOR METAL EOS --------------- #


    # -------------- Iron-rich melt Adiabatic T ------------- #

    def T_S_Metal(x,gamma):
        """
        Temperature relative to T0 for constant Gruneisen parameter
        and compression factor x = V/V_0 = rho_0/rho for liquid metal
        """
        Vo=29.283 # reference T for Vo at zero pressure
        V=Vo*x
        T0=1811.0
        T_S=T0+T0*gamma*((Vo-V)/Vo)
        
        return T_S

    # -------------------------------------------------------- #


    # ----------- Iron-rich melt Gruneisen parameter --------- #

    def gamma_Metal(x):
        """
        Gruneisen parameter for metal From Kuwayama et al. (2020).
        x is V/V_0 = rho_0/rho.
        """
        gamma_0=2.02 # Gruneisen parameter at zero P
        b=0.63
        gamma=gamma_0*(x**b)
        return gamma

    # -------------------------------------------------------- #


    # -------------- FUNCTION FOR IRON-rich melt EOS ------------- #
     
    def func_Metal(x,P,rho_0):
        """
        Provides P - (Vinet EOS + Pthermal) for Mie-Gruneisen EoS after Kuwayama + (2020).
        1. x is the compressibility factor written here as V/Vo, or rho_o/rho.
        2. P is total pressure, both static and thermal combined.
        3. rho_0 is the density at 0 P;
        Based on EOS of Kuwayama et al. (2020) 
        """
        
        P_GPa=P/1.0e9 # Convert input pressure in Pa to GPa
        
        # CONSTANTS
        cm3_per_A3=1.0e-24
        J_per_cm3_to_GPa=0.001
        Avagadros_no=6.023e23 # atoms per mole
        Vo=29.283 # units = cm^3/mole, based on fcc iron
        k=82.1 # GPa Kuwayama et al.
        kprime= 5.80 # Kuwayama et al.
        gamma_0=2.02 # Gruneisen parameter at zero P
        b=0.63
        
        # DENSITY CALCULATION
        #rho_0=7.03*(1-xrho) # g/cc, zero pressure, 1811 K, the melting point of Fe
        rho = rho_0/x # x is the volumetric compression factor V/Vo, rho_o/rho

        # Gruneisen parameter, gamma(P)
        gamma=gamma_Metal(x)
        
        # Temperature, due to compression of liquid metal only
        T_S=T_S_Metal(x,gamma)
        
        # Thermal pressure in GPa
        def Eth(x,T):
            Rgas=8.3144598
            e0=0.68e-4 # K^-1
            gmetal=-1.0 # at 1811
            Eth=3.0*Rgas*(T+e0*(x**gmetal))
            return Eth
        V=Vo*x # cm^3/mole
        Tref = 1811.0
        dPth=0.001*(gamma/V)*(Eth(x,T_S)-Eth(x,Tref))
        
        # Return function to mimimize in order to find eta = rho/rho_o at this pressure
        func = (P_GPa - 3.0 * k * x**(-2.0/3.0)*(1.0-x**(1.0/3.0))
               *exp((3.0/2.0)*(kprime-1.0)*(1.0-x**(1.0/3.0)))-dPth)
               
        return func
    # -------------------------------------------------------- #



    # ---------- Metal melt density calculation ------------ #

    def EOS_Metal(P,xrho):
        """
        Returns density of Fe-rich metal melt in kg / m^3.
        In 'structure_ODEs', metal corresponds to layer 1.
        See Seager et al. (2007)
        """
        rho_0=7.03*(1-xrho) # 1800 K, zero pressure
        
        x=optimize.root_scalar(func_Metal,bracket=[1.0e-5,1.0e6],args=(P,rho_0),method='brentq')
        rho=rho_0/x.root
        
        return rho*1000 # return density in SI units, kg/m^3

    # -------------------------------------------------------- #



    # ------------------ DIFFERENTIAL EQUATIONS -------------- #
    #  (modify to include temperature for equations of state)

    def structure_ODEs(
        m,
        variables,
        layer,
        xrho,
        rho_0,
        T0,
        cmf_mix=0.0,           # Fe mass fraction in the single-phase melt (w_Fe)
        apply_mix=False,
        phi_site=0.505,         # fraction of MgSiO3 molar volume for Mg/Fe site, trades off with fe_corr_rho
        fe_state='Fe2+_HS',    # 'Fe2+_HS', 'Fe3+_HS', 'Fe3+_LS'
        fe_corr_rho=1.0,     # empirical correction on Fe component density (from Fe0.66–MgO datum)
        wtfracH2=0.0           # weight fraction of H2 in the melt (passed to EOS_MgSiO3)
    ):
        """
        ODEs:
          dr/dm = 1/(4π r^2 ρ)
          dP/dm = -G m/(4π r^4)

        EOS:
          layer==0: metal -> EOS_Metal(P, xrho)
          layer==1: MgSiO3(+H2 scaffold) -> EOS_MgSiO3(P, rho_0, T0, wtfracH2) = ρ12

        Mixing (single-phase, apply_mix=True):
          Fe component uses Fe-occupied site volume ONLY:
            Vbar_site = φ * Vbar_MgSiO3
            Vbar_Fe_eff = α * Vbar_site
            rho_Fe_eff  = fe_corr_rho * (M_Fe / Vbar_Fe_eff)  # fe_corr_rho scales Fe density
          Then:
            1/ρ = (1-wFe)/ρ12 + wFe/ρ_Fe_eff
        """
        r, P = variables

        # --- constants (kg/mol) ---
        M_MgSiO3 = 100.39e-3
        M_Fe     = 55.85e-3

        # --- ionic radii (Å) — Shannon (VI coordination) ---
        r_Mg_VI = 0.72
        R_FE = {'Fe2+_HS': 0.78, 'Fe3+_HS': 0.645, 'Fe3+_LS': 0.55}
        r_Fe_VI = R_FE.get(fe_state, R_FE['Fe2+_HS'])

        # clamps
        wFe = min(max(cmf_mix, 0.0), 0.999999)
        phi = min(max(phi_site, 0.0), 1.0)

        if layer == 0:
            rho = EOS_Metal(P, xrho)

        elif layer == 1:
            rho_12 = EOS_MgSiO3(P, rho_0, T0, wtfracH2)  # kg/m^3

            if apply_mix and (wFe > 0.0):
                # reference MgSiO3-only density at this P,T to get Vbar_MgSi
                MW_sil, MW_H2 = 100.39, 2.01588   # g/mol (for rho_0_mix)
                rho0sil, rho0H2 = 2.5, 0.09       # g/cc
                rho0_pure_sil = rho_0_mix(0.0, MW_sil, MW_H2, rho0sil, rho0H2)  # g/cc
                rho_Si_pure   = EOS_MgSiO3(P, rho0_pure_sil, T0, wtfracH2)                # kg/m^3

                Vbar_MgSi = M_MgSiO3 / max(rho_Si_pure, 1e-30)  # m^3/mol
                alpha = (r_Fe_VI / r_Mg_VI) ** 3
                Vbar_site    = phi * Vbar_MgSi
                Vbar_Fe_eff  = alpha * Vbar_site
                rho_Fe_eff   = fe_corr_rho* M_Fe / max(Vbar_Fe_eff, 1e-30)

                inv_rho = (1.0 - wFe) / rho_12 + wFe / rho_Fe_eff
                rho = 1.0 / inv_rho
            else:
                rho = rho_12
        else:
            raise Exception(f"I don't have an EOS for layer {layer}")

        r2 = r * r
        drdm = 1.0 / (4.0 * pi * r2 * rho)
        dPdm = - G * m / (4.0 * pi * r2 * r2)
        return [drdm, dPdm]

    # ------------- SOLVE DIFFERENTIAL EQUATIONS ------------- #

    def integrate_ODEs(
        Pc, Mp, structure, cmf, cmf_mix, xrho, rho_0, T0,
        wtfracH2=0.0,
        phi_site=0.505,
        fe_state='Fe2+_HS',
        fe_corr_rho=1.0
    ):
        """
        - Single-phase mixing only if structure == [1]; cmf_mix is Fe mass fraction wFe.
        - Fe component uses Fe-occupied site volume ONLY; fe_corr_rho scales Fe density (0.771 from Fe0.66–MgO).
        """
        initial_values = [1e-9, Pc]
        apply_mix = (structure == [1])

        # constants (kg/mol)
        M_MgSiO3 = 100.39e-3
        M_Fe     = 55.85e-3

        # ionic radii (Å)
        r_Mg_VI = 0.72
        R_FE = {'Fe2+_HS': 0.78, 'Fe3+_HS': 0.645, 'Fe3+_LS': 0.55}
        r_Fe_VI = R_FE.get(fe_state, R_FE['Fe2+_HS'])
        phi = min(max(phi_site, 0.0), 1.0)
        wFe = min(max(cmf_mix, 0.0), 0.999999)

        # MgSiO3-only reference for Vbar_MgSi
        MW_sil, MW_H2 = 100.39, 2.01588  # g/mol
        rho0sil, rho0H2 = 2.5, 0.09      # g/cc
        rho0_pure_sil = rho_0_mix(0.0, MW_sil, MW_H2, rho0sil, rho0H2)  # g/cc

        for i in range(len(structure)):

            if structure[i] == 0:
                _m_min, _m_max = 0.0, cmf * Mp
            elif structure[i] == 1:
                _m_min, _m_max = cmf * Mp, Mp
            else:
                raise Exception(f"I don't have a structure model for {structure[i]}")

            sol = solve_ivp(
                structure_ODEs,
                [_m_min, _m_max],
                initial_values,
                args=(structure[i], xrho, rho_0, T0, cmf_mix, apply_mix, phi_site, fe_state, fe_corr_rho, wtfracH2),
                max_step=0.005 * Mp
            )
            initial_values = [sol.y[0, -1], sol.y[1, -1]]

            # Build rho profile consistent with structure_ODEs
            if structure[i] == 0:
                rho_val = np.array([EOS_Metal(Pj, xrho) for Pj in sol.y[1, :]])

            elif structure[i] == 1:
                rho_12_arr = np.array([EOS_MgSiO3(Pj, rho_0, T0, wtfracH2) for Pj in sol.y[1, :]])

                if apply_mix and (wFe > 0.0):
                    rho_Si_pure_arr = np.array([EOS_MgSiO3(Pj, rho0_pure_sil, T0, wtfracH2) for Pj in sol.y[1, :]])
                    Vbar_MgSi_arr = M_MgSiO3 / np.maximum(rho_Si_pure_arr, 1e-30)

                    alpha = (r_Fe_VI / r_Mg_VI) ** 3
                    Vbar_site_arr   = phi * Vbar_MgSi_arr
                    Vbar_Fe_eff_arr = alpha * Vbar_site_arr
                    rho_Fe_eff_arr  = fe_corr_rho * M_Fe / np.maximum(Vbar_Fe_eff_arr, 1e-30)

                    inv_rho_arr = (1.0 - wFe) / rho_12_arr + wFe / rho_Fe_eff_arr
                    rho_val = 1.0 / inv_rho_arr
                else:
                    rho_val = rho_12_arr

            if i == 0:
                m, r, P, rho = sol.t, sol.y[0, :], sol.y[1, :], rho_val
                Rcmb = r[-1] if structure == [0, 1] else 0.0
            else:
                m = np.append(m, sol.t)
                r = np.append(r, sol.y[0, :])
                P = np.append(P, sol.y[1, :])
                rho = np.append(rho, rho_val)

        return m, r, P, rho, Rcmb

    # ----- SOLVE SYSTEM OF EQUATIONS FOR Psurf = 0 ---- #

    def func_MR(Pc, Psurf, Mp, structure, cmf, cmf_mix, xrho, rho_0, T0, wtfracH2=0.0):
        """
        This function returns the surface pressure for a given central 
        pressure, Pc (Pa); planet mass, Mp (kg); core mass-fraction cmf [0,1];
        and metal density deficit, xrho [0,1].

        The layering model is set with the structure argument (see documentation
        of integrate_ODEs)
        """
        
        #Tref, output = optimize.newton(func_Tref, 100000, args=(Psurf, Mp, structure, cmf, cmf_mix, xrho,rho_0,Pc,T0),
                        #tol=50, maxiter=100, full_output=True, disp=False)
                                     
        m, r, P, rho, Rcmb = integrate_ODEs(Pc, Mp, structure, cmf, cmf_mix, xrho, rho_0, T0, wtfracH2)
        
        index_cmb = np.argmax(r > Rcmb)
        index_surf=np.argmax(r)
        
        diff=P[-1] - Psurf

        return diff



    def solve_for_structure(Mp, cmf, cmf_mix, xrho, rho_0, Tsurf, Psurf, structure=[0,1], wtfracH2=0.0):
        """
        This function returns arrays for mass (kg); radius (m); pressure (Pa);
        density (kg / m^3) and T (K) for a planet of mass Mp (kg); core mass-fraction cmf [0,1];
        metal density deficit, xrho [0,1], and surface T and P, Tsurf and Psurf, respectively.

        The layering model is set with the structure argument (see documentation
        of integrate_ODEs)
        """

        if cmf == 0.0 and structure == [0,1]:
            raise Exception("You've specified a 2 layer model of metal and MgSiO3, but also set a cmf = 0. Either set structure = [1] for a pure silicate planet, or increase core mass-fraction")
        if cmf > 0.0 and structure == [1]:
            print("    You are running a one-phase MgSiO3-H2-Fe mixture.")
            cmf = 0.0
        if cmf != 1.0 and structure == [0]:
            print("    Warning: You've specified a 1 layer model of metal, but also set a cmf != 1.0. I'll set cmf = 1 for this model.")
            cmf = 1.0

        # --- NEW: convert boundary T at finite P to potential Tpot ---
        Tpot = compute_T0_potential_from_surface(Psurf, rho_0, Tsurf, wtfracH2)

        # calculate central pressure, Pc, such that Psurf = desired value
        Pc, output = optimize.newton(func_MR, 200e9, args=(Psurf, Mp, structure, cmf, cmf_mix, xrho, rho_0, Tpot, wtfracH2),
                                     tol=1.0, maxiter=100, full_output=True, disp=False)

        # Structure
        m, r, P, rho, Rcmb = integrate_ODEs(Pc, Mp, structure, cmf, cmf_mix, xrho, rho_0, Tpot, wtfracH2)
        
        # Temperatures
        
        eta=rho/(rho_0*1000)
        
        # Treat whole body as silicate for initial T estimates
        vector_T_S_MgSiO3=np.vectorize(T_S_MgSiO3)
        T=vector_T_S_MgSiO3(eta,Tpot)
        
        # Replace with iron temperatures in the core as defined by Rcmb
        index_cmb = np.argmax(r > Rcmb)
        T_cmb= T_S_MgSiO3(eta[index_cmb],Tpot)
        rho_metal_0=7.03*(1-xrho)*1000
        
        for i in range(index_cmb,-1,-1):
            x=rho[i+1]/rho[i]
            # Skip over density contrast between silicate and metal to avoid spurious compression factor
            if x < 0.98:
                x=0.999
            # Use the definition of the metal Gruneisen parameter, gamma(V), to compute dT
            gamma=gamma_Metal(x)
            dT=T[i+1]*(gamma*(1-x))
            T[i]=T[i+1]+dT
            
        return m, r, P, rho, Rcmb,T
        

    def normalized_moment_of_inertia(densities_orig, radii, index):
        """
        Calculate the normalized moment of inertia (I / MR^2).
        
        Parameters:
            densities (array-like): Array of density values (kg/m^3).
            radii (array-like): Array of radial positions from the center (m).
                               Must be in ascending order and represent shell boundaries.
            index is the index for the arrays indicating the full size of the body. 
        
        Returns:
            float: Normalized moment of inertia (I / MR^2).
        """
        # Length of densities must be one less than length of radii
        if len(densities_orig) != len(radii) - 1:
            densities=densities_orig[:-1]
            
        # Calculate the total mass and moment of inertia stepwise
        total_mass = 0
        moment_of_inertia = 0
        
        for i in range(index):
            r_outer = radii[i + 1]
            r_inner = radii[i]
            
            # Shell volume
            shell_volume = (4.0/3.0) * np.pi * (r_outer**3.0 - r_inner**3.0)
            # Trap for phase change with infinite d(rho)/dr (dr = 0)
            if shell_volume == 0.0:
                r_outer = radii[i] + (radii[i+2]-radii[i])/2.0
            
            # Shell mass
            shell_mass = densities[i] * shell_volume
            
            # Contribution to moment of inertia (I = (2/5) m r^2 for spherical shells)
            # radius term is a weighted average of inner and outer shells
            
            shell_moi = (2.0/5.0) * shell_mass * ((r_outer**5.0 - r_inner**5.0) / (r_outer**3.0 - r_inner**3.0))

            
            # Accumulate mass and moment of inertia
            #total_mass += shell_mass
            #moment_of_inertia += shell_moi
            total_mass=total_mass + shell_mass
            moment_of_inertia = moment_of_inertia + shell_moi
        
        # Normalize by MR^2
        R = radii[index]  # Total radius defining the body size
        normalized_I = moment_of_inertia / (total_mass * R**2)
        
        return normalized_I


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #                                      ATMOSPHERE FUNCTIONS

    #---------------------------------------------------------------------------------------------
    def gridding(rmin, rmax, n, sharp, do_plot=False):
        """
        Monotone radius grid from rmin to rmax with smaller spacing near rmin
        and larger spacing near rmax, using logistic weights.

        Parameters
        ----------
        rmin, rmax : float
            Endpoints [m]. Output satisfies r[0]=rmin and r[-1]=rmax exactly.
        n : int
            Number of grid points (>= 2).
        sharp : float
            Controls how rapidly spacing transitions from small (near base)
            to large (aloft). Typical: 6–20. Larger => more concentrated near base.
        do_plot : bool
            If True, plots dr vs index.

        Returns
        -------
        r : ndarray shape (n,)
            Monotone increasing radii.
        """
        import numpy as np

        rmin = float(rmin)
        rmax = float(rmax)
        n = int(n)
        if n < 2:
            raise ValueError("n must be >= 2")
        if not (rmax > rmin):
            raise ValueError("Require rmax > rmin")

        # normalized index in [0,1]
        x = np.linspace(0.0, 1.0, n)

        # logistic weights: small near 0, large near 1
        # center at 0.5 so transition happens mid-column by default
        w = 1.0 / (1.0 + np.exp(-sharp * (x - 0.5)))

        # cumulative coordinate s in [0,1] (strictly increasing)
        s = np.cumsum(w)
        s = (s - s[0]) / (s[-1] - s[0])

        # map to radii, enforce exact endpoints
        r = rmin + (rmax - rmin) * s
        r[0] = rmin
        r[-1] = rmax

        if do_plot:
            import matplotlib.pyplot as plt
            dr = np.diff(r)
            plt.figure()
            plt.plot(x[:-1], dr)
            plt.xlabel("normalized index x")
            plt.ylabel("dr (m)")
            plt.grid(True)
            plt.show()

        return r
         


    #---- Convert mole fraction xH2 in silicate to weight percent ------
    def mole_fraction_to_weight_percent(x_H2, MWH2=2.016, MWsil=100.39):
        """
        Convert mole fraction of H2 in a binary mixture with MgSiO3 to weight percent of H2.
        
        Parameters:
        x_H2 : float or array-like
            Mole fraction of H2
        MWH2 : float
            Molecular weight of H2 (default = 2.016 g/mol)
        MWsil : float
            Molecular weight of MgSiO3 (default = 100.39 g/mol)
        
        Returns:
        wt_percent_H2 : float or array-like
            Weight percent of H2
        """
        numerator = x_H2 * MWH2
        denominator = numerator + (1 - x_H2) * MWsil
        wt_percent_H2 = (numerator / denominator) * 100
        return wt_percent_H2

    #----- Convert weight percent H2 in silicate to mole fraction ------
    def weight_percent_to_mole_fraction(wt_H2, MWH2=2.016, MWsil=100.39):
        """
        Convert weight percent of H2 in a binary mixture with MgSiO3 to mole fraction of H2.
        
        Parameters:
        wt_H2 : float or array-like
            Weight percent of H2 (0–100)
        MWH2 : float
            Molecular weight of H2 (default = 2.016 g/mol)
        MWsil : float
            Molecular weight of MgSiO3 (default = 100.39 g/mol)
        
        Returns:
        x_H2 : float or array-like
            Mole fraction of H2
        """
        w = wt_H2 / 100
        numerator = w / MWH2
        denominator = numerator + (1 - w) / MWsil
        x_H2 = numerator / denominator
        return x_H2
        

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #                            MgSiO3 - H2 SUBREGULAR SOLN MODEL functions
    A = 622000
    B = -4950.0
    #---------------------------------------------------------------------------------------------
    # Subregular G mixing functions based on analytical Gilmore and Stixrude G
    #
    # --- Glmore and Stixrude Gmix function ---
    def Gmix_H(x, T_G, P_G):
        """
        Equivalent to a subregular soln model where:
            x(1-x)(Co + C1x) whee
            Co = B
            C1 = A - B
        """
        R = 8.314
        tau = 4350
        ppi = -35
        x2 = 1 - x
        factor = (1 - T_G / tau + P_G / ppi)
        if x <= 0 or x >= 1:
            return np.inf
        return (A * x + B * x2) * x * x2 * factor + R * T_G * (x * np.log(x) + x2 * np.log(x2))

    #--- Critical temperature ---
    def critical_temperature(P):
        return 4223*(1+P/-35.0) # 4200*(1+P/-35.0)

    # --- Chemical potential using pen and paper ---
    def mu_H(x, T, P):
        R = 8.314
        tau = 4350
        ppi = -35.0
        factor = (1.0 - T / tau + P / ppi)
        term1 = factor * ( (2.0 * x * (A - 2.0*B))+ (3.0 * x * x * (B - A)) + B)
        term2 = R * T * (np.log(x)- np.log(1.0-x))
        return term1+ term2
        
    # --- Second derivative of G ---
    def d2Gmix_H(x,T, P):
        R = 8.314
        tau = 4350
        ppi = -35.0
        factor = (1.0 - T / tau + P / ppi)
        term1 = factor * ( 6 * x * (B - A) + 2*A- 4*B )
        term2 = R * T * (1.0/x+ 1.0/(1.0-x))
        return term1+ term2
        
    # --- Third derivative of Gmixing ---
    def d3Gmix_H(x, T, P):
        """
        Third derivative of Gmix with respect to x, needed for critical point.
        """
        R = 8.314
        tau = 4350
        ppi = -35.0
        factor = (1.0 - T / tau + P / ppi)
        # derivative of term1 in d2Gmix_H:
        dterm1 = 6 * (B - A) * factor
        # derivative of term2 in d2Gmix_H:
        dterm2 = R * T * (-1.0 / x**2 + 1.0 / (1.0 - x)**2)
        return dterm1 + dterm2

    # --- Critical point, x and T together  ---
    def critical_temperature(P, A=622000, B=-4950, T_guess=3000.0, x_guess=0.5):
        """
        Solve for the critical point (x, T) where d2G/dx2 = 0 and d3G/dx3 = 0.
        """
        def equations(vars):
            x, T = vars
            if x <= 0 or x >= 1:
                return [1e6, 1e6]  # avoid log(0) issues
            d2 = d2Gmix_H(x, T, P)
            d3 = d3Gmix_H(x, T, P)
            return [d2, d3]

        sol = root(equations, x0=[x_guess, T_guess], method='hybr')

        if sol.success:
            x_c, T_c = sol.x
            return T_c
        else:
            return {"x": np.nan, "T": np.nan, "success": False, "message": sol.message}

    # ---------- Binodal search at a single T and P --------------
    def coexistence_conditions(vars, T_bin, PGPa_bin):
        """
        Function derives conditions for coexistence, deviations
        from chemical potentials being equal, and free energy of whole
        system being minimized (common tanget criterion). 
        
        vars = x1 and x2 mole fractions of component in phase 1 and phase2, respectively.  
        
        f1 = difference between chemical potentials in phases 1 and 2 at x1 and x2. 
        f2 = difference between dG/dx at x2 and chemical potential at x1. 
        
        A third criterion employed is that the 2nd derivatives about x1 and x2 are positive,
        ensuring compositions represent stable, rather than metastable, solutions. 
        """
        Tcrit = critical_temperature(PGPa_bin)

        x1, x2 = vars

        # Basic domain checks
        if not (0 < x1 < 1) or not (0 < x2 < 1) or x1 >= x2:
            return [1e6, 1e6]  # Huge penalty if invalid

        if abs(x2 - x1) < 0.001:  # Enforce minimum separation 0.05
            return [1e5, 1e5]

        # Calculate
        mu1 = mu_H(x1, T_bin, PGPa_bin)
        mu2 = mu_H(x2, T_bin, PGPa_bin)
        G1 = Gmix_H(x1, T_bin, PGPa_bin)
        G2 = Gmix_H(x2, T_bin, PGPa_bin)

        # Check stability with (d²G/dx² > 0)
        d2G1 = d2Gmix_H(x1, T_bin, PGPa_bin)
        d2G2 = d2Gmix_H(x2, T_bin, PGPa_bin)

        if d2G1 <= -10 or d2G2 <= -10:
            return [1e5, 1e5]

        # Coexistence conditions
        f1 = (mu1 - mu2) / 1000.0  # Rescaled for better numerical behavior
        f2 = ((G2 - G1) / (x2 - x1) - mu1) / 1000.0  # Common tangent condition, so system G is minimum

        return [f1, f2]
        

    # ----------- Binodal search at low T ----------
    def find_coexisting_composition_soft_lowT(T_bin, PGPa_bin, bulk_wtpercentH2):
        """
        Approximate low-T coexistence: assume x2 ≈ 1,
        and find x1 such that dG/dx = 0 (i.e., Gmix minimum).
        """
        Tcrit = critical_temperature(PGPa_bin)

        # Fix x2 to near 1 (gas phase)
        x2_fixed = 1.0 - 1e-7  # effectively pure H2

        # Objective: minimize Gmix(x) to find stable melt composition
    #    def objective(x1):
    #        if x1 <= 0 or x1 >= 1:
    #            return 1e6
    #        d2G1 = d2Gmix_H(x1, T_bin, PGPa_bin)
    #        if d2G1 <= -10:
    #            return 1e5  # melt phase must be stable
    #        return Gmix_H(x1, T_bin, PGPa_bin)  # we want minimum Gmix

        def objective(x1):
            if x1 <= 0 or x1 >= 1:
                return 1e6

            d2G1 = d2Gmix_H(x1, T_bin, PGPa_bin)
            if d2G1 <= -10:
                return 1e5  # melt must be stable

            mu1 = mu_H(x1, T_bin, PGPa_bin)
            if np.isnan(mu1):
                return 1e6

            # --- Inner function to find x2 such that mu2 ≈ mu1 ---
            def mu_match(x2):
                if x2 <= 0 or x2 >= 1:
                    return 1e6
                mu2 = mu_H(x2, T_bin, PGPa_bin)
                if np.isnan(mu2):
                    return 1e6
                return (mu2 - mu1)**2

            # --- Use bounded minimization to find x2 ---
            Tfrac = T_bin/Tcrit
            x2_lower = 0.999 - 0.01 * (0.7 - Tfrac) / 0.7  # e.g., drops to ~0.989 near T=0
            x2_lower = max(0.95, x2_lower)
            res = minimize_scalar(mu_match, bounds=(x2_lower, 0.999999), method='bounded')
            #res = minimize_scalar(mu_match, bounds=(0.98, 0.999999), method='bounded')
            if not res.success:
                return 1e6

            x2 = res.x
            d2G2 = d2Gmix_H(x2, T_bin, PGPa_bin)
            if d2G2 <= -10:
                return 1e5  # gas phase must also be stable

            # --- Evaluate common tangent mismatch (optional) ---
            G1 = Gmix_H(x1, T_bin, PGPa_bin)
            G2 = Gmix_H(x2, T_bin, PGPa_bin)
            slope = (G2 - G1) / (x2 - x1)
            tangent_error = (slope - mu1)**2

            # --- Final objective: prioritize matching μ + smooth tangent ---
            return tangent_error  # or use: return tangent_error + 0.01 * G1 if you want Gmin bias
        

        # Minimize Gmix over melt phase range
        res = minimize_scalar(objective, bounds=(1e-6, 0.3), method='bounded')
        
        if res.success:
            x1 = res.x
            x2 = x2_fixed

            # Convert mole fractions x1 and x2 to weight fractions
            MW1 = 2.02
            MW2 = 100.3
            Rmass1 = (MW1 / MW2) * (x1 / (1.0 - x1))
            wtpercent1 = 100.0 * (Rmass1 / (Rmass1 + 1.0))

            Rmass2 = (MW1 / MW2) * (x2 / (1.0 - x2))
            wtpercent2 = 100.0 * (Rmass2 / (Rmass2 + 1.0))

            delta = wtpercent2 - wtpercent1
            wt_frac_sil = (wtpercent2 - bulk_wtpercentH2) / delta
            wt_frac_atm = (bulk_wtpercentH2 - wtpercent1) / delta

            Tcrit = critical_temperature(PGPa_bin)
            return x1, x2, Tcrit, wt_frac_sil, wt_frac_atm
            print(f"Found low-T coexistence (dG/dx = 0): x1 = {x1:.5f}, x2 = {x2:.5f} at T = {T:.1f} K")

        else:
            raise RuntimeError(f"Soft low-T solver (Gmin) failed at T = {T:.1f} K.")



    #----- Fnc to find coexisting compositions x1 and x2 -----
    def subregular(Tsubreg, bulk_wtpercentH2, PGPa_sub, last_guess=None, lowT_cutoff=0.7):
        """
        Find coexisting compositions at given Tsubreg(K), P(GPa).
        Automatically handles low-T regime.
        Tries soft-lowT solver first, falls back to hard-lowT solver if needed.
        
        Parameters:
            Tsubreg = the value to be evaluated for obtaining compositions;
            bulk_wtpercentH2 = the bulk composition of interest in wt% H2;
            PGPa_sub (GPa) = the pressure. 
        
        Returns:
            x1 (xH2 for silicate)
            x2 (xH2 for envelope)
            Tcrit = top of solvus
            wt_frac_sil = weight fraction silicate using lever rule and input bulk
            wt_frac_atm = weight fraction atmosphere using lever rule
        """

        Tcrit = critical_temperature(PGPa_sub)
        
        # --- hard guard: never allow None/invalid T into subregular ---
        if (Tsubreg is None) or (not np.isfinite(Tsubreg)):
            print(f"[subregular warn] bad Tsubreg={Tsubreg}; forcing Tsubreg=2500 K "
                  f"(bulk={bulk_wtpercentH2}, Ps={PGPa_sub})")
            Tsubreg = 1500.0

        # optional: keep T above some floor to avoid low-T branch weirdness
        Tsubreg = max(float(Tsubreg), 1500.0)

        bulk_wtpercentH2 = max(bulk_wtpercentH2, 1.0e-3)

        # --- Low T case ---
        if Tsubreg/Tcrit < lowT_cutoff:
            try:
                return find_coexisting_composition_soft_lowT(Tsubreg, PGPa_sub, bulk_wtpercentH2)
            except RuntimeError as e:
                print(f"  [Warning] Soft low-T solver failed at Tsubreg = {Tsubreg:.1f} K. Falling back to fixed-x2 low-T solver...")
                return find_coexisting_composition_soft_lowT(Tsubreg, PGPa_sub, bulk_wtpercentH2)
        else:
            # --- Normal T case ---
            Tfrac = Tsubreg / Tcrit

            # --- Define bounds depending on temperature regime ---
            if Tfrac < 0.73:  # 0.73
                # Tightly narrow search for x2 near pure H₂ gas
                x1_bounds = (1e-8, 0.25)
                x2_bounds = (0.999, 0.999995)
            else:
                # Normal range
                x1_bounds = (1e-6, 0.999)
                x2_bounds = (1e-6, 0.99999)

            # --- Choose initial guess ---
            if last_guess is not None:
                x1_init, x2_init = last_guess
            else:
                if Tfrac <= 0.96:
                    x1_init = 0.001
                    x2_init = 0.999
                elif Tfrac > 0.9975:
                    x1_init = 0.6
                    x2_init = 0.85
                else:
                    x1_init = 0.1
                    x2_init = 0.9

            # --- Clamp guess inside bounds ---
            x1_init = np.clip(x1_init, x1_bounds[0] + 1e-10, x1_bounds[1] - 1e-10)
            x2_init = np.clip(x2_init, x2_bounds[0] + 1e-10, x2_bounds[1] - 1e-10)

            # --- Call solver ---
            sol = least_squares(
                lambda vars: coexistence_conditions(vars, Tsubreg, PGPa_sub),
                x0=[x1_init, x2_init],
                bounds=([x1_bounds[0], x2_bounds[0]], [x1_bounds[1], x2_bounds[1]]),
                xtol=1e-10,
                ftol=1e-10
            )

        # --- Extract solution if successful ---
        if sol.success:
            x1, x2 = sol.x

            # --- Evaluate errors ---
            errs = coexistence_conditions([x1, x2], Tsubreg, PGPa_sub)
            total_error = np.sqrt(errs[0]**2 + errs[1]**2)

            # --- Adaptive acceptance ---
            min_separation = 0.01 * (1.0 - Tsubreg/Tcrit + 0.02)
            error_tolerance = 1.0e-3 * (1.0 - Tsubreg/Tcrit + 0.02) #1.0e-3

            if (0 < x1 < 1 and 0 < x2 < 1 and x1 < x2 and
                abs(x2 - x1) > min_separation and total_error < error_tolerance):
                #print(f"Found coexistence: x1 = {x1:.5f}, x2 = {x2:.5f} at Tsubreg = {Tsubreg:.1f} K")
                
                # Convert mole fractions x1 and x2 to weight fractions
                MW1=2.02 #H2
                MW2=100.3 #MgSiO3
                Rmass1=(MW1/MW2)*(x1/(1.0-x1)) # mass ratio species 1/2, 1 refers to first mole fraction of solvus, xH2 silicate
                wtpercent1=100.0*(Rmass1/(Rmass1+1.0)) # Weight per cent H2 in silicate

                Rmass2=(MW1/MW2)*(x2/(1.0-x2)) # mass ratio 1/2, 2 refers here to 2nd mole fraction of solvus, xH2 atmosphere
                wtpercent2=100.0*(Rmass2/(Rmass2+1.0)) # Weight per cent H2 in atmosphere

                # LEVER RULE:  A negative weight fraction indicates being above the solvus at the bulk specified.
                delta=wtpercent2-wtpercent1
                wt_frac_sil=(wtpercent2-bulk_wtpercentH2)/delta # Weight fraction of silicate condensed phase
                wt_frac_atm=(bulk_wtpercentH2-wtpercent1)/delta # Weight fraction of atmosphere phase
                        
                return x1, x2, Tcrit, wt_frac_sil, wt_frac_atm
                
            else:
                raise RuntimeError(f"Solver succeeded but solution bad: Δx = {x2-x1:.4e}, error = {total_error:.2e}")
        else:
            raise RuntimeError("Root-finding failed.")
        
    #------------------------------- Find contact with solvus ------------------------
    def find_contact_T(Tcrit,H2melt,PGPa_cont):
        """
        Find first contact with solvus for input H2 of melt
        
        Parameters:
            Tcrit is the crest of the solvus;
            H2melt is the weight percent H2 of the melt being tested.
        Returns:
            T of contact with solvus for melt wt percent H2 composition (H2melt)
        """
        Tcrit=critical_temperature(PGPa_cont)
        T_test = Tcrit-1
        
        bulk_xH2 = weight_percent_to_mole_fraction(H2melt, MWH2=2.016, MWsil=100.39)
        
        j=0
        while j < 5000:
            if H2melt > 2.0:
                T_test=T_test - 0.07  # This value is important near the top of the solvus
                err_tol  = 1.0e-1 * (1.0 - T_test/Tcrit + 0.02)
            else:
                T_test=T_test - 0.5
                err_tol  = 1.0e-2 * (1.0 - T_test/Tcrit + 0.02)
            if H2melt < 0.3:
                T_test = T_test - 0.1
                err_tol  = 0.07
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x1, x2, Tcrest, wt_silicate,wt_atmosphere = subregular(T_test,H2melt,PGPa_cont)

            if x1 <= bulk_xH2 <= x2:
                Tc=T_test
                print('    found T contact at step ', j)
                return Tc
                break
            
            if j == 4999:
                print('WARNING: search for surface T did not converge in alotted steps')
            
            j=j+1
    #-----------------------------------------------------------------------------------
        


    #---------------------------------------------------------------------------------------------
    def L_rad_thick(dTdr, T_thick, kappa, rho, r):
        """
        Function to calculate luminosity from Eddington's equation for radiative diffusion.
        T_thick = temperature (K)
        dTdr = T gradient (K/meter)
        kappa = opacity, m^2/kg
        rho = kg/m^3
        r = radial position, m
        """
        sigma = 5.6703744e-8
        pi = 3.1415927
        
        L = - dTdr*(64.0*pi*sigma*(T_thick**3.0)*r**2.0)/(3.0*kappa*rho)
        
        return L


    def clamp_T_contact(T, fallback=2350.0, Tmin=1000.0, Tmax=20000.0):
        """
        Return a safe float temperature for interface/solvus calls.
        - If T is None / non-finite: use fallback
        - Else clip into [Tmin, Tmax]
        """
        if (T is None) or (not np.isfinite(T)):
            return float(fallback)
        T = float(T)
        if T < Tmin: return float(Tmin)
        if T > Tmax: return float(Tmax)
        return T

    #-----------------------------------------------------------------------------------------
    # FUNCTIONS FOR MASS, PRESSURE, and DENSITY
    def dmdr(r,rho):
        """
        dm/dr for gas
        """
        pi=3.1415926
        G=6.67408e-11 #m^3/(kg s^2), SI units
        dmdr=(4.0*pi*r**2.0)*rho
        return dmdr
        
    def dPdr(r,rho,m):
        """
        dP/dr for gas from g and rho
        """
        pi=3.1415926
        G=6.67408e-11 #m^3/(kg s^2), SI units
        g=(-G*m/r**2)
        dPdr=g*rho
        return dPdr
        
    def drhodmw(T,P):
        """
        drho/dMW for gas
        """
        drhodmw=(P/(Rgas*T))
        return drhodmw

    #-----------------------------------------------------------------------------------------
    # FUNCTIONS FOR ENERGY DERIVATIVES
    def dEdr_core(r,c,Mr,T,rho,option):
        """
        Returns dE/dr for the core thermal energy (option = 1),
        for gravitational potential energy (option = 2), or total
        energy (option = 3).
        T is temperature K, c is specific heat in J/(K kg), r is radius in meters,
        rho is melt density in kg/m^3, and Mr is mass contained within radisu r. 
        Makes use of np.pi for pi.
        Supply an array of radii, dr, to scipy.integrate.quad to integrate dEdr to
        obtain total energy E, thermal + gravitational potential. 
        """
        if option == 1:
            dEdr=(c*T)*rho*4*np.pi*r**2
        
        if option == 2:
            dEdr=(-G*Mr/r)*rho*4*np.pi*r**2
        
        if option == 3:
            dEdr=(-G*Mr/r + c*T)*rho*4*np.pi*r**2
        
        return dEdr
        
    def dEdr_atm(r,Mr,m,T,gamma,rho,option):
        """
        Returns dE/dr for atmosphere thermal energy (option = 1),
        for gravitational potential energy (option = 2), or total
        energy (option = 3).
        Gravitational potential energy = -GM/r; thermal energy = TxCv/m,
        where Cv = kb/(gamma-1).  Mr is the mass inside of radius r, 
        gamma is the heat capacity ratio Cp/Cv for the gas, m is the 
        mean molecular mass for the gas (kg/molecule), and rho is the gas
        density (kg/m^3). Supply an array of radii, dr, to scipy.integrate.quad 
        to integrate dEdr to obtain total energy E.
        """
        if option == 1:
            dEdr=((1/(gamma-1))*kb*T/m)*4*np.pi*rho*r**2
            
        if option == 2:
            dEdr=(-G*Mr/r )*4*np.pi*rho*r**2
        
        if option == 3:
            dEdr=(-G*Mr/r + (1/(gamma-1))*kb*T/m)*4*np.pi*rho*r**2
        
        return dEdr
        

    #------------------------------------------------------------------------------------------
    def mole_fractions(x_cond,xH2sil,xH2atm,MWgas,MWmelt):
        """
         Function to pass mole fractions for system required by pseudoadiabat from Graham et al.
         (2021, PSJ).
         xd = mole fraction of non-volatile component (H2)
         xv = mole fraction of condensible still in vapor phase (MgSiO3)
         xc = mole fraction of condensible that now exists as condensed liquid (MgSiO3)
         These three mole fractions sum to unity and refer to total moles of atmosphere,
         including condensed phase.
        
         x_cond = mole fraction of atmosphere that is condensed
         xH2sil = xH2 for the silicate melt phase
         xH2atm = xH2 for the vapor phase
         MWi = kg/mole for specified phase i
        """


        xsil_as_melt=(x_cond)*(1-xH2sil) # mole fraction of MgSiO3 as melt rel to whole atmosphere
        xc=xsil_as_melt

        xsil_as_gas=(1-x_cond)*(1-xH2atm) # mole fraction of MgSiO3 as vapor rel to whole atmosphere
        xv=xsil_as_gas
        
        sum_silicate=xsil_as_gas+xsil_as_melt
        
        # Mole fraction of H2 as gas relative to the entire atmosphere system, melt + vapor
        xd=1.0-xv-xc

        return xd,xv,xc

    #---------------------------------------------------------------------------------------------
    def nabla_conv(T, L, molecmass, g, xd, xv, xc, alpha_con, P):
        """
        Pseudoadiabatic convective lapse (-dT/dr) with EOS scaling via Chabrier grad_ad.

        Steps:
          1) Compute moist-adiabatic dlnT/dlnP (ideal core) per Graham et al. (2021).
          2) Convert to k_ideal = (m * g / kB) * dlnT/dlnP.
          3) Scale k_ideal by  φ = grad_ad^{Chabrier}(T,P) / (Rgas / Cp_H2),
             so in the dry-H2 limit, match the EOS adiabatic gradient.

        Parameters
        ----------
        T : float          Temperature [K]
        L : float          Latent heat of the condensation reaction [J/mol]
        molecmass : float  Mean molecular mass [kg/molecule]  (NB: per molecule, not per mole)
        g : float          Local gravity [m/s^2]
        xd, xv, xc : float Mole fractions of dry gas, volatile gas, condensed volatile (≈ sum to 1)
        alpha_con : float  Retained fraction of condensate (mole basis)
        P : float          Pressure [Pa] at which to evaluate EOS correction

        Returns
        -------
        k_nonideal : float  EOS-scaled pseudoadiabatic lapse, k = -dT/dr [K/m]
        """
        Rgas = 8.3145
        kB   = 1.38065e-23

        # Soft check on composition closure
        s = xd + xv + xc
        if s < 0.995 or s > 1.05:
            print(f'...[nabla_conv] warning: xd+xv+xc ≈ {s:.3f} (expected ~1)')

        # --- 1) Moist-adiabatic (ideal) per Graham et al. (2021) ---
        beta  = L / (Rgas * T)

        # Heat capacities
        Cpd    = Cp_H2()
        CpSiO  = Cp_SiO()
        CpO2   = Cp_O2()
        CpMg   = Cp_Mg()
        Cpmelt = Cp_melt(T)

        # Mixture Cp of the volatile gas pseudo-binary (renormalize xd,xv only)
        denom_x = max(xd + xv, 1e-12)
        Cpv     = (xd/denom_x)*Cpd + (xv/denom_x)*(0.333*CpMg + 0.333*CpO2 + 0.333*CpSiO)

        # Graham numerator/denominator
        fnum = xd*Cpd + xv*(Cpv - Rgas*beta + Rgas*beta**2) + alpha_con*xc*Cpmelt
        fden = Rgas * (xd + xv*beta)
        if abs(fden) < 1e-18:
            fden = 1e-18 if fden >= 0.0 else -1e-18

        denom_dln = xd*(fnum/fden) + xv*beta
        dlnTdlnP  = (xd + xv) / max(denom_dln, 1e-30)

        # --- 2) Ideal lapse (per molecule mass!) ---
        k_ideal = (molecmass * g / kB) * dlnTdlnP

        # --- 3) EOS scaling using Chabrier table grad_ad ---
        # Dry ideal H2 adiabatic gradient: R/Cp_H2
        grad_ideal_dry = Rgas / max(Cpd, 1e-30)

        # Table adiabatic gradient at (T,P) for H2 (dimensionless dlnT/dlnP)
        grad_table = grad_H2_Chabrier(T, P)

        # Scaling factor φ; clamp mildly to avoid rare spikes from table edges
        phi = grad_table / max(grad_ideal_dry, 1e-30)
        phi = max(0.3, min(phi, 3.0))

        k_nonideal = k_ideal * phi
        return k_nonideal


    #---------------------------------------------------------------------------------------------
    def nabla_cond(T,m_molec,Cp_gas,xH2atm,Lrad,r_rad,r):
        """
        Function to calculate CONDUCTIVE dT/dr.
        Lrad is the luminosity to be transmitted
        xH2atm is the mole fraction of H2 (relative to MgSiO3)
        T is temperature, m_molec is molecular mass in kg, Cp_gas is molar heat capacity of gas
        """

        nucleon_mass = 1.6726e-27  # kg/nucleon
        kb = 1.38065e-23 # J/K
        Rgas=8.3144598
        pi = 3.14159265
        Av_number=6.02214e23 # particles per mole
        
        d=xH2atm*289.0e-12+(1-xH2atm)*(0.3333*346.0e-12+0.3333*200.0e-12+0.3333*300.0e-12)
        Cv=Cp_gas/Av_number-kb # heat capacity per atom at fixed volume
        v=np.sqrt(kb*T/m_molec) # average velocity of molecules
        k=2.0/(3.0*pi**(3/2))*(Cv/d**2)*v
        nabla=(1/k)*Lrad/(4.0*pi*r**2)
        return nabla
        

    #---------------------------------------------------------------------------------------------
    # Function to calculate isobaric heat capacity for H2.
    def Cp_H2():
        """
        H2 ideal gas isobaric heat capacity
        """
        Rgas=8.3145
        gamma=7.0/5.0
        Cp=(gamma/(gamma-1.0))*Rgas
        return Cp

    # Function to calculate isobaric heat capacity for SiO.
    def Cp_SiO():
        """
        SiO ideal gas isobaric heat capacity
        """
        Rgas=8.3145
        gamma=7.0/5.0
        Cp=(gamma/(gamma-1.0))*Rgas
        return Cp
        
    # Function to calculate isobaric heat capacity for O2.
    def Cp_O2():
        """
        O2 ideal gas heat capacity
        """
        Rgas=8.3145
        gamma=7.0/5.0
        Cp=(gamma/(gamma-1.0))*Rgas
        return Cp
        
    # Function to calculate isobaric heat capacity for Mg.
    def Cp_Mg():
        """
        Mg ideal gas heat capacity
        """
        Rgas=8.3145
        gamma=5.0/3.0
        Cp=(gamma/(gamma-1.0))*Rgas
        return Cp
        
    def Cp_melt(TK):
        """
        Function to calculate heat capacity for MgSiO3 melt. Compressibility is ignored, so no P input.
        Thermodynamic parameters are in SI units, per mole.  Source = NIST, Chase 1998, JANAF tables.
        """
        Rgas=8.3145
        #MgSiO3 melt
        # from NIST
        ti=TK/1000.0
        a=146.440
        b=-1.499926e-7
        c=6.220145e-8
        d=-8.733222e-09
        e=-3.144171e-8
        Cpmelt=a+b*ti+c*(ti**2.0)+d*(ti**3.0)+e/ti**2
        
        return Cpmelt

    #-------------------------------------------------------------------------------------------------------------
    def S_gas(T,P,xH2):
        """
        Function to calculate entropy of gas in J/(mol K).  Nist values for S(298,1bar) are used as the base values.
        Thermodynamic parameters are in SI units, per mole.
        Input P should be in Pascal, T in K.
        """
        Rgas=8.3145
        CpH2=Cp_H2()
        CpSiO=Cp_SiO()
        CpO2=Cp_O2()
        CpMg=Cp_Mg()
        
        S_H2=130.0+CpH2*ln(T/298.15)-Rgas*ln(P/1.0e5)
        S_Mg=148.65+CpMg*ln(T/298.15)-Rgas*ln(P/1.0e5)
        S_SiO=211.58+CpSiO*ln(T/298.15)-Rgas*ln(P/1.0e5)
        S_O2=205+CpO2*ln(T/298.15)-Rgas*ln(P/1.0e5)
        
        S=xH2*S_H2+(1.0-xH2)*(0.333*S_Mg+0.333*S_O2+0.333*S_SiO)
        return S

    #-------------------------------------------------------------------------------------------------------------
    def melt_thermo(TK):
        """
        Function to calculate thermodynamic parameters for MgSiO3 melt. Compressibility is ignored, so no P input.
        Thermodynamic parameters are in SI units, per mole
        """
        Rgas=8.3145
        #MgSiO3 melt
        # from NIST
        ti=TK/1000.0
        a=146.440
        b=-1.499926e-7
        c=6.220145e-8
        d=-8.733222e-09
        e=-3.144171e-8
        f=-1563.306
        g=220.6679
        h=-1494.864
        HmeltMgSiO3=-1494.86+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HmeltMgSiO3=HmeltMgSiO3*1000.0 #convert kJ to J
        SmeltMgSiO3=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g
        GmeltMgSiO3=HmeltMgSiO3-TK*SmeltMgSiO3  #apparent G of formation of MgO liquid at T and 1 bar
        
        return SmeltMgSiO3, HmeltMgSiO3, GmeltMgSiO3
        
        
    #-------------------------------------------------------------------------------------------------------------
    def H2_thermo(TK,P):
        """
        Function to calculate thermodynamic parameters for H2 gas phase. H2 is treated as ideal in that enthalpy
        has no pressure effect.  The effect is ascribed to entropy only.
        P is input pressure in Pa.
        Thermodynamic parameters are in SI units, per mole.
        """
        Rgas=8.3145
        #H2 std state data
        # from NIST
        if TK < 6000.0:
            ti=TK/1000.0
            a=43.41356
            b=-4.293079
            c=1.272428
            d=-0.096876
            e=-20.533862
            f=-38.515158
            g=162.081354
            h=0.000
            HgasH2=0.000+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
            HgasH2=HgasH2*1000.0 #convert kJ to J
            SgasH2=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g-Rgas*ln(P/1.0e5)
            GgasH2=HgasH2-TK*SgasH2 #apparent G of formation of H2 gas at T and 1 bar
        if TK < 2500.0:
            ti=TK/1000.0
            a=18.563083
            b=12.257357
            c=-2.859786
            d=0.268238
            e=1.977990
            f=-1.147438
            g=156.288133
            h=0.00
            HgasH2=0.000+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
            HgasH2=HgasH2*1000.0 #convert kJ to J
            SgasH2=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g-Rgas*ln(P/1.0e5)
            GgasH2=HgasH2-TK*SgasH2 #apparent G of formation of H2 gas at T and 1 bar
        if TK < 1000.0:
            ti=TK/1000.0
            a=33.066178
            b=-11.363417
            c=11.432816
            d=-2.772874
            e=-0.158558
            f=-9.980797
            g=172.707974
            h=0.00
            HgasH2=0.000+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
            HgasH2=HgasH2*1000.0 #convert kJ to J
            SgasH2=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g-Rgas*ln(P/1.0e5)
            GgasH2=HgasH2-TK*SgasH2 #apparent G of formation of H2 gas at T and 1 bar
            
        return SgasH2, HgasH2, GgasH2
        
    #-------------------------------------------------------------------------------------------------------------

    def O2_thermo(TK,P):
        """
        Function to calculate thermodynamic parameters for O2 gas phase. O2 is treated as ideal in that enthalpy
        has no pressure effect.  The effect is ascribed to entropy only.
        P is input pressure in Pa.
        Thermodynamic parameters are in SI units, per mole
        """
        Rgas=8.3145
        #O2 gas std state of 1 bar 2000 to 6000K
        ti=TK/1000.0
        a=20.9111
        b=10.72071
        c=-2.020498
        d=0.146449
        e=9.245722
        f=5.337651
        g=237.6185
        h=0.00
        HgasO2=0.000+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
        HgasO2=HgasO2*1000.0 #convert kJ to J
        SgasO2=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g-Rgas*ln(P/1.0e5)
        GgasO2=HgasO2-TK*SgasO2  #apparent G of formation of O2 gas at T and 1 bar
        
        return SgasO2, HgasO2, GgasO2
        
    #-------------------------------------------------------------------------------------------------------------
    def SiO_thermo(TK,P):
        """
        Function to calculate thermodynamic parameters for SiO gas phase. SiO is treated as ideal in that enthalpy
        has no pressure effect.  The effect is ascribed to entropy only.
        P is input pressure in Pa.
        """
        Rgas=8.3145
        #SiO gas std state of 1 bar
        if TK < 1100.0:
            ti=TK/1000.0
            a=19.52413
            b=37.46370
            c=-30.51805
            d=9.094050
            e=0.148934
            f=-107.1514
            g=226.1506
            h=-100.4160
            HgasSiO=-100.42+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
            HgasSiO=HgasSiO*1000.0 #convert kJ to J
            SgasSiO=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g-Rgas*ln(P/1.0e5)
            GgasSiO=HgasSiO-TK*SgasSiO #apparent G of formation of SiO gas at T and 1 bar
        else:
            ti=TK/1000.0
            a=35.69893
            b=1.731252
            c=-0.509348
            d=0.059404
            e=-1.248055
            f=-114.6019
            g=249.1911
            h=-100.416
            HgasSiO=-100.42+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
            HgasSiO=HgasSiO*1000.0 #convert kJ to J
            SgasSiO=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g-Rgas*ln(P/1.0e5)
            GgasSiO=HgasSiO-TK*SgasSiO #apparent G of formation of SiO gas at T and 1 bar
        
        return SgasSiO, HgasSiO, GgasSiO

    #-------------------------------------------------------------------------------------------------------------
    def Mg_thermo(TK,P):
        """
        Function to calculate thermodynamic parameters for Mg gas phase. Mg gas is treated as ideal in that enthalpy
        has no pressure effect.  The effect is ascribed to entropy only.
        P is input pressure in Pa.
        """
        Rgas=8.3145
        #Mg gas std state of 1 bar
        if TK < 2200.0:
            ti=TK/1000.0
            a=20.77306
            b=0.035592
            c=-0.031917
            d=0.009109
            e=0.000461
            f=140.9071
            g=173.7799
            h=147.1002
            HgasMg=147.1+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
            HgasMg=HgasMg*1000.0 #convert kJ to J
            SgasMg=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g-Rgas*ln(P/1.0e5)
            GgasMg=HgasMg-TK*SgasMg  #apparent G of formation of Mg gas at T and 1 bar
        else:
            ti=TK/1000.0
            a=47.60848
            b=-15.40875
            c=2.875965
            d=-0.120806
            e=-27.01764
            f=97.40017
            g=177.2305
            h=147.1002
            HgasMg=147.1+a*ti+(b*ti**2.0)/2.0+(c*ti**3.0)/3.0+(d*ti**4.0)/4.0-(e/ti)+f-h
            HgasMg=HgasMg*1000.0 #convert kJ to J
            SgasMg=a*ln(ti)+b*ti+(c*ti**2.0)/2.0+(d*ti**3.0)/3.0-(e/(2.0*ti**2.0))+g-Rgas*ln(P/1.0e5)
            GgasMg=HgasMg-TK*SgasMg  #apparent G of formation of Mg gas at T and 1 bar
        
        return SgasMg, HgasMg, GgasMg


    #-------------------------------------------------------------------------------------------------------------
    def my_kappa(T, P_Pa, zmetal):
        """
        Rosseland mean opacity (Freedman-style fit), returns kappa in cm^2/g.

        Inputs:
          T [K]
          P_Pa [Pa]
          zmetal = [M/H] dex relative to solar
        """
        # Pa -> dyn/cm^2
        Pcgs = float(P_Pa) * 10.0

        if T <= 0.0 or Pcgs <= 0.0:
            raise ValueError("Non-positive T or P in my_kappa")

        # ---- USE log10 consistently ----
        logT = np.log10(float(T))
        logP = np.log10(float(Pcgs))

        # Coefficients
        c1, c2, c3, c4, c5, c6, c7 = 10.602, 2.882, 6.09e-15, 2.954, -2.526, 0.843, -5.490
        c8, c9, c10, c11, c12, c13 = -14.051, 3.055, 0.024, 1.877, -0.445, 0.8321

        if T > 800.0:
            c8, c9, c10, c11, c12, c13 = 82.241, -55.456, 8.754, 0.7048, -0.0414, 0.8321

        # ---- low-P branch ----
        # Note: c4 wasn't used;  leaving  as-is.
        # Replace exp(logT - c5) with 10**(logT - c5) because logT is log10(T).
        try:
            term_exp = 10.0**(logT - c5)          # = 10^(log10T - c5)
            logkappa_lowP = (
                c1 * np.arctan(logT - c2)
                - (c3 / max(logP, -99.0)) * (term_exp**2)
                + c6 * zmetal
                + c7
            )
        except Exception:
            logkappa_lowP = -10.0

        # ---- high-P branch ----
        try:
            metallicity_term = c13 * zmetal * (0.5 + (1.0 / np.pi) * np.arctan((logT - 2.5) / 0.2))
            logkappa_highP = (
                c8 + c9 * logT + c10 * logT**2
                + logP * (c11 + c12 * logT)
                + metallicity_term
            )
            logkappa_highP = min(logkappa_highP, 50.0)
        except Exception:
            logkappa_highP = -10.0

        # ---- combine in log10 space  ----
        max_logkappa = max(logkappa_lowP, logkappa_highP)
        delta_low  = 10.0**(logkappa_lowP  - max_logkappa)
        delta_high = 10.0**(logkappa_highP - max_logkappa)
        logkappa_total = max_logkappa + np.log10(delta_low + delta_high)

        kappa = 10.0**logkappa_total  # cm^2/g
        return float(kappa)
        
    #-------------------------------------------------------------------------------------------------------------
    def compute_metallicity(metal_mole_fraction, solar_metallicity=1.34e-3):
        """
        Computes log(X/X_solar) given the mole fraction of metals in the gas.

        Parameters:
        metal_mole_fraction (float): Mole fraction of metals in the H₂ gas (X)
        solar_metallicity (float): Solar metallicity value to compare against (X_solar)

        Returns:
        float: Metallicity in logarithmic scale (log10(X/X_solar))
        """
        
        if metal_mole_fraction <= 0 or solar_metallicity <= 0:
            raise ValueError("Both metal mole fraction and solar metallicity must be positive numbers.")
        
        z = np.log10(metal_mole_fraction / solar_metallicity)
        if z < 0:
            z = 0
        return z




    #-------------------------------------------------------------------------------------------------------------
        
    def T_eddington(tau, T_int, T_eq, gamma=None):
        """
        Guillot (2010) Eq. 27: analytic grey atmosphere with isotropic irradiation.

        tau   : IR optical depth (downward from TOA)
        T_int : intrinsic temperature (internal heat flux)
        T_eq  : equilibrium temperature (stellar irradiation)
        gamma : kappa_vis / kappa_th.  If None (default), uses the value of
                gamma_vis_th passed to Planet_LAB.
        """

        tau = np.asarray(tau, dtype=float)
        if gamma is None:
            gamma = gamma_vis_th
        gamma = float(gamma)

        # --- Internal flux term ---
        T4_int = (3.0/4.0) * T_int**4 * (tau + 2.0/3.0)

        # --- Irradiation term: EXACT Guillot Eq. 27 bracket ---
        bracket = (
            (2.0/3.0)
            + (2.0/(3.0*gamma))
            + (gamma/3.0 - 2.0/(3.0*gamma)) * np.exp(-gamma * tau)
        )

        T4_irr = (3.0/4.0) * T_eq**4 * bracket

        # --- Total ---
        T4 = T4_int + T4_irr
        return T4**0.25

    #-------------------------------------------------------------------------------------------------------------
    
    def dbg_r_samples(r_atm, tag=""):
        r = np.asarray(r_atm)
        n = len(r)

        print(f"\n[DBG r_atm] {tag}")
        print("------------------------------------------------")

        idx = [0, 1, 2,
               n//10,
               n//2,
               n-3, n-2, n-1]

        for i in idx:
            if 0 <= i < n:
                print(f" i={i:6d}   r={r[i]:.6e} m")

        # Step sizes near base and top
        dr = np.diff(r)

        print("\n--- local dr ---")
        print(f" dr[0]     = {dr[0]:.3e} m")
        print(f" dr[mid]   = {dr[n//2]:.3e} m")
        print(f" dr[-1]    = {dr[-1]:.3e} m")
        
    

    
    # Array defintitions
    n=int(nr)
    T=np.zeros(n)
    P=np.zeros(n)
    Pthermal=np.zeros(n)
    N=np.zeros(n)
    mass=np.zeros(n)
    tau=np.zeros(n)
    density=np.zeros(n)
    H=np.zeros(n)
    k_ratio=np.zeros(n)
    gradT_saved=np.zeros(n)
    k_rad_saved=np.zeros(n)
    k_cond_saved=np.zeros(n)
    k_conv_saved=np.zeros(n)
    k_nc_saved=np.zeros(n)
    gradP=np.zeros(n)
    Matm=0.0
    mass_H2_atm=0.0
    m_molec=np.zeros(n)
    MW_gas=np.zeros(n)
    MW_melt=np.zeros(n)
    mean_mol_wt_gas=np.zeros(n)
    xH2atmosphere=np.zeros(n)
    mass_frac_H2atm=np.zeros(n)
    xH2melt=np.zeros(n)
    dEntropy=np.zeros(n)
    Sgas=np.zeros(n)
    Smelt=np.zeros(n)
    Gmelt=np.zeros(n)
    Hmelt=np.zeros(n)
    x_cond=np.zeros(n)
    wt_frac_condensed=np.zeros(n)
    S_sys_kg=np.zeros(n)
    Sgaskg=np.zeros(n)
    Smeltkg=np.zeros(n)
    H_cond=np.zeros(n)
    x_inhib=np.zeros(n)
    dPdr_save=np.zeros(n)

    # RADII and HEIGHTS
    # Bondi radius based on the radiation temperature and the mass of the planet core
    Cs=np.sqrt(gamma_gas*kb*Teq/m_main)
    RB=2.0*G*Mp_target/(Cs**2)
    print(' Bondi radius = %10.4e km' %(RB/1000.0))
    print(' Bondi radius/planet radius = %10.3e' %(RB/R_s))

    # SET MAXIMUM RADIUS to nominal value for searching for Teq, and minimum to surface, all in meters
    RB_over_rmax_search=RB_over_rmax
    rmax=RB/RB_over_rmax_search

    # SETUP RADIAL POSITIONS IN ATMOSPHERE
    R_earth_meters=6371.0*1000.0
    R_s=6371.0*1000.0*(Mp_target_earth-(bulk/100)*Mp_target_earth)**(0.25) # meters
    r_atm = gridding(R_s, rmax, int(nr), sharp)
    
    #dbg_r_samples(r_atm, tag="after gridding()")


    # Print results
    print('\n r = ', r_atm)
    print(' dr[1] =', (r_atm[2] - r_atm[1]))
    print(' dr[n] = ', (r_atm[-1] - r_atm[-2]),'\n ')


    # SET INITIAL GRAVITATIONAL ACCELERATION
    g_s=G*Mp_target/(R_s)**2.0 #SI units, subscript s means "surface"
    g=G*Mp_target/(r_atm)**2.0  # g array for all radii

    # SET INITIAL CORE MASS TO TARGET PLANET MASS
    Mc=Mp_target

    # Ps DETERMINES THE MASS OF ATMOSPHERE AND MAY BE ADJUSTED BELOW
    Ps=target

    #-------------------------------------------------------------------------------------------------------------
    def model(option, r, Ra =1.0e12, T_contact_in=None, Ps_in = None):
        """
        FUNCTION COMPRISING THE ATMOSPHERE MODEL 
        
        Ps is the surface pressure in GPa
        
        option = 0 means return a cost function, otherwise return all results.
        Use option = 0 when passing this model function for optimization basal luminosity
        
            T = temperatures (K) for each atmosphere shell
            P = pressure (Pa) for each atmosphere shell
            L_s = basal net luminosity
            density = density of each atmosphere shell
            mass = cumulative mass of gas phase below each shell
            mass_cond = cumulative mass of condensate below each shell
            dPdr_save = derivative for atmosphere
            MW_gas = mol wt gas in kg/mole, for atmosphere
            MW_melt = mol wt melt in kg/mole, for atmosphere
            m_molec = mean molecular weight of molecules in gas
            k_ratio = nabla_convection/nabla_non-convection, nabla = dT/dr
            mass_H2_atm = mass of H2 residing as gas in each shell
            mass_H2_cond = mass of H2 residing as condensate in each shell
            x_cond = mole fraction of condensate relative to gas + condensate
            xH2atmosphere = mole fraction of H2 in gas phase
            wt_frac_condensed = mass fraction of "atmosphere" composed of condensate
            k_rad_saved = -dT/dr for radiation
            k_conv_saved = -dT/dr for convection
            k_nc_saved = -dT/dr due to radiation and/or conduction
            gradT_saved = effective -dT/dr due to all processes, note sign!
            x_inhib = mole fraction of MgSiO3 leading to convective inhibition in gas
            Sgaskg = Entropy of gas J/(K kg) in atmosphere
            Smeltkg = Entropy of melt in atmosphere
            S_sys_kg = Entropy of system, atmosphere
        """
        r = np.asarray(r_atm, dtype=float)
        adiabat=False

        # CONSTANTS IN SI UNITS
        G=6.67e-11 # m^3 kg^-1 s^-2
        nucleon_mass = 1.6726e-27  # kg/nucleon
        kb = 1.38065e-23 # J/K
        Rgas=8.3144598 # J/(mol K)
        pi = 3.14159265
        Av_number=6.02214e23 # particles per mole
        sigma=5.670374419e-8 # W/(m^2 K^4)
        L_solar=3.846e26 # Watts
        
        T_contact_use = T_contact_in if (T_contact_in is not None) else T_contact
        Ps_use = Ps_in if (Ps_in is not None) else Ps

        # PREALLOCATE ARRAYS
        n=int(nr)
        T=np.zeros(n)
        P=np.zeros(n)
        Pthermal=np.zeros(n)
        N=np.zeros(n)
        mass=np.zeros(n)
        tau=np.zeros(n)
        density=np.zeros(n)
        H=np.zeros(n)
        k_ratio=np.zeros(n)
        gradT_saved=np.zeros(n)
        k_rad_saved=np.zeros(n)
        k_cond_saved=np.zeros(n)
        k_conv_saved=np.zeros(n)
        k_nc_saved=np.zeros(n)
        gradP=np.zeros(n)
        Matm=0.0
        mass_H2_atm=np.zeros(n)
        m_molec=np.zeros(n)
        MW_gas=np.zeros(n)
        MW_melt=np.zeros(n)
        mean_mol_wt_gas=np.zeros(n)
        xH2atmosphere=np.zeros(n)
        mass_frac_H2atm=np.zeros(n)
        xH2melt=np.zeros(n)
        dEntropy=np.zeros(n)
        Sgas=np.zeros(n)
        Smelt=np.zeros(n)
        Gmelt=np.zeros(n)
        Hmelt=np.zeros(n)
        x_cond=np.zeros(n)
        wt_frac_condensed=np.zeros(n)
        mass_cond=np.zeros(n)
        S_sys_kg=np.zeros(n)
        Sgaskg=np.zeros(n)
        Smeltkg=np.zeros(n)
        H_cond=np.zeros(n)
        x_inhib=np.zeros(n)
        dPdr_save=np.zeros(n)
        xMgSiO3_atm=np.zeros(n)
        mass_H2_cond=np.zeros(n)
        wt_frac_H2_melt=np.zeros(n)
        drhodmw_s=np.zeros(n)
        Lrad=np.zeros(n)
        Teddington=np.zeros(n)
        bulk_Rayleigh=np.zeros(n)
        tau_above      = np.zeros(n)
        kappa_si_save  = np.zeros(n)
        
        # --- radiative–convective attachment state (must be in model() scope) ---
        tau_attach = None
        T_attach   = None
        i_attach   = None
        
        ledoux_limited = False
        
        
        #dbg_r_samples(r, tag="top of model() call")
        
        P_i_set = None   # must exist first

        def chkP(i, val, tag=""):
            global P_i_set
            P_i_set = val
            print(f"[CHK P] i={i} P={val:.6e} ({tag})")
                        
        
        # INITIALIZE................................................................
        # Initial values
        for i in range(0,n):
            T[i]=Teq
            m_molec[i]=m_main
            Smelt[i],Hmelt[i],Gmelt[i]=melt_thermo(T[i])
        
        
        # Get solvus crest T
        Tcrest = critical_temperature(Ps)

        # Estimate of specific entropy for atm imparted at the base

        xH2sil, xH2atm, Tcrest, wt_silicate, wt_atmosphere = subregular(T_contact, bulk, Ps)
        
        xH2atmosphere[0]=xH2atm
        xH2melt[0]=xH2sil
        xMgSiO3_atm[0]=1-xH2atmosphere[0]
        mean_mol_wt_gas[0]=xH2atmosphere[0]*MWH2+(1.0-xH2atmosphere[0])*(0.333*32/1000+0.333*(28+16)/1000+0.333*24/1000) #mean of kg/mole for gas species combined
        m_molec[0]=mean_mol_wt_gas[0]/Av_number # mean kg/molecule
        MWgas= mean_mol_wt_gas[0] # kg/mole for gas molecules
        MWmelt=xH2sil*MWH2+(1.0-xH2sil)*MWMgSiO3 # kg/mole
        # Convert bulk mass fraction H2 (%) specified by user to to molar basis
        Rkg=(bulk/100)/(1.0-bulk/100)
        Rmol=Rkg*MWMgSiO3/MWH2
        xH2bulk=Rmol/(Rmol+1)
        # initial gas has the molar composition defined by the initial contact of the bulk with solvus
        Sg=S_gas(T_contact,(Ps*1.0e9),xH2atm)
        Sm,Hm,Gm=melt_thermo(T_contact)
        # Define system entropy from weight fractions of melt and silicate defined by the solvus at first contact
        S_system=wt_atmosphere*Sg/MWgas+wt_silicate*Sm/MWmelt
        for i in range(n):
            S_sys_kg[i]=S_system #in kg/mole

        # SET GRAVITATIONAL ACCELERATION
        g_s=G*Mc/(R_s)**2.0 #SI units, subscript s means "surface"
        g=G*Mc/(r)**2.0  # g array for all radii

        # INITIALIZE T AND P
        T[0] = float(T_contact_use)
        P[0] = Ps_use * 1.0e9
        
        print('    T surface =',T[0])
        

        # INITIALIZE GAS DENSITY at surface
        density[0]=rho_gas(P[0],T[0],mean_mol_wt_gas[0])
        
        # Initialize z = rho_ideal/rho
        zgas = 1
        
        # Initialize atmosphere mass, zero at the surface
        rho=density[0]
        rlower=r[0]
        rupper=r[1]
        msoln=quad(dmdr,rlower,rupper,args=(rho))
        dm=msoln[0]
        mass[0]=dm
        N[0]=mass[0]/(4.0*pi*r[0]**2.0)

        # INITIALIZE COMPOSITIONS, partly repeated from above
        xH2atmosphere[0]=xH2atm
        xH2melt[0]=xH2sil
        wt_frac_H2_melt[0] = mole_fraction_to_weight_percent(xH2melt[0])/100 # weight fraction H2 in melt
        mass_frac_H2atm[0]= mole_fraction_to_weight_percent(xH2atmosphere[0])/100 # weight fraction H2 in gas
            
        # Update the effective bulk for the next gas shell, affords perfect Rayleigh distillation
        bulk_Rayleigh[0] = mass_frac_H2atm[0] * 100
        
        MW_gas[0]=MWgas
        MW_melt[0]=MWmelt
        wt_frac_condensed[0] = 0.0
        # Account for rounding errors
        if wt_frac_condensed[0] >= 1.00000:
            wt_frac_condensed[0] = 1.0000 - 1.0e-6
        if wt_frac_condensed[0] <= 0.0:
            wt_frac_condensed[0] = 1.0e-8
        dm_cond = 0
        mass_cond[0]=dm_cond
        
        # Initialize entropy of gas
        Sgas[0]=S_gas(T_contact,P[0],xH2atmosphere[0]) #J/mole K H2 corrected from 298 to T_contact

        # Initialize latent heat of condensation in the atmosphere
        Sm,Hm,Gm=melt_thermo(T_contact)
        SO2,HO2,GO2=O2_thermo(T_contact,P[0])
        SSiO,HSiO,GSiO=SiO_thermo(T_contact,P[0])
        SMg,HMg,GMg=Mg_thermo(T_contact,P[0])
        H_cond[0]=abs((Hm-(HO2+HSiO+HMg)))*Hmultiplier
        
        #epsilon for threshold
        MW_vap = (0.333*32/1000 + 0.333*(28+16)/1000 + 0.333*24/1000)
        epsilon = MW_vap/MWH2
        L=H_cond[0]
        x_inhib[0]=1.0/((L/(Rgas*T[0])-1 )*(epsilon-1))


        # Calculate Tint from surface properties
        convect=True
        Trcb=1.0
        
        
        metal_mole_fraction = 1-xH2atmosphere[0]
        Zatm = compute_metallicity(metal_mole_fraction)
        kappa_s=my_kappa(T[0],P[0],Zatm) #
        tau_s=kappa_s*0.1*P[0]/g[0]
        tau[0]=tau_s
        tau_int=1
        
        # Magma interface luminosity cap:
        Hsurf = Rgas*T[0]/(mean_mol_wt_gas[0]*g[0])
        delta_bl = 5.0*Hsurf/Ra**(1/3)
        tau_bl = kappa_s*0.1*density[0] * delta_bl
        L_bb = 4.0*pi*(r[0]**2)*sigma*(float(T[0])**4)  # no attenuation
        Lint = L_bb * (4.0/(3.0*(1.0 + tau_bl)))  # attenuated by tau_for_Tint
        Tint = T[0]*(4.0/(3.0*(tau_bl + 1)))**(0.25)
        
        
        print(f"    [DBG] DELTA ={delta_bl:.6e}, H={Hsurf:.6e}, Ra/Rc={Ra:.6e} ")
        
        
        # Radial position for T = Teq+DT = Trad
        r_rad=RB
        
        # THE EDDINGTON TWO-STREAM refers to flux balance, so Tint should be corrected for radial positon
        # relative to the maximum value compatible with Lint at the base. Do this throughout to ensure
        # it is luminosity that is constant, not flux(r).
        r_call = max(r[i-1], 1e-30)
        Tint_r = Tint * (r[0] / r_call)**0.5
        Trad = T_eddington(0,Tint_r,Teq)
        
        # DRY ADIABAT FOR H2 from the surface for calculating the "cross-over" definition of the Rrcb
        T_adiabat=np.zeros(n)
        i_rcb_analytical=n-1
        for i in range(0,n):
            T_adiabat[i]=Trad
        T_adiabat[0]=T[0]
        for i in range(1,n):
            CpH2=Cp_H2()
            CpSiO=Cp_SiO()
            CpO2=Cp_O2()
            CpMg=Cp_Mg()
            Cp_gas=xH2atmosphere[i]*CpH2+(1.0-xH2atmosphere[i])*(0.333*CpO2+0.333*CpSiO+0.333*CpMg)
            k_conv=(Rgas/Cp_gas)*(m_molec[i]*g[i]/kb)
            T_adiabat[i]=(T_adiabat[i-1])-(k_conv)*(r[i]-r[i-1])
        for i in range(1,n):
            if T_adiabat[i] < Trad:
                i_rcb_analytical=i
                break
        Rrcb_analytical=r[i_rcb_analytical]
        
        # Initialize dP/dr
        rlower=r[0]
        rupper=r[1]
        dr=rupper-rlower
        m=Mc
        rho=density[0]
        argtuple=(rho,m)
        psoln=quad(dPdr,rlower,rupper,args=argtuple) #integral of dP/dr from  rlower to rupper
        dP=psoln[0]
        dPdr_save[0]=dP/dr # Derivative estimate used for density derivative
        
        # Initialize density derivatives
        drhodmw_s[0] = drhodmw(T[0],P[0])
        
        if option > 0:
            print('    Tint r[0] = %.2f' %Tint)
            print('    tau surface = %.3e' %tau[0])
        
        r_call = max(r[0], 1e-30)
        Tint_r = Tint * (r[0] / r_call)**0.5
        Teddington[0] = T_eddington(0,Tint_r,Teq)
        
        dtau_dr = kappa_s*0.1*density[0]
        dT_dtau = (1/(4*T[0]**3))*((3/4)*Tint**4 + (3/8)*Teq**4 * exp(-tau[0]))
        k_rad_rad = dT_dtau*dtau_dr
        
        
        # Initialize gradT, save kappa at surface for future use
        if Inhibit:
            dh=r[1]-r[0]
            L_r = Lint
            
            print('    Lint at base of atm = %.3e' %L_r)
            
            k_rad_deep=(3.0*kappa_s*0.1*density[0]/(64.0*pi*sigma*(T[0]**3.0)*r[0]**2.0))*L_r
            gradT_saved[0]=k_rad_deep
        else:
            xd,xv,xc=mole_fractions(wt_frac_condensed[0],xH2melt[0],xH2atmosphere[0],MW_gas[0],MW_melt[0])
            alpha_c=0.0
            k_conv=nabla_conv(T[0],H_cond[0],m_molec[0],g[0],xd,xv,xc,alpha_c,P[0])
            #k_conv=nabla_conv(Cp_gas,m_molec[i-1],g[i-1])
            gradT_saved[0]=k_conv
            
        # Initialize index for position of the rcb
        i_rcb = None
        
        # Initialize index for tracking the final Ledoux layer top
        ledoux_top_i_final = None
        
        
        print("    len(r_atm) =", len(r), "n =", n)
        drs = np.diff(r)
        print("    dr min/median/max =", drs.min(), np.median(drs), drs.max())
        print("    r[0], r[1], r[-1] =", r[0], r[1], r[-1])
        

        
        
        # START STEPPING NUMERICALLY THROUGH ATMOSPHERE FROM THE SURFACE OUTWARD
        # ------------------------------------------------------------
        #
        # Multi-pass TOA optical depth closure
        # ------------------------------------------------------------
        N_PASS  = 2
        TOL_REL = 1e-4     # stop early if T converges

        tau_above_prev = None  # will hold τ_above from previous pass

        # Outer loop begins
        for ipass in range(N_PASS):

            if option > 0:
                print(f"\n    [TOA multipass] pass {ipass+1}/{N_PASS}")

            T_prev_pass = T.copy()
            
            
            # ============================================================
            # RESET (multipass-safe): reset accumulators + τ bookkeeping ONLY
            # Do NOT restart the profile arrays (T,P,rho,composition).
            # Largely unnecessary, but a precaution.
            # ============================================================

            # Reset cumulative gas mass (prevents potential double-counting across passes)
            mass[:] = 0.0
            mass_cond[:] =0.0
            N[:]    = 0.0

            # Re-seed mass[0] using convention (shell r0->r1)
            rho0 = float(density[0])          # keep current base rho from previous pass
            dm0  = (4.0*pi/3.0) * rho0 * (float(r[1])**3 - float(r[0])**3)
            mass[0] = dm0
            N[0] = mass[0] / (4.0*pi*float(r[0])**2)

            # Clear τ_above and κ saves (OK)
            tau_above[:] = 0.0
            kappa_si_save[:] = 0.0

            # ============================================================
            
            
            T_prev_pass = T.copy()
            
            
            # reset attachment markers each pass (they depend on τ used)
            i_schw = None
            tau_schw = None
            tau_attach = None
            T_attach   = None
            i_attach   = None

            # (optional) clear τ_above and κ saves
            tau_above[:] = 0.0
            kappa_si_save[:] = 0.0

            # ----------------------------
            # OUTWARD INTEGRATION
            # ----------------------------
            inhib_strength_prev = 0.0
            
            # Ledoux cap state (per pass)
            ledoux_active = False

            ledoux_lnP0  = None
            ledoux_lnmu0 = None
            ledoux_best_dlnP  = 0.0
            ledoux_best_dlnmu = 0.0
            ledoux_i_start = None

            # "pending exit" (we arm exit early, finalize cap later after nabla_ad exists)
            ledoux_pending_exit = False
            ledoux_exit_i = None
            ledoux_exit_dlnP_best  = 0.0
            ledoux_exit_dlnmu_best = 0.0
            ledoux_exit_dlnP_min   = 0.0
            
            for i in range(1, n-1):

                # ------------------------------------------------------------
                # 0) geometry + progress
                # ------------------------------------------------------------
                dr = float(r[i] - r[i-1])
                if (not np.isfinite(dr)) or (dr <= 0.0):
                    raise RuntimeError(f"[DBG] bad dr at i={i}: r[i]={r[i]:.6e}, r[i-1]={r[i-1]:.6e}, dr={dr}")

                percent = int((i/(n-1))*100)
                if i % max(int(n/20), 1) == 0 and i > 1:
                    time.sleep(0.05)
                    sys.stdout.write(f"\r{percent}%")
                    sys.stdout.flush()

                # shorthand at i-1
                Tprev    = float(T[i-1])
                Pprev    = float(P[i-1])
                rprev    = float(r[i-1])
                gprev    = float(g[i-1])
                rho_prev = float(density[i-1])

                if (not np.isfinite(Tprev)) or (Tprev <= 0.0):
                    raise RuntimeError(f"[DBG] bad Tprev at i={i}: T[i-1]={Tprev}")
                    
                    
                # ------------------------------------------------------------
                # Ledoux inhibition (STEP-LEVEL): local transport control
                # ------------------------------------------------------------
                # Purpose:
                #   Decide, at each radial step (i-1 -> i), whether convection should be
                #   suppressed due to a stabilizing mean-molecular-weight (mu) gradient,
                #   and smoothly modify the local temperature gradient used to integrate T.
                #
                # Conceptual role:
                #   This is a LOCAL buoyancy/transport-mode controller. It changes how steep
                #   dT/dr is in the current cell by blending between the baseline gradient
                #   (convective if unstable, otherwise radiative/required) and the gradient
                #   required to carry the imposed flux.
                #
                # What it does NOT do:
                #   - Does NOT cap or redefine the global intrinsic luminosity Lint.
                #   - Does NOT compute any integrated (layer-spanning) Ledoux constraint.
                #   - Acts continuously and locally, not as a one-time layer bottleneck.
                #
                # Numerics:
                #   A smooth inhibition strength w in [0,1] is built from:
                #     (i) a local estimate of nabla_mu = d ln(mu) / d ln(P),
                #     (ii) a condensation/threshold criterion (e.g., Markham) indicating when
                #          composition stratification is relevant.
                #   Then:
                #     nabla_used = (1-w)*nabla_base + w*nabla_req
                #   and T is updated using nabla_used for this step.

                # 1) mu-gradient gate from MW_gas (stabilizing if mu increases inward)
                # Guard first few steps to avoid initialization wiggles.
                if i < 3:
                    nabla_mu_step = 0.0
                    mu_gate_step  = 0.0
                else:
                    # define "deep" and "shallow" so dlnP is positive
                    P_deep  = float(max(P[i-2], 1.0))
                    P_shal  = float(max(P[i-1], 1.0))
                    mu_deep = float(max(MW_gas[i-2], 1e-30))
                    mu_shal = float(max(MW_gas[i-1], 1e-30))

                    dlnP  = float(np.log(P_deep)  - np.log(P_shal))   # >0 if P_deep > P_shal
                    dlnmu = float(np.log(mu_deep) - np.log(mu_shal))  # >0 if mu_deep > mu_shal

                    # if pressure change is too tiny, don't trust the gradient
                    if abs(dlnP) < 1e-12:
                        nabla_mu_step = 0.0
                        mu_gate_step  = 0.0
                    else:
                        nabla_mu_step = dlnmu / dlnP

                        # smooth gate
                        mu0 = 1e-3
                        q   = 4.0
                        t = (max(nabla_mu_step, 0.0) / mu0)**q
                        mu_gate_step = t / (1.0 + t)

                # 2) Markham threshold at i-1
                heavy_step = 1.0 - float(xH2atmosphere[i-1])
                xthr_step  = float(x_inhib[i-1])

                # boolean inhib (for clarity / branching if needed)
                inhib_step = bool(Inhibit and (heavy_step > xthr_step) and (mu_gate_step > 0.0))

                # smooth strength (recommended)
                p = 4.0
                ratio = heavy_step / max(xthr_step, 1e-30)
                s = ratio**p
                markham_strength_step = s / (1.0 + s)
                inhib_strength_step = float(Inhibit) * float(mu_gate_step) * float(markham_strength_step)
                # --- NEW: radial smoothing to kill 1-cell shelves at the top of the layer ---
                beta = 0.5  # 0=no smoothing, 0.3 light, 0.5 moderate, 0.8 strong
                inhib_strength_step = (1.0 - beta) * float(inhib_strength_step) + beta * float(inhib_strength_prev)
                inhib_strength_prev = float(inhib_strength_step)

                if (i < 5):
                    print("    [inhib_step dbg] i=", i,
                          " Pprev=", Pprev,
                          " MW_prev=", MW_gas[i-1],
                          " nabla_mu_step=", nabla_mu_step,
                          " mu_gate_step=", mu_gate_step,
                          " heavy_step=", heavy_step,
                          " xthr_step=", xthr_step,
                          " inhib_strength_step=", inhib_strength_step)
                # --- END: step-level inhibition control ---

                # ------------------------------------------------------------
                # Ledoux cap on Lint: ENTER / ACTIVE (early). EXIT is ARMED here.
                # Final cap is computed later after nabla_ad exists.
                # ------------------------------------------------------------

                in_ledoux = (Inhibit and (inhib_strength_step > 0.0))

                # ENTER
                if in_ledoux and (not ledoux_active):
                    ledoux_active = True

                    ledoux_lnP0  = float(np.log(max(Pprev, 1.0)))
                    ledoux_lnmu0 = float(np.log(max(MW_gas[i-1], 1e-30)))

                    ledoux_best_dlnP  = 0.0
                    ledoux_best_dlnmu = 0.0
                    ledoux_i_start = i

                # ACTIVE: update best baseline reached
                if ledoux_active:
                    lnP_now  = float(np.log(max(Pprev, 1.0)))
                    lnmu_now = float(np.log(max(MW_gas[i-1], 1e-30)))

                    dlnP  = float(ledoux_lnP0  - lnP_now)   # >0 outward if P drops outward
                    dlnmu = float(ledoux_lnmu0 - lnmu_now)

                    if dlnP > ledoux_best_dlnP:
                        ledoux_best_dlnP  = dlnP
                        ledoux_best_dlnmu = dlnmu

                # EXIT: ARM the exit (do NOT compute Lmax here; nabla_ad/kappa not ready yet)
                if (not in_ledoux) and ledoux_active:
                    ledoux_active = False

                    ledoux_pending_exit = True
                    ledoux_exit_i = i
                    ledoux_exit_dlnP_best  = float(ledoux_best_dlnP)
                    ledoux_exit_dlnmu_best = float(ledoux_best_dlnmu)

                    # baseline guard based on local scale height (uses i-1 quantities, available now)
                    Hp_loc = float(Pprev) / max(float(rho_prev) * float(gprev), 1e-30)
                    ledoux_exit_dlnP_min = max(1e-6, 3.0 * abs(float(dr)) / max(Hp_loc, 1e-30))

                    # clear entry markers
                    ledoux_lnP0 = None
                    ledoux_lnmu0 = None
                    ledoux_best_dlnP = 0.0
                    ledoux_best_dlnmu = 0.0
                    ledoux_i_start = None

                                 
                
                # ------------------------------------------------------------
                # 1) TRANSPORT / gradT block -> compute T[i] ONCE
                # ------------------------------------------------------------
                metal_mole_fraction = 1.0 - float(xH2atmosphere[i-1])
                Zatm = compute_metallicity(metal_mole_fraction)

                kappa_t_cgs = float(my_kappa(Tprev, Pprev, Zatm))
                kappa_si = max(kappa_t_cgs * 0.1, 0.0)
                kappa_si_save[i-1] = kappa_si  # <-- SAVE κ at i-1 for τ_above sweep

                # local proxy τ (still compute, for diagnostics)
                tau_loc = kappa_si * Pprev / max(gprev, 1e-30)
                tau_loc = max(float(tau_loc), 1e-30)

                # ============================================================
                #  MULTIPASS KEY
                # Use τ_above from previous pass once available; else use tau_loc.
                # This is the τ one should feed into Eddington/TOA blending logic.
                # ============================================================
                if (ipass == 0) or (tau_above_prev is None):
                    tau_f = tau_loc
                else:
                    tau_f = max(float(tau_above_prev[i-1]), 1e-30)

                # store τ for plotting/output:
                # On pass0: tau is proxy; on later passes: it becomes τ_above
                tau[i] = tau_f
                

                # ------------------------------------------------------------
                #   - replace tau_loc_f -> tau_f where it represents "the τ used for handoff"
                #   - keep tau_loc_f where "proxy τ used only as diagnostic"
                # ------------------------------------------------------------

                r_loc = float(r[i-1])
                r2    = max(r_loc*r_loc, 1e-30)

                
                Lrad[i] = Lint

                rho = max(float(density[i-1]), 1e-30)

                # --- total and internal proxy powers (unchanged) ---
                T3_tot = max(Tprev**3, 1e-30)
                T4_int = max(Tprev**4 - Teq**4, 1e-30)
                T3_int = max(T4_int**0.75, 1e-30)

                # --- anchor internal diffusion correction to Schwarzschild crossover ---
                # CHANGED: use tau_f consistently (not tau_loc)
                tau_anchor = tau_f if (tau_schw is None) else tau_schw

                TAU_INT_START = 1000.0 * tau_anchor
                TAU_INT_END   =   10.0 * tau_anchor

                # CHANGED: use tau_f for window
                if tau_f >= TAU_INT_START:
                    f_int = 0.0
                elif tau_f <= TAU_INT_END:
                    f_int = 1.0
                else:
                    x = (np.log(tau_f) - np.log(TAU_INT_END)) / (np.log(TAU_INT_START) - np.log(TAU_INT_END))
                    f_int = 1.0 - float(x)

                T3_eff = (1.0 - f_int) * T3_tot + f_int * T3_int
                T3_eff = max(float(T3_eff), 1e-30)

                k_rad_dif = (3.0 * float(kappa_si) * float(rho) * float(Lint)) / (64.0 * float(pi) * float(sigma) * T3_eff * float(r2))

                k_rad = float(k_rad_dif)
                if (not np.isfinite(k_rad)) or (k_rad < 0.0):
                    k_rad = 0.0

                # optically-thin correction:
                TAU_RAD_MAX = 100.0
                # CHANGED: use tau_f as the τ in exp(-τ) etc.
                if tau_f <= TAU_RAD_MAX:

                    dtau_dr = float(kappa_si) * float(rho)
                    if (not np.isfinite(dtau_dr)) or (dtau_dr < 0.0):
                        dtau_dr = 0.0

                    expfac = np.exp(-min(float(tau_f), 700.0))

                    dT_dtau = (1.0 / (4.0 * T3_eff)) * (
                        (3.0/4.0) * float(Tint)**4 +
                        (3.0/8.0) * float(Teq)**4 * expfac
                    )

                    k_rad_rad = float(dT_dtau) * float(dtau_dr)
                    if (not np.isfinite(k_rad_rad)) or (k_rad_rad < 0.0):
                        k_rad_rad = 0.0

                    k_rad = (k_rad_dif * k_rad_rad) / (k_rad_dif + k_rad_rad + 1e-99)
                    if (not np.isfinite(k_rad)) or (k_rad < 0.0):
                        k_rad = 0.0

                # conduction / convection etc. (UNCHANGED)
                CpH2 = Cp_H2()
                CpSiO = Cp_SiO()
                CpO2 = Cp_O2()
                CpMg = Cp_Mg()
                Cp_gas = float(xH2atmosphere[i-1]) * CpH2 + (1.0 - float(xH2atmosphere[i-1])) * (0.333*CpO2 + 0.333*CpSiO + 0.333*CpMg)

                k_cond = float(nabla_cond(Tprev, m_molec[i-1], Cp_gas, xH2atmosphere[i-1], Lint, RB, rprev))
                k_cond = max(k_cond, 1e-30)

                k_nc = (k_rad * k_cond) / (k_rad + k_cond + 1e-99)

                Sm, Hm, Gm = melt_thermo(Tprev)
                SO2, HO2, GO2 = O2_thermo(Tprev, Pprev)
                SSiO, HSiO, GSiO = SiO_thermo(Tprev, Pprev)
                SMg, HMg, GMg = Mg_thermo(Tprev, Pprev)
                H_cond[i-1] = abs((Hm - (HO2 + HSiO + HMg))) * Hmultiplier

                xd, xv, xc = mole_fractions(wt_frac_condensed[i-1], xH2melt[i-1], xH2atmosphere[i-1], MW_gas[i-1], MW_melt[i-1])
                k_conv = float(nabla_conv(Tprev, H_cond[i-1], m_molec[i-1], gprev, xd, xv, xc, 0.0, Pprev))

                k_conv_saved[i] = k_conv
                k_cond_saved[i] = k_cond
                k_nc_saved[i]   = k_nc
                k_rad_saved[i]  = k_rad

                # ------------------------------------------------------------
                # Schwarzschild / attachment:
                # CHANGED: use tau_f as “the τ coordinate”
                # ------------------------------------------------------------
                rho_loc = max(rho_prev, 1e-30)
                g_loc   = max(gprev,    1e-30)
                P_loc   = max(Pprev,    1e-30)
                T_loc   = max(Tprev,    1e-30)

                def _nabla_from_k(k_val):
                    return (k_val * P_loc) / (T_loc * rho_loc * g_loc)

                def _k_from_nabla(nabla_val):
                    return nabla_val * (T_loc * rho_loc * g_loc / P_loc)

                nabla_req = _nabla_from_k(k_nc)
                nabla_ad  = _nabla_from_k(k_conv)
                
                # ------------------------------------------------------------
                # Ledoux cap on Lint: global flux bottleneck
                # ------------------------------------------------------------
                # Purpose:
                #   Treat a finite-thickness Ledoux-stable layer as a GLOBAL bottleneck on
                #   intrinsic luminosity. While inside the layer we accumulate the best
                #   (largest) lnP baseline; on EXIT we compute a single Lmax_Ledoux and cap
                #   Lint once.
                #
                # Conceptual role:
                #   This is a NON-LOCAL throughput constraint: it answers "what is the maximum
                #   intrinsic luminosity that can pass through the ENTIRE Ledoux layer without
                #   exceeding the Ledoux-stable critical gradient?"
                #
                # Physics:
                #   Define:
                #     nabla_crit = nabla_ad + nabla_mu,   where nabla_mu = d ln(mu)/d ln(P) (stabilizing if >0)
                #   Radiative diffusion gives:
                #     Lmax_Ledoux = (64*pi*sigma_sb*G*M*T^4 / (3*kappa*P)) * nabla_crit
                #   On EXIT:
                #     Lint <- min(Lint, Lmax_Ledoux)
                #     Tint <- (Lint / (4*pi*sigma_sb*r0^2))^(1/4)   (keep Tint consistent with active Lint)
                #
                # What it does NOT do:
                #   - Does NOT directly change the cell-by-cell temperature gradient (that is Block 1).
                #   - Does NOT repeatedly clamp Lint inside the layer; it clamps once per layer on exit.
                #   - Uses integrated baselines to avoid grid-step noise and fine tuning.
                
                if ledoux_pending_exit:

                    i_end = int(ledoux_exit_i)
                    dlnP_best  = float(ledoux_exit_dlnP_best)
                    dlnmu_best = float(ledoux_exit_dlnmu_best)
                    dlnP_min   = float(ledoux_exit_dlnP_min)
                    
                    # Save this as the final Ledoux layer top index
                    ledoux_top_i_final = i_end

                    if dlnP_best > dlnP_min:

                        nabla_mu = max(0.0, dlnmu_best / max(dlnP_best, 1e-30))

                        # one very soft sanity guard (optional; keep or remove)
                        if nabla_mu > 1e3:
                            nabla_mu = 1e3

                        nabla_crit = float(nabla_ad) + float(nabla_mu)

                        P_loc  = float(max(Pprev, 1e-30))
                        T_loc  = float(max(Tprev, 1.0))
                        kappai = float(max(kappa_si, 1e-20))
                        Mi     = float(max(float(Mc) + float(mass[i-1]) + float(mass_cond[i-1]), float(Mc)))
                        sigma_sb = 5.670374419e-8

                        Lmax = (64.0 * pi * sigma_sb * G * Mi * T_loc**4 / (3.0 * kappai * P_loc)) * float(nabla_crit)

                        if np.isfinite(Lmax) and (Lmax > 0.0):
                            Lint_old = float(Lint)
                            Lint = min(float(Lint), float(Lmax))
                            ledoux_limited = True

                            # keep Tint consistent with active Lint
                            if (np.isfinite(Lint)) and (Lint > 0.0):
                                Tint = (Lint / (4.0 * pi * sigma * (r[0]**2)))**0.25
                            else:
                                Tint = 0.0

                            print(
                                f"    [Ledoux EXIT] i_end={i_end} r/r0={float(r[i_end])/float(r[0]):.6f} "
                                f"dlnP_best={dlnP_best:.3e} (min={dlnP_min:.3e}) "
                                f"nabla_mu={nabla_mu:.3e} nabla_crit={nabla_crit:.3e} "
                                f"Lmax Ledoux={Lmax:.3e} Lint_old={Lint_old:.3e} Lint_new={Lint:.3e}"
                            )
                        else:
                            print(
                                f"    [Ledoux EXIT] i_end={i_end} r/r0={float(r[i_end])/float(r[0]):.6f} "
                                f"dlnP_best={dlnP_best:.3e} (min={dlnP_min:.3e}) "
                                f"(bad Lmax={Lmax}) no cap applied"
                            )

                    else:
                        print(
                            f"    [Ledoux EXIT] i_end={i_end} r/r0={float(r[i_end])/float(r[0]):.6f} "
                            f"dlnP_best={dlnP_best:.3e} < dlnP_min={dlnP_min:.3e} "
                            f"(baseline too small) no cap applied"
                        )

                    # clear pending exit
                    ledoux_pending_exit = False
                    ledoux_exit_i = None
                    ledoux_exit_dlnP_best = 0.0
                    ledoux_exit_dlnmu_best = 0.0
                    ledoux_exit_dlnP_min = 0.0

                is_convective = (nabla_req > nabla_ad)

                if (tau_attach is None) and (nabla_req <= nabla_ad):
                    tau_attach = tau_f
                    T_attach   = Tprev
                    i_attach   = i

                if (tau_schw is None) and (tau_attach is not None):
                    tau_schw = tau_attach
                    i_schw   = i_attach

                # ------------------------------------------------------------
                # Below/above attachment logic:
                # CHANGED: compare using tau_f, evaluate Eddington at tau_f
                # ------------------------------------------------------------
                if (tau_attach is None) or (tau_f >= tau_attach):
                    nabla_base = nabla_ad if is_convective else nabla_req # new
                    # --- inhibition: push gradient toward radiative-required gradient ---
                    w = float(min(max(inhib_strength_step, 0.0), 1.0)) # new
                    nabla_used = (1.0 - w) * float(nabla_base) + w * float(nabla_req) # new

                    k_used = max(_k_from_nabla(nabla_used), 0.0)

                    alpha = alpha_smooth_gradT
                    gradT_saved[i] = alpha * k_used + (1.0 - alpha) * gradT_saved[i-1]
                    T_new = Tprev - gradT_saved[i] * dr

                    r_call = max(r[i-1], 1e-30)
                    Tint_r = Tint * (r[0] / r_call)**0.5
                    Ttry = T_eddington(tau_f, Tint_r, Teq)
                    if np.isfinite(Ttry) and (Ttry > 0.0):
                        Teddington[i] = Ttry
                    else:
                        Teddington[i] = Teddington[i-1]

                else:

                    # ---- TEST 1: pure Eddington above attachment (no stitch/shift) ----
                    r_call = max(r[i-1], 1e-30)
                    Tint_r = Tint * (r[0] / r_call)**0.5
                    Tedd_here = T_eddington(tau_f, Tint_r, Teq)

                    if (not np.isfinite(Tedd_here)) or (Tedd_here <= 0.0):
                        T_new = Tprev
                        Teddington[i] = Teddington[i-1]
                    else:
                        T_new = Tedd_here
                        Teddington[i] = Tedd_here
                        
                

                # limiter (unchanged)
                if (not np.isfinite(T_new)) or (T_new <= 1.0):
                    T_new = max(1.0, 0.99*Tprev)
                else:
                    dT = Tprev - T_new
                    dT_max = 0.10 * max(Tprev, 1.0)
                    if dT > dT_max:
                        T_new = Tprev - dT_max

                T[i] = T_new

                if (i_rcb is None) and (i_attach is not None):
                    i_rcb = i_attach

                k_ratio[i] = nabla_req / (nabla_ad + 1e-30)

                # ------------------------------------------------------------
                # 2) MASS / COLUMN (uses rho_prev)
                # ------------------------------------------------------------
                msoln = quad(dmdr, r[i-1], r[i], args=(rho_prev,))
                dm = float(msoln[0])
                mass[i] = mass[i-1] + dm
                N[i] = mass[i] / (4.0*pi*(r[i]**2.0))

                # ------------------------------------------------------------
                # 3) SOLVUS / COMPOSITION at this level
                # ------------------------------------------------------------
                T_solvus = float(Tprev)  # keep i-1 convention for solvus call
                if T_solvus > 1500.0:
                    PinputGPa = Pprev / 1.0e9
                    bulk_in = float(np.clip(bulk_Rayleigh[i-1], 1e-6, 100.0 - 1e-6))
                    nudges = [0.0, -1e-4, +1e-4, -5e-4, +5e-4]

                    solved = False
                    last_err = None
                    bulk_in = float(np.clip(bulk_in, 1e-4, 100.0 - 1e-9))
                    for dP in nudges:
                        try:
                            xH2sil, xH2atm, Tcrest, wt_silicate, wt_atmosphere = subregular(T_solvus, bulk_in, PinputGPa + dP)
                            solved = True
                            break
                        except RuntimeError as e:
                            last_err = e
                            continue

                    if not solved:
                        xH2atm = xH2atmosphere[i-1]
                        xH2sil = xH2melt[i-1]
                        wt_silicate = wt_frac_condensed[i-1]
                else:
                    xH2atm = 1.0 - 1e-6
                    xH2sil = 1e-7
                    wt_silicate = 1e-8

                # commit compositions
                xH2atmosphere[i] = float(np.clip(xH2atm, 1e-9, 1.0-1e-9))
                xH2melt[i]       = float(np.clip(xH2sil, 1e-9, 1.0-1e-9))
                wt_frac_condensed[i] = float(np.clip(wt_silicate, 1e-8, 1.0-1e-6))

                # ------------------------------------------------------------
                # 3b) RESTORE YOUR CONDENSATE BOOKKEEPING (right place)
                # ------------------------------------------------------------
                # weight percent conversions (use these downstream a lot)
                wt_frac_H2_melt[i] = mole_fraction_to_weight_percent(xH2melt[i]) / 100.0
                mass_frac_H2atm[i] = mole_fraction_to_weight_percent(xH2atmosphere[i]) / 100.0

                # H2 mass in *this* gas shell (increment, not cumulative)
                dmass_H2_atm = dm * mass_frac_H2atm[i]
                mass_H2_atm[i] = dmass_H2_atm

                # Perfect Rayleigh: next bulk is current gas wt% (convention)
                bulk_Rayleigh[i] = mass_frac_H2atm[i-1] * 100.0

                # mass bookkeeping
                # avoid division by zero if wt_frac_condensed is extremely close to 1
                wf_prev = float(np.clip(wt_frac_condensed[i], 1e-12, 1.0 - 1e-12))
                dm_cond = dm * wf_prev / (1.0 - wf_prev)
                mass_cond[i] = mass_cond[i-1] + dm_cond
                mass_H2_cond[i] = dm_cond * wt_frac_H2_melt[i-1]


                # (MW_gas/MW_melt defined just below; temporarily use i-1 if needed)
                # We'll compute MWs next and then overwrite x_cond[i] cleanly after MW exists.

                # ------------------------------------------------------------
                # 3c) MW bookkeeping
                # ------------------------------------------------------------
                MW_O2 = 23/1000
                MW_SiO = (29+16)/1000
                MW_Mg = 24/1000

                mean_mol_wt_gas[i] = xH2atmosphere[i]*MWH2 + (1.0 - xH2atmosphere[i]) * (0.333*MW_O2 + 0.333*MW_SiO + 0.333*MW_Mg)
                MW_gas[i]  = mean_mol_wt_gas[i]
                m_molec[i] = MW_gas[i] / Av_number
                

                MW_melt[i] = xH2melt[i]*MWH2 + (1.0 - xH2melt[i])*MWMgSiO3

                # Now that MWs exist at i, set x_cond[i]
                mole_ratio = float(wt_frac_condensed[i]) / (1.0 - float(wt_frac_condensed[i])) * (float(MW_gas[i]) / float(MW_melt[i]))
                x_cond[i] = mole_ratio / (mole_ratio + 1.0)


                # ------------------------------------------------------------
                # 3d) Inhibition parameter update (Markham threshold x_inhib)
                # ------------------------------------------------------------
                MW_vap = (0.333*32/1000 + 0.333*(28+16)/1000 + 0.333*24/1000)
                epsilon = MW_vap / MWH2

                Ti = float(max(T[i], 1.0))
                phi = float(H_cond[i]) / (float(Rgas) * Ti)   # = H_cond/(R T)

                # --- ΔH trap: if phi <= 1, Markham threshold formula is not meaningful ---
                # In this regime, prevent x_inhib from collapsing to ~0 (which would make inhibition always "on").
                if (not np.isfinite(phi)) or (phi <= 1.0):
                    x_inhib[i] = float(np.clip(x_inhib[i-1], 1e-12, 1.0 - 1e-12))
                else:
                    denom = (phi - 1.0) * (float(epsilon) - 1.0)
                    if (np.isfinite(denom)) and (abs(denom) > 1e-30):
                        xtry = 1.0 / denom
                        x_inhib[i] = float(np.clip(xtry, 1e-12, 1.0 - 1e-12))
                    else:
                        x_inhib[i] = float(np.clip(x_inhib[i-1], 1e-12, 1.0 - 1e-12))

#                # “keep away from 1”
#                if xH2atmosphere[i] > 0.9999:
#                    xH2atmosphere[i] = 0.9990
#                xMgSiO3_atm[i] = 1.0 - xH2atmosphere[i]
                

                # Smooth floor on heavy fraction h = 1 - xH2 to avoid --> 0
                hmin = 1.0e-3     # matches the old hard clamp to x=0.9990
                h0   = 3.0e-4     # transition softness (smaller = sharper; tune)
                x = float(xH2atmosphere[i])
                h = 1.0 - x
                if h < hmin:
                    # smooth "max": h -> hmin + softplus(h - hmin)
                    # softplus(z) = h0 * ln(1 + exp(z/h0))
                    z = (h - hmin) / h0
                    # guard exp overflow
                    if z > 50.0:
                        h_eff = h
                    else:
                        h_eff = hmin + h0 * float(np.log1p(np.exp(z)))
                    x = 1.0 - h_eff
                # keep away from exactly 1 for safety
                xH2atmosphere[i] = float(min(max(x, 1e-12), 1.0 - 1e-12))
                xMgSiO3_atm[i]   = 1.0 - xH2atmosphere[i]
                


                # ------------------------------------------------------------
                # 4) GRAVITY at i (uses updated masses)
                # ------------------------------------------------------------
                m_under = Mc + mass[i] + mass_cond[i]
                g[i] = G * m_under / (r[i]**2.0)

                # ------------------------------------------------------------
                # 5) PRESSURE update (DEBUG-STABLE): pure HSE from i-1 -> i
                # ------------------------------------------------------------
                dP_hse = -float(density[i-1]) * float(gprev) * dr
                P[i] = max(1.0, float(Pprev) + dP_hse)
                dPdr_save[i] = dP_hse / dr

                # ------------------------------------------------------------
                # 6) DENSITY update (EOS, one-way): rho[i] = μ P / (Z R T)
                # ------------------------------------------------------------
                mmw = max(float(MW_gas[i]), 1e-30)

                Z_here = H2_density(float(T[i]), float(P[i]))
                if (not np.isfinite(Z_here)) or (Z_here <= 0.0):
                    print(f"[DBG EOS] bad Z at i={i}: Z={Z_here}  T={T[i]}  P={P[i]}")
                    Z_here = 1.0

                density[i] = max(1e-12, (mmw * float(P[i])) / (Z_here * Rgas * float(T[i])))

                drhodmw_s[i] = float(P[i]) / (max(Z_here, 1e-12) * Rgas * float(T[i]))

                # ------------------------------------------------------------
                # 7) Entropy / adiabat toggle
                # ------------------------------------------------------------
                rcb_limit = 1.0
                if (not adiabat) and (k_ratio[i] < rcb_limit):
                    adiabat = True
                    ircb = i

            # ----------------------------
            # OUTWARD integration finished
            # ----------------------------
            
            
            # ============================================================
            # POST-PASS: 2-sided slope-match near Schwarzschild and blend
            # Put this AFTER the for i in range(...) loop, BEFORE tau_above sweep
            # ============================================================

            if i_schw is not None and 2 <= i_schw < n-2:

                # ---- knobs (simple) ----
                SEARCH_UP   = 5000     # points above i_schw (toward TOA)
                SEARCH_DOWN = 5000     # points below i_schw (deeper)
                TOL_REL_SLOPE = 0.10   # accept when slopes match to 10% (try 0.05–0.20)
                BLEND_HALF = int(0.0167*n)       # blend window half-width in index space (try 0.0167*n)

                # --- helper: slope magnitude |dT/dr| at index j using central difference ---
                def slope_mag_arr(Tarr, j):
                    drc = r[j+1] - r[j-1]
                    if drc <= 0.0:
                        return None
                    return abs((Tarr[j+1] - Tarr[j-1]) / drc)

                # --- helper: Eddington temperature at index j (guarded) ---
                def Tedd_at_index(j):
                    # spherical flux correction: F ∝ 1/r^2  ⇒  Tint(r) ∝ r^{-1/2}
                    rj = max(r[j], 1e-30)
                    Tint_r = Tint * (r[0] / rj)**0.5

                    tj = T_eddington(tau[j], Tint_r, Teq)
                    if (not np.isfinite(tj)) or (tj <= 0.0):
                        return None
                    return tj

                # --- helper: Eddington slope magnitude at j ---
                def slope_mag_edd(j):
                    t_m1 = Tedd_at_index(j-1)
                    t_p1 = Tedd_at_index(j+1)
                    if (t_m1 is None) or (t_p1 is None):
                        return None
                    drc = r[j+1] - r[j-1]
                    if drc <= 0.0:
                        return None
                    return abs((t_p1 - t_m1) / drc)

                i0 = i_schw

               # ============================================================
                # Post-pass RCB Patch: Slope-Matched Hermite Bridge
                # ============================================================
                #
                # Purpose
                # -------
                # After each TOA multipass iteration, the outward-integrated
                # interior temperature profile T(r) (convective/diffusive)
                # may not join smoothly onto the imposed irradiated outer
                # atmosphere solution T_eddington(τ).
                #
                # This mismatch typically appears near the radiative–convective
                # boundary (RCB) as a kink or plateau, because the two solutions
                # are effectively satisfying different boundary conditions:
                #
                #   • Interior march outward: controlled by gradT transport
                #   • Exterior Eddington: controlled by Teq + TOA radiative balance
                #
                # The goal of this patch is NOT to replace the physics,
                # but to prevent an unphysical sharp jump in dT/dr across the handoff.
                #
                #
                # Strategy Overview
                # -----------------
                # (1) Locate a "best" patch center index best_j near the nominal RCB.
                #
                #     Instead of forcing a match exactly at i0 (the first Schwarzschild-
                #     stable point), we search within a small window around i0 for the
                #     radius where the local slope magnitudes agree:
                #
                #         sb(j) = |dT/dr| from the interior-integrated profile
                #         se(j) = |dT/dr| from the Eddington solution
                #
                #     The best_j minimizes:
                #
                #         rel = |se - sb| / sb
                #
                #     This avoids locking to a pathological local slope at i0.
                #
                #
                # (2) Define a blending/patch window around best_j:
                #
                #         [j0, j1] = [best_j - BLEND_HALF, best_j + BLEND_HALF]
                #
                #     Inner endpoint (j0): use the interior solution T[j0]
                #     Outer endpoint (j1): pin to the Eddington temperature Tedd(j1)
                #
                #
                # (3) Compute endpoint slopes:
                #
                #     • m0 = dT/dr from the interior profile at the inner edge
                #     • m1 = dT/dr from the Eddington curve at the outer edge
                #
                #     These represent the physical flux slopes on each side.
                #
                #
                # (4) Build a smooth C¹ bridge using a cubic Hermite interpolant:
                #
                #     The Hermite polynomial enforces continuity of BOTH:
                #
                #         • Temperature T(r)
                #         • First derivative dT/dr (i.e., radiative/convective flux proxy)
                #
                #     across the entire patch window.
                #
                #     This produces a convex-hull-like connector without ringing.
                #
                #
                # (5) Optional slope clamping ("tautness")
                #
                #     To prevent overshoot from extreme endpoint slopes,
                #     we clamp m0 and m1 relative to the mean slope across the window.
                #     This keeps the bridge well-behaved without introducing new knobs.
                #
                #
                # Outcome
                # -------
                # The atmosphere remains:
                #
                #   • Pure transport physics well below the RCB
                #   • Pure Eddington radiative equilibrium well above the RCB
                #   • Smoothly connected across a narrow numerical boundary layer
                #
                # This patch suppresses unphysical kinks while preserving the
                # correct TOA boundary condition and a realistic interior gradient.
                #
                # ============================================================

                # We'll search for a j where the below-solution slope matches the Eddington slope.
                # But instead of locking to s_b(i0), compare s_b(j) to s_e(j) at each candidate j.
                # This lets "escape" a pathological local slope at i0.
                best_j = None
                best_obj = 1e300
                best_rel = None
                best_sb = None
                best_se = None

                j_lo = max(2, i0 - SEARCH_DOWN)
                j_hi = min(n-3, i0 + SEARCH_UP)

                for j in range(j_lo, j_hi + 1):

                    sb = slope_mag_arr(T, j)
                    if sb is None or sb <= 0.0:
                        continue

                    se = slope_mag_edd(j)
                    if se is None or se <= 0.0:
                        continue

                    rel = abs(se - sb) / max(sb, 1e-30)

                    # objective: primarily slope match
                    obj = rel

                    # optional small bias to stay near i0
                    # makes the algorithm prefer nearby matches if multiple are similar
                    obj += 0.001 * abs(j - i0) / max(1.0, float(SEARCH_UP + SEARCH_DOWN))

                    if obj < best_obj:
                        best_obj = obj
                        best_j = j
                        best_rel = rel
                        best_sb = sb
                        best_se = se

                    # early accept: if we found a good slope match, stop (but only after checking both sides near i0)
                    # simplest: break immediately when rel < tol
                    if rel <= TOL_REL_SLOPE:
                        best_j = j
                        best_rel = rel
                        best_sb = sb
                        best_se = se
                        break

                # If we found a candidate, patch around it
                if best_j is not None:

                    # Blend/patch window centered at best_j (clipped)
                    j_bl_lo = max(1, best_j - BLEND_HALF)
                    j_bl_hi = min(n-2, best_j + BLEND_HALF)

                    j0 = j_bl_lo
                    j1 = j_bl_hi

                    # Need enough room for slopes
                    if j1 > j0 + 2:

                        r0 = r[j0]
                        r1 = r[j1]
                        L  = r1 - r0

                        if L > 0.0:

                            # Endpoint temperatures:
                            # inner endpoint uses current integrated profile,
                            # outer endpoint pins to Eddington.
                            T0 = T[j0]
                            T1 = Tedd_at_index(j1)

                            if (T1 is not None) and np.isfinite(T1) and (T1 > 0.0):

                                # --- slopes dT/dr at endpoints ---
                                # inner slope: from current T array (centered if possible)
                                if j0 >= 2:
                                    m0 = (T[j0] - T[j0-2]) / max(r[j0] - r[j0-2], 1e-30)
                                else:
                                    m0 = (T[j0+1] - T[j0]) / max(r[j0+1] - r[j0], 1e-30)

                                # outer slope: from Eddington curve (centered if possible)
                                if j1 <= n-3:
                                    Te_m1 = Tedd_at_index(j1-1)
                                    Te_p1 = Tedd_at_index(j1+1)
                                    if (Te_m1 is not None) and (Te_p1 is not None) and np.isfinite(Te_m1) and np.isfinite(Te_p1):
                                        m1 = (Te_p1 - Te_m1) / max(r[j1+1] - r[j1-1], 1e-30)
                                    else:
                                        m1 = 0.0
                                else:
                                    Te_m1 = Tedd_at_index(j1-1)
                                    Te_0  = Tedd_at_index(j1)
                                    if (Te_m1 is not None) and (Te_0 is not None) and np.isfinite(Te_m1) and np.isfinite(Te_0):
                                        m1 = (Te_0 - Te_m1) / max(r[j1] - r[j1-1], 1e-30)
                                    else:
                                        m1 = 0.0

                                # --- optional “tautness” clamp to avoid ringing ---
                                # (keeps the bridge convex-hull-ish; not a new knob)
                                mcap = 3.0 * abs((T1 - T0) / max(L, 1e-30))
                                if np.isfinite(mcap) and (mcap > 0.0):
                                    m0 = np.clip(m0, -mcap, mcap)
                                    m1 = np.clip(m1, -mcap, mcap)

                                # --- Hermite cubic bridge ---
                                for j in range(j0, j1 + 1):

                                    t  = (r[j] - r0) / max(L, 1e-30)   # 0..1
                                    t2 = t*t
                                    t3 = t2*t

                                    h00 =  2.0*t3 - 3.0*t2 + 1.0
                                    h10 =        t3 - 2.0*t2 + t
                                    h01 = -2.0*t3 + 3.0*t2
                                    h11 =        t3 -       t2

                                    Tj = h00*T0 + h10*(L*m0) + h01*T1 + h11*(L*m1)
                                    T[j] = max(Tj, 1.0)

                                # keep diagnostics array consistent (optional, but nice)
                                for j in range(j0, j1 + 1):
                                    Tej = Tedd_at_index(j)
                                    if (Tej is not None) and np.isfinite(Tej) and (Tej > 0.0):
                                        Teddington[j] = Tej

                    if option > 0:
                        print(
                            f"    [post-pass patch 2side] i0={i0} best_j={best_j} "
                            f"rel={best_rel:.3e} sb={best_sb:.3e} se={best_se:.3e} "
                            f"blend=[{j_bl_lo},{j_bl_hi}]"
                        )

                        # ------------------------------------------------------------
                        # CONSISTENCY CHECKS: recompute slopes the same way as the search
                        # (This catches cases where the stored best_* were from pre-patch,
                        #  but T[] has been modified by the patch.)
                        # ------------------------------------------------------------
                        sb_check = slope_mag_arr(T, best_j)
                        se_check = slope_mag_edd(best_j)

                        # show checks (guard None)
                        if sb_check is None:
                            sb_check_val = np.nan
                        else:
                            sb_check_val = sb_check

                        if se_check is None:
                            se_check_val = np.nan
                        else:
                            se_check_val = se_check

                        print(
                            f"    [check slopes] best_sb(stored)={best_sb:.3e}  sb_check(now)={sb_check_val:.3e} | "
                            f"best_se(stored)={best_se:.3e}  se_check(now)={se_check_val:.3e}"
                        )

                        # ------------------------------------------------------------
                        # EXTRA DIAGNOSTIC: show temps + slope mismatch at best_j
                        # using one consistent definition per quantity.
                        # ------------------------------------------------------------
                        rj = max(r[best_j], 1e-30)
                        Tint_r = Tint * (r[0] / rj)**0.5
                        T_edd_0 = T_eddington(max(tau[best_j], 1e-30), Tint_r, Teq)

                        # Use the SAME functions used in the objective
                        slope_int_fd = sb_check
                        slope_edd_fd = se_check

                        print("    [patch detail]")
                        print(f"    tau(best)   = {tau[best_j]:.3e}")
                        print(f"    T(best)     = {T[best_j]:.3f} K   (after patch)")
                        print(f"    T_edd(best) = {T_edd_0:.3f} K   (raw Eddington)")

                        if slope_int_fd is not None:
                            print(f"    slope_int   = {slope_int_fd:.3e} K/m   (same as sb_check)")
                        else:
                            print(f"    slope_int   = None")

                        if slope_edd_fd is not None:
                            print(f"    slope_edd   = {slope_edd_fd:.3e} K/m   (same as se_check)")
                        else:
                            print(f"    slope_edd   = None")

                        # temperature mismatch at best_j (also useful)
                        if np.isfinite(T_edd_0) and (T_edd_0 > 0.0):
                            dT_abs = abs(T[best_j] - T_edd_0)
                            dT_rel = dT_abs / max(T_edd_0, 1.0)
                            dT4_rel = abs(T[best_j]**4 - T_edd_0**4) / max(T_edd_0**4, 1e-30)
                            print(f"    |ΔT|        = {dT_abs:.3f} K   (rel {dT_rel:.3e})")
                            print(f"    Δ(T^4)/T^4  = {dT4_rel:.3e}")

                        # NOTE: blend window tau order: j increases outward, tau decreases outward
                        print(f"    blend_tau   = [{tau[j_bl_hi]:.3e},{tau[j_bl_lo]:.3e}]")

            # ============================================================
            # κ SAFETY: fill top cell opacity before downward τ_above sweep
            # ============================================================

            # The first sweep step needs kappa_si_save[n-1]
            if kappa_si_save[n-1] == 0.0:

                Tj = float(max(T[n-1], 1.0))
                Pj = float(max(P[n-1], 1.0))

                metal_mole_fraction = 1.0 - float(xH2atmosphere[n-1])
                Zatm_j = compute_metallicity(metal_mole_fraction)

                kappa_si_save[n-1] = max(
                    float(my_kappa(Tj, Pj, Zatm_j)) * 0.1,
                    0.0
                )

            # ----------------------------
            # TOA -> downward τ_above sweep (end of pass)
            # ----------------------------
            tau_above[n-1] = 0.0

            # ensure κ is available at the topmost cell too (if needed)
            # compute/save κ at n-2 if it wasn’t set (cheap safety)
            if kappa_si_save[n-2] == 0.0:
                Tj = float(max(T[n-2], 1.0))
                Pj = float(max(P[n-2], 1.0))
                metal_mole_fraction = 1.0 - float(xH2atmosphere[n-2])
                Zatm_j = compute_metallicity(metal_mole_fraction)
                kappa_si_save[n-2] = max(float(my_kappa(Tj, Pj, Zatm_j)) * 0.1, 0.0)

            for j in range(n-2, -1, -1):
                dr_abs = abs(float(r[j+1] - r[j])) + 1e-30

                # κ,ρ at j+1 (consistent with τ_above definition)
                kappa_si_j = float(kappa_si_save[j+1])
                if kappa_si_j == 0.0:
                    Tj = float(max(T[j+1], 1.0))
                    Pj = float(max(P[j+1], 1.0))
                    metal_mole_fraction = 1.0 - float(xH2atmosphere[j+1])
                    Zatm_j = compute_metallicity(metal_mole_fraction)
                    kappa_si_j = max(float(my_kappa(Tj, Pj, Zatm_j)) * 0.1, 0.0)

                rho_j = float(max(density[j+1], 1e-30))
                tau_above[j] = tau_above[j+1] + kappa_si_j * rho_j * dr_abs

            # store for next pass
            tau_above_prev = tau_above.copy()

            # overwrite tau[] with τ_above for output after pass>=1
            if ipass >= 1:
                tau[:] = tau_above[:]

            # convergence check (early exit)
            rel = np.max(np.abs(T - T_prev_pass) / np.maximum(T, 1.0))
            if option > 0:
                print(f"    [TOA multipass] max rel ΔT = {rel:.3e}")
                if tau_attach is not None:
                    print(f"    [TOA multipass] tau_attach={tau_attach:.3e}  i_attach={i_attach}")

            if (ipass >= 1) and (rel < TOL_REL):
                break

        # STOP STEPPING NUMERICALLY THROUGH ATMOSPHERE FROM THE SURFACE OUTWARD
        

        if i < 200 and float(P[i]) != P_check:
            print(f"[DBG P] OVERWRITE: i={i} P changed after set: {P_check:.6e} -> {P[i]:.6e}")
        
        if option > 0:
            # report result
            Tdiff=(T[n-2]-Teq)
            print('')
            print('    Teq =',Teq)
            print('    Trad[n-2] =',T[n-2])
            print('    Trad-Teq = ',Tdiff)
            #print('   P = %5.3e' %P[i-1])
            print('    k_conv[3]= %10.3e' %k_conv_saved[3])
            print('    k_nc[3] = %10.3e' %k_nc_saved[3])
            print('    gradT[3] = %10.3e' %gradT_saved[3])
            print('    RB/Rmax = %10.2f' %RB_over_rmax_search)
            print('    Number of radial steps =',n)
            print('')
        
        
        
#        # ============================================================
#        # FINAL CONSISTENCY SWEEP (after multipass + any T patching)
#        # Recompute P, rho, mass to match the final T(r)
#        #
#        # UNECESSARY but retained in case of future use
#        # ============================================================
#
#        # Re-enforce base BCs (safe / optional but consistent)
#        T[0] = float(T_contact_use)
#        P[0] = float(Ps_use) * 1.0e9
#        density[0] = rho_gas(P[0], T[0], mean_mol_wt_gas[0])
#
#        # Reset gas mass only (avoid double-counting); KEEP condensate mass profile
#        mass[:] = 0.0
#        # mass_cond[:] = 0.0   # <-- DO NOT DO THIS HERE
#
#        # Seed mass[0] with your convention: shell mass from r[0] -> r[1]
#        rho0   = float(density[0])
#        rlower = float(r[0])
#        rupper = float(r[1])
#        dm0    = float(quad(dmdr, rlower, rupper, args=(rho0,))[0])
#        mass[0] = dm0
#        N[0]    = mass[0] / (4.0*pi*(r[0]**2.0))
#
#        # Base gravity includes condensate already present below
#        g[0] = G * float(Mc + mass[0] + mass_cond[0]) / (r[0]**2.0)
#
#        for i in range(1, n-1):
#
#            dr = float(r[i] - r[i-1])
#            if (not np.isfinite(dr)) or (dr <= 0.0):
#                raise RuntimeError(f"[final sweep] bad dr at i={i}: dr={dr}")
#
#            gprev = float(g[i-1])
#
#            # 1) HSE pressure update
#            dP_hse = -float(density[i-1]) * gprev * dr
#            P[i] = max(1.0, float(P[i-1]) + dP_hse)
#            dPdr_save[i] = dP_hse / dr
#
#            # 2) EOS density update using FINAL T[i]
#            mmw = max(float(MW_gas[i]), 1e-30)
#
#            Z_here = H2_density(float(T[i]), float(P[i]))
#            if (not np.isfinite(Z_here)) or (Z_here <= 0.0):
#                Z_here = 1.0
#
#            density[i] = max(1e-12, (mmw * float(P[i])) / (Z_here * Rgas * float(T[i])))
#            drhodmw_s[i] = float(P[i]) / (max(Z_here, 1e-12) * Rgas * float(T[i]))
#
#            # 3) Shell gas mass increment (rho treated constant over the cell, consistent with args=(rho_prev,))
#            rho_prev = float(density[i-1])
#            rlower   = float(r[i-1])
#            rupper   = float(r[i])
#            dm       = float(quad(dmdr, rlower, rupper, args=(rho_prev,))[0])
#
#            mass[i] = mass[i-1] + dm
#            N[i]    = mass[i] / (4.0*pi*(r[i]**2.0))
#
#            # 4) Gravity using updated cumulative gas mass + (unchanged) condensate mass
#            m_under = float(Mc) + float(mass[i]) + float(mass_cond[i])
#            g[i] = G * m_under / (r[i]**2.0)
#
#        # ============================================================
        
        
        
        # RETURN TEMPERATURE OBJECTIVE FUNCTION, COST, OR RESULTS
        # ensure it’s always set before returning from model
        if i_rcb is None:
            i_rcb = i_rcb_analytical
        
        # Calculate Ledoux layer top height in meters
        if ledoux_top_i_final is not None:
            ledoux_top_height_m = r[ledoux_top_i_final] - r[0]  # height above atmosphere base (R_s)
        else:
            ledoux_top_height_m = 0.0
        
        T_upper=T[n-2]
        rj = max(r[best_j], 1e-30)
        Tint_r = Tint * (r[0] / r[n-2])**0.5
        cost = (T_upper-T_eddington(0.0,Tint_r,Teq))**2
        #return cost,T_upper
        if option == 0:
            print('  T[n-2]=',T[n-2])
            return cost
        else:
            print(
                "    [Model return] "
                f"Lint={Lint:.3e}  "
                f"Tint={Tint:.2f} K  "
                f"Rrcb/Rc={r[i_rcb]/r[0]:.3f}  "
                f"source={'Ledoux cap' if ledoux_limited else 'non-Ledoux'}"
            )
            return T,P,density,mass,mass_cond,dPdr_save,MW_gas,MW_melt,m_molec,k_ratio,\
    mass_H2_atm,mass_H2_cond,x_cond,xH2atmosphere,wt_frac_condensed,k_rad_saved,k_conv_saved,k_cond_saved,\
    k_nc_saved,gradT_saved,\
    x_inhib,Sgaskg,Smeltkg,S_sys_kg,tau,Tint, i_rcb, Lint, delta_bl, ledoux_top_height_m
           
    #--------------------------------------------------------------------------------------------------------
    # FIRST, SEARCH FOR FIRST-CONTACT SOLVUS TEMPERATURE AT SPECIFIED BULK COMPOSITION
    # Supress warnings for this exercise

    # Use the subregular fnc to test crest of solvus temperature at this pressure

    # Search for the temperature that first contacts with solvus, starting at the crest of the solvus
    # and working downward in temperature

    Tcrest = critical_temperature(Ps)
    print('')
    print('Searching for T contact at P = %.3f GPa' %Ps)
    print('...and melt = %.3f H2 %%' %bulk)

    # --- Contact temperature search with hard safety ---
    # Choose fallback from something guaranteed to exist
    T_contact_fallback = float(T[0]) if ('T' in locals() and T is not None) else 2700.0

    T_contact_try = find_contact_T(Tcrest, bulk, Ps)
    print('    Contact solvus temperature try =', T_contact_try)
    T_contact = clamp_T_contact(T_contact_try)
    print('    Contact solvus temperature =', T_contact)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #_______________________________________________ PLANET MODEL ________________________________________________#

    #-------------------------------------------------------------------------------------------------------
    def H2_mass_balance():
        """
        Function to keep track of H2. Use to update H2 distribution for each Planet Model calculation. 
        
        Inputs (from calling program, not as arguments): 
        mass_H2_atm = array that contains mass H2 in gas at each radial position
        mass_H2_cond = array containing mass of H2 in condensates at each radial position
        mass = array with cumulative mass of gaseous atm, last term is total mass
        mass_cond = array with cumulative mass of condensate in atm, last term is total mass
        
        Returns:
        mass_H2atm = mass of H2 that resides as gaseous atmosphere
        mass_H2cond = mass of H2 that resides as condensates that are returned to core
        matm_Mp = ratio of mass of H2 existing as gas/mass of planet
        massfracH2_atm = mass of H2 in the gas/mass of gaseous atmosphere
        mass_H2_total = mass of H2 for the whole planet
        mass_H2_core = mass of H2 comprising the core, original bulk with H2 of gaseous atm removed
        massfracH2_core = fraction of the core composed of H2, by mass
        
        Currently setup for perfect Rayleigh removal of condensate from each layer of the atm
        and return condensate to core.
        """
        mass_H2atm=np.sum(mass_H2_atm)
        mass_H2cond=np.sum(mass_H2_cond)
        Matm_Mp_final=mass[n-2]/Mp # mass of atmosphere/mass of planet
        Mcond_Mp_final=mass_cond[n-2]/Mp # mass of condensate/mass of planet
        mH2atm_Mp=mass_H2atm/Mp # mass of H2 in gas phase/mass planet
        mH2cond_Mp=mass_H2cond/Mp # mass of H2 in condensate/mass planet
        massfracH2_atm=mass_H2atm/Matm # mass fraction of gas composed of H2
        mass_H2_total=(bulk/100)*Mp # mass fraction of H2 for whole planet
        mass_H2_core=(mass_H2_total-mass_H2atm) # remove H2 in gas but not condensate from core
        massfracH2_core=mass_H2_core/Mc
        
        return mass_H2atm, mass_H2cond, Matm_Mp_final, Mcond_Mp_final, mH2atm_Mp, \
        mH2cond_Mp, massfracH2_atm, mass_H2_total, mass_H2_core, massfracH2_core
    #-------------------------------------------------------------------------------------------------------

    # 1. CALCULATE THE ATMOSPHERE MODEL
    r_atm = gridding(R_s, rmax, int(nr), sharp)
    T,P,density,mass,mass_cond,dPdr_save,MW_gas,MW_melt,m_molec,k_ratio,mass_H2_atm,mass_H2_cond,x_cond,xH2atmosphere,wt_frac_condensed,\
    k_rad_saved,k_conv_saved,k_cond_saved,k_nc_saved,gradT_saved,x_inhib,Sgaskg,Smeltkg,S_sys_kg,tau,Tint, i_rcb, Lint, delta_bl, ledoux_top_height_m =model(2, r_atm, Ra=Ra)

    if (i_rcb is None): # trap the rcb pointer if for some reason not found
        i_rcb = 1
    print('    i_rcb =',i_rcb)
    print('    T at rcb = ',T[i_rcb])


    # 2. CALCULATE CORE USING OUTPUT FROM ATMOSPHERE MODEL
    # use structure=[0,1], or [1], or [0] to choose silicate with metal core,
    # pure silicate, or pure metal, respectively
    T0=T[0]
    P_surface=P[0] # Base pressure, from atmosphere, Pascal
    # Use new composition for core to calculate rho_0 for silicate in core
    # Add mass of atmosphere to total mass of planet, Mp
    Matm=mass[n-2]

    # Set planet mass to target value
    Mp=Mp_target

    # Set core mass to target planet mass - mass of atmosphere
    Mc=Mp_target - mass[n-2]

    # H2 MASS BALANCE
    mass_H2atm, mass_H2cond, Matm_Mp_final, Mcond_Mp_final, mH2atm_Mp, \
    mH2cond_Mp, massfracH2_atm, mass_H2_total, mass_H2_core, massfracH2_core =H2_mass_balance()

    print('    massfrac H2 atm =',massfracH2_atm)
    print('    massfrac H2 core after atmosphere extraction =',massfracH2_core)

    # Trap SURFACE PRESSURE AND THUS MASS OF ATMOSPHERE to avoid negative mass fractions of H2 in silicate
    Ps_old    = float(Ps)
    Ps_oldold = float(Ps) * 1.001  # tiny offset so denom != 0 on first use
    massfracH2_core_old = float(massfracH2_core)
    
    # While loop below is for first pass on atmosphere only, later is a similar mitigation for subsequent passes
    
    while massfracH2_core < -0.00001:
        print('Not enough H2,reducing mass of atmosphere...')
        Ps = Ps*0.8
        print('New surface pressure = %.4f GPa' %Ps)
        
        #--------------------------------------------------------------------------------------------------------
        # FIRST, SEARCH FOR FIRST-CONTACT SOLVUS TEMPERATURE AT SPECIFIED BULK COMPOSITION
        # Supress warnings for this exercise

        # Use the subregular fnc to test crest of solvus temperature at this pressure

        # Search for the temperature that first contacts with solvus, starting at the crest of the solvus
        # and working downward in temperature

        # Get solvus crest T
        Tcrest = critical_temperature(Ps)
        print('')
        print('Searching for T contact at P = %.3f GPa' %Ps)
        print('...and melt = %.3f H2 %%' %(100*massfracH2_core))
        T_contact_try = find_contact_T(Tcrest,(100*massfracH2_core),Ps)
        T_contact = clamp_T_contact(T_contact_try)
        print(' Contact solvus temperature =', T_contact)
        
        
        # RECALCULATE THE ATMOSPHERE MODEL WITH NEW T_contact
        T,P,density,mass,mass_cond,dPdr_save,MW_gas,MW_melt,m_molec,k_ratio,mass_H2_atm,mass_H2_cond,x_cond,xH2atmosphere,wt_frac_condensed,\
        k_rad_saved,k_conv_saved,k_cond_saved,k_nc_saved,gradT_saved,x_inhib,Sgaskg,Smeltkg,S_sys_kg,tau,Tint,i_rcb, Lint, delta_bl, ledoux_top_height_m=model(2, r_atm, Ra = Ra,T_contact_in = T_contact, Ps_in = Ps)

        T0=T_contact
        P_surface=P[0] # Pascal
        # Use new composition for core to calculate rho_0 for silicate in core
        # Add mass of atmosphere to total mass of planet, Mp
        
        # TOTAL PLANET MASS
        Matm=mass[n-2]
        # Set core mass to target planet mass - mass of atmosphere
        Mc=Mp_target-mass[n-2] # remove atm from mass of core, condensate retunrs to the core
        # Calculate total planet mass from core and atmosphere masses
        Mp = Mc + mass[n-2]
        
        
        # H2 MASS BALANCE
        mass_H2atm, mass_H2cond, Matm_Mp_final, Mcond_Mp_final, mH2atm_Mp, \
        mH2cond_Mp, massfracH2_atm, mass_H2_total, mass_H2_core, massfracH2_core =H2_mass_balance()
        
        print('massfrac H2 atm =',massfracH2_atm)
        print('massfrac H2 core after atmosphere extraction =',massfracH2_core)
            
        

    if rho_0_in == 0.0:
        rho_0=rho_0_mix(massfracH2_core,MW2,MW1,2.5,0.09)
        print('Auto calculation of silicate core rho triggered')
    else:
        rho_0=rho_0_in
    print('')
    print('CORE CALCULATION:')
    m4, r4, P4, rho4, Rcmb4, Tcore = solve_for_structure(Mc, cmf, cmf_mix, xrho, rho_0,T0,P_surface,structure)
    print('    rho_0 surface = %8.3f g/cm^3' %rho4[-1])

    R_s=r4[-1]

    # Report core result
    bulk_rho=Mc/((4/3)*pi*r4[-1]**3)
    print('    Mass of core in Earth units = %.5f ' %(Mc/Mearth))
    print(r'    Radius of core in Earth units = %.4f ' %(r4[-1]/R_earth_meters))
    print('    Radius of metal core in Earth units = %.3f ' %(Rcmb4/R_earth_meters))
    print('    Mass frac H2 core after atmosphere extraction = %.4f' %massfracH2_core)
    print('    Bulk core density = %.3f kg/m^3' %bulk_rho)

    # SETUP NEW ATMOSPHERE MODEL BASED ON CORE RESULT, MAINTAIN TARGET MASS OF PLANET
    R_s=r4[-1]

    # RADII and HEIGHTS
    # Bondi radius based on the radiation temperature and the mass of the planet core
    Cs=np.sqrt(gamma_gas*kb*Teq/m_main)
    RB=2.0*G*Mc/(Cs**2)

    # Reset minimum and maxium radii for atmosphere model
    RB_over_rmax_search=RB_over_rmax
    rmax=RB/RB_over_rmax_search
    rmin=R_s
    r_atm = gridding(R_s, rmax, int(nr), sharp)

    # 3. RECALCULATE THE ATMOSPHERE MODEL based on new core size
    print('')
    print('ATMOSPHERE:')
    T,P,density,mass,mass_cond,dPdr_save,MW_gas,MW_melt,m_molec,k_ratio,mass_H2_atm,mass_H2_cond,x_cond,xH2atmosphere,wt_frac_condensed,\
    k_rad_saved,k_conv_saved,k_cond_saved,k_nc_saved,gradT_saved,x_inhib,Sgaskg,Smeltkg,S_sys_kg,tau,Tint,i_rcb, Lint, delta_bl, ledoux_top_height_m=model(2, r_atm, Ra = Ra,T_contact_in = T_contact, Ps_in = Ps)



    # REPEAT TO ENSURE CONVERGENCE BETWEEN CORE AND ATMOSPHERE

    j_iterate   = 0
    nrepeat     = 7
    save_T_top  = np.zeros(nrepeat)

    # Keep history for the Ps two-cycle guard
    Ps_hist = []
    f_hist  = []

    # --- mixer ---
    def _mix(old, new, lam):
        return (1.0 - lam)*old + lam*new

    # --- ADDED: gentle adaptive mixing config for main loop ---
    lam_outer       = 0.02      # start gentler than before
    LAM_MIN         = 1.0e-8    # floor
    LAM_MAX         = 0.3      # cap (was 0.30; gentler now)
    GROW_F          = 1.03      # slow ramp-up if truly improving
    DECAY_F         = 0.50      # halve on trouble
    IMPROVE_TOL     = 1e-3      # ~0.1% improvement needed to count
    calm_streak     = 0         # counts consecutive good steps
    E_prev          = None      # previous composite error for comparison
    # --------------------------------------------------------

    # Seed "previous" values; these names assumed already exist before loop begins
    T_contact_prev    = T_contact
    T0_prev           = T0
    P_surface_prev    = P_surface
    R_s_prev          = R_s
    R_s_prevprev      = R_s
    T_top_prev        = None
    T_top_prevprev    = None
    # --- Seed for Patch 2/3: make sure a raw value exists on iter 0 ---
    T_contact_raw = T_contact

    while j_iterate < nrepeat:
        # Save radius of core from last iteration
        R_s_earth_old = R_s / R_earth_meters
        
        print('----------------------------------------')
        print('')
        print("ITERATE between core and atmosphere solns...step = ", j_iterate)
        print(f'  [lam_outer] {lam_outer:.4e}')
        print('')

        # ----- ATMOSPHERE (already computed in prior outer step) provides P_surface, T_contact -----

        # SETUP CORE CALCULATION USING OUTPUT FROM PREVIOUS ATMOSPHERE MODEL
        # include update to surface pressure if required

        # --- Boundary hand-off uses the RAW contact temperature (no blending) ---
        # Assumes T_contact_raw has just been set by step 1 (or seeded before the loop)
        T0        = T_contact_raw
        P_surface = P[0]  # base pressure from the (current) atmosphere pass

        # Use new composition for core to calculate rho_0 for silicate in core
        # Add mass of atmosphere to total mass of planet, Mp, for mass balance
        
            
        Ps_lo = None
        g_lo  = None
        Ps_hi = None
        g_hi  = None
        
        # FIND LOWER PRESSURE to accommodate too much atmosphere
        f_target = 2.2e-03  # target for minimum xH2_melt, f
        while massfracH2_core < f_target:
            massfracH2_core_old = massfracH2_core
            
            # SETUP NEW ATMOSPHERE MODEL BASED ON CORE RESULT, MAINTAIN TARGET MASS OF PLANET
            R_s=r4[-1]

            # RADII and HEIGHTS
            # Bondi radius based on the radiation temperature and the mass of the planet core
            Cs=np.sqrt(gamma_gas*kb*Teq/m_main)
            RB=2.0*G*Mc/(Cs**2)

            # Reset minimum and maxium radii for atmosphere model
            RB_over_rmax_search=RB_over_rmax
            rmax=RB/RB_over_rmax_search
            rmin=R_s
            r_atm = gridding(R_s, rmax, int(nr), sharp)

            # RECALCULATE THE OPTIMIZED ATMOSPHERE MODEL based on new core size, 2 in call indicates not a search
            # You already computed/mixed T0 above
            T_contact = T0          # << enforce continuity at the interface for this iteration
            P_surface = P[0]        # fine as-is

            print('\nATMOSPHERE:')
            T, P, density, mass, mass_cond, dPdr_save, MW_gas, MW_melt, m_molec, k_ratio, \
            mass_H2_atm, mass_H2_cond, x_cond, xH2atmosphere, wt_frac_condensed, \
            k_rad_saved, k_conv_saved, k_cond_saved, k_nc_saved, gradT_saved, \
            x_inhib, Sgaskg, Smeltkg, S_sys_kg, tau, Tint, i_rcb, Lint, delta_bl, ledoux_top_height_m = model(2, r_atm, Ra = Ra,T_contact_in = T_contact, Ps_in = Ps)
            
            
            # TOTAL PLANET MASS
            Matm = mass[n-2]
            # Set core mass to target planet mass - mass of atmosphere, condensate returns to core
            Mc   = Mp_target - mass[n-2]
            # Calculate total planet mass from core and atmosphere masses
            Mp   = Mc + mass[n-2]
            Matm_Mp_final = mass[n-2] / Mp
            print('Matm/Mp = ', Matm_Mp_final)
            
            # H2 MASS BALANCE
            mass_H2atm, mass_H2cond, Matm_Mp_final, Mcond_Mp_final, mH2atm_Mp, \
            mH2cond_Mp, massfracH2_atm, mass_H2_total, mass_H2_core, massfracH2_core = H2_mass_balance()
        
            print('massfrac H2 core after atmosphere extraction =', massfracH2_core)
            
            # --- BEGIN inner loop: enforce raw binodal + build predictor (do NOT overwrite raw) ---
            Tcrest = critical_temperature(Ps)
            print('')
            print('Searching for T contact at P = %.3f GPa' % Ps)

            # Clamp ONLY what we pass into find_contact_T (do not overwrite massfracH2_core)
            xH2_core_wt_pct_raw  = 100.0 * float(massfracH2_core)
            XH2_WT_PCT_FLOOR     = 1e-12   # safe tiny floor (wt%); bump to 1e-9 or 1e-6 if needed
            XH2_WT_PCT_CEIL      = 100.0   # physical upper bound

            xH2_core_wt_pct_safe = min(XH2_WT_PCT_CEIL, max(XH2_WT_PCT_FLOOR, xH2_core_wt_pct_raw))

            if xH2_core_wt_pct_safe != xH2_core_wt_pct_raw:
                print(f"[clamp] core H2 wt% {xH2_core_wt_pct_raw:.6e} -> {xH2_core_wt_pct_safe:.6e}")

            print('...and melt = %.3f H2 %%' % (xH2_core_wt_pct_safe))

            # 1) True binodal (raw)  [raw T logic; safe x input]
            T_contact_try = find_contact_T(Tcrest, xH2_core_wt_pct_safe, Ps)
            T_contact_raw = clamp_T_contact(T_contact_try)
            print(' Contact solvus temperature (raw) =', T_contact_raw)

            # 2) Smoothed predictor for controllers only (never used as boundary)
            if j_iterate > 0:
                T_contact_pred = _mix(T_contact_prev, T_contact_raw, lam_outer)
            else:
                T_contact_pred = T_contact_raw
            # --- END ---
            
            
            mH2_Mp = mass_H2atm / Mp
        
            # ------------------------------------------------------------
            # Single Ps update for next iteration (bracketed regula falsi + safe guards)
            # Target: massfracH2_core >= f_target
            # Define g(Ps) = massfracH2_core(Ps) - f_target
            # We want g >= 0. Negative g means "still too much atmosphere" → LOWER Ps.
            #
            # Requires persistent vars initialized OUTSIDE the while-loop:
            #   Ps_lo = None; g_lo = None   # store a POSITIVE-side point (g>0)
            #   Ps_hi = None; g_hi = None   # store a NEGATIVE-side point (g<0)
            # ------------------------------------------------------------
            f_new  = float(massfracH2_core)
            Ps_cur = float(Ps)

            g_new = f_new - float(f_target)   # <-- key change: target is f_target, not 0

            # --- update bracket endpoints (keep BEST/latest on each side of g=0) ---
            if g_new < 0.0:
                Ps_hi, g_hi = Ps_cur, g_new   # "too much atmosphere" side
            elif g_new > 0.0:
                Ps_lo, g_lo = Ps_cur, g_new   # "enough H2 in core" side

            # --- choose next Ps ---
            if g_new < 0.0:
                # Still below target → must reduce Ps
                if (Ps_lo is None) or (g_lo is None) or (not np.isfinite(Ps_lo)) or (not np.isfinite(g_lo)):
                    # no valid above-target bracket yet → geometric march down
                    Ps_next = 0.90 * Ps_cur   # 0.95 gentler; 0.85 faster
                    mode = "geom"
                else:
                    # bracketed → regula falsi (robust, but guard signs)
                    if not (g_hi < 0.0 and g_lo > 0.0):
                        Ps_next = 0.90 * Ps_cur
                        mode = "geom(bad_bracket)"
                    else:
                        denom = (g_hi - g_lo)
                        if (not np.isfinite(denom)) or (abs(denom) < 1e-30):
                            Ps_next = 0.90 * Ps_cur
                            mode = "geom(bad_denom)"
                        else:
                            # regula falsi root between Ps_lo (g>0) and Ps_hi (g<0)
                            Ps_next = (Ps_lo * g_hi - Ps_hi * g_lo) / denom
                            # clamp to bracket
                            Ps_next = max(min(Ps_next, Ps_hi), Ps_lo)
                            mode = "rf"
            else:
                # g_new >= 0 means we have met/exceeded the target; do not push Ps further down here
                Ps_next = Ps_cur
                mode = "hold" if (g_new > 0.0) else "done"

            # --- HARD INVARIANTS / SANITY GUARDS ---
            Ps_next = float(max(0.0, Ps_next))

            if g_new < 0.0:
                # never increase Ps while below target
                if Ps_next > Ps_cur:
                    print(f"  [Ps guard] preventing increase: Ps_next {Ps_next:.6f} > Ps_cur {Ps_cur:.6f}; forcing decrease")
                    Ps_next = 0.95 * Ps_cur  # or 0.90 for faster

                # minimum progress while below target (prevents tiny RF steps stalling you)
                min_drop_frac = 0.02  # try 0.02–0.05
                Ps_next = min(Ps_next, (1.0 - min_drop_frac) * Ps_cur)

            # final clamp again after guards
            Ps_next = float(max(0.0, Ps_next))

            print(f"  [Ps update:{mode}] Ps: {Ps_cur:.6f} -> {Ps_next:.6f} GPa  "
                  f"(f={f_new:.3e}, f_target={float(f_target):.3e}, g={g_new:.3e})")

            Ps = Ps_next
        # ---------------------------------------------------------------------------------------------

        # --- CORE SETUP (rho_0) ---
        rho_0_old = rho_0
        if rho_0_in == 0.0:
            rho_0 = rho_0_mix(massfracH2_core, MW2, MW1, 2.5,0.09)
        else:
            rho_0 = rho_0_in
        rho_0 = (rho_0 + rho_0_old) / 2

        print('')
        print('CORE:')
        
        # CORE CALCULATION
        P_surface = P[0]

        m4, r4, P4, rho4, Rcmb4, Tcore = solve_for_structure(
            Mc, cmf, cmf_mix, xrho, rho_0, T0, P_surface, structure,
            wtfracH2=massfracH2_core)
        print('    rho surface = %8.3f g/cm^3' % rho4[-1])
        print('    T surface = %.3f K' % T0)
        

        # Report next core solution
        bulk_rho = Mc / ((4/3)*pi*r4[-1]**3)
        print('    Mass of core in Earth units = %.5f ' % (Mc/Mearth))
        print(r'    Radius of core in Earth units = %.4f ' % (r4[-1]/R_earth_meters))
        print('    Mass frac H2 core after atmosphere extraction = %.4f' % massfracH2_core)
        print('    Bulk core density = %.3f kg/m^3' % bulk_rho)

        # SETUP NEW ATMOSPHERE MODEL BASED ON CORE RESULT, MAINTAIN TARGET MASS OF PLANET
        R_s=r4[-1]

        # RADII and HEIGHTS
        # Bondi radius based on the radiation temperature and the mass of the planet core
        Cs=np.sqrt(gamma_gas*kb*Teq/m_main)
        RB=2.0*G*Mc/(Cs**2)

        # Reset minimum and maxium radii for atmosphere model
        RB_over_rmax_search=RB_over_rmax
        rmax=RB/RB_over_rmax_search
        rmin=R_s
        r_atm = gridding(R_s, rmax, int(nr), sharp)


        # RECALCULATE THE ATMOSPHERE MODEL based on new core size, 2 in call indicates not a search
        # You already computed/mixed T0 above
        T_contact = T0          # << enforce continuity at the interface for this iteration
        P_surface = P[0]        #

        print('\nATMOSPHERE:')
        T, P, density, mass, mass_cond, dPdr_save, MW_gas, MW_melt, m_molec, k_ratio, \
        mass_H2_atm, mass_H2_cond, x_cond, xH2atmosphere, wt_frac_condensed, \
        k_rad_saved, k_conv_saved, k_cond_saved, k_nc_saved, gradT_saved, \
        x_inhib, Sgaskg, Smeltkg, S_sys_kg, tau, Tin, i_rcb, Lint, delta_bl, ledoux_top_height_m = model(2, r_atm, Ra = Ra,T_contact_in = T_contact, Ps_in = Ps)
        
        # SAVE ATMOSPHERE MASS
        Matm = mass[n-2]
        # UPDATE CORE MASS & PLANET MASS
        Mc   = Mp_target - mass[n-2]
        Mp   = Mc + mass[n-2]
        
        # H2 MASS BALANCE
        mass_H2atm, mass_H2cond, Matm_Mp_final, Mcond_Mp_final, mH2atm_Mp, \
        mH2cond_Mp, massfracH2_atm, mass_H2_total, mass_H2_core, massfracH2_core = H2_mass_balance()
        
        # --- BEGIN: compute raw binodal + predictor (do NOT overwrite raw) ---
        Tcrest = critical_temperature(Ps)
        print('')
        print('Searching for T contact at P = %.3f GPa' % Ps)
        print('...and melt = %.3f H2 %%' % (100*massfracH2_core))

        # 1) True binodal (raw) — used as the boundary given to the core
        T_contact_try = find_contact_T(Tcrest, (100*massfracH2_core), Ps)
        T_contact_raw = clamp_T_contact(T_contact_try)
        print(' Contact solvus temperature (raw) =', T_contact_raw)

        # 2) Smoothed predictor for controllers/diagnostics only (NEVER used as boundary)
        if j_iterate > 0:
            T_contact_pred = _mix(T_contact_prev, T_contact_raw, lam_outer)
        else:
            T_contact_pred = T_contact_raw
        # --- END: raw binodal ---

        # TEST FOR CONVERGENCE TO DESIRED THERMAL STATE
        T_top = T[n-2]
        save_T_top[j_iterate] = T_top
        
        # =========================
        # ADAPTIVE lam_outer UPDATE
        # =========================
        # Composite error measuring both radiation match and radius drift
        Tint_r = Tint * (r_atm[0] / r_atm[n-2])**0.5
        e_rad = abs(T_top - T_eddington(0.0, Tint_r, Teq))
        e_rad = max(e_rad, 1e-9)      # guard
        e_r   = abs(R_s/R_earth_meters - R_s_earth_old)
        E     = 0.7*e_rad + 0.3*e_r

        # Flip detection using last two steps (safe if histories exist)
        rs_flip = ( (R_s - R_s_prev) * (R_s_prev - R_s_prevprev) < 0.0 )
        t_flip  = False
        if (T_top_prev is not None) and (T_top_prevprev is not None):
            t_flip = ( (T_top - T_top_prev) * (T_top_prev - T_top_prevprev) < 0.0 )

        if rs_flip or t_flip:
            lam_outer   = max(DECAY_F * lam_outer, LAM_MIN)
            calm_streak = 0
            print(f'  [lam_outer ↓ flip] {lam_outer:.4e}')
        elif (E_prev is not None) and (E >= E_prev*(1 - IMPROVE_TOL)):
            # error not improving → tighten
            lam_outer   = max(DECAY_F * lam_outer, LAM_MIN)
            calm_streak = 0
            print(f'  [lam_outer ↓ worsen/flat] {lam_outer:.4e}')
        else:
            # error improved and no flip
            calm_streak += 1
            if calm_streak >= 3:
                lam_outer   = min(GROW_F * lam_outer, LAM_MAX)
                calm_streak = 0
                print(f'  [lam_outer ↑ improve] {lam_outer:.4e}')

        # Update error history
        E_prev = E

        # CRITERIA FOR EXIT DUE TO CONVERGENCE, with notifications (unchanged except for raw use)
        if j_iterate > 5: # do at least several iterations
        
            radius_accuracy = 1.0e-5
            Trad_accuracy   = 1.0
            print('\n')
            Tint_r = Tint * (r_atm[0] / r_atm[n-2])**0.5
            radtest = T_top - T_eddington(0.0, Tint_r, Teq)
            print('T_top - Trad = %10.3e' % radtest)
            if abs(radtest) < Trad_accuracy:
                print(' Success Trad...step ', j_iterate)
            print('Rc(j_iterate) - Rc(j_iterate-1) = %10.3e ' % (R_s/R_earth_meters - R_s_earth_old) )
            if abs((R_s/R_earth_meters) - R_s_earth_old) < radius_accuracy:
                print(' Success radius...step ', j_iterate)
            
            if abs(radtest) < Trad_accuracy:
                if abs((R_s/R_earth_meters) - R_s_earth_old) < radius_accuracy:
                    if massfracH2_core >= -0.0005:
                        # Right before break, snap and FREEZE final values (use RAW contact T)
                        R_s_final       = float(r4[-1])         # raw core radius at convergence (meters)
                        T_contact_final = float(T_contact_raw)  # RAW binodal contact T
                        T0_final        = float(T_contact_raw)  # keep consistency with boundary
                        P_surface_final = float(P[0])

                        # Prevent any subsequent accidental reuse/mutation
                        R_s = R_s_final
                        T0  = T0_final
                        P_surface = P_surface_final
                        break
            if j_iterate > 2:
                if (abs(save_T_top[j_iterate] - save_T_top[j_iterate-1]) <= Trad_accuracy and
                    abs(save_T_top[j_iterate - 1] - save_T_top[j_iterate-2]) <= Trad_accuracy and
                    abs(save_T_top[j_iterate - 2] - save_T_top[j_iterate-3]) <= Trad_accuracy):
                    if abs((R_s/R_earth_meters) - R_s_earth_old) < radius_accuracy:
                        if massfracH2_core >= -0.0005:
                            # Right before break, snap and FREEZE final values (use RAW contact T)
                            R_s_final       = float(r4[-1])         # raw core radius at convergence (meters)
                            T_contact_final = float(T_contact_raw)  # RAW binodal contact T
                            T0_final        = float(T_contact_raw)  # keep consistency with boundary
                            P_surface_final = float(P[0])

                            # Prevent any subsequent accidental reuse/mutation
                            R_s = R_s_final
                            T0  = T0_final
                            P_surface = P_surface_final
                            break

        # --- advance outer-path histories for next iteration ---
        R_s_prevprev         = R_s_prev
        R_s_prev             = R_s
        T_top_prevprev       = T_top_prev
        T_top_prev           = T_top
        T0_prev              = T0
        P_surface_prev       = P_surface
        T_contact_prev       = T_contact_pred     # blended predictor (used only for stability)
        T_contact_raw_prev   = T_contact_raw      # raw binodal for auditing
        # -------------------------------------------------------------

        j_iterate += 1

    print('Done!')

    # If the loop finished without triggering the break/convergence snap,
    # fall back to the *last raw* core geometry and boundary T.
    if 'R_s_final' not in locals():          # i.e., no convergence snap happened
        R_s = float(r4[-1])                  # last raw core radius
        T0  = float(T_contact_raw)           # last raw binodal contact T
        P_surface = float(P[0])              # last base pressure


    #____________________________________________PLANET MODEL COMPLETE ________________________________________________#

    P_surf_GPa=P_surface/1.0e9


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #_______________________________________ ANCILLARY PARAMETERS ________________________________________________#


    # Radiation temperature out the top of the atmosphere, Trad - Teq controls cooling
    Trad=T[n-2]

    # Create an adiabat from the surface for calculating the "cross-over" definition of the Rrcb
    T_adiabat=np.zeros(n)
    i_rcb_analytical=n-1
    T_adiabat[0]=T[0]
    for i in range(1,n):
        CpH2=Cp_H2()
        CpSiO=Cp_SiO()
        CpO2=Cp_O2()
        CpMg=Cp_Mg()
        Cp_gas=xH2atmosphere[i]*CpH2+(1.0-xH2atmosphere[i])*(0.333*CpO2+0.333*CpSiO+0.333*CpMg)
        k_conv=(Rgas/Cp_gas)*(m_molec[i]*g[i]/kb)
        T_adiabat[i]=(T_adiabat[i-1])-(k_conv)*(r_atm[i]-r_atm[i-1])
    for i in range(1,n):
        if T_adiabat[i] < Trad+1:
            i_rcb_analytical=i
            break
        
        
    # TOTAL PLANET MASS
    Matm=mass[n-2]
    # Set core mass to target planet mass - mass of atmosphere
    Mc=Mp_target - mass[n-2]
    # Calculate total planet mass from core and atmosphere masses
    Mp = Mc + mass[n-2]
    Matm_Mp_final=mass[n-2]/Mp
    print('Matm/Mp = ',Matm_Mp_final)
    
    # RADII and HEIGHTS
    # Bondi radius based on the radiation temperature and the mass of the planet core
    Cs=np.sqrt(gamma_gas*kb*Teq/m_main)
    RB=2.0*G*Mc/(Cs**2)

    # H2 MASS BALANCE
    mass_H2atm, mass_H2cond, Matm_Mp_final, Mcond_Mp_final, mH2atm_Mp, \
    mH2cond_Mp, massfracH2_atm, mass_H2_total, mass_H2_core, massfracH2_core =H2_mass_balance()
    massfracH2_cond=mass_H2cond/mass_cond[n-2] # Extra information about the condensate sent back to core

    print('massfrac H2 atm =',massfracH2_atm)
    print('massfrac H2 core after atmosphere extraction =',massfracH2_core)


    # Mass balance check for hydrogen, two values printed below should match
    massfrac_H2_total=(mass_H2_core + mass_H2atm )/Mp
    #print('massfrac H2 total =',massfrac_H2_total)
    #print('total H2 mass frac initial =',mass_H2_total/Mp)

    # Get numerical position of radiative-convective boundary as defined as equal contributions of each
    #i_rcb=n-1
    #rcb_limit=1.0 # limit to compare k_ratio to for finding index for rcb
    #for i in range(n):
    #    if k_ratio[i] > rcb_limit:
    #        i_rcb=i
    #        break

    # Fix  some arrays at endpoints for plotting
    T[n-1]=T[n-2]
    k_cond_saved[0]=k_cond_saved[1]
    k_ratio[0]=k_ratio[1]
    k_ratio[n-1]=k_ratio[n-2]
    gradT_saved[n-1]=gradT_saved[n-2]
    k_rad_saved[0]=k_rad_saved[1]
    k_rad_saved[n-1]=k_rad_saved[n-2]
    k_cond_saved[0]=k_cond_saved[1]
    k_cond_saved[n-1]=k_cond_saved[n-2]
    k_conv_saved[0]=k_conv_saved[1]
    k_conv_saved[n-1]=k_conv_saved[n-2]
    k_nc_saved[0]=k_nc_saved[1]
    k_nc_saved[n-1]=k_nc_saved[n-2]
    wt_frac_condensed[0]=wt_frac_condensed[1]
    xH2atmosphere[-1]=xH2atmosphere[-2]
    mass[-1]=mass[-2]

    #--------------------------------------------------------------------------------------
    # OPTICAL DEPTHS

    # ----------------------------------------------------------------------
    # OPTICAL DEPTH FROM TOP OF ATMOSPHERE DOWN TO LEVEL i
    # τ_i = ∫_{top}^{r_i} κ(T,P,Z) ρ dr  (integrated shell-by-shell)
    # my_kappa returns opacity in cm^2/g; the 0.1 converts to m^2/kg.
    # ----------------------------------------------------------------------
    tau_above = np.zeros(n)
    tau_above[-1] = 0.0  # by definition at the topmost cell

    for i in range(n-2, -1, -1):  # i = n-2, n-3, ..., 0
        # shell thickness (assumes r increases outward)
        dr = r_atm[i+1] - r_atm[i]
        if dr <= 0.0:
            dr = abs(dr)  # guard in case of reversed/duplicate radii

        # midpoint values for stability
        rho_mid  = 0.5 * (density[i] + density[i+1])
        T_mid    = 0.5 * (T[i] + T[i+1])
        P_mid    = 0.5 * (P[i] + P[i+1])
        xH2_mid  = 0.5 * (xH2atmosphere[i] + xH2atmosphere[i+1])
        Z_mid    = compute_metallicity(1.0 - xH2_mid)

        # opacity: cm^2/g -> m^2/kg via ×0.1
        kappa_m2kg = my_kappa(T_mid, P_mid, Z_mid) * 0.1

        # increment optical depth
        tau_above[i] = tau_above[i+1] + kappa_m2kg * rho_mid * dr

    # optional: smooth boundary artifact at i=0
    tau_above[0] = tau_above[1]



    # CHORD OPTICAL DEPTH (Guillot 2010, Eq. 56)
    # tau = 2 integral_b^inf kappa*rho*r/(sqrt(r^2-b^2)) dr where
    # ds/dr = r/sqrt(r^2 - b^2)
    # arrays increasing outward: r, density, T,
    n = len(r_atm)
    tau_chord = np.zeros(n)

    # shell thicknesses 
    dr_c = np.empty(n)
    dr_c[1:] = r_atm[1:] - r_atm[:-1]
    dr_c[0]  = dr_c[1]

    UNIT = 0.1  # Convert kappa to SI units, used below

    for i in range(0, n-1):                # tangent (impact) radius r_p = r_atm[i]
        rp = r_atm[i]

        # tangent-cell evaluated analytically (from z=0 to top half of cell)
        if i < n-1:
            dz0 = max(r_atm[i+1] - r_atm[i], 0.0)  # top half-thickness above tangent
            ds0 = np.sqrt(dz0*dz0 + 2.0*rp*dz0)   # integral of ds across tangent sliver
            Nr_i_tangent = 2.0 * density[i] * ds0
        else:
            Nr_i_tangent = 0.0

        # shells strictly above the tangent cell evaluated by Riemann sum (j = i+1 .. end)
        if i+1 < n:
            rj  = r_atm[i+1:]
            den = np.sqrt(np.maximum(rj*rj - rp*rp, 0.0))
            geom = rj / den                              # ds/dr
            Nr_i_above = 2.0 * np.sum(density[i+1:] * geom * dr_c[i+1:])
        else:
            Nr_i_above = 0.0

        Nr_i = Nr_i_tangent + Nr_i_above

        # Opacity: evaluate at tangent level (or switch to layer-dependent if needed)
        kappa_i = my_kappa(T[i], P[i], 0.0)
        tau_chord[i] = UNIT * Nr_i * kappa_i

    # Topmost has no column above
    tau_chord[-1] = 0.0
    tau_chord[n-1]=tau_chord[n-2]
    tau_chord[0]=tau_chord[1]

    i_chord=100 # DUMMY to prevent crashes during coding
    for i in range(n-1,0,-1):
        if tau_chord[i] > tau_critical:
            i_chord=i
            break
    r_chord=r_atm[i_chord]
    r_chord_earth_units=r_chord/R_earth_meters
    
    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    #___________________________________________ REPORT STRUCTURE ________________________________________________#

    # DIAGNOSTIC: Smooth out oscillations in pressure due to oscillations in MW near the surface for large dTcdP values
    def exponential_moving_average(arr, alpha_exp):
        """
        Smooth an array using an exponential moving average.
        
        Parameters:
            arr (numpy array): The input array to smo---+oth.
            alpha_exp (float): Smoothing factor between 0 and 1.
        
        Returns:
            numpy array: The smoothed array.
        """
        smoothed = np.zeros_like(arr)
        smoothed[0] = arr[0]
        for i in range(1, len(arr)):
            smoothed[i] = alpha_exp * arr[i] + (1 - alpha_exp) * smoothed[i - 1]
        return smoothed
    smoothed_P =exponential_moving_average(P,alpha_exp=0.3)


    # Save mass of planet in MEarth units
    Mp_Earth=Mp/Mearth

    # Volume of core
    vol_core=(4.0/3.0)*pi*(R_s**3)

    # Pressure due to weight of atmosphere
    P_s_atm=g[0]*Matm/(4*pi*r_atm[0]**2)


    #  CONSOLIDATE CORE AND ATMOSPHERE
    
    #dbg_r_samples(r_atm, tag="r_atm right before append r4, r_atm")
    
    def dbg_stitch_radii(r4, r_atm, k=8, label="", verbose=False):
        """
        Diagnostic stitch helper.
        Set verbose=False to suppress all prints while preserving return behavior.
        """

        def _p(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        r4   = np.asarray(r4, dtype=float)
        ratm = np.asarray(r_atm, dtype=float)

        _p("\n===================================================")
        _p(f"[DBG STITCH] {label}")
        _p("===================================================")

        # --- basic sanity ---
        _p(f"len(r4)   = {len(r4)}")
        _p(f"len(r_atm)= {len(ratm)}")

        if len(r4) < 3 or len(ratm) < 3:
            _p("!!! Too short to diagnose stitch robustly.")
            return

        # --- monotonicity checks ---
        def _mono_check(arr, name):
            d = np.diff(arr)
            bad = np.where(d <= 0)[0]
            if len(bad) > 0:
                i = bad[0]
                _p(f"!!! {name} NOT strictly increasing at i={i}: "
                   f"{name}[i]={arr[i]:.6e}, {name}[i+1]={arr[i+1]:.6e}, dr={d[i]:.3e}")
            else:
                _p(f"{name} monotone ✅  (min dr = {np.min(d):.3e} m, max dr = {np.max(d):.3e} m)")
            return d

        dr4   = _mono_check(r4,   "r4")
        dratm = _mono_check(ratm, "r_atm")

        # --- tail/head print ---
        kk4   = min(k, len(r4))
        kkatm = min(k, len(ratm))

        _p("\n--- r4 tail ---")
        for i in range(len(r4)-kk4, len(r4)):
            _p(f" r4[{i:6d}] = {r4[i]:.6e}")

        _p("\n--- r_atm head ---")
        for i in range(0, kkatm):
            _p(f" r_atm[{i:6d}] = {ratm[i]:.6e}")

        # --- join geometry ---
        r4_last   = r4[-1]
        ratm0     = ratm[0]
        gap       = ratm0 - r4_last

        _p("\n--- join metrics ---")
        _p(f" r4[-1]      = {r4_last:.6e} m")
        _p(f" r_atm[0]    = {ratm0:.6e} m")
        _p(f" gap (atm0 - core_last) = {gap:.6e} m")

        # Detect duplicate join
        if abs(gap) < 1e-6:
            _p("⚠️  Looks like r4[-1] and r_atm[0] are (nearly) identical → duplicate radius at stitch is likely.")
        elif gap < 0:
            _p("❌ r_atm[0] < r4[-1] → overlap/reversal at stitch (will create negative dr in r_planet).")
        else:
            _p("Join ordering OK ✅")

        # --- step-size jump at join ---
        dr4_last   = r4[-1] - r4[-2]
        dratm0     = ratm[1] - ratm[0]

        _p("\n--- step sizes near join ---")
        _p(f" dr4_last   = r4[-1]-r4[-2]     = {dr4_last:.3e} m")
        _p(f" dratm0     = r_atm[1]-r_atm[0] = {dratm0:.3e} m")

        if dr4_last > 0 and dratm0 > 0:
            ratio = dratm0 / dr4_last
            _p(f" dr ratio (atm/core) = {ratio:.3e}")
            if ratio > 1e3 or ratio < 1e-3:
                _p("⚠️  Huge step-size jump at stitch → expect kinks/zigzags in derived quantities vs r.")
        else:
            _p("❌ Non-positive dr near join — check array construction.")

        # --- build combined + interrogate neighborhood ---
        r_planet = np.append(r4, ratm)
        # stitch without duplicating R_s :
        # r_planet = np.concatenate([r4, r_atm[1:]])

        drp = np.diff(r_planet)
        j = len(r4) - 1   # last core index in combined

        _p("\n--- combined array stitch neighborhood ---")
        for ii in range(max(0, j-5), min(len(r_planet), j+6)):
            tag = ""
            if ii == j: tag = "<-- core last"
            if ii == j+1: tag = "<-- atm first"
            _p(f" r_planet[{ii:6d}] = {r_planet[ii]:.6e} {tag}")

        _p("\n--- dr around stitch in r_planet ---")
        for ii in range(max(0, j-5), min(len(drp), j+5)):
            tag = ""
            if ii == j: tag = "<-- dr across stitch (core_last→atm0)"
            _p(f" dr_planet[{ii:6d}] = {drp[ii]:.3e} {tag}")

        badp = np.where(drp <= 0)[0]
        if len(badp) > 0:
            i = badp[0]
            _p(f"\n❌ r_planet NOT monotone at i={i}: r[{i}]={r_planet[i]:.6e}, r[{i+1}]={r_planet[i+1]:.6e}, dr={drp[i]:.3e}")
        else:
            _p("\nr_planet monotone ✅")

        return r_planet
            

    #dbg_stitch_radii(r4, r_atm, k=10, label="before append")
    #r_planet = np.append(r4, r_atm)
    r_planet = dbg_stitch_radii(r4, r_atm, k=10, label="stitch check")
    
    T_planet = np.append(Tcore,T)
    rho_planet = np.append(rho4,density)
    P_planet = np.append(P4,P)
    P_planet_smoothed=np.append(P4,smoothed_P)

    # Find index for planet transit radius using new combined arrays
    #i_chord=0 # DUMMY to prevent crashes during coding
    i_chord_p=0
    for i in range(len(r_planet)):
        if r_planet[i] > r_chord:
            i_chord_p=i
            break
        else:
            i_chord_p=i
    print('')
    i_chord_length=i_chord_p/len(r_planet)
    print('')
    print('len(planet arrays) = %d' %len(r_planet))
    print('i_chord_p = %d' %i_chord_p)
    print('i_chord_p/len(arrays) = %.3e' %i_chord_length)
    print('Planet radius chord = %.3e' %(r_planet[i_chord_p]/R_earth_meters))

    # Find index for radial position for 1 bar pressure
    i_1bar_p=0
    for i in range(len(P_planet)):
        if P_planet[i] < 1.0e5:  # 1e5 Pa = 1 bar
            i_1bar_p=i
            break
        
    print('Planet radius at 1 bar =',(r_planet[i_1bar_p]/R_earth_meters))

    # Compute normalized moment of inertia, use 1 bar radius for MR integration
    kMI=normalized_moment_of_inertia(rho_planet, r_planet,i_1bar_p)


    # ENERGY

    # TOTAL ENERGY FOR CORE UP TO SURFACE, arrays are cumulative
    # dEdr_core(r,c,Mr,T,rho)
    E_core_gravPE=np.zeros(len(r4))
    E_core_thermal=np.zeros(len(r4))
    for i in range(1,len(r4)):
        if r4[i] < Rcmb4:
            Cv=3.0*Rgas
            c = Cv/0.0558 # MW of Fe, J/(mole K)*(mole/kg Fe)
        else:
            Cv=4.0*5*Rgas
            c = Cv/0.100
        Ttemp=Tcore[i]
        rhotemp=rho4[i]
        rlower=r_planet[i-1]
        rupper=r_planet[i+1]
        Esoln=quad(dEdr_core,rlower,rupper,args=(c,m4[i],Ttemp,rhotemp,2))
        dE=Esoln[0]/2
        E_core_gravPE[i]=E_core_gravPE[i-1]+dE
        Esoln=quad(dEdr_core,rlower,rupper,args=(c,m4[i],Ttemp,rhotemp,1))
        dE=Esoln[0]/2
        E_core_thermal[i]=E_core_thermal[i-1]+dE
        
    # TOTAL ENERGY FOR GAS PHASE up to transit radius, start with core energy values, values are cumulative
    n_atm = len(r_atm)
    E_gas_gravPE  = np.zeros(n_atm)
    E_gas_thermal = np.zeros(n_atm)

    E_gas_gravPE[0]  = E_core_gravPE[-1]
    E_gas_thermal[0] = E_core_thermal[-1]

    # stop at len(r_atm)-2 because we use i+1
    for i in range(1, len(r_atm) - 1):
        m       = m_molec[i]
        Ttemp   = T[i]
        rhotemp = density[i]

        rlower = r_atm[i-1]
        rupper = r_atm[i+1]

        Esoln = quad(dEdr_atm, rlower, rupper,
                     args=((Mc + mass[i]), m, Ttemp, gamma_gas, rhotemp, 2))
        dE = 0.5 * Esoln[0]
        E_gas_gravPE[i] = E_gas_gravPE[i-1] + dE

        Esoln = quad(dEdr_atm, rlower, rupper,
                     args=((Mc + mass[i]), m, Ttemp, gamma_gas, rhotemp, 1))
        dE = 0.5 * Esoln[0]
        E_gas_thermal[i] = E_gas_thermal[i-1] + dE

    # fill the last point (cheap and consistent)
    E_gas_gravPE[-1]  = E_gas_gravPE[-2]
    E_gas_thermal[-1] = E_gas_thermal[-2]
        

    ## Add to the core the conversion of chemical energy to thermal energy via latent heat of reaction for
    ## the solvus reaction.
    S_H2,H_H2,G_H2=H2_thermo(T0,P_surface)
    S_melt,H_melt,G_melt=melt_thermo(T0)
    dH_solvus=Matm*((xH2atmosphere[0])*H_H2/MWH2+(1-xH2atmosphere[0])*H_melt/MWMgSiO3 -xH2melt[0]*H_H2/MWMgSiO3 + (1-xH2melt[0])*H_melt/MWMgSiO3)

    # Consolidate energy into whole-planet arrays
    E_thermal_planet = np.append(E_core_thermal, E_gas_thermal)
    E_gravPE_planet = np.append(E_core_gravPE, E_gas_gravPE)
    E_total_planet = E_thermal_planet + E_gravPE_planet
    

    # Derived quantities:
    R_earth_meters = 6.371e6
    Mearth = 5.972e24
    pi = np.pi

    bulk_rho_core = Mc / ((4.0/3.0) * pi * r4[-1]**3)
    r_core_m = r4[-1]
    r_core_Earth = r_core_m / R_earth_meters

    # Mass-weighted temperature:
    Tavg = np.sum(T * P * r_atm**2) / np.sum(P * r_atm**2)

    # Atmosphere volumes/densities:
    rchord = r_chord_earth_units * R_earth_meters
    vol_atm_chord = (4.0/3.0)*pi*(rchord**3 - R_s**3)
    density_atm_chord = Matm / vol_atm_chord
    vol_atm_1bar = (4.0/3.0)*pi*(r_planet[i_1bar_p]**3 - R_s**3)
    density_atm_1bar = Matm / vol_atm_1bar

    # Planet volumes/fractions:
    vol_planet = (4.0/3.0)*pi*(rchord**3)
    vol_frac_core = vol_core / vol_planet
    vol_frac_atm  = vol_atm_chord / vol_planet
    bulk_rho_planet = bulk_rho_core*vol_frac_core + density_atm_chord*vol_frac_atm

    # Gas/cond mass fractions already computed:
    ratio = Matm_Mp_final
    massfrac_atm = ratio/(1.0 + ratio)
    ratio = Mcond_Mp_final
    massfrac_cond = ratio/(1.0 + ratio)
    massfracH2gas = (mH2atm_Mp)/(1.0 + mH2atm_Mp)

    # Escape:
    Cs = np.sqrt(gamma_gas*kb*Trad/m_main)
    lambda_surface = (G*Mp*m_molec[0]/R_s)/(kb*T[0])
    lambda_rcb = (G*Mp*m_molec[i_rcb]/r_atm[i_rcb])/(kb*T[i_rcb])
    
    #dbg_r_samples(r_atm, tag="r_atm right before PROFILES")
    
   
    # Build the profiles dict (only what you need downstream) to be loaded into results
    profiles = {
        "r_planet_m": np.asarray(r_planet),
        "rho_planet_kgm3": np.asarray(rho_planet),
        "r_atm": np.asarray(r_atm),
        "P_planet_Pa": np.asarray(P_planet),
        "P_planet_smoothed_Pa": np.asarray(P_planet_smoothed),
        "T_planet_K": np.asarray(T_planet),
        "core_radius_m": np.asarray(r4),
        "core_rho_kgm3": np.asarray(rho4),
        "core_T_K": np.asarray(Tcore),
        "core_P_Pa": np.asarray(P4),
        "atm_P_bar": np.asarray(P)/1.0e5,
        "atm_P_smoothed_bar": np.asarray(smoothed_P)/1.0e5,
        "atm_T_K": np.asarray(T),
        "atm_column_density": np.asarray(N),
        "atm_rho_kgm3": np.asarray(density),
        "tau_above": np.asarray(tau_above),
        "R_over_Rc": np.asarray(r_atm)/R_s,
        "m_molec": np.asarray(m_molec),  # if this is kg/mol already
        "xH2_atmosphere": np.asarray(xH2atmosphere),
        "mw_condensate_gmol": np.asarray(MW_melt)*1000.0,
        "tau_chord": np.asarray(tau_chord),
        "wt_frac_condensed": np.asarray(wt_frac_condensed),
        "x_cond": np.asarray(x_cond),
        "x_inhib": np.asarray(x_inhib),
        "dPdr": np.asarray(dPdr_save),
        "gradT": np.asarray(gradT_saved),
        "g_m_per_s2": np.asarray(g),
        "E_total_planet_J_series": np.asarray(E_total_planet),
        "E_gravPE_planet_J_series": np.asarray(E_gravPE_planet),
        "E_thermal_planet_J_series": np.asarray(E_thermal_planet),
    }

    # Optional: write files (reuses open_with_suffix; suffix/out_dir already set above)
    written = []
    if write_files:

        def dump(name, arr, fmt="%12.7e\n", transform=None):
            fp = open_with_suffix(name, "w")
            try:
                if transform is None:
                    for v in arr:
                        fp.write(fmt % v)
                else:
                    for v in arr:
                        fp.write(fmt % transform(v))
            finally:
                fp.close()

            # record the *actual* file path that was written
            base, ext = os.path.splitext(name)
            written.append(str(out_dir / f"{base}{suffix}{ext}"))

        # mirror previous file outputs (examples; complete as needed)
        dump('Planet_radii_meters.txt', r_planet)
        dump('Planet_rho_kgm3.txt', rho_planet)
        dump('Planet_pressure_Pa.txt', P_planet)
        dump('Planet_smoothed_pressure_Pa.txt', P_planet_smoothed)
        dump('Planet_TK.txt', T_planet)
        dump('Planet_core_radii.txt', r4)
        dump('Planet_core_rho_g_per_cm3.txt', rho4, transform=lambda x: x/1000.0)
        dump('Planet_core_T.txt', Tcore, fmt="%12.3f\n")
        dump('Planet_core_P_GPa.txt', P4, fmt="%12.3f\n", transform=lambda x: x/1.0e9)
        dump('Atmosphere_P_bar_Planet.txt', P[:-1], fmt="%11.6e\n", transform=lambda x: x/1.0e5)
        dump('Atmosphere_smoothed_P_bar_Planet.txt', smoothed_P[:-1], fmt="%11.6e\n", transform=lambda x: x/1.0e5)
        dump('Atmosphere_T_K_Planet.txt', T[:-1], fmt="%10.5f\n")
        dump('Atmosphere_column_density_from_top_Planet.txt', N[:-1], fmt="%10.5f\n")
        dump('Atmosphere_density_Planet.txt', density[:-1], fmt="%10.5e\n")
        dump('Atmosphere_radii_planet.txt', r_atm[:-1], fmt="%.5e\n")
        dump('tau_from_above_Planet.txt', tau_above[:-1], fmt="%.5e\n")
        dump('Atmosphere_R_over_RP_Planet.txt', (r_atm[:-1]/R_s), fmt="%10.5e\n")
        dump('Molec_wt_gas_Planet.txt', (m_molec[:-1]/nucleon_mass), fmt="%10.5e\n")
        dump('xH2atmosphere_Planet.txt', xH2atmosphere[:-1], fmt="%10.5e\n")
        dump('Molec_wt_condensate_Planet.txt', (MW_melt[:-1]*1000.0), fmt="%10.5e\n")
        dump('chord_tau_Planet.txt', tau_chord[:-1], fmt="%10.5e\n")
        dump('kappa_cgs_Planet.txt', [my_kappa(T[i], P[i], 0.0) for i in range(0, n-1)], fmt="%10.5e\n")
        dump('Condensed_mass_fraction_Planet.txt', wt_frac_condensed[:-1], fmt="%10.5e\n")
        dump('Condensed_mole_fraction_Planet.txt', x_cond[:-1], fmt="%10.5e\n")
        dump('dPdr_atmosphere.txt', dPdr_save[:-1], fmt="%10.5e\n")
        dump('gradT_atmosphere.txt', gradT_saved[:-1], fmt="%10.5e\n")
        dump('Atm_g.txt', g[:-1], fmt="%10.5e\n")
        dump('Total_E_planet.txt', E_total_planet, fmt="%10.5e\n")
        dump('E_gravPE_planet.txt', E_gravPE_planet, fmt="%10.5e\n")
        dump('E_thermal_planet.txt', E_thermal_planet, fmt="%10.5e\n")

    # Instance of class PlanetLABResults() above.
    results = PlanetLABResults(
        # ---- Core ----
        Mc=Mc,
        r_core_m=r_core_m,
        Rcmb_m=Rcmb4,
        vol_core_m3=vol_core,
        bulk_rho_core=bulk_rho_core,
        E_core_thermal_J=E_core_thermal[-1],
        E_core_gravPE_J=E_core_gravPE[-1],
        T_central_K=T_planet[0],
        # ---- Atmosphere ----
        T_contact_K=T_contact,
        P_surface_bar=P[0]/1.0e5,
        P_surface_GPa=P[0]/1.0e9,
        P_weight_bar=P_s_atm/1.0e5,
        Matm_kg=Matm,
        Mp_kg=Mp,
        Mp_Earth=Mp_Earth,
        Matm_over_Mp=Matm_Mp_final,
        massfrac_atm=massfrac_atm,
        massfrac_cond=massfrac_cond,
        massfracH2_atm=massfracH2_atm,
        massfracH2_cond=massfracH2_cond,
        massfracH2_core=massfracH2_core,
        massfracH2_total=massfrac_H2_total,
        massfracH2_gas=massfracH2gas,
        RB_km=RB/1000.0,
        RB_over_Rc=RB/R_s,
        i_rcb=i_rcb,
        i_chord=i_chord,
        i_chord_p=i_chord_p,
        tau_rcb=tau_above[i_rcb],
        rcb_height_km=(r_atm[i_rcb]-R_s)/1000.0,
        Rrcb_over_Rc=r_atm[i_rcb]/R_s,
        Trcb_K=T[i_rcb],
        P_rcb_bar=P[i_rcb]/1.0e5,
        Rrcb_cross_over=r_atm[i_rcb_analytical]/R_s,
        Teq_K=Teq,
        Trad_K=Trad,
        R1bar_over_Rc=r_planet[i_1bar_p]/R_s,
        T_mass_weighted_K=Tavg,
        T_surface_K=T[0],
        Tint=Tint,
        # ---- Planet ----
        r_chord_Earth=r_chord_earth_units,
        r_1bar_Earth=(r_planet[i_1bar_p]/R_earth_meters),
        r_core_Earth=r_core_Earth,
        density_atm_chord=density_atm_chord,
        vol_atm_chord_m3=vol_atm_chord,
        density_atm_1bar=density_atm_1bar,
        vol_atm_1bar_m3=vol_atm_1bar,
        vol_planet_m3=vol_planet,
        vol_frac_core=vol_frac_core,
        vol_frac_atm=vol_frac_atm,
        bulk_rho_planet=bulk_rho_planet,
        dH_solvus_J=dH_solvus,
        E_thermal_planet_J=E_thermal_planet[-1],
        E_gravPE_planet_J=E_gravPE_planet[-1],
        E_total_planet_J=E_total_planet[-1],
        kMI=kMI,
        # ---- other ----
        Cs_m_per_s=Cs,
        lambda_surface=lambda_surface,
        lambda_rcb=lambda_rcb,
        Lint=Lint,
        delta_bl = delta_bl,
        ledoux_top_height_m=ledoux_top_height_m,
        Ra=Ra,
        # ---- Profiles and files ----
        profiles=profiles,
        files_written=written if write_files else None,
        output_dir=str(out_dir)
    )

    # Optional: write a compact summary file
    if write_files:
        try:
            with open_with_suffix('Planet_SUMMARY.txt', 'w') as fp:
                fp.write(results.summary() if hasattr(results, 'summary') else str(results))
                fp.write('\n')
        except Exception:
            import traceback
            print("\n======= EXCEPTION SWALLOWED AT WRTIE FILES ================")
            traceback.print_exc()
            print("=========================================================\n")
            raise

    return results
    

from math import pi

def legacy_summary_text(res, *, use_color=False) -> str:
    """
    Build a legacy-style summary string using PlanetLABResults (res) and res.profiles.

    - Assumes profiles store atm_P_bar (bar), atm_T_K (K), r (m), etc.
    - Uses res.<scalars> whenever available.
    - Returns a single string (suitable for printing OR writing to a file).
    """
    prof = res.profiles or {}

    # ---------- Constants (only for unit conversions / derived values) ----------
    Mearth = 5.972e24
    R_earth_m = 6.371e6

    # ---------- Pull arrays from profiles (preferred) ----------
    # Atmosphere arrays
    r = prof.get("r_atm", None)                    # m
    T = prof.get("atm_T_K", None)              # K
    P_bar = prof.get("atm_P_bar", None)        # bar
    tau_above = prof.get("tau_above", None)    # dimensionless optical depth-from-top

    # Planet/core arrays (for some legacy derived calcs)
    r4 = prof.get("core_radius_m", None)       # m
    rho4 = prof.get("core_rho_kgm3", None)     # kg/m^3
    T_planet = prof.get("T_planet_K", None)    # K
    E_thermal_planet = prof.get("E_thermal_planet_J_series", None)
    E_gravPE_planet = prof.get("E_gravPE_planet_J_series", None)
    E_total_planet = prof.get("E_total_planet_J_series", None)

    # Planet profiles
    r_planet = prof.get("r_planet_m", None)
    P_planet_Pa = prof.get("P_planet_Pa", None)

    # ---------- Convert pressure to Pa where needed ----------
    P_Pa = None
    if P_bar is not None:
        P_Pa = np.asarray(P_bar) * 1.0e5
    elif P_planet_Pa is not None:
        # fallback for atm_P_bar
        P_Pa = np.asarray(P_planet_Pa)

    # ---------- Pull scalars from res ----------
    Mc = getattr(res, "Mc", None)
    Mp = getattr(res, "Mp_kg", None)
    Mp_Earth = getattr(res, "Mp_Earth", Mp / Mearth if Mp is not None else None)

    Rcmb_m = getattr(res, "Rcmb_m", None)
    r_core_m = getattr(res, "r_core_m", None)

    vol_core = getattr(res, "vol_core_m3", None)
    bulk_rho_core = getattr(res, "bulk_rho_core", None)

    massfracH2_core = getattr(res, "massfracH2_core", None)

    E_core_thermal = getattr(res, "E_core_thermal_J", None)
    E_core_gravPE = getattr(res, "E_core_gravPE_J", None)

    T_contact = getattr(res, "T_contact_K", None)
    P_weight_bar = getattr(res, "P_weight_bar", None)

    Matm = getattr(res, "Matm_kg", None)
    Matm_over_Mp = getattr(res, "Matm_over_Mp", None)
    massfrac_atm = getattr(res, "massfrac_atm", None)
    massfrac_cond = getattr(res, "massfrac_cond", None)

    massfracH2_atm = getattr(res, "massfracH2_atm", None)
    massfracH2_cond = getattr(res, "massfracH2_cond", None)
    massfracH2_total = getattr(res, "massfracH2_total", None)
    massfracH2_gas = getattr(res, "massfracH2_gas", None)

    RB_km = getattr(res, "RB_km", None)
    RB_over_Rc = getattr(res, "RB_over_Rc", None)

    i_rcb = getattr(res, "i_rcb", None)
    Trcb = getattr(res, "Trcb_K", None)
    P_rcb_bar = getattr(res, "P_rcb_bar", None)
    rcb_height_km = getattr(res, "rcb_height_km", None)
    Rrcb_over_Rc = getattr(res, "Rrcb_over_Rc", None)
    Rrcb_cross_over = getattr(res, "Rrcb_cross_over", None)
    Lint = getattr(res, "Lint", None)

    Teq = getattr(res, "Teq_K", None)
    Trad = getattr(res, "Trad_K", None)
    Tint = getattr(res, "Tint", None)
    
    Ra = getattr(res, "Ra", None)
    delta_bl = getattr(res, "delta_bl", None)

    r_chord_E = getattr(res, "r_chord_Earth", None)
    r_1bar_E = getattr(res, "r_1bar_Earth", None)
    r_core_E = getattr(res, "r_core_Earth", None)

    density_atm_chord = getattr(res, "density_atm_chord", None)
    vol_atm_chord = getattr(res, "vol_atm_chord_m3", None)
    density_atm_1bar = getattr(res, "density_atm_1bar", None)
    vol_atm_1bar = getattr(res, "vol_atm_1bar_m3", None)

    vol_frac_core = getattr(res, "vol_frac_core", None)
    vol_frac_atm = getattr(res, "vol_frac_atm", None)
    bulk_rho_planet = getattr(res, "bulk_rho_planet", None)

    dH_solvus = getattr(res, "dH_solvus_J", None)
    E_thermal_planet_last = getattr(res, "E_thermal_planet_J", None)
    E_gravPE_planet_last = getattr(res, "E_gravPE_planet_J", None)
    E_total_planet_last = getattr(res, "E_total_planet_J", None)
    kMI = getattr(res, "kMI", None)
    
    lambda_surface = getattr(res, "lambda_surface", None)
    lambda_rcb = getattr(res, "lambda_rcb", None)

    tau_rcb = getattr(res, "tau_rcb", None)

    # ---------- Derived / fallback computations ----------
    # Core radius: prefer res.r_core_m; else last of r4
    if r_core_m is None and r4 is not None:
        r_core_m = float(np.asarray(r4)[-1])

    # Core volume/density: prefer res.*; else derive
    if (vol_core is None) and (r_core_m is not None):
        vol_core = (4.0/3.0) * pi * r_core_m**3
    if (bulk_rho_core is None) and (Mc is not None) and (vol_core is not None) and vol_core > 0:
        bulk_rho_core = Mc / vol_core

    # Atmosphere base radius (legacy R_s): revised to Rcmb_m for atmosphere base
    R_s = Rcmb_m if Rcmb_m is not None else r_core_m

    # tau at rcb: prefer res.tau_rcb; else pull from tau_above array
    if tau_rcb is None and (tau_above is not None) and (i_rcb is not None):
        tau_rcb = float(np.asarray(tau_above)[int(i_rcb)])

    # Mass-weighted temperature legacy-style: using r, P_Pa, T
    Tavg = None
    if (r is not None) and (P_Pa is not None) and (T is not None):
        rr = np.asarray(r)
        PP = np.asarray(P_Pa)
        TT = np.asarray(T)
        # match lengths safely
        nmin = min(len(rr), len(PP), len(TT))
        rr, PP, TT = rr[:nmin], PP[:nmin], TT[:nmin]
        denom = np.sum(PP * rr**2)
        if denom != 0:
            Tavg = float(np.sum(TT * PP * rr**2) / denom)

    # ---------- Formatting helpers ----------
    def fnum(x, fmt):
        return fmt % x if x is not None else "NA"

    def fexp(x, fmt="%10.3e"):
        return (fmt % x) if x is not None else "NA"

    # optional coloring (kept off by default to avoid file pollution)
    BLUE = "\033[94m" if use_color else ""
    RESET = "\033[0m" if use_color else ""

    # ---------- Build text ----------
    lines = []
    lines.append("--------------------------------------------------------------------------")
    lines.append("")
    lines.append("CORE STRUCTURE:")
    lines.append(f" Mass of core in Earth units = {fnum(Mc/Mearth if Mc is not None else None, '%.5f')}")
    lines.append(f" Radius of core in Earth units = {fnum((r_core_m/R_earth_m) if r_core_m is not None else None, '%.5f')}")
    lines.append(f" Radius of metal core in Earth units = {fnum((Rcmb_m/R_earth_m) if Rcmb_m is not None else None, '%10.3f')}")
    lines.append(f" Mass frac H2 core after atmosphere extraction = {fnum(massfracH2_core, '%10.3f')}")
    lines.append(f" Volume of core = {fexp(vol_core)} m^3")
    lines.append(f" Bulk core density = {fnum(bulk_rho_core, '%10.3f')} kg/m^3")
    lines.append(f" Total thermal energy E(Joules) for core = {fexp(E_core_thermal, '%10.5e')}")
    lines.append(f" Total grav pot energy E(Joules) for core = {fexp(E_core_gravPE, '%10.5e')}")
    # central temperature: prefer res.T_central_K; else T_planet[0]
    T_central = getattr(res, "T_central_K", None)
    if T_central is None and T_planet is not None:
        T_central = float(np.asarray(T_planet)[0])
    lines.append(f" Central temperature = {fnum(T_central, '%.4f')} K")
    lines.append(" ")
    lines.append("ATMOSPHERE STRUCTURE:")
    lines.append(f" Surface temperature = {fnum(T_contact, '%10.3f')} K")
    lines.append(f" Patm at base from weight at fixed g[0] = {fexp(P_weight_bar, '%10.3e')} bar")
    # Surface pressure from arrays or res
    Psurf_bar = getattr(res, "P_surface_bar", None)
    Psurf_GPa = getattr(res, "P_surface_GPa", None)
    if Psurf_bar is None and P_bar is not None:
        Psurf_bar = float(np.asarray(P_bar)[0])
    lines.append(f" Surface pressure = {fexp(Psurf_bar, '%10.3e')} bar")
    lines.append(f"{BLUE} Surface pressure = {fexp(Psurf_GPa, '%10.3e')} GPa{RESET}")
    lines.append(f" Mass of atmosphere = {fexp(Matm, '%10.4e')} kg")
    lines.append(f" Mass of planet = {fexp(Mp, '%10.3e')} kg")
    lines.append(f" Mass of planet = {fnum(Mp_Earth, '%10.3f')} MEarth")
    lines.append(f" Matm/Mp = {fexp(Matm_over_Mp, '%10.4e')}")

    lines.append(f"{BLUE}   Gas mass as percent of planet  = {fnum((massfrac_atm*100.0) if massfrac_atm is not None else None, '%8.3f')} % {RESET}")
    lines.append(f"{BLUE}   Condensate mass as percent planet, all returned to planet core  = {fnum((massfrac_cond*100.0) if massfrac_cond is not None else None, '%10.4e')} % {RESET}")

    lines.append(f" Massfraction H2 comprising gas = {fnum(massfracH2_atm, '%7.4f')}")
    lines.append(f" Massfraction H2 comprising condensate = {fnum(massfracH2_cond, '%7.4f')}")
    lines.append(f" Massfraction H2 comprising core= {fnum(massfracH2_core, '%7.4f')}")
    lines.append(f" Massfraction H2 for whole planet = {fnum(massfracH2_total, '%7.4f')}")
    lines.append(f" Mass fraction of planet composed of H2 in atm = {fnum(massfracH2_gas, '%7.4f')}")

    # Bondi radius
    if RB_km is not None:
        lines.append(f" Bondi radius = {fexp(RB_km, '%10.4e')} km")
    else:
        lines.append(" Bondi radius = NA km")
    lines.append(f" Bondi radius/Rc = {fnum(RB_over_Rc, '%10.4f')}")

    # tau at RCB
    tau_for_print =  tau_rcb
    lines.append(f" Optical depth at Rrcb = {fexp(tau_for_print, '%10.2e')}")

    lines.append(f" Height of rcb = {fexp(rcb_height_km, '%10.3e')} km")
    lines.append(f" Rrcb/Rc = {fnum(Rrcb_over_Rc, '%.4f')}")
    lines.append(f" Trcb = {fnum(Trcb, '%10.3f')}")
    lines.append(f" Pressure at Rrcb = {fexp(P_rcb_bar, '%.4e')} bar")
    lines.append(f" Cross-over Rrcb = {fnum(Rrcb_cross_over, '%.4f')}")
    lines.append(f" Teq = {fnum(Teq, '%10.3f')} K")
    lines.append(f" Trad = {fnum(Trad, '%10.3f')} K")
    lines.append(f" Ra envelope = {fnum(Ra, '%.4e')}")
    lines.append(f" Boundary layer envelope = {fnum(delta_bl, '%.4e')} meters")
    lines.append(f" Ledoux layer top height = {fnum(getattr(res, 'ledoux_top_height_m', None), '%.4e')} meters")
    lines.append(f" Lint = {fnum(Lint, '%.4e')} W")

    # R(1 bar)/Rc is stored as scalar on res
    lines.append(f" R(1 bar)/Rc = {fnum(getattr(res, 'R1bar_over_Rc', None), '%8.3f')}")
    if Tavg is not None:
        lines.append(f" Mass-weighted temperature = {fnum(Tavg, '%10.1f')} K")
    if T is not None:
        lines.append(f"{BLUE} Surface temperature = {fnum(float(np.asarray(T)[0]), '%10.1f')} K{RESET}")

    lines.append("")
    lines.append("PLANET STRUCTURE:")
    lines.append(f" Mass of planet = {fnum(Mp_Earth, '%8.3f')} MEarth")
    lines.append(f" Planet radius in Earth radii using chord tau = {fnum(r_chord_E, '%10.3f')}")
    lines.append(f" Planet radius in Earth radii using R(1 bar) = {fnum(r_1bar_E, '%10.3f')}")
    lines.append(f" Planet core radius in Earth units = {fnum(r_core_E, '%10.3f')}")

    lines.append(f" Average density of atm using R_chord = {fnum(density_atm_chord, '%10.3f')} kg/m^3")
    lines.append(f" Volume of atm using R_chord = {fexp(vol_atm_chord, '%10.3e')} m^3")

    lines.append(f" Average density of atm using R(1 bar) = {fnum(density_atm_1bar, '%10.3f')} kg/m^3")
    lines.append(f" Volume of atm using R(1 bar) = {fexp(vol_atm_1bar, '%10.3e')} m^3")

    lines.append(f" Planet volume fraction comprising core = {fnum(vol_frac_core, '%10.3f')}")
    lines.append(f" Planet volume fraction comprising atm (chord) = {fnum(vol_frac_atm, '%10.3f')}")
    lines.append(f"{BLUE} Bulk density of planet (chord) = {fnum(bulk_rho_planet, '%10.3f')} kg/m^3{RESET}")

    lines.append(f" Latent heat to produce atmosphere = {fexp(dH_solvus, '%.3e')} Joules")
    lines.append(f" Total thermal energy E(Joules) for planet = {fexp(E_thermal_planet_last, '%10.5e')}")
    lines.append(f" Total grav pot energy E(Joules) for planet = {fexp(E_gravPE_planet_last, '%10.5e')}")
    lines.append(f" Total energy (Joules) = {fexp(E_total_planet_last, '%10.6e')}")
    lines.append(f" Normalized moment of inertia to 1bar, k = {fnum(kMI, '%8.3f')}")
    
    lines.append(f" Lambda rcb = {fnum(lambda_rcb, '%10.6f')}")

    return "\n".join(lines)

    
# --- useful for testing ----
if __name__ == "__main__":
        result = Planet_LAB()
        print(result.summary())

