# Simulation of atmospheric vertical wind plus settling
#
# Standard library
import os
import sys
import time
import re
import warnings
from math import sqrt, log as ln, log10 as log, isnan
from random import uniform, randint
from statistics import mean, stdev
from multiprocessing.pool import Pool
from multiprocessing import get_start_method, get_context

from pathlib import Path

# NumPy
import numpy as np
from math import pi

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

# PyQt5 and PyQtGraph
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

# Import the model code
from Planet_Lab_v33_fnc import Planet_LAB, legacy_summary_text

print('  ')
print('  ')
print ('  +-----------------------------------------+')
print ('  |               PLANET MODEL              |')
print ('  |                                         |')
print ('  |                                         |')
print ('  |         Binary MgSiO3-H2 system;        |')
print ('  |      Integrate upward from solvus       |')
print ('  |  that defines the magma ocean surface.  |')
print ('  |                                         |')
print ('  |         Self-consistent core EOS.       |')
print ('  | Includes harmonic mixed rho for 1-phase |')
print ('  |     mixed silicate-H2-Fe solutions.     |')
print ('  |                                         |')
print ('  |          (EdY February, 2026)           |')
print ('  |                                         |')
print ('  +-----------------------------------------+')
print('')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#-----------------------------------------------
# FRONT MATTER FOR GUI INPUT
# -----------------------------------------------
# SCIENTIFIC NOTATION version of QDoubleSpinBox()
# Regular expression to find floats. Match groups are the whole string, the
# whole coefficient, the decimal part of the coefficient, and the exponent
# part.



_float_re = re.compile(r'(([+-]?\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)')

def valid_float_string(string):
    match = _float_re.search(string)
    return match.groups()[0] == string if match else False


class FloatValidator(QtGui.QValidator):

    def validate(self, string, position):
        if valid_float_string(string):
            state = QtGui.QValidator.Acceptable
        elif string == "" or string[position-1] in 'e.-+':
            state=QtGui.QValidator.Intermediate
        else:
            state=QtGui.QValidator.Invalid
        return (state,string,position)

    def fixup(self, text):
        match = _float_re.search(text)
        return match.groups()[0] if match else ""


class ScientificDoubleSpinBox(QtWidgets.QDoubleSpinBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(-np.inf)
        self.setMaximum(np.inf)
        self.validator = FloatValidator()
        self.setDecimals(1000)
        self.setValue(np.inf)
        self.setSingleStep(0.01)

    def validate(self, text, position):
        return self.validator.validate(text, position)

    def fixup(self, text):
        return self.validator.fixup(text)

    def valueFromText(self, text):
        return float(text)

    def textFromValue(self, value):
        return format_float(value)

    def stepBy(self, steps):
        text = self.cleanText()
        groups = _float_re.search(text).groups()
        decimal = float(groups[1])
        decimal += steps
        new_string = "{:g}".format(decimal) + (groups[3] if groups[3] else "")
        self.lineEdit().setText(new_string)


def format_float(value):
    """Modified form of the 'g' format specifier."""
    string = "{:g}".format(value).replace("e+", "e")
    string = re.sub("e(-?)0*(\d+)", r"e\1\2", string)
    return string
# END scientific notation version of QDoubleSpinBox()

#____________________________ INPUTS _____________________________#

# BEGIN GUI interface itself
## Always start by initializing Qt (only once per application)
app = QtWidgets.QApplication([])

## Define a top-level widget to hold everything
w = QtWidgets.QWidget()
w.setWindowTitle('MAGMA OCEAN PLANET ')

# CREATE WIDGETS
label1 = QtWidgets.QLabel('ATMOSPHERE:')

label2 = QtWidgets.QLabel('Bulk planet wt% H2')
box2 = QtWidgets.QDoubleSpinBox()
box2.setMinimum(0.0)
box2.setSingleStep(0.1)
box2.setMaximum(100.00)
box2.setSuffix("  %")
box2.setValue(3.0)

label4 = QtWidgets.QLabel('Mass of planet')
box4 = QtWidgets.QDoubleSpinBox()
box4.setSingleStep(0.01)
box4.setMinimum(0.01)
box4.setMaximum(20.0)
box4.setSuffix(" MEarth")
box4.setValue(6.0)

label17 = QtWidgets.QLabel('Radius Carrier Molec')
box17 = QtWidgets.QDoubleSpinBox()
box17.setMinimum(0.000)
box17.setMaximum(9000.0)
box17.setSuffix(" pm")
box17.setValue(289.0)

label8 = QtWidgets.QLabel('Molec wt main gas')
box8 = QtWidgets.QDoubleSpinBox()
box8.setMinimum(1.0)
box8.setMaximum(500.00)
box8.setSingleStep(0.1)
box8.setSuffix("  amu")
box8.setValue(2.3)

label27 = QtWidgets.QLabel('Trad - Teq')
box27 = QtWidgets.QDoubleSpinBox()
box27.setDecimals(5)
box27.setMinimum(0.00)
box27.setMaximum(20000.00)
box27.setSingleStep(0.1)
box27.setSuffix("  K")
box27.setValue(1.0)

label9 = QtWidgets.QLabel('Teq')
box9 = QtWidgets.QDoubleSpinBox()
box9.setDecimals(5)
box9.setMinimum(0.0000)
box9.setMaximum(20000.00)
box9.setSingleStep(1.0)
box9.setSuffix("  K")
box9.setValue(1000.0)

label10 = QtWidgets.QLabel('Molec wt silicate')
box10 = QtWidgets.QDoubleSpinBox()
box10.setMinimum(1.0)
box10.setMaximum(500.00)
box10.setSingleStep(0.1)
box10.setSuffix("  amu")
box10.setValue(100.0)

label11 = QtWidgets.QLabel('Tau critical (chord)')
box11 = QtWidgets.QDoubleSpinBox()
box11.setDecimals(5)
box11.setMinimum(0.01)
box11.setMaximum(500.0)
box11.setSingleStep(0.1)
box11.setValue(1.0)
box11.setSuffix( " ")

label19 = QtWidgets.QLabel('Surface P')
box19 = QtWidgets.QDoubleSpinBox()
box19.setMinimum(0.0)
box19.setMaximum(500.0)
box19.setSingleStep(0.05)
box19.setValue(3.7)
box19.setSuffix( "  GPa")

label21 = QtWidgets.QLabel('Gamma gas')
box21 = QtWidgets.QDoubleSpinBox()
box21.setSingleStep(0.010)
box21.setMinimum(0.0001)
box21.setMaximum(100.000)
box21.setSuffix(" = Cp/Cv")
box21.setValue(1.4)

label22 = QtWidgets.QLabel('Number of Radial Steps')
box22 = QtWidgets.QDoubleSpinBox()
box22.setDecimals(0)
box22.setMinimum(0)
box22.setMaximum(1000000000)
box22.setSingleStep(1)
box22.setValue(120000)

label26 = QtWidgets.QLabel('RB/rmax')
box26 = QtWidgets.QDoubleSpinBox()
box26.setDecimals(3)
box26.setMinimum(1.000)
box26.setMaximum(500.000)
box26.setSingleStep(1)
box26.setValue(13.00)

label23 = QtWidgets.QLabel('alpha smooth')
box23 = QtWidgets.QDoubleSpinBox()
box23.setMinimum(0.0000)
box23.setMaximum(1.0)
box23.setSingleStep(0.05)
box23.setValue(1.00)
box23.setSuffix(" ")

label24 = QtWidgets.QLabel('Initial search P at RB ')
box24 = ScientificDoubleSpinBox()
box24.setMinimum(1.0e-21)
#box24.setMaximum(1000.0)
#box24.setSingleStep(0.1)
box24.setValue(1.0e-19)
box24.setSuffix(" bar")

label25 = QtWidgets.QLabel('Inhibit conv ON (OFF) ')
box25 = QtWidgets.QLineEdit()
box25.setText("ON")


label34 = QtWidgets.QLabel('rho(0) silicate (0.0 = auto)')
box34 = QtWidgets.QDoubleSpinBox()
box34.setMinimum(0.00)
box34.setMaximum(10.00)
box34.setSingleStep(0.1)
box34.setSuffix("  g/cc")
box34.setValue(0.0)

label35 = QtWidgets.QLabel('Mass deficit metal')
box35 = QtWidgets.QDoubleSpinBox()
box35.setDecimals(3)
box35.setMinimum(0.0)
box35.setMaximum(1.0)
box35.setSingleStep(0.01)
box35.setSuffix("  ")
box35.setValue(0.01)

label30 = QtWidgets.QLabel('CORE:')

label36 = QtWidgets.QLabel('1=silicate, 2=metal, 3=silicate+metal')
box36 = QtWidgets.QDoubleSpinBox()
box36.setDecimals(0)
box36.setSingleStep(1)
box36.setSuffix("  ")
box36.setValue(1)

label37 = QtWidgets.QLabel('Mass fraction metal core')
box37 = QtWidgets.QDoubleSpinBox()
box37.setDecimals(3)
box37.setMinimum(0.0)
box37.setMaximum(1.0)
box37.setSingleStep(0.01)
box37.setSuffix("  ")
box37.setValue(0.33)

label40 = QtWidgets.QLabel('Suffix for output files ')
box40 = QtWidgets.QLineEdit()
box40.setText("_default")

label41 = QtWidgets.QLabel('Plot radius limit (0.0 = auto)')
box41 = QtWidgets.QDoubleSpinBox()
box41.setDecimals(3)
box41.setMinimum(0.0)
box41.setMaximum(20.0)
box41.setSingleStep(0.1)
box41.setSuffix(" REarth ")
box41.setValue(4.0)

label42 = QtWidgets.QLabel('Ra gas')
box42 = ScientificDoubleSpinBox()
box42.setMinimum(1.0)
box42.setMaximum(1.0e20)
box42.setValue(1.0e12)

label43 = QtWidgets.QLabel('dr logist slope')
box43 = QtWidgets.QDoubleSpinBox()
box43.setDecimals(0)
box43.setMinimum(1)
box43.setMaximum(20)
box43.setSingleStep(1.0)
box43.setValue(14)


b1 = QtWidgets.QPushButton("Enter")
b1.setStyleSheet('background-color: grey; color: white')

## Create a grid layout to manage the widgets size and position
layout = QtWidgets.QGridLayout()
w.setLayout(layout)
layout.setVerticalSpacing(7)


## Add widgets to the layout in their proper positions
layout.addWidget(label1,0,0)
layout.addWidget(label2, 1, 0)   # Column 1
layout.addWidget(box2, 2, 0)   #
layout.addWidget(label9, 3, 0)  #
layout.addWidget(box9, 4, 0)  #
layout.addWidget(label27, 5, 0)
layout.addWidget(box27, 6, 0)
layout.addWidget(label11, 7, 0)
layout.addWidget(box11, 8, 0)
layout.addWidget(label42, 9, 0)
layout.addWidget(box42, 10, 0)

layout.addWidget(label8, 1, 1)  # Column 2
layout.addWidget(box8, 2, 1)  #
layout.addWidget(label21,3,1)
layout.addWidget(box21,4,1)
layout.addWidget(label17, 5, 1)
layout.addWidget(box17, 6, 1)
layout.addWidget(label24, 7, 1)
layout.addWidget(box24, 8, 1)
layout.addWidget(label23,9,1)
layout.addWidget(box23,10,1) # smoothing factor for gradT
layout.addWidget(label43,11,1)
layout.addWidget(box43,12,1)

layout.addWidget(label10, 1, 2) # heavy molecule weight
layout.addWidget(box10, 2, 2)
layout.addWidget(label22,3,2) # radial steps
layout.addWidget(box22,4,2) # radial steps
layout.addWidget(label26,5,2) # RB/rmax
layout.addWidget(box26,6,2) # RB/rmax
layout.addWidget(label19,7,2) # Target pressure
layout.addWidget(box19,8,2) # Target pressure
layout.addWidget(label25,9,2)
layout.addWidget(box25,10,2)
layout.addWidget(label41,11,2)
layout.addWidget(box41,12,2) # Maximum radius for plot

# Core inputs, column 3
layout.addWidget (label30,0,3)
layout.addWidget(label4, 1, 3)  #
layout.addWidget(box4, 2, 3)  #
layout.addWidget(label34, 3, 3)
layout.addWidget(box34, 4, 3)
layout.addWidget(label35,5,3)
layout.addWidget(box35,6,3)
layout.addWidget(label36,7,3)
layout.addWidget(box36,8,3)
layout.addWidget(label37,9,3)
layout.addWidget(box37,10,3)
layout.addWidget(label40,11,3)
layout.addWidget(box40,12,3) # suffix



layout.addWidget(b1, 0,4) # Enter button third column

b1.clicked.connect(QtCore.QCoreApplication.instance().quit)

## Display the widget as a new window
w.show()

## Start the Qt event loop
app.exec_()

print(' ')
print(' ')

bulk = box2.value()
print("Entered H2 wt % = ",bulk,' %')

nr=box22.value()
print('Entered number of radial steps = ',nr)

RB_over_rmax=box26.value()
print('Entered Rb/rmax = ',RB_over_rmax)

Mp_target_earth = box4.value()
Mp_target=Mp_target_earth*5.972e24
print("Entered Mass of planet = %8.4f Earth masses" %Mp_target_earth)

m_main_amu_in=box8.value()
print("Entered mass of carrier = ",m_main_amu_in,' amu')

m_heavy_amu_in=box10.value()
print("Entered total molec wt silicate = ",m_heavy_amu_in,' amu')

r_main=box17.value()
r_main=r_main*1.0e-12
print('Entered radius carrier molecule = ',r_main,' meters')

gamma_gas=box21.value()
print('Entered gamma for main gas = ', gamma_gas)

DT=box27.value()
print('Entered Trad - Teq = %10.4f K' %DT)

Teq=box9.value()
print("Entered Teq = %10.4f K" %Teq)

target=box19.value()
print('Target base pressure = ', target,' GPa')

Ra_gas = box42.value()

tau_critical=box11.value()
print('Entered critical optical depth from top for chord = ', tau_critical)

alpha_smooth_gradT=box23.value()
print("alpha_smooth_gradT = %.3f K" %alpha_smooth_gradT)

Pnebula_bar=box24.value()
Pnebula=Pnebula_bar*1.0e5
print('Entered Pressure at top = %10.3e bar' %Pnebula_bar)

sharp = box43.value()

tempstring=box25.text()
tempstring
if tempstring == "OFF":
#Inhibit=bool(tempstring)
    Inhibit = False
else:
    Inhibit=True
print('Inhibiting convection is ',Inhibit)

suffix=box40.text()
print('Suffix for output files is ',suffix)

rad_lim=box41.value()

rho_0_in=box34.value()
#print('Entered rho(P=0) silicate melt = %.3f g/cc' % rho_0_in)

xrho = box35.value()
#print("Fractional mass deficit for metal = %8.4f" %xrho)

layering = box36.value()
if layering == 1:
    structure=[1]
    #print('Selected pure silicate planet')
elif layering == 2:
    structure=[0]
    #print('Selected pure Fe planet')
else:
    structure=[0,1]
    #print('Selected silicate with metal core')
    
cmf = box37.value()
cmf_mix = cmf

#----------------------- Function to add a suffix to all output files ---------------------------------------
def open_with_suffix(file_name, mode, suffix="_modified"):
    """
    Opens a file after placing it in a folder named after the suffix.
    The file name will have the suffix added before its extension.

    Parameters:
        file_name (str): Original file name.
        mode (str): Mode to open the file ('r', 'w', etc.).
        suffix (str): Suffix used both for the folder name and file name modification.

    Returns:
        file object: Opened file object.
    """
    # Split the file name into base name and extension
    base_name, ext = os.path.splitext(file_name)
    
    # Create folder with the suffix as its name (if not exists)
    folder_name = suffix.strip("_")
    os.makedirs(folder_name, exist_ok=True)
    
    # Add the suffix to the file name and place it in the folder
    new_file_name = f"{base_name}{suffix}{ext}"
    new_file_path = os.path.join(folder_name, new_file_name)
    
    # Open the file with the new name inside the folder
    return open(new_file_path, mode)



# END GUI INPUT VALUES
#-----------------------------------------------------------------------------------------------------------

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

# MODIFY UNITS OF INPUT MOLECULAR WEIGHTS
mw=m_main_amu_in/1000.0 # kg per mole
m_main=m_main_amu_in*nucleon_mass # kg per single molecule of main gas, H2
mw_h=m_heavy_amu_in/1000.0
m_heavy=m_heavy_amu_in*nucleon_mass # kg per single molecule of the second species, usually MgSiO3 in this app

# ADDITIONAL MOLECULAR WEIGHT TERMS USED BY SEVERAL FUNCTIONS, mw, mw_h = input converted to kg/mole
MWH2=mw # kg/mole
MWMgSiO3=mw_h # kg/mole
MW1=mw*1000.0 # amu for light main gas
MW2=mw_h*1000.0 # amu for heavy

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
# ------------------------------------------ RUN MODEL ---------------------------------------------------------

res = Planet_LAB(
        bulk_wt_H2_percent=bulk,
        Mp_target_kg=Mp_target_earth * 5.972e24,
        cmf_core_metal=cmf,
        n_radial_steps=nr,
        RB_over_rmax=RB_over_rmax,
        plot_radius_limit_RE=rad_lim,
        surface_P_GPa=target,
        Teq_K=Teq,
        mw_main_amu=m_main_amu_in,
        mw_silicate_amu=m_heavy_amu_in,
        gamma_gas=gamma_gas,
        r_main_m=r_main,
        rho0_silicate_gcc=rho_0_in,
        metal_mass_deficit=xrho,
        structure_layers=structure,
        DeltaTrad_K=DT,
        Ra = Ra_gas,
        tau_critical_chord=tau_critical,
        alpha_smooth_gradT=alpha_smooth_gradT,
        Inhibit=Inhibit,
        output_suffix=suffix,
        write_files=True,
        output_dir=".",
    )
if res is None:
    raise RuntimeError("Planet_LAB returned None — upstream exception was caught. Search upward in the terminal output for the original traceback.")

# --- Unpack results from Planet_LAB() so the legacy plotting code below can run unchanged ---
profiles = res.profiles or {}  # If res.profiles exists as a dictionary, and is not empty, use it

# --- Scalars from results object ---
Mp_Earth = res.Mp_Earth
Mp       = res.Mp_kg

Rcmb4    = res.Rcmb_m          # metal core radius (legacy name)
R_s      = res.r_core_m        # atmosphere base radius (legacy name Rc)

Mc       = res.Mc
vol_core = res.vol_core_m3
bulk_rho_core = res.bulk_rho_core
massfracH2_core = res.massfracH2_core

Ra_envelope = res.Ra
delta_boundary_layer = res.delta_bl

T_contact = res.T_contact_K
Matm      = res.Matm_kg
Matm_Mp_final = res.Matm_over_Mp

massfrac_atm = res.massfrac_atm
massfrac_cond = res.massfrac_cond

massfracH2_atm    = res.massfracH2_atm
massfracH2_cond   = res.massfracH2_cond
massfrac_H2_total = res.massfracH2_total
massfracH2gas     = res.massfracH2_gas

RB   = res.RB_km * 1000.0
Teq  = res.Teq_K
Trad = res.Trad_K

i_rcb     = res.i_rcb
i_chord   = res.i_chord
i_chord_p = res.i_chord_p
r_chord_earth_units = res.r_chord_Earth
Tint      = res.Tint

tau_above_rcb = res.tau_rcb
Lint_eff_W    = res.Lint

# --- Planet/core profiles ---
r_planet = profiles.get("r_planet_m")
T_planet = profiles.get("T_planet_K")
P_planet = profiles.get("P_planet_Pa")
rho_planet = profiles.get("rho_planet_kgm3")
E_total_planet = profiles.get("E_total_planet_J_series")
E_gravPE_planet = profiles.get("E_gravPE_planet_J_series")
E_thermal_planet = profiles.get("E_thermal_planet_J_series")

# Core arrays (if you still want them)
r4   = profiles.get("core_radius_m")
rho4 = profiles.get("core_rho_kgm3")
Tcore= profiles.get("core_T_K")
P4   = profiles.get("core_P_Pa")

# --- Atmosphere arrays (as stored) ---
r       = profiles.get("r_atm")                       # meters
T       = profiles.get("atm_T_K")                 # K
P_bar   = profiles.get("atm_P_bar")               # bar
density = profiles.get("atm_rho_kgm3")            # kg/m3
tau_above = profiles.get("tau_above")             # dimensionless
x_inhib = profiles.get("x_inhib")

gradT_saved = profiles.get("gradT")
dPdr_save   = profiles.get("dPdr")
g           = profiles.get("g_m_per_s2")

xH2atmosphere     = profiles.get("xH2_atmosphere")
wt_frac_condensed = profiles.get("wt_frac_condensed")
x_cond            = profiles.get("x_cond")
tau_chord         = profiles.get("tau_chord")
m_molec           = profiles.get("m_molec")

# Convenience if legacy code expects P in Pa:
P = P_bar * 1.0e5

# Output folder name used by writer
folder_name = str(suffix).strip('_')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#___________________________________________ REPORT STRUCTURE ________________________________________________


# Execute the summary text function within the Planet_LAB source file
# fp.write(legacy_summary_text(res)) # of sending to some file
print(legacy_summary_text(res))




#-----------------------------------------------------------------------------------
#  COMPARISONS AGAINST ANALYTICAL SOLNS

# Radiative diffusion from Teq at RB downward using derivation for T(tau) from Eddington's equation
# --- Analytic grey (irradiated) radiative layer temperature profile ---
n = int(nr)
T_tau = np.full(n, np.nan)

tau_above = res.profiles["tau_above"]
tau_above_rcb = res.tau_rcb      # scalar: tau_above at i_rcb
Tint          = res.Tint
i_rcb         = res.i_rcb

# Define the "radiative layer" as everything above the RCB (shallower than it):
rad_mask = (tau_above <= tau_above_rcb)

# Simple grey atmosphere (one common form; consistent with using Tint and Teq)
# T^4 = (3/4) Tint^4 (tau + 2/3) + (3/4) Teq^4
T_tau[rad_mask] = ((3.0/4.0) * Tint**4 * (tau_above[rad_mask] + 2.0/3.0)
                   + (3.0/4.0) * Teq**4) ** 0.25

# Optional: enforce continuity at the RCB by setting the RCB cell explicitly
T_tau[i_rcb] = res.Trcb_K
    


# Scaling for convective layer T after Ginzburg+ (2016, eq 5)
i_rcb = res.i_rcb
Teq = res.Teq_K
RB = res.RB_km * 1000.0        # meters
RBprime = (1.0 - 1.0/gamma_gas) * RB
# Using Trcb rather than Teq here adjusts the scale in Temperature
T_Ginzburg = res.Trcb_K * (1.0 + RBprime/r - RBprime/r[i_rcb]) #vectorized for r





# Radiation temperature out the top of the atmosphere
Trad_temp=T[n-2]

# RADII and HEIGHTS
# Bondi radius based on the radiation temperature and the mass of the planet core
Cs=np.sqrt(gamma_gas*kb*Teq/m_main)
RB=2.0*G*Mc/(Cs**2)

# Create an adiabat from the surface for calculating the "cross-over" definition of the Rrcb
T_adiabat=np.zeros(n)
i_rcb_analytical=n-1
T_adiabat[0]=T[0]
for i in range(1,n):
    Cp_gas=gamma_gas * Rgas / (gamma_gas - 1.0)
    k_conv=(Rgas/Cp_gas)*(m_molec[i]*g[i]/kb)
    T_adiabat[i]=(T_adiabat[i-1])-(k_conv)*(r[i]-r[i-1])
for i in range(1,n):
    if T_adiabat[i] < Trad_temp+1:
        i_rcb_analytical=i
        break
    

def dbg_r_samples(r_atm, tag=""):
    import numpy as np
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
  
#-----------------------------------------------------------------------------------
# MAKE PLOTS

# Assign path from Planet_LAB
out_dir = Path(res.output_dir)


heightkm=np.zeros(n)
for i in range(0,n):
    heightkm[i]=(r[i]-R_s)/1000.0

ymax=1.5*r_planet[i_chord_p]/R_s
ymin=1.0

planet_radius=np.zeros(n)
for i in range(n):
    planet_radius[i]=R_s/R_earth_meters


print("len(r4)      =", len(r4))
print("len(r_atm)   =", len(r))
print("len(r_planet)=", len(r_planet))

dbg_r_samples(r, tag="before PLOTTING call")


# Plot atmosphere pressure profile
plt.figure(3)
plt.plot(P/1.0e5,r/R_s,color='grey',label='Calculated')
#plt.plot(smoothed_P/1.0e5,r/R_s,color='black',label='smoothed')
#plt.plot(((density*Rgas*T/mean_mol_wt_gas)/1.0e-5),r/R_s,color='blue',label='Posterior calc')
plt.xlabel("Pressure (bar))")
plt.ylabel("R/Rc")
plt.ylim((1.0,ymax))
plt.legend()
plotname = 'Atm_P_profile_planetv4' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot atmosphere density profile
plt.figure(4)
plt.plot(density,r/R_s,label='Model')
#plt.plot(rho_ginz,r/R_s,label='Scaling estimate')
plt.xlabel("Density (kg/m$^3$)")
plt.ylabel("R/Rc")
plt.ylim((1.0,ymax))
#plt.legend()
plotname = 'Atm_density_profile_planetv4' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

## Plot atmosphere mass profile
#plt.figure(5)
#plt.plot(mass/Matm,r/R_s)
#plt.xlabel("Mass(R)/Mass ")
#plt.ylabel("R/Rc")
#plt.ylim(1.0,ymax)
##plt.legend()
#plotname = 'Mass_profile_planetv4' + suffix +'.pdf'
## Generate the new file path by placing the file in the suffix-named folder
#new_file_path = os.path.join(folder_name, plotname)
#plt.savefig(new_file_path,bbox_inches='tight',dpi=1000)

# Plot optical depth tau for atmosphere above
plt.figure(6)
plt.plot(tau_above,r/R_s,color='black')
plt.xlabel("Optical depth ")
plt.ylabel("R/Rc")
plt.ylim(1.0,ymax)
plt.xscale("log")
#plt.xlim(-0.5,100.0)
#plt.legend()
plotname = 'Optical_depth_planetv4' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot atmosphere CHORD tau profile integrated from above
plt.figure(7)
plt.plot(tau_chord,r/R_earth_meters,color='black')
plt.xlabel("Chord optical depth ")
plt.ylabel("R/R$_{Earth}$")
plt.ylim(1.0,ymax)
plt.xscale("log")
#plt.xlim(-0.5,100.0)
#plt.legend()
plotname = 'Chord_optical_depth_planetv4' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot atmosphere temperature profile with xover
plt.figure(8)
plt.plot(T,r/R_s,label=r'Model: $\nabla T $',color='black')
plt.plot(T_adiabat,r/R_s,'--',label=r'Model: $\nabla T = =\nabla_{conv}$',color='grey')
plt.plot(T_tau,r/R_s,'--',label=r'Model: $\nabla T_{nc} = =\nabla_{rad}$',color='green')
plt.xlabel("Temperture (K)")
plt.ylabel("R/Rc")
plt.legend(frameon=False)
plt.xlim(-0.1*T[0],1.2*T[0])
plt.ylim((1.0,ymax))
plotname = 'Atm_T_profile_with_xover_planetv4' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot atmosphere xH2
plt.figure(9)
plt.plot(xH2atmosphere,r/R_s,label=r'xH$_2$ atmosphere',color='black')
plt.xlabel("xH$_2$ Atmosphere")
plt.ylabel("R/Rc")
plt.legend(frameon=False)
plt.ylim((1.0,ymax))
plotname = 'Atm_xH2_profile_planetv4' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)


# Plot molec wt atmosphere, kg/(kg/nucleon) = nucleons = amu
mw_atm = m_molec/nucleon_mass
plt.figure(11)
plt.plot(mw_atm,r/R_s,label=r'mol wt atmosphere',color='black')
plt.xlabel("Mean molecular weight atmosphere (amu)")
plt.ylabel("R/Rc")
plt.legend(frameon=False)
plt.ylim((1.0,ymax))
plotname = 'Atm_molwt_profile_planet' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot atmosphere xMgSiO3
plt.figure(12)
plt.plot((1-xH2atmosphere),r/R_s,label=r'xMgSiO$_3$ atmosphere',color='black')
plt.plot(x_inhib,r/R_s,label='Critical for convective inhibition')
plt.xlabel("xMgSiO$_3$ Atmosphere")
plt.ylabel("R/Rc")
plt.legend(frameon=False)
plt.ylim((1.0,ymax))
plt.xlim((0.0,1.0))
plotname = 'Gas_xMgSiO3_planet' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot atmosphere temperature profile
plt.figure(13)
plt.plot(T,r/R_s,label=r'Model: $\nabla T = \nabla_{rad} \nabla_{conv}/(\nabla_{rad}+\nabla_{conv})$',color='black')
#plt.plot(T_adiabat,r/R_s,'--',label=r'Model: $\nabla T = =\nabla_{conv}$',color='grey')
plt.xlabel("Temperture (K)")
plt.ylabel("R/Rc")
plt.legend(frameon=False)
plt.xlim(-0.1*T[0],1.2*T[0])
plt.ylim((1.0,ymax))
plotname = 'Atm_T_profile_planetv4' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot atmosphere optical depth profile
plt.figure(14)
plt.plot(T,tau_above,label=r'Model: $\nabla T = \nabla_{rad} \nabla_{conv}/(\nabla_{rad}+\nabla_{conv})$',color='black')
#plt.plot(T_adiabat,r/R_s,'--',label=r'Model: $\nabla T = =\nabla_{conv}$',color='grey')
plt.xlabel("Temperture (K)")
plt.ylabel("Optical depth")
plt.legend(frameon=False)
plt.xlim(-0.1*T[0],1.2*T[0])
#plt.ylim((1.0,ymax))
plt.yscale("log")
plotname = 'Atm_T_vs_tau' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot atmosphere temperature vs Pbar profile
plt.figure(18)
plt.plot(T,P/1.0e5)
plt.xlabel("Temperture (K)")
plt.ylabel("P (bar)")
plt.yscale("log")
plt.ylim(1.0e6,1.0e-6)
#plt.legend(frameon=False)
plt.xlim(-0.1*T[0],1.2*T[0])
plotname = 'Atm_T_vs_P' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

## Plot core density
#plt.figure(14)
#plt.plot(r4 / R_earth_meters, rho4 / 1e3, label=f"$P_0$ = {P_surf_GPa:.2f} GPa, $T_0$ = {T0} K")
#plt.xlabel(r'Radius (R$_\oplus$)',labelpad=12,fontsize=13)
#plt.ylabel(r'Density (g cm$^-3$)',labelpad=12,fontsize=13)
#plt.legend(fontsize=11, handletextpad=0.5, frameon=False)
#plt.tight_layout()
#plotname = 'Mass_radius_melts_rho_vs_R' + suffix +'.pdf'
#plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot Pressure for planet
plt.figure(15)
plt.plot(r_planet / R_earth_meters, P_planet/1.0e9)
plt.xlabel(r'Radius (R$_\oplus$)',labelpad=12,fontsize=13)
plt.ylabel(r'Pressure (GPa)',labelpad=12,fontsize=13)
#plt.legend(fontsize=11, handletextpad=0.5, frameon=False)
plt.xlim(0.0,1.1*r_chord_earth_units)
plt.tight_layout()
plotname = 'Planet_P_vs_Rearth' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot Temperature for planet
plt.figure(16)
plt.plot(r_planet / R_earth_meters, T_planet)
plt.xlabel(r'Radius (R$_\oplus$)',labelpad=12,fontsize=13)
plt.ylabel(r'Temperature (K)',labelpad=12,fontsize=13)
#plt.legend(fontsize=11, handletextpad=0.5, frameon=False)
plt.xlim(0.0,1.1*r_chord_earth_units)
plt.tight_layout()
plotname = 'Planet_T_vs_Rearth' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)

# Plot density for planet
plt.figure(17)
plt.plot(r_planet / R_earth_meters, rho_planet, label=f"Mass = {Mp_Earth:.2f} $M_\oplus$, \n Radius = {r_chord_earth_units:.2f} $R_\oplus$")
plt.plot(r_planet[i_chord_p]/R_earth_meters,rho_planet[i_chord_p], 'o',color='black',label=r"$\tau = 1$")
plt.xlabel(r'Radius (R$_\oplus$)',labelpad=12,fontsize=13)
plt.ylabel(r'Density (kg m$^-3$)',labelpad=12,fontsize=13)
plt.legend(fontsize=11, handletextpad=0.5, frameon=False)
plt.xlim(0.0,1.1*r_chord_earth_units)
plt.tight_layout()
plotname = 'Planet_rho_vs_Rearth' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)


# Plot total energy for planet
plt.figure(20)
#plt.plot(r_planet[0:i_chord_p-1] / R_earth_meters, E_total_planet, label=f"Total energy")
plt.plot(r_planet[:i_chord_p-1] / R_earth_meters, E_total_planet[:i_chord_p-1], label="Total energy")
plt.xlabel(r'Radius (R$_\oplus$)',labelpad=12,fontsize=13)
plt.ylabel(r'Total Energy (J)',labelpad=12,fontsize=13)
#plt.yscale("log")
plt.legend(fontsize=11, handletextpad=0.5, frameon=False)
plt.xlim(0.0,1.1*r_chord_earth_units)
plt.tight_layout()
plotname = 'Planet_Total_Energy_vs_Rearth' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)


# Plot all forms of energy for planet
plt.figure(23)
plt.plot(r_planet[0:i_chord_p-1] / R_earth_meters, E_thermal_planet[:i_chord_p-1], label =f"Thermal energy")
plt.plot(r_planet[0:i_chord_p-1] / R_earth_meters, E_gravPE_planet[:i_chord_p-1], label =f"Gravitational potential energy")
plt.plot(r_planet[0:i_chord_p-1] / R_earth_meters, E_total_planet[:i_chord_p-1], label=f"Total energy")
plt.xlabel(r'Radius (R$_\oplus$)',labelpad=12,fontsize=13)
plt.ylabel(r'Energy (J)',labelpad=12,fontsize=13)
#plt.yscale("log")
plt.legend(fontsize=11, handletextpad=0.5, frameon=False)
plt.xlim(0.0,1.1*r_chord_earth_units)
plt.tight_layout()
plotname = 'Planet_Energy_vs_Rearth' + suffix +'.pdf'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=1000)



# PLOT DIAGRAM OF THE PLANET
#Radius of metal core in Earth units = (Rcmb4/R_earth_meters)
#Radius of core in Earth units (r4[-1]/R_earth_meters)
# Planet radius in Earth radii using chord tau = r_chord_earth_units

radii = [(Rcmb4/R_earth_meters), (r4[-1]/R_earth_meters), r_chord_earth_units]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot a circle contoured with log(density) values indicated by a sequential color map
radius_max=radii[2] # transit radius is the maximum
num_rings=i_chord_p

print(' i_chord_p = ',i_chord_p)
values=np.zeros(i_chord_p)
for i in range(i_chord_p):
    values[i]=rho_planet[i]

# Normalize values for colormap to be between 0 and 1 for color map
#norm = colors.LogNorm(vmin=np.min(values), vmax=np.max(values)) # Use data to determine limits
norm = colors.LogNorm(vmin=np.min(values), vmax=3.0e4) # Use data to determine limits

# Define custom colormap
#colors_list = [(0, "lightblue"), (1, "gold")]
colors_list = ['lightblue', 'deepskyblue', 'lightgreen','yellow','lightsalmon','salmon','darkred']
new_cmap = LinearSegmentedColormap.from_list("new_cmap", colors_list, N=i_chord_p) # Makes smooth transitions between colors


# Make rings
for i in range(1, num_rings):
    r_inner=r_planet[i-1] / R_earth_meters
    r_outer=r_planet[i] / R_earth_meters
    #color = new_cmap(values[i])
    color = new_cmap(norm(values[i]))
    
    # Create a full-circle wedge (theta1=0, theta2=360)
    ring = Wedge(center=(0, 0), r=r_outer, theta1=0, theta2=360, width=r_outer - r_inner, color=color)
    ax.add_patch(ring)

# Set aspect of the plot to be equal
#ax.set_aspect('equal', adjustable='datalim')
ax.set_aspect('equal')

# Set limits for the axes
rad_lim=4.0
if r_chord_earth_units > rad_lim:
    rad_lim = r_chord_earth_units
if rad_lim == 0:
    ax.set_xlim(-max(radii)-1, max(radii)+1)
    ax.set_ylim(-max(radii)-1, max(radii)+1)
# Alternative fixed limits for axes in Earth units
else:
    ax.set_xlim(-rad_lim, rad_lim)
    ax.set_ylim(-rad_lim, rad_lim)

# Add centered x and y axes
ax.spines['left'].set_position('zero')
ax.spines['left'].set_color('gray')
ax.spines['left'].set_linewidth(0.8)

ax.spines['bottom'].set_position('zero')
ax.spines['bottom'].set_color('gray')
ax.spines['bottom'].set_linewidth(0.8)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Define a custom formatter to remove labels at 0
def custom_formatter(value, tick_number):
    if value == 0:
        return ""  # Suppress label for 0
    return str(int(value))  # Convert other values to integers

# Apply the custom formatter
ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

# Add minor ticks
ax.minorticks_on()
ax.tick_params(axis='both', which='major', length=6, width=1.2, color='gray')
ax.tick_params(axis='both', which='minor', length=3, width=0.8, color='gray')

# Add labels and text
plt.xlabel('')
plt.ylabel('')
if rad_lim == 0:
    x_pos=max(radii)+1.0 # Rightmost position
    y_pos=max(radii) # Upper limit for y axis
else:
    x_pos=rad_lim
    y_pos=max(radii)
    
ax.text(x_pos,y_pos,'Transit radius = %.3f (R$_\oplus$) \n Surface = %.3f (R$_\oplus$)' %(radii[2],radii[1]), ha='right',va='bottom',fontsize=9)

# Add colorbar
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=new_cmap), label=r'$\rho$ (kg/m$^3$)',ax = ax)
# Invert the color bar axis
cbar.ax.invert_yaxis()  # Use invert_xaxis() for a horizontal color bar

#plt.grid(True, linestyle='--', alpha=0.7)
plotname='Planet_radii' + suffix +'.png'
plt.savefig(out_dir / plotname, bbox_inches='tight', dpi=300)

# SHOW ALL PLOTS
plt.show()  # Including only one plt.show command makes all plots in their own windows

print('Done...')
exit()


