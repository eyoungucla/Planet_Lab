#------------------------------------------------------------------------------
# Interior model of rotating fluid planet.
#------------------------------------------------------------------------------
import numpy as np
import cms
import warnings
from timeit import default_timer as timer

class CMSPlanet:
    """Interior model of rotating fluid planet.

    This class implements a model of a rotating fluid planet using Concentric
    Maclaurin Spheroids to calculate the hydrostatic equilibrium shape and
    resulting gravity field. A CMSPlanet object is defined by a density profile
    rho(a), supplied by the user and stored in the column vectors obj.ai and
    obj.rhoi, indexed from the surface in. To complete the definition the user
    must also specify a mass, equatorial radius, and rotation period. With
    these a gravity field and equilibrium shape can be determined, with a call
    to obj.relax_to_HE().

    Note that the oblate shape calculated with relax_to_HE() preserves the
    equatorial radius of the planet but not its mass. If fixmass is True
    (default: True) the density will be re-normalized to match the reference
    mass, modifying the implied 1-bar density. It is not possible to define
    mass, radius, and density simultaneously.
    """
    def __init__(self, obs=None, **kwargs):
        self._G = 6.67430e-11 # m^3 kg^-1 s^-2 (2018 NIST reference)
        self.opts = _default_opts(kwargs) # holds user configurable options
        if obs is None:
            obs = _default_planet() # undocumented defaults for quick testing
        self.set_observables(obs)
        self.Pi = None
        self.CMS = None

    def set_observables(self,obs):
        """Copy physical properties from an observables struct."""
        self.name = obs.pname
        self.mass = obs.M
        self.radius = obs.a0
        self.P0 = obs.P0
        self.period = obs.P
        self._GM = self._G*self.mass

    def relax_to_HE(self, fixmass=True):
        """Call cms() to obtain equilibrium shape and gravity."""

        if (self.opts['verbosity'] > 1):
            print('  Relaxing to hydrostatic equilibrium...')
        tic = timer()

        self.Js, self.CMS = cms.cms(self.ai, self.rhoi, self._qrot(),
            dJtol=self.opts['dJtol'], maxiter=self.opts['MaxIterHE'],
            xlayers=self.opts['xlayers'])
        toc = timer() - tic

        if (self.opts['verbosity'] > 1):
            print('  Relaxing to hydrostatic equilibrium...done.')
            print(f' Elapsed time {toc:g} sec.')

        if fixmass:
            self.rhoi = self.rhoi*self.mass/self._M()

    # def level_surfaces(self,mus):
    #     # Normalized r(cos(theta)) from shape functions
    #     ss = self.ss
    #     s0 = ss[0]; s2 = ss[1]; s4 = ss[2]; s6 = ss[3]; s8 = ss[4]
    #     shp = (np.outer(s0,_Pn(0,mus)) + np.outer(s2,_Pn(2,mus)) +
    #            np.outer(s4,_Pn(4,mus)) + np.outer(s6,_Pn(6,mus)) +
    #            np.outer(s8,_Pn(8,mus)))
    #     return np.flipud(1 + shp)

    def NMoI(self):
        # TODO: implement
        return None

    ### Private methods slash pseudo properties
    def _M(self): #TODO implement
        return self.mass
    
    def si(self):
        return None
    
    def _qrot(self):
        return (2*np.pi/self.period)**2*self.radius**3/(self._G*self._M())

    def bi(self):
        return self.s0*self.level_surfaces(1) # level surfaces polar radii

    ### Visualizers
    def plot_rho_of_r(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,6))
        x = np.append(self.ai/self.a0, 0)
        y = np.append(self.rhoi, self.rhoi[-1])/1000
        plt.plot(x, y, lw=2, label=self.name)
        plt.xlabel(r'Layer radius, $a/a_0$', fontsize=12)
        plt.ylabel(r'$\rho$ [1000 kg/m$^3$]', fontsize=12)
        plt.show(block=False)

    def plot_P_of_r(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,6))
        x = self.ai/self.a0
        y = self.Pi/1e9
        plt.plot(x, y, lw=2, label=self.name)
        plt.xlabel(r'Layer radius, $a/a_0$', fontsize=12)
        plt.ylabel(r'$P$ [GPa]', fontsize=12)
        plt.show(block=False)

    def plot_P_of_rho(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,6))
        x = self.rhoi/1000
        y = self.Pi/1e9
        plt.plot(x, y, lw=2, label=self.name)
        plt.xlabel(r'$\rho$ [1000 kg/m$^3$]', fontsize=12)
        plt.ylabel(r'$P$ [GPa]', fontsize=12)
        plt.show(block=False)

    def plot_rho_of_P(self):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,6))
        x = self.Pi/1e9
        y = self.rhoi/1000
        plt.loglog(x, y, lw=2, label=self.name)
        plt.xlabel(r'$P$ [GPa]', fontsize=12)
        plt.ylabel(r'$\rho$ [1000 kg/m$^3$]', fontsize=12)
        plt.xlim(left=1e-3)
        plt.show(block=False)

    ## Reporters/exporters
    def to_ascii(self, fname=None):
        fname = fname if fname else (self.name+'.txt')
        with open(fname, 'wt') as fid:
            # The header
            fid.write('# Equilibrium shape and gravity solved with ')
            fid.write("Concentric Maclaurin Spheroids\n")
            fid.write('#\n')
            fid.write(f'# Model name: {self.name}\n')
            fid.write('#\n')
            fid.write(f'# Scalar quantities:\n')
            fid.write(f'# N layers = {len(self.ai)}\n')
            fid.write(f'# xlayers = {self.opts["xlayers"]}\n')
            fid.write(f'# Mass M = {self.M} kg\n')
            fid.write(f'# Mean radius       s0 = {self.s0:0.6e} m\n')
            fid.write(f'# Equatorial radius a0 = {self.a0:0.6e} m\n')
            fid.write(f'# Rotation period P = {self.period:0.6g} s\n')
            fid.write(f'# Normalized MOI = {self.NMoI:0.6f}\n')
            fid.write(f'#\n')
            fid.write(f'# Calculated gravity zonal harmonics (x 10^6):\n')
            fid.write(f'# J0  = {self.Js[0]*1e6:12.6f}\n')
            fid.write(f'# J2  = {self.Js[1]*1e6:12.6f}\n')
            fid.write(f'# J4  = {self.Js[2]*1e6:12.6f}\n')
            fid.write(f'# J6  = {self.Js[3]*1e6:12.6f}\n')
            fid.write(f'# J8  = {self.Js[4]*1e6:12.6f}\n')
            fid.write(f'#\n')
            fid.write('# Column data description (MKS):\n')
            fid.write('# i     - layer index (increasing with depth)\n')
            fid.write('# a_i   - equatorial radius of layer i\n')
            fid.write('# rho_i - density inside layer i\n')
            fid.write('# P_i   - pressure at top of layer i\n')
            fid.write('#\n')

            # The data
            fid.write('# Column data:\n')
            fid.write(f'# {"i":>4}')
            fid.write(f'{"a_i":>8}')
            fid.write(f'{"rho_i":>14s}')
            fid.write(f'{"P_i":>10}')
            fid.write('\n')
            for k in range(len(self.si)):
                fid.write(f'  {k:-4d}  ')
                fid.write(f'{self.ai[k]:10.4e}  ')
                fid.write(f'{self.rhoi[k]:10.4e}  ')
                fid.write(f'{self.Pi[k]:10.4e}  ')
                fid.write(f'\n')
        return

# Class-related functions
def _Pn(n, x):
    # Fast implementation of ordinary Legendre polynomials of low even degree.
    if n == 0:
        y = np.ones_like(x)
    elif n == 2:
        y = 0.5*(3*x**2 - 1)
    elif n == 4:
        y = (1/8)*(35*x**4 - 30*x**2 + 3)
    elif n == 6:
        y = (1/16)*(231*x**6 - 315*x**4 + 105*x**2 - 5)
    elif n == 8:
        y = (1/128)*(6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35)
    else:
        raise(Exception("Unimplemented order"))
    return y

def _mass_int(svec, dvec):
    """Trapz-integrate mass from rho(r) data."""
    from scipy.integrate import trapz
    return -4*np.pi*trapz(dvec*svec**2, x=svec)

def _default_opts(kwargs):
    """Return options dict used by TOFPlanet class methods."""
    opts = {'dJtol':1e-7,
            'drottol':1e-6,
            'MaxIterHE':60,
            'xlayers':-1,
            'verbosity':1
            }
    for kw, v in kwargs.items():
        if kw in opts:
            opts[kw] = v
    return opts

class _default_planet:
    """Use this to prefill critical TP fields with reasonable values."""
    pname = 'planet'
    M  = 1898.187e24
    a0 = 71492e3
    s0 = 69911e3
    P0 = 1e5
    P = 0.41354*24*3600

def _test(N,nx):
    obs = _default_planet()
    cp = CMSPlanet(obs,xlayers=nx)
    zvec = np.linspace(1, 1/N, N)
    dvec = -3000*zvec**2 + 3000
    cp.ai = zvec*cp.radius
    cp.rhoi = dvec
    #a = - 15*obs.M/8/np.pi/obs.s0**3
    tic = timer()
    it = cp.relax_to_HE()
    toc = timer()
    print()
    print(f"{N=}, {nx=}")
    print(f"cms finished in {toc-tic:.3g} sec.")
    print("J0 = {}".format(cp.Js[0]))
    print("J2 = {}".format(cp.Js[1]))
    print("J4 = {}".format(cp.Js[2]))
    print("J6 = {}".format(cp.Js[3]))
    print("J8 = {}".format(cp.Js[4]))
    print("I = {}".format(cp.NMoI()))
    print("")

if __name__ == '__main__':
    _test(128,-1)
