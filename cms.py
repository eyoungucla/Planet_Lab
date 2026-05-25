#---------------------------------------------------------------------------------
#  Concentric Maclaurin Spheroids gravity calculator
#
# Author: Naor Movshovitz (nmovshov at gee mail dot com)
#---------------------------------------------------------------------------------
from __future__ import division
import sys
import numpy as np
from scipy.interpolate import CubicSpline as spline
import warnings

# numba is optional — use it for speed if available, otherwise fall back
# to pure Python (slower but correct)
try:
    from numba import jit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        """No-op decorator when numba is not installed."""
        def decorator(fn):
            return fn
        return decorator if (args and callable(args[0])) else decorator

def cms(zvec, dvec, qrot, dJtol=1e-6, maxiter=100, xlayers=64, J0s=None):
    """Return gravity coefficients of density profile in hydrostatic equilibrium.

    Js,out = cms(zvec, dvec, qrot) returns the even harmonic gravity
    coefficients from J0 to J30 in the vector Js, so that Js[0] is J0, Js[1] is
    J2, Js[2] is J4, etc. The mandatory inputs are a vector of equatorial radii
    zvec, vector of corresponding layer densities dvec, and rotation parameter
    qrot. The struct out holds diagnostic values and the full hydrostatic
    spheroid shapes.

    Required inputs
    ----------
    zvec : ndarray, 1d, positive
        Equatorial radii of constant density layers, indexed from the outside
        in, i.e., zvec[0]=a0 is the outer radius of the outermost layer,
        zvec[1] is the inner radius of the outermost layer as well as the outer
        radius of the next layer, etc. The innermost layer extends all the way
        to the center, so that zvec[-1] is the outer radius of a central
        spheroid layer. Units of zvec are unimportant as values will be
        normalized to outer radius.
    dvec : ndarray, 1d, positive
        Layer densities. The layer lying between zvec[i] and zvec[i+1] has
        constant density dvec[i]. Units are unimportant as values will be
        normalized to the mean (bulk) density. Density is expected to be
        monotonically non-increasing with zvec, but this is not enforced.
        (Note: these are layer densities and NOT the density "deltas" of
        concentric spheroids.)
    qrot : float, scalar, nonnegative
        Dimensionless rotation parameter. Recall q = w^2a0^3/GM.

    Optional parameters
    ----------
    dJtol : scalar, positive (dJtol=1e-6)
        Convergence tolerance on fractional change in Js in successive
        iterations.
    maxiter : scalar, positive, integer, (maxiter=100)
        Limit iterations of CMS algorithm.
    xlayers : scalar or vector, nonnegative, integer (xlayers=64)
        Layers whose shape will be explicitly calculated. The shape functions
        (zetas) will be explicitly calculated for these layers and
        spline-interpolated in between. This can result in significant speedup
        with minimal loss of precision, if the xlayers are chosen by trial and
        error to fit the required precision and the spacing of density layers.
        A scalar value is interpreted as a number of xlayers to be uniformaly
        distributed among the density layers. For example, a smooth-density,
        1024-layer model can benefit from almost 16x-speedup by specifying
        xlayers=64 while retaining a 10^-6 relative precision on J2. A vector
        value is interpreted as indices of layers to be used as xlayers. (A
        negative value is a shortcut to flag a full calculation instead of
        skip-n-spline, useful for debugging.)
    J0s : struct
       J-like values representing initial state. This is not just for speeding
       up convergence. Mostly it's a mechanism to preserve state between calls.

    Outputs
    -------
    Js : 1-by-16 vector, real
        Even harmonic gravity coefficients J0 to J30. Typically only J2 to J10
        are helpful. J0 is included as a sanity check and test of convergence.
    out : struct
        A structure holding other quantities calculated in the course of
        running cms. Including out.zetas and out.JLike that together define the
        converged hydrostatic shape.
    """

    # Minimal input control
    zvec = np.array(zvec)
    dvec = np.array(dvec)
    assert zvec.shape == dvec.shape
    assert qrot >= 0

    # Normalize and flip radii and densities (it's safe to normalize a normal)
    p = np.argsort(zvec)
    zvec = zvec[p]
    dvec = dvec[p]
    if zvec[0] == 0:
        zvec[0] = np.spacing(1)
    dro = np.hstack((dvec[-1], np.diff(np.flipud(dvec))))
    m = sum(dro*np.flipud(zvec)**3)
    robar = m/zvec[-1]**3
    zvec = np.flipud(zvec/zvec[-1])
    dvec = np.flipud(dvec/robar)

    ## Define and initialize local variables (in CMS notation)
    lambdas = zvec
    deltas = np.hstack((dvec[0], np.diff(dvec)))
    nlay = len(lambdas)
    nangles = 48
    kmax = 30

    # Define down-sampled variabels (for skip-n-spline)
    if np.isscalar(xlayers):
        sskip = max(nlay//xlayers, 1)
        xind = np.arange(0,nlay,sskip)
    else:
        xind = np.array(xlayers)
    xlambdas = lambdas[xind]
    xdvec = dvec[xind]
    xdeltas = np.hstack((xdvec[0], np.diff(xdvec)))
    nxlay = len(xlambdas)

    # Initialize zetas as spherical
    zetas = np.ones((nlay, nangles))
    xzetas = np.ones((nxlay, nangles))

    # Initialize J-like quantities for spherical planet
    if J0s is None:
        Jlike = _allocate_spherical_Js(nxlay, kmax, xlambdas, xdeltas)
    else:
        Jlike = J0s

    # Abscissas and weights for Gaussian quadrature
    mus, gws = _gauleg(0, 1, nangles)

    # Precompute Legendre polynomials for fixed colatitudes (for gauss quad)
    class Ps:
        pass
    Ps.Pnmu = np.zeros((kmax+1,nangles))
    Ps.Pnzero = np.zeros((kmax+1,))
    for k in range(kmax+1):
        Ps.Pnmu[k,:] = _Pn(k, mus)
        Ps.Pnzero[k] = _Pn(k, 0)

    # Precompute powers of ratios of lambdas (only for explicit layers)
    lamratpow = _powers_of_ratios(xlambdas, kmax)

    # The loop (see Hubbard, 2012 and ./notes/CMS.pdf)
    Js = Jlike.Jn # J0=0 ensures at least one iteration
    eps = np.finfo(float).eps
    for iter in range(maxiter):
        # Update shape with current gravity
        new_xzetas = _update_zetas(Jlike, Ps, lamratpow, qrot, xzetas)

        #... and spline
        for alfa in range(nangles):
            x = np.flipud(xlambdas)
            y = np.flipud(new_xzetas[:,alfa])
            # Guard against non-strictly-increasing x caused by bracket
            # failures that leave duplicate zeta values.  Deduplicate by
            # keeping only points where x strictly increases.
            keep = np.concatenate(([True], np.diff(x) > 0))
            x_k  = x[keep]
            y_k  = y[keep]
            if len(x_k) >= 4:
                cs = spline(x_k, y_k)
                zetas[:,alfa] = cs(lambdas)
            else:
                # Too few unique points — fall back to linear interpolation
                zetas[:,alfa] = np.interp(lambdas, x_k[::-1], y_k[::-1])
        assert np.all(np.isfinite(zetas)), "interpolation error"

        # Update gravity with current shape
        new_Jlike = _update_Js(lambdas, deltas, zetas, xind, Ps, gws)
        new_Js = new_Jlike.Jn

        # Check for convergence of J0-J8 to terminate...
        dJs = np.abs((Js - new_Js)/(Js+eps))
        if np.all(dJs[0:4] < dJtol):
            break
        # ... or update to new values
        Jlike = new_Jlike
        Js = new_Js
        xzetas = new_xzetas

    # It's not necessarily terrible to reach maxiter, but we'd want to know
    if iter == (maxiter - 1):
        warnings.warn('Spheroid shapes may not be fully converged.')

    # Return
    Js = new_Js
    class out:
        pass
    out.dJs = dJs
    out.iter = iter
    out.zetas = zetas
    out.lambdas = lambdas
    out.deltas = deltas
    out.Jlike = new_Jlike
    out.mus = mus
    out.gws = gws
    out.Ps = Ps
    out.xind = xind

    return (Js, out)

#--------------------------------
#  Helper functions
#--------------------------------
@jit(nopython=True)
def _powers_of_ratios(xlam, kmax):
    """Return array of powers of ratios of spheroid radii."""
    nx = len(xlam)
    lamrat = np.zeros((kmax+2,nx,nx))
    for ii in range(nx):
        for jj in range(nx):
            for kk in range(kmax+2):
                lamrat[kk,ii,jj] = (xlam[ii]/xlam[jj])**(kk)
    return lamrat

def _allocate_spherical_Js(nlay,nmom,lambdas,deltas):
    """Return a struct of J-like quantities representing a spherical planet."""
    class Js:
        pass
    Js.tilde = np.zeros((nlay,(nmom+1)))
    Js.tildeprime = np.zeros((nlay,(nmom+1)))
    Js.Jn = np.zeros((nmom//2+1,))
    den = sum(deltas*lambdas**3)
    Js.tilde[:,0] = -1*(deltas*lambdas**3)/den
    Js.tildeprime[:,0] = -1.5*(deltas*lambdas**3)/den
    Js.tildeprimeprime = 0.5*(deltas*lambdas**3)/den
    Js.Jn[0] = sum(Js.tilde[:,0])
    return Js

def _gauleg(x1, x2, n):
    """Return abscissas and weights for Gauss-Legendre n-point quadrature.

    x,w = GAULEG(x1,x2,n) returns the abscissas x and weights w that can be
    used to evaluate the definite integral, I, of a function well approximated
    by an (2n - 1) degree polynomial in the interval [x1,x2] using the
    Gauss-Legendre formula:
        I = sum(w.*f(x))

    Algorithm:
      This function is based on the C++ implementation of a routine with the
      same name in Numerical Recipes, 3rd Edition. But in several places I opt
      for readability over performance, on the assumption that this function is
      most likely to be called in a setup routine rather than in an inner-loop
      computation.

    Example:
      fun = np.sin
      x,w = gauleg(0, np.pi, 6)
      I_quad = scipy.integrate.quad(fun, 0, np.pi)
      I_gaussleg = sum(w*fun(x))

    Reference: William H. Press, Saul A. Teukolsky, William T. Vetterling, and
    Brian P. Flannery. 2007. Numerical Recipes 3rd Edition: The Art of
    Scientific Computing (3 ed.). Cambridge University Press, New York, NY,
    USA.
    """

    # Minimal assertions
    assert np.isfinite(x1), "x1 must be real and finite"
    assert np.isfinite(x2), "x2 must be real and finite"
    assert int(n) == n and n > 2, "n must be positive integer > 2"
    assert x2 > x1, "Interval must be positive"

    # Local variables
    tol = 1e-14
    m = int(np.ceil(n/2))
    xmid = (x1 + x2)/2
    dx = (x2 - x1)
    x = np.nan*np.ones((n,))
    w = np.nan*np.ones((n,))

    # Main loop
    for j in range(m):
        # Get j-th root of Legendre polynomial Pn, along with Pn' value there.
        z = np.cos(np.pi*(j + 0.75)/(n + 0.5)) # initial guess for j-th root
        while True:
            # Calculate Pn(z) and Pn-1(z) and Pn'(z)
            p = np.nan*np.ones((n+1,))
            p[0] = 1
            p[1] = z
            for k in range(2,n+1):
                pkm1 = p[k-1]
                pkm2 = p[k-2]
                pk = (1/k)*((2*k - 1)*z*pkm1 - (k - 1)*pkm2)
                p[k] = pk
            pn = p[-1]
            pp = (n*p[-2] - n*z*p[-1])/(1 - z**2)

            # And now Newton's method (we are hopefully very near j-th root)
            oldz = z
            z = z - pn/pp
            if np.abs(z - oldz) < tol:
                break

        # Now use j-th root to get 2 abscissas and weights
        x[j]     = xmid - z*dx/2 # Scaled abscissa left of center
        x[n-1-j] = xmid + z*dx/2 # Scaled abscissa right of center
        w[j]     = dx/((1 - z**2)*pp**2)
        w[n-1-j] = w[j]

    # Verify and return
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(w))
    return (x,w)

def _Pn(n, x):
    """Fast implementation of ordinary Legendre polynomials of low degree.

    y = Pn(n,x) returns the ordinary Legendre polynomial of degree n evaulated
    at x. For n <= 12 the polynomials are implemented explicitly resutling in
    faster calculation compared with the recursion formula. For n > 12 we fall
    back on scipy.special.eval_legendre(n,x).

    Note: in keeping with the premise of an optimized implementation this
    function performs no input checks at all. Use with care.
    """
    x = np.array(x, dtype=float)

    if n == 0:
        y = np.ones(x.shape)
    elif n == 1:
        y = x
    elif n == 2:
        y = 0.5*(3*x**2 - 1)
    elif n == 3:
        y = 0.5*(5*x**3 - 3*x)
    elif n == 4:
        y = (1/8)*(35*x**4 - 30*x**2 + 3)
    elif n == 5:
        y = (1/8)*(63*x**5 - 70*x**3 + 15*x)
    elif n == 6:
        y = (1/16)*(231*x**6 - 315*x**4 + 105*x**2 - 5)
    elif n == 7:
        y = (1/16)*(429*x**7 - 693*x**5 + 315*x**3 - 35*x)
    elif n == 8:
        y = (1/128)*(6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35)
    elif n == 9:
        y = (1/128)*(12155*x**9 - 25740*x**7 + 18018*x**5 - 4620*x**3 + 315*x)
    elif n == 10:
        y = (1/256)*(46189*x**10 - 109395*x**8 + 90090*x**6 - 30030*x**4 + 3465*x**2 - 63)
    elif n == 11:
        y = (1/256)*(88179*x**11 - 230945*x**9 + 218790*x**7 - 90090*x**5 + 15015*x**3 - 693*x)
    elif n == 12:
        y = (1/1024)*(676039*x**12 - 1939938*x**10 + 2078505*x**8 - 1021020*x**6 + 225225*x**4 - 18018*x**2 + 231)
    else:
        from scipy.special import eval_legendre
        y = eval_legendre(n,x)

    return y

def _zeta_j_of_alfa(j, alfa, Js, Ps, lamrats, qrot, oldzeta):
    """Call fzero on _eq52 to find equilibrium zeta."""
    from scipy.optimize import brentq
    Jt = Js.tilde
    Jtp = Js.tildeprime
    Jtpp = Js.tildeprimeprime
    P0 = Ps.Pnzero
    Pmu = Ps.Pnmu[:,alfa]
    def fun(x):
        return _eq52(x,j,Jt,Jtp,Jtpp,P0,Pmu,lamrats,qrot)

    # Scan a wide range of brackets including [0.5,1.0] and [0.8,1.05]
    # to handle both highly flattened and near-spherical layers.
    # Physical zeta should be in (0, 1] but we allow slightly above 1
    # in the scan to avoid missing a root near 1.
    brackets = [(0.95, 1.0), (0.8, 1.0), (0.6, 1.0),
                (0.5, 1.0), (0.3, 1.0), (0.99, 1.05)]
    for lo, hi in brackets:
        try:
            fa = fun(lo)
            fb = fun(hi)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                sol = brentq(fun, lo, hi, xtol=1e-9, rtol=1e-9)
                # Clamp to physical range before returning
                return float(np.clip(sol, 0.3, 1.0))
        except Exception:
            continue

    # Last resort: secant from previous zeta
    from scipy.optimize import root_scalar
    try:
        x0 = float(np.clip(oldzeta, 0.3, 1.0))
        x1 = x0 * 0.99
        sol = root_scalar(fun, x0=x0, x1=x1, method='secant',
                          xtol=1e-9, rtol=1e-9)
        if sol.converged and np.isfinite(sol.root):
            return float(np.clip(sol.root, 0.3, 1.0))
    except Exception:
        pass

    # Final fallback: return a safe physical value (near-spherical)
    # clamped to [0.5, 1.0] so it cannot blow up the power series in _eq52.
    import warnings
    safe = float(np.clip(oldzeta, 0.5, 1.0))
    warnings.warn(f'_zeta_j_of_alfa: bracket search failed for layer {j}, '
                  f'angle {alfa}. Clamping to {safe:.6f}.')
    return safe

@jit(nopython=True)
def _eq52(zja, jl, Jt, Jtp, Jtpp, P0, Pmu, lamrats, qrot):
    # locals
    nlay = lamrats.shape[1]
    kmax = len(P0)-1
    lamj3 = lamrats[3,jl,0]
    q = qrot
    pows = np.arange(kmax+2)
    zetpow = zja**pows
    zetipow = zja**-pows

    # first double sum
    y1 = 0
    for ii in range(jl,nlay):
        for kk in range(0,kmax,2):
            y1 = y1 + lamrats[kk,ii,jl]*Jt[ii,kk]*zetipow[kk]*Pmu[kk]

    # second double sum
    y2 = 0
    for ii in range(jl,nlay):
        for kk in range(0,kmax,2):
            y2 = y2 + lamrats[kk,ii,jl]*Jt[ii,kk]*P0[kk]

    # third double sum
    y3 = 0
    for ii in range(jl):
        for kk in range(0,kmax,2):
            y3 = y3 + lamrats[kk+1,jl,ii]*Jtp[ii,kk]*zetpow[kk+1]*Pmu[kk]

    # and forth double sum
    y4 = 0
    for ii in range(jl):
        for kk in range(0,kmax,2):
            y4 = y4 + lamrats[kk+1,jl,ii]*Jtp[ii,kk]*P0[kk]

    # a single sum
    y5 = 0
    for ii in range(jl):
        y5 = y5 + lamrats[3,jl,ii]*Jtpp[ii]*zetpow[3]

    # another single sum
    y6 = 0
    for ii in range(jl):
        y6 = y6 + lamrats[3,jl,ii]*Jtpp[ii]

    # and the rotation term
    y7 = -(1/3)*q*lamj3*zja**2*(1 - Pmu[2]) + (1/2)*q*lamj3

    # Now combine
    y = (1/zja)*(y1 + y3 + y5) - y2 - y4 - y6 + y7

    # And BYO
    return y

def _update_zetas(Js, Ps, lamrats, qrot, oldzetas):
    """Update layer shapes using current value of Js.

    For shells whose density delta is negligibly small (near-massless
    outer atmosphere), _eq52 becomes too flat for Brent's method.
    These shells are identified by their tildeprimeprime value (which is
    proportional to delta*lambda^3) and assigned the leading-order
    analytic test-particle zeta instead of solving _eq52.
    """
    nlay    = oldzetas.shape[0]
    nangles = oldzetas.shape[1]
    newzetas = np.nan*np.zeros((nlay, nangles))

    # Identify near-massless shells from tildeprimeprime (mass proxy).
    # tildeprimeprime[i] = 0.5 * delta_i * lambda_i^3 / denom
    # A shell is near-massless if its mass weight is < 0.1% of the max.
    tpp      = np.abs(Js.tildeprimeprime)
    max_tpp  = np.max(tpp) if np.max(tpp) > 0 else 1.0
    massless_threshold = 1e-3   # 0.1% of peak mass weight

    # Leading-order analytic zeta for massless test-particle limit:
    # zeta*(theta) = 1 - (5q/4)*(1 - P2(cos theta))
    analytic_eps = 5.0 * qrot / 4.0

    # Note: tildeprimeprime has shape (nxlay,) not (nlay,).
    # Map xlay indices to full layer indices via nearest neighbour.
    nxlay = len(tpp)
    xind_approx = np.round(np.linspace(0, nlay-1, nxlay)).astype(int)

    # Build per-layer massless flag
    is_massless = np.zeros(nlay, dtype=bool)
    for xi, j in enumerate(xind_approx):
        if tpp[xi] / max_tpp < massless_threshold:
            is_massless[j] = True

    for j in range(nlay):
        for alfa in range(nangles):
            if is_massless[j]:
                p2 = Ps.Pnmu[2, alfa]
                zeta_analytic = 1.0 - analytic_eps * (1.0 - p2)
                newzetas[j, alfa] = float(np.clip(zeta_analytic, 0.85, 1.0))
            else:
                oldzeta = oldzetas[j, alfa]
                newzetas[j, alfa] = _zeta_j_of_alfa(
                    j, alfa, Js, Ps, lamrats, qrot, oldzeta)

    return newzetas

@jit(nopython=True)
def _eqs48(lambdas, deltas, zetas, xlambdas, xdeltas, xzetas, pnmu, gws):
    """A nopython version of eqs. 48."""

    nlay = lambdas.shape[0]
    nxlay = xlambdas.shape[0]
    kmax = pnmu.shape[0]-1

    # Precompute common denominator in eqs. (48) (USING FULL DENSITY PROFILE)
    denom = 0
    for j in range(nlay):
        fun = zetas[j,:]**3
        I = np.dot(fun,gws) # the gauss quadrature formula
        denom = denom + I*deltas[j]*lambdas[j]**3

    # Do J tilde, eq. (48a)
    tilde = np.zeros((nxlay,kmax+1))
    for ii in range(nxlay):
        for kk in range(kmax+1):
            if (kk%2 > 0):
                continue
            fun = pnmu[kk,:]*xzetas[ii,:]**(kk+3)
            I = np.dot(gws,fun) # gauss quad formula
            tilde[ii,kk] = -(3/(kk + 3))*xdeltas[ii]*xlambdas[ii]**3*I/denom

    # Do J tilde prime, eqs. (48b and 48c)
    tprime = np.zeros((nxlay,kmax+1))
    for ii in range(nxlay):
        for kk in range(kmax+1):
            if (kk%2 > 0):
                continue
            if kk == 2: # eq. (48c)
                fun = pnmu[2,:]*np.log(xzetas[ii,:])
                I = np.dot(gws,fun) # gauss quad formula
                tprime[ii,kk] = -3*xdeltas[ii]*xlambdas[ii]**3*I/denom
            else:       # eq. (48b)
                fun = pnmu[kk,:]*xzetas[ii,:]**(2 - kk)
                I = np.dot(gws,fun) # gauss quad formula
                tprime[ii,kk] = -(3/(2 - kk))*xdeltas[ii]*xlambdas[ii]**3*I/denom

    # Do J tilde double prime, eq. (48d)
    tpprime = np.zeros((nxlay,))
    for ii in range(nxlay):
        tpprime[ii] = 0.5*xdeltas[ii]*xlambdas[ii]**3/denom

    # And finally, the EXTERNAL Js deserve full grid resolution
    full_tilde = np.zeros((nlay,kmax+1))
    for ii in range(nlay):
        for kk in range(kmax+1):
            if (kk%2 < 0):
                continue
            fun = pnmu[kk,:]*zetas[ii,:]**(kk+3)
            I = np.dot(gws,fun) # gauss quad formula
            full_tilde[ii,kk] = -(3/(kk + 3))*deltas[ii]*lambdas[ii]**3*I/denom

    # Return the arrays in a tuple
    return (tilde, tprime, tpprime, full_tilde)

def _update_Js(lambdas, deltas, zetas, xind, Ps, gws):
    """Single-pass update of gravitational moments by Gaussian quad."""

    nlay = len(lambdas)
    kmax = len(Ps.Pnzero)-1
    xlambdas = lambdas[xind]
    dvec = np.cumsum(deltas)
    xdvec = dvec[xind]
    xdeltas = np.hstack((xdvec[0], np.diff(xdvec)))
    xzetas = zetas[xind, :]
    nxlay = len(xlambdas)

    (new_tilde, new_tprime, new_tpprime, full_tilde) = _eqs48(lambdas,
            deltas, zetas, xlambdas, xdeltas, xzetas, Ps.Pnmu, gws)

    # Return updated Js struct
    class newJs:
        pass
    newJs.tilde = new_tilde
    newJs.tildeprime = new_tprime
    newJs.tildeprimeprime = new_tpprime
    newJs.fulltilde = full_tilde
    n = np.arange(0,kmax+1,2)
    newJs.Jn = np.zeros((len(n),))
    for k in range(len(n)):
        newJs.Jn[k] = np.sum(newJs.fulltilde[:,n[k]]*lambdas**n[k])

    return newJs

def _test(N,nx):
    import time
    zvec = np.linspace(1, 1.0/N, N)
    dvec = np.linspace(1/N,2,N)
    qrot = 0.1
    tic = time.time()
    Js, out = cms(zvec,dvec,qrot,xlayers=nx)
    toc = time.time()
    print(Js[0:3])
    print("Elapsed time {} seconds".format(toc-tic))

if __name__ == '__main__':
    N = int(sys.argv[1])
    nx = int(sys.argv[2])
    _test(N, nx)
    sys.exit(0)
