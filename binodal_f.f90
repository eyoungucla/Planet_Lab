! binodal_f.f90 -- Fortran transcription of binodal_numpy.py (validated vs Planet_LAB subregular).
! Build:  python3 -m numpy.f2py -c -m binodal_f binodal_f.f90    (needs gfortran)
! Call in Python:  import binodal_f
!                  x1,x2,Tcrit,wsil,watm = binodal_f.binodal_mod.subregular_f(T, bulk, P)
!
! subregular_f(T[K], bulk[wt% H2], P[GPa]) -> x1(melt), x2(atm), Tcrit, wt_frac_sil, wt_frac_atm
! Reproduces _subregular_exact: 2x2 Newton (normal-T), golden-section (low-T), same guards.
!
! NOTE: everything is `double precision` (f2py-friendly; no named kind parameter), and only
! subregular_f is PUBLIC so f2py wraps just that one entry point.

module binodal_mod
  implicit none
  private
  public :: subregular_f

  double precision, parameter :: AA=622000.0d0, BB=-4950.0d0
  double precision, parameter :: TAU=4350.0d0, PPI=-35.0d0, RG=8.314d0
  double precision, parameter :: GR=0.6180339887498949d0
contains

  double precision function fac(T,P)
    double precision, intent(in) :: T,P
    fac = 1.0d0 - T/TAU + P/PPI
  end function fac

  double precision function Gmix(x,T,P)
    double precision, intent(in) :: x,T,P
    double precision :: x2
    if (x <= 0.0d0 .or. x >= 1.0d0) then
      Gmix = 1.0d300; return
    end if
    x2 = 1.0d0 - x
    Gmix = (AA*x + BB*x2)*x*x2*fac(T,P) + RG*T*(x*log(x) + x2*log(x2))
  end function Gmix

  double precision function mu(x,T,P)
    double precision, intent(in) :: x,T,P
    double precision :: f
    f = fac(T,P)
    mu = f*(2.0d0*x*(AA-2.0d0*BB) + 3.0d0*x*x*(BB-AA) + BB) + RG*T*(log(x) - log(1.0d0-x))
  end function mu

  double precision function d2G(x,T,P)
    double precision, intent(in) :: x,T,P
    double precision :: f
    f = fac(T,P)
    d2G = f*(6.0d0*x*(BB-AA) + 2.0d0*AA - 4.0d0*BB) + RG*T*(1.0d0/x + 1.0d0/(1.0d0-x))
  end function d2G

  double precision function d3G(x,T,P)
    double precision, intent(in) :: x,T,P
    double precision :: f
    f = fac(T,P)
    d3G = 6.0d0*(BB-AA)*f + RG*T*(-1.0d0/x**2 + 1.0d0/(1.0d0-x)**2)
  end function d3G

  ! --- critical temperature: 2x2 Newton on {d2G=0, d3G=0} in (x,T) ---
  double precision function crit_T(P)
    double precision, intent(in) :: P
    double precision :: x,T,f1,f2,df1dx,df1dT,df2dx,df2dT,det,dx,dT,s
    integer :: it
    x = 0.5d0; T = 3000.0d0
    do it = 1, 80
      f1 = d2G(x,T,P); f2 = d3G(x,T,P)
      df1dx = d3G(x,T,P)
      df1dT = (-1.0d0/TAU)*(6.0d0*x*(BB-AA) + 2.0d0*AA - 4.0d0*BB) + RG*(1.0d0/x + 1.0d0/(1.0d0-x))
      df2dx = RG*T*(2.0d0/x**3 + 2.0d0/(1.0d0-x)**3)
      df2dT = 6.0d0*(BB-AA)*(-1.0d0/TAU) + RG*(-1.0d0/x**2 + 1.0d0/(1.0d0-x)**2)
      det = df1dx*df2dT - df1dT*df2dx
      if (abs(det) < 1.0d-30) exit
      dx = -(df2dT*f1 - df1dT*f2)/det
      dT = -(df1dx*f2 - df2dx*f1)/det
      s = 1.0d0
      do while ((x+s*dx <= 1.0d-9 .or. x+s*dx >= 1.0d0-1.0d-9) .and. s > 1.0d-6)
        s = s*0.5d0
      end do
      x = x + s*dx; T = T + s*dT
      if (abs(f1) < 1.0d-6 .and. abs(f2) < 1.0d-9) exit
    end do
    crit_T = T
  end function crit_T

  ! --- coexistence residual {mu1-mu2, common tangent - mu1} ---
  subroutine coex_res(x1,x2,T,P,f1,f2)
    double precision, intent(in) :: x1,x2,T,P
    double precision, intent(out) :: f1,f2
    f1 = mu(x1,T,P) - mu(x2,T,P)
    f2 = (Gmix(x2,T,P) - Gmix(x1,T,P))/(x2-x1) - mu(x1,T,P)
  end subroutine coex_res

  ! --- damped 2x2 Newton (numeric Jacobian) with bound projection + backtracking ---
  subroutine coex_newton(x1,x2,T,P,lo1,lo2,hi1,hi2)
    double precision, intent(inout) :: x1,x2
    double precision, intent(in) :: T,P,lo1,lo2,hi1,hi2
    double precision :: f1,f2,f1a,f2a,f1b,f2b,h1,h2,j11,j21,j12,j22,det,dx1,dx2,s,r0,g1,g2,aa2,bb2
    integer :: it,k
    logical :: found
    do it = 1, 100
      call coex_res(x1,x2,T,P,f1,f2)
      if (f1*f1 + f2*f2 < 1.0d-18) exit
      h1 = max(1.0d-9, 1.0d-7*abs(x1)); h2 = max(1.0d-9, 1.0d-7*abs(x2))
      call coex_res(x1+h1,x2,T,P,f1a,f2a)
      call coex_res(x1,x2+h2,T,P,f1b,f2b)
      j11 = (f1a-f1)/h1; j21 = (f2a-f2)/h1
      j12 = (f1b-f1)/h2; j22 = (f2b-f2)/h2
      det = j11*j22 - j12*j21
      if (abs(det) < 1.0d-30) exit
      dx1 = -(j22*f1 - j12*f2)/det
      dx2 = -(j11*f2 - j21*f1)/det
      s = 1.0d0; r0 = f1*f1 + f2*f2; found = .false.
      do k = 1, 40
        aa2 = min(max(x1+s*dx1, lo1), hi1)
        bb2 = min(max(x2+s*dx2, lo2), hi2)
        if (bb2 > aa2) then
          call coex_res(aa2,bb2,T,P,g1,g2)
          if (g1*g1 + g2*g2 < r0) then
            x1 = aa2; x2 = bb2; found = .true.; exit
          end if
        end if
        s = s*0.5d0
      end do
      if (.not. found) exit
    end do
  end subroutine coex_newton

  ! --- golden-section: argmin of (mu(z)-mu1)^2 over [a,b] (inner x2 solve) ---
  double precision function inner_x2(mu1,T,P,a_in,b_in)
    double precision, intent(in) :: mu1,T,P,a_in,b_in
    double precision :: a,b,c,d,fc,fd
    integer :: it
    a = a_in; b = b_in
    c = b - GR*(b-a); d = a + GR*(b-a)
    fc = (mu(c,T,P)-mu1)**2; fd = (mu(d,T,P)-mu1)**2
    do it = 1, 200
      if (b-a < 1.0d-8) exit
      if (fc < fd) then
        b = d; d = c; fd = fc; c = b - GR*(b-a); fc = (mu(c,T,P)-mu1)**2
      else
        a = c; c = d; fc = fd; d = a + GR*(b-a); fd = (mu(d,T,P)-mu1)**2
      end if
    end do
    inner_x2 = 0.5d0*(a+b)
  end function inner_x2

  ! --- low-T objective for outer x1 minimization ---
  double precision function objective(x1,T,P,Tfrac)
    double precision, intent(in) :: x1,T,P,Tfrac
    double precision :: mu1,x2low,x2,G1,G2
    if (x1 <= 0.0d0 .or. x1 >= 1.0d0) then
      objective = 1.0d6; return
    end if
    if (d2G(x1,T,P) <= -10.0d0) then
      objective = 1.0d5; return
    end if
    mu1 = mu(x1,T,P)
    x2low = max(0.95d0, 0.999d0 - 0.01d0*(0.7d0 - Tfrac)/0.7d0)
    x2 = inner_x2(mu1,T,P,x2low,0.999999d0)
    if (d2G(x2,T,P) <= -10.0d0) then
      objective = 1.0d5; return
    end if
    G1 = Gmix(x1,T,P); G2 = Gmix(x2,T,P)
    objective = ((G2-G1)/(x2-x1) - mu1)**2
  end function objective

  subroutine low_T(T,P,x1,x2,Tcrit)
    double precision, intent(in) :: T,P
    double precision, intent(out) :: x1,x2,Tcrit
    double precision :: Tfrac,a,b,c,d,fc,fd
    integer :: it
    Tcrit = crit_T(P); Tfrac = T/Tcrit
    x2 = 1.0d0 - 1.0d-7
    a = 1.0d-6; b = 0.3d0
    c = b - GR*(b-a); d = a + GR*(b-a)
    fc = objective(c,T,P,Tfrac); fd = objective(d,T,P,Tfrac)
    do it = 1, 200
      if (b-a < 1.0d-8) exit
      if (fc < fd) then
        b = d; d = c; fd = fc; c = b - GR*(b-a); fc = objective(c,T,P,Tfrac)
      else
        a = c; c = d; fc = fd; d = a + GR*(b-a); fd = objective(d,T,P,Tfrac)
      end if
    end do
    x1 = 0.5d0*(a+b)
  end subroutine low_T

  ! ================= public entry =================
  subroutine subregular_f(T_in, bulk_in, P, x1, x2, Tcrit, wsil, watm)
    double precision, intent(in) :: T_in, bulk_in, P
    double precision, intent(out) :: x1, x2, Tcrit, wsil, watm
    double precision :: T, bulk, Tfrac, xi1, xi2, lo1, lo2, hi1, hi2, w1, w2, delta, rr1, rr2
    Tcrit = crit_T(P)
    if (T_in /= T_in) then           ! NaN guard
      T = 1500.0d0
    else
      T = max(T_in, 1500.0d0)
    end if
    bulk = max(bulk_in, 1.0d-3)
    if (T/Tcrit < 0.7d0) then
      call low_T(T, P, x1, x2, Tcrit)
    else
      Tfrac = T/Tcrit
      if (Tfrac < 0.73d0) then
        lo1 = 1.0d-8; hi1 = 0.25d0; lo2 = 0.999d0; hi2 = 0.999995d0
      else
        lo1 = 1.0d-6; hi1 = 0.999d0; lo2 = 1.0d-6; hi2 = 0.99999d0
      end if
      if (Tfrac <= 0.96d0) then
        xi1 = 0.001d0; xi2 = 0.999d0
      else if (Tfrac > 0.9975d0) then
        xi1 = 0.6d0; xi2 = 0.85d0
      else
        xi1 = 0.1d0; xi2 = 0.9d0
      end if
      xi1 = min(max(xi1, lo1+1.0d-10), hi1-1.0d-10)
      xi2 = min(max(xi2, lo2+1.0d-10), hi2-1.0d-10)
      x1 = xi1; x2 = xi2
      call coex_newton(x1, x2, T, P, lo1, lo2, hi1, hi2)
    end if
    ! lever rule (MW1=2.02, MW2=100.3)
    rr1 = (2.02d0/100.3d0)*(x1/(1.0d0-x1)); w1 = 100.0d0*(rr1/(rr1+1.0d0))
    rr2 = (2.02d0/100.3d0)*(x2/(1.0d0-x2)); w2 = 100.0d0*(rr2/(rr2+1.0d0))
    delta = w2 - w1
    wsil = (w2 - bulk)/delta
    watm = (bulk - w1)/delta
  end subroutine subregular_f

end module binodal_mod
