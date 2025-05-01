# -*- coding: utf-8 -*-
# Author: Vladislav S. Yakovlev

# A set of useful functions and classes

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate

class AtomicUnits:
    meter = 5.2917720859e-11 # atomic unit of length in meters
    nm = 5.2917720859e-2 # atomic unit of length in nanometres
    second = 2.418884328e-17 # atomic unit of time in seconds
    fs = 2.418884328e-2 # atomic unit of time in femtoseconds
    Joule = 4.359743935e-18 # atomic unit of energy in Joules
    eV = 27.21138383 # atomic unit of energy in electronvolts
    Volts_per_meter = 5.142206313e+11 # atomic unit of electric field in V/m
    Volts_per_Angstrom = 51.42206313 # atomic unit of electric field in V/Angström
    speed_of_light = 137.036 # vacuum speed of light in atomic units
    Coulomb = 1.60217646e-19 # atomic unit of electric charge in Coulombs
    PW_per_cm2_au = 0.1553661415 # PW/cm^2 in atomic units


def nextpow2(i):
    """ Compute the exponent for the smallest power of 2 larger than the input. """
    return int(np.ceil(np.log2(i)))


def soft_window(x_grid, x_begin, x_end):
    """ Compute a soft window.

    Given a vector 'x_grid' and two numbers ('x_begin'and 'x_end'),
    the function returns a vector that contains a 'soft window',
    which gradually changes from 1 at x=x_begin to 0 at x=x_end,
    being constant outside of this range. The value of 'x_begin'
    is allowed to be larger than that of 'x_end'.
    """
    window = np.zeros(len(x_grid))
    # determine the indices where the window begins and ends
    x_min = min((x_begin, x_end))
    x_max = max((x_begin, x_end))
    u = np.nonzero(x_grid < x_min)[0]
    i1 = min((u[-1] + 1, len(x_grid))) if len(u) > 0 else 0
    u = np.nonzero(x_grid > x_max)[0]
    i2 = u[0] if len(u) > 0 else len(x_grid)
    # evaluate the window function
    if x_begin <= x_end:
        window[:i1] = 1
        if i2 > i1:
            window[i1:i2] = \
                np.cos(np.pi/2.0 * (x_grid[i1:i2]-x_min) / (x_max - x_min))**2
    else:
        window[i2:] = 1
        if i2 > i1:
            window[i1:i2] = \
                np.sin(np.pi/2.0 * (x_grid[i1:i2]-x_min) / (x_max - x_min))**2
    return window


def SignificantPart1(A, threshold=1e-8):
    """ Return a tuple (i1,i2) such that A[:i1] and A[i2:] are very small """
    abs_A = np.abs(A)
    i0 = np.argmax(abs_A)
    A_max = abs_A[i0]
    Y = np.nonzero(abs_A[:i0]>=threshold*A_max)[0]
    i1 = Y[0] if len(Y)>0 else i0
    Y = np.nonzero(abs_A[i0:]>=threshold*A_max)[0]
    i2 = i0 + Y[-1] + 1 if len(Y)>0 else i0 + 1
    return (i1, i2)


def SignificantPart2(A, threshold=1e-8):
    """ Return a tuple (i1,i2) such that none of the elements A[i1:i2] is small """
    abs_A = np.abs(A)
    i0 = np.argmax(abs_A)
    A_max = abs_A[i0]
    Y = np.nonzero(abs_A[:i0]<threshold*A_max)[0]
    i1 = Y[-1] if len(Y)>0 else 0
    Y = np.nonzero(abs_A[i0:]<threshold*A_max)[0]
    i2 = i0 + Y[0] + 1 if len(Y)>0 else len(A)
    return (i1, i2)


def Fourier_filter(Y, dt, spectral_window, periodic=False):
    """ Apply a Fourier filter to data.

    Given a function Y(t) (or a set of functions) on an equidistant
    grid, this function evaluates the Fourier transform, applies the
    spectral window to suppress some of the frequency components,
    and returns the inverse Fourier transform. This function does not
    apply zero padding--it works fastest if the number of temporal
    nodes is a power of two.
 
    Parameters
    ----------
    Y : a matrix (N_t,N) where each column specifies some function of time;
        it's also possible to pass the function a 1D array, in which case
        a 1D array will be returned;
    dt : a time step (a scalar)
    spectral_window : a matrix (N_window, 2) where the first column contains
        circular frequencies in the ascending order, and the second contains
        a spectral window function, W(omega) the spectra of the given functions
        will be multiplied with this window
    periodic : if 'true', then the given functions are assumed to be
        periodic otherwise, the function takes some precaution to avoid
        artifacts induced by the fact that discrete Fourier transform
        implicitly assumes periodic data
    """
    if len(Y.shape) == 1:
        output_is_1D = True
        Y = Y.copy()
        N_t = Y.size
        N = 1
        Y = Y.reshape((N_t, 1))
    else:
        output_is_1D = False
        N_t = Y.shape[0]
        N = Y.shape[1]
    N_window = spectral_window.shape[0]
    if N_window == 0:
        return Y
    # Fourier transform the input data
    if periodic:
        F = np.fft.fftshift(np.fft.fft(Y, axis=0), axes=0)
    else:
        F = np.fft.fftshift(np.fft.fft(np.vstack((Y, Y[::-1, :])), axis=0), axes=0)
        N_t *= 2
    dw = 2*np.pi / (N_t * dt)
    w_grid = dw * (1 - np.ceil(N_t/2.) + np.arange(N_t).reshape((N_t,1)))
    # apply the spectral filter
    W = np.interp(np.abs(w_grid), spectral_window[:,0], spectral_window[:,1], left=1.0, right=0.0)
    ## plt.figure()
    ## plt.plot(w_grid, np.abs(F[:, 0]))
    ## plt.plot(w_grid, W[:,0] * np.abs(F[:, 0]))
    ## plt.show()
    ## import sys
    ## sys.exit(0)
    F *= W.reshape((N_t, 1))
    # inverse Fourer transform
    Y1 = np.fft.ifft(np.fft.ifftshift(F, axes=0), axis=0)
    # if the original functions were extended to make them periodic, drop that artificial part
    if not periodic:
        Y1 = Y1[:N_t//2, :]
    # if the input data was real, make the output data real as well
    if np.all(np.isreal(Y)):
        Y1 = Y1.real
    if output_is_1D:
        Y1 = Y1.reshape(Y1.size)
    return Y1


def PolyfitWithWeights(x, y, w, deg):
    """ Do the least-squares polynomial fit with weights.

    Parameters
    ----------
    x : vector of sample points
    y : vector or 2D array of values to fit
    w : vector of weights
    deg : degree of the fitting polynomial

    Do a best fit polynomial of degree 'deg' of 'x' to 'y'.  Return value is a
    vector of polynomial coefficients [pn ... p1 p0].

    pn*x0^n + ... +  p1*x0 + p0 = y1
    pn*x1^n + ... +  p1*x1 + p0 = y1
    pn*x2^n + ... +  p1*x2 + p0 = y2
    .....
    pn*xk^n + ... +  p1*xk + p0 = yk
    """
    deg += 1
    a = np.empty((deg,deg), np.float64)
    w2 = w * w
    for i in range(deg):
        for j in range(i,deg): a[i,j] = np.sum(w2 * x**(i+j))
    for i in range(deg):
        for j in range(i): a[i,j] = a[j,i]
    b = np.empty(deg,float)
    for i in range(deg): b[i] = np.sum(w2 * y * x**i)
    solution = scipy.linalg.solve(a, b)
    return solution[::-1]


def FourierTransform(t, Y, omega=None, is_periodic=False, t0=None):
    r""" Apply the FFT in an easy-to-use way.

    Parameters
    ----------
    t (N_t): time discretization of Y(t); the array must be sorted in
        ascending order;
    Y (N_t, N_data): time-dependent data to Fourier transform;
    omega (N_omega): circular frequencies; the array must be sorted in
        ascending order;
    is_periodic: if 'true', then Y(t) is assumed to be periodic, and the
        time step is expected to be constant, so that Y(t[-1]+dt) = Y(t[0]);
        if 'false', the function takes some precaution to avoid artifacts
        induced by the fact that discrete Fourier transform implicitly
        assumes periodic data;
    t0 (scalar or an array of length N_data): if the input data represents
        a pulse that is not centered in the middle of its time window, then
        knowing the center of the pulse helps avoid interpolation artifacts;
        this parameter specifies the central time of each pulse;

    Returns
    -------
    If omega is None, the function returns a tuple of two elements: the
    Fourier-transformed 'Y' and an array of circular frequencies;
    otherwise, the function returns F[Y] at the specified frequencies:
        $ F[Y](\omega) = \int_{-\infty}^\infty dt Y(t) \exp(i \omega t). $
    """
    max_N_fft = 2**20
    N_t = len(t)
    shape_length = len(Y.shape)
    if shape_length == 1:
        Y = Y.reshape((N_t, 1))
    N_data = Y.shape[1]
    assert(N_t > 1)
    dt = np.min(np.diff(t))
    if is_periodic:
        T = t[-1] - t[0] + np.mean(np.diff(t)) # period assuming a constant step
    if omega is None:
        dw = 2 * np.pi / (dt * N_t)
    else:
        if len(omega) == 1: # no need to do FFT
            integrand = Y * np.exp(1j * omega * t.reshape((N_t, 1)))
            result = np.trapz(integrand, t, axis=0)
            if is_periodic:
                result += 0.5 * dt * (integrand[0] + integrand[-1])
            return result
        dw = np.min(np.diff(omega))
    # FFT discretization
    if is_periodic:
        N_fft = N_t
        while 2 * np.pi / (dt * N_fft) > 1.1 * dw and N_fft < max_N_fft:
            N_fft *= 2
    else:
        N_fft = 2**int(round(np.log(2 * np.pi / (dt * dw)) / np.log(2.0)))
        while (N_fft-1)*dt < t[-1] - t[0] and N_fft < max_N_fft:
            N_fft *= 2
        if not (omega is None):
            while (omega[-1] > np.pi / dt * (1 - 2.0/N_fft) or \
                -omega[0] > np.pi / dt) and N_fft < max_N_fft:
                dt /= 2.0
                N_fft *= 2
    dw = 2 * np.pi / (dt * N_fft)
    if t0 is None:
        tc = 0.5 * (t[0] + t[-1])
        i0 = np.argmax(t <= tc) # this element will be treated as t=0
        t0 = t[i0] * np.ones(N_data)
    t_grid = dt * np.fft.ifftshift(np.arange(N_fft) - N_fft//2)
    Z = np.zeros((N_fft, N_data), dtype=Y.dtype)
    for j in range(N_data):
        if np.isscalar(t0):
            tt = t - t0
        else:
            tt = t - t0[j]
        if is_periodic:
            Z[:,j] = np.interp(t_grid, tt, Y[:,j], period=T)
        else:
            Z[:,j] = np.interp(t_grid, tt, Y[:,j], left=0.0, right=0.0)
    if np.isscalar(t0):
        t0 = t0 * np.ones((1, N_data))
    else:
        t0 = np.reshape(t0, (1, N_data))
    # FFT
    Z = np.fft.fftshift(np.fft.ifft(Z, axis=0), axes=0) * (N_fft * dt)
    w_grid = dw * (np.arange(N_fft) - N_fft//2)
    if omega is None:
        return (Z * np.exp(1j * t0 * w_grid.reshape((N_fft, 1))), w_grid)
        ## return (Z[N_fft//2:,:], w_grid[N_fft//2:])
    # interpolate the result
    result = np.zeros((len(omega), N_data), dtype=np.complex128)
    for j in range(N_data):
        result[:,j] = np.interp(omega, w_grid, Z[:,j], left=0.0, right=0.0)
    # correct for t0
    result = result * np.exp(1j * t0 * omega.reshape((len(omega), 1)))
    if shape_length == 1:
        result = result.reshape(len(omega))
    return result


def InverseFourierTransform(omega, Y, t_array=None, is_periodic=False, omega0=None):
    r""" Apply the FFT in an easy-to-use way.

    Parameters
    ----------
    omega (N_omega): an array of circular frequencies specifying
        the time discretization of Y(omega); the array must be sorted in
        ascending order;
    Y (N_omega, N_data): frequency-dependent data to Fourier transform;
    t_array (N_t): either None or an array of time nodes; the array must be
        sorted in ascending order;
    is_periodic: if 'true', then Y(omega) is assumed to be periodic, and the
        frequency step is expected to be constant, so that Y(omega[-1]+dw) = Y(omega[0]);
        if 'false', the function takes some precaution to avoid artifacts
        induced by the fact that discrete Fourier transform implicitly
        assumes periodic data;
    omega0 (scalar or an array of length N_data): if the input data is not centered in
        the middle of its frequency window, then knowing the central frequency helps avoid
        interpolation artifacts; this parameter specifies the central frequency of each spectrum
        (note that this parameter should be set to zero if the outcome of the inverse Fourier
        transform is supposed to be a real-valued function of time);

    Returns
    -------
    If t_array is None, the function returns a tuple of two elements: the
    Fourier-transformed 'Y' and an array of time nodes; otherwise, the function
    returns F[Y] at the specified frequencies:
        $ F^{-1}[Y](t) = 1 / (2 \pi) \int_{-\infty}^\infty d\omega Y(omega) \exp(-i t omega). $
    """
    if t_array is None:
        Z, t_grid = FourierTransform(omega, Y, None, is_periodic, omega0)
        Z = Z[::-1] / (2 * np.pi)
        t_grid = -t_grid[::-1]
        return Z, t_grid
    return FourierTransform(omega, Y, -t_array[::-1], is_periodic, omega0)[::-1] / (2 * np.pi)
    ## max_N_fft = 2**20
    ## N_omega = len(omega)
    ## shape_length = len(Y.shape)
    ## if shape_length == 1:
    ##     Y = Y.reshape((N_omega, 1))
    ## N_data = Y.shape[1]
    ## assert(N_omega > 1)
    ## d_omega = np.min(np.diff(omega))
    ## if is_periodic:
    ##     period = omega[-1] - omega[0] + np.mean(np.diff(omega)) # period assuming a constant step
    ## if t_array is None:
    ##     dt = 2 * np.pi / (d_omega * N_omega)
    ## else:
    ##     if len(t_array) == 1: # no need to do FFT
    ##         integrand = Y * np.exp(-1j * t_array * omega.reshape((N_omega, 1)))
    ##         result = np.trapz(integrand, omega, axis=0) / (2 * np.pi)
    ##         if is_periodic:
    ##             result += 0.5 * d_omega * (integrand[0] + integrand[-1])
    ##         return result
    ##     dt = np.min(np.diff(t_array))
    ## # FFT discretization
    ## if is_periodic:
    ##     N_fft = N_omega
    ##     while 2 * np.pi / (d_omega * N_fft) > 1.1 * dt and N_fft < max_N_fft:
    ##         N_fft *= 2
    ## else:
    ##     N_fft = 2**int(round(np.log(2 * np.pi / (d_omega * dt)) / np.log(2.0)))
    ##     while (N_fft-1)*d_omega < omega[-1] - omega[0] and N_fft < max_N_fft:
    ##         N_fft *= 2
    ##     if not (t_array is None):
    ##         while (t_array[-1] > np.pi / d_omega * (1 - 2.0/N_fft) or \
    ##             -t_array[0] > np.pi / d_omega) and N_fft < max_N_fft:
    ##             d_omega /= 2.0
    ##             N_fft *= 2
    ## dt = 2 * np.pi / (d_omega * N_fft)
    ## omega_central = 0.5 * (omega[0] + omega[-1])
    ## i0 = np.argmax(omega >= omega_central) # this element will be treated as omega=0
    ## omega0 = omega[i0]
    ## omega_shifted = omega - omega0
    ## omega_grid = d_omega * np.fft.ifftshift(np.arange(N_fft) - N_fft//2)
    ## Z = np.zeros((N_fft, N_data), dtype=Y.dtype)
    ## if is_periodic:
    ##     for j in range(N_data):
    ##         Z[:,j] = np.interp(omega_grid, omega_shifted, Y[:,j], period=period)
    ## else:
    ##     for j in range(N_data):
    ##         Z[:,j] = np.interp(omega_grid, omega_shifted, Y[:,j], left=0.0, right=0.0)
    ## # FFT
    ## Z = np.fft.ifftshift(np.fft.fft(Z, axis=0), axes=0) * (d_omega / (2 * np.pi))
    ## t_grid = dt * (np.arange(N_fft) - N_fft//2)
    ## if t_array is None:
    ##     return (Z * np.exp(1j * omega0 * t_grid.reshape((N_fft, 1))), t_grid)
    ##     ## return (Z[N_fft//2:,:], t_grid[N_fft//2:])
    ## # interpolate the result
    ## result = np.zeros((len(t_array), N_data), dtype=np.complex)
    ## for j in range(N_data):
    ##     result[:,j] = np.interp(t_array, t_grid, Z[:,j], left=0.0, right=0.0)
    ## # correct for omega0
    ## result = result * np.exp(1j * omega0 * t_array.reshape((len(t_array), 1)))
    ## if shape_length == 1:
    ##     result = result.reshape(len(t_array))
    ## return result


def find_zero_crossings(X, Y):
    """ Find all the zero crossings: y(x) = 0

Parameters
----------
X : a 1D float array of x-values sorted in ascending order;
    the array may not have identical elements;
Y : a float array of the same shape as X;

Returns
-------
out : an array of x-values where the linearly interpolated function y(x)
has zero values (an empty list if there are no zero crossings).
"""
    Z = Y[:-1] * Y[1:]
    out = []
    for i in np.nonzero(Z <= 0)[0]:
        if Z[i] == 0:
            if Y[i] == 0:
                out.append(X[i])
        else:
            # there is a zero crossing between X[i] and X[i+1]
            out.append((X[i]*Y[i+1] - X[i+1]*Y[i]) / (Y[i+1] - Y[i]))
    return np.array(out)


def find_extrema_positions(X, Y):
    """ Find all the extrema of the given function

Parameters
----------
X : a 1D float array of x-values sorted in ascending order;
    the array may not have identical elements;
Y : a float array of the same shape as X;

Returns
-------
out : an array of x-values where the linearly interpolated y'(x)
has zero values (an empty list if there are no such x-values).
"""
    dY_dX = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])
    return find_zero_crossings(0.5 * (X[1:] + X[:-1]), dY_dX)

def minimize_imaginary_parts(Z_array):
    """ Multiplies Z_array with a phase factor to make Z_array possibly real-valued

    The function returns Z_array*np.exp(1j*phi) evaluated for a rotation angle phi that
    minimizes np.sum(np.imag(np.exp(1j*phi) * Z_vector)**2).
    """
    numerator = 2 * np.sum(Z_array.real * Z_array.imag)
    denominator = np.sum(Z_array.imag**2 - Z_array.real**2)
    phi = 0.5 * np.arctan2(numerator, denominator)
    y1 = np.sum(np.imag(np.exp(1j*phi) * Z_array)**2)
    y2 = np.sum(np.imag(np.exp(1j*(phi + 0.5*np.pi)) * Z_array)**2)
    if y2 < y1:
        phi += 0.5*np.pi
    phi -= np.pi * np.round(phi / np.pi)
    return Z_array * np.exp(1j * phi)

def integrate_oscillating_function(X, f, phi, phase_step_threshold=1e-3):
    r""" The function evaluates \int dx f(x) exp[i phi(x)] using an algorithm
    suitable for integrating quickly oscillating functions.
    
    Parameters
    ----------
    X: a vector of sorted x-values;
    f: either a vector or a matrix where each column contains
        the values of a function f(x);
    phi: either a vector or a matrix where each column contains
        the values of a real-valued phase phi(x);
    phase_step_threshold (float): a small positive number; the formula that
        approximates the integration of an oscillating function over a
        small interval contains phi(x+dx)-phi(x) in the denominator;
        this parameter prevents divisions by very small numbers.
    
    Returns
    -------
    result: a row vector where elements correspond to the columns in
      'f' and 'phi'.
    """
    
    # check that the input data is OK
    assert(X.shape[0] == f.shape[0])
    assert(X.shape[0] == phi.shape[0])
    assert(np.all(np.imag(phi)) == 0)
    # evaluate the integral(s)
    dx = X[1:] - X[:-1]
    f1 = f[:-1, ...]
    f2 = f[1:, ...]
    df = f2 - f1
    dphi = phi[1:, ...] - phi[:-1, ...]
    s = np.ones((f.ndim), dtype=np.int)
    s[0] = dx.size
    Z = dx.reshape(s) * np.exp(0.5j*(phi[1:, ...] + phi[:-1, ...]))
    s = (np.abs(dphi) < phase_step_threshold)
    if np.any(s):
        Z[s] = Z[s] * (0.5 * (f1[s] + f2[s]) + 0.125j * dphi[s] * df[s])
    s = np.logical_not(s)
    if np.any(s):
        exp_term = np.exp(0.5j * dphi[s])
        Z[s] = Z[s] / dphi[s]**2 * (exp_term * (df[s] - 1j*f2[s]*dphi[s]) -
            (df[s] - 1j*f1[s]*dphi[s]) / exp_term)
    return np.sum(Z, axis=0)


def permittivity_from_delta_polarization(dt, P_delta, omega_array,
    momentum_relaxation_rate=0,  dephasing_time=None,
    disregard_drift_current=False, allow_for_linear_displacement=True):
    r""" Evaluate the permittivity from the polarization induced by E(t) = delta(t)

If the polarization response induced by a delta spike smoothly approached zero
by the end of the P_delta array, the task of evaluating the permittivity would be
as simple as Fourier transforming P_delta. Unfortunately, it is not so simple.
If there are free charge carriers, the interaction of a delta spike with a solid
results in a drift residual current, which corresponds to a steadily growing
polarization, while interband coherences make the polarization oscillate. The
oscillations can be damped by assuming some decoherence, but the linearly growing
polarization is harder to handle. This function solves the problem by linearly
extrapolating the given polarization response and analytically integrating to infinity.

Parameters
----------
dt (scalar) : the time step (in atomic units) of the grid, on which J(t) is given;
P_delta (1D array) : the polarization response (in atomic units) induced by a delta
    spike of the electric field: E(t) = delta(t); the first element in this array
    must correspond to t=dt because P_delta(0) is necessarily zero as long as the
    electric current has a finite magnitude;
omega_array (1D array) : circular frequencies (in atomic units), for which the
    permittivity of the medium will be calculated; all frequencies must be nonzero;
momentum_relaxation_rate (scalar) : if nonzero, model momentum relaxation within the
    Drude-model (the input data must not contain any effect of momentum relaxation);
dephasing_time (scalar or None) : if this parameter is neither None nor zero, then
    an exponential decay will be enforced on the coherent dipole oscillations in P_delta;
    the rate of this decay is 1/dephasing_time;
disregard_drift_current (boolean) : if True, the linear component of the
    polarization, which is due to the residual drift current and linear displacement
    of charge carriers, will have no effect on the result; (if there are no free charge
    carriers, a weak probe pulse cannot induce a residual drift current, but, in a
    simulation, a parasitic drift current and a parasitic displacement may appear);
allow_for_linear_displacement (boolean) : if True, the general linear fit is used to
    approximate the residual linear polarization: P(t) \approx J_drift * t + P_offset;
    if P_offset is nonzero, then a weak probe pulse can displace charge carriers in
    addition to inducing some residual drift current (in other words, the drift
    current will not account for all the displacement); if this parameter is False,
    then a more restrictive model is used for fitting: P(t) \approx J_drift * t.


Returns
-------
permittivity : a complex array of the same shape as omega_array; each element
    of this array contains the complex permittivity:
    $\epsilon(\omega) = 1 + 4 \pi \chi(\omega)$,
    $P(\omega) = \chi(\omega) E(\omega)$.
"""
    assert(np.all(omega_array != 0))
    N_t = 1 + P_delta.size
    t_grid = dt * np.arange(N_t)
    t_max = t_grid[-1]
    P = np.zeros(N_t)
    P[1:] = P_delta # P[0] == 0
    permittivity = np.zeros(omega_array.shape, dtype=np.complex)
    # separate P(t) into a linear function of time and the rest
    nn = t_grid.size // 2
    if allow_for_linear_displacement:
        polynomial_coefficients = np.polyfit(t_grid[nn:], P[nn:], 1)
        J_drift = polynomial_coefficients[0]
        P_offset = polynomial_coefficients[1]
    else:
        J_drift = np.sum(P[nn:] * t_grid[nn:]) / np.sum(t_grid[nn:]**2)
        P_offset = 0
    P -= J_drift * t_grid
    # attenuate the coherent oscillations of the polarization
    if (dephasing_time is None) or (dephasing_time == 0):
        window = soft_window(t_grid, 0.5 * t_max, t_max)
    else:
        window = np.exp(- t_grid / dephasing_time) * soft_window(t_grid, 0.5 * t_max, t_max)
    P = P_offset + window * (P - P_offset)
    # evaluate the permittivity at each frequency
    for i_omega, omega in enumerate(omega_array):
        chi = integrate_oscillating_function(t_grid, P, omega * t_grid)
        chi += P_offset * 1j * np.exp(1j * omega * t_max) / omega
        if not disregard_drift_current:
            chi -= J_drift / (omega * (omega + 1j * momentum_relaxation_rate))
        permittivity[i_omega] = 1 + 4*np.pi * chi
    return permittivity
