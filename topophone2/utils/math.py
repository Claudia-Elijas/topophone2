"""
.. container:: module-intro

    The ``topophone2.utils.math`` module provides mathematical convenience functions:

        :func:`bessel_function` - Computes the Bessel function of the first kind or the spherical Bessel function.

        :func:`legendre_polynomial` - Computes the associated Legendre polynomial.

"""

#________ IMPORT ________#

#__ Numerical __
import numpy as np

#__ Harmonic functions __
from scipy.special import jv, jvp, spherical_jn, lpmv


#__ Bessel functions __
def bessel_function(
    argument : float|np.ndarray,
    order    : int = 0,
    shape    : str = 'cylindrical',
    derivative : bool = False
):
    r"""
    
    Input
    -----
    argument : ``float`` or ``numpy.ndarray``
        Argument to Bessel function

    order : ``int`` (default ``0``)
        Order of Bessel function (can broadcast)

    shape : ``str`` (default ``'cylindrical'``)
        Shape of Bessel function (``'cylindical'`` for `J`, ``'spherical'`` for `j`)

    derivative : ``bool`` (default `False`)
        Return derivative of the Bessel function

    Output
    ------
    ``numpy.ndarray``
        Function output


    .. math::

        J_n(x) = \sum_{m=0}^{\infty} \frac{(-1)^m}{m! \Gamma(m+n+1)} \left(\frac{x}{2}\right)^{2m+n}



    """

    #__ Cylindical Bessel function __
    if (shape == 'cylindrical'):
        if not derivative: return jv(order,  argument)
        elif derivative:   return jvp(order, argument, n=1)

    #__ Spherical Bessel function __
    elif (shape == 'spherical'):
        return spherical_jn(order, argument, derivative=derivative)

    #__ Error for `shape` __
    else: print("**shape** must be ``'cylindrical'`` or ``'spherical'``")

#__ Legendre polynomial __
def legendre_polynomial(
    argument : np.ndarray,
    degree   : int,
    order    : int,
    condon_shortley : bool = False
):
    r"""
    
    Function to compute the associated Legendre polynomial $P_n^m(x)$ using ``scipy.special.lpmv``.

    Input
    -----
    argument : ``numpy.ndarray``
        Argument to polynomial

    degree : ``int``
        Degree of polynomial ($n$)

    order : ``int``
        Order of polynomial ($m$)

    condon_shortley : ``bool`` (default ``False``)
        Include the Condon-Shortley phase $(-1)^m$


    .. math::

        P_n^m(x) = (-1)^m \frac{(1-x^2)^{m/2}}{2^n n!} \frac{d^{n+m}}{dx^{n+m}} (x^2 - 1)^n


    """

    #__ Prefactor __
    if condon_shortley: prefactor = 1
    else: prefactor = (-1)**order

    #__ Return output __
    return prefactor * lpmv(order, degree, argument)

#__ Density function parameters __
def density_params(
    xi    : np.ndarray,
    sigma : np.ndarray = None,
    corr  : np.ndarray = None,
    alpha : np.ndarray = None
):
    """
    Construct summary quantities for density functions (point, normal, skewnormal) in cylindrical/spherical coordinates.

    Input
    -----
    xi : ``numpy.ndarray``
        Location vector [s.m-1]
        (model, direction)

    sigma : ``numpy.ndarray`` (optional)
        Scale vector [s.m-1]
        (model, direction)

    corr : ``numpy.ndarray`` (optional)
        Correlation matrix off-diagonal (upper-triangle order)
        (model, direction pair)

    alpha : ``numpy.ndarray`` (optional)
        Shape vector [s.m-1]
        (model, direction)

    Output
    ------
    wd : ``numpy.ndarray``
        Skewness-direction vector [s.m-1]
        (model, direction)

    mean : ``numpy.ndarray``
        Mean vector [s.m-1]
        (model, direction)

    xi_circ : ``numpy.ndarray``
        Location vector in cylindrical/spherical coordinates [s.m-1, deg from North[, deg from horizontal]]
        (model, direction)

    mean_circ : ``numpy.ndarray``
        Mean vector in cylindrical/spherical coordinates [s.m-1, deg from North[, deg from horizontal]]
        (model, direction)

    wd_circ : ``numpy.ndarray``
        Skewness-direction vector in cylindrical/spherical coordinates about xi [s.m-1, deg from North[, deg from horizontal]]
        (model, direction)
    """

    #__ np.einsum subscripts __
    #__ [i] model
    #__ [l],[p] direction

    #__ Instantiate outputs __
    wd, mean, xi_circ, mean_circ, wd_circ = [None]*5

    #__ Construct correlation matrix __
    if corr is not None:
        N_model, N_dir     = xi.shape
        _iu0_dir, _iu1_dir = np.triu_indices(N_dir, k=1)
        Omega_z = np.ones((N_model, N_dir, N_dir), dtype=float)
        Omega_z[:,_iu0_dir,_iu1_dir] = corr
        Omega_z[:,_iu1_dir,_iu0_dir] = corr

    #__ Calculate skewness direction vector __
    if alpha is not None:
        Oza   = np.einsum('ilp,ip->il', Omega_z, alpha, optimize=True)
        N     = 1.0 / np.sqrt(1.0 + np.einsum('il,il->i', alpha, Oza, optimize=True))
        delta = N[:,None] * Oza

        #__ Scale skewness-direction vector __
        wd = sigma * delta

    #__ Calculate mean __
    if alpha is not None:
        mean =   xi + wd * np.sqrt(2/np.pi)
    else: mean = xi

    #__ Cartesian -> cylindrical/spherical __
    def to_circ(v):
        r   = np.linalg.norm(v, axis=1)
        phi = np.degrees(np.arctan2(v[:,0], v[:,1])) % 360
        if (N_dir == 2): return np.stack([r, phi], axis=1)
        theta = np.degrees(np.arctan2(v[:,2], np.hypot(v[:,0], v[:,1])))
        return np.stack([r, phi, theta], axis=1)

    #__ Project to cylindrical/spherical coordinates __
    xi_circ   = to_circ(xi)
    mean_circ = to_circ(mean)

    #__ Local orthonormal basis at xi __
    if alpha is not None:
        r     = xi_circ[:,0]
        phi   = np.radians(xi_circ[:,1])
        theta = np.radians(xi_circ[:,2]) if (N_dir == 3) else np.zeros(N_model)
        ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
        if (N_dir == 2):
            e_r   = np.stack([sp, cp], axis=1)
            e_phi = np.stack([cp, -sp], axis=1)
        else:
            e_r     = np.stack([ct*sp, ct*cp, st], axis=1)
            e_phi   = np.stack([cp, -sp, np.zeros(N_model)], axis=1)
            e_theta = np.stack([-st*sp, -st*cp, ct], axis=1)

        #__ Skewness-direction as linearised changes in xi __
        dr   = np.sum(wd * e_r, axis=1)
        dphi = np.degrees(np.sum(wd * e_phi, axis=1) / (r * ct))
        if (N_dir == 2): wd_circ = np.stack([dr, dphi], axis=1)
        else:
            dtheta  = np.degrees(np.sum(wd * e_theta, axis=1) / r)
            wd_circ = np.stack([dr, dphi, dtheta], axis=1)

    #__ Return outputs __
    return wd, mean, xi_circ, mean_circ, wd_circ

#__ Cylindrical/spherical harmonics __
def harmonic2D_density(
    az : np.ndarray,
    a  : np.ndarray,
    b  : np.ndarray,
    a0 : float = 1.0
):
    """
    Azimuthal density of a 2D noise field from its cylindrical-harmonic coefficients.

    Input
    -----
    az : ``numpy.ndarray``
        Azimuthal angles [deg, clockwise from North]
        (angle,)

    a : ``numpy.ndarray``
        Cosine coefficients
        ([model,] order)

    b : ``numpy.ndarray``
        Sine coefficients
        ([model,] order)

    a0 : ``float`` (default ``1.0``)
        Zeroth-order coefficient

    Output
    ------
    rho : ``numpy.ndarray``
        Normalised azimuthal density [rad-1]
        (model, angle)
    """

    #__ Shapes __
    a = np.atleast_2d(a) ; b = np.atleast_2d(b)
    M = a.shape[-1]

    #__ Compass angle __
    theta = np.radians(90.0 - az)

    #__ Harmonic series (model, angle) __
    m     = np.arange(1, M+1)
    cos_m = np.cos(m[:,None] * theta[None,:])        # (order, angle)
    sin_m = np.sin(m[:,None] * theta[None,:])
    rho   = (a0 + 2.0 * (a @ cos_m + b @ sin_m)) / (2*np.pi*a0)

    #__ Return output __
    return rho