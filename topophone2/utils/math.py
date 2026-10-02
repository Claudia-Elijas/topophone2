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
from scipy.special import jv, jvp, spherical_jn, lpmv, log_ndtr


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

# #__ Density function parameters __
# def density_params(
#     xi    : np.ndarray,
#     sigma : np.ndarray = None,
#     corr  : np.ndarray = None,
#     alpha : np.ndarray = None
# ):
#     """
#     Construct summary quantities for density functions (point, normal, skewnormal) in cylindrical/spherical coordinates.

#     Input
#     -----
#     xi : ``numpy.ndarray``
#         Location vector [s.m-1]
#         (model, direction)

#     sigma : ``numpy.ndarray`` (optional)
#         Scale vector [s.m-1]
#         (model, direction)

#     corr : ``numpy.ndarray`` (optional)
#         Correlation matrix off-diagonal (upper-triangle order)
#         (model, direction pair)

#     alpha : ``numpy.ndarray`` (optional)
#         Shape vector
#         (model, direction)

#     Output
#     ------
#     wd : ``numpy.ndarray``
#         Skewness-direction vector [s.m-1]
#         (model, direction)

#     mean : ``numpy.ndarray``
#         Mean vector [s.m-1]
#         (model, direction)

#     xi_circ : ``numpy.ndarray``
#         Location vector in cylindrical/spherical coordinates [s.m-1, deg from North[, deg from horizontal]]
#         (model, direction)

#     mean_circ : ``numpy.ndarray``
#         Mean vector in cylindrical/spherical coordinates [s.m-1, deg from North[, deg from horizontal]]
#         (model, direction)

#     sigma_circ : ``numpy.ndarray``
#         Scale vector in cylindrical/spherical coordinates about xi [s.m-1, deg from North[, deg from horizontal]]
#         (model, direction)

#     wd_circ : ``numpy.ndarray``
#         Skewness-direction vector in cylindrical/spherical coordinates about xi [s.m-1, deg from North[, deg from horizontal]]
#         (model, direction)
#     """

#     #__ np.einsum subscripts __
#     #__ [i] model
#     #__ [l],[p] direction

#     #__ Instantiate outputs __
#     wd, mean, xi_circ, mean_circ, sigma_circ, wd_circ = [None]*6

#     #__ Construct correlation matrix __
#     if corr is not None:
#         N_model, N_dir     = xi.shape
#         _iu0_dir, _iu1_dir = np.triu_indices(N_dir, k=1)
#         Omega_z = np.ones((N_model, N_dir, N_dir), dtype=float)
#         Omega_z[:,_iu0_dir,_iu1_dir] = corr
#         Omega_z[:,_iu1_dir,_iu0_dir] = corr

#     #__ Calculate skewness direction vector __
#     if alpha is not None:
#         Oza   = np.einsum('ilp,ip->il', Omega_z, alpha, optimize=True)
#         N     = 1.0 / np.sqrt(1.0 + np.einsum('il,il->i', alpha, Oza, optimize=True))
#         delta = N[:,None] * Oza

#         #__ Scale skewness-direction vector __
#         wd = sigma * delta

#     #__ Calculate mean __
#     if alpha is not None:
#         mean =   xi + wd * np.sqrt(2/np.pi)
#     else: mean = xi

#     #__ Cartesian -> cylindrical/spherical __
#     def to_circ(v):
#         r   = np.linalg.norm(v, axis=1)
#         phi = np.degrees(np.arctan2(v[:,0], v[:,1])) % 360
#         if (N_dir == 2): return np.stack([r, phi], axis=1)
#         theta = np.degrees(np.arctan2(v[:,2], np.hypot(v[:,0], v[:,1])))
#         return np.stack([r, phi, theta], axis=1)

#     #__ Project to cylindrical/spherical coordinates __
#     xi_circ   = to_circ(xi)
#     mean_circ = to_circ(mean)

#     #__ Local orthonormal basis at xi __
#     if (sigma is not None) or (alpha is not None):
#         r     = xi_circ[:,0]
#         phi   = np.radians(xi_circ[:,1])
#         theta = np.radians(xi_circ[:,2]) if (N_dir == 3) else np.zeros(N_model)
#         ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
#         if (N_dir == 2):
#             e_r   = np.stack([sp, cp], axis=1)
#             e_phi = np.stack([cp, -sp], axis=1)
#         else:
#             e_r     = np.stack([ct*sp, ct*cp, st], axis=1)
#             e_phi   = np.stack([cp, -sp, np.zeros(N_model)], axis=1)
#             e_theta = np.stack([-st*sp, -st*cp, ct], axis=1)

#         #__ Scale vector linearised about xi __
#         if (sigma is not None):
#             dr   = np.sum(sigma * e_r, axis=1)
#             dphi = np.degrees(np.sum(sigma * e_phi, axis=1) / (r * ct))
#             if (N_dir == 2): sigma_circ = np.stack([dr, dphi], axis=1)
#             else:
#                 dtheta  = np.degrees(np.sum(sigma * e_theta, axis=1) / r)
#                 sigma_circ = np.stack([dr, dphi, dtheta], axis=1)

#         #__ Skewness-direction as linearised changes in xi __
#         if (alpha is not None):
#             dr   = np.sum(wd * e_r, axis=1)
#             dphi = np.degrees(np.sum(wd * e_phi, axis=1) / (r * ct))
#             if (N_dir == 2): wd_circ = np.stack([dr, dphi], axis=1)
#             else:
#                 dtheta  = np.degrees(np.sum(wd * e_theta, axis=1) / r)
#                 wd_circ = np.stack([dr, dphi, dtheta], axis=1)

#     #__ Return outputs __
#     return wd, mean, xi_circ, mean_circ, sigma_circ, wd_circ

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
        Shape vector [dimensionless]
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
 
    sigma_circ : ``numpy.ndarray``
        Standard deviation along the local radial, azimuthal[, elevation] directions at the mean,
        from the covariance (Omega - 2/pi wd wd^T for the skew-normal) [s.m-1, deg[, deg]]
        (model, direction)
 
    wd_circ : ``numpy.ndarray``
        Skewness-direction vector resolved along the local radial, azimuthal[, elevation] directions
        at the mean; azimuthal component positive clockwise [s.m-1, deg[, deg]]
        (model, direction)
    """
 
    #__ Dimensions __
    N_model, N_dir = xi.shape
    wd, sigma_circ, wd_circ = None, None, None
 
    #__ Correlation matrix __
    Omega_z = np.tile(np.eye(N_dir), (N_model, 1, 1))
    if corr is not None:
        iu0, iu1 = np.triu_indices(N_dir, k=1)
        Omega_z[:, iu0, iu1] = Omega_z[:, iu1, iu0] = corr
 
    #__ Skewness-direction vector and mean __
    if alpha is not None:
        Oza   = np.einsum('ilp,ip->il', Omega_z, alpha)
        N     = 1.0 / np.sqrt(1.0 + np.einsum('il,il->i', alpha, Oza))
        wd    = sigma * N[:,None] * Oza
        mean  = xi + np.sqrt(2/np.pi) * wd
    else:
        mean  = xi
 
    #__ Cartesian -> cylindrical/spherical (compass bearing, elevation) __
    def to_circ(v):
        r   = np.linalg.norm(v, axis=1)
        phi = np.degrees(np.arctan2(v[:,0], v[:,1])) % 360
        if (N_dir == 2): return np.stack([r, phi], axis=1)
        theta = np.degrees(np.arctan2(v[:,2], np.hypot(v[:,0], v[:,1])))
        return np.stack([r, phi, theta], axis=1)
    xi_circ   = to_circ(xi)
    mean_circ = to_circ(mean)
 
    #__ Local orthonormal basis at xi __
    r     = xi_circ[:,0]
    phi   = np.radians(xi_circ[:,1])
    theta = np.radians(xi_circ[:,2]) if (N_dir == 3) else np.zeros(N_model)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    if (N_dir == 2):
        basis  = [np.stack([sp, cp], axis=1),                      # radial
                  np.stack([cp, -sp], axis=1)]                     # azimuthal (clockwise)
        scale  = [np.ones(N_model), r]                             # arc length -> angle
    else:
        basis  = [np.stack([ct*sp, ct*cp, st], axis=1),            # radial
                  np.stack([cp, -sp, np.zeros(N_model)], axis=1),  # azimuthal (clockwise)
                  np.stack([-st*sp, -st*cp, ct], axis=1)]          # elevation (upward)
        scale  = [np.ones(N_model), r*ct, r]
    to_angle = [False] + [True]*(N_dir-1)                          # convert all but radial to degrees
 
    #__ Standard deviations: project the covariance onto the local basis __
    if sigma is not None:
        Cov = Omega_z * sigma[:,:,None] * sigma[:,None,:]
        if alpha is not None:
            Cov = Cov - (2/np.pi) * np.einsum('il,ip->ilp', wd, wd)
        sigma_circ = np.stack([
            (np.degrees(s) if ang else s)
            for s, ang in zip(
                [np.sqrt(np.einsum('il,ilp,ip->i', e, Cov, e)) / k for e, k in zip(basis, scale)],
                to_angle)
        ], axis=1)
 
    #__ Skewness direction: resolve the vector onto the local basis __
    if alpha is not None:
        wd_circ = np.stack([
            (np.degrees(c) if ang else c)
            for c, ang in zip(
                [np.einsum('il,il->i', wd, e) / k for e, k in zip(basis, scale)],
                to_angle)
        ], axis=1)
 
    #__ Return outputs __
    return wd, mean, xi_circ, mean_circ, sigma_circ, wd_circ

#__ Density functions __
def density_function(
    s     : tuple,
    xi    : np.ndarray,
    sigma : np.ndarray,
    corr  : np.ndarray = None,
    alpha : np.ndarray = None
) -> np.ndarray:
    """
    Evaluate the normal or skew-normal probability density on a slowness grid.

    Input
    -----
    s : ``tuple`` of ``numpy.ndarray``
        Slowness grid from ``np.meshgrid`` [s.m-1], one array per direction
        (*grid) each

    xi : ``numpy.ndarray``
        Location vector [s.m-1]
        (model, direction)

    sigma : ``numpy.ndarray``
        Scale vector [s.m-1]
        (model, direction)

    corr : ``numpy.ndarray`` (optional)
        Correlation matrix off-diagonal (upper-triangle order)
        (model, direction pair)

    alpha : ``numpy.ndarray`` (optional)
        Shape vector [dimensionless]
        (model, direction)

    Output
    ------
    p : ``numpy.ndarray``
        Probability density [(s.m-1)^-N_dir]
        (model, *grid)
    """

    #__ Dimensions __
    S          = np.stack(s, axis=-1)
    grid_shape = S.shape[:-1]
    N_model, N_dir = xi.shape
    expand     = (N_model,) + (1,)*len(grid_shape)

    #__ Correlation matrix (Omega_z) __
    Omega_z = np.tile(np.eye(N_dir), (N_model, 1, 1))
    if corr is not None:
        iu = np.triu_indices(N_dir, k=1)
        Omega_z[:, iu[0], iu[1]] = Omega_z[:, iu[1], iu[0]] = corr

    #__ Positive-definiteness check __
    sign, logdet_Oz = np.linalg.slogdet(Omega_z)
    valid = sign > 0
    Omega_z[~valid] = np.eye(N_dir)

    #__ Standardised offsets z = (s - xi) / sigma __
    z = (S[None] - xi.reshape(expand + (N_dir,))) / sigma.reshape(expand + (N_dir,))

    #__ Normal log-density: phi(z; Omega_z) / prod(sigma) __
    quad     = np.einsum('m...i,mij,m...j->m...', z, np.linalg.inv(Omega_z), z)
    log_norm = -0.5*(N_dir*np.log(2*np.pi) + logdet_Oz) - np.log(sigma).sum(axis=-1)
    log_p    = -0.5*quad + log_norm.reshape(expand)

    #__ Skew-normal factor: 2 Phi(alpha^T z), in log space for stability __
    if alpha is not None:
        log_p += np.log(2.0) + log_ndtr(np.einsum('m...i,mi->m...', z, alpha))

    #__ Return output __
    p = np.exp(log_p)
    p[~valid] = np.nan
    return p

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

    a0 : ``numpy.ndarray`` (default ``1.0``)
        Zeroth-order coefficient
        ([model,])

    Output
    ------
    rho : ``numpy.ndarray``
        Normalised azimuthal density [rad-1]
        (model, angle)
    """

    #__ Shapes __
    a = np.atleast_2d(a) ; b = np.atleast_2d(b)
    a0 = np.atleast_1d(a0)
    M = a.shape[-1]

    #__ Compass angle __
    theta = np.radians(90.0 - az)

    #__ Harmonic series (model, angle) __
    m     = np.arange(1, M+1)
    cos_m = np.cos(m[:,None] * theta[None,:])        # (order, angle)
    sin_m = np.sin(m[:,None] * theta[None,:])
    rho   = (a0[:,None] + 2.0 * (a @ cos_m + b @ sin_m)) / (2*np.pi*a0[:,None])

    #__ Return output __
    return rho