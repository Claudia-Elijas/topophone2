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
