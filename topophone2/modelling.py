"""
.. container:: module-intro

    The ``topophone2.modelling`` module provides functionality for array-covariance modelling. It contains the :class:`ArrayCovarianceModel` class to predict the array covariance 
    for a given geometry and frequency vector recording any combination of energy distributions:

        - point (plane wave)
        - normal distribution
        - skew-normal distribution
        - spherically-/cylindrically-harmonic field.

"""

#________ IMPORT ________#

#__ System __
import itertools
from typing import Tuple, Optional, Dict, Callable

#__ Numerical __
import numpy as np
from scipy.special import dawsn
#from topophone2.utils.math import bessel_function, legendre_polynomial
from .utils.math import bessel_function, legendre_polynomial

#________ topophone2/utils/math.py ________#

#__ Import __
from scipy.special import jv, jvp, spherical_jn, lpmv


#________ DEFINE CLASS ________#

#__ Class to model array covariance __
class ArrayCovarianceModel():
    '''
        Class to model the array covariance for a given geometry
        and frequency vector recording any combination of energy distributions: point (plane wave),
        normal distribution, skew-normal distribution, and spherically-/cylindrically-harmonic field.
               
        Methods
        -------
        - :meth:`cf_point` : characteristic function for point (plane wave)
        - :meth:`cf_normal` : characteristic function for normal distribution
        - :meth:`cf_skewnormal` : characteristic function for skew-normal distribution
        - :meth:`cf_isonoise` : characteristic function for isotropic noise
        - :meth:`cf_noise2D` : characteristic function for 2D noise
        - :meth:`cf_noise3D` : characteristic function for 3D noise
        - :meth:`forward` : predict array covariance for a given model vector
    
        Input
        -----
        x : ``numpy.ndarray``
            Sensor location vectors 
            (sensor, direction)
    
        freq : ``numpy.ndarray``
            Angular frequency vector
            (frequency)
    
        signal_types : ``list`` of ``str`` or ``dict``
            Signal components to superpose. 
            
            a)  ``str``: density functions ('point', 'normal', 'skewnormal')
            
            b) ``dict``: harmonic functions for noise field ('noise2D', 'noise3D') with optional parameters.
                    
                    **sig** : ``str`` (required) - signal type ('isonoise', 'noise2D', 'noise3D')
                    
                    **s** : ``float`` (optional) - local slowness 
                    
                    **a** : ``numpy.ndarray`` (optional) - cosine coefficients
                    
                    **b** : ``numpy.ndarray`` (optional) - sine coefficients
                    
                    **max_order** : ``int`` (optional) - maximum order of harmonic basis (if **a**/**b** not specified)
                    
        weight_units : ``str`` (default 'dB')
            Units of the signal weights in the model vector
            ('dB' or 'linear') 

        calc_noise_only : ``bool`` (default ``False``)
            Include an additional noise-only calculation
    
        custom_cf : ``dict`` (default `{}`)
            Custom characteristic functions keyed by signal name,
            containing tuple of number of parameters and function with arguments:

                **t**   ``numpy.ndarray`` : Characteristic function argument (frequency, sensor pair, direction)
                
                **m_i** ``numpy.ndarray`` : Model vector (model, parameter)
            
            and returns:
               
                **cf**     ``numpy.ndarray`` : Characteristic function (model, frequency, sensor pair)
                
                **dcf_dm** ``numpy.ndarray``) : Characteristic function gradient wrt. model parameters (model, frequency, sensor pair, parameter)    

        Note
        ----
        If **s**, **a**, or **b** is unspecified, then treated as free parameters.
    
        
        Attributes
        ----------
        dist : ``numpy.ndarray``
            Inter-sensor distances (sensor pair)
    
        az : ``numpy.ndarray``
            Inter-sensor azimuths (sensor pair)
    
        pol : ``numpy.ndarray``
            Inter-sensor polar angles (sensor pair)
    
        t : ``numpy.ndarray``
            Characteristic function argument (frequency, sensor pair, direction)
    
        weight_units : ``str``
            Units of the signal weights in the model vector
    
        signal_types : ``list``
            Signal components to superpose
    
        freq : ``numpy.ndarray``
            Angular frequency vector
    
        N_pair : ``int``
            Number of sensor pairs
    
        N_dir : ``int``
            Number of directional components
    
        N_freq : ``int``
            Number of frequency bins
    
        N_corr : ``int``
            Number of correlation coefficients (upper triangle of Omega_z)
    
        max_order_2D : ``int``
            Maximum order of 2D harmonic basis
    
        max_order_3D : ``int``
            Maximum order of 3D harmonic basis
    
        order2D : ``numpy.ndarray``
            2D harmonic orders
    
        order3D : ``numpy.ndarray``
            3D harmonic orders
    
        degree3D : ``numpy.ndarray``
            3D harmonic degrees
    
        cos_2D : ``numpy.ndarray``
            2D harmonic cosine basis (sensor pair, order)
    
        sin_2D : ``numpy.ndarray``
            2D harmonic sine basis (sensor pair, order)
    
        cos_3D : ``numpy.ndarray``
            3D harmonic cosine basis (sensor pair, order)
    
        sin_3D : ``numpy.ndarray``
            3D harmonic sine basis (sensor pair, order)
    
        e_m : ``numpy.ndarray``
            2D harmonic cosine/sine normalization factors (order)
    
        i_m : ``numpy.ndarray``
            2D harmonic imaginary unit powers (order)
    
        i_n : ``numpy.ndarray``
            3D harmonic imaginary unit powers (degree)
    
        P_nm : ``numpy.ndarray``
            3D harmonic Legendre polynomials (sensor pair, order, degree)
    
        N_tri : ``int``
            Number of triangular (m <= n) index pairs
        
        '''
    
    #__ Initialise class __
    def __init__(
        self,
        x            : np.ndarray,
        freq         : np.ndarray,
        signal_types : list, 
        weight_units : str = 'dB',
        calc_noise_only : bool = False,
        custom_cf    : Dict[str,Tuple[int,Callable]] = {}
    ):
        
        #__ Store arguments as attributes __
        self.x               = x             # Sensor location [m] (sensor, direction)
        self.signal_types    = signal_types  # Signal components to superimpose
        self.weight_units    = weight_units  # Units of signal weights in the model vector
        self.freq            = freq          # Angular-frequency vector [rad] (frequency)
        self.N_freq          = len(freq)     # Number of frequency bins
        self._calc_noise_only = calc_noise_only  # Include an additional noise-only calculation
        self.custom_cf       = custom_cf

        #__ Array geometry __
        self.N_sta  = x.shape[0]                      # Number of sensors
        self.N_dir  = x.shape[1]                      # Number of directional components
        self._iu0_sta, self._iu1_sta = np.triu_indices(self.N_sta, k=0)   # Upper-triangle index set for unique station pairs (including diagonal)
        self._il0_sta, self._il1_sta = np.tril_indices(self.N_sta, k=-1)  # Lower-triangle index set for unique station pairs (excluding diagonal)
        self.N_pair = len(self._iu0_sta)              # Number of sensor pairs
        self.r = (x[None,:,:] - x[:,None,:])[self._iu0_sta, self._iu1_sta, :]  # Inter-sensor separation [m] (sensor pair, direction)
        self.dist = np.linalg.norm(self.r, axis=1)         # Inter-sensor distance [m] (sensor pair)
        self.az   = np.arctan2(self.r[:,1], self.r[:,0])   # Inter-sensor azimuth [rad, anticlockwise from +X] (sensor pair)
        if (self.N_dir == 3):
            with np.errstate(invalid='ignore', divide='ignore'):
                self.pol = np.arccos(self.r[:,2] / self.dist)  # Inter-sensor polar angle [rad, from +Z] (sensor pair)
            self.pol[self.dist == 0.0] = 0.0                   # Undefined at zero separation
        else: self.pol = None

        #__ Self noise __
        if ('isoselfnoise' in self.signal_types):
            self._cf_isoselfnoise = np.eye(self.N_sta)[None,None,self._iu0_sta, self._iu1_sta]  # Isotropic self-noise characteristic function (model, frequency, sensor pair)
        self._NOISE_SIGS = ['isonoise', 'noise2D', 'noise3D', 'isoselfnoise']                   # Possible noise signals

        #__ Characteristic-function argument __
        self.t = self.freq[:,None,None] * self.r[None,:,:]  # Characteristic-function argument [rad.s-1.m] (frequency, sensor pair, direction)

        #__ Weight units __
        if (self.weight_units == 'dB'):        # Weights in dB scale
            self._wscale = np.log(10.0)/10.0   # Gradient (wrt. dB weight) scalar 
        elif (self.weight_units == 'linear'):  # Weights in linear scale (e.g., Pa2.Hz-1)
            self._wscale = 1.0

        #__ Indices for upper-triangle of Omega_z __
        self._iu0_dir, self._iu1_dir = np.triu_indices(self.N_dir, k=1)  # Index set for unique direction pairs (excluding diagonal)
        self.N_corr = len(self._iu0_dir)  # Number of direction pairs

        #__ Initialise harmonic bases (for noise modelling) __
        self._init_harmonic_bases() 
        # self._max_order_2D, self._max_order_3D, 
        # self.order2D, self.order3D, self.degree3D, 
        # self.cos_2D, self.sin_2D, self.cos_3D, self.sin_3D,
        # self.i_m, self.i_n, self.e_m, self.P_nm, self.N_tri

        #__ Bessel cache for signals whose slowness is held fixed __
        self._bessel_cache = {}

    #__ Harmonic bases evaluation 
    def _init_harmonic_bases(self) -> dict:
        r"""Return a dictionary of harmonic bases for the current geometry."""
       
        #__ Geometry-dependent harmonic quantities __
        self._max_order_2D = None
        self._max_order_3D = None
        for signal_type in self.signal_types:
            if isinstance(signal_type, dict):
                if (signal_type.get('sig') == 'noise2D'):
                    self._max_order_2D = self._infer_max_order(signal_type)  # Maximum order of cylindrical basis
                elif (signal_type.get('sig') == 'noise3D'):
                    self._max_order_3D = self._infer_max_order(signal_type)  # Maximum order of spherical basis
        self._base = self.freq[None,:,None] * self.dist[None,None,:]  # Geometry-independent part of Bessel function argument

        #__ 2D harmonic basis __
        if self._max_order_2D is not None:
            self.order2D = np.arange(self._max_order_2D + 1)                 # Cylindrical-harmonic order vector (order)
            self.cos_2D    = np.cos(self.order2D[None,:] * self.az[:,None])  # Cylindrical-harmonic cosine basis (sensor pair, order)
            self.sin_2D    = np.sin(self.order2D[None,:] * self.az[:,None])  # Cylindrical-harmonic sine basis (sensor pair, order)
            self.i_m       = 1j ** self.order2D                              # Cylindrical-harmonic imaginary unit powers (order)
            self.e_m       = np.ones(len(self.order2D), dtype=float) ; self.e_m[1:] = 2.0  # Cylindrical-harmonic cosine/sine normalization factors (order)
        else:
            self.order2D = self.cos_2D = self.sin_2D = self.i_m = self.e_m = None

        #__ 3D harmonic basis __
        if self._max_order_3D is not None:
            if self.pol is None:
                raise ValueError("noise3D requires 3-component separation vectors.")
            self.order3D  = np.arange(self._max_order_3D + 1)             # Spherical-harmonic order vector (order)
            self.degree3D = np.arange(self._max_order_3D + 1)             # Spherical-harmonic degree vector (degree)
            self.cos_3D = np.cos(self.order3D[None,:] * self.az[:,None])  # Spherical-harmonic cosine basis (sensor pair, order)
            self.sin_3D = np.sin(self.order3D[None,:] * self.az[:,None])  # Spherical-harmonic sine basis (sensor pair, order)
            self.i_n    = 1j ** self.degree3D                             # Spherical-harmonic imaginary unit powers (degree)

            #__ Legendre polynomials __
            polynomial_argument = np.cos(self.pol)
            self.P_nm = np.empty((self.N_pair, len(self.order3D), len(self.degree3D)), dtype=float)  # Spherical-harmonic Legendre polynomials (sensor pair, order, degree)
            for (l, m_l), (p, n_p) in itertools.product(enumerate(self.order3D),
                                                        enumerate(self.degree3D)):
                self.P_nm[:,l,p] = legendre_polynomial(polynomial_argument, n_p, m_l)  # Populate P_nm

            #__ Triangular (m <= n) index set __
            self._tri_ord, self._tri_deg = np.tril_indices(self._max_order_3D + 1)  # Degree/order index set for lower triangle of degree-order pair matrix
            self.N_tri   = len(self._tri_ord)                              # Number of spherical-harmonic order-degree pairs
            self._a_free = ~((self._tri_ord == 0) & (self._tri_deg == 0))  # Free spherical-harmonic cosine-basis coefficients
            self._b_free =   (self._tri_ord != 0)                          # Free spherical-harmonic sine-basis coefficients
            self._n_a3D  = int(self._a_free.sum())                         # Number of free spherical-harmonic cosine-basis coefficients
            self._n_b3D  = int(self._b_free.sum())                         # Number of free spherical-harmonic sine-basis coefficients
        else:
            self.P_nm, self.N_tri = None, 0

    #__ Helper: read the truncation order out of a signal spec __
    @staticmethod
    def _infer_max_order(signal_type: dict) -> int:
        if ('max_order' in signal_type): return int(signal_type['max_order'])
        if ('a' in signal_type):         return int(np.shape(signal_type['a'])[-1]) - 1
        raise ValueError("Harmonic signal needs 'max_order' or explicit 'a'/'b'.")

    #__ Cached Bessel evaluation, value and derivative together __
    def _bessel(
        self,
        argument   : np.ndarray,
        max_order  : int,
        shape      : str,
        need_deriv : bool,
        cache_key  : Optional[tuple] = None
    ):
        r"""
        Return (value, derivative) for orders {0, ..., max_order}.

        Results are cached when the slowness is held fixed (`cache_key`).
        """

        #__ Cache lookup __
        if cache_key is not None:
            key = (cache_key, shape, need_deriv, max_order)
            if key in self._bessel_cache: return self._bessel_cache[key]

        #__ Cylindrical Bessel function __
        if (shape == 'cylindrical'):
            if need_deriv:
                oe = np.arange(-1, max_order+2)[None,None,None,:]
                W  = bessel_function(argument, order=oe, shape=shape)
                val = W[..., 1:max_order+2]
                der = 0.5 * (W[..., 0:max_order+1] - W[..., 2:max_order+3])
            else:
                o   = np.arange(max_order+1)[None,None,None,:]
                val = bessel_function(argument, order=o, shape=shape)
                der = None

        #__ Spherical Bessel function __
        else:
            o   = np.arange(max_order+1)[None,None,None,:]
            val = bessel_function(argument, order=o, shape=shape)
            der = (bessel_function(argument, order=o, shape=shape, derivative=True)
                   if need_deriv else None)

        #__ Cache and return __
        if cache_key is not None: self._bessel_cache[key] = (val, der)
        return val, der

    #________ CHARACTERISTIC FUNCTIONS ________#

    #__ Point (plane wave) __
    def cf_point(
        self,
        xi : np.ndarray,
        grad : bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        r"""

        Function that retuns the characteristic function for a point (plane wave) distribution.

        Input
        -----
        xi : ``numpy.ndarray``
            Location vector (slowness)
            (model, direction)

        grad : ``bool`` (default ``True``)
            Calculate gradients

        Output
        ------
        cf : ``numpy.ndarray``
            Characteristic function
            (model, frequency, sensor pair)

        grad_cf : ``dict``
            Gradient of the characteristic function, keyed by parameter name. 

            - **xi** : (``numpy.ndarray``) -- (model, frequency, sensor pair, direction)

        
        The characteristic function for a point (plane wave) distribution is given by

        .. math::
                
                    \varphi_\mathbf{s} \left(\mathbf{t} ; \mathbf{\xi}\right) = \exp \left[i \mathbf{t}^T \mathbf{\xi}\right]

        where :math:`\mathbf{t}` is the characteristic function argument (frequency, sensor pair, direction) and :math:`\mathbf{\xi}` is the location vector (slowness).
                
        The gradients of the characteristic function are given by the following expression:

        .. math::

                    \frac{\partial \varphi_\mathbf{s}}{\partial \xi_i} = i t_i \varphi_\mathbf{s}
        
    
        """

        #__ np.einsum subscripts:  
        #__ [i] model 
        #__ [j] frequency
        #__ [k] sensor pair
        #__ [l] direction

        #__ Calculate characteristic function (model, frequency, sensor pair) __
        L  = 1j * np.einsum('jkl,il->ijk', self.t, xi, optimize=True)
        cf = np.exp(L)

        #__ Calculate gradient __
        grad_cf = {}
        if grad:
            grad_cf['xi'] = 1j * self.t[None,...] * cf[...,None]

        #__ Return output __
        return cf, grad_cf

    #__ Normal distribution __
    def cf_normal(
        self,
        xi    : np.ndarray,
        sigma   : np.ndarray,
        Omega_z : np.ndarray,
        grad    : bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        r"""

        Function that retuns the characteristic function for a normal distribution.
        
        Input
        -----
        xi : ``numpy.ndarray``
            Location vector
            (model, direction)

        sigma : ``numpy.ndarray``
            Scale vector
            (model, direction)

        Omega_z : ``numpy.ndarray``
            Correlation matrix
            (model, direction, direction)

        grad : ``bool`` (default ``True``)
            Calculate gradients

        Output
        ------
        cf : ``numpy.ndarray``
            Characteristic function
            (model, frequency, sensor pair)

        grad_cf : ``dict``
            Gradient of the characteristic function, keyed by parameter name:

            - **xi** : (``numpy.ndarray``) -- (model, frequency, sensor pair, direction)

            - **sigma** : (``numpy.ndarray``) -- (model, frequency, sensor pair, direction)

            - **corr** : (``numpy.ndarray``) -- (model, frequency, sensor pair, direction pair)
                    
        Note
        ----
        The correlation matrix is already flattened to upper triangle of ``Omega_z``


        The characteristic function for a normal distribution is given by

        .. math::

                \varphi_\mathbf{s} \left(\mathbf{t} ; \mathbf{\xi}, \Omega\right) = \exp \left[i \mathbf{t}^T \mathbf{\xi} - \frac{1}{2} \mathbf{t}^T \Omega \mathbf{t}\right]

                \Omega = W \Omega_z W

                W = \mathrm{diag} \left(\mathbf{\sigma}\right)

                \Omega_z = \left\|\rho_{ij}\right\|

        where :math:`\mathbf{t}` is the characteristic function argument (frequency, sensor pair, direction), :math:`\mathbf{\xi}` is the location vector (slowness), :math:`\Omega` is the covariance matrix, :math:`W` is the diagonal scale matrix, and :math:`\Omega_z` is the correlation matrix.
        The gradients of the characteristic function are given by the following expressions:

        .. math::
                
                \frac{\partial \varphi_\mathbf{s}}{\partial \xi_i} = i t_i \varphi_\mathbf{s}

                \frac{\partial \varphi_\mathbf{s}}{\partial \sigma_i} = - \sum_j rho_{ij} \sigma_j t_i t_j \varphi_\mathbf{s}

                \frac{\partial \varphi_\mathbf{s}}{\partial \rho_{ij}} = - \sigma_i \sigma_j t_i t_j \varphi_\mathbf{s}

        where

        """

        #__ np.einsum subscripts:
        #__ [i] model
        #__ [j] frequency
        #__ [k] sensor pair
        #__ [l],[p] direction

        #__ Intermediates (model, frequency, pair, direction) __
        w   = sigma[:,None,None,:] * self.t[None,...]
        Ozw = np.einsum('ilp,ijkp->ijkl', Omega_z, w, optimize=True)

        #__ Calculate characteristic function (model, frequency, sensor pair) __
        Q  = 0.5 * np.einsum('ijkl,ijkl->ijk', w, Ozw, optimize=True)
        L  = 1j * np.einsum('jkl,il->ijk', self.t, xi, optimize=True) - Q
        cf = np.exp(L)

        #__ Calculate gradients __
        grad_cf = {}
        if grad:

            #__ Location (model, frequency, sensor pair, direction) __
            grad_cf['xi'] = 1j * self.t[None,...] * cf[...,None]

            #__ Scale (model, frequency, sensor pair, direction) __
            grad_cf['sigma'] = -cf[...,None] * self.t[None,...] * Ozw

            #__ Correlation (model, frequency, sensor pair, direction pair) __
            grad_cf['corr'] = -cf[...,None] * w[...,self._iu0_dir] * w[...,self._iu1_dir]

        #__ Return output __
        return cf, grad_cf

    #__ Skew-normal distribution __
    def cf_skewnormal(
        self,
        xi    : np.ndarray,
        sigma   : np.ndarray,
        Omega_z : np.ndarray,
        alpha   : np.ndarray,
        grad    : bool = True
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        r"""
        
        Function that retuns the characteristic function for a skew-normal distribution.

        Input
        -----
        xi : ``numpy.ndarray``
            Location vector
            (model, direction)

        sigma : ``numpy.ndarray``
            Scale vector
            (model, direction)

        Omega_z : ``numpy.ndarray``
            Correlation matrix
            (model, direction, direction)

        alpha : ``numpy.ndarray``
            Shape vector
            (model, direction)

        grad : ``bool`` (default ``True``)
            Calculate gradients

        Output
        ------
        cf : ``numpy.ndarray``
            Characteristic function
            (model, frequency, sensor pair)

        grad_cf : ``dict``
            Gradient of the characteristic function, keyed by parameter name:

            - **xi** : (``numpy.ndarray``) -- (model, frequency, sensor pair, direction)

            - **sigma** : (``numpy.ndarray``) -- (model, frequency, sensor pair, direction)

            - **corr** : (``numpy.ndarray``) -- (model, frequency, sensor pair, direction pair)

            - **alpha** : (``numpy.ndarray``) -- (model, frequency, sensor pair, direction pair)


        Note
        ----
        The correlation matrix is already flattened to upper triangle of ``Omega_z``.


        The characteristic function for a skew-normal distribution is given by

        .. math::

            \varphi_\mathbf{s} \left(\mathbf{t} ; \mathbf{\xi}, \Omega, \mathbf{\alpha}\right) = \exp \left[i \mathbf{t}^T \mathbf{\xi} - \frac{1}{2} \mathbf{t}^T \Omega \mathbf{t}\right] \left[1 + i \erfi \left(\frac{\sqrt{2}}{2} \mathbf{\delta}^T W \mathbf{t}\right)\right]

            \Omega = W \Omega_z W

            W = \mathrm{diag} \left(\mathbf{\sigma}\right)

            \Omega_z = \left\|\rho_{ij}\right\|

            \mathbf{\delta} = \frac{\Omega_z \mathbf{\alpha}}{\sqrt{1 + \mathbf{\alpha}^T \Omega_z \mathbf{\alpha}}}


        where :math:`\mathbf{t}` is the characteristic function argument (frequency, sensor pair, direction), :math:`\mathbf{\xi}` is the location vector (slowness), :math:`\Omega` is the covariance matrix, :math:`W` is the diagonal scale matrix, :math:`\Omega_z` is the correlation matrix, and :math:`\mathbf{\delta}` is the skewness-direction vector.
        The gradients of the characteristic function are given by the following expressions:

        .. math::

            \frac{\partial \varphi_\mathbf{s}}{\partial \xi_i} = i t_i \varphi_\mathbf{s}
            
            \frac{\partial \varphi_\mathbf{s}}{\partial \sigma_i} = - \sum_j \rho_{ij} \sigma_j t_i t_j \varphi_\mathbf{s} + \sqrt{\frac{2}{\pi}} i \exp \left[u^2 + L\right] \delta_i t_i

            u = \frac{\sqrt{2}}{2} \mathbf{\delta}^T W \mathbf{t}

            L = i \mathbf{t}^T \mathbf{\xi} - \frac{1}{2} \mathbf{t}^T \Omega \mathbf{t}
            
            \frac{\partial \varphi_\mathbf{s}}{\partial \rho_{ij}} = - \sigma_i \sigma_j t_i t_j \varphi_\mathbf{s} + \sqrt{\frac{2}{\pi}} i N \exp \left[u^2 + L\right] \left(\sigma_i t_i \alpha_j + \sigma_j t_j \alpha_i - N^2 \alpha_i \alpha_j \sum_{k, l} \sigma_k t_k \rho_{kl} \alpha_l\right)
            
            N = \sqrt{1 + \mathbf{\alpha}^T \Omega_z \mathbf{\alpha}}

            \frac{\partial \varphi_\mathbf{s}}{\partial \alpha_i} = \sqrt{\frac{2}{\pi}} i \exp \left[u^2 + L\right] \sum_{j, k, l} \sigma_j t_j \left(\rho_{ji} N - \rho_{il} \alpha_l \rho_{jk} \alpha_k N^3\right)
            
        
        """

        #__ np.einsum subscripts:
        #__ [i] model
        #__ [j] frequency
        #__ [k] sensor pair
        #__ [l],[p] direction

        #__ Intermediates (model, frequency, pair, direction) __
        w   = sigma[:,None,None,:] * self.t[None,...]
        Ozw = np.einsum('ilp,ijkp->ijkl', Omega_z, w, optimize=True)

        #__ Calculate skewness-direction vector (model, direction) __
        Oza   = np.einsum('ilp,ip->il', Omega_z, alpha, optimize=True)
        N     = 1.0 / np.sqrt(1.0 + np.einsum('il,il->i', alpha, Oza, optimize=True))
        delta = N[:,None] * Oza

        #__ Calculate characteristic function (model, frequency, sensor pair) __
        Q  = 0.5 * np.einsum('ijkl,ijkl->ijk', w, Ozw, optimize=True)
        L  = 1j * np.einsum('jkl,il->ijk', self.t, xi, optimize=True) - Q
        u  = np.einsum('il,ijkl->ijk', delta, w, optimize=True) / np.sqrt(2.0)
        #cf = (1.0 + 1j * erfi(u)) * np.exp(L)
        K  = np.sqrt(2.0/np.pi) * 1j * np.exp(u**2 + L)
        cf = np.exp(L) + np.sqrt(2.0) * dawsn(u) * K  # Dawson function replacing erfi to keep within computational bounds

        #__ Calculate gradients __
        grad_cf = {}
        if grad:

            #__ Shared prefactor __
            #K = np.sqrt(2.0/np.pi) * 1j * np.exp(u**2 + L)

            #__ Location (model, frequency, sensor pair, direction) __
            grad_cf['xi'] = 1j * self.t[None,...] * cf[...,None]

            #__ Scale (model, frequency, sensor pair, direction) __
            grad_cf['sigma'] = (-cf[...,None] * self.t[None,...] * Ozw
                                + K[...,None] * delta[:,None,None,:] * self.t[None,...])

            #__ Correlation (model, frequency, sensor pair, direction pair) __
            w0, w1 = w[...,self._iu0_dir], w[...,self._iu1_dir]
            a0     = alpha[:,None,None,self._iu0_dir]
            a1     = alpha[:,None,None,self._iu1_dir]
            Nb     = N[:,None,None,None]
            grad_cf['corr'] = (-cf[...,None] * w0 * w1
                               + K[...,None] * (Nb * (w0*a1 + a0*w1)
                                                - np.sqrt(2.0) * Nb**2 * u[...,None] * a0 * a1))

            #__ Shape (model, frequency, sensor pair, direction) __
            grad_cf['alpha'] = (K[...,None] * N[:,None,None,None]
                                * (Ozw - np.sqrt(2.0) * u[...,None] * delta[:,None,None,:]))

        #__ Return output __
        return cf, grad_cf

    #__ Isotropic noise __
    def cf_isonoise(
        self,
        s          : np.ndarray|float,
        dim        : int = 2,
        grad_s     : bool = False,
        fixed_key  : Optional[tuple] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        r""" 
        
        Function that retuns the characteristic function for an isotropic noise field.

        Input
        -----
        s : ``numpy.ndarray`` or ``float``
            Local slowness
            (model)

        dim : ``int`` (default 2)
            Dimensions of noise (`2` or `3`)

        grad_s : ``bool`` (default ``False``)
            Calculate gradient wrt. `s`

        fixed_key : ``tuple`` (optional)
            Cache key; supplied only when `s` is held fixed

        Output
        ------
        cf : ``numpy.ndarray``
            Characteristic function
            (model, frequency, sensor pair)

        grad_cf : ``dict``
            Gradient of the characteristic function, keyed by parameter name:

            - **s** : (``numpy.ndarray``) -- (model, frequency, sensor pair)
            

        The characteristic function for an isotropic noise field is given by

        .. math::

            \varphi_{\mathbf{s}}(\mathbf{t};S_{tr}) = 
            \begin{cases}
                J_0\left(\ S_{tr} |\mathbf{t}\| \right), & \text{2D (cylindrical)} \\
                j_0\left(\ S_{tr} |\mathbf{t}\| \right), & \text{3D (spherical)} 
            \end{cases}

        where :math:`J_0` is the Bessel function of order 0 and :math:`j_0` is the spherical Bessel function of order 0.
        The gradients of the characteristic function are given by the following expressions:
        
        .. math::

            \frac{\partial \varphi_{\mathbf{s}}}{\partial S_{tr}} = 
            \begin{cases}
                -|\mathbf{t}\| J_1 \left(\ S_{tr} |\mathbf{t}\| \right), & \text{2D (cylindrical)} \\
                -|\mathbf{t}\| j_1 \left(\ S_{tr} |\mathbf{t}\| \right), & \text{3D (spherical)}
            \end{cases}

        """

        #__ Sort inputs (model) __
        s = np.atleast_1d(np.asarray(s, dtype=float))

        #__ Select Bessel shape __
        if   (dim == 2): shape = 'cylindrical'
        elif (dim == 3): shape = 'spherical'
        else: raise ValueError("`dim` should be 2 or 3.")

        #__ Bessel argument (model, frequency, sensor pair) __
        argument = self._base * s[:,None,None]

        #__ Calculate characteristic function __
        J_0, dJ = self._bessel(argument[...,None], 0, shape, grad_s, fixed_key)
        cf = J_0[...,0]

        #__ Calculate gradient (model, frequency, sensor pair) __
        grad_cf = {}
        if grad_s:
            grad_cf['s'] = self._base * dJ[...,0]

        #__ Return output __
        return cf, grad_cf

    #__ Noise (2D) __
    def cf_noise2D(
        self,
        a         : np.ndarray,
        b         : np.ndarray,
        s         : np.ndarray|float,
        grad_ab   : bool = False,
        grad_s    : bool = False,
        fixed_key : Optional[tuple] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        r"""

        Function that retuns the characteristic function for a 2D isotropic noise field.

        Input
        -----
        a, b : ``numpy.ndarray``
            Cosine / sine components
            ([model], order)

        s : ``numpy.ndarray`` or ``float``
            Local slowness
            ([model])

        grad_ab, grad_s : ``bool``
            Calculate gradients

        Output
        ------
        cf : ``numpy.ndarray``
            Characteristic function
            (model, frequency, sensor pair)

        grad_cf : ``dict``
            Gradient of the characteristic function, keyed by parameter name:

            - **a** : (``numpy.ndarray``) -- (model, frequency, sensor pair, order)

            - **b** : (``numpy.ndarray``) -- (model, frequency, sensor pair, order)

            - **s** : (``numpy.ndarray``) -- (model, frequency, sensor pair)

        
        The characteristic function for a 2D isotropic noise field is given by
        
        .. math::

            \varphi_{\mathbf{s}}(\mathbf{t}, \phi; S_{tr}, \mathbf{b}, \mathbf{b}) = \sum_{m=0}^{M} i^m \epsilon_m C_m (\phi; a_m , b_m) J_m(S_{tr} |\mathbf{t}\|)

        where :math:`C_m = a_m \cos(m \phi) + b_m \sin(m \phi)` and :math:`J_m` is the Bessel function of order :math:`m`.
        The gradients of the characteristic function are given by the following expressions:

        .. math::

            \frac{\partial \varphi_{\mathbf{s}}}{\partial a_m} = i^m \epsilon_m \cos(m \phi) J_m(S_{tr} |\mathbf{t}\|)
            
        
        .. math::

            \frac{\partial \varphi_{\mathbf{s}}}{\partial b_m} = i^m \epsilon_m \sin(m \phi) J_m(S_{tr} |\mathbf{t}\|)


        .. math::
        
            \frac{\partial \varphi_{\mathbf{s}}}{\partial S_{tr}} = - \sum_{m=0}^{M} i^m \epsilon_m C_m (\phi; a_m , b_m) S_{tr} J'_m(S_{tr} |\mathbf{t}\|)

        where 
        
        .. math::

            \mathbf{\epsilon} = (1, 2, \ldots, 2) 

            C_m = a_m \cos(m \phi) + b_m \sin(m \phi)

            J_m '(S_{tr} |\mathbf{t}\|) = \frac{1}{2} \left[ J_{m-1}(S_{tr} |\mathbf{t}\|) - J_{m+1}(S_{tr} |\mathbf{t}\|) \right]
            
            """    

        #__ np.einsum subscripts:
        #__ [i] model
        #__ [j] frequency
        #__ [k] sensor pair
        #__ [l] order

        #__ Sort inputs __
        s = np.atleast_1d(np.asarray(s, dtype=float))
        a = np.atleast_2d(a) ; b = np.atleast_2d(b)

        #__ Calculate coefficient (model, sensor pair, order) __
        C_m = (np.einsum('il,kl->ikl', a, self.cos_2D, optimize=True)
               + np.einsum('il,kl->ikl', b, self.sin_2D, optimize=True))

        #__ Bessel argument (model, frequency, sensor pair) __
        argument = self._base * s[:,None,None]

        #__ Bessel functions (model, frequency, sensor pair, order) __
        J_m, dJ = self._bessel(argument[...,None], self.order2D[-1],
                                  'cylindrical', grad_s, fixed_key)

        #__ Calculate characteristic function (model, frequency, sensor pair) __
        cf = np.einsum('l,l,ikl,ijkl->ijk', self.i_m, self.e_m, C_m, J_m, optimize=True)

        #__ Gradients __
        grad_cf = {}
        if grad_ab:
            prefactor = (self.i_m * self.e_m)                              # (order)
            grad_cf['a'] = prefactor * self.cos_2D[None,None,:,:] * J_m
            grad_cf['b'] = prefactor * self.sin_2D[None,None,:,:] * J_m
        if grad_s:
            grad_cf['s'] = self._base * np.einsum('l,l,ikl,ijkl->ijk',
                                            self.i_m, self.e_m, C_m, dJ, optimize=True)

        #__ Return outputs __
        return cf, grad_cf

    #__ Noise (3D) __
    def cf_noise3D(
        self,
        a         : np.ndarray,
        b         : np.ndarray,
        s         : np.ndarray|float,
        grad_ab   : bool = False,
        grad_s    : bool = False,
        fixed_key : Optional[tuple] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:

        r"""
        
        Function that retuns the characteristic function for a 3D isotropic noise field.

        Input
        -----
        a, b : ``numpy.ndarray``
            Cosine / sine components
            ([model], order, degree)

        s : ``numpy.ndarray`` or float
            Local slowness
            ([model])

        Output
        ------
        cf : ``numpy.ndarray``
            Characteristic function
            (model, frequency, sensor pair)
        
        grad_cf : ``dict``
            Gradient of the characteristic function, keyed by parameter name.

            - **a** : (``numpy.ndarray``) -- (model, frequency, sensor pair, triangle)

            - **b** : (``numpy.ndarray``) -- (model, frequency, sensor pair, triangle)

            - **s** : (``numpy.ndarray``) -- (model, frequency, sensor pair)

        Note
        ----
        **a** and **b** are already flattened to :math:`m \leq n` triangle.



        The characteristic function for a 3D isotropic noise field is given by

        .. math::
        
            \varphi_{\mathbf{s}}(\mathbf{t}, \phi, \theta; S_{tr}, A, B) = \sum_{n=0}^{N} \sum_{m=0}^{n} i^n C_{nm} (\phi ; A, B) P_n^m(\cos(\theta)) j_n(S_{tr} |\mathbf{t}\|)

        where :math:`P_n^m` is the associated Legendre polynomial of order :math:`m` and degree :math:`n`, and :math:`j_n` is the spherical Bessel function of order :math:`n`.
        The azimuthal angle :math:`\phi` is measured from the x-axis in the xy-plane, and the polar angle :math:`\theta` is measured from the z-axis. 
        The coefficients :math:`A_n^m` and :math:`B_n^m` are the cosine and sine components of the harmonic expansion, respectively.
        The harmonic coefficient :math:`C_n^m` is given by

        .. math::
        
            C_n^m = A_n^m \cos(m \phi) + B_n^m \sin(m \phi)

        
        The derivatives of the characteristic function with respect to the model parameters are:
            
        .. math:: 
        
            \frac{\partial \varphi_{\mathbf{s}}}{\partial S_{tr}} = \sum_{n=0}^{N} \sum_{m=0}^{n} i^n C_n^m P_n^m S_{tr} j'_n(S_{tr} |\mathbf{t}\|)
       
        where :math:`j'_n` is the derivative of the spherical Bessel function of order :math:`n` with respect to its argument, and is given by
        
        .. math::
        
            j'_n(S_{tr} |\mathbf{t}\|) ) = \frac{n j_{n-1}(S_{tr} |\mathbf{t}\|) - (n+1) j_{n+1}(S_{tr} |\mathbf{t}\|)}{2n + 1}
             
        .. math::   
        
            \frac{\partial \varphi_{\mathbf{s}}}{\partial A_n^m} = i^n \cos(m \phi) P_n^m j_n(S_{tr} |\mathbf{t}\|)

        .. math::
            
            \frac{\partial \varphi_{\mathbf{s}}}{\partial B_n^m} = i^n \sin(m \phi) P_n^m j_n(S_{tr} |\mathbf{t}\|)
            

        """

        #__ np.einsum subscripts:
        #__ [i] model
        #__ [j] frequency
        #__ [k] sensor pair
        #__ [l] order  
        #__ [p] degree

        #__ Sort inputs __
        s = np.atleast_1d(np.asarray(s, dtype=float))
        if (a.ndim == 2): a = a[None,:,:]
        if (b.ndim == 2): b = b[None,:,:]

        #__ Calculate coefficient (model, sensor pair, order, degree) __
        C_nm = (np.einsum('ilp,kl->iklp', a, self.cos_3D, optimize=True)
                + np.einsum('ilp,kl->iklp', b, self.sin_3D, optimize=True))

        #__ Bessel argument (model, frequency, sensor pair) __
        argument = self._base * s[:,None,None]

        #__ Spherical Bessel functions (model, frequency, sensor pair, degree) __
        j_n, djn = self._bessel(argument[...,None], self.degree3D[-1],
                                   'spherical', grad_s, fixed_key)

        #__ Calculate characteristic function (model, frequency, sensor pair) __
        cf = np.einsum('p,iklp,klp,ijkp->ijk', self.i_n, C_nm, self.P_nm, j_n, optimize=True)

        #__ Gradients __
        grad_cf = {}
        if grad_ab:  # (model, frequency, sensor pair, degree-order pair)
            P_tri = self.P_nm[:, self._tri_ord, self._tri_deg]
            i_tri = self.i_n[self._tri_deg]
            jn_tri = j_n[..., self._tri_deg]
            grad_cf['a'] = (i_tri * P_tri[None,None,:,:]
                            * self.cos_3D[None,None,:,self._tri_ord] * jn_tri)
            grad_cf['b'] = (i_tri * P_tri[None,None,:,:]
                            * self.sin_3D[None,None,:,self._tri_ord] * jn_tri)
        if grad_s:  # (model, frequency, sensor pair)
            grad_cf['s'] = self._base * np.einsum('p,iklp,klp,ijkp->ijk',
                                           self.i_n, C_nm, self.P_nm, djn, optimize=True)

        #__ Return outputs __
        return cf, grad_cf

    #________ FORWARD FUNCTION ________#

    #__ Number of parameters per signal __
    def _n_params(self, signal_type : str|dict) -> int:
        r"""Length of this signal's slice of the model vector (excluding weight)."""
        if isinstance(signal_type, str):
            if (signal_type == 'point'):      return self.N_dir
            if (signal_type == 'normal'):     return 2*self.N_dir + self.N_corr
            if (signal_type == 'skewnormal'): return 3*self.N_dir + self.N_corr
            if (signal_type == 'isoselfnoise'):  return 0
            raise ValueError(f"Unknown signal type '{signal_type}'.")
        sig, n = signal_type['sig'], 0
        if ('s' not in signal_type): n += 1
        if not (('a' in signal_type) and ('b' in signal_type)):
            if   (sig == 'noise2D'): n += 2 * self._max_order_2D
            elif (sig == 'noise3D'): n += self._n_a3D + self._n_b3D
        return n

    #__ Unpack model vector __
    def _unpack(
        self,
        m    : np.ndarray,
        grad : bool = True
    ) -> Tuple[list,list,list,list]:
        r"""
        Input
        -----
        `m` : `numpy.ndarray`
            Model vector
            (model, parameter)

        Output
        ------
        `sig_list`    : signal identifiers
        `kwargs_list` : arguments for the characteristic functions
        `idx_list`    : model-vector boundaries (leading 0, one extra on the end)
        `weight_list` : weight of each signal, shaped to broadcast
        """

        #__ Instantiate outputs __
        sig_list, kwargs_list, idx_list, weights_list = [], [], [0], []

        #__ Loop through signal types __
        for signal_type in self.signal_types:
            idx = idx_list[-1]

            #__ Weight __
            if (self.weight_units == 'dB'):
                weights_list.append(10**(m[:,idx][:,None,None]/10))
            elif (self.weight_units == 'linear'):
                weights_list.append(m[:,idx][:,None,None])
            idx += 1

            #__ Density functions __
            if isinstance(signal_type, str):
                sig = signal_type
                sig_list.append(sig)

                #__ Location __
                if (sig in ['point', 'normal', 'skewnormal']):
                    xi = m[:,idx:idx+self.N_dir] ; idx += self.N_dir
                    kwargs = dict(xi = xi, grad = grad)

                #__ Scale __
                if (sig in ['normal', 'skewnormal']):
                    sigma = m[:,idx:idx+self.N_dir] ; idx += self.N_dir
                    corr  = m[:,idx:idx+self.N_corr] ; idx += self.N_corr
                    Omega_z = np.ones((m.shape[0], self.N_dir, self.N_dir), dtype=float)
                    Omega_z[:,self._iu0_dir,self._iu1_dir] = corr
                    Omega_z[:,self._iu1_dir,self._iu0_dir] = corr
                    kwargs.update(sigma=sigma, Omega_z=Omega_z)

                #__ Shape __
                if (sig in ['skewnormal']):
                    alpha = m[:,idx:idx+self.N_dir] ; idx += self.N_dir
                    kwargs.update(alpha = alpha)

                #__ Custom function __
                if (sig in self.custom_cf):
                    N_custom = self.custom_cf[sig][0]
                    mi = m[:,idx:idx+N_custom] ; idx += N_custom
                    kwargs = dict(t = self.t, mi = mi)

                #__ Self noise __
                if (sig == 'isoselfnoise'): kwargs = {}

            #__ Harmonic functions __
            elif isinstance(signal_type, dict):
                sig = signal_type['sig']
                sig_list.append(sig)
                kwargs = {}

                #__ Slowness __
                if ('s' in signal_type):
                    kwargs.update(s = signal_type['s'], grad_s = False,
                                  fixed_key = (sig, float(np.ravel(signal_type['s'])[0])))
                else:
                    kwargs.update(s = m[:,idx], grad_s = grad) ; idx += 1

                #__ Isotropic noise __
                if (sig == 'isonoise'):
                    kwargs['dim'] = signal_type.get('dim', 2)

                #__ Noise (2D) __
                elif (sig == 'noise2D'):
                    if ('a' in signal_type) and ('b' in signal_type):
                        kwargs.update(a=signal_type['a'], b=signal_type['b'],
                                      grad_ab=False)
                    else:
                        n_rat = self._max_order_2D
                        one   = np.ones ((m.shape[0], 1))
                        zero  = np.zeros((m.shape[0], 1))
                        kwargs.update(a = np.concatenate([one,  m[:,idx        :idx+n_rat  ]], axis=1),
                                    b = np.concatenate([zero, m[:,idx+n_rat  :idx+2*n_rat]], axis=1),
                                    grad_ab = grad)
                        idx += 2 * n_rat

                #__ Noise (3D) __
                elif (sig == 'noise3D'):
                    if ('a' in signal_type) and ('b' in signal_type):
                        kwargs.update(a=signal_type['a'], b=signal_type['b'],
                                      grad_ab=False)
                    else:
                        n_ord, n_tri = self._max_order_3D + 1, self.N_tri
                        na, nb = self._n_a3D, self._n_b3D
                        a_flat = np.zeros((m.shape[0], n_tri)) ; b_flat = np.zeros((m.shape[0], n_tri))
                        a_flat[:, ~self._a_free] = 1.0
                        a_flat[:,  self._a_free] = m[:, idx      : idx+na]
                        b_flat[:,  self._b_free] = m[:, idx+na   : idx+na+nb]
                        a = np.zeros((m.shape[0], n_ord, n_ord)) ; b = np.zeros((m.shape[0], n_ord, n_ord))
                        a[:,self._tri_ord,self._tri_deg] = a_flat
                        b[:,self._tri_ord,self._tri_deg] = b_flat
                        idx += na + nb
                        kwargs.update(a=a, b=b, grad_ab=grad)
                else: raise ValueError(f"Unknown signal type '{sig}'.")

            #__ Record __
            idx_list.append(idx) ; kwargs_list.append(kwargs)

        #__ Return outputs __
        return sig_list, kwargs_list, idx_list, weights_list

    #__ Write gradient blocks straight into a preallocated view __
    def _write_grad(
        self,
        signal_type : str|dict,
        grad_cf     : dict,
        out         : np.ndarray,
        weight      : np.ndarray|float
    ) -> None:
        r"""
        Fill `out` (a view on dcov_dm, shape (model, freq, pair, n_param)) with
        this signal's gradients, scaled by its mixture weight.
        """

        #__ Column cursor within `out` __
        c = 0
        def put(block):
            nonlocal c
            n = block.shape[-1]
            np.multiply(block, weight[...,None] if np.ndim(weight) else weight,
                        out=out[..., c:c+n], casting='unsafe')
            c += n

        #__ Density functions __
        if isinstance(signal_type, str):
            if (signal_type in ['point','normal','skewnormal']):
                put(grad_cf['xi'])
            if (signal_type in ['normal','skewnormal']):
                put(grad_cf['sigma'])
                put(grad_cf['corr'])
            if (signal_type in ['skewnormal']):
                put(grad_cf['alpha'])
            if (signal_type in self.custom_cf):
                put(grad_cf)

        #__ Harmonic functions __
        elif isinstance(signal_type, dict):
            if ('s' not in signal_type):
                put(grad_cf['s'][...,None])
            if not (('a' in signal_type) and ('b' in signal_type)):
                if (signal_type['sig'] == 'noise2D'):
                    put(grad_cf['a'][...,1:])
                    put(grad_cf['b'][...,1:])
                elif (signal_type['sig'] == 'noise3D'):
                    put(grad_cf['a'][..., self._a_free])
                    put(grad_cf['b'][..., self._b_free])

        #__ Consistency check __
        if (c != out.shape[-1]):
            raise RuntimeError(f"Gradient block filled {c} of {out.shape[-1]} columns.")

    #__ Dispatcher function __
    def _cf(
        self,
        sig    : str,
        kwargs : dict
    ) -> Tuple[np.ndarray,dict]:
        r"""Route to the characteristic function named by `sig`."""
        dispatch = {
            'point'      : self.cf_point,
            'normal'     : self.cf_normal,
            'skewnormal' : self.cf_skewnormal,
            'isonoise'   : self.cf_isonoise,
            'noise2D'    : self.cf_noise2D,
            'noise3D'    : self.cf_noise3D,
        }
        dispatch.update({sig_i:self.custom_cf[sig_i][1] for sig_i in self.custom_cf})
        if (sig == 'isoselfnoise'): return self._cf_isoselfnoise, {}
        if sig not in dispatch: raise ValueError(f"Unknown signal type '{sig}'.")
        return dispatch[sig](**kwargs)

    #__ Forward function __
    def forward(
        self,
        m    : np.ndarray,
        grad : bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
       
        r"""

        Compute the covariance and its gradient with respect to the model parameters.


        Input
        -----
        m : ``numpy.ndarray``
            Model vector
            (model, parameter)

        grad : ``bool`` (default ``True``)
            Calculate gradients

        Output
        ------
        cov : ``numpy.ndarray``
            Covariance
            (model, frequency, sensor pair)

        dcov_dm : ``numpy.ndarray`` or None
            Gradient of covariance wrt. model parameters
            (model, frequency, sensor pair, parameter)

        cov_noise : ``numpy.ndarray``
            Covariance (noise-only)
            (model, frequency, sensor pair)
        
        dcov_dm_noise : ``numpy.ndarray`` or None
            Gradient of covariance wrt. model parameters (noise-only)
            (model, frequency, sensor pair, parameter)

        
        Note
        ----
        
        If requested noise-only calculation (``calc_noise_only`` is ``True``), **cov_noise** and **dcov_dm_noise** will be returned in addition to **cov** and **dcov_dm**. 

        """

        #__ Instantiate covariance __
        N_model, N_param = m.shape
        cov = np.zeros((N_model, self.N_freq, self.N_pair), dtype='complex')
        dcov_dm = (np.zeros((N_model, self.N_freq, self.N_pair, N_param),
                            dtype='complex') if grad else None)
        if self._calc_noise_only:
            cov_noise = np.zeros((N_model, self.N_freq, self.N_pair), dtype='complex')
            dcov_dm_noise = (np.zeros((N_model, self.N_freq, self.N_pair, N_param),
                                    dtype='complex') if grad else None)

        #__ Unpack model vector __
        sig_list, kwargs_list, idx_list, weight_list = self._unpack(m, grad=grad)
        
        #__ Loop through signals __
        for signal_type, sig, kwargs, istart, iend, weight in zip(
                self.signal_types, sig_list, kwargs_list,
                idx_list[:-1], idx_list[1:], weight_list):

            #__ Calculate characteristic function __
            cf, grad_cf = self._cf(sig, kwargs)

            #__ Accumulate covariance __
            cov += weight * cf
            if self._calc_noise_only and (sig in self._NOISE_SIGS):
                cov_noise += weight * cf

            #__ Write gradients directly into the output view __
            if grad:
                if (self.weight_units == 'dB'):
                    dcov_dm[...,istart] = self._wscale * weight * cf
                    if self._calc_noise_only and (sig in self._NOISE_SIGS):
                        dcov_dm_noise[...,istart] = self._wscale * weight * cf
                elif (self.weight_units == 'linear'):
                    dcov_dm[...,istart] = cf
                    if self._calc_noise_only and (sig in self._NOISE_SIGS):
                        dcov_dm_noise[...,istart] = cf
                self._write_grad(signal_type, grad_cf,
                dcov_dm[...,istart+1:iend], weight)
                if self._calc_noise_only and (sig in self._NOISE_SIGS):
                    self._write_grad(signal_type, grad_cf,
                    dcov_dm_noise[...,istart+1:iend], weight)

        #__ Return output __
        if self._calc_noise_only: return cov, dcov_dm, cov_noise, dcov_dm_noise
        else: return cov, dcov_dm

    #__ Expand sensor pair axis __
    def expand(self, cov, dcov_dm):
        r"""(sensor pair) -> (sensor, sensor)"""

        #__ Shape of full matrix __
        N_model, _, _ = cov.shape
        if (dcov_dm is not None): _, _, _, N_param = dcov_dm.shape
        shp = (N_model, self.N_freq, self.N_sta, self.N_sta)

        #__ Array-covariance matrix __
        Gamma = np.zeros(shp, dtype=cov.dtype)
        Gamma[..., self._iu0_sta, self._iu1_sta] = cov
        Gamma[..., self._il0_sta, self._il1_sta] = Gamma[..., self._il1_sta, self._il0_sta].conj()

        #__ Gradient of array-covariance matrix __
        dGamma_dm = None
        if dcov_dm is not None:
            dGamma_dm = np.zeros(shp + (N_param,), dtype=dcov_dm.dtype)
            dGamma_dm[..., self._iu0_sta, self._iu1_sta, :] = dcov_dm
            dGamma_dm[..., self._il0_sta, self._il1_sta, :] = dGamma_dm[..., self._il1_sta, self._il0_sta, :].conj()
        
        #__ Return output
        return Gamma, dGamma_dm