"""
.. container:: module-intro

    The ``topophone2.utils.dsp`` module provides utilities for digital signal processing. Note that the engineering phase convention is used throughout.
"""

#________ IMPORT ________#

#__ Numerical __
import numpy as np

#__ Signal processing __
from scipy.fft import next_fast_len
from scipy.signal.windows import tukey, dpss

#________ DOMAIN TRANSFORMS ________#

#__ Time -> Frequency __
def rfft(
    u_t   : np.ndarray,
    dt    : float,
    nfft  : int = None,
    alpha : float = None
):
    """
    Transform real time series to frequency domain (engineering phase convention).

    Input
    -----
    u_t : ``numpy.ndarray``
        Time series
        (..., time)

    dt : ``float``
        Sampling interval [s]

    nfft : ``int`` (optional)
        Number of points in FFT

    alpha : ``float`` (optional)
        Pre-FFT taper fraction

    Output
    ------
    u_f : ``numpy.ndarray``
        Complex spectrum
        (..., frequency)

    f : ``numpy.ndarray``
        Frequency vector [Hz]
        (frequency,)

    nfft : ``int``
        Number of points in FFT
    """

    #__ Calculate number of points in FFT __
    npts = u_t.shape[-1]
    if (nfft is None):
        nfft = 2 * next_fast_len(-(-npts//2), real=True)

    #__ Apply taper __
    if (alpha is not None): u_t = u_t * tukey(npts, alpha)

    #__ Perform FFT __
    u_f = np.fft.rfft(u_t, n=nfft, axis=-1)

    #__ Calculate frequency vector __
    f = np.fft.rfftfreq(n=nfft, d=dt)

    #__ Return output __
    return u_f, f, nfft

#__ Frequency -> Time __
def irfft(
    u_f  : np.ndarray,
    dt   : float,
    nfft : int,
    npts : int = None
):
    """
    Transform complex spectrum to real time series (engineering phase convention).

    Input
    -----
    u_f : ``numpy.ndarray``
        Complex spectrum
        (..., frequency)

    dt : ``float``
        Sampling interval [s]

    nfft : ``int``
        Number of points in FFT

    npts : ``int`` (optional)
        Number of points in output time series

    Output
    ------
    u_t : ``numpy.ndarray``
        Time series
        (..., time)

    t : ``numpy.ndarray``
        Time vector [s]
        (time,)

    npts : ``int``
        Number of points in output time series
    """

    #__ Perform inverse FFT __
    u_t = np.fft.irfft(u_f, n=nfft, axis=-1)

    #__ Truncate padding __
    if (npts is not None): u_t = u_t[..., :npts]
    else: npts = nfft

    #__ Calculate time vector [s] __
    t = np.arange(npts) * dt

    #__ Return output __
    return u_t, t, npts

#________ BEAMFORMING ________#

#__ Calculate time shifts __
def _calc_shift(
    x        : np.ndarray,
    p_bounds : list,
    dp       : float
):
    """
    Calculate the time shift for an ensemble of slowness vectors.

    Input
    -----
    x : ``numpy.ndarray``
        Receiver location [m]
        (receiver, direction)

    p_bounds : ``list``
        (Minimum, maximum) slowness to search in each direction [s.m-1]

    dp : ``float``
        Slowness search interval [s.m-1]

    Output
    ------
    shift : ``numpy.ndarray``
        Time shifts [s]
        (slowness, receiver)

    p : ``numpy.ndarray``
        Slowness vectors [s.m-1]
        (slowness, direction)
    """

    #__ Construct slowness vectors __
    p_cube = np.meshgrid(*[
        np.linspace(p_min, p_max, int(round((p_max-p_min)/dp))+1)
        for (p_min,p_max) in p_bounds
    ], indexing='ij')
    p = np.vstack([p_i.ravel() for p_i in p_cube]).T

    #__ Calculate travel times __
    shift = np.einsum('ik,jk->ij', p, x)

    #__ Return output __
    return shift, p

#__ Apply time shift __
def _apply_shift(
    u_f         : np.ndarray,
    f           : np.ndarray,
    shift       : np.ndarray,
    dt          : float = None,
    nfft        : int   = None,
    npts        : int   = None,
    return_time : bool  = False
):
    """
    Apply a time shift to a signal in the frequency domain.

    Input
    -----
    u_f : ``numpy.ndarray``
        Complex spectrum
        (receiver, frequency)

    f : ``numpy.ndarray``
        Frequency vector [Hz]
        (frequency,)

    shift : ``numpy.ndarray``
        Time shift [s]
        (slowness, receiver)

    dt : ``float`` (optional)
        Sampling interval [s]

    nfft : ``int`` (optional)
        Number of points in FFT

    npts : ``int`` (optional)
        Number of points in time series

    return_time : ``bool`` (default ``False``)
        Return signal in the time domain

    Output
    ------
    ushift_f/ushift_t : ``numpy.ndarray``
        Shifted signal in the frequency/time domain
        (slowness, receiver, frequency/time)

    phase_shift : ``numpy.ndarray``
        Phase-rotation scalar applied to signal
        (slowness, receiver, frequency)
    """

    #__ Calculate phase shift __
    w           = 2 * np.pi * f
    phase_shift = np.exp(1j * shift[...,None] * w)

    #__ Apply phase shift __
    ushift_f = u_f[None,...] * phase_shift

    #__ Transform to time domain __
    if return_time: ushift_t, _, _ = irfft(ushift_f, dt=dt, nfft=nfft, npts=npts)

    #__ Return output __
    if return_time: return ushift_t, phase_shift
    else:           return ushift_f, phase_shift

#__ Apply time window to signal __
def apply_window(
    u_t   : np.ndarray,
    t     : np.ndarray,
    t_w   : float,
    T_w   : float
):
    """
    Apply time window to signal.

    Input
    -----
    u_t : ``numpy.ndarray``
        Time series
        (..., time)

    t : ``numpy.ndarray``
        Time vector [s]
        (time,)

    t_w : ``float``
        Time-window centre [s]

    T_w : ``float``
        Time-window half-width [s]

    Output
    ------
    u_t_w : ``numpy.ndarray``
        Windowed time series
        (..., time)

    npts_w : ``int``
        Length of time window
    """

    #__ Calculate window __
    window = (np.abs(t - t_w) < T_w)
    npts_w = np.sum(window)

    #__ Window __
    u_t_w = u_t[...,window]

    #__ Return output __
    return u_t_w, npts_w

#__ Calculate the best beam DOA __
def best_beam_DOA(
    u_t      : np.ndarray,
    t        : np.ndarray,
    x        : np.ndarray,
    p_bounds : list,
    dp       : float,
    t_w      : float = None,
    T_w      : float = None,
    alpha_maj : float = None,
    alpha_min : float = None
):
    """
    Estimate the direction-of-arrival of the best beam.

    Input
    -----
    u_t : ``numpy.ndarray``
        Time series
        (receiver, time)

    t : ``numpy.ndarray``
        Time vector [s]
        (time,)

    x : ``numpy.ndarray``
        Receiver location [m]
        (receiver, direction)

    p_bounds : ``list``
        (Minimum, maximum) slowness to search in each direction [s.m-1]

    dp : ``float``
        Slowness search interval [s.m-1]

    t_w : ``float`` (optional)
        Time-window centre [s]

    T_w : ``float`` (optional)
        Time-window half-width [s]

    alpha_maj : ``float`` (optional)
        Pre-FFT taper fraction

    alpha_min : ``float`` (optional)
        Time-window taper fraction

    Output
    ------
    ushift_t_best : ``numpy.ndarray``
        Time-shifted time series
        (receiver, time)

    p_best : ``numpy.ndarray``
        Best-beam slowness vector [s.m-1]
        (direction,)

    shift_best : ``numpy.ndarray``
        Best-beam time shift [s]
        (receiver,)

    npts_w : ``int``
        Number of points in time window
    """

    #__ Extract time variables __
    N_rec, npts = u_t.shape
    dt          = t[1] - t[0]

    #__ Convert to frequency domain __
    u_f, f, nfft = rfft(u_t, dt, alpha=alpha_maj)

    #__ Calculate time shifts __
    shift, p = _calc_shift(x, p_bounds, dp)

    #__ Apply time shifts __
    ushift_t, phase_shift = _apply_shift(u_f, f, shift, dt, nfft, npts, return_time=True)

    #__ Apply window __
    if (t_w is not None) and (T_w is not None):
        ushift_t_w, npts_w = apply_window(ushift_t, t, t_w, T_w)
    else: ushift_t_w, npts_w = ushift_t, npts

    #__ Calculate F-stat taper __
    if (alpha_min is not None): taper = tukey(npts_w, alpha_min)
    else:                       taper = np.ones(npts_w)
    
    #__ Calculate F-statistic __
    sum_square_mean = np.sum(np.mean((ushift_t_w*taper), axis=1)**2, axis=1)
    sum_mean_square = np.sum(np.mean((ushift_t_w*taper)**2, axis=1), axis=1)
    F_statistic     = (N_rec - 1) * sum_square_mean / (sum_mean_square - sum_square_mean)
    
    #__ Extract maximum __
    idx_best      = np.argmax(F_statistic)
    p_best        = p[idx_best]
    shift_best    = shift[idx_best]
    ushift_t_best = ushift_t_w[idx_best]

    #__ Return output __
    return ushift_t_best, p_best, shift_best, npts_w

#__ Create Slepian tapers __
def _slepian(
    npts : int,
    dt   : float,
    bw   : float
):
    """
    Calculate set of Slepian tapers.

    Input
    -----
    npts : ``int``
        Number of points in time vector

    dt : ``float``
        Sampling interval [s]

    bw : ``float``
        Full bandwidth [Hz]

    Output
    ------
    taper : ``numpy.ndarray``
        Slepian tapers
        (taper, time)

    T_bw : ``float``
        Time-bandwidth product

    N : ``int``
        Number of tapers
    """

    #__ Slepian-sequence parameters __
    T    = npts * dt              # Total time duration [s]
    T_bw = T * bw                 # Time-bandwidth product
    N    = int(np.floor(T_bw)-1)  # Number of tapers

    #__ Calculate Slepian sequences __
    taper = dpss(M=npts, NW=T_bw/2, Kmax=N)

    #__ Return output __
    return taper, T_bw, N

#__ Calculate sample-covariance matrix __
def sample_covariance_matrix(
    u_t   : np.ndarray,
    npts  : int,
    dt    : float,
    bw    : float,
    correction : np.ndarray = None
):
    """
    Calculate the sample-covariance matrix.

    Input
    -----
    u_t : ``numpy.ndarray``
        Time series
        (receiver, time)

    npts : ``int``
        Number of points in time series

    dt : ``float``
        Sampling interval [s]

    bw : ``float``
        Slepian-sequence full bandwidth [Hz]

    correction : ``numpy.ndarray`` (optional)
        Time shifts to be applied post-FFT [s]
        (receiver,)

    Output
    ------
    scm : ``numpy.ndarray``
        Sample-covariance matrix
        (receiver, receiver, frequency)

    f : ``numpy.ndarray``
        Frequency vector [Hz]
        (frequency,)

    N : ``int``
        Number of tapers used to estimate array-covariance

    T_bw : ``float``
        Time-bandwidth product of Slepian sequences
    """

    #__ Calculate Slepian tapers __
    taper, T_bw, N = _slepian(npts, dt, bw)

    #__ Apply Slepian tapers __
    utap_t = taper[:,None,:] * u_t[None,:,:]

    #__ Transform to frequency domain __
    utap_f, f, nfft = rfft(utap_t, dt)

    #__ Apply correction __
    if (correction is not None):
        utap_f, _ = _apply_shift(utap_f, f, correction)
        utap_f    = utap_f[0]

    #__ Calculate sample-covariance matrix __
    scm = np.mean(
        utap_f[:,:,None,:] * np.conj(utap_f[:,None,:,:]),
        axis=0
    )

    #__ Return output __
    return scm, f, N, T_bw