"""
.. container:: module-intro

    The ``topophone2.utils.dsp`` module provides utilities for digital signal processing.
"""

#________ IMPORT ________#

#__ Numerical __
import numpy as np

#__ Signal processing __
from scipy.fft import next_fast_len

#________ DOMAIN TRANSFORMS ________#

#__ Time -> Frequency __
def rfft(
    u_t   : np.ndarray,
    dt    : float,
    nfft  : int = None,
    taper : np.ndarray = None
):
    """
    Tranform real time series to frequency domain.

    Input
    -----
    u_t : ``numpy.ndarray``
        Time series
        (..., time)

    dt : ``float``
        Sampling interval [s]

    nfft : ``int`` (optional)
        Number of points in FFT

    taper : ``numpy.ndarray`` (optional)
        Apply taper before FFT
        (time,)

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
    if (nfft is None):
        npts = u_t.shape[-1]
        nfft = next_fast_len(npts)
        if nfft % 2: nfft = next_fast_len(nfft+1)

    #__ Apply taper __
    if (taper is not None): u_t *= taper

    #__ Perform FFT __
    u_f = np.fft.rfft(u_t, nfft=nfft, axis=-1)

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
    Transform complex spectrum to real time series.

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
        Slowness search interval [m.s-1]

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
        np.arange(p_min, p_max+dp, dp)
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
    phase_shift = np.exp(- 1j * shift[...,None] * w)

    #__ Apply phase shift __
    ushift_f = u_f[None,...] * phase_shift

    #__ Transform to time domain __
    if return_time: ushift_t, _, _ = irfft(ushift_f, dt=dt, nfft=nfft, npts=npts)

    #__ Return output __
    if return_time: return ushift_t, phase_shift
    else:           return ushift_f, phase_shift

#__ Calculate the best beam DOA __
def best_beam_DOA(
    u_t      : np.ndarray,
    t        : np.ndarray,
    x        : np.ndarray,
    p_bounds : list,
    dp       : float,
    t_w      : float,
    T_w      : float,
    taper    : np.ndarray = None,
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
        Slowness search interval [m.s-1]

    t_w : ``float``
        Central time of time interval [s]

    T_w : ``float``
        Half-width of time interval [s]

    taper : ``np.ndarray`` (optional)
        Apply taper before FFT
        (time,)

    Output
    ------
    p_best : ``numpy.ndarray`
        Best-beam slowness vector [s.m-1]
        (direction,)

    shift_best : ``numpy.ndarray``
        Best-beam time shift [s]
        (receiver,)
    """

    #__ Extract time variables __
    N_rec, npts = u_t.shape
    dt          = t[1] - t[0]
    window      = (np.abs(t - t_w) < T)

    #__ Convert to frequency domain __
    u_f, f, nfft = rfft(u_t, dt, taper=taper)

    #__ Calculate time shifts __
    shift, p = _calc_shift(x, p_bounds, dp)

    #__ Apply time shifts __
    ushift_t, phase_shift = _apply_shift(u_f, f, shift, dt, nfft, npts, return_time=True)

    #__ Calculate F-statistic __
    sum_square_mean = np.sum(np.mean(ushift_t[...,window], axis=1)**2, axis=1)
    sum_mean_square = np.sum(np.mean(ushift_t[...,window]**2, axis=1), axis=1)
    F_statistic     = (N_rec - 1) * sum_square_mean / (sum_mean_square - sum_square_mean)
    
    #__ Extract maximum __
    idx_best   = np.argmax(F_statistic)
    p_best     = p[idx_best]
    shift_best = shift[idx_best]

    #__ Return output __
    return p_best, shift_best