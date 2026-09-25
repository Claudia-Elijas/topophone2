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
        Apply taper befor FFT
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


# !!! APPLY TIME SHIFT FUNCTION TO TAKE INPUT FREQUENCY OR TIME AND OUTPUT FREQUENCY OR TIME !!! #

#__ Apply time shift __
def apply_time_shift(
    u_t   : np.ndarray,
    dt    : float,
    shift : np.ndarray,
    taper : np.ndarray = None,
    return_time : bool = False
):
    """
    Apply a time shift in the frequency domain.

    Input
    -----
    u_t : ``numpy.ndarray``
        Time series
        (..., time)

    dt : ``float``
        Sampling interval [s]

    shift : ``numpy.ndarray``
        Time shift [s]
        (...)

    taper : ``numpy.ndarray``
        Apply taper before FFT
        (time,)

    return_time : ``bool`` (default ``False``)
        Return in time domain
    """

    #__ Transform to frequency domain __
    u_f, f, nfft = rfft(u_t=u_t, dt=dt, taper=taper)

    #__ Calculate phase shifts __
    w = 2 * np.pi * f
    phase_shift = np.exp(-1j * w * shift[...,None])

    #__ Apply phase shift __
    u_shift = u_f * phase_shift

    #__ Transform to time domain __
    if return_time:
        npts = u_t
        u_shift, _, _ = irfft(u_f=u_shift, dt=dt)