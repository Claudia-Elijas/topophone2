"""
    The ``topophone2.topophone`` module provides the ``decompose`` function, which performs the decomposition of an observed array-covariance matrix into its constituent signal components using a variational inference approach. The function takes in various parameters related to the observed data, sensor positions, signal types, and model configurations, and returns the estimated model parameters and loss history.

"""


from topophone2 import inversion, modelling
from typing import int 
import numpy as np

def decompose(
    Gamma         : np.ndarray,
    Gamma_n       : np.ndarray,
    Nobs          : int,
    Nobs_n        : int,
    x             : np.ndarray,
    freq          : np.ndarray,
    sigs          : list,
    weight_units  : str,
    custom_cf     : dict,
    prior_bounds  : tuple,
    N_flow        : int,
    mean_field    : bool,
    ini_dist      : str,
    learning_rate : float,
    outpath       : str ,
    iterations    : int,
    nsample       : int,
    nout          : int,
    save_intermediate_result : bool,
    verbose       : bool,
    N_samples     : int,
    model_outpath : str,
    loss_outpath  : str,
) -> np.ndarray:
    r'''
    
    Function to perform the decomposition of an observed array-covariance matrix into its constituent signal components using a variational inference approach.
    
    Inputs
    ------

    Gamma : ``numpy.ndarray``
        Observed array-covariance matrix
        (frequency, sensor, sensor)

    Gamma_n : ``numpy.ndarray``
        Observed noise-only array-covariance matrix
        (frequency, sensor, sensor)

    Nobs: ``int``
        Degrees of freedom

    Nobs_n : ``int``
        Degrees of freedom (noise)

    x : ``numpy.ndarray``
        Sensor position [m]
        (sensor, direction)

    freq : ``numpy.ndarray``
        Angular frequency vector [rad.s-1]
        (frequency)

    sigs : ``list``
        List of signals

    weight_units : ``str``
        Units of the signal weights in the model vector
        ('dB' or 'linear')

    custom_cf : ``dict``
        Custom characteristic functions keyed by signal name

    prior_bounds : ``(numpy.ndarray,numpy.ndarray)``
        Prior bounds

    N_flow : ``int``
        Number of normalising flows

    mean_field : ``bool``
        Use mean-field approximation

    ini_dist : ``str``
        Initial distribution to transform

    learning_rate : ``float``
        Learning rate for the variational inference

    outpath : ``str``
        Output path for saving intermediate results


    '''

    #__ Set up array-covariance model __
    if Gamma_n : calc_noise_only = True
    else:        calc_noise_only = False
    model = modelling.ArrayCovarianceModel(
        x               = x,
        freq            = freq,
        signal_types    = sigs,
        weight_units    = weight_units,
        calc_noise_only = calc_noise_only,
        custom_cf       = custom_cf
    )

    #__ Set up prior pdf __
    prior = inversion.Uniform(*prior_bounds)

    #__ Set up posterior pdf __
    posterior = inversion.Posterior(
        Gamma_obs             = Gamma,
        N_obs                 = Nobs,
        arraycovariance_model = model,
        prior                 = prior,
        Gamma_obs_noise       = Gamma_n,
        N_obs_noise           = Nobs_n
    )

    #__ Set up normalising flow model __
    flow = inversion.LinearFlowsModel(
        N_param      = len(prior_bounds[0]),
        prior_bounds = prior_bounds,
        N_flow       = N_flow,
        mean_field   = mean_field,
        ini_dist     = ini_dist
    )

    #__ Set up inverse problem __
    inverse = inversion.VI(
        flow_model    = flow,
        posterior     = posterior,
        learning_rate = learning_rate,
        outpath       = outpath
    )

    #__ Perform inversion __
    inverse.invert(
        iterations = iterations,
        nsample    = nsample,
        nout       = nout,
        save_intermediate_result = save_intermediate_result,
        verbose    = verbose
    )
    loss = inverse.loss_his

    #__ Sample posterior __
    m = inverse.sample(N_samples=N_samples)

    #__ Save models __
    if model_outpath: np.savetxt(model_outpath, m)
    if loss_outpath : np.savetxt(loss_outpath,  loss)

