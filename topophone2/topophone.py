"""
.. container:: module-intro

    The ``topophone2.topophone`` module provides the :func:`decompose` function,
    which performs the decomposition of the underlying wavefield using the observed array-covariance matrix
    using a variational inference approach.
    The function takes in various parameters related to the observed data, sensor positions, signal types, and model configurations,
    and saves the estimated model parameters and loss history.

"""

#________ IMPORT ________#

#__ Import: Numerical __
import numpy as np

#__ Import: Topophone2 __
from topophone2.modelling import ArrayCovarianceModel
from topophone2.inversion import Uniform, Posterior, LinearFlowsModel, Inference

#________ HELPER FUNCTION ________#

#__ Function to run Topophone2 core functionality __
def decompose(
    Gamma_obs       : np.ndarray = None,
    Gamma_obs_noise : np.ndarray = None,
    N_obs           : int        = None,
    N_obs_noise     : int        = None,
    x               : np.ndarray = None,
    freq            : np.ndarray = None,
    sigs            : list       = None,
    weight_units    : str        = 'dB',
    custom_cf       : dict       = {},
    prior_bounds    : tuple      = None,
    N_flow          : int        = None,
    mean_field      : bool       = False,
    ini_dist        : str        = 'Uniform',
    learning_rate   : float      = None,
    outpath         : str        = None,
    N_iterations    : int        = None,
    N_montecarlo    : int        = None,
    N_out           : int        = 0,
    save_intermediate_result : bool = False,
    verbose         : bool       = False,
    N_samples       : int        = None,
    calc_likelihood : bool       = False
):
    r'''
    
    Function to perform the decomposition of an observed array-covariance matrix into its constituent signal components using a variational inference approach.
    
    Input
    -----

    Gamma_obs : ``numpy.ndarray``
        Observed array-covariance matrix
        (frequency, receiver, receiver)

    Gamma_obs_noise : ``numpy.ndarray`` (optional)
        Observed noise-only array-covariance matrix
        (frequency, receiver, receiver)

    N_obs: ``int``
        Degrees of freedom

    N_obs_noise : ``int`` (optional)
        Degrees of freedom (noise)

    x : ``numpy.ndarray``
        Receiver position [m]
        (receiver, direction)

    freq : ``numpy.ndarray``
        Angular frequency vector [rad.s-1]
        (frequency)

    sigs : ``list``
        List of signals

    weight_units : ``str`` (default ``'dB'``)
        Units of the signal weights in the model vector
        ('dB' or 'linear')

    custom_cf : ``dict`` (optional)
        Custom characteristic functions keyed by signal name

    prior_bounds : ``(numpy.ndarray,numpy.ndarray)``
        Prior bounds

    N_flow : ``int``
        Number of normalising flows

    mean_field : ``bool`` (default ``False``)
        Use mean-field approximation

    ini_dist : ``str`` (default ``'Uniform'``)
        Initial distribution to transform

    learning_rate : ``float``
        Learning rate for the variational inference

    outpath : ``str``
        Output path for saving results

    N_iterations : ``int``
        Number of iterations to run optimiser

    N_montecarlo : ``int``
        Number of samples to estimate ELBO

    N_out : ``int`` (default ``0``)
        Number of print statements and/or intermediate saves

    save_intermediate_result : ``bool`` (default ``False``)
        Save intermediate results during iteration

    verbose : ``bool`` (default ``False``)
        Print progress statements

    N_samples : ``int`` (optional)
        Number of samples to draw from posterior solution

    calc_likelihood : ``bool`` (default ``False``)
        Calculate likelihood
    '''

    #__ Forward modeller __
    if (Gamma_obs_noise is not None) : calc_noise_only = True
    else:        calc_noise_only = False
    array_covariance_model = ArrayCovarianceModel(
        x               = x,
        freq            = freq,
        sigs            = sigs,
        weight_units    = weight_units,
        calc_noise_only = calc_noise_only,
        custom_cf       = custom_cf
    )

    #__ Uniform prior pdf __
    prior = Uniform(*prior_bounds)

    #__ Posterior pdf __
    posterior = Posterior(
        Gamma_obs             = Gamma_obs,
        N_obs                 = N_obs,
        array_covariance_model = array_covariance_model,
        prior                 = prior,
        Gamma_obs_noise       = Gamma_obs_noise,
        N_obs_noise           = N_obs_noise
    )

    #__ Normalising flow model __
    flow_model = LinearFlowsModel(
        N_param      = len(prior_bounds[0]),
        prior_bounds = prior_bounds,
        N_flow       = N_flow,
        mean_field   = mean_field,
        ini_dist     = ini_dist
    )

    #__ Set up inverse problem __
    inverse_problem = Inference(
        flow_model    = flow_model,
        posterior     = posterior,
        learning_rate = learning_rate,
        outpath       = outpath
    )

    #__ Perform inversion __
    inverse_problem.invert(
        N_iterations = N_iterations,
        N_montecarlo = N_montecarlo,
        N_out        = N_out,
        save_intermediate_result = save_intermediate_result,
        verbose      = verbose
    )

    #__ Save flow model __
    inverse_problem.save()

    #__ Sample posterior __
    if (N_samples is not None):
        m = inverse_problem.sample(N_samples=N_samples)
        np.savetxt(f"{outpath}/model-samples_iteration-{N_iterations:05d}.txt", m)

        #__ Calculate likelihood __
        if calc_likelihood:
            lnp_dm, _ = posterior.log_likelihood(m=m, grad=False)
            np.savetxt(f"{outpath}/log-likelihood_iteration-{N_iterations:05d}.txt", lnp_dm)