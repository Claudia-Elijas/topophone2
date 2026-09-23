"""
    The ``topophone2.topophone`` module provides the ``decompose`` function,
    which performs the decomposition of an observed array-covariance matrix
    into its constituent signal components
    using a variational inference approach.
    The function takes in various parameters related to the observed data, receiver positions, signal types, and model configurations,
    and saves the estimated model parameters and loss history.

"""


from topophone2 import inversion, modelling
import numpy as np

def decompose(
    Gamma_obs         : np.ndarray,
    Gamma_obs_noise       : np.ndarray,
    N_obs          : int,
    N_obs_noise        : int,
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
    N_iterations    : int,
    N_montecarlo       : int,
    N_out          : int,
    save_intermediate_result : bool,
    verbose       : bool,
    N_samples     : int,
    model_outpath : str,
    loss_outpath  : str,
):
    r'''
    
    Function to perform the decomposition of an observed array-covariance matrix into its constituent signal components using a variational inference approach.
    
    Inputs
    ------

    Gamma_obs : ``numpy.ndarray``
        Observed array-covariance matrix
        (frequency, receiver, receiver)

    Gamma_obs_noise : ``numpy.ndarray``
        Observed noise-only array-covariance matrix
        (frequency, receiver, receiver)

    N_obs: ``int``
        Degrees of freedom

    N_obs_noise : ``int``
        Degrees of freedom (noise)

    x : ``numpy.ndarray``
        Receiver position [m]
        (receiver, direction)

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
    if (Gamma_obs_noise is not None) : calc_noise_only = True
    else:        calc_noise_only = False
    array_covariance_model = modelling.ArrayCovarianceModel(
        x               = x,
        freq            = freq,
        sigs    = sigs,
        weight_units    = weight_units,
        calc_noise_only = calc_noise_only,
        custom_cf       = custom_cf
    )

    #__ Set up prior pdf __
    prior = inversion.Uniform(*prior_bounds)

    #__ Set up posterior pdf __
    posterior = inversion.Posterior(
        Gamma_obs             = Gamma_obs,
        N_obs                 = N_obs,
        array_covariance_model = array_covariance_model,
        prior                 = prior,
        Gamma_obs_noise       = Gamma_obs_noise,
        N_obs_noise           = N_obs_noise
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
    inverse = inversion.Inference(
        flow_model    = flow,
        posterior     = posterior,
        learning_rate = learning_rate,
        outpath       = outpath
    )

    #__ Perform inversion __
    inverse.invert(
        N_iterations = N_iterations,
        N_montecarlo    = N_montecarlo,
        N_out       = N_out,
        save_intermediate_result = save_intermediate_result,
        verbose    = verbose
    )
    loss = inverse.loss_his

    #__ Sample posterior __
    m = inverse.sample(N_samples=N_samples)

    #__ Save models __
    if model_outpath: np.savetxt(model_outpath, m)
    if loss_outpath : np.savetxt(loss_outpath,  loss)

