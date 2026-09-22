"""

.. container:: module-intro 

    
    The ``topophone2.inversion`` module provides functionality for inversion by variational inference.
    It contains classes and functions to perform variational Bayesian inference
    by normalising flows, inverting observed array covariance for the
    parameterised incoherently-distributed wavefield energy.


"""

#________ IMPORT ________#

#__ System __
from typing import Tuple, Optional, Dict, Callable

#__ Numerical __
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch
torch.set_default_dtype(torch.float64)

#__ Inversion __
from geopvi.nfvi.models import FlowsBasedDistribution, VariationalInversion
from geopvi.nfvi.flows import Real2Constr, Constr2Real, Linear
from geopvi.forward.swi.posterior import ForwardModel

#________ PROBABILITY DENSITY FUNCTIONS ________#

#__ Uniform prior __
class Uniform():

    '''

        Class to calculate log-prior for uniform distribution with lower and upper bounds.

        Methods
        -------

        - :meth:`log_prior` : Calculate log-prior for uniform distribution

        Input
        -----

        lower : ``numpy.ndarray``
            Lower bounds for uniform distribution
            (parameter,)

        upper : ``numpy.ndarray``
            Upper bounds for uniform distribution
            (parameter,)

        
        Attributes
        ----------
        lower : ``torch.Tensor``
            Lower bounds for uniform distribution
            (parameter,)
        
        upper : ``torch.Tensor``
            Upper bounds for uniform distribution
            (parameter,)
            
    '''

    #__ Initialise class __
    def __init__(
        self,
        lower : np.ndarray,
        upper : np.ndarray
    ):

        #__ Set prior bound arguments __
        self.lower = torch.from_numpy(lower)
        self.upper = torch.from_numpy(upper)

    #__ Calculate log-prior __
    def log_prior(
        self,
        m : np.ndarray
    ):

        '''
        Function to calculate the log-prior for a uniform distribution.

        Input
        -----
        m : ``numpy.ndarray``
            Model vector
            (model, parameter)

        Output
        ------ 
        lnp_m : ``torch.Tensor``
            Log-prior for uniform distribution
            (model,)

        '''

        #__ Calculate if model in bounds __
        in_bounds = torch.logical_and(
            m >= self.lower,
            m <= self.upper
        )

        #__ Calculate log-prior __
        lnp_mi = torch.where(
            in_bounds,
            - torch.log(self.upper - self.lower),
            torch.full_like(m, float('-inf'))
        )
        lnp_m  = lnp_mi.sum(axis=-1)

        #__ Return output __
        return lnp_m

#__ Posterior distribution __
class Posterior():
    r'''
    
    Class to calculate log-posterior for array-covariance inversion.

    Methods 
    -------
    - :meth:`log_likelihood` : Calculate log-likelihood of data given model. 
    - :meth:`log_posterior` : Calculate log-posterior for model given data and prior. 

    Input
    -----
    Gamma_obs : ``numpy.ndarray``
        Observed array-covariance matrix
        (frequency, sensor, sensor)
    
    N_obs : ``int``
        Number of independent samples used to estimate observed expectation

    arraycovariance_model : ``Callable``
        Array covariance model (:class:`ArrayCovarianceModel`)

    prior : ``Callable``
        Prior distribution (:class:`Uniform`)

    Gamma_obs_noise : ``numpy.ndarray`` (optional)
        Observed noise-only array-covariance matrix
        (frequency, sensor, sensor)

    N_obs_noise : ``int`` (optional)
        Number of independent samples used to estimate observed noise-only expectation


    Attributes
    ----------
    N_obs : ``float``
        Number of independent samples used to estimate observed expectation
    N_obs_noise : ``float``
        Number of independent samples used to estimate observed noise-only expectation
    prior : ``Callable``
        Prior distribution
    forward : ``Callable``
        Function to predict array covariance from model vector
  
    '''

    def __init__(
        self,
        Gamma_obs : np.ndarray,
        N_obs     : int,
        arraycovariance_model : Callable,
        prior     : Callable,
        Gamma_obs_noise : Optional[np.ndarray] = None,
        N_obs_noise     : Optional[int]        = None
    ):
        

        #__ Throw error if noise-only not calculated but noise ACM specified here, or warning if specified but not provided __
        self._invert_noise_only = (Gamma_obs_noise is not None)  # Invert an additional noise-only data set

        #__ Store arguments as attributes __
        self.N_obs       = float(N_obs)            # Number of independent samples used to estimate observed expectation
        if self._invert_noise_only:
            self.N_obs_noise = float(N_obs_noise)  # Number of independent samples used to estimate observed noise-only expectation
        self.prior       = prior                   # Prior distribution

        #__ Store required attributes from model __
        self.forward  = arraycovariance_model.forward   # Function to predict array covariance from model vector
        self._iu0_sta = arraycovariance_model._iu0_sta  # Upper-triangle index set for unique station pairs (including diagonal)
        self._iu1_sta = arraycovariance_model._iu1_sta  # ^
        self._N_sta   = arraycovariance_model.N_sta     # Number of sensors
        self._N_freq  = arraycovariance_model.N_freq    # Number of frequency bins
        self._calc_noise_only = arraycovariance_model._calc_noise_only  # Include an additional noise-only calculation
        self._unpack  = arraycovariance_model._unpack   # Unpack model vector
        self._expand  = arraycovariance_model.expand    # Unflatten array-covariance

        #__ Lower-tirangle index set for unique station pairs (excluding diagonal) __
        self._il0_sta, self._il1_sta = np.tril_indices(self._N_sta, k=-1)

        #__ Check Hermitian symmetry of observation __
        def check_hermitian(Gamma_obs, var_name):
            if not np.allclose(Gamma_obs,
                            np.conj(np.swapaxes(Gamma_obs, -1, -2))):
                raise ValueError(f"{var_name} is not Hermitian.")
        check_hermitian(Gamma_obs, 'Gamma_obs')
        if self._invert_noise_only: check_hermitian(Gamma_obs_noise, 'Gamma_obs_noise')

        #__ Scaled Gamma_obs (A in Goodman [1963]) __
        self._A = self.N_obs * Gamma_obs  # (frequency, sensor, sensor)
        if self._invert_noise_only: self._A_noise = self.N_obs_noise * Gamma_obs_noise

        #__ Cholesky factorisation of A __
        self._L_A = self._safe_cholesky(self._A, 'Gamma_obs')
        if self._invert_noise_only: self._L_A_noise = self._safe_cholesky(self._A_noise, 'Gamma_obs_noise')

    #__ Safe Cholesky decomposition __
    @staticmethod
    def _safe_cholesky(A, var_name):
        try: L_A = np.linalg.cholesky(A)
        except np.linalg.LinAlgError as err:
            raise ValueError(f"{var_name} is not positive definite.") from err
        return L_A

    #__ Calculate log-likelihood helper __
    def _lnL_and_grad(self, Gamma, dGamma_dm, A, N_obs, grad, Gamma_name):

        #__ Cholesky decomposition of predicted array-covariance matrix __
        L = self._safe_cholesky(Gamma, Gamma_name)  # (model, freq, s, s)

        #__ Log-determinant of Gamma __
        diag   = np.real(np.diagonal(L, axis1=-2, axis2=-1))  # (model, freq, sensor)
        logdet = 2.0 * np.sum(np.log(diag), axis=-1)          # (model, freq)

        #__ Inverse of Gamma __
        Ginv = np.linalg.inv(Gamma)  # (model, freq, sensor, sensor)

        #__ tr(G^-1 A) __
        tr = np.real(np.sum(Ginv * np.conj(A)[None,...], axis=(-2, -1)))  # (model, freq)

        #__ Sum over independent frequency bins __
        lnp_d_m = np.sum(-N_obs * logdet - tr, axis=-1)   # (model)
        if not grad: return lnp_d_m, None

        #__ Calculate gradient (model, parameter) __
        B            = A[None, ...] - N_obs * Gamma
        G            = Ginv @ B @ Ginv
        grad_lnp_d_m = np.real(np.einsum('mfij,mfjik->mk', G, dGamma_dm, optimize=True))

        #__ Return output __
        return lnp_d_m, grad_lnp_d_m

    #__ Calculate log-likelihood __
    def log_likelihood(self, m, grad=True):
        r"""

        Function to calculate the log-likelihood of the observed array-covariance matrix given a model vector.

        Input
        -----
        m : ``numpy.ndarray``
            Model vector
            (model, parameter)

        grad : ``bool`` (default: ``True``)
            If ``True``, calculate gradient of log-likelihood wrt. model parameters

        Output
        ------
        lnp_d_m : ``numpy.ndarray``
            Log-likelihood of observed array-covariance matrix given model vector
            (model,)

        grad_lnp_d_m : ``numpy.ndarray``
            Gradient of log-likelihood wrt. model parameters
            (model, parameter)

        """

        #__ Extract model shape (model, parameter) __
        if not isinstance(m, np.ndarray):
            m = m.detach().numpy()
        m = np.atleast_2d(m)
        N_model, N_param = m.shape

        #__ Synthesise array-covariance matrix __
        if self._calc_noise_only: cov, dcov_dm, cov_noise, dcov_dm_noise = self.forward(m, grad=grad)  # (model, frequency, sensor pair, [parameter])
        else: cov, dcov_dm = self.forward(m, grad=grad)  # (model, frequency, sensor pair, [parameter])
        Gamma, dGamma_dm = self._expand(cov, dcov_dm if grad else None)
        if self._invert_noise_only: Gamma_noise, dGamma_dm_noise = self._expand(cov_noise, dcov_dm_noise if grad else None)

        #__ Calculate likelihood __
        lnp_d_m, grad_lnp_d_m = self._lnL_and_grad(Gamma, dGamma_dm, self._A, self.N_obs, grad, 'Gamma')
        if self._invert_noise_only:
            lnp_d_m_noise, grad_lnp_d_m_noise = self._lnL_and_grad(Gamma_noise, dGamma_dm_noise, self._A_noise, self.N_obs_noise, grad, 'Gamma_noise')
            lnp_d_m += lnp_d_m_noise
            if (grad_lnp_d_m is not None): grad_lnp_d_m += grad_lnp_d_m_noise

        #__ Return output __
        return lnp_d_m, grad_lnp_d_m

    #__ Calculate log-posterior __
    def log_posterior(
        self,
        m : np.ndarray
    ):

        r'''
        
        Function to calculate the log-posterior for a model vector given observed array-covariance matrix and prior distribution.

        Input
        -----
        m : ``numpy.ndarray``
            Model vector
            (model, parameter)

        Output
        ------
        lnp_m_d : ``torch.Tensor``
            Log-posterior for model vector given observed array-covariance matrix and prior distribution
            (model,)
        
        
        '''

        #__ Calculate log-likelihood __
        lnp_d_m = ForwardModel.apply(m, self.log_likelihood)

        #__ Calculate log-prior __
        lnp_m = self.prior.log_prior(m)

        #__ Calculate log-posterior __
        lnp_m_d = lnp_m + lnp_d_m

        #__ Return output __
        return lnp_m_d

#________ NORMALISING FLOWS MODEL ________#

class LinearFlowsModel():

    ''' 

    Class to define a normalising flows model for variational inference.

    Methods
    -------
    - :meth:`def_flow_model` : Define a normalising flows model for variational inference.
    - :meth:`get_flow_param` : Get the parameters of the normalising flows model.

    Input 
    -----
    N_param : ``int``
        Number of model parameters

    N_flow : ``int``
        Number of normalising flows 

    prior_bounds : ``tuple`` (default: ``None``)
        Lower and upper bounds for uniform prior distribution

    mean_field : ``bool`` (default: ``False``)
        If ``True``, use mean-field approximation for flows model
    
    ini_dist : ``str`` (default: ``"Uniform"``)
        Initial distribution for flows model. Options are ``"Uniform"`` or ``"Normal"`` 

    
    Attributes
    ----------
    N_param : ``int``
        Number of model parameters
    N_flow : ``int``
        Number of normalising flows
    prior_bounds : ``tuple``
        Lower and upper bounds for uniform prior distribution
    mean_field : ``bool``
        If ``True``, use mean-field approximation for flows model
    ini_dist : ``str``
        Initial distribution for flows model. Options are ``"Uniform"`` or ``"Normal"``

    '''

    #__ Initialise model __
    def __init__(self, N_param, N_flow, prior_bounds=None, mean_field=False, ini_dist="Uniform"):
        self.N_param      = N_param
        self.prior_bounds = prior_bounds
        self.N_flow       = N_flow
        self.mean_field   = mean_field
        self.ini_dist     = ini_dist

    #__ Flows model __
    def def_flow_model(self):

        '''
        Function to define a normalising flows model for variational inference.
        '''

        #__ Initialise flow __
        flow_type = "Linear"
        flow  = eval(flow_type)
        param = None

        #__ Select kernel __
        if self.mean_field: 
            kernel = "diagonal"
        else:
            kernel = "fullrank"

        #__ Create flow model __
        mask = np.ones((self.N_param, self.N_param))
        flows = [flow(dim=self.N_param, kernel=kernel, mask=mask, param=param) for _ in range(self.N_flow)]

        #__ Add transforms between real and constrained spaces __
        if (self.ini_dist == "Uniform"):
            flows.insert(0, Constr2Real(dim=self.N_param, lower=0, upper=1))
        flows.append(Real2Constr(dim=self.N_param, lower=self.prior_bounds[0], upper=self.prior_bounds[1]))

        #__ Create flows model __
        self.variational = FlowsBasedDistribution(flows, base=self.ini_dist)

        #__ Return output __
        return self

    #__ Get flow parameters __
    def get_flow_param(self):
        '''
        Function to get the parameters of the normalising flows model for variational inference.
        '''
        flow = self.variational.flows[-2]
        mus = flow.u.detach().numpy()
        sigmas = np.exp(flow.diag.detach().numpy())
        if flow.non_diag is None:
            return np.hstack([mus, sigmas])
        else:
            non_diag = flow.non_diag.detach().numpy().flatten()
            return np.hstack([mus, sigmas, non_diag])

#________ INVERSION CLASS ________#

#__ Inversion __
class VI(): 

    '''

    Class that performs variational inference using normalising flows.

    Methods
    -------
    - :meth:`invert` : Perform variational inference to invert observed array covariance for model parameters.
    - :meth:`save` : Save the results of the inversion.
    - :meth:`load` : Load the results of the inversion.
    - :meth:`sample` : Sample from the posterior distribution of model parameters.
    - :meth:`unpack` : Unpack the model vector into its constituent parameters.

    Input
    -----
    flow_model : ``Callable``
        The normalising flows model to use for variational inference (e.g., ``LinearFlowsModel``).
    posterior : ``Callable``
        The posterior distribution to infer (e.g., ``Posterior``).
    learning_rate : ``float``
        The learning rate for the optimiser.
    outpath : ``str``
        The path to save the results.
    
    Attributes
    ----------
    flow_model : ``Callable``
        The normalising flows model to use for variational inference (e.g., ``LinearFlowsModel``).
    inversion : ``Callable``
        The variational inference object that performs the inversion.
    optimizer : ``torch.optim.Optimizer``
        The optimiser used for variational inference.
    learning_rate : ``float``
        The learning rate for the optimiser.
    loss_his : ``list``
        The history of the loss function during the inversion.
    outpath : ``str``
        The path to save the results.
    

    '''

    #__ Initialise class __
    def __init__(self, flow_model, posterior, learning_rate, outpath):
        self.flow_model = flow_model.def_flow_model()
        self.inversion = VariationalInversion(variationalDistribution=self.flow_model.variational, log_posterior=posterior.log_posterior)
        self.optimizer = torch.optim.Adam(self.flow_model.variational.parameters(), lr=learning_rate)
        self.learning_rate, self.loss_his, self.outpath = learning_rate, [], outpath
        self._unpack = posterior._unpack

        ## WARN IF ONLY USING SINGLE POINT SOURCE AS LIKELIHOOD ILL-DEFINED

    #__ Perform inversion __
    def invert(self, iterations, nsample, nout=0, save_intermediate_result=False, verbose=False):

        '''
        Function to perform variational inference to invert observed array covariance for model parameters.

        Input
        -----
        iterations : ``int``
            Number of iterations to run the inversion.
        nsample : ``int``
            Number of samples to estimate evidence.
        nout : ``int`` (default: ``0``)
            Number of samples to output for intermediate results.

        save_intermediate_result : ``bool`` (default: ``False``)
            If ``True``, save intermediate results during the inversion.

        verbose : ``bool`` (default: ``False``)
            If ``True``, print progress during the inversion.
        '''
        if (nout == 0):
            nout = 1
            save_intermediate_result = False
        self.loss_his.extend(self.inversion.update(
            optimizer = 'torch.optim.Adam',
            lr        = self.learning_rate,
            n_iter    = iterations,
            nsample   = nsample,
            n_out     = nout,
            verbose   = verbose,
            save_intermediate_result = save_intermediate_result,
            outpath   = self.outpath
        ))
        return self

    #__ Save flows __
    def save(self):
        '''
        Function to save the current flow model.
        '''
        iteration = len(self.loss_his)
        self.flow_model.variational.eval()
        param = self.flow_model.get_flow_param()
        param_fname = f"{self.outpath}/parameter_iteration-{iteration:05d}.npy"
        np.save(param_fname, param)
        loss_fname = f"{self.outpath}/loss_iteration-{iteration:05d}.txt"
        np.savetxt(loss_fname, self.loss_his)
        model_fname = f"{self.outpath}/model_iteration-{iteration:05d}.pt"
        torch.save(
            {
                'iteration':iteration,
                'model_state_dict':self.flow_model.variational.state_dict(),
                'loss':self.loss_his
            }, model_fname
        )
        return self

    #__ Load checkpoint __
    def load(self, iteration):
        '''
        Function to load a checkpoint from a previous iteration.

        Input
        -----
        iteration : ``int``
            The iteration number of the checkpoint to load.
        '''
        model_fname = f"{self.outpath}/model_iteration-{iteration:05d}.pt"
        checkpoint  = torch.load(model_fname, weights_only=False)
        self.flow_model.variational.load_state_dict(checkpoint['model_state_dict'])
        return self

    #__ Sample posterior __
    def sample(self, N_samples):
        '''
        Function to sample current normalising flows model.

        Input
        -----
        N_samples : ``int``
            Number of samples to draw from the posterior distribution.

        Output
        ------
        samples : ``numpy.ndarray``
            Samples drawn from the posterior distribution
            (N_samples, parameter)
        '''
        self.flow_model.variational.eval()
        samples = self.flow_model.variational.sample(N_samples)
        if not isinstance(samples, np.ndarray):
            samples = samples.detach().numpy()
        return samples

    #__ Unpack model vector __
    def unpack(self, m):
        '''
        Function to unpack the model vector.

        Input
        -----
        m : ``numpy.ndarray``
            The model vector to unpack.
            (model, parameter)

        Output
        ------
        The unpacked model.
        '''
        return self._unpack(m)