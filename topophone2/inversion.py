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

#__ Statistics __
import harmonic as hm
from scipy.stats import gaussian_kde, hmean

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
        (frequency, receiver, receiver)
    
    N_obs : ``int``
        Number of independent samples used to estimate observed expectation

    array_covariance_model : ``Callable``
        Array covariance model (:class:`ArrayCovarianceModel`)

    prior : ``Callable``
        Prior distribution (:class:`Uniform`)

    Gamma_obs_noise : ``numpy.ndarray`` (optional)
        Observed noise-only array-covariance matrix
        (frequency, receiver, receiver)

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
        array_covariance_model : Callable,
        prior     : Callable,
        Gamma_obs_noise : Optional[np.ndarray] = None,
        N_obs_noise     : Optional[int]        = None
    ):
        

        #__ Throw error if noise-only not calculated but noise ACM specified here, or warning if specified but not provided __
        self._invert_noise_only = (Gamma_obs_noise is not None)
        if self._invert_noise_only and not array_covariance_model._calc_noise_only:
            raise ValueError('Noise data provided, must specify `calc_noise_only` in ArrayCovarianceModel.')

        #__ Store arguments as attributes __
        self.N_obs       = float(N_obs)            # Number of independent samples used to estimate observed expectation
        if self._invert_noise_only:
            self.N_obs_noise = float(N_obs_noise)  # Number of independent samples used to estimate observed noise-only expectation
        self.prior       = prior                   # Prior distribution

        #__ Store required attributes from model __
        self.forward  = array_covariance_model.forward   # Function to predict array covariance from model vector
        self._iu0_rec = array_covariance_model._iu0_rec  # Upper-triangle index set for unique receiver pairs (including diagonal)
        self._iu1_rec = array_covariance_model._iu1_rec  # ^
        self._N_rec   = array_covariance_model.N_rec     # Number of receivers
        self._N_freq  = array_covariance_model.N_freq    # Number of frequency bins
        self._calc_noise_only = array_covariance_model._calc_noise_only  # Include an additional noise-only calculation
        self._unpack  = array_covariance_model._unpack   # Unpack model vector
        self._expand  = array_covariance_model.expand    # Unflatten array-covariance

        #__ Lower-triangle index set for unique receiver pairs (excluding diagonal) __
        self._il0_rec, self._il1_rec = np.tril_indices(self._N_rec, k=-1)

        #__ Check Hermitian symmetry of observation __
        def check_hermitian(Gamma_obs, var_name):
            if not np.allclose(Gamma_obs,
                            np.conj(np.swapaxes(Gamma_obs, -1, -2))):
                raise ValueError(f"{var_name} is not Hermitian.")
        check_hermitian(Gamma_obs, 'Gamma_obs')
        if self._invert_noise_only: check_hermitian(Gamma_obs_noise, 'Gamma_obs_noise')

        #__ Scaled Gamma_obs (A in Goodman [1963]) __
        self._A = self.N_obs * Gamma_obs  # (frequency, receiver, receiver)
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
    def _lnp_dm_and_grad(self, Gamma, dGamma, A, N_obs, grad, var_name):

        #__ Cholesky decomposition of predicted array-covariance matrix __
        L = self._safe_cholesky(Gamma, var_name)  # (model, freq, s)

        #__ Log-determinant of Gamma __
        diag   = np.real(np.diagonal(L, axis1=-2, axis2=-1))  # (model, freq, receiver)
        logdet = 2.0 * np.sum(np.log(diag), axis=-1)          # (model, freq)

        #__ Inverse of Gamma __
        Gamma_inv = np.linalg.inv(Gamma)  # (model, freq, receiver, receiver)

        #__ tr(Gamma^-1 A) __
        tr = np.real(np.sum(Gamma_inv * np.conj(A)[None,...], axis=(-2, -1)))  # (model, freq)

        #__ Sum over independent frequency bins __
        lnp_dm = np.sum(-N_obs * logdet - tr, axis=-1)   # (model)
        if not grad: return lnp_dm, None

        #__ Calculate gradient (model, parameter) __
        B            = A[None, ...] - N_obs * Gamma
        G            = Gamma_inv @ B @ Gamma_inv
        dlnp_dm = np.real(np.einsum('mfij,mfjik->mk', G, dGamma, optimize=True))

        #__ Return output __
        return lnp_dm, dlnp_dm

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
        lnp_dm : ``numpy.ndarray``
            Log-likelihood of observed array-covariance matrix given model vector
            (model,)

        dlnp_dm : ``numpy.ndarray``
            Gradient of log-likelihood wrt. model parameters
            (model, parameter)

        """

        #__ Extract model shape (model, parameter) __
        if not isinstance(m, np.ndarray):
            m = m.detach().numpy()
        m = np.atleast_2d(m)
        N_model, N_param = m.shape

        #__ Synthesise array-covariance matrix __
        if self._calc_noise_only: cov, dcov, cov_noise, dcov_noise = self.forward(m, grad=grad)  # (model, frequency, receiver pair, [parameter])
        else: cov, dcov = self.forward(m, grad=grad)  # (model, frequency, receiver pair, [parameter])
        Gamma, dGamma = self._expand(cov, dcov if grad else None)
        if self._invert_noise_only: Gamma_noise, dGamma_noise = self._expand(cov_noise, dcov_noise if grad else None)

        #__ Calculate likelihood __
        lnp_dm, dlnp_dm = self._lnp_dm_and_grad(Gamma, dGamma, self._A, self.N_obs, grad, 'Gamma')
        if self._invert_noise_only:
            lnp_dm_noise, dlnp_dm_noise = self._lnp_dm_and_grad(Gamma_noise, dGamma_noise, self._A_noise, self.N_obs_noise, grad, 'Gamma_noise')
            lnp_dm += lnp_dm_noise
            if (dlnp_dm is not None): dlnp_dm += dlnp_dm_noise

        #__ Return output __
        return lnp_dm, dlnp_dm

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
        lnp_md : ``torch.Tensor``
            Log-posterior for model vector given observed array-covariance matrix and prior distribution
            (model,)
        
        
        '''

        #__ Calculate log-likelihood __
        lnp_dm = ForwardModel.apply(m, self.log_likelihood)

        #__ Calculate log-prior __
        lnp_m = self.prior.log_prior(m)

        #__ Calculate log-posterior __
        lnp_md = lnp_m + lnp_dm

        #__ Return output __
        return lnp_md

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
class Inference(): 

    '''

    Class that performs variational inference using normalising flows.

    Methods
    -------
    - :meth:`invert` : Perform variational inference to invert observed array covariance for model parameters.
    - :meth:`save` : Save the results of the inversion.
    - :meth:`load` : Load the results of the inversion.
    - :meth:`sample` : Sample from the posterior distribution of model parameters.
    - :meth:`unpack` : Unpack the model vector into its constituent parameters.
    - :meth:`get_lnELBO` : Function to return logarithmic evidence lower bound.

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
    optimiser : ``torch.optim.Optimizer``
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
        self.optimiser = torch.optim.Adam(self.flow_model.variational.parameters(), lr=learning_rate)
        self.learning_rate, self.loss_his, self.outpath = learning_rate, [], outpath
        self._unpack = posterior._unpack

        ## WARN IF ONLY USING SINGLE POINT SOURCE AS LIKELIHOOD ILL-DEFINED

    #__ Perform inversion __
    def invert(self, N_iterations, N_montecarlo, N_out=0, save_intermediate_result=False, verbose=False):

        '''
        Function to perform variational inference to invert observed array covariance for model parameters.

        Input
        -----
        N_iterations : ``int``
            Number of iterations to run the inversion.
        N_montecarlo : ``int``
            Number of samples to estimate ELBO and gradient estimate.
        N_out : ``int`` (default: ``0``)
            Number of samples to output for intermediate results.

        save_intermediate_result : ``bool`` (default: ``False``)
            If ``True``, save intermediate results during the inversion.

        verbose : ``bool`` (default: ``False``)
            If ``True``, print progress during the inversion.
        '''
        if (N_out == 0):
            N_out = 1
            save_intermediate_result = False
        self.loss_his.extend(self.inversion.update(
            optimizer = 'torch.optim.Adam',
            lr        = self.learning_rate,
            n_iter    = N_iterations,
            nsample   = N_montecarlo,
            n_out     = N_out,
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

    #__ Return ELBO __
    def get_lnELBO(self):
        '''
        Function to return logarithmic evidence lower bound.

        Output
        ------
        lnELBO : ``numpy.ndarray``
            Log-ELBO.
            (iteration,)
        '''

        #__ Extract log-ELBO __
        lnELBO = -np.array(self.loss_his)

        #__ Return output __
        return lnELBO

#________ POST-PROCESSING ________#

#__ Class for marginal statistics __
class MarginalStats():
    ''''''

    #__ Initialise class __
    def __init__(self, m, lnp_dm=None, posterior=None, prior_bounds=None):
        self.m   = m
        if (lnp_dm is not None):
            self.lnp_dm = lnp_dm
        elif (posterior is not None):
            self.lnp_dm, _ = posterior.log_likelihood(m, grad=False)
        else:
            self.lnp_dm = None
        if (prior_bounds is not None):
            self.Vi = prior_bounds[1] - prior_bounds[0]
        else:
            self.Vi = None

    #__ Calculate evidence __
    def calc_evidence(self, nchains=10, temperature=0.8):
        '''
        Function to calculate evidence using the learnt harmonic mean estimator [McEwan et al., 2021, ArXiv].
        
        Input
        -----
        nchains : ``int`` (default ``10``)
            Number of data sets to split model chain into.

        temperature : ``float`` (default ``0.8``)
            Temperature of learnt model.

        Output
        ------
        lnp_d : ``float``
            Log-evidence.

        lnp_d_std : ``float``
            Standard deviation of estimate.
        '''

        #__ Reshape model vector __
        n = self.m.shape[-1]                # Number of parameters
        m = self.m.reshape(nchains, -1, n)  # (chain, sample, parameter)

        #__ Calculate log-posterior __
        lnV = np.sum(np.log(self.Vi))                # Log-hypervolume
        lnp_md = (self.lnp_dm - lnV).reshape(nchains, -1)  # (chain, sample)

        #__ Instantiate Chains class __
        chains = hm.Chains(n)
        chains.add_chains_3d(m, lnp_md)

        #__ Split into training and inference chains __
        chains_train, chains_infer = hm.utils.split_data(chains, training_proportion=0.5)

        #__ Train ML model __
        model = hm.model.RQSplineModel(n, standardize=True, temperature=temperature)
        model.fit(chains_train.samples)

        #__ Instantiate Evidence class __
        ev = hm.Evidence(chains_infer.nchains, model)
        ev.add_chains(chains_infer)
        
        #__ Compute evidence __
        lnp_d, lnp_d_std = ev.compute_ln_evidence()

        #__ Return output __
        return lnp_d, lnp_d_std

    #__ Fit Gaussian KDE __
    def _gaussian_kde(self, idx, thin=None):
        '''
        Fit a Gaussian KDE to a subset of the model parameters.

        Input
        -----
        idx : ``list``
            List of indices in subset.

        thin : ``int`` (optional)
            Thin model samples for KDE estimation.

        Output
        ------
        ???
        '''

        #__ Thin model selection __
        m_s   = self.m[:,idx].T
        m_fit = m_s[:,::thin]

        #__ Fit KDE __
        kde = gaussian_kde(m_fit)

        #__ Return output __
        return kde, m_s

    #__ Calculate information gain __
    def calc_information(self, idx=None, thin=2):
        '''
        Calculate the information gain relative to a bounded uniform prior.

        Input
        -----
        idx : ``list`` (optional)
            List of indices in subset.

        thin : ``int`` (default ``2``)
            Thin model samples for KDE estimation.

        Output
        ------
        lnD : ``float``
            Information gain [nats]
        '''

        #__ Full model parameter set __
        if (idx is None):

            #__ Calculate evidence __
            lnp_d, _ = self.calc_evidence()

            #__ Calculate information gain __
            lnD = np.mean(self.lnp_dm) - lnp_d

        #__ Partial model parameter set __
        else:

            #__ Calculate sub-hypervolume __
            lnVs = np.sum(np.log(self.Vi[idx]))

            #__ Fit KDE __
            kde, m_s = self._gaussian_kde(idx, thin)

            #__ Calculate information gain __
            lnD = lnVs + np.mean(kde.logpdf(m_s[:,1::thin]))

        #__ Return output __
        return lnD

    #__ Calculate mean __
    def marginal_mean(self, idx, harmonic=False):
        '''
        Calculate the marginal means.

        Input
        -----
        idx : ``list``
            List of indices in subset.

        harmonic : ``bool`` (default ``False``)
            Perform harmonic averaging, rather than arithmetic.

        Output
        ------
        mean : ``np.ndarray``
            Marginal mean
            (parameter,)
        '''

        #__ Subset models __
        m_s = self.m[:,idx].T

        #__ Calculate mean __
        if harmonic: mean = hmean(m_s,   axis=1)
        else:        mean = np.mean(m_s, axis=1)

        #__ Return output __
        return mean

    #__ Calculate credible intervals __
    def credible_interval(self, inner_pc=95., idx=None):
        '''
        Calculate the credible interval.

        Input
        -----
        inner_pc : ``float`` (default ``95.0``)
            Interval percentage [%].

        idx : ``list`` (optional)
            List of indices in subset.

        Output
        ------
        ci_lo : ``np.ndarray``
            Lower bound of credible interval.
            (parameter,)

        ci_hi : ``np.ndarray``
            Upper bound of credible interval.
            (parameter,)
        '''

        #__ Subset models __
        m_s = self.m[:,idx].T if idx is not None else self.m.T

        #__ Calculate percentile bounds __
        lo, hi = 0.5*(100-inner_pc), 0.5*(100+inner_pc)

        #__ Compute percentiles __
        ci = np.percentile(m_s, [lo,hi], axis=1)
        ci_lo, ci_hi = ci[0], ci[1]

        #__ Return output __
        return ci_lo, ci_hi

    #__ Estimate probability density function __
    def probability_density(self, i, npts=500, bounds=None, thin=None, cumulative=None):
        '''
        Calculate marginal [(reverse) cumulative] probability density estimate.

        Input
        -----
        i : ``int``
            Model parameter index.

        npts : ``int`` (default ``500``)
            Number of points in KDE vector.

        bounds : ``Tuple[float,float]`` (optional)
            Bounds of KDE vector.

        thin : ``int`` (optional)
            Thin model samples for KDE estimation.

        cumulative : ``str`` (optional)
            If specified, calculate cumulative distribution in ``'forward'`` or ``'reverse'`` direction.
        
        Output
        ------
        density : ``np.ndarray``
            Density function.

        vec : ``np.ndarray``
            Density function vector.
        '''

        #__ Fit KDE __
        kde, m_s = self._gaussian_kde(np.atleast_1d(i), thin)

        #__ Calculate vector __
        lo, hi = bounds if bounds is not None else (m_s.min(), m_s.max())
        vec    = np.linspace(lo, hi, npts)

        #__ Calculate density function __
        if (cumulative is None):
            density = kde(vec)
        elif (cumulative == 'forward'):
            density = np.array([kde.integrate_box_1d(-np.inf, vec_i) for vec_i in vec])
        elif (cumulative == 'reverse'):
            density = np.array([kde.integrate_box_1d(vec_i, np.inf)  for vec_i in vec])

        #__ Return output __
        return density, vec