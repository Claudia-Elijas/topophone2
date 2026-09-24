#__ Import: Numerical __
import numpy as np

#__ Import: Topophone2 __
from topophone2.modelling import ArrayCovarianceModel
from topophone2.inversion import Uniform, Posterior, LinearFlowsModel, Inference
from topophone2.utils.read import read_txt

#__ Import: Plotting __
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmcrameri.cm as cmc

#__ Set up: Plotting __
theme = ['#DC4424', '#74A466', '#24448C', '#F8BC5C']
small, large = 8, 10
plt.rcParams["font.sans-serif"] = ['Arial']

#__ Set up: Random seed __
np.random.seed(2026)

#__ Set up: File paths __
data_dir  = '../data/TRIAD_0DB'
model_dir = '../models/TRIAD_0DB'

#__ Read: Signal + noise __
out = read_txt(f"{data_dir}/TRIAD_FULL_0DB_topophone2.txt")
freq_Hz   = out[0]  # Frequency vector         [Hz]        (frequency)
x         = out[1]  # Receiver position        [m]         (receiver, direction)
Gamma_obs = out[2]  # Array-covariance matrix  [uPa2.Hz-1] (frequency, receiver, receiver)
N_obs     = out[3]  # Independent observations
freq      = 2 * np.pi * freq_Hz  # Angular frequency vector [rad.s-1] (frequency)

#__ Read: Noise only __
out = read_txt(f"{data_dir}/TRIAD_NOISE_0DB_topophone2.txt")
Gamma_obs_noise = out[2]  # Array-covariance matrix  [uPa2.Hz-1] (frequency, receiver, receiver)
N_obs_noise     = out[3]  # Independent observations

#__ Noise apparent slowness [s.m-1] __
synthetics = np.load('../data/SYNTHETICS.npz')
s0_J       = synthetics['s0'][0]

#__ Signals present __
sigs = [
    'normal',                   # Gaussian density-function field
    dict(
        sig       = 'noise2D',  # Cylindrically-harmonic noise field
        s         = s0_J,       # Local apparent slowness [s.m-1]
        max_order = 1           # Maximum cylindrical-harmonic order
    )
]

#__ Forward modeller __
array_covariance_model = ArrayCovarianceModel(
    x = x,                  # Receiver locations       [m]       (receiver, direction)
    freq = freq,            # Angular frequency vector [rad.s-1] (frequency)
    sigs = sigs,            # Signals present
    calc_noise_only = True  # Calculate an additional noise-only array-covariance matrix
)

#__ Prior bounds __
lower = np.array([                                 # Lower bound
    50.,  -5.0e-4, 8.0e-5, 1.0e-6, 1.0e-6, -0.99,  #     'normal' parameters
    50., -0.8, -0.8                                #     'noise2D' parameters
])
upper = np.array([                                 # Upper bound
    100., -4.2e-4, 1.6e-4, 5.0e-5, 5.0e-5,  0.99,  #     'normal' parameters
    100., 0.8,  0.8                                #     'noise2D parameters
])

#__ Uniform prior pdf __
prior = Uniform(lower=lower, upper=upper)

#__ Posterior pdf __
posterior = Posterior(
    Gamma_obs       = Gamma_obs,        # Observed array-covariance matrix
    N_obs           = N_obs,            # Number of independent observations
    Gamma_obs_noise = Gamma_obs_noise,  # Observed (noise-only) array-covariance matrix
    N_obs_noise     = N_obs_noise,      # Number of independent (noise-only) observations
    prior           = prior,            # Uniform prior pdf
    array_covariance_model = array_covariance_model  # Forward modeller
)

#__ Number of flows __
N_flow = 8

#__ Normalising flow model __
flow_model = LinearFlowsModel(
    N_param      = len(lower),      # Number of model parameters
    prior_bounds = (lower, upper),  # Uniform prior bounds
    N_flow       = N_flow           # Number of flows
)

#__ Inversion parameters __
learning_rate = 5e-4                       # Learning rate for the optimiser
N_iterations  = 30000                      # Number of iterations for optimiser
N_montecarlo  = 100                        # Number of samples to estimate ELBO

#__ Set up inverse problem __
inverse_problem = Inference(
    flow_model    = flow_model,     # Normalising flows model
    posterior     = posterior,      # Posterior pdf
    learning_rate = learning_rate,  # Learning rate
    outpath       = f"{model_dir}"  # Path to save results
)

#__ Run inversion __
inverse_problem.invert(
    N_iterations = N_iterations,  # Number of iterations for optimiser
    N_montecarlo = N_montecarlo,  # Number of samples to estimate ELBO
    verbose      = True,          # Print progress
    N_out        = 6              # Number of print statements
)

#__ Save flows model for later use __
inverse_problem.save()

#__ Number of samples to draw __
N_samples = 20000

#__ Draw model samples from posterior pdf __
m = inverse_problem.sample(N_samples=N_samples)        # (sample, parameter)

#__ Calculate log-likelihood of each sample __
lnp_dm, _ = posterior.log_likelihood(m=m, grad=False)  # (sample,)

#__ Save for later __
np.savez_compressed(
    f"{model_dir}/inversion_results.npz",
    m            = m,
    lnp_dm       = lnp_dm,
    prior_bounds = (lower, upper)
)