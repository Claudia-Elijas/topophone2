.. topophone2 master file, created by
   sphinx-quickstart on Thu Sep 10 10:10:30 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. The toctree is hidden: it only feeds the top navigation bar, so the pages
   are not listed a second time on the home page.

.. toctree::
   :hidden:
   :maxdepth: 1

   setup
   documentation
   examples
   license


Topophone2
==========

.. container:: lead

   ``Topophone2`` is a Python package for probabilistic wavefield decomposition
   using sensor arrays. Given the array covariance matrix (cross-spectrum), it estimates
   parameters of the mixture of incoming wavefield components that produced the recording:
   e.g., plane waves, directionally distributed arrivals, propagating environental noise and 
   sensor self-noise.

   For each component, Topophone2 estimates the full posterior probability distribution
   (not just a single best-fit value) of the parameters that define its
   energy distribution, such as the direction of arrival, slowness and directional
   spread. Topophone2 can also use these posteriors to calculate information gain and
   compare competeing models through the Bayes factor.

.. container:: topophone-banner

   **Topophone1** The original topophone was an acoustic instrument
   patented in in 1880 by Alfred M. Mayer, with the purpose of finding the 
   direction of a sound source,
   such as a ship's fog horn. It used a pair of ear trumpets set apart from
   each other: the listener turned the device until the sound in both ears was
   balanced. 
   

The problem
-----------

Arrays of sensors (e.g., seismometers, hydrophones or microbarometers) often record several
wavefields simultaneously in the real world. Coherently propagating signals from distinct 
primary and secondarysources overlap and energy is spread over a range of directions.
Many array-processing methods assume (a) a far-field point source and (b) incoherent noise.
In practice, however, (a) the energy is often distributed over a range of directions of arrival, and
(b) ambient environmental wavefields propagate coherently across the array. Distributed wavefields
cause a loss of coherence between sensors, conventionally seen as a nuisance.
However, inter-sensor coherence loss carries information about the underlying wavefield structure, hence
the array covariance matrix can be used to infer properties of parameterised wavefield components, e.g., plane
waves, directionally distributed arrivals, propagating environmental noise and sensor self-noise.
This is a nonlinear, non-unique and often ill-conditioned inverse problem. Therefore, a nonlinear 
probabilistic approach is needed to properly quantify the uncertainty
in the inferred parameters: this is important for location and monitoring decisions.

``Topophone2`` uses the array covariance to decompompose the underlying wavefield into its
constituent parts within a probabilistic framework, estimating the full posterior probability
distributions of their parameters.


The approach
------------

``Topophone2`` treats decomposition as a Bayesian inverse problem, in three steps:

.. container:: home-grid steps

   .. container:: home-card

      .. rubric:: 1. Forward model

      :class:`~topophone2.modelling.ArrayCovarianceModel` predicts the array
      covariance for a superposition of parameterised, incoherently-distributed
      directional energy distributions given the array geometry and frequency band.
      The array covariance is a sum of the characteristic functions of each component, weighted
      by their total energies.

   .. container:: home-card

      .. rubric:: 2. Likelihood

      The predicted and observed array-covariance matrices are compared, using an
      exact complex Wishart likelihood. Uniform priors bound every parameter.

   .. container:: home-card

      .. rubric:: 3. Variational inference

      Normalising flows (built on
      `GeoPVI <https://github.com/XuebinZhaoZXB/GeoPVI>`_) learn an
      approximation to the posterior. Samples drawn from it can be summarised
      using marginal statistics such as means, credible intervals and information gain.
      The Bayes factor can be used to compare different models.

The whole workflow runs from a single call to
:func:`topophone2.topophone.decompose`.


Functionality
-------------

.. container:: home-grid

   .. container:: home-card

      .. rubric:: Applications

      ``Topophone2`` works with any array of sensors recording scalar wavefields
      distributed incoherently according to an analytical distribution with a known
      characteristic function. Typical uses include:

      - Direction-of-arrival estimation, with uncertainties
      - Estimating signal-to-noise ratios
      - Characterising the directional spread of energy from distributed primary or secondary sources
      - Quantifying the directionality of ambient enviromental wavefields
      - Comparing competing source models through Bayesian evidence 

   .. container:: home-card

      .. rubric:: Features

      - Analytical array-covariance forward models for degenerate (plane wave),
        normally and skew-normally distributed energy in slowness,
        (an)isotropic angular-harmonic fields (2D or 3D),
        and user-defined characteristic functions.
      - Exact gradients of the forward model, for efficient gradient-based
        optimisation.
      - Probabilistic inversion by normalising-flow variational inference, with
        optional mean-field approximation.
      - Optional joint inversion with noise-only data to constrain noise components.
      - Posterior analysis tools: marginal densities, credible intervals,
        information content and evidence (via `harmonic <https://github.com/astro-informatics/harmonic>`_).
      - Some utilities for reading and writing input
        data and configuration files and for signal processing (FFT, windowing, Slepian tapers,
        sample-covariance estimation, beamforming).


Quick start
-----------

After :doc:`installing <setup>` the package, decompose an array covariance matrix with a single call to
:func:`topophone2.topophone.decompose` and :func:`topophone2.utils.read.read_txt` to read the input data. 


.. code-block:: python

   from topophone2.utils.read import read_txt
   from topophone2.topophone import decompose

   # Observed array covariance, receiver positions and frequencies
   freq, x, Gamma_obs, N_obs = read_txt("data_input.txt")

   decompose(
       Gamma_obs    = Gamma_obs,
       N_obs        = N_obs,
       x            = x,
       freq         = freq,
       sigs         = ["point", "normal"],   # signal components to fit
       prior_bounds = (lower, upper),        # uniform prior bounds per parameter
       N_flow       = 10,
       learning_rate = 1e-2,
       N_iterations = 2000,
       N_montecarlo = 10,
       N_samples    = 5000,
       outpath      = "results",
   )

Where to go next:

- :doc:`examples`: complete worked examples, from preparing input data to
  real-data applications.
- :doc:`documentation`: the API reference for every module, class and
  function.
- `GitHub repository <https://github.com/Claudia-Elijas/topophone2>`_: the
  source code.



How to cite
-----------

If you use ``Topophone2`` in your research, please cite the following paper:

.. container:: cite-card cite-main

   .. rubric:: topophone2

   Smith, J.-A., & Elijas Parra, C. (in prep.). Probabilistic wavefield
   decomposition from array covariance.

   .. raw:: html

      <details><summary>BibTeX</summary>

   .. code-block:: bibtex

      @unpublished{smith_topophone2,
        author = {Smith, Jack-Andrew and Elijas Parra, Claudia},
        title  = {Probabilistic wavefield decomposition from array covariance},
        note   = {In preparation}
      }

   .. raw:: html

      </details>

``Topophone2`` is built on the following open-source packages, which should
also be cited if used in your research:

.. container:: cite-grid

   .. container:: cite-card

      .. rubric:: GeoPVI

      Zhao, X., & Curtis, A. (2026). GeoPVI: a Python package for
      geoscientific inversion using parametric variational inference.
      *Geophysical Journal International*, 245(3), ggag138.
      `doi:10.1093/gji/ggag138 <https://doi.org/10.1093/gji/ggag138>`_

      .. raw:: html

         <details><summary>BibTeX</summary>

      .. code-block:: bibtex

         @article{zhao_geopvi_2026,
           author  = {Zhao, Xuebin and Curtis, Andrew},
           title   = {{GeoPVI}: a {Python} package for geoscientific inversion
                      using parametric variational inference},
           journal = {Geophysical Journal International},
           volume  = {245},
           number  = {3},
           pages   = {ggag138},
           month   = jun,
           year    = {2026},
           doi     = {10.1093/gji/ggag138}
         }

      .. raw:: html

         </details>



Project information
-------------------

.. container:: project-banner

   .. container:: project-banner-item

      .. rubric:: Funding

      This project has received funding from the NERC Edinburgh Earth Ecology
      and Environment Doctoral Training Partnership grant NE/S007407/1.

   .. container:: project-banner-item

      .. rubric:: Contacts

      For any questions, suggestions or collaborations, please contact:

      - Lead developer: Jack-Andrew Smith (jack.andrew.smith53@gmail.com)
      - Code curator and site maintainer: Claudia Elijas Parra
        (claudia.elijas.parra@gmail.com)
