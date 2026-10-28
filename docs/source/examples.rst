Examples
========

Here are some examples of how to use the Topophone2 package:



1. Hydroacoustics example
-------------------------

In this example, we demonstrate how to carry out probabilistic wavefield decomposition on a synthetic abyssal T-phase. 
We use the posteriors to estimate information gain for an extra sensor and increased noise level. 
We use the posteriors to discriminate between a point-source and a distributed-source model.  


.. nbgallery::
    :glob: 

    examples/example_1/example_1.1.ipynb
    examples/example_1/example_1.2.ipynb
    examples/example_1/example_1.3.ipynb


2. Infrasound example
---------------------

In this example, we demonstrate how to carry out probabilistic wavefield decomposition on a thermospheric infrasound return of an explosion event. 
We show how to create a sample-covariance matrix from a time series. 
We invert for a skew-normally distributed signal, angular-harmonic noise, and sensor self-noise component. 
We test the hypothesis that the signal is skewed back towards the source with respect to the deviation of the maximum energy direction. 

.. nbgallery::
    :glob: 

    examples/example_2/example_2.1.ipynb
    examples/example_2/example_2.2.ipynb

The waveform data used in this example is available from the virtual Data Exploitation Centre of the International Data Centre at https://www.ctbto.org/specials/vdec/ . 
 
3. Synthetic testing
--------------------

In this example, we show how to generate data for a synthetic wavefield.
We also demonstrate how to populate and read the configuration file with the existing code functionality. 

.. nbgallery::
    :glob: 

    examples/example_3/example_3.1.ipynb
    examples/example_3/example_3.2.ipynb





