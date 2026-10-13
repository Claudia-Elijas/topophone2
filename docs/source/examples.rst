Examples
========

Here are some examples of how to use the topophone2 package:



1. Hydroacoustics example
-------------------------

In this example, we demonstrate how to carry out probabilistic wavefield decomposition on a synthetic abyssal T-phase. 
We use the posteriors to estimate information gain for an extra sensor and increased noise level. 
We use the posterios to discriminate between a point-source and a distributed-source model.  


.. nbgallery::
    :glob: 

    examples/example_1/example_1.1.ipynb
    examples/example_1/example_1.2.ipynb
    examples/example_1/example_1.3.ipynb


2. Infrasound example
---------------------

In this example, we demonstrate how to carry out probabilistic wavefield decomposition on a thermospheric infrasound return of an explosion event. 
We show how to create a sample covariance matrix from a time series. 
We inverted for a skew-normal, environmental noise and sensor self-noise component. 
We test the hypothesis that the signal is skewed back towards the source with respect to its bulk deviation. 

.. nbgallery::
    :glob: 

    examples/example_2/example_2.1.ipynb
    examples/example_2/example_2.2.ipynb


 
3. Synthetic testing
--------------------

We show how to generate data for a synthetic wavefield.
It also demonstrates how to populate and read the configuration file with the existing code functionality. 

.. nbgallery::
    :glob: 

    examples/example_3/example_3.1.ipynb
    examples/example_3/example_3.2.ipynb





