# Topophone2

**Topophone2** is a Python package for probabilistic wavefield decomposition
using receiver arrays. Given the array covariance matrix (cross-spectrum)
derived from recordings made by a set of receivers, it estimates
the mixture of incoming wavefields that produced the recorded signal: inluding 
plane waves, directionally distributed arrivals, propagating environental noise and sensor self-noise.

For every component, it estimates the full posterior probability distribution
(not just a single best-fit value) of the parameters that define the signal's
energy distribution, such as the direction of arrival, slowness and directional
spread. It also quantifies how well the data constrain each component, and
compare competing models through Bayesian evidence.

>[!NOTE]
>We would like to remind the prospective users that this package is under active development to ensure the implemented methodology is accessible and easy to employ for a wider community. We welcome feedback, feature suggestions, and collaboration requests. Please feel free to contact the developers by email or use the GitHub Discussions, Issues, or Pull Requests to get in touch and contribute to the project. 

## Getting started 

To install **topophone2** follow the steps below: 

#### 1. Clone the repository

Create a copy of the source code on your local machine using ``git``.

```bash
git clone https://github.com/Claudia-Elijas/topophone2.git
```

#### 2. Run the setup.sh script

This script creates the topophone2 conda environment, installs dependencies, and installs the package locally in your machine. 

```bash
cd topophone2
bash setup.sh
```

#### 3. Activate the topophone2 environment for use

The script does not leave the environment active, so activate it before using
**topophone2**:

```bash 
conda activate topophone2
```

To check that everything is installed, run:

```bash
python -c "import topophone2.inversion; print('topophone2 is ready')"
```


![Completed](https://img.shields.io/badge/completed-✓-brightgreen)


## Examples

You can find several examples at https://claudia-elijas.github.io/topophone2/examples/ 


## Documentation

The API documentation can be found at https://claudia-elijas.github.io/topophone2/ 


## How to cite

If you use **topophone2** in your research, please cite the following paper:

> Smith, J.-A., & Elijas Parra, C. (in prep.). Probabilistic wavefield
> decomposition from array covariance.

<details>
<summary>BibTeX</summary>

```bibtex
@unpublished{smith_topophone2,
  author = {Smith, Jack-Andrew and Elijas Parra, Claudia},
  title  = {Probabilistic wavefield decomposition from array covariance},
  note   = {In preparation}
}
```

</details>

**topophone2** is built on the following open-source packages, which should
also be cited if used in your research:

**GeoPVI**

> Zhao, X., & Curtis, A. (2026). GeoPVI: a Python package for geoscientific
> inversion using parametric variational inference. *Geophysical Journal
> International*, 245(3), ggag138.
> [doi:10.1093/gji/ggag138](https://doi.org/10.1093/gji/ggag138)

<details>
<summary>BibTeX</summary>

```bibtex
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
```

</details>

**harmonic**

> McEwen, J. D., Wallis, C. G. R., Price, M. A., & Docherty, M. M. (2021).
> Machine learning assisted Bayesian model comparison: learnt harmonic mean
> estimator. *arXiv preprint* arXiv:2111.12720.
> [doi:10.48550/arXiv.2111.12720](https://doi.org/10.48550/arXiv.2111.12720)

<details>
<summary>BibTeX</summary>

```bibtex
@article{mcewen_harmonic_2021,
  author  = {McEwen, J. D. and Wallis, C. G. R. and
             Price, M. A. and Docherty, M. M.},
  title   = {Machine learning assisted {Bayesian} model comparison:
             learnt harmonic mean estimator},
  journal = {arXiv preprint arXiv:2111.12720},
  year    = {2021},
  doi     = {10.48550/arXiv.2111.12720}
}
```

</details>
