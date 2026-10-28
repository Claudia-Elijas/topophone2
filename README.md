# Topophone2

**Topophone2** is a Python package for probabilistic wavefield decomposition
using sensor arrays. Given the array covariance matrix (cross-spectrum), it estimates
parameters of the mixture of incoming wavefield components that produced the recording:
e.g., plane waves, directionally distributed arrivals, propagating environental noise and sensor self-noise.

For each component, Topophone2 estimates the full posterior probability distribution
(not just a single best-fit value) of the parameters that define its
energy distribution, such as the direction of arrival, slowness and directional
spread. Topophone2 can also use these posteriors to calculate information gain and
compare competeing models through the Bayes factor.

Visit the [package website](https://claudia-elijas.github.io/topophone2/) for more information!

>[!NOTE]
>We would like to remind the prospective users that this package is under active development to ensure the implemented methodology is accessible and easy to employ for a wider community. We welcome feedback, feature suggestions, and collaboration requests. Please feel free to contact the developers by [email](mailto:jack.andrew.smith53@gmail.com) or use [Issues](https://github.com/Claudia-Elijas/topophone2/issues) or [Pull Requests](https://github.com/Claudia-Elijas/topophone2/pulls) to get in touch and contribute to the project.

## Getting started 

To install **Topophone2**, follow the steps below, also described in the [installation guide](https://claudia-elijas.github.io/topophone2/setup.html).

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

You can find several examples in the [examples gallery](https://claudia-elijas.github.io/topophone2/examples.html).

The API documentation can be found in the [documentation](https://claudia-elijas.github.io/topophone2/documentation.html).


## How to cite

If you use **Topophone2** in your research, please cite the following paper:

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

**Topophone2** is built on the following open-source packages, which should
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
