Installation
============

In this section, we will guide you through the installation process of the **topophone2** package.

.. admonition:: Requirements
   :class: banner

   The **topophone2** package requires a Linux system (on Windows, use
   `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_), with
   ``git`` and ``conda`` (or ``mamba``) installed.

The installation will be done through the terminal, following the steps below.

1. Clone the repository
-----------------------

First, create a copy of the **topophone2** code on your local machine using ``git clone``:

.. code-block:: bash

   git clone https://github.com/Claudia-Elijas/topophone2.git

.. note::

   This downloads the latest code from the GitHub repository. To install a
   specific release instead, pass its tag after ``--checkout``. For example, for
   version 1.0.0:

   .. code-block:: bash

      git clone https://github.com/Claudia-Elijas/topophone2.git 
      git fetch --quiet origin 
      git checkout --quiet v1.0.0

2. Run the setup script
-----------------------

Then run the ``setup.sh`` script from inside the ``topophone2`` folder:

.. code-block:: bash

   cd topophone2
   bash setup.sh

The script:

1. creates (or updates, if it already exists) a conda environment called
   ``topophone2`` from ``conda_env.yml``;
2. downloads `GeoPVI <https://github.com/XuebinZhaoZXB/GeoPVI>`_ into
   ``external/GeoPVI``, compiles its solver and installs it;
3. installs **topophone2** in editable mode, so changes to the source code take
   effect without reinstalling.

This can take several minutes the first time. Running it again later is safe,
and brings the environment up to date with ``conda_env.yml``.

3. Activate the environment for use
-----------------------------------

The script does not leave the environment active, so activate it before using
**topophone2**:

.. code-block:: bash

   conda activate topophone2

To check that everything is installed, run:

.. code-block:: bash

   python -c "import topophone2.inversion; print('topophone2 is ready')"

For more details, please refer to the `topophone2 GitHub repository <https://github.com/Claudia-Elijas/topophone2>`_.
