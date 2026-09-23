Installation
============

Welcome to the installation of topophone2!

There are several ways to install topophone2, depending on your preferences and needs. 


1. **Using GitHub repository**: You can clone the topophone2 repository from GitHub and install it locally. This method allows you to access the latest version of the code and contribute to its development if you wish.
    
    First, clone the repository and navigate to the project directory:

    .. code-block:: bash
        
        git clone https://github.com/Claudia-Elijas/topophone2.git
        cd topophone2
    
    Then, you can install the required dependencies from the environment yml file:

    .. code-block:: bash

        conda env create -f topophone2_env.yml
    
    And finally, install the package:

    .. code-block:: bash

        pip install -e .
    
    Note that the `-e` flag allows you to install the package in editable mode, which means that any changes you make to the code will be reflected in the installed package without needing to reinstall it. This is particularly useful for development and testing purposes.

For more details, please refer to the `topophone2 GitHub repository <https://github.com/Claudia-Elijas/topophone2>`_   


2. **Using pip**: You can install topophone2 using pip, which **will be available soon on PyPI**. This method is straightforward and allows you to easily manage the package and its dependencies.
     
    First make sure you have pip installed and updated:
    
    .. code-block:: bash

        pip install --upgrade pip
    
    Then, you can install topophone2 using pip:

    .. code-block:: bash 

        pip install topophone2






