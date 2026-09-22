# setup file
from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='topophone2',
    version='1.0.0',
    description='.',
    long_description=open('README.md').read(),
    author='.',
    author_email='.',
    url='https://github.com/Claudia-Elijas/topophone2',
    packages=['topophone2'],
    install_requires=requirements,
    license="GPL-3.0",
    classifiers=[
        # How mature is this project? 
        # Development Status :: 3 - Alpha",
        # Development Status :: 4 - Beta",
        # Development Status :: 5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # supported Python versions
        "Programming Language :: Python :: 3",
        # license
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        # operating systems 
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        # indicate the type of application
        "Topic :: Scientific/Engineering :: Physics"
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.11',

)