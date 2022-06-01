#!/usr/bin/env python

import pathlib
from setuptools import find_packages, setup



# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="mat2qubit", # Replace with your own username
    version="0.0.1",
    author="Nicolas P. D. Sawaya",
    author_email="nicolas.sawaya@intel.com",
    description="Package for encoding matrices into qubits.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/mat2qubit",
    license='Apache 2',
    #package_dir={'': 'src'},
    packages=['mat2qubit'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "sympy>=1.3",
        "scipy>=1.1",
        "openfermion>=1.0"
    ],
    extras_require={"dev": [
        "black~=22.3.0",
        "flake8~=4.0.1",
        "isort~=5.10.1",
    ]}
)







