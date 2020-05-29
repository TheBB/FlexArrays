#!/usr/bin/env python

from pathlib import Path
from setuptools import setup, find_packages
from distutils.extension import Extension

setup(
    name='FlexArrays',
    version='0.1.0',
    description='Flexible and symbolically indexed block arrays',
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'multipledispatch',
    ],
)
