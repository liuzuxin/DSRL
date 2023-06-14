#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("dsrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> str:
    return [
        'gym>=0.26.0',
        'pybullet>=3.0.6',
        "bullet_safety_gym==1.4.0",
        "safety-gymnasium==0.4.0",
        'numpy',
    ]


def get_extras_require() -> str:
    req = {
        "metadrive":
        ["metadrive-simulator@git+https://github.com/HenryLHH/metadrive_clean.git@main"],
    }
    return req


setup(
    name="dsrl",
    version=get_version(),
    description="Datasets for Offline Safe Reinforcement Learning",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/liuzuxin/dsrl",
    author="DSRL contributors",
    author_email="zuxin1997@gmail.com",
    license="Apache",
    python_requires=">=3.8",
    classifiers=[
        # How mature is this project? Common values are 3 - Alpha 4 - Beta 5 -
        #   Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: Apache Software License",
        # Specify the Python versions you support here. In particular, ensure that you
        # indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="datasets for offline safe reinforcement learning",
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)