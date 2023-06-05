#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("dsrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> str:
    return ['gym>=0.26.0', 'numpy', 'pybullet>=3.0.6']


setup(
    name="dsrl",
    version=get_version(),
    description="Offline Safe Reinforcement Learning.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    python_requires=">=3.6",
    keywords="offlien safe reinforcement learning dataset",
    packages=find_packages(exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]),
    install_requires=get_install_requires(),
)