#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import itertools
from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("dsrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> str:
    return ['gym>=0.26.0', 'numpy', 'pybullet>=3.0.6', "bullet_safety_gym@git+https://github.com/SvenGronauer/Bullet-Safety-Gym.git@master"]


def get_extras_require() -> str:
    req = {
        "mujoco": ["safety-gymnasium@git+https://github.com/PKU-Alignment/safety-gymnasium.git@main"],
        "metadrive": ["metadrive-simulator@git+https://github.com/HenryLHH/metadrive_clean.git@main"],
    }
    
    all_groups = set(req.keys())
    req["all"] = list(
        set(itertools.chain.from_iterable(map(lambda group: req[group], all_groups)))
    )
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
    python_requires=">=3.6",
    keywords="offline safe reinforcement learning dataset",
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)