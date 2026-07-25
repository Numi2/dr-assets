# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# Copyright (c) 2026, Dr.Anmar Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the Dr.Anmar asset extension.

The ``orbit.surgical.assets`` namespace is retained for compatibility with
Dr.Anmar and ORBIT-Surgical-derived task code.
"""

import os
import toml

from setuptools import find_namespace_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Installation operation
setup(
    name="dranmar-assets",
    author="Dr.Anmar Project Developers",
    maintainer="Dr.Anmar Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    include_package_data=True,
    python_requires=">=3.10",
    packages=find_namespace_packages(
        include=["orbit.surgical.assets", "orbit.surgical.assets.*"]
    ),
    install_requires=["toml>=0.10"],
    classifiers=[
        "Natural Language :: English",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=False,
)
