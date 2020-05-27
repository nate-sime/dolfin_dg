#!/usr/bin/env python

import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

version = '2019.02.10'

setup(name="dolfin_dg",
      description="Utility module for automatic generation of DG FE formulations.",
      version=version,
      author="Nate Sime",
      author_email="njcs4@cam.ac.uk",
      url="",
      license="",
      package_dir={"dolfin_dg": "dolfin_dg"},
      packages=["dolfin_dg"])
