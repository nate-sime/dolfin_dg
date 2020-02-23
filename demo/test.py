import os
import subprocess
import sys

import pytest

demo_dirs = [
    "dolfin/burgers",
    # "dolfin/compressible_navier_stokes_naca0012",
    # "dolfin/heat_equation",
    # "dolfin/hyperlasticity_cantilever",
    # "dolfin/hyperlasticity_compresson",  # remove fenics-tools requirement
    "dolfin/nonlinear_advection",
    # "dolfin/paper_examples",
    # "dolfin/poisson",
    # "dolfin/sod_shock_tube",             # very slow
    "dolfin/stokes",
]

# Build list of demo programs
demos = []
for demo_dir in demo_dirs:
    demo_files = [f for f in os.listdir(demo_dir) if f.endswith(".py")]
    for demo_file in demo_files:
        demos.append((os.path.abspath(demo_dir), demo_file))


@pytest.mark.serial
@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):
    ret = subprocess.run([sys.executable, name],
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
    assert ret.returncode == 0
