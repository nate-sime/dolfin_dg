import os
import subprocess
import sys
import flake8.api.legacy as flake8

import pytest

fast_demo_dirs = [
    "dolfin/burgers",
    "dolfin/nonlinear_advection",
    "dolfin/stokes",
    "dolfinx/advection",
    "dolfinx/poisson",
]

slow_demo_dirs = [
    "dolfin/compressible_navier_stokes_naca0012",
    "dolfin/hyperelasticity",
    "dolfin/time_dependent",
    "dolfinx/stokes"
]

paper_demo_dirs = [
    "dolfin/paper_examples"
]

supported_demo_dirs = fast_demo_dirs + slow_demo_dirs

all_demo_dirs = fast_demo_dirs + slow_demo_dirs + paper_demo_dirs


# Build list of demo programs
def build_demo_list(demo_dirs):
    demos = []
    for demo_dir in demo_dirs:
        demo_files = [f for f in os.listdir(demo_dir) if f.endswith(".py")]
        for demo_file in demo_files:
            demos.append((os.path.abspath(demo_dir), demo_file))
    return demos


def dispatch_demo(path, name):
    ret = subprocess.run([sys.executable, name],
                         cwd=str(path),
                         env={**os.environ, 'MPLBACKEND': 'agg'},
                         check=True)
    return ret


@pytest.mark.serial
@pytest.mark.parametrize("path,name", build_demo_list(fast_demo_dirs))
def test_fast_demos(path, name):
    ret = dispatch_demo(path, name)
    assert ret.returncode == 0


@pytest.mark.serial
@pytest.mark.parametrize("path,name", build_demo_list(supported_demo_dirs))
def test_supported_demos(path, name):
    ret = dispatch_demo(path, name)
    assert ret.returncode == 0


@pytest.mark.serial
@pytest.mark.parametrize("path,name", build_demo_list(paper_demo_dirs))
def test_paper_example_demos(path, name):
    ret = dispatch_demo(path, name)
    assert ret.returncode == 0


@pytest.mark.serial
@pytest.mark.parametrize("path,name", build_demo_list(supported_demo_dirs))
def test_flake8_demos(path, name):
    report = flake8.get_style_guide().check_files(
        paths=[os.path.join(path, name)])
    assert len(report.get_statistics("F")) == 0
    assert len(report.get_statistics("E")) == 0
