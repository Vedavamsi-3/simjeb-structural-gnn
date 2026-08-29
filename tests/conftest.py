"""Shared test fixtures.

Every test runs against SimJEB model 148, the one model shipped in the dataset's
sample bundle with all file types present (.fem, .vtk, .obj, .stp, field.csv). It is
the only data the local machine needs -- the full dataset is processed on Kaggle.

Point ``SIMJEB_SAMPLE_DIR`` at the extracted sample bundle if it lives elsewhere.
"""

import os
from pathlib import Path

import pytest

DEFAULT_SAMPLE_DIR = Path(
    r"D:\Vamsi_courses\Projects\3D_deep_learning_project_2\_sample"
)

# Facts about model 148, read directly off the files. Hard-coded on purpose: if a
# refactor changes what the parser returns, these catch it.
FIXTURE_ID = 148
FIXTURE_N_MESH_NODES = 129_260   # POINTS in 148.vtk
FIXTURE_N_GRID = 129_265         # GRID cards: mesh nodes + 5 rigid reference nodes


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    path = Path(os.environ.get("SIMJEB_SAMPLE_DIR", DEFAULT_SAMPLE_DIR))
    if not path.is_dir():
        pytest.skip(f"sample bundle not found at {path}; set SIMJEB_SAMPLE_DIR")
    return path


@pytest.fixture(scope="session")
def fem_path(sample_dir: Path) -> Path:
    path = sample_dir / f"{FIXTURE_ID}.fem"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return path


@pytest.fixture(scope="session")
def csv_path(sample_dir: Path) -> Path:
    path = sample_dir / f"{FIXTURE_ID}field.csv"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return path


@pytest.fixture(scope="session")
def vtk_path(sample_dir: Path) -> Path:
    path = sample_dir / f"{FIXTURE_ID}.vtk"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    return path


@pytest.fixture(scope="session")
def deck(fem_path: Path):
    """Parsed deck for model 148. Session-scoped -- the file is ~44 MB."""
    from src.data.parse_fem import parse_fem

    return parse_fem(fem_path)
