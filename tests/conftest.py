"""
Shared fixtures for the test suite.

Run from the repo root:

    pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
DATA = REPO_ROOT / "data"

# Project settings, mirroring what run.sh exports.
# utils.py and features/preprocessing.py read PROJECT_DIR at import time, so it must
# be set *before* any src/ module is imported -- which is why this happens in conftest.
os.environ["PROJECT_DIR"] = str(REPO_ROOT)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def inventory():
    """The raw ABIDE inventory: one row per participant."""
    path = DATA / "data_inventory.csv"
    if not path.exists():
        pytest.skip(f"{path} not found")
    return pd.read_csv(path, low_memory=False)


@pytest.fixture(scope="session")
def manifest():
    """The sagittal manifest that train_baseline.py consumes: one row per scan."""
    path = DATA / "manifest_sagittal.csv"
    if not path.exists():
        pytest.skip(f"{path} not found (run run.sh -m)")
    return pd.read_csv(path)


@pytest.fixture
def synthetic_manifest():
    """Build a small in-memory manifest for exercising the split logic.

    Sized so that every select_data subset has enough subjects of both classes to
    stratify -- which the real manifest cannot guarantee (it has no aq_outlier
    rows at all).

    Age is constant within a site on purpose. The stratification key is
    (label, site, age), so varying age per subject would create hundreds of
    three-row strata; those fall below MIN_STRATUM_SIZE, get merged into
    label_rare, and the resulting proportions become hard to reason about. One
    age per site keeps six well-populated strata.
    """

    def _make(n_per_site=60, sites=("NYU", "PITT", "YALE")):
        rows = []
        sub_id = 50000
        for site_idx, site in enumerate(sites):
            for i in range(n_per_site):
                rows.append({
                    "sub_id": sub_id,
                    "site": site,
                    "label": i % 2,
                    "age": 20.0 + 5 * site_idx,
                    "sex": 1 if i % 2 else 2,
                    "aq_outlier": (i % 4) < 2,
                    "unclear_diag": i % 3 == 0,
                })
                sub_id += 1
        return pd.DataFrame(rows)

    return _make