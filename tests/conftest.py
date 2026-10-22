import numpy as np
import pytest

from sim.config import AnalyteSet, CRFConfig, InstrumentConfig


@pytest.fixture(scope="session")
def inst() -> InstrumentConfig:
    return InstrumentConfig.load()


@pytest.fixture(scope="session")
def crf_cfg() -> CRFConfig:
    return CRFConfig.load()


@pytest.fixture(scope="session")
def three_analytes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A weakly retained, a typical and a strongly retained analyte: covers all three regimes."""
    log_kw = np.array([0.6, 3.0, 4.3])
    S = np.array([5.0, 6.0, 5.0])
    dH = np.array([-1500.0, -2000.0, -2500.0])
    return log_kw, S, dH


@pytest.fixture(scope="session")
def demo_set() -> AnalyteSet:
    return AnalyteSet.from_rows("demo", [
        (2.0, 5.0, -1500.0), (2.5, 6.5, -2000.0), (3.0, 6.0, -1200.0),
        (3.05, 6.2, -2500.0), (3.8, 7.0, -1800.0), (4.2, 5.0, -2200.0),
    ])
