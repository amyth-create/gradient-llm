"""Pydantic models for every input and output of the simulator.

Units: lengths as named in the field (mm, um), volumes in mL, flow in mL/min,
time in min, temperature in degrees C at the interface (kelvin inside the
retention model).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
ANALYTE_DIR = CONFIG_DIR / "analytes"

# Data roots. The project root holds the generated sets; every extra root mirrors the
# same layout (configs/analytes, tables, results) for a family of sets kept apart, so
# an analyte set is found by name wherever its YAML lives and its optimum, table and
# results stay next to it.
PROJECT_ROOT = CONFIG_DIR.parent
EXTRA_ROOTS: tuple[Path, ...] = (PROJECT_ROOT / "hplcsim",)
ROOTS: tuple[Path, ...] = (PROJECT_ROOT, *EXTRA_ROOTS)


def set_root(set_name: str) -> Path:
    """The root whose ``configs/analytes/<set_name>.yaml`` exists; the project root for a set that does not exist yet."""
    for root in ROOTS:
        if (root / "configs" / "analytes" / f"{set_name}.yaml").exists():
            return root
    return PROJECT_ROOT


def analyte_dir(set_name: str) -> Path:
    return set_root(set_name) / "configs" / "analytes"


def table_dir(set_name: str) -> Path:
    return set_root(set_name) / "tables"


RESULTS_DIR_VAR = "HPLC_RESULTS_DIR"


def results_dir(set_name: str) -> Path:
    """``<set root>/results``, or ``$HPLC_RESULTS_DIR`` for every set when that is set.

    The override keeps one run's outputs apart from the committed ones, e.g. the LLM
    transcripts and tables of a second deployment under ``deployments/<name>/results``;
    without it the resumable LLM runner would pick up the first model's transcripts.
    """
    override = os.environ.get(RESULTS_DIR_VAR)
    return Path(override).expanduser() if override else set_root(set_name) / "results"

# Order of the decision vector used by optimizers: a box in (phi0, dphi, tG, T).
VECTOR_NAMES: tuple[str, ...] = ("phi0", "dphi", "tG", "T")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Instrument
# --------------------------------------------------------------------------- #
class ViscosityModel(_Frozen):
    """Water/acetonitrile viscosity surface, mPa.s. Used only by the standalone ``sim.pressure`` helper.

    eta(phi, T) = [eta_water + a_mix*phi*(1-phi) - c_lin*phi] * exp(-T_coef*(T - T_ref_C))
    """

    eta_water: float = 0.89
    a_mix: float = 0.87
    c_lin: float = 0.52
    T_coef: float = 0.021
    T_ref_C: float = 25.0

    @property
    def phi_max(self) -> float:
        """Composition at which the (concave) viscosity polynomial peaks."""
        if self.a_mix <= 0:
            return 0.0
        return float(np.clip((self.a_mix - self.c_lin) / (2.0 * self.a_mix), 0.0, 1.0))


class MethodBounds(_Frozen):
    """Box visible to the optimizer. phi1 is bounded by phi0 + min_dphi below and phi1_max above."""

    phi0: tuple[float, float] = (0.05, 0.60)
    phi1_max: float = 0.95
    min_dphi: float = 0.10
    tG: tuple[float, float] = (2.0, 40.0)
    T: tuple[float, float] = (25.0, 60.0)

    @property
    def dphi(self) -> tuple[float, float]:
        """Box bounds on dphi = phi1 - phi0 (a superset: phi1 <= phi1_max is re-checked)."""
        return (self.min_dphi, self.phi1_max - self.phi0[0])


class DetectorConfig(_Frozen):
    """Detector used by ``sim.chromatogram`` to turn a retention table into a trace.

    The retention model and the CRF never look at it. ``noise_sd`` and ``drift``
    are the defaults that ``sim.chromatogram.trace`` falls back to when a call
    does not set them; at 0.0 every trace is deterministic and noise-free.
    """

    sampling_rate_hz: float = Field(default=10.0, gt=0.0, description="acquisition rate; samples are 1/(60 f) min apart")
    noise_sd: float = Field(default=0.0, ge=0.0, description="white-noise standard deviation, signal units")
    drift: float = Field(default=0.0, description="total linear baseline rise over the run, signal units")


class InstrumentConfig(_Frozen):
    L_mm: float = 100.0
    dc_mm: float = 2.1
    dp_um: float = 1.8
    eps: float = 0.65
    F_mL_min: float = 0.5
    VD_mL: float = 0.10
    N: float = 15000.0
    sigma_ec_min: float = 0.02
    t_max_min: float = 45.0
    t_eq_min: float = 3.0
    flow_resistance: float = 500.0
    T_ref_C: float = 30.0
    viscosity: ViscosityModel = ViscosityModel()
    method_bounds: MethodBounds = MethodBounds()
    detector: DetectorConfig = DetectorConfig()

    # ---- derived quantities ------------------------------------------------
    @property
    def V0_mL(self) -> float:
        """Dead volume V0 = eps * pi * (dc/2)^2 * L (mL)."""
        r_cm = self.dc_mm / 20.0
        L_cm = self.L_mm / 10.0
        return self.eps * math.pi * r_cm**2 * L_cm

    @property
    def t0(self) -> float:
        """Dead time t0 = V0 / F (min)."""
        return self.V0_mL / self.F_mL_min

    @property
    def tD(self) -> float:
        """Dwell time tD = VD / F (min)."""
        return self.VD_mL / self.F_mL_min

    @property
    def t_end(self) -> float:
        """End of data acquisition t_end = t_max - t_eq (min): peaks with t_R > t_end are not observed."""
        return self.t_max_min - self.t_eq_min

    # ---- optimizer-facing bounds ------------------------------------------
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Box bounds on the decision vector (phi0, dphi, tG, T)."""
        b = self.method_bounds
        return {"phi0": b.phi0, "dphi": b.dphi, "tG": b.tG, "T": b.T}

    def box(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) arrays over VECTOR_NAMES."""
        bd = self.bounds()
        lo = np.array([bd[n][0] for n in VECTOR_NAMES], dtype=float)
        hi = np.array([bd[n][1] for n in VECTOR_NAMES], dtype=float)
        return lo, hi

    # ---- io -----------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "InstrumentConfig":
        path = Path(path) if path is not None else CONFIG_DIR / "instrument.yaml"
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))


# --------------------------------------------------------------------------- #
# Analytes (hidden from optimizers)
# --------------------------------------------------------------------------- #
class Analyte(_Frozen):
    log_kw: float = Field(description="log10 retention factor in pure water at T_ref")
    S: float = Field(description="solvent-strength slope, d(log k)/d(phi)")
    dH: float = Field(description="van 't Hoff term, K*log10 units; negative -> retention falls as T rises")
    name: str | None = None


class AnalyteSet(_Frozen):
    name: str
    analytes: list[Analyte]
    seed: int | None = None

    def __len__(self) -> int:
        return len(self.analytes)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(log_kw, S, dH) as float arrays of shape (n,)."""
        log_kw = np.array([a.log_kw for a in self.analytes], dtype=float)
        S = np.array([a.S for a in self.analytes], dtype=float)
        dH = np.array([a.dH for a in self.analytes], dtype=float)
        return log_kw, S, dH

    @classmethod
    def from_rows(cls, name: str, rows: Iterable[Sequence[float]], seed: int | None = None) -> "AnalyteSet":
        return cls(name=name, seed=seed,
                   analytes=[Analyte(log_kw=float(r[0]), S=float(r[1]), dH=float(r[2])) for r in rows])

    @classmethod
    def load(cls, name_or_path: str | Path) -> "AnalyteSet":
        p = Path(name_or_path)
        if p.suffix != ".yaml":
            p = analyte_dir(str(name_or_path)) / f"{name_or_path}.yaml"
        with open(p) as fh:
            data = yaml.safe_load(fh)
        if "name" not in data:
            data["name"] = p.stem
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "seed": self.seed,
            "analytes": [a.model_dump(exclude_none=True) for a in self.analytes],
        }
        with open(path, "w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)


# --------------------------------------------------------------------------- #
# CRF
# --------------------------------------------------------------------------- #
class CRFConfig(_Frozen):
    R_target: float = 1.5
    t_target: float = 10.0
    w_t: float = 0.5

    @classmethod
    def load(cls, path: str | Path | None = None) -> "CRFConfig":
        path = Path(path) if path is not None else CONFIG_DIR / "crf.yaml"
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))


# --------------------------------------------------------------------------- #
# Method and Result
# --------------------------------------------------------------------------- #
class Method(_Frozen):
    """One linear gradient: phi0 -> phi1 over tG minutes at temperature T, then hold at phi1.

    Flow rate is fixed by the instrument. Bounds are *not* enforced here: an
    out-of-bounds method is a legal input that the simulator reports as
    ``feasible = False`` with a reason.
    """

    phi0: float = Field(ge=0.0)
    phi1: float = Field(ge=0.0)
    tG: float = Field(gt=0.0)
    T: float = Field(gt=-273.15, description="column temperature, degrees C")

    @property
    def dphi(self) -> float:
        return self.phi1 - self.phi0

    def to_vector(self) -> np.ndarray:
        """(phi0, dphi, tG, T) — the box parametrisation used by optimizers."""
        return np.array([self.phi0, self.dphi, self.tG, self.T], dtype=float)

    @classmethod
    def from_vector(cls, v: Sequence[float]) -> "Method":
        phi0, dphi, tG, T = (float(x) for x in v)
        return cls(phi0=phi0, phi1=phi0 + dphi, tG=tG, T=T)


class Result(_Frozen):
    """What one injection returns.

    The run ends at ``InstrumentConfig.t_end`` (t_max - t_eq). Analytes still on
    the column then are not observed: the table is shorter, ``n_missing`` counts
    them, and the CRF scores each lost peak like a fully co-eluting one with the
    time term charged at t_end. Only an out-of-bounds method is infeasible.
    """

    retention_table: list[tuple[float, float]] = Field(
        description="(t_R [min], sigma [min]) sorted by t_R, one row per observed analyte (t_R <= t_end), "
                    "no identities; empty if infeasible")
    crf: float = Field(description="CRF score over the observed peaks (lost peaks score as co-eluting); nan if infeasible")
    run_time: float = Field(description="min(t_R of last peak, t_end) + re-equilibration (min)")
    feasible: bool
    reason: str | None = Field(default=None, description="why the method is infeasible (bounds), if it is")
    n_missing: int = Field(default=0, ge=0,
                           description="analytes not observed: eluting after t_end, or all of them if infeasible")

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2)
