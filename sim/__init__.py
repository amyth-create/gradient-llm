"""HPLC gradient simulator.

Public interface for everything downstream (gate, optimizers):

    from sim import Simulator, Method, Result, InstrumentConfig

Nothing outside ``sim/`` should import ``AnalyteSet``.
"""
from .config import CRFConfig, InstrumentConfig, Method, Result
from .simulator import Simulator

__all__ = ["Simulator", "Method", "Result", "InstrumentConfig", "CRFConfig"]
