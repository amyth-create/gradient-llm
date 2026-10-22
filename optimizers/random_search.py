"""Random search: a seeded permutation of the table rows, one injection each."""
from __future__ import annotations

import numpy as np

from .oracle import SearchRun, TableOracle


def random_order(oracle: TableOracle, seed: int) -> np.ndarray:
    """The seeded row order shared by random search and BO's initial design."""
    return np.random.default_rng(seed).permutation(oracle.n_candidates)


def random_search(oracle: TableOracle, budget: int = 25, seed: int = 0) -> SearchRun:
    run = oracle.new_run("random", seed, budget)
    for idx in random_order(oracle, seed)[:budget]:
        oracle.query(run, int(idx))
    return run
