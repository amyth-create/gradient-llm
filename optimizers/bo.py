"""Bayesian optimisation over the table: Gaussian process surrogate + expected improvement.

The surrogate is a GP with an ARD Matern-5/2 kernel on the unit-cube features
(phi0, phi1, tG, T). Hyperparameters are refit at every step by maximising the
log marginal likelihood plus a lognormal prior on the lengthscales (a MAP fit);
the prior keeps the model sensible while there are only a handful of points,
and fades as data accumulate. The acquisition is expected improvement evaluated
on every not-yet-queried table row, so the candidate set is exactly the oracle's
grid. A truncated run (peaks still on the column when the run ends) is an
ordinary observation at its CRF: the score already carries the penalty. Only an
out-of-bounds injection (no chromatogram; none in the default tables) enters the
surrogate at a fixed penalty value so the model learns to avoid that region.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import norm

from .oracle import SearchRun, TableOracle
from .random_search import random_order


@dataclass(frozen=True)
class BOConfig:
    n_init: int = 8                                   # random injections before the surrogate takes over (2 x dims)
    xi: float = 0.01                                  # EI exploration margin (target units)
    infeasible_value: float | None = -0.5             # surrogate target for a bounds-infeasible injection;
                                                      # None = running minimum of feasible CRFs minus 0.1
    ls_bounds: tuple[float, float] = (0.05, 2.0)      # lengthscale bounds, unit-cube units
    sf2_bounds: tuple[float, float] = (0.05, 5.0)     # signal variance bounds (standardised targets)
    noise_bounds: tuple[float, float] = (1e-4, 0.1)   # noise variance bounds (standardised targets)
    ls_prior: tuple[float, float] | None = (0.5, 0.7) # lognormal prior on lengthscales: (median, sd of log)
    restarts: int = 1                                 # random restarts on top of two fixed starts


def matern52(A: np.ndarray, B: np.ndarray, lengthscales: np.ndarray, signal_var: float) -> np.ndarray:
    d = np.sqrt(5.0) * cdist(A / lengthscales, B / lengthscales)
    return signal_var * (1.0 + d + d**2 / 3.0) * np.exp(-d)


@dataclass
class GP:
    """Zero-mean GP on standardised targets."""

    X: np.ndarray
    y: np.ndarray
    lengthscales: np.ndarray
    signal_var: float
    noise_var: float

    def __post_init__(self):
        self.y_mean = float(self.y.mean())
        self.y_std = float(self.y.std()) or 1.0
        z = (self.y - self.y_mean) / self.y_std
        K = matern52(self.X, self.X, self.lengthscales, self.signal_var) + self.noise_var * np.eye(len(z))
        self.L = cholesky(K, lower=True)
        self.alpha = cho_solve((self.L, True), z)

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation in the original target units."""
        Ks = matern52(Xs, self.X, self.lengthscales, self.signal_var)
        mu = Ks @ self.alpha
        v = solve_triangular(self.L, Ks.T, lower=True)
        var = np.maximum(self.signal_var - (v**2).sum(axis=0), 1e-12)
        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std

    @staticmethod
    def _objective(theta: np.ndarray, X: np.ndarray, z: np.ndarray, prior: tuple[float, float] | None) -> float:
        """Negative log marginal likelihood, plus the negative log prior on lengthscales if given."""
        d = X.shape[1]
        ls, sf2, noise = np.exp(theta[:d]), float(np.exp(theta[d])), float(np.exp(theta[d + 1]))
        K = matern52(X, X, ls, sf2) + noise * np.eye(len(z))
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e25
        alpha = cho_solve((L, True), z)
        nlml = 0.5 * z @ alpha + np.log(np.diag(L)).sum() + 0.5 * len(z) * np.log(2.0 * np.pi)
        if prior is not None:
            median, sd = prior
            nlml += float(((theta[:d] - np.log(median)) ** 2).sum() / (2.0 * sd**2))
        return float(nlml)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, rng: np.random.Generator, cfg: BOConfig = BOConfig()) -> "GP":
        """MAP fit of lengthscales, signal variance and noise from fixed starts plus random restarts."""
        d = X.shape[1]
        y_std = float(y.std()) or 1.0
        z = (y - y.mean()) / y_std
        log = np.log
        bounds = [(log(cfg.ls_bounds[0]), log(cfg.ls_bounds[1]))] * d + \
                 [(log(cfg.sf2_bounds[0]), log(cfg.sf2_bounds[1])), (log(cfg.noise_bounds[0]), log(cfg.noise_bounds[1]))]
        lo, hi = np.array(bounds)[:, 0], np.array(bounds)[:, 1]
        ls0 = cfg.ls_prior[0] if cfg.ls_prior else 0.5
        starts = [np.clip(np.r_[np.full(d, log(ls0)), 0.0, log(1e-3)], lo, hi),
                  np.clip(np.r_[np.full(d, log(2.0 * ls0)), 0.0, log(1e-2)], lo, hi)]
        starts += [rng.uniform(lo, hi) for _ in range(cfg.restarts)]
        best_theta, best_val = starts[0], np.inf
        for theta0 in starts:
            res = minimize(cls._objective, theta0, args=(X, z, cfg.ls_prior), method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 200})
            if res.fun < best_val:
                best_theta, best_val = res.x, res.fun
        return cls(X=X, y=y, lengthscales=np.exp(best_theta[:d]),
                   signal_var=float(np.exp(best_theta[d])), noise_var=float(np.exp(best_theta[d + 1])))


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    gain = mu - best - xi
    z = gain / sd
    return gain * norm.cdf(z) + sd * norm.pdf(z)


def surrogate_targets(run: SearchRun, cfg: BOConfig) -> np.ndarray:
    """Observed CRFs with (bounds-)infeasible injections replaced by the configured penalty."""
    crf = np.array([o.crf for o in run.observations], dtype=float)
    feasible = np.isfinite(crf)
    if cfg.infeasible_value is None:
        penalty = (crf[feasible].min() - 0.1) if feasible.any() else -0.5
    else:
        penalty = cfg.infeasible_value
    return np.where(feasible, crf, penalty)


def bo_search(oracle: TableOracle, budget: int = 25, seed: int = 0, config: BOConfig | None = None,
              **overrides) -> SearchRun:
    """GP-EI Bayesian optimisation. The first ``n_init`` injections are the same rows random
    search uses for this seed, so the two strategies share a starting point."""
    cfg = replace(config or BOConfig(), **overrides)
    rng = np.random.default_rng(seed + 10_000)
    run = oracle.new_run("bo", seed, budget)
    for idx in random_order(oracle, seed)[:min(cfg.n_init, budget)]:
        oracle.query(run, int(idx))
    while len(run) < budget:
        idx = np.array(run.indices())
        y = surrogate_targets(run, cfg)
        gp = GP.fit(oracle.X_unit[idx], y, rng, cfg)
        mask = np.ones(oracle.n_candidates, dtype=bool)
        mask[idx] = False
        cand = np.flatnonzero(mask)
        mu, sd = gp.predict(oracle.X_unit[cand])
        ei = expected_improvement(mu, sd, float(y.max()), cfg.xi)
        ei = ei + 1e-12 * rng.random(ei.shape)          # seeded tie-break
        oracle.query(run, int(cand[np.argmax(ei)]))
    return run
