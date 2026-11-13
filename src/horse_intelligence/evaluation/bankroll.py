"""Simulation de bankroll (Monte-Carlo) pour une règle de mise fixe.

On ne cherche pas ici à savoir si la règle est rentable (c'est le rôle des tests sur
période non vue), mais à montrer à quoi ressemble une année réelle de paris : séries de
défaites, creux du budget, probabilité de finir en perte ou d'épuiser son budget.

Méthode : on tire au hasard, avec remise, des paris réellement joués par la règle
(gain net au rapport officiel pour 1 €) pour fabriquer des milliers d'années possibles.
Comme la rentabilité réelle de la règle est incertaine, chaque simulation est refaite
sous plusieurs hypothèses de rentabilité : les rapports gagnants sont mis à l'échelle
pour que le ROI moyen vaille la valeur visée, sans changer la fréquence des gains ni la
forme de la distribution des rapports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    roi: float | None  # None : rentabilité observée, sans ajustement


@dataclass
class BankrollResult:
    scenario: str
    roi: float
    n_bets: int
    final_quantiles: dict[str, float]   # gain final en mises (P5, P25, médiane, P75, P95)
    p_profit: float                     # probabilité de finir l'année en gain
    longest_losing_streak: dict[str, float]  # médiane, P90, P99
    max_drawdown: dict[str, float]           # creux maximal en mises : médiane, P90, P99
    p_ruin: dict[int, float]            # budget (en mises) -> probabilité de l'épuiser


def rescale_to_roi(profits: np.ndarray, target_roi: float) -> np.ndarray:
    """Met à l'échelle les rapports gagnants pour que le gain moyen par mise vaille
    `target_roi`. Les paris perdants restent à −1."""
    returns = profits + 1.0
    mean_return = float(returns.mean())
    if mean_return <= 0:
        raise ValueError("aucun pari gagnant : impossible d'ajuster la rentabilité")
    return returns * (1.0 + target_roi) / mean_return - 1.0


def _longest_losing_streak(paths: np.ndarray) -> np.ndarray:
    """Plus longue série de paris perdants (gain −1) pour chaque ligne."""
    losing = paths <= -1.0 + 1e-12
    best = np.zeros(len(paths), dtype=int)
    run = np.zeros(len(paths), dtype=int)
    for j in range(paths.shape[1]):
        run = np.where(losing[:, j], run + 1, 0)
        best = np.maximum(best, run)
    return best


def simulate(profits: np.ndarray, n_bets: int, n_sims: int = 10_000,
             bankrolls: tuple[int, ...] = (50, 100, 150, 200), seed: int = 0,
             scenario: str = "observé") -> BankrollResult:
    """`profits` : gain net de chaque pari historique pour 1 € (−1 si perdu)."""
    if len(profits) == 0 or n_bets <= 0:
        raise ValueError("aucun pari à simuler")
    rng = np.random.default_rng(seed)
    paths = profits[rng.integers(0, len(profits), size=(n_sims, n_bets))]
    cum = paths.cumsum(axis=1)
    final = cum[:, -1]
    peak = np.maximum.accumulate(np.concatenate([np.zeros((n_sims, 1)), cum], axis=1),
                                 axis=1)[:, 1:]
    drawdown = (peak - cum).max(axis=1)
    lowest = np.minimum(cum.min(axis=1), 0.0)
    streak = _longest_losing_streak(paths)

    def q(x: np.ndarray, *ps: float) -> list[float]:
        return [float(np.quantile(x, p)) for p in ps]

    f5, f25, f50, f75, f95 = q(final, 0.05, 0.25, 0.5, 0.75, 0.95)
    s50, s90, s99 = q(streak.astype(float), 0.5, 0.9, 0.99)
    d50, d90, d99 = q(drawdown, 0.5, 0.9, 0.99)
    return BankrollResult(
        scenario=scenario, roi=float(profits.mean()), n_bets=n_bets,
        final_quantiles={"P5": f5, "P25": f25, "médiane": f50, "P75": f75, "P95": f95},
        p_profit=float((final > 0).mean()),
        longest_losing_streak={"médiane": s50, "P90": s90, "P99": s99},
        max_drawdown={"médiane": d50, "P90": d90, "P99": d99},
        p_ruin={b: float((lowest <= -b).mean()) for b in bankrolls})


DEFAULT_SCENARIOS = (
    Scenario("observé sur 2 ans", None),
    Scenario("neutre", 0.0),
    Scenario("prudent", -0.05),
    Scenario("pessimiste (rang 3 ordinaire)", -0.13),
)


def run_scenarios(profits: np.ndarray, n_bets: int, n_sims: int = 10_000,
                  scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
                  bankrolls: tuple[int, ...] = (50, 100, 150, 200),
                  seed: int = 0) -> list[BankrollResult]:
    out = []
    for i, sc in enumerate(scenarios):
        p = profits if sc.roi is None else rescale_to_roi(profits, sc.roi)
        out.append(simulate(p, n_bets, n_sims, bankrolls, seed + i, sc.name))
    return out
