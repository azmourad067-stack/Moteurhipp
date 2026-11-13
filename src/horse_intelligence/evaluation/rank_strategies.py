"""Stratégies « par rang » : jouer chaque course le cheval classé k par le modèle ou le marché.

Même protocole que les autres simulations : probabilités hors échantillon, règlement au
rapport officiel, période de sélection puis période de test postérieure. Toutes les
règles essayées sont publiées : sur 60 règles, deux ou trois peuvent sembler rentables
sur une période par pur hasard ; seule une règle rentable sur la sélection ET sur le
test, avec un intervalle de confiance au-dessus de zéro, mériterait d'être suivie.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation.betting_simulator import payout_lookup

RANKS = (1, 2, 3, 4, 5)
BETS = ("SIMPLE_GAGNANT", "SIMPLE_PLACE")
SUBSETS = ("toutes", "quinte", "pick5")
SUBSET_BET = {"quinte": "QUINTE_PLUS", "pick5": "PICK5"}
SUBSET_LABEL = {"toutes": "toutes", "quinte": "quinté", "pick5": "pick 5"}


@dataclass(frozen=True)
class RankRule:
    source: str      # colonne de probabilité (modèle) ou "marché"
    rank: int
    bet_type: str
    subset: str

    @property
    def name(self) -> str:
        who = "marché" if self.source == "marché" else "modèle"
        bet = "gagnant" if self.bet_type == "SIMPLE_GAGNANT" else "placé"
        nth = "1er" if self.rank == 1 else f"{self.rank}e"
        return f"{nth} du {who}, {bet}, {SUBSET_LABEL[self.subset]}"


@dataclass
class RankResult:
    n_bets: int
    hit_rate: float
    roi: float
    roi_ci: tuple[float, float]


def _settle(picks: pd.DataFrame, lookup: dict[tuple[int, str, str], float], bet_type: str,
            n_boot: int = 2000, seed: int = 0) -> RankResult:
    ret = np.array([lookup.get((int(r), bet_type, str(int(n))), 0.0)
                    for r, n in zip(picks["race_id"], picks["num_pmu"], strict=True)])
    if len(ret) == 0:
        return RankResult(0, float("nan"), float("nan"), (float("nan"), float("nan")))
    profit = ret - 1.0
    rng = np.random.default_rng(seed)
    boots = profit[rng.integers(0, len(ret), size=(n_boot, len(ret)))].mean(axis=1)
    return RankResult(len(ret), float((ret > 0).mean()), float(profit.mean()),
                      (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))))


def rank_strategy_table(pred: pd.DataFrame, payouts: pd.DataFrame, model_col: str,
                        selection_share: float = 0.5) -> list[dict[str, Any]]:
    """Une ligne par règle : résultats sur la sélection, le test et toute la période."""
    lookup = payout_lookup(payouts, "INTERNET")
    paid = {k[0] for k in lookup if k[1] == "SIMPLE_GAGNANT"}
    subsets = {k: set(payouts.loc[payouts["bet_type"] == bt, "race_id"].astype(int))
               for k, bt in SUBSET_BET.items()}
    pred = pred[pred["race_id"].isin(paid)].copy()
    pred["rank_marché"] = pred.groupby("race_id")["mkt_prob"].rank(ascending=False,
                                                                     method="first")
    pred["rank_modèle"] = pred.groupby("race_id")[model_col].rank(ascending=False,
                                                                   method="first")
    dates = sorted(pred["race_date"].unique())
    cut = dates[int(len(dates) * selection_share)]
    rows = []
    for source, rank, bet, subset in product(("modèle", "marché"), RANKS, BETS, SUBSETS):
        rule = RankRule("marché" if source == "marché" else model_col, rank, bet, subset)
        p = pred[pred[f"rank_{source}"] == rank]
        if subset in subsets:
            p = p[p["race_id"].isin(subsets[subset])]
        rows.append({"rule": rule.name, **asdict(rule),
                     "selection": asdict(_settle(p[p["race_date"] < cut], lookup, bet)),
                     "test": asdict(_settle(p[p["race_date"] >= cut], lookup, bet)),
                     "all": asdict(_settle(p, lookup, bet))})
    return rows


def robust_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Règles rentables sur la sélection et sur le test, IC du test au-dessus de zéro."""
    return [r for r in rows if r["selection"]["roi"] > 0 and r["test"]["roi"] > 0
            and r["test"]["roi_ci"][0] > 0]
