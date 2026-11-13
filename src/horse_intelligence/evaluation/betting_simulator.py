"""Simulateur de paris historique (analyse de rentabilité THÉORIQUE).

Règles anti-biais :
- la DÉCISION n'utilise que des informations antérieures à l'instant de décision :
  probabilités hors échantillon (walk-forward) et cote de référence publiée avant ;
- le RÈGLEMENT utilise le rapport définitif officiel du pari mutuel (le gain réel
  qu'aurait perçu un parieur), jamais la cote de référence ;
- les règles de sélection sont choisies sur une période de SÉLECTION puis appliquées
  sans modification à une période de TEST postérieure : seul le test est annoncé ;
- une course sans rapport collecté est exclue (et comptée), jamais supposée perdante
  ou gagnante.

Aucune espérance positive estimée n'est une garantie de rentabilité.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.models.ranking import paid_places, place_probabilities

BET_TYPES = ("SIMPLE_GAGNANT", "SIMPLE_PLACE")


@dataclass(frozen=True)
class Strategy:
    name: str
    bet_type: str = "SIMPLE_GAGNANT"
    prob_col: str = "lgbm_market"
    min_ev: float = 0.0              # p × cote_ref − 1 minimal (simple gagnant)
    min_prob: float = 0.0            # probabilité minimale (gagnant ou placé selon le pari)
    max_odds: float = 1e9
    one_per_race: bool = True        # meilleure sélection seulement
    rank_by: str = "ev"              # ev | prob : critère de choix dans la course
    staking: str = "flat"            # flat | kelly
    kelly_fraction: float = 0.1      # Kelly fractionné (théorique)
    max_stake_share: float = 0.02    # mise maximale en part de bankroll


@dataclass
class SimulationResult:
    strategy: dict[str, Any]
    n_races_considered: int
    n_races_without_payouts: int
    n_bets: int
    staked: float
    returned: float
    roi: float
    roi_ci: tuple[float, float]
    hit_rate: float
    max_drawdown: float
    longest_losing_streak: int
    final_bankroll: float | None
    equity: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def payout_lookup(payouts: pd.DataFrame, pool: str) -> dict[tuple[int, str, str], float]:
    p = payouts[payouts["pool"] == pool]
    p = p[p["payout_per_eur"].notna()]
    return {(int(rid), str(bt), str(comb)): float(val) for rid, bt, comb, val in zip(
        p["race_id"], p["bet_type"], p["combination"], p["payout_per_eur"], strict=True)}


def add_place_probabilities(pred: pd.DataFrame, prob_col: str) -> pd.Series:
    out = pd.Series(np.nan, index=pred.index)
    for _, g in pred.groupby("race_id"):
        p = g[prob_col].to_numpy()
        if np.isnan(p).any():
            continue
        declared = g["declared_runners"].iloc[0]
        n = paid_places(int(declared) if pd.notna(declared) else len(g))
        if n:
            out.loc[g.index] = place_probabilities(p, n)
    return out


def select_bets(pred: pd.DataFrame, s: Strategy) -> pd.DataFrame:
    f = pred.dropna(subset=[s.prob_col, "mkt_odds"]).copy()
    f["p"] = f[s.prob_col]
    f["ev"] = f["p"] * f["mkt_odds"] - 1.0
    if s.bet_type == "SIMPLE_GAGNANT":
        f = f[(f["ev"] >= s.min_ev) & (f["p"] >= s.min_prob) & (f["mkt_odds"] <= s.max_odds)]
        score = f["ev"] if s.rank_by == "ev" else f["p"]
    else:  # pas de cote placé avant la course : sélection sur la probabilité seulement
        f["p_place"] = add_place_probabilities(pred, s.prob_col).loc[f.index]
        f = f[(f["p_place"] >= s.min_prob) & (f["mkt_odds"] <= s.max_odds)]
        score = f["p_place"]
    if s.one_per_race and not f.empty:
        # départage déterministe des égalités (cotes identiques) : plus petit numéro
        order = f.assign(_score=score.round(10)).sort_values(
            ["race_id", "_score", "num_pmu"], ascending=[True, False, True])
        f = f.loc[order.drop_duplicates("race_id").index]
    return f.sort_values(["race_date", "race_id"])


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))
    return float((peak[1:] - equity).max()) if equity.size else 0.0


def _longest_losing_streak(results: np.ndarray) -> int:
    best = cur = 0
    for r in results:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return best


def simulate(pred: pd.DataFrame, payouts: pd.DataFrame, s: Strategy, pool: str = "INTERNET",
             bankroll: float = 100.0, n_boot: int = 2000, seed: int = 0) -> SimulationResult:
    lookup = payout_lookup(payouts, pool)
    races_with_payouts = {k[0] for k in lookup if k[1] == "SIMPLE_GAGNANT"}
    considered = pred["race_id"].nunique()
    missing = len(set(pred["race_id"]) - races_with_payouts)
    bets = select_bets(pred[pred["race_id"].isin(races_with_payouts)], s)

    stakes: list[float] = []
    returns: list[float] = []
    rows: list[dict[str, Any]] = []
    bank = bankroll
    for b in bets.to_dict("records"):
        p, odds = float(b["p"]), float(b["mkt_odds"])
        if s.staking == "kelly":
            # Kelly sur la cote de référence : le rapport réel sera différent (pari mutuel)
            edge = (p * odds - 1) / (odds - 1) if odds > 1 else 0.0
            stake = round(min(max(edge, 0) * s.kelly_fraction, s.max_stake_share) * bank, 2)
            if stake <= 0:
                continue
        else:
            stake = 1.0
        race_id, num = int(b["race_id"]), int(b["num_pmu"])
        payout = lookup.get((race_id, s.bet_type, str(num)), 0.0)
        ret = stake * payout
        bank += ret - stake
        stakes.append(stake)
        returns.append(ret)
        rows.append({"date": str(b["race_date"]), "race_id": race_id, "num_pmu": num,
                     "stake": stake, "return": ret, "bankroll": bank})
    st, rt = np.array(stakes), np.array(returns)
    profit = rt - st
    staked = float(st.sum())
    roi = float(profit.sum() / staked) if staked else float("nan")
    if len(st) >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(st), size=(n_boot, len(st)))
        boots = profit[idx].sum(axis=1) / st[idx].sum(axis=1)
        ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    else:
        ci = (float("nan"), float("nan"))
    return SimulationResult(
        strategy=asdict(s), n_races_considered=int(considered),
        n_races_without_payouts=missing, n_bets=len(st), staked=staked,
        returned=float(rt.sum()), roi=roi, roi_ci=ci,
        hit_rate=float((rt > 0).mean()) if len(rt) else float("nan"),
        max_drawdown=_max_drawdown(np.cumsum(profit)),
        longest_losing_streak=_longest_losing_streak(profit),
        final_bankroll=bank if s.staking == "kelly" else None, equity=rows)


def strategy_grid(prob_col: str) -> list[Strategy]:
    grid = [Strategy(f"{prob_col} EV≥{ev:.2f} p≥{mp:.2f} cote≤{mo:g}", "SIMPLE_GAGNANT",
                     prob_col, min_ev=ev, min_prob=mp, max_odds=mo)
            for ev, mp, mo in product((0.0, 0.05, 0.10, 0.20), (0.0, 0.05, 0.10),
                                      (10.0, 30.0, 1e9))]
    grid += [Strategy(f"{prob_col} placé p≥{mp:.2f}", "SIMPLE_PLACE", prob_col, min_prob=mp)
             for mp in (0.5, 0.6, 0.7)]
    return grid


def reference_strategies() -> list[Strategy]:
    """Repères sans modèle : ils mesurent le coût du prélèvement du pari mutuel."""
    return [Strategy("favori du marché (gagnant)", "SIMPLE_GAGNANT", "mkt_prob", min_ev=-1.0,
                     rank_by="prob"),
            Strategy("favori du marché (placé)", "SIMPLE_PLACE", "mkt_prob")]


@dataclass
class SelectionOutcome:
    selection_period: tuple[str, str]
    test_period: tuple[str, str]
    n_strategies_tried: int
    chosen: dict[str, Any] | None
    chosen_on_selection: dict[str, Any] | None
    test: dict[str, Any] | None
    references_test: list[dict[str, Any]]
    message: str


MIN_BETS_SELECTION = 30


def select_then_test(pred: pd.DataFrame, payouts: pd.DataFrame, prob_col: str,
                     selection_share: float = 0.5,
                     grid: Callable[[str], list[Strategy]] = strategy_grid) -> SelectionOutcome:
    """Choisit la meilleure règle sur la période de sélection, l'applique au test."""
    dates = sorted(pred["race_date"].unique())
    cut = dates[int(len(dates) * selection_share)]
    sel, test = pred[pred["race_date"] < cut], pred[pred["race_date"] >= cut]
    periods = ((str(dates[0]), str(max(sel["race_date"]))),
               (str(cut), str(dates[-1])))
    candidates = grid(prob_col)
    results = [(s, simulate(sel, payouts, s)) for s in candidates]
    eligible = [(s, r) for s, r in results if r.n_bets >= MIN_BETS_SELECTION]
    refs = [simulate(test, payouts, s).as_dict() for s in reference_strategies()]
    if not eligible:
        return SelectionOutcome(*periods, len(candidates), None, None, None, refs,
                                "Aucune règle n'a produit assez de paris en sélection.")
    best_s, best_r = max(eligible, key=lambda t: t[1].roi)
    if not best_r.roi > 0:
        return SelectionOutcome(*periods, len(candidates), asdict(best_s), best_r.as_dict(),
                                None, refs,
                                "Aucune règle n'est rentable sur la période de sélection : "
                                "aucune stratégie n'est retenue pour le test.")
    test_r = simulate(test, payouts, best_s)
    msg = ("Rentabilité observée sur le test, à confirmer (IC du ROI)." if test_r.roi > 0 else
           "La règle retenue en sélection n'est pas rentable sur le test.")
    return SelectionOutcome(*periods, len(candidates), asdict(best_s), best_r.as_dict(),
                            test_r.as_dict(), refs, msg)
