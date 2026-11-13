"""Simulation historique des paris combinés (couplé, trio, 2 sur 4) — analyse THÉORIQUE.

Mêmes règles anti-biais que le simulateur du simple (`betting_simulator`) :
- décision sur probabilités hors échantillon (walk-forward) uniquement ;
- règlement au rapport définitif officiel de la combinaison ;
- règle choisie sur une période de sélection, puis appliquée telle quelle à une période
  de test postérieure ; une course sans rapport pour le pari est exclue.

Probabilité d'une combinaison : arrivées simulées selon le modèle de Plackett-Luce
(tirage de Gumbel) à partir des probabilités de victoire. Comparer la probabilité du
modèle à celle déduite du marché du simple revient à chercher les combinaisons que le
pari mutuel paierait trop cher si sa masse était cohérente avec celle du simple.
Le prix réel de la combinaison avant la course n'est pas connu ici : seul le rapport
final l'est. Aucune espérance positive estimée n'est une garantie de rentabilité.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd

# pari -> (chevaux choisis, rang d'arrivée maximal) ; combinaisons sans ordre
COMBO_BETS: dict[str, tuple[int, int]] = {
    "COUPLE_GAGNANT": (2, 2),     # les deux premiers
    "COUPLE_PLACE": (2, 3),       # deux chevaux parmi les trois premiers
    "TRIO": (3, 3),               # les trois premiers
    "DEUX_SUR_QUATRE": (2, 4),    # deux chevaux parmi les quatre premiers
}
N_SAMPLES = 4000
MIN_BETS_SELECTION = 30

Combo = tuple[int, ...]


def combo_probabilities(nums: np.ndarray, p: np.ndarray, k: int, m: int,
                        rng: np.random.Generator,
                        n_samples: int = N_SAMPLES) -> dict[Combo, float]:
    """Probabilité que les k chevaux de chaque combinaison finissent tous dans les m premiers."""
    n = len(nums)
    if n < m or not np.isfinite(p).all() or p.sum() <= 0:
        return {}
    logp = np.log(np.clip(p / p.sum(), 1e-12, None))
    keys = logp + rng.gumbel(size=(n_samples, n))
    top = np.sort(np.argpartition(-keys, m - 1, axis=1)[:, :m], axis=1)
    counts: dict[Combo, int] = {}
    for sub in combinations(range(m), k):
        rows, cnt = np.unique(top[:, list(sub)], axis=0, return_counts=True)
        for r, c in zip(rows, cnt, strict=True):
            key = tuple(int(nums[i]) for i in r)
            counts[key] = counts.get(key, 0) + int(c)
    return {c: v / n_samples for c, v in counts.items()}


def winning_combos(payouts: pd.DataFrame, pool: str = "INTERNET"
                   ) -> dict[tuple[int, str], dict[Combo, float]]:
    """(course, pari) -> {combinaison triée: rapport pour 1 €}, hors lignes de non-partants."""
    p = payouts[(payouts["pool"] == pool) & payouts["bet_type"].isin(COMBO_BETS)
                & payouts["payout_per_eur"].notna()]
    out: dict[tuple[int, str], dict[Combo, float]] = {}
    for rid, bt, comb, val in zip(p["race_id"], p["bet_type"], p["combination"],
                                  p["payout_per_eur"], strict=True):
        parts = str(comb).split("-")
        if not all(x.isdigit() for x in parts) or len(parts) < COMBO_BETS[bt][0]:
            continue  # rapport de remboursement / non-partant
        if bt in ("COUPLE_GAGNANT", "TRIO"):
            combos = [tuple(sorted(int(x) for x in parts))]
        else:  # couplé placé, 2 sur 4 : une ligne par paire payée
            combos = [tuple(sorted(int(x) for x in parts[:2]))]
        for c in combos:
            out.setdefault((int(rid), bt), {})[c] = float(val)
    return out


@dataclass
class ComboCandidates:
    """Pour chaque (course, pari) : combinaisons et probabilités modèle / marché."""
    table: pd.DataFrame  # race_id, race_date, bet_type, combo, p_model, p_market, payout


def build_candidates(pred: pd.DataFrame, payouts: pd.DataFrame, model_col: str,
                     market_col: str, seed: int = 0) -> ComboCandidates:
    wins = winning_combos(payouts)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for key, g in pred.groupby("race_id", sort=False):
        race_id = int(key)  # type: ignore[arg-type]
        bets = [bt for bt in COMBO_BETS if (race_id, bt) in wins]
        if not bets or g[[model_col, market_col]].isna().any().any():
            continue
        nums = g["num_pmu"].to_numpy()
        for bt in bets:
            k, m = COMBO_BETS[bt]
            pm = combo_probabilities(nums, g[model_col].to_numpy(), k, m, rng)
            pk = combo_probabilities(nums, g[market_col].to_numpy(), k, m, rng)
            paid = wins[(race_id, bt)]
            for c in set(pm) | set(pk):
                rows.append({"race_id": race_id, "race_date": g["race_date"].iloc[0],
                             "bet_type": bt, "combo": c, "p_model": pm.get(c, 0.0),
                             "p_market": pk.get(c, 0.0), "payout": paid.get(c, 0.0)})
    return ComboCandidates(pd.DataFrame(rows))


@dataclass(frozen=True)
class ComboStrategy:
    name: str
    bet_type: str
    rank_by: str              # model | market | ratio
    min_ratio: float = 0.0    # p_modèle / p_marché minimal (rank_by = ratio)
    min_prob: float = 0.0     # probabilité modèle minimale


@dataclass
class ComboResult:
    strategy: dict[str, Any]
    n_races: int
    n_bets: int
    hit_rate: float
    roi: float
    roi_ci: tuple[float, float]


def simulate_combos(cand: pd.DataFrame, s: ComboStrategy, n_boot: int = 2000,
                    seed: int = 0) -> ComboResult:
    """Au plus une combinaison par course, mise de 1 €, réglée au rapport officiel."""
    c = cand[cand["bet_type"] == s.bet_type]
    n_races = int(c["race_id"].nunique())
    c = c[c["p_model"] >= s.min_prob]
    if s.rank_by == "ratio":
        c = c.assign(score=c["p_model"] / c["p_market"].clip(lower=1e-4))
        c = c[c["score"] >= s.min_ratio]
    else:
        c = c.assign(score=c["p_model"] if s.rank_by == "model" else c["p_market"])
    bets = c.sort_values("score", ascending=False).drop_duplicates("race_id")
    ret = bets["payout"].to_numpy()
    profit = ret - 1.0
    if len(ret) >= 2:
        rng = np.random.default_rng(seed)
        boots = profit[rng.integers(0, len(ret), size=(n_boot, len(ret)))].mean(axis=1)
        ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    else:
        ci = (float("nan"), float("nan"))
    return ComboResult(asdict(s), n_races, len(ret),
                       float((ret > 0).mean()) if len(ret) else float("nan"),
                       float(profit.mean()) if len(ret) else float("nan"), ci)


def reference_strategies(bet_type: str) -> list[ComboStrategy]:
    return [ComboStrategy(f"{bet_type} favori du marché", bet_type, "market"),
            ComboStrategy(f"{bet_type} favori du modèle", bet_type, "model")]


def value_grid(bet_type: str) -> list[ComboStrategy]:
    return [ComboStrategy(f"{bet_type} valeur ×{r:g} p≥{mp:.2f}", bet_type, "ratio", r, mp)
            for r, mp in product((1.1, 1.25, 1.5, 2.0), (0.01, 0.03, 0.08))]


@dataclass
class ComboSelection:
    bet_type: str
    selection_period: tuple[str, str]
    test_period: tuple[str, str]
    references_test: list[dict[str, Any]]
    chosen_on_selection: dict[str, Any] | None
    test: dict[str, Any] | None
    message: str


def select_then_test_combos(cand: pd.DataFrame, bet_type: str,
                            selection_share: float = 0.5) -> ComboSelection:
    c = cand[cand["bet_type"] == bet_type]
    dates = sorted(c["race_date"].unique())
    cut = dates[int(len(dates) * selection_share)]
    sel, test = c[c["race_date"] < cut], c[c["race_date"] >= cut]
    periods = ((str(dates[0]), str(max(sel["race_date"]))), (str(cut), str(dates[-1])))
    refs = [asdict(simulate_combos(test, s)) for s in reference_strategies(bet_type)]
    tried = [(s, simulate_combos(sel, s)) for s in value_grid(bet_type)]
    eligible = [(s, r) for s, r in tried if r.n_bets >= MIN_BETS_SELECTION]
    if not eligible:
        return ComboSelection(bet_type, *periods, refs, None, None,
                              "Aucune règle n'a produit assez de paris en sélection.")
    best_s, best_r = max(eligible, key=lambda t: t[1].roi)
    if not best_r.roi > 0:
        return ComboSelection(bet_type, *periods, refs, asdict(best_r), None,
                              "Aucune règle n'est rentable sur la période de sélection.")
    test_r = simulate_combos(test, best_s)
    msg = ("Rentable sur le test, à confirmer (IC du ROI)." if test_r.roi > 0 else
           "La règle retenue en sélection n'est pas rentable sur le test.")
    return ComboSelection(bet_type, *periods, refs, asdict(best_r), asdict(test_r), msg)
