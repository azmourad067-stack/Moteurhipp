"""Découverte automatique de sous-groupes où le marché se trompe systématiquement.

Méthode :
1. variables discrétisées en conditions lisibles (ex. « changement de monte »,
   « 2e-3e favori », « distance 2100-2700 m ») ; conditions simples et paires ;
2. période de DÉCOUVERTE : pour chaque sous-groupe, victoires observées A contre
   victoires attendues par le marché E = Σ p_marché ; z = (A−E)/√Σp(1−p) ;
3. correction des tests multiples (Benjamini-Hochberg) sur TOUS les tests réalisés ;
4. période de CONFIRMATION, postérieure et disjointe : le même sous-groupe est
   re-mesuré ; seules les hypothèses confirmées sont dites « validées hors échantillon ».

Un sous-groupe validé est une association statistique, pas une cause, et pas une
stratégie rentable (prélèvement du pari mutuel, cote finale différente).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

MIN_RUNNERS = 80
MIN_EXPECTED = 8.0
FDR_LEVEL = 0.10
CONFIRM_ALPHA = 0.05

STATUS_DETECTED = "hypothèse détectée"
STATUS_PENDING = "en cours de validation"
STATUS_VALIDATED = "validée hors échantillon"
STATUS_NOT_CONFIRMED = "non confirmée"
STATUS_INVALIDATED = "invalidée"


def _bands(s: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(s, bins=[-np.inf, *edges, np.inf], labels=labels, right=True).astype(object)


CONDITIONS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "discipline": lambda f: f["discipline"].astype(object),
    "distance": lambda f: _bands(f["distance_m"], [1600, 2100, 2700],
                                 ["≤ 1600 m", "1601-2100 m", "2101-2700 m", "> 2700 m"]),
    "partants": lambda f: _bands(f["n_runners"], [8, 12], ["≤ 8 partants", "9-12 partants",
                                                            "≥ 13 partants"]),
    "rang_cote": lambda f: _bands(f["mkt_rank"], [1, 3, 6],
                                  ["favori", "2e-3e cote", "4e-6e cote", "7e cote et +"]),
    "derniere_place": lambda f: _bands(f["mus_last_pos"], [3, 9],
                                       ["dernière course : 1-3", "dernière course : 4-9",
                                        "dernière course : non placé/incident"]),
    "tendance": lambda f: _bands(f["mus_trend"], [-1.5, 1.5],
                                 ["forme en baisse", "forme stable", "forme en hausse"]),
    "changement_monte": lambda f: f["rider_change"].map({1.0: "changement de monte",
                                                         0.0: "même monte"}),
    "corde": lambda f: _bands(f["draw_rel"], [1 / 3, 2 / 3],
                              ["corde intérieure", "corde médiane", "corde extérieure"]),
    "age": lambda f: _bands(f["age"], [4, 7], ["≤ 4 ans", "5-7 ans", "≥ 8 ans"]),
    "ferrure": lambda f: f["shoes"].astype(object).map(
        lambda v: None if pd.isna(v) else ("déferré" if "DEFERRE" in str(v) else "ferré")),
    "repos": lambda f: _bands(f["horse_days_since_db"], [14, 30],
                              ["repos ≤ 14 j", "repos 15-30 j", "repos > 30 j"]),
}


def condition_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: fn(frame) for k, fn in CONDITIONS.items()}, index=frame.index)


@dataclass(frozen=True)
class SubgroupStat:
    n: int
    actual: float
    expected: float
    z: float
    p_value: float

    @property
    def ae(self) -> float:
        return self.actual / self.expected if self.expected > 0 else float("nan")


def subgroup_stat(won: np.ndarray, p: np.ndarray) -> SubgroupStat:
    from scipy.stats import norm

    a, e = float(won.sum()), float(p.sum())
    var = float((p * (1 - p)).sum())
    z = (a - e) / np.sqrt(var) if var > 0 else 0.0
    return SubgroupStat(len(won), a, e, float(z), float(2 * norm.sf(abs(z))))


def ae_interval(actual: float, expected: float, level: float = 0.90) -> tuple[float, float]:
    """Intervalle exact (Poisson) du ratio A/E."""
    from scipy.stats import chi2

    alpha = 1 - level
    lo = chi2.ppf(alpha / 2, 2 * actual) / 2 if actual > 0 else 0.0
    hi = chi2.ppf(1 - alpha / 2, 2 * actual + 2) / 2
    return float(lo / expected), float(hi / expected)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def _subgroups(cond: pd.DataFrame, max_order: int) -> list[tuple[tuple[str, str], ...]]:
    singles = [(c, v) for c in cond.columns for v in cond[c].dropna().unique()]
    groups: list[tuple[tuple[str, str], ...]] = [(s,) for s in singles]
    if max_order >= 2:
        groups += [(a, b) for a, b in combinations(singles, 2) if a[0] != b[0]]
    return groups


def _mask(cond: pd.DataFrame, group: tuple[tuple[str, str], ...]) -> np.ndarray:
    m = np.ones(len(cond), dtype=bool)
    for col, val in group:
        m &= (cond[col] == val).to_numpy()
    return m


def label(group: tuple[tuple[str, str], ...]) -> str:
    return " + ".join(str(v) for _, v in group)


def discover(discovery: pd.DataFrame, confirmation: pd.DataFrame, max_order: int = 2
             ) -> tuple[pd.DataFrame, int]:
    """Renvoie les hypothèses retenues en découverte (q < FDR) et leur confirmation,
    ainsi que le nombre total de tests réalisés (à publier avec les résultats)."""
    disc = discovery.dropna(subset=["mkt_prob", "won"])
    conf = confirmation.dropna(subset=["mkt_prob", "won"])
    cd, cc = condition_frame(disc), condition_frame(conf)
    wd, pd_ = disc["won"].to_numpy(), disc["mkt_prob"].to_numpy()
    wc, pc = conf["won"].to_numpy(), conf["mkt_prob"].to_numpy()

    rows = []
    for group in _subgroups(cd, max_order):
        m = _mask(cd, group)
        if m.sum() < MIN_RUNNERS or pd_[m].sum() < MIN_EXPECTED:
            continue
        st = subgroup_stat(wd[m], pd_[m])
        rows.append({"group": group, "label": label(group), "order": len(group),
                     "n": st.n, "actual": st.actual, "expected": st.expected, "ae": st.ae,
                     "z": st.z, "p_value": st.p_value})
    n_tests = len(rows)
    if not rows:
        return pd.DataFrame(), 0
    res = pd.DataFrame(rows)
    res["q_value"] = benjamini_hochberg(res["p_value"].to_numpy())
    res = res[res["q_value"] < FDR_LEVEL].copy()

    out = []
    for r in res.to_dict("records"):
        r["ae_low"], r["ae_high"] = ae_interval(r["actual"], r["expected"])
        if r["order"] == 2:  # interaction : écart au produit des effets simples
            singles = [subgroup_stat(wd[_mask(cd, (g,))], pd_[_mask(cd, (g,))]).ae
                       for g in r["group"]]
            r["interaction_lift"] = r["ae"] / (singles[0] * singles[1])
        m2 = _mask(cc, r["group"]) if len(cc) else np.zeros(0, dtype=bool)
        if m2.sum() < MIN_RUNNERS / 2 or pc[m2].sum() < MIN_EXPECTED / 2:
            r.update(status=STATUS_PENDING, oos_n=int(m2.sum()))
        else:
            st2 = subgroup_stat(wc[m2], pc[m2])
            same = np.sign(st2.z) == np.sign(r["z"])
            one_sided = st2.p_value / 2 if same else 1 - st2.p_value / 2
            lo2, hi2 = ae_interval(st2.actual, st2.expected)
            r.update(oos_n=st2.n, oos_ae=st2.ae, oos_ae_low=lo2, oos_ae_high=hi2,
                     oos_p=one_sided,
                     status=STATUS_VALIDATED if same and one_sided < CONFIRM_ALPHA else
                     STATUS_INVALIDATED if not same and st2.p_value < CONFIRM_ALPHA else
                     STATUS_NOT_CONFIRMED)
        out.append(r)
    if not out:  # aucun sous-groupe ne survit à la correction des tests multiples
        return pd.DataFrame(), n_tests
    table = pd.DataFrame(out).sort_values("q_value").reset_index(drop=True)
    return table, n_tests


def weekly_stability(frame: pd.DataFrame, group: tuple[tuple[str, str], ...],
                     direction: float) -> float:
    """Part des semaines où le sous-groupe va dans le sens de l'effet détecté."""
    f = frame.dropna(subset=["mkt_prob", "won"])
    m = _mask(condition_frame(f), group)
    sub = f[m]
    if sub.empty:
        return float("nan")
    week = pd.to_datetime(sub["race_date"]).dt.isocalendar().week
    res = (sub["won"] - sub["mkt_prob"]).groupby(week).sum()
    return float((np.sign(res) == np.sign(direction)).mean())
