"""Profils de partants atypiques (Isolation Forest) et test de leur lien avec le marché.

L'atypicité seule n'est pas un signal : on vérifie hors échantillon si les partants
les plus atypiques gagnent plus (ou moins) souvent que ne l'anticipent les cotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from horse_intelligence.intelligence.pattern_discovery import ae_interval, subgroup_stat

TOP_SHARE = 0.05


@dataclass(frozen=True)
class AnomalyTest:
    n: int
    ae: float
    ae_low: float
    ae_high: float
    p_value: float
    message: str


def _matrix(frame: pd.DataFrame, features: list[str], center: pd.Series,
            scale: pd.Series) -> np.ndarray:
    X = frame[features].apply(pd.to_numeric, errors="coerce")
    Z = ((X - center) / scale).fillna(0.0)
    return np.asarray(Z.to_numpy(dtype=np.float64))


def fit_scores(train: pd.DataFrame, score_on: pd.DataFrame, features: list[str],
               seed: int = 0) -> np.ndarray:
    """Score d'anomalie (plus haut = plus atypique), modèle appris sur `train` seulement."""
    from sklearn.ensemble import IsolationForest

    X = train[features].apply(pd.to_numeric, errors="coerce")
    center = X.median()
    scale = (X.quantile(0.75) - X.quantile(0.25)).replace(0, 1.0).fillna(1.0)
    forest = IsolationForest(n_estimators=200, random_state=seed)
    forest.fit(_matrix(train, features, center, scale))
    return np.asarray(-forest.score_samples(_matrix(score_on, features, center, scale)))


def test_top_anomalies(frame: pd.DataFrame, scores: np.ndarray) -> AnomalyTest:
    f = frame.assign(_s=scores).dropna(subset=["mkt_prob", "won"])
    top = f[f["_s"] >= f["_s"].quantile(1 - TOP_SHARE)]
    if len(top) < 50:
        return AnomalyTest(len(top), float("nan"), float("nan"), float("nan"), float("nan"),
                           "Échantillon insuffisant.")
    st = subgroup_stat(top["won"].to_numpy(), top["mkt_prob"].to_numpy())
    lo, hi = ae_interval(st.actual, st.expected)
    msg = ("Les profils atypiques s'écartent des cotes (p < 0,05)." if st.p_value < 0.05
           else "Les profils atypiques ne s'écartent pas significativement des cotes.")
    return AnomalyTest(st.n, st.ae, lo, hi, st.p_value, msg)
