"""Détection de dérive : changement de distribution des variables et de la qualité du marché."""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_ALERT = 0.25
PSI_WATCH = 0.10
# Variables d'historique interne : elles évoluent mécaniquement tant que la base se
# construit (compteurs croissants, lissage qui s'affaiblit). Ce n'est pas un changement
# de régime.
STRUCTURAL = {"horse_db_runs", "horse_days_since_db", "horse_db_mean_pct", "jockey_db_runs",
              "trainer_db_runs", "pair_db_runs", "jockey_win_rate", "trainer_win_rate",
              "rider_change"}


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index sur des classes de quantiles de la référence."""
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if len(ref) < 50 or len(cur) < 50:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "psi": psi(reference[f], current[f])} for f in features]
    t = pd.DataFrame(rows).dropna()
    t["niveau"] = np.where(t["psi"] >= PSI_ALERT, "dérive forte",
                           np.where(t["psi"] >= PSI_WATCH, "à surveiller", "stable"))
    structural = t["feature"].isin(STRUCTURAL) & (t["niveau"] != "stable")
    t.loc[structural, "niveau"] = "structurelle (historique en construction)"
    return t.sort_values("psi", ascending=False).reset_index(drop=True)


def weekly_market_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Log-loss hebdomadaire du marché : une rupture signale un changement de régime."""
    f = frame.dropna(subset=["mkt_prob", "won"])
    w = f[f["won"] == 1].copy()
    w["week"] = pd.to_datetime(w["race_date"]).dt.to_period("W").dt.start_time
    t = w.groupby("week").agg(races=("race_id", "nunique"),
                              log_loss=("mkt_prob", lambda p: float(-np.log(p).mean())))
    return t.reset_index()
