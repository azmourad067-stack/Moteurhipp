"""Suivi des prédictions enregistrées avant départ, une fois les résultats connus."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from horse_intelligence.evaluation.metrics import compute_metrics, paired_bootstrap


def settled_predictions(engine: Engine) -> pd.DataFrame:
    """Dernière prédiction antérieure au départ pour chaque course terminée."""
    sql = text("""
        SELECT p.prediction_id, p.race_id, p.decision_time, p.model_id, r.race_date,
               r.scheduled_start_utc, e.num_pmu, e.p_win, e.mkt_prob, e.mkt_odds,
               res.finish_position
        FROM predictions p
        JOIN races r ON r.race_id = p.race_id
        JOIN prediction_entries e ON e.prediction_id = p.prediction_id
        LEFT JOIN race_results res ON res.race_id = p.race_id AND res.num_pmu = e.num_pmu
        WHERE p.decision_time < r.scheduled_start_utc AND r.result_final = :final""")
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"final": True})
    if df.empty:
        return df
    last = df.groupby("race_id")["prediction_id"].transform("max")
    df = df[df["prediction_id"] == last].copy()
    has_result = df.groupby("race_id")["finish_position"].transform(lambda s: s.notna().any())
    df = df[has_result]
    df["won"] = (df["finish_position"] == 1).astype(float)
    # marché renormalisé sur les partants prédits (non-partants de dernière minute exclus)
    df["mkt_prob"] = df["mkt_prob"] / df.groupby("race_id")["mkt_prob"].transform("sum")
    df["n_runners"] = df.groupby("race_id")["num_pmu"].transform("size")
    return df


def live_tracking(engine: Engine) -> dict[str, Any]:
    df = settled_predictions(engine)
    if df.empty:
        return {"races": 0, "message": "Aucune prédiction enregistrée n'a encore été réglée."}
    model = compute_metrics(df, df["p_win"])
    out: dict[str, Any] = {"races": model.n_races, "model": model.as_dict(),
                           "first": str(df["race_date"].min()),
                           "last": str(df["race_date"].max())}
    mkt = df[df.groupby("race_id")["mkt_prob"].transform(lambda s: s.notna().all())]
    if not mkt.empty:
        out["market"] = compute_metrics(mkt, mkt["mkt_prob"]).as_dict()
        out["vs_market"] = paired_bootstrap(mkt, mkt["p_win"], mkt["mkt_prob"])
    out["message"] = (f"{model.n_races} courses prédites avant le départ puis réglées."
                      if model.n_races >= 30 else
                      f"{model.n_races} courses réglées : échantillon encore trop petit.")
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in out.items()}
