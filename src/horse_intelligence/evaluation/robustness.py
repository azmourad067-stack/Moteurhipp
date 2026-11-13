"""Robustesse : résultats par segment, stabilité temporelle, données manquantes, aléa."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation.metrics import compute_metrics, paired_bootstrap

MIN_SEGMENT_RACES = 30
DISTANCE_BANDS = ([0, 1600, 2100, 2700, 99999], ["≤ 1600 m", "1601-2100 m", "2101-2700 m",
                                                 "> 2700 m"])
FIELD_BANDS = ([0, 8, 12, 99], ["≤ 8 partants", "9-12 partants", "≥ 13 partants"])


def _segment_columns(pred: pd.DataFrame) -> pd.DataFrame:
    f = pred.copy()
    f["discipline_seg"] = f["discipline"].astype(object).fillna("inconnue")
    f["distance_seg"] = pd.cut(f["distance_m"], DISTANCE_BANDS[0],
                               labels=DISTANCE_BANDS[1]).astype(object)
    f["field_seg"] = pd.cut(f["n_runners"], FIELD_BANDS[0], labels=FIELD_BANDS[1]).astype(object)
    top = f.groupby("racecourse", observed=True)["race_id"].nunique().nlargest(8).index
    f["racecourse_seg"] = f["racecourse"].astype(object).where(
        f["racecourse"].isin(top), "autres hippodromes")
    f["period_seg"] = "pli " + f["fold"].astype(int).astype(str)
    return f


SEGMENTS = {"discipline_seg": "Discipline", "racecourse_seg": "Hippodrome",
            "distance_seg": "Distance", "field_seg": "Nombre de partants",
            "period_seg": "Période"}


def segment_table(pred: pd.DataFrame, model: str, reference: str) -> list[dict[str, Any]]:
    f = _segment_columns(pred.dropna(subset=[model, reference]))
    rows = []
    for col, title in SEGMENTS.items():
        for value, g in f.groupby(col):
            n = g["race_id"].nunique()
            if n < MIN_SEGMENT_RACES:
                continue
            m, r = compute_metrics(g, g[model]), compute_metrics(g, g[reference])
            d = paired_bootstrap(g, g[model], g[reference], n_boot=1000)
            rows.append({"dimension": title, "segment": str(value), "races": n,
                         "log_loss_model": m.log_loss, "log_loss_reference": r.log_loss,
                         "diff": d["mean_diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                         "top1_model": m.top1, "top1_reference": r.top1})
    return rows


def stability_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Le modèle est-il régulièrement meilleur, ou seulement sur quelques segments ?"""
    periods = [r for r in rows if r["dimension"] == "Période"]
    better = sum(r["diff"] < 0 for r in periods)
    return {"periods": len(periods), "periods_better": better,
            "message": (f"Meilleur que la référence sur {better}/{len(periods)} périodes."
                        if periods else "Pas assez de périodes.")}


def mask_features(frame: pd.DataFrame, features: list[str], share: float,
                  seed: int = 0) -> pd.DataFrame:
    """Rend manquante une part des valeurs numériques (robustesse aux données absentes)."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for c in features:
        if pd.api.types.is_numeric_dtype(out[c]):
            out.loc[rng.random(len(out)) < share, c] = np.nan
    return out
