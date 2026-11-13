"""Validation chronologique glissante (walk-forward).

Pour chaque pli : apprentissage sur toutes les dates antérieures au pli (fenêtre
croissante), test sur les `test_days` dates suivantes. Dans l'apprentissage, les
`calib_days` dernières dates servent à la calibration/combinaison (jamais au modèle
de base). Aucune donnée du pli de test n'intervient dans l'apprentissage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from horse_intelligence.models.ranking import FloatArray

ModelRunner = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], FloatArray]
"""(base, calib, test) -> probabilités sur test."""


@dataclass(frozen=True)
class Fold:
    train_dates: list[date]
    calib_dates: list[date]
    test_dates: list[date]


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    predictions: pd.DataFrame  # race_id, num_pmu, fold, et une colonne par modèle
    errors: dict[str, list[str]] = field(default_factory=dict)


def make_folds(dates: list[date], min_train_days: int, test_days: int,
               calib_days: int) -> list[Fold]:
    dates = sorted(set(dates))
    folds = []
    start = min_train_days
    while start < len(dates):
        train = dates[:start]
        test = dates[start:start + test_days]
        folds.append(Fold(train_dates=train[:-calib_days], calib_dates=train[-calib_days:],
                          test_dates=test))
        start += test_days
    return folds


def run_walk_forward(frame: pd.DataFrame, models: dict[str, ModelRunner],
                     min_train_days: int = 21, test_days: int = 7,
                     calib_days: int = 7) -> WalkForwardResult:
    labelled = frame[frame["won"].notna()]
    folds = make_folds(list(labelled["race_date"].unique()), min_train_days, test_days,
                       calib_days)
    parts = []
    errors: dict[str, list[str]] = {}
    for i, fold in enumerate(folds):
        base = labelled[labelled["race_date"].isin(fold.train_dates)]
        calib = labelled[labelled["race_date"].isin(fold.calib_dates)]
        test = labelled[labelled["race_date"].isin(fold.test_dates)]
        if not (base["race_date"].max() < calib["race_date"].min()
                and calib["race_date"].max() < test["race_date"].min()):
            raise RuntimeError("chevauchement temporel interdit entre les jeux du pli")
        out = test[["race_id", "race_date", "num_pmu"]].copy()
        out["fold"] = i
        for name, runner in models.items():
            try:
                out[name] = runner(base, calib, test)
            except Exception as exc:  # un modèle en échec n'invalide pas les autres
                errors.setdefault(name, []).append(f"pli {i}: {exc}")
                out[name] = np.nan
        parts.append(out)
    preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return WalkForwardResult(folds, preds, errors)
