"""Modèles de référence : uniforme, marché brut, marché calibré, logit conditionnel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_intelligence.models.ranking import (
    ConditionalLogit,
    FloatArray,
    group_codes,
    normalize_within_race,
)


def uniform_proba(frame: pd.DataFrame) -> FloatArray:
    codes, n = group_codes(frame["race_id"])
    return normalize_within_race(np.ones(len(frame)), codes, n)


def market_proba(frame: pd.DataFrame) -> FloatArray:
    """Probabilités implicites des cotes de référence, marge retirée (proportionnel)."""
    return np.asarray(frame["mkt_prob"], dtype=np.float64)


class CalibratedMarket:
    """Marché corrigé du biais favori/outsider : p ∝ q^α, α appris (logit conditionnel)."""

    def __init__(self) -> None:
        self.model = ConditionalLogit(l2=1e-6)

    def fit(self, frame: pd.DataFrame) -> CalibratedMarket:
        self.model.fit(frame[["mkt_log_prob"]], frame["race_id"], frame["won"],
                       ["mkt_log_prob"])
        return self

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        return self.model.predict_proba(frame[["mkt_log_prob"]], frame["race_id"])


class FormLogit:
    """Logit conditionnel sur variables interprétables (sans marché)."""

    def __init__(self, features: list[str], l2: float = 5.0) -> None:
        self.features = features
        self.model = ConditionalLogit(l2=l2)

    def fit(self, frame: pd.DataFrame) -> FormLogit:
        self.model.fit(frame[self.features], frame["race_id"], frame["won"], self.features)
        return self

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        return self.model.predict_proba(frame[self.features], frame["race_id"])
