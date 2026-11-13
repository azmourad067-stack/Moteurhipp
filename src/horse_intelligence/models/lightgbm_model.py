"""LightGBM + calibration par course, avec ou sans combinaison au marché.

Schéma (sans fuite) :
1. LightGBM (objectif binaire « gagne ») appris sur la partie ancienne de l'apprentissage ;
2. son score logit est converti en probabilités par course via un logit conditionnel
   appris sur la partie récente de l'apprentissage (calibration de température) ;
3. variante « + marché » : le logit conditionnel combine le score et log(p_marché)
   (approche de type Benter), ce qui mesure l'information AJOUTÉE au marché.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.models.ranking import ConditionalLogit, FloatArray

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 40,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
    "verbose": -1, "seed": 42, "deterministic": True, "num_threads": 2,
}
NUM_ROUNDS = 300


def _logit(p: FloatArray) -> FloatArray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(p / (1 - p)), dtype=np.float64)


@dataclass
class LgbmRaceModel:
    features: list[str]
    categorical: list[str]
    with_market: bool = False
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_rounds: int = NUM_ROUNDS
    booster: Any = None
    calibrator: ConditionalLogit | None = None
    categories: dict[str, list[str]] = field(default_factory=dict)

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[self.features].copy()
        for c in self.categorical:
            # catégories figées à l'apprentissage : une modalité inconnue devient NaN
            X[c] = pd.Categorical(X[c].astype("object"), categories=self.categories[c])
        return X

    def raw_score(self, frame: pd.DataFrame) -> FloatArray:
        p = self.booster.predict(self._matrix(frame))
        return _logit(np.asarray(p, dtype=np.float64))

    def _calib_inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = {"lgbm_score": self.raw_score(frame)}
        if self.with_market:
            cols["mkt_log_prob"] = frame["mkt_log_prob"].to_numpy()
        return pd.DataFrame(cols, index=frame.index)

    def fit(self, base: pd.DataFrame, calib: pd.DataFrame) -> LgbmRaceModel:
        import lightgbm as lgb  # dépendance chargée seulement si nécessaire

        self.categories = {c: sorted(map(str, base[c].dropna().astype("object").unique()))
                           for c in self.categorical}
        data = lgb.Dataset(self._matrix(base), label=base["won"].to_numpy(),
                           categorical_feature=self.categorical, free_raw_data=False)
        self.booster = lgb.train(self.params, data, num_boost_round=self.num_rounds)
        Z = self._calib_inputs(calib)
        self.calibrator = ConditionalLogit(l2=1e-3).fit(Z, calib["race_id"], calib["won"],
                                                        list(Z.columns))
        return self

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        if self.calibrator is None:
            raise RuntimeError("modèle non entraîné")
        return self.calibrator.predict_proba(self._calib_inputs(frame), frame["race_id"])

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.booster.feature_importance("gain"),
                         index=self.booster.feature_name()).sort_values(ascending=False)

    # --- persistance (texte LightGBM + JSON, jamais pickle) -----------------
    def to_dict(self) -> dict[str, Any]:
        if self.calibrator is None:
            raise RuntimeError("modèle non entraîné")
        return {"features": self.features, "categorical": self.categorical,
                "with_market": self.with_market, "params": self.params,
                "num_rounds": self.num_rounds, "categories": self.categories,
                "calibrator": self.calibrator.to_dict(),
                "booster": self.booster.model_to_string()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LgbmRaceModel:
        import lightgbm as lgb

        m = cls(features=d["features"], categorical=d["categorical"],
                with_market=d["with_market"], params=d["params"], num_rounds=d["num_rounds"])
        m.categories = d["categories"]
        m.calibrator = ConditionalLogit.from_dict(d["calibrator"])
        m.booster = lgb.Booster(model_str=d["booster"])
        return m
