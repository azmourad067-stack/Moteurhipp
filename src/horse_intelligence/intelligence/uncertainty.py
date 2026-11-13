"""Incertitude des estimations et détection des situations hors distribution.

- Ensemble bootstrap : K modèles appris sur des rééchantillonnages de COURSES
  (avec remise). La dispersion de leurs estimations mesure l'incertitude d'estimation
  (et non l'aléa de la course elle-même).
- Score hors distribution : écarts robustes (médiane / écart interquartile) des
  variables d'un partant par rapport à l'apprentissage.
- Statut de course : seuils appris sur la distribution d'apprentissage (quantiles),
  jamais fixés arbitrairement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.models.lightgbm_model import LgbmRaceModel
from horse_intelligence.models.ranking import FloatArray

N_MEMBERS = 8
INTERVAL = (0.10, 0.90)
TOP_K_DEVIATIONS = 3


def resample_races(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    groups = frame.groupby("race_id", sort=False).indices
    keys = list(groups)
    picked = rng.integers(0, len(keys), size=len(keys))
    rows = np.concatenate([groups[keys[i]] for i in picked])
    out = frame.iloc[rows].copy()
    # chaque tirage devient une course distincte (une course tirée deux fois = deux courses)
    out["race_id"] = np.repeat(np.arange(len(picked)), [len(groups[keys[i]]) for i in picked])
    return out


def fit_members(base: pd.DataFrame, calib: pd.DataFrame, features: list[str],
                categorical: list[str], with_market: bool, n_members: int = N_MEMBERS,
                seed: int = 0) -> list[LgbmRaceModel]:
    rng = np.random.default_rng(seed)
    members = []
    for k in range(n_members):
        m = LgbmRaceModel(features, categorical, with_market=with_market)
        m.params = {**m.params, "seed": seed + k + 1}
        members.append(m.fit(resample_races(base, rng), resample_races(calib, rng)))
    return members


def member_intervals(members: list[LgbmRaceModel], frame: pd.DataFrame
                     ) -> tuple[FloatArray, FloatArray]:
    preds = np.vstack([m.predict(frame) for m in members])
    lo, hi = np.quantile(preds, INTERVAL, axis=0)
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


@dataclass
class OodScorer:
    features: list[str]
    median: dict[str, float] = field(default_factory=dict)
    iqr: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame) -> OodScorer:
        for c in self.features:
            s = pd.to_numeric(frame[c], errors="coerce").dropna()
            if s.empty:
                continue
            q1, q3 = np.quantile(s, [0.25, 0.75])
            self.median[c] = float(s.median())
            self.iqr[c] = float(q3 - q1) if q3 > q1 else float(s.std() or 1.0)
        return self

    def runner_scores(self, frame: pd.DataFrame) -> FloatArray:
        cols = [c for c in self.features if c in self.median]
        z = np.column_stack([
            np.abs((pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=np.float64)
                    - self.median[c]) / self.iqr[c]) for c in cols])
        z = np.nan_to_num(z, nan=0.0)
        top = -np.sort(-z, axis=1)[:, :TOP_K_DEVIATIONS]
        return np.asarray(top.mean(axis=1), dtype=np.float64)

    def top_deviations(self, row: pd.Series, k: int = 3) -> list[tuple[str, float]]:
        devs = []
        for c in self.features:
            if c in self.median and pd.notna(row.get(c)):
                devs.append((c, float((row[c] - self.median[c]) / self.iqr[c])))
        return sorted(devs, key=lambda t: -abs(t[1]))[:k]

    def to_dict(self) -> dict[str, Any]:
        return {"features": self.features, "median": self.median, "iqr": self.iqr}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OodScorer:
        return cls(d["features"], d["median"], d["iqr"])


def race_aggregate(values: FloatArray, race_ids: pd.Series, how: str = "mean") -> pd.Series:
    return pd.Series(values, index=race_ids.index).groupby(race_ids).agg(how)


@dataclass(frozen=True)
class UncertaintyThresholds:
    ood_race: float          # quantile 99 % du score moyen par course
    rel_width: float         # quantile 90 % de la largeur relative (top 3)
    disagreement: float      # quantile 90 % de l'écart modèle / marché

    def to_dict(self) -> dict[str, float]:
        return {"ood_race": self.ood_race, "rel_width": self.rel_width,
                "disagreement": self.disagreement}


def relative_width(p: FloatArray, lo: FloatArray, hi: FloatArray) -> FloatArray:
    return np.asarray((hi - lo) / np.clip(p, 1e-6, None), dtype=np.float64)


def top3_mask(p: FloatArray, race_ids: pd.Series) -> FloatArray:
    rank = pd.Series(p, index=race_ids.index).groupby(race_ids).rank(ascending=False,
                                                                      method="first")
    return np.asarray(rank.to_numpy() <= 3, dtype=np.float64)


def learn_thresholds(frame: pd.DataFrame, p: FloatArray, lo: FloatArray, hi: FloatArray,
                     ood: FloatArray) -> UncertaintyThresholds:
    rid = frame["race_id"]
    top = top3_mask(p, rid).astype(bool)
    width = race_aggregate(relative_width(p, lo, hi)[top], rid[top], "max")
    dis = race_aggregate(np.abs(p - frame["mkt_prob"].to_numpy()), rid, "max")
    return UncertaintyThresholds(
        ood_race=float(race_aggregate(ood, rid).quantile(0.99)),
        rel_width=float(width.quantile(0.90)), disagreement=float(dis.quantile(0.90)))


def race_status(p: FloatArray, lo: FloatArray, hi: FloatArray, ood: FloatArray,
                mkt: FloatArray | None, th: dict[str, float]) -> tuple[str, list[str]]:
    """Statut : `exploitable` | `incertaine` | `hors distribution`, avec les raisons."""
    reasons = []
    if float(np.mean(ood)) > th["ood_race"]:
        reasons.append("profil de course inhabituel par rapport à l'historique d'apprentissage")
    order = np.argsort(-p)[:3]
    width = float(np.max(relative_width(p[order], lo[order], hi[order])))
    if width > th["rel_width"]:
        reasons.append("estimations instables d'un rééchantillonnage à l'autre")
    if mkt is not None and float(np.max(np.abs(p - mkt))) > th["disagreement"]:
        reasons.append("fort désaccord entre le modèle et le marché")
    if reasons and reasons[0].startswith("profil"):
        return "hors distribution", reasons
    return ("incertaine" if reasons else "exploitable"), reasons
