"""Prédiction d'une course : probabilités, places, incertitude simple et comparaison au marché.

Aucune prédiction n'est produite si les données sont insuffisantes (contrôle qualité
bloquant, partants absents) ou si aucun modèle n'est disponible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, select

from horse_intelligence.data.processors.validation import DataStatus
from horse_intelligence.data.storage import database as db
from horse_intelligence.intelligence.explainability import (
    relative_contributions,
    strengths_weaknesses,
)
from horse_intelligence.intelligence.feature_engineering import build_dataset, load_raw
from horse_intelligence.intelligence.pattern_discovery import condition_frame
from horse_intelligence.intelligence.uncertainty import OodScorer, member_intervals, race_status
from horse_intelligence.models.lightgbm_model import LgbmRaceModel
from horse_intelligence.models.ranking import ConditionalLogit, paid_places, place_probabilities
from horse_intelligence.models.registry import latest_model, load_artifact
from horse_intelligence.services.discovery import validated_patterns

PRIMARY_ORDER = ("lgbm_market", "lgbm_form")
MIN_EDGE_UNCERTAINTY = 0.02


@dataclass
class RacePrediction:
    race_id: int
    status: str                       # verified | partial | insufficient | unavailable
    message: str
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    primary: str | None = None
    n_places: int = 0
    has_market: bool = False
    in_sample: bool = False           # course incluse dans l'apprentissage du modèle
    uncertainty: str = "non évaluée"  # exploitable | incertaine | hors distribution
    uncertainty_reasons: list[str] = field(default_factory=list)


def _quality_status(engine: Engine, race_id: int) -> tuple[str, list[dict[str, Any]]]:
    r, q = db.races, db.data_quality_reports
    with engine.connect() as conn:
        race = conn.execute(select(r.c.race_date, r.c.reunion_num, r.c.course_num)
                            .where(r.c.race_id == race_id)).first()
        if race is None:
            return DataStatus.UNAVAILABLE.value, []
        key = f"{race.race_date.isoformat()}-R{race.reunion_num}C{race.course_num}"
        row = conn.execute(select(q.c.status, q.c.checks_json)
                           .where(q.c.scope == "race", q.c.scope_ref == key)
                           .order_by(q.c.report_id.desc()).limit(1)).first()
    if row is None:
        return DataStatus.UNAVAILABLE.value, []
    return str(row.status), list(row.checks_json or [])


@dataclass
class LoadedFamily:
    """Modèle chargé et désérialisé une fois (réutilisable entre prédictions)."""

    meta: dict[str, Any]
    payload: dict[str, Any]
    lgbm: LgbmRaceModel | None = None
    members: list[LgbmRaceModel] = field(default_factory=list)
    calibrator: ConditionalLogit | None = None


def load_models(engine: Engine) -> dict[str, LoadedFamily]:
    out: dict[str, LoadedFamily] = {}
    for family in (*PRIMARY_ORDER, "market_calibrated"):
        meta = latest_model(engine, family)
        if meta is None:
            continue
        payload = load_artifact(engine, meta["artifact_uri"], meta["artifact_sha256"])
        if family == "market_calibrated":
            out[family] = LoadedFamily(meta, payload,
                                       calibrator=ConditionalLogit.from_dict(payload["calibrator"]))
        else:
            out[family] = LoadedFamily(
                meta, payload, lgbm=LgbmRaceModel.from_dict(payload),
                members=[LgbmRaceModel.from_dict(d) for d in payload.get("members", [])])
    return out


def models_use_source_aggregates(models: dict[str, LoadedFamily]) -> bool:
    return all(bool(f.payload.get("use_source_aggregates", True)) for f in models.values())


def build_frame(engine: Engine, until: date, use_source_aggregates: bool = True
                ) -> pd.DataFrame:
    return build_dataset(load_raw(engine, until=until),
                         use_source_aggregates=use_source_aggregates).frame


def matched_patterns(race: pd.DataFrame, validated: list[dict[str, Any]]) -> list[str]:
    """Libellés des patterns VALIDÉS hors échantillon qui s'appliquent à chaque partant."""
    if not validated:
        return [""] * len(race)
    cond = condition_frame(race)
    out = []
    for i in range(len(race)):
        hits = [f"{p['definition_json']['label']} (A/E {p['effect_size']:.2f})"
                for p in validated
                if all(cond.iloc[i][c] == v for c, v in p["definition_json"]["group"])]
        out.append(" ; ".join(hits))
    return out


def predict_race(engine: Engine, race_id: int, frame: pd.DataFrame | None = None,
                 models: dict[str, LoadedFamily] | None = None) -> RacePrediction:
    """`frame` et `models` peuvent être fournis déjà construits (cache de l'interface)."""
    status, _checks = _quality_status(engine, race_id)
    if status in (DataStatus.INSUFFICIENT.value, DataStatus.UNAVAILABLE.value):
        return RacePrediction(race_id, status, "Données insuffisantes : aucune prédiction.")
    with engine.connect() as conn:
        race_date = conn.execute(select(db.races.c.race_date)
                                 .where(db.races.c.race_id == race_id)).scalar_one()
    try:
        models = models if models is not None else load_models(engine)
    except (OSError, ValueError) as exc:
        return RacePrediction(race_id, DataStatus.UNAVAILABLE.value,
                              f"Échec du chargement des modèles : {exc}")
    if not any(f in models for f in PRIMARY_ORDER):
        return RacePrediction(race_id, DataStatus.UNAVAILABLE.value,
                              "Aucun modèle entraîné : lancez scripts/train_models.py.")

    validated = validated_patterns(engine)
    if frame is None:
        frame = build_frame(engine, race_date, models_use_source_aggregates(models))
    race = frame[frame["race_id"] == race_id].copy()
    if len(race) < 2:
        return RacePrediction(race_id, DataStatus.INSUFFICIENT.value,
                              "Moins de deux partants identifiés : aucune prédiction.")
    has_market = bool(race["mkt_prob"].notna().all())

    probs: dict[str, np.ndarray] = {}
    lgbm: dict[str, LgbmRaceModel] = {}
    meta_out: dict[str, dict[str, Any]] = {}
    for family, loaded in models.items():
        meta = loaded.meta
        meta_out[family] = {"version": meta["version"], "status": meta["status"],
                            "train_end": meta["train_end"], "model_id": meta["model_id"]}
        if loaded.calibrator is not None:
            if has_market:
                probs[family] = loaded.calibrator.predict_proba(race[["mkt_log_prob"]],
                                                                race["race_id"])
            continue
        m = loaded.lgbm
        if m is None or (m.with_market and not has_market):
            continue
        lgbm[family] = m
        probs[family] = m.predict(race)
    primary = next((f for f in PRIMARY_ORDER if f in probs), None)
    if primary is None:
        return RacePrediction(race_id, DataStatus.UNAVAILABLE.value,
                              "Aucun modèle applicable à cette course.")

    p = probs[primary]
    declared = race["declared_runners"].iloc[0]
    n_places = paid_places(int(declared) if pd.notna(declared) else len(race))
    table = race[["num_pmu", "horse_id", "mkt_odds", "mkt_prob", "mkt_pool",
                  "form_string", "age", "sex"]].copy()
    table["p_win"] = p
    table["rank"] = pd.Series(p, index=race.index).rank(ascending=False, method="min")
    table["p_place"] = place_probabilities(p, n_places) if n_places else np.nan
    for family, pr in probs.items():
        table[f"p_{family}"] = pr
    # désaccord entre modèles : écart maximal entre estimations disponibles
    stack = np.vstack(list(probs.values()) + ([race["mkt_prob"].to_numpy()] if has_market
                                               else []))
    table["disagreement"] = stack.max(axis=0) - stack.min(axis=0)
    # --- incertitude : ensemble bootstrap, hors distribution, statut de course
    payload = models[primary].payload
    members = models[primary].members
    if members:
        lo, hi = member_intervals(members, race)
    else:
        lo = hi = np.full(len(race), np.nan)
    table["p_low"], table["p_high"] = lo, hi
    unc: str = "non évaluée"
    reasons: list[str] = []
    if members and "ood" in payload and "thresholds" in payload:
        ood = OodScorer.from_dict(payload["ood"]).runner_scores(race)
        unc, reasons = race_status(p, lo, hi, ood,
                                   race["mkt_prob"].to_numpy() if has_market else None,
                                   payload["thresholds"])

    # --- explications (relatives au lot) et signaux faibles validés
    contrib = relative_contributions(lgbm[primary], race)
    sw = [strengths_weaknesses(contrib.loc[i]) for i in race.index]
    table["strengths"] = [", ".join(s) for s, _ in sw]
    table["weaknesses"] = [", ".join(w) for _, w in sw]
    table["weak_signals"] = matched_patterns(race, validated)

    if has_market:
        table["edge"] = table["p_win"] - table["mkt_prob"]
        # espérance théorique pour 1 € sur la cote de référence ; le rapport final du
        # pari mutuel sera différent : ce n'est PAS un gain attendu garanti
        table["ev_theoretical"] = table["p_win"] * table["mkt_odds"] - 1.0
        # signal seulement si la borne basse dépasse le marché et la course est exploitable
        table["value_signal"] = ((table["edge"] > MIN_EDGE_UNCERTAINTY)
                                 & (table["p_low"] > table["mkt_prob"])
                                 & (unc == "exploitable"))
    msg = ("Modèle validé." if meta_out[primary]["status"] == "active" else
           "Modèle CANDIDAT non validé : il n'a pas démontré d'amélioration significative "
           "par rapport au marché. Estimations à titre expérimental.")
    train_end = meta_out[primary]["train_end"]
    in_sample = train_end is not None and race_date <= train_end
    if in_sample:
        msg += (" Cette course fait partie de la période d'apprentissage : l'estimation "
                "n'est pas indépendante de son résultat.")
    return RacePrediction(race_id, status, msg, table.sort_values("rank").reset_index(drop=True),
                          meta_out, primary, n_places, has_market, in_sample, unc, reasons)
