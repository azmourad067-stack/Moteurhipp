"""Laboratoire de découverte : patterns, acteurs, anomalies et dérive (traitement hors ligne).

Toutes les analyses suivent le même schéma : une période de DÉCOUVERTE (la plus
ancienne) pour formuler les hypothèses, une période de CONFIRMATION (postérieure et
disjointe) pour les tester. Le nombre total de tests est publié avec les résultats.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, insert, select, update

from horse_intelligence.data.storage import blob_store
from horse_intelligence.data.storage import database as db
from horse_intelligence.intelligence.anomaly_detection import fit_scores, test_top_anomalies
from horse_intelligence.intelligence.bayesian_analysis import (
    actor_table,
    dispersion_message,
    persistence_test,
)
from horse_intelligence.intelligence.feature_engineering import (
    FORM_FEATURES,
    build_dataset,
    load_raw,
)
from horse_intelligence.intelligence.leakage_audit import audit_source_aggregates
from horse_intelligence.intelligence.pattern_discovery import discover, weekly_stability
from horse_intelligence.intelligence.uncertainty import OodScorer
from horse_intelligence.monitoring.drift_detection import feature_drift, weekly_market_quality

CONFIRM_SHARE = 0.4
LAB_REPORT_KEY = "reports/lab-latest.json"
RECENT_DAYS = 7
ANOMALY_FEATURES = [f for f in FORM_FEATURES if f not in ("n_runners", "distance_m",
                                                          "log_prize")]


@dataclass
class LabReport:
    created_at: str
    discovery_period: tuple[str, str]
    confirmation_period: tuple[str, str]
    n_tests: int
    patterns: list[dict[str, Any]]
    actors: dict[str, dict[str, Any]]
    anomalies: dict[str, Any]
    drift: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _names(engine: Engine, table: Any, pk: str) -> dict[int, str]:
    with engine.connect() as conn:
        return {int(r[0]): str(r[1]) for r in conn.execute(
            select(table.c[pk], table.c.display_name))}


def _jsonable(records: list[dict[Any, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        clean: dict[str, Any] = {}
        for key, v in r.items():
            k = str(key)
            if isinstance(v, float | np.floating):
                clean[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, tuple):
                clean[k] = [list(x) for x in v]
            else:
                clean[k] = v
        out.append(clean)
    return out


def _store_patterns(engine: Engine, patterns: list[dict[str, Any]], disc: tuple[str, str],
                    now: datetime) -> None:
    dp = db.detected_patterns
    with engine.begin() as conn:
        existing = {json.dumps(r.definition_json["group"]): r for r in conn.execute(
            select(dp.c.pattern_id, dp.c.definition_json, dp.c.history_json))}
        for p in patterns:
            key = json.dumps(p["group"])
            entry = {"run_at": now.isoformat(), "status": p["status"], "ae": p["ae"],
                     "oos_ae": p.get("oos_ae")}
            values = {
                "definition_json": {"group": p["group"], "label": p["label"]},
                "discovery_start": datetime.fromisoformat(disc[0]).date(),
                "discovery_end": datetime.fromisoformat(disc[1]).date(),
                "n_obs": p["n"], "effect_size": p["ae"], "ci_low": p["ae_low"],
                "ci_high": p["ae_high"], "p_value": p["p_value"], "q_value_fdr": p["q_value"],
                "oos_effect": p.get("oos_ae"), "oos_ci_low": p.get("oos_ae_low"),
                "oos_ci_high": p.get("oos_ae_high"), "status": p["status"],
            }
            if key in existing:
                row = existing[key]
                history = [*(row.history_json or []), entry]
                conn.execute(update(dp).where(dp.c.pattern_id == row.pattern_id)
                             .values(**values, history_json=history))
            else:
                conn.execute(insert(dp).values(**values, history_json=[entry]))


def run_lab(engine: Engine, confirm_share: float = CONFIRM_SHARE,
            store: bool = True) -> LabReport:
    raw = load_raw(engine)
    ds = build_dataset(raw)
    warnings: list[str] = []
    if not audit_source_aggregates(ds.frame).passed:
        warnings.append("Agrégats de la source exclus (audit de fuite non concluant).")
        ds = build_dataset(raw, use_source_aggregates=False)
    frame = ds.frame[ds.frame["won"].notna()]
    dates = sorted(frame["race_date"].unique())
    if len(dates) < 10:
        raise ValueError("historique insuffisant pour le laboratoire (10 jours minimum)")
    cut = int(len(dates) * (1 - confirm_share))
    disc = frame[frame["race_date"] < dates[cut]]
    conf = frame[frame["race_date"] >= dates[cut]]
    disc_period = (str(dates[0]), str(dates[cut - 1]))
    conf_period = (str(dates[cut]), str(dates[-1]))
    now = datetime.now(UTC)

    # --- patterns
    table, n_tests = discover(disc, conf)
    patterns: list[dict[str, Any]] = []
    if not table.empty:
        table["stability"] = [weekly_stability(disc, g, z)
                              for g, z in zip(table["group"], table["z"], strict=True)]
        patterns = _jsonable(table.to_dict("records"))
        if store:
            _store_patterns(engine, patterns, disc_period, now)

    # --- acteurs (jockeys / entraîneurs) : postérieur sur toute la période + persistance
    actors: dict[str, dict[str, Any]] = {}
    for key, table_ref, pk, title in (("jockey_id", db.jockeys, "jockey_id", "jockeys"),
                                      ("trainer_id", db.trainers, "trainer_id", "trainers")):
        tab, prior = actor_table(frame, key)
        names = _names(engine, table_ref, pk)
        tab["name"] = tab[key].map(lambda i, n=names: n.get(int(i), "?"))
        pers = persistence_test(disc, conf, key)
        actors[title] = {"prior_a": prior.a, "n_entities": prior.n_entities,
                         "dispersion": dispersion_message(prior),
                         "persistence": asdict(pers),
                         "table": _jsonable(tab.drop(columns=[key]).to_dict("records"))}

    # --- anomalies : forêt apprise sur la découverte, évaluée sur la confirmation
    feats = [f for f in ANOMALY_FEATURES if disc[f].notna().mean() > 0.5]
    scores = fit_scores(disc, conf, feats)
    test = test_top_anomalies(conf, scores)
    scorer = OodScorer(feats).fit(disc)
    horses = _names(engine, db.horses, "horse_id")
    top = conf.assign(score=scores).nlargest(15, "score")
    examples = [{"date": str(r["race_date"]), "race_id": int(r["race_id"]),
                 "num_pmu": int(r["num_pmu"]), "horse": horses.get(int(r["horse_id"]), "?"),
                 "score": float(r["score"]), "won": float(r["won"]),
                 "mkt_prob": None if pd.isna(r["mkt_prob"]) else float(r["mkt_prob"]),
                 "deviations": [[f, round(v, 2)] for f, v in scorer.top_deviations(r)]}
                for _, r in top.iterrows()]
    anomalies = {"features": feats, "test": asdict(test), "examples": examples}

    # --- dérive
    recent = frame[frame["race_date"] >= dates[-RECENT_DAYS]]
    drift = {"psi": _jsonable(feature_drift(disc, recent, FORM_FEATURES).to_dict("records")),
             "weekly_market": [{"week": str(r["week"].date()), "races": int(r["races"]),
                                "log_loss": float(r["log_loss"])}
                               for r in weekly_market_quality(frame).to_dict("records")]}

    report = LabReport(now.isoformat(), disc_period, conf_period, n_tests, patterns, actors,
                       anomalies, drift, warnings)
    body = json.loads(json.dumps(asdict(report), default=str))
    blob_store.put_json(engine, f"reports/lab-{now:%Y%m%d-%H%M%S}.json", body)
    blob_store.put_json(engine, LAB_REPORT_KEY, body)
    return report


def latest_lab_report(engine: Engine) -> dict[str, Any] | None:
    data: dict[str, Any] | None = blob_store.get_json(engine, LAB_REPORT_KEY)
    return data


def validated_patterns(engine: Engine) -> list[dict[str, Any]]:
    dp = db.detected_patterns
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(
            select(dp).where(dp.c.status == "validée hors échantillon")).mappings()]
