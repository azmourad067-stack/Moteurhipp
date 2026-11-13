"""Entraînement, évaluation walk-forward et enregistrement des modèles (hors interface)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from horse_intelligence.data.storage import blob_store
from horse_intelligence.evaluation.metrics import (
    calibration_table,
    compute_metrics,
    paired_bootstrap,
)
from horse_intelligence.evaluation.walk_forward import ModelRunner, run_walk_forward
from horse_intelligence.intelligence.feature_engineering import (
    CATEGORICAL,
    FEATURE_SET_VERSION,
    FORM_FEATURES,
    RELATIVE_FEATURES,
    build_dataset,
    load_raw,
)
from horse_intelligence.intelligence.leakage_audit import audit_source_aggregates
from horse_intelligence.intelligence.uncertainty import (
    N_MEMBERS,
    OodScorer,
    fit_members,
    learn_thresholds,
    member_intervals,
)
from horse_intelligence.models.baseline import (
    CalibratedMarket,
    FormLogit,
    market_proba,
    uniform_proba,
)
from horse_intelligence.models.lightgbm_model import LgbmRaceModel
from horse_intelligence.models.ranking import FloatArray
from horse_intelligence.models.registry import register_model, save_artifact

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = FORM_FEATURES + RELATIVE_FEATURES
LGBM_FEATURES = NUMERIC_FEATURES + CATEGORICAL
REFERENCE_MODEL = "market_calibrated"
CANDIDATES = ("lgbm_form", "lgbm_market", "form_logit")
MIN_RACES_FOR_PROMOTION = 300
WF_REPORT_KEY = "reports/walk_forward-latest.json"
WF_PREDICTIONS_KEY = "reports/walk_forward_predictions-latest.parquet"
OOS_COLUMNS = ["race_id", "race_date", "num_pmu", "fold", "won", "finish_position",
               "n_runners", "declared_runners", "discipline", "distance_m", "racecourse",
               "mkt_prob", "mkt_odds", "mkt_pool", "mkt_rank"]


@dataclass
class TrainingReport:
    created_at: str
    feature_set: str
    period: tuple[str, str]
    n_races: int
    n_runners: int
    n_races_with_market: int
    source_aggregates_audit: dict[str, Any]
    folds: list[dict[str, Any]]
    metrics_all_races: dict[str, dict[str, float]]
    metrics_market_races: dict[str, dict[str, float]]
    vs_reference: dict[str, dict[str, float]]
    calibration: dict[str, list[dict[str, Any]]]
    feature_importance: dict[str, Any]
    registered: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: dict[str, list[str]] = field(default_factory=dict)


def _with_market(f: pd.DataFrame) -> pd.DataFrame:
    return f[f.groupby("race_id")["mkt_prob"].transform(lambda s: s.notna().all())]


def _masked(test: pd.DataFrame, sub: pd.DataFrame, proba: FloatArray) -> FloatArray:
    out = pd.Series(np.nan, index=test.index)
    out.loc[sub.index] = proba
    return np.asarray(out.to_numpy(), dtype=np.float64)


def model_runners() -> dict[str, ModelRunner]:
    def market_cal(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        train = _with_market(pd.concat([base, calib]))
        sub = _with_market(test)
        return _masked(test, sub, CalibratedMarket().fit(train).predict(sub))

    def form_logit(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        return FormLogit(NUMERIC_FEATURES).fit(pd.concat([base, calib])).predict(test)

    def lgbm_form(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        m = LgbmRaceModel(LGBM_FEATURES, CATEGORICAL).fit(base, calib)
        return m.predict(test)

    def lgbm_market(base: pd.DataFrame, calib: pd.DataFrame, test: pd.DataFrame) -> FloatArray:
        m = LgbmRaceModel(LGBM_FEATURES, CATEGORICAL, with_market=True)
        m.fit(base, _with_market(calib))
        sub = _with_market(test)
        return _masked(test, sub, m.predict(sub))

    return {
        "uniform": lambda b, c, t: uniform_proba(t),
        "market": lambda b, c, t: market_proba(t),
        REFERENCE_MODEL: market_cal,
        "form_logit": form_logit,
        "lgbm_form": lgbm_form,
        "lgbm_market": lgbm_market,
    }


def _final_with_uncertainty(base: pd.DataFrame, cal: pd.DataFrame, with_market: bool,
                            n_members: int) -> dict[str, Any]:
    """Modèle final + ensemble bootstrap + score hors distribution + seuils de statut.

    Les seuils sont des quantiles mesurés sur la période de calibration (sur laquelle le
    calibrateur a été appris) : ils décrivent la dispersion habituelle, ce ne sont pas
    des mesures de performance.
    """
    cal_fit = _with_market(cal) if with_market else cal
    model = LgbmRaceModel(LGBM_FEATURES, CATEGORICAL, with_market=with_market)
    model.fit(base, cal_fit)
    members = fit_members(base, cal_fit, LGBM_FEATURES, CATEGORICAL, with_market, n_members)
    ood = OodScorer(NUMERIC_FEATURES).fit(base)
    ref = _with_market(cal)
    p = model.predict(ref)
    lo, hi = member_intervals(members, ref)
    thresholds = learn_thresholds(ref, p, lo, hi, ood.runner_scores(ref))
    return {**model.to_dict(), "members": [m.to_dict() for m in members],
            "ood": ood.to_dict(), "thresholds": thresholds.to_dict()}


def _version() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def train_and_evaluate(engine: Engine, min_train_days: int = 21, test_days: int = 7,
                       calib_days: int = 7, register: bool = True,
                       n_members: int = N_MEMBERS) -> TrainingReport:
    raw = load_raw(engine)
    ds = build_dataset(raw)
    audit = audit_source_aggregates(ds.frame)
    warnings: list[str] = []
    if not audit.passed:
        warnings.append(f"Agrégats de la source exclus : {audit.message}")
        ds = build_dataset(raw, use_source_aggregates=False)
    frame = ds.frame[ds.frame["won"].notna() & (ds.frame["n_runners"] >= 2)]
    if frame.empty:
        raise ValueError("aucune course avec résultat : collectez des données d'abord")

    wf = run_walk_forward(frame, model_runners(), min_train_days, test_days, calib_days)
    if wf.predictions.empty:
        raise ValueError(f"historique insuffisant : il faut plus de {min_train_days} jours "
                         "de courses avec résultats")
    evald = frame.merge(wf.predictions.drop(columns=["race_date"]), on=["race_id", "num_pmu"])
    names = [n for n in model_runners() if evald[n].notna().any()]
    all_races = {n: compute_metrics(evald, evald[n]).as_dict()
                 for n in names if not n.startswith(("market", "lgbm_market"))}
    mkt = _with_market(evald)
    mkt = mkt[mkt.groupby("race_id")[names].transform(lambda s: s.notna().all()).all(axis=1)]
    market_races = {n: compute_metrics(mkt, mkt[n]).as_dict() for n in names}
    vs_ref = {n: paired_bootstrap(mkt, mkt[n], mkt[REFERENCE_MODEL])
              for n in names if n != REFERENCE_MODEL}
    calib: dict[str, list[dict[str, Any]]] = {
        n: [{str(k): v for k, v in row.items()} for row in
            calibration_table(mkt[n], mkt["won"]).to_dict("records")]
        for n in names if len(mkt) >= 10}

    # --- modèles finaux : apprentissage sur tout l'historique disponible
    dates = sorted(frame["race_date"].unique())
    calib_dates = set(dates[-calib_days:])
    base = frame[~frame["race_date"].isin(calib_dates)]
    cal = frame[frame["race_date"].isin(calib_dates)]
    finals: dict[str, dict[str, Any]] = {
        family: _final_with_uncertainty(base, cal, with_market, n_members)
        for family, with_market in (("lgbm_form", False), ("lgbm_market", True))
    }
    finals[REFERENCE_MODEL] = {"calibrator": CalibratedMarket().fit(_with_market(frame))
                               .model.to_dict()}
    importance = LgbmRaceModel.from_dict(finals["lgbm_form"]).feature_importance()

    n_market_races = int(mkt["race_id"].nunique())
    if n_market_races < MIN_RACES_FOR_PROMOTION:
        warnings.append(f"Échantillon d'évaluation réduit ({n_market_races} courses) : "
                        f"aucune promotion possible sous {MIN_RACES_FOR_PROMOTION} courses.")

    report = TrainingReport(
        created_at=datetime.now(UTC).isoformat(), feature_set=FEATURE_SET_VERSION,
        period=(str(dates[0]), str(dates[-1])), n_races=int(frame["race_id"].nunique()),
        n_runners=len(frame), n_races_with_market=int(_with_market(frame)["race_id"].nunique()),
        source_aggregates_audit=asdict(audit),
        folds=[{"train": [str(f.train_dates[0]), str(f.train_dates[-1])],
                "calib": [str(f.calib_dates[0]), str(f.calib_dates[-1])],
                "test": [str(f.test_dates[0]), str(f.test_dates[-1])]} for f in wf.folds],
        metrics_all_races=all_races, metrics_market_races=market_races, vs_reference=vs_ref,
        calibration=calib, feature_importance={str(k): float(v) for k, v in
                            importance.head(20).round(2).items()},
        warnings=warnings, errors=wf.errors,
    )

    if register:
        version = _version()
        for family, payload in finals.items():
            cmp = vs_ref.get(family)
            promote = family == REFERENCE_MODEL or bool(
                cmp and n_market_races >= MIN_RACES_FOR_PROMOTION and cmp["ci_high"] < 0)
            payload = {**payload, "family": family, "version": version,
                       "feature_set": FEATURE_SET_VERSION,
                       "use_source_aggregates": audit.passed}
            uri, sha = save_artifact(engine, family, version, payload)
            m = market_races.get(family, {})
            metric_rows = [{"eval_protocol": "walk_forward", "period_start": dates[0],
                            "period_end": dates[-1], "segment": "courses_avec_marche",
                            "metric": k, "value": v, "n_races": m.get("n_races")}
                           for k, v in m.items() if k not in ("n_races", "n_runners")]
            if cmp:
                metric_rows.append({"eval_protocol": "walk_forward", "period_start": dates[0],
                                    "period_end": dates[-1], "segment": "courses_avec_marche",
                                    "metric": f"log_loss_diff_vs_{REFERENCE_MODEL}",
                                    "value": cmp["mean_diff"], "ci_low": cmp["ci_low"],
                                    "ci_high": cmp["ci_high"], "n_races": cmp["n_races"]})
            model_id = register_model(engine, family, version, uri, sha, dates[0], dates[-1],
                                      {"min_train_days": min_train_days,
                                       "test_days": test_days, "calib_days": calib_days},
                                      metric_rows, promote)
            report.registered[family] = {"model_id": model_id, "version": version,
                                         "status": "active" if promote else "candidate",
                                         "artifact": uri}

    body = asdict(report)
    blob_store.put_json(engine, f"reports/walk_forward-{_version()}.json", body)
    blob_store.put_json(engine, WF_REPORT_KEY, body)
    # prédictions hors échantillon : seule base légitime pour simuler des paris (phase 5)
    keep = [*OOS_COLUMNS, *names]
    oos = evald[[c for c in keep if c in evald.columns]].copy()
    for c in ("discipline", "racecourse"):
        oos[c] = oos[c].astype(object)
    blob_store.put_frame(engine, WF_PREDICTIONS_KEY, oos)
    _prune_artifacts(engine)
    return report


KEEP_MODEL_VERSIONS = 3
KEEP_REPORTS = 3


def _prune_artifacts(engine: Engine) -> None:
    """Base gratuite limitée : on garde les 3 dernières versions de chaque famille (retour
    arrière possible) et toujours l'artefact du modèle actif."""
    from horse_intelligence.models.registry import latest_model

    active = {m["artifact_uri"].removeprefix(blob_store.URI_PREFIX)
              for f in ("lgbm_market", "lgbm_form", REFERENCE_MODEL)
              if (m := latest_model(engine, f, ("active",))) is not None}
    for family in ("lgbm_market", "lgbm_form", REFERENCE_MODEL):
        blob_store.prune(engine, f"models/{family}-", KEEP_MODEL_VERSIONS, active)
    for kind in ("walk_forward-", "lab-", "validation-"):
        blob_store.prune(engine, f"reports/{kind}", KEEP_REPORTS)


def latest_report(engine: Engine) -> dict[str, Any] | None:
    data: dict[str, Any] | None = blob_store.get_json(engine, WF_REPORT_KEY)
    return data


def report_period(report: dict[str, Any]) -> tuple[date, date]:
    a, b = report["period"]
    return date.fromisoformat(a), date.fromisoformat(b)


def load_oos_predictions(engine: Engine) -> pd.DataFrame | None:
    return blob_store.get_frame(engine, WF_PREDICTIONS_KEY)
