"""Construction des variables prédictives « as-of » (sans fuite de données).

Règles :
- l'instant de décision d'une course est `scheduled_start_utc - DECISION_OFFSET` ;
- l'historique interne (chevaux, jockeys, entraîneurs) n'utilise que les courses
  de dates STRICTEMENT antérieures ;
- les cotes de marché sont des cotes de référence publiées avant l'instant de décision ;
- les champs déclaratifs d'un engagement (cheval, monte, corde, poids, ferrure…)
  et les agrégats de la source (musique, cumuls de carrière) sont utilisés tels
  qu'observés : leur caractère « avant course » est vérifié par
  intelligence/leakage_audit.py à chaque entraînement ;
- les résultats ne servent qu'à la cible et à l'historique des courses ultérieures.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from horse_intelligence.intelligence.musique import musique_features

DECISION_OFFSET = timedelta(minutes=10)
FEATURE_SET_VERSION = "fs-2026.10.3"

# Lissage bayésien empirique des taux de réussite (pseudo-observations)
SHRINK_RUNS = 20.0
# Taux de victoire vers lequel on lisse : constante fixée a priori (≈ 1 / partants moyens).
# Une moyenne calculée sur l'historique inclurait des résultats postérieurs à la course.
PRIOR_WIN_RATE = 0.08
CAREER_SHRINK = 5.0

CATEGORICAL = ["discipline", "racecourse", "sex", "blinkers", "shoes"]

FORM_FEATURES = [
    "age", "draw", "draw_rel", "weight_carried_kg", "weight_vs_field", "handicap_extra_m",
    "n_runners", "distance_m", "log_prize",
    "mus_n", "mus_last_pos", "mus_mean_pos_3", "mus_mean_pos_6", "mus_trend", "mus_wins_6",
    "mus_top3_6", "mus_incidents_6", "mus_same_disc_share",
    "career_starts", "career_win_rate", "career_place_rate", "log_earnings_per_start",
    "horse_db_runs", "horse_days_since_db", "horse_db_mean_pct",
    "jockey_win_rate", "jockey_db_runs", "trainer_win_rate", "trainer_db_runs",
    "pair_db_runs", "rider_change",
]
# Variables ramenées à la course (centrées dans la course) : utiles au logit conditionnel
RELATIVE_SOURCES = ["mus_mean_pos_6", "career_win_rate", "jockey_win_rate",
                    "trainer_win_rate", "log_earnings_per_start", "weight_carried_kg"]
RELATIVE_FEATURES = [f"{c}_rel" for c in RELATIVE_SOURCES]
MARKET_FEATURES = ["mkt_prob", "mkt_log_prob", "mkt_rank"]


@dataclass
class Dataset:
    frame: pd.DataFrame          # une ligne par partant (non-partants exclus)
    feature_set_version: str = FEATURE_SET_VERSION

    @property
    def model_features(self) -> list[str]:
        return FORM_FEATURES + RELATIVE_FEATURES

    def races(self) -> pd.Index:
        return pd.Index(self.frame["race_id"].unique())


def _read(engine: Engine, sql: str, **params: Any) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="mixed")


def load_raw(engine: Engine, until: date | None = None) -> dict[str, pd.DataFrame]:
    until = until or date(2100, 1, 1)
    races = _read(engine, """
        SELECT r.race_id, r.race_date, r.reunion_num, r.course_num, r.discipline,
               r.distance_m, r.prize_eur, r.declared_runners, r.scheduled_start_utc,
               r.result_final, r.country, rc.code AS racecourse
        FROM races r LEFT JOIN racecourses rc ON rc.racecourse_id = r.racecourse_id
        WHERE r.race_date <= :until""", until=until)
    entries = _read(engine, """
        SELECT e.* FROM race_entries e JOIN races r ON r.race_id = e.race_id
        WHERE r.race_date <= :until""", until=until)
    results = _read(engine, """
        SELECT res.race_id, res.num_pmu, res.finish_position, res.incident
        FROM race_results res JOIN races r ON r.race_id = res.race_id
        WHERE r.race_date <= :until""", until=until)
    odds = _read(engine, """
        SELECT o.race_id, o.num_pmu, o.pool, o.odds, o.published_at, o.observed_at
        FROM odds_snapshots o JOIN races r ON r.race_id = o.race_id
        WHERE r.race_date <= :until AND o.odds_kind = 'reference'
          AND o.bet_type = 'SIMPLE_GAGNANT'""", until=until)
    for frame, cols in ((races, ["distance_m", "prize_eur", "declared_runners"]),
                        (entries, ["age", "draw", "weight_kg", "weight_carried_kg",
                                   "handicap_distance_m", "career_starts", "career_wins",
                                   "career_places", "career_earnings"]),
                        (results, ["finish_position"]), (odds, ["odds"])):
        for c in cols:  # une colonne entièrement vide arrive en type objet
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float64")
    races["race_date"] = pd.to_datetime(races["race_date"]).dt.date
    races["scheduled_start_utc"] = _utc(races["scheduled_start_utc"])
    entries["observed_at"] = _utc(entries["observed_at"])
    odds["published_at"] = _utc(odds["published_at"])
    odds["observed_at"] = _utc(odds["observed_at"])
    return {"races": races, "entries": entries, "results": results, "odds": odds}


def _entries_at_decision(entries: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Version de chaque engagement connue à l'instant de décision.

    Si aucune version n'a été observée avant la décision (collecte a posteriori),
    la plus ancienne observation est retenue : les champs déclaratifs ne dépendent pas
    de l'issue de la course (vérifié par l'audit de fuite).
    """
    e = entries.merge(races[["race_id", "decision_time"]], on="race_id")
    e = e.sort_values("observed_at")
    before = e[e["observed_at"] <= e["decision_time"]]
    latest_before = before.drop_duplicates(["race_id", "num_pmu"], keep="last")
    earliest = e.drop_duplicates(["race_id", "num_pmu"], keep="first")
    missing = earliest.merge(latest_before[["race_id", "num_pmu"]], how="left",
                             on=["race_id", "num_pmu"], indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    out = pd.concat([latest_before, missing], ignore_index=True)
    out["entry_seen_before_decision"] = out["observed_at"] <= out["decision_time"]
    return out


def _market(odds: pd.DataFrame, races: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    """Probabilités implicites des cotes de référence disponibles à la décision.

    Par course, une seule masse est utilisée (INTERNET de préférence, sinon OFFLINE) et
    seulement si tous les partants ont une cote ; sinon le marché est absent (NaN).
    """
    o = odds.merge(races[["race_id", "decision_time", "scheduled_start_utc"]], on="race_id")
    o = o[o["published_at"].notna() & (o["published_at"] < o["scheduled_start_utc"])
          & (o["published_at"] <= o["decision_time"])]
    o = o.sort_values("published_at").drop_duplicates(["race_id", "num_pmu", "pool"],
                                                      keep="last")
    o = o.merge(runners[["race_id", "num_pmu"]], on=["race_id", "num_pmu"])
    n_runners = runners.groupby("race_id").size().rename("n")
    cover = o.groupby(["race_id", "pool"]).size().rename("k").reset_index()
    cover = cover.merge(n_runners, left_on="race_id", right_index=True)
    cover = cover[cover["k"] == cover["n"]]
    cover["pref"] = (cover["pool"] != "INTERNET").astype(int)
    chosen = cover.sort_values("pref").drop_duplicates("race_id")[["race_id", "pool"]]
    o = o.merge(chosen, on=["race_id", "pool"])
    o["implied"] = 1.0 / o["odds"]
    total = o.groupby("race_id")["implied"].transform("sum")
    o["mkt_prob"] = o["implied"] / total
    o["mkt_overround"] = total
    o["mkt_log_prob"] = np.log(o["mkt_prob"])
    o["mkt_rank"] = o.groupby("race_id")["odds"].rank(method="min")
    return o[["race_id", "num_pmu", "pool", "odds", "mkt_prob", "mkt_log_prob", "mkt_rank",
              "mkt_overround"]].rename(columns={"pool": "mkt_pool", "odds": "mkt_odds"})


def _asof_before(df: pd.DataFrame, hist: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Joint à chaque ligne l'état d'historique de la DERNIÈRE date strictement antérieure.

    `hist` contient des cumuls INCLUANT leur propre date ; la jointure exclut la date de
    la course (allow_exact_matches=False) : aucune information du jour même n'est utilisée.
    """
    left = df.assign(_d=pd.to_datetime(df["race_date"])).sort_values("_d")
    right = hist.assign(_d=pd.to_datetime(hist["race_date"]),
                        _hist_date=pd.to_datetime(hist["race_date"]))
    right = right.drop(columns="race_date").sort_values("_d")
    for k in keys:  # merge_asof exige des types de clés identiques
        left[k] = left[k].astype("float64")
        right[k] = right[k].astype("float64")
    merged = pd.merge_asof(left, right, on="_d", by=keys, direction="backward",
                           allow_exact_matches=False)
    return merged.drop(columns="_d")


def _actor_history(hist: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    d = hist.dropna(subset=[key]).groupby([key, "race_date"]).agg(
        runs=("won", "size"), wins=("won", "sum")).reset_index().sort_values("race_date")
    g = d.groupby(key)
    d[f"{prefix}_cum_runs"] = g["runs"].cumsum()
    d[f"{prefix}_cum_wins"] = g["wins"].cumsum()
    return d[[key, "race_date", f"{prefix}_cum_runs", f"{prefix}_cum_wins"]]


def _horse_history(hist: pd.DataFrame) -> pd.DataFrame:
    d = hist.groupby(["horse_id", "race_date"]).agg(
        runs=("won", "size"), pct=("finish_pct", "mean"),
        horse_last_jockey=("jockey_id", "last")).reset_index()
    d = d.sort_values(["horse_id", "race_date"])
    g = d.groupby("horse_id")
    d["horse_db_runs"] = g["runs"].cumsum()
    d["horse_db_mean_pct"] = g["pct"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return d[["horse_id", "race_date", "horse_db_runs", "horse_db_mean_pct",
              "horse_last_jockey"]]


def build_dataset(raw: dict[str, pd.DataFrame], use_source_aggregates: bool = True) -> Dataset:
    races = raw["races"].copy()
    races["decision_time"] = races["scheduled_start_utc"] - DECISION_OFFSET
    races = races[races["scheduled_start_utc"].notna()]

    e = _entries_at_decision(raw["entries"], races)
    e = e[e["status"].fillna("").str.upper() != "NON_PARTANT"]
    df = e.merge(races, on=["race_id", "decision_time"])
    df = df.merge(raw["results"], on=["race_id", "num_pmu"], how="left")

    n = df.groupby("race_id")["num_pmu"].transform("size")
    df["n_runners"] = n.astype(float)
    df["won"] = (df["finish_position"] == 1).astype(float)
    df["has_result"] = df.groupby("race_id")["finish_position"].transform(
        lambda s: s.notna().any()) & df["result_final"].astype(bool)
    # rang d'arrivée en percentile (0 = premier, 1 = dernier ; non classé = 1)
    df["finish_pct"] = ((df["finish_position"] - 1) / (n - 1).clip(lower=1)).clip(0, 1)
    df.loc[df["finish_position"].isna(), "finish_pct"] = 1.0
    df.loc[~df["has_result"], ["won", "finish_pct"]] = np.nan

    # --- conditions de course et engagement
    df["draw_rel"] = df["draw"] / n
    df["log_prize"] = np.log1p(df["prize_eur"])
    df["weight_vs_field"] = df["weight_carried_kg"] - df.groupby("race_id")[
        "weight_carried_kg"].transform("mean")
    df["handicap_extra_m"] = df["handicap_distance_m"] - df["distance_m"]

    # --- agrégats de la source
    mus = pd.DataFrame([musique_features(f, d) for f, d in
                        zip(df["form_string"], df["discipline"], strict=True)], index=df.index)
    df = pd.concat([df, mus], axis=1)
    starts = df["career_starts"]
    df["career_win_rate"] = (df["career_wins"] + 0.1 * CAREER_SHRINK) / (starts + CAREER_SHRINK)
    df["career_place_rate"] = (df["career_places"] + 0.3 * CAREER_SHRINK) / (
        starts + CAREER_SHRINK)
    df["log_earnings_per_start"] = np.log1p(df["career_earnings"] / starts.where(starts > 0))
    if not use_source_aggregates:
        source_cols = [*mus.columns, "career_starts", "career_win_rate", "career_place_rate",
                       "log_earnings_per_start"]
        df[source_cols] = np.nan

    # --- historique interne (dates strictement antérieures)
    hist = df[df["has_result"]]
    prior = PRIOR_WIN_RATE
    for key, prefix in (("jockey_id", "jockey"), ("trainer_id", "trainer")):
        df = _asof_before(df, _actor_history(hist, key, prefix), [key]).drop(
            columns="_hist_date")
        runs = df.pop(f"{prefix}_cum_runs").fillna(0.0)
        wins = df.pop(f"{prefix}_cum_wins").fillna(0.0)
        df[f"{prefix}_db_runs"] = runs
        df[f"{prefix}_win_rate"] = (wins + SHRINK_RUNS * prior) / (runs + SHRINK_RUNS)
    df = _asof_before(df, _horse_history(hist), ["horse_id"])
    df["horse_days_since_db"] = (pd.to_datetime(df["race_date"]) - df.pop("_hist_date")).dt.days
    df["horse_db_runs"] = df["horse_db_runs"].fillna(0.0)
    # changement de jockey/driver depuis la dernière course connue (NaN sans historique)
    last = df.pop("horse_last_jockey")
    df["rider_change"] = np.where(last.isna(), np.nan,
                                  (df["jockey_id"].astype("float64") != last).astype(float))
    pair = hist.groupby(["horse_id", "jockey_id", "race_date"]).size().rename("r").reset_index()
    pair = pair.sort_values("race_date")
    pair["pair_db_runs"] = pair.groupby(["horse_id", "jockey_id"])["r"].cumsum()
    df = _asof_before(df, pair[["horse_id", "jockey_id", "race_date", "pair_db_runs"]],
                      ["horse_id", "jockey_id"]).drop(columns="_hist_date")
    df["pair_db_runs"] = df["pair_db_runs"].fillna(0.0)

    # --- variables relatives à la course
    for c in RELATIVE_SOURCES:
        df[f"{c}_rel"] = df[c] - df.groupby("race_id")[c].transform("mean")

    # --- marché
    df = df.merge(_market(raw["odds"], races, df), on=["race_id", "num_pmu"], how="left")

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return Dataset(df.sort_values(["race_date", "race_id", "num_pmu"]).reset_index(drop=True))
