"""Schéma relationnel (SQLAlchemy Core) — compatible SQLite et PostgreSQL.

Principes (docs/00_audit_et_architecture.md §4) :
- tout fait observable porte `observed_at` (moment de NOTRE collecte, UTC) et, si
  connu, `published_at` (moment de publication par la source) ;
- les partants sont historisés en ajout seul (une ligne par version observée) ;
- chaque ligne référence la réponse brute dont elle provient (`payload_id`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Dialect,
    Engine,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """Horodatage toujours stocké et restitué en UTC explicite."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("horodatage naïf refusé : préciser le fuseau (UTC)")
        return value.astimezone(UTC)

    def process_result_value(self, value: Any, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


metadata = MetaData()

# --- Sources et traçabilité -------------------------------------------------
data_sources = Table(
    "data_sources", metadata,
    Column("source_key", String(40), primary_key=True),
    Column("name", String(200), nullable=False),
    Column("base_url", String(500)),
    Column("kind", String(40)),
    Column("enabled", Boolean, nullable=False),
    Column("status_audit", String(40)),
    Column("last_status", String(40)),
    Column("last_checked_at", UTCDateTime),
    Column("last_message", Text),
)

raw_payloads = Table(
    "raw_payloads", metadata,
    Column("payload_id", Integer, primary_key=True, autoincrement=True),
    Column("source_key", String(40), nullable=False),
    Column("endpoint", String(80), nullable=False),
    Column("url", String(1000), nullable=False),
    Column("params_json", JSON),
    Column("fetched_at", UTCDateTime, nullable=False),
    Column("http_status", Integer, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("storage_uri", String(1000), nullable=False),
    Index("ix_raw_payloads_url_fetched", "url", "fetched_at"),
)

collection_runs = Table(
    "collection_runs", metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("target_date", Date, nullable=False),
    Column("started_at", UTCDateTime, nullable=False),
    Column("finished_at", UTCDateTime),
    Column("status", String(20), nullable=False),  # running | success | partial | failed
    Column("races_seen", Integer, default=0),
    Column("races_ok", Integer, default=0),
    Column("errors_json", JSON),
)

# --- Référentiels -----------------------------------------------------------
racecourses = Table(
    "racecourses", metadata,
    Column("racecourse_id", Integer, primary_key=True, autoincrement=True),
    Column("code", String(20), nullable=False, unique=True),
    Column("name", String(200)),
    Column("country", String(10)),
    Column("lat", Float),   # renseigné manuellement ; jamais deviné
    Column("lon", Float),
)

horses = Table(
    "horses", metadata,
    Column("horse_id", Integer, primary_key=True, autoincrement=True),
    Column("canonical_name", String(200), nullable=False),
    Column("display_name", String(200), nullable=False),
    Column("birth_year", Integer),
    Column("breed", String(80)),
    Column("sex_last", String(20)),
    UniqueConstraint("canonical_name", "birth_year", name="uq_horse_identity"),
)


def _person_table(name: str, pk: str) -> Table:
    return Table(
        name, metadata,
        Column(pk, Integer, primary_key=True, autoincrement=True),
        Column("canonical_name", String(200), nullable=False, unique=True),
        Column("display_name", String(200), nullable=False),
    )


jockeys = _person_table("jockeys", "jockey_id")      # jockeys et drivers
trainers = _person_table("trainers", "trainer_id")
owners = _person_table("owners", "owner_id")

# --- Courses ------------------------------------------------------------------
races = Table(
    "races", metadata,
    Column("race_id", Integer, primary_key=True, autoincrement=True),
    Column("race_date", Date, nullable=False),
    Column("reunion_num", Integer, nullable=False),
    Column("course_num", Integer, nullable=False),
    Column("racecourse_id", ForeignKey("racecourses.racecourse_id")),
    Column("country", String(10)),
    Column("label", String(300)),
    Column("discipline", String(40)),
    Column("discipline_raw", String(40)),
    Column("specialite_raw", String(40)),
    Column("distance_m", Integer),
    Column("corde", String(40)),
    Column("prize_eur", Float),
    Column("declared_runners", Integer),
    Column("sex_condition", String(40)),
    Column("conditions_text", Text),
    Column("scheduled_start_utc", UTCDateTime),
    Column("status", String(40)),
    Column("result_final", Boolean, default=False),
    Column("source_key", String(40), nullable=False),
    Column("first_observed_at", UTCDateTime, nullable=False),
    Column("last_observed_at", UTCDateTime, nullable=False),
    Column("payload_id", ForeignKey("raw_payloads.payload_id")),
    UniqueConstraint("race_date", "reunion_num", "course_num", name="uq_race_key"),
    Index("ix_races_date", "race_date"),
)

race_entries = Table(
    "race_entries", metadata,
    Column("entry_id", Integer, primary_key=True, autoincrement=True),
    Column("race_id", ForeignKey("races.race_id"), nullable=False),
    Column("num_pmu", Integer, nullable=False),
    Column("horse_id", ForeignKey("horses.horse_id"), nullable=False),
    Column("jockey_id", ForeignKey("jockeys.jockey_id")),
    Column("trainer_id", ForeignKey("trainers.trainer_id")),
    Column("owner_id", ForeignKey("owners.owner_id")),
    Column("age", Integer),
    Column("sex", String(20)),
    Column("draw", Integer),
    Column("weight_kg", Float),           # poids du handicap / des conditions
    Column("weight_carried_kg", Float),   # poids réellement porté (après décharge)
    Column("handicap_distance_m", Integer),
    Column("blinkers", String(40)),
    Column("shoes", String(60)),
    Column("status", String(30)),
    # Agrégats fournis par la source : ambigus temporellement (voir audit §2.3).
    # Utilisables uniquement si observed_after_start est False.
    Column("form_string", String(200)),
    Column("career_starts", Integer),
    Column("career_wins", Integer),
    Column("career_places", Integer),
    Column("career_earnings", Float),
    Column("observed_at", UTCDateTime, nullable=False),
    Column("observed_after_start", Boolean, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("payload_id", ForeignKey("raw_payloads.payload_id")),
    Index("ix_entries_race_num_obs", "race_id", "num_pmu", "observed_at"),
    Index("ix_entries_horse", "horse_id"),
)

race_results = Table(
    "race_results", metadata,
    Column("result_id", Integer, primary_key=True, autoincrement=True),
    Column("race_id", ForeignKey("races.race_id"), nullable=False),
    Column("num_pmu", Integer, nullable=False),
    Column("horse_id", ForeignKey("horses.horse_id"), nullable=False),
    Column("finish_position", Integer),  # NULL = non classé / disqualifié / non-partant
    Column("incident", String(60)),
    Column("time_s", Float),
    Column("reduction_km_s", Float),       # trot : temps au kilomètre
    Column("observed_at", UTCDateTime, nullable=False),
    Column("payload_id", ForeignKey("raw_payloads.payload_id")),
    UniqueConstraint("race_id", "num_pmu", name="uq_result"),
    Index("ix_results_horse", "horse_id"),
)

odds_snapshots = Table(
    "odds_snapshots", metadata,
    Column("odds_id", Integer, primary_key=True, autoincrement=True),
    Column("race_id", ForeignKey("races.race_id"), nullable=False),
    Column("num_pmu", Integer, nullable=False),
    Column("bet_type", String(40), nullable=False),
    Column("pool", String(20), nullable=False),  # OFFLINE (points de vente) | INTERNET
    # reference : cote de référence publiée ; live : relevée AVANT le départ ;
    # post_start : relevée APRÈS le départ (≈ finale) → interdite pour la décision
    Column("odds_kind", String(20), nullable=False),
    Column("odds", Float, nullable=False),
    Column("published_at", UTCDateTime),
    Column("observed_at", UTCDateTime, nullable=False),
    Column("payload_id", ForeignKey("raw_payloads.payload_id")),
    Index("ix_odds_race_obs", "race_id", "observed_at"),
)

payouts = Table(
    "payouts", metadata,
    Column("payout_id", Integer, primary_key=True, autoincrement=True),
    Column("race_id", ForeignKey("races.race_id"), nullable=False),
    Column("bet_type", String(60), nullable=False),
    Column("pool", String(20), nullable=False),
    Column("combination", String(100)),
    Column("payout_per_eur", Float),
    Column("observed_at", UTCDateTime, nullable=False),
    Column("payload_id", ForeignKey("raw_payloads.payload_id")),
    UniqueConstraint("race_id", "pool", "bet_type", "combination", name="uq_payout"),
)

weather_conditions = Table(
    "weather_conditions", metadata,
    Column("weather_id", Integer, primary_key=True, autoincrement=True),
    Column("racecourse_id", ForeignKey("racecourses.racecourse_id"), nullable=False),
    Column("valid_time_utc", UTCDateTime, nullable=False),
    Column("kind", String(20), nullable=False),  # forecast | observed
    Column("temp_c", Float),
    Column("precip_mm", Float),
    Column("wind_ms", Float),
    Column("issued_at", UTCDateTime, nullable=False),
    Column("observed_at", UTCDateTime, nullable=False),
)

racecourse_conditions = Table(
    "racecourse_conditions", metadata,
    Column("condition_id", Integer, primary_key=True, autoincrement=True),
    Column("race_id", ForeignKey("races.race_id"), nullable=False),
    Column("going_label", String(80)),
    Column("penetrometer", String(20)),
    Column("observed_at", UTCDateTime, nullable=False),
)

data_quality_reports = Table(
    "data_quality_reports", metadata,
    Column("report_id", Integer, primary_key=True, autoincrement=True),
    Column("scope", String(40), nullable=False),     # race | day | source
    Column("scope_ref", String(100), nullable=False),
    Column("run_at", UTCDateTime, nullable=False),
    Column("status", String(30), nullable=False),    # verified | partial | insufficient
    Column("checks_json", JSON, nullable=False),
)

# --- Tables des phases suivantes (créées dès maintenant, alimentées plus tard) ----
feature_sets = Table(
    "feature_sets", metadata,
    Column("feature_set_id", Integer, primary_key=True, autoincrement=True),
    Column("version", String(40), nullable=False, unique=True),
    Column("code_hash", String(64), nullable=False),
    Column("definition_json", JSON, nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
)

model_versions = Table(
    "model_versions", metadata,
    Column("model_id", Integer, primary_key=True, autoincrement=True),
    Column("family", String(60), nullable=False),
    Column("version", String(40), nullable=False),
    Column("feature_set_id", ForeignKey("feature_sets.feature_set_id")),
    Column("train_start", Date),
    Column("train_end", Date),
    Column("hyperparams_json", JSON),
    Column("artifact_uri", String(1000)),
    Column("artifact_sha256", String(64)),
    Column("git_commit", String(40)),
    Column("status", String(20), nullable=False),  # candidate | active | retired
    Column("created_at", UTCDateTime, nullable=False),
    UniqueConstraint("family", "version", name="uq_model_version"),
)

model_metrics = Table(
    "model_metrics", metadata,
    Column("metric_id", Integer, primary_key=True, autoincrement=True),
    Column("model_id", ForeignKey("model_versions.model_id"), nullable=False),
    Column("eval_protocol", String(60), nullable=False),
    Column("period_start", Date),
    Column("period_end", Date),
    Column("segment", String(100)),
    Column("metric", String(60), nullable=False),
    Column("value", Float),
    Column("ci_low", Float),
    Column("ci_high", Float),
    Column("n_races", Integer),
    Column("n_entries", Integer),
)

predictions = Table(
    "predictions", metadata,
    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
    Column("race_id", ForeignKey("races.race_id"), nullable=False),
    Column("model_id", ForeignKey("model_versions.model_id"), nullable=False),
    Column("decision_time", UTCDateTime, nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
    Column("data_status", String(30), nullable=False),
    Column("input_snapshot_hash", String(64), nullable=False),
)

prediction_entries = Table(
    "prediction_entries", metadata,
    Column("prediction_id", ForeignKey("predictions.prediction_id"), primary_key=True),
    Column("num_pmu", Integer, primary_key=True),
    Column("p_win", Float), Column("p_place", Float),
    Column("p_win_lo", Float), Column("p_win_hi", Float),
    Column("rank", Integer), Column("model_disagreement", Float), Column("ood_score", Float),
    Column("mkt_prob", Float), Column("mkt_odds", Float),
)

prediction_explanations = Table(
    "prediction_explanations", metadata,
    Column("explanation_id", Integer, primary_key=True, autoincrement=True),
    Column("prediction_id", ForeignKey("predictions.prediction_id"), nullable=False),
    Column("num_pmu", Integer, nullable=False),
    Column("factor", String(100), nullable=False),
    Column("contribution", Float),
    Column("direction", String(10)),
    Column("text", Text),
)

detected_patterns = Table(
    "detected_patterns", metadata,
    Column("pattern_id", Integer, primary_key=True, autoincrement=True),
    Column("definition_json", JSON, nullable=False),
    Column("discovery_start", Date), Column("discovery_end", Date),
    Column("n_obs", Integer), Column("effect_size", Float),
    Column("ci_low", Float), Column("ci_high", Float),
    Column("p_value", Float), Column("q_value_fdr", Float),
    Column("oos_effect", Float), Column("oos_ci_low", Float), Column("oos_ci_high", Float),
    Column("status", String(30), nullable=False),
    Column("history_json", JSON),
)

research_documents = Table(
    "research_documents", metadata,
    Column("doc_id", Integer, primary_key=True, autoincrement=True),
    Column("url", String(1000), nullable=False),
    Column("source_name", String(200)),
    Column("published_at", UTCDateTime),
    Column("collected_at", UTCDateTime, nullable=False),
    Column("reliability", String(20)),
    Column("expires_at", UTCDateTime),
    Column("content_sha256", String(64)),
    Column("summary", Text),
    Column("structured_json", JSON),
)


stored_files = Table(
    "stored_files", metadata,
    Column("key", String(300), primary_key=True),
    Column("content", LargeBinary, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("content_type", String(80), nullable=False),
    Column("created_at", UTCDateTime, nullable=False),
)


SUPPORTED_SCHEMES = ("sqlite:///", "postgresql+psycopg://", "postgresql://")


class DatabaseUrlError(ValueError):
    """Adresse de base mal formée (la valeur n'est jamais affichée : elle contient un secret)."""


def make_engine(database_url: str) -> Engine:
    if not database_url.startswith(SUPPORTED_SCHEMES):
        scheme = database_url.split(":", 1)[0] if ":" in database_url else "?"
        hint = (" Il s'agit de l'URL du projet Supabase : il faut la chaîne de connexion "
                "PostgreSQL (Supabase → Connect → Session pooler)." if scheme == "https" else "")
        raise DatabaseUrlError(
            f"DATABASE_URL invalide (schéma « {scheme} »). Format attendu : "
            f"postgresql+psycopg://utilisateur:motdepasse@hote:5432/base?sslmode=require.{hint}")
    if database_url.startswith("postgresql://"):
        database_url = "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
    if database_url.startswith("sqlite:///"):
        path = database_url.removeprefix("sqlite:///")
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url, future=True)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    return engine


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
