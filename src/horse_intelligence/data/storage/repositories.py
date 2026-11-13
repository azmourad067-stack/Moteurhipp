"""Accès aux données : écriture traçable et lectures « as-of »."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import Engine, Table, and_, delete, func, insert, select, update
from sqlalchemy.engine import Connection, CursorResult

from horse_intelligence.config.settings import Settings
from horse_intelligence.data.collectors.base import HealthReport, RawResponse
from horse_intelligence.data.processors.cleaning import canonical_name
from horse_intelligence.data.processors.validation import QualityReport
from horse_intelligence.data.schemas.domain import (
    EntryRecord,
    PayoutRecord,
    RaceRecord,
    RaceSnapshot,
)
from horse_intelligence.data.storage import database as db
from horse_intelligence.data.storage.raw_store import RawStore

_ENTRY_HASH_FIELDS = (
    "horse_canonical", "rider_name", "trainer_name", "owner_name", "draw", "weight_kg",
    "weight_carried_kg",
    "handicap_distance_m", "blinkers", "shoes", "status", "form_string", "career_starts",
    "career_wins", "career_places", "career_earnings",
)


def _entry_hash(e: EntryRecord) -> str:
    payload = {k: getattr(e, k) for k in _ENTRY_HASH_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _pk(result: CursorResult[Any]) -> int:
    key = result.inserted_primary_key
    if key is None:
        raise RuntimeError("clé primaire non renvoyée par l'insertion")
    return int(key[0])


class Repository:
    def __init__(self, engine: Engine, raw_store: RawStore) -> None:
        self.engine = engine
        self.raw_store = raw_store

    # ------------------------------------------------------------------ sources
    def sync_sources(self, settings: Settings) -> None:
        with self.engine.begin() as conn:
            for key, src in settings.sources.items():
                values = {"name": src.name, "base_url": src.base_url, "kind": src.kind,
                          "enabled": src.enabled, "status_audit": src.status_audit}
                exists = conn.execute(select(db.data_sources.c.source_key).where(
                    db.data_sources.c.source_key == key)).first()
                if exists:
                    conn.execute(update(db.data_sources)
                                 .where(db.data_sources.c.source_key == key).values(**values))
                else:
                    conn.execute(insert(db.data_sources).values(source_key=key, **values))

    def record_health(self, report: HealthReport) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(db.data_sources)
                         .where(db.data_sources.c.source_key == report.source)
                         .values(last_status=report.status.value,
                                 last_checked_at=report.checked_at,
                                 last_message=report.message))

    # ---------------------------------------------------------------- brut
    def store_raw(self, raw: RawResponse) -> int:
        path = self.raw_store.write(raw)
        with self.engine.begin() as conn:
            res = conn.execute(insert(db.raw_payloads).values(
                source_key=raw.source, endpoint=raw.endpoint, url=raw.url,
                params_json=raw.params, fetched_at=raw.fetched_at, http_status=raw.http_status,
                sha256=raw.sha256, size_bytes=len(raw.body), storage_uri=str(path),
            ))
            return _pk(res)

    # ---------------------------------------------------------------- runs
    def start_run(self, target_date: date) -> int:
        with self.engine.begin() as conn:
            res = conn.execute(insert(db.collection_runs).values(
                target_date=target_date, started_at=datetime.now(UTC), status="running"))
            return _pk(res)

    def finish_run(self, run_id: int, status: str, races_seen: int, races_ok: int,
                   errors: list[dict[str, str]]) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(db.collection_runs).where(db.collection_runs.c.run_id == run_id)
                         .values(finished_at=datetime.now(UTC), status=status,
                                 races_seen=races_seen, races_ok=races_ok, errors_json=errors))

    def day_completed(self, target_date: date) -> bool:
        """Vrai si une collecte réussie a eu lieu après la fin de la journée cible."""
        with self.engine.connect() as conn:
            rows = conn.execute(select(db.collection_runs.c.started_at).where(and_(
                db.collection_runs.c.target_date == target_date,
                db.collection_runs.c.status == "success"))).all()
        end_of_day = datetime.combine(target_date, datetime.max.time(), tzinfo=UTC)
        return any(r.started_at > end_of_day for r in rows)

    # ---------------------------------------------------------------- référentiels
    @staticmethod
    def _get_or_create(conn: Connection, table: Table, pk: str, key: dict[str, Any],
                       extra: dict[str, Any]) -> int:
        cond = and_(*[table.c[k] == v if v is not None else table.c[k].is_(None)
                      for k, v in key.items()])
        row = conn.execute(select(table.c[pk]).where(cond)).first()
        if row:
            return int(row[0])
        res = conn.execute(insert(table).values(**key, **extra))
        return _pk(res)

    def _racecourse(self, conn: Connection, race: RaceRecord) -> int:
        return self._get_or_create(conn, db.racecourses, "racecourse_id",
                                   {"code": race.racecourse_code},
                                   {"name": race.racecourse_name, "country": race.country})

    def upsert_race(self, race: RaceRecord, observed_at: datetime, source_key: str,
                    payload_id: int | None) -> int:
        with self.engine.begin() as conn:
            rc_id = self._racecourse(conn, race)
            values = {k: v for k, v in asdict(race).items()
                      if k in db.races.c and k not in ("race_date", "reunion_num", "course_num")}
            values.update(racecourse_id=rc_id, source_key=source_key,
                          last_observed_at=observed_at, payload_id=payload_id)
            key = and_(db.races.c.race_date == race.race_date,
                       db.races.c.reunion_num == race.reunion_num,
                       db.races.c.course_num == race.course_num)
            row = conn.execute(select(db.races.c.race_id).where(key)).first()
            if row:
                conn.execute(update(db.races).where(key).values(**values))
                race_id = int(row[0])
            else:
                res = conn.execute(insert(db.races).values(
                    race_date=race.race_date, reunion_num=race.reunion_num,
                    course_num=race.course_num, first_observed_at=observed_at, **values))
                race_id = _pk(res)
            if race.going_label or race.penetrometer:
                conn.execute(insert(db.racecourse_conditions).values(
                    race_id=race_id, going_label=race.going_label,
                    penetrometer=race.penetrometer, observed_at=observed_at))
            return race_id

    # ---------------------------------------------------------------- instantanés
    # Chaque écriture regroupe ses requêtes : la base distante (Supabase, ~0,1 s par
    # aller-retour depuis GitHub Actions) impose de ne pas interroger ligne par ligne.
    @staticmethod
    def _resolve(conn: Connection, table: Table, pk: str, keys: Sequence[str],
                 wanted: dict[tuple[Any, ...], dict[str, Any]]) -> dict[tuple[Any, ...], int]:
        """Identifiants des lignes de référentiel `wanted` (clé -> valeurs d'insertion),
        créées si absentes. La première colonne de `keys` sert de filtre."""
        if not wanted:
            return {}
        cols = [table.c[k] for k in keys]

        def fetch() -> dict[tuple[Any, ...], int]:
            rows = conn.execute(select(table.c[pk], *cols).where(
                cols[0].in_({k[0] for k in wanted}))).all()
            return {tuple(r[1:]): int(r[0]) for r in rows}

        found = fetch()
        missing = [{**dict(zip(keys, k, strict=True)), **extra}
                   for k, extra in wanted.items() if k not in found]
        if missing:
            conn.execute(insert(table), missing)
            found = fetch()
        return {k: found[k] for k in wanted}

    def _people(self, conn: Connection, table: Table, pk: str,
                names: Sequence[str | None]) -> dict[str, int]:
        wanted: dict[tuple[Any, ...], dict[str, Any]] = {}
        for name in names:
            canon = canonical_name(name)
            if canon is not None and name is not None:
                wanted.setdefault((canon,), {"display_name": name})
        return {k[0]: v for k, v in self._resolve(conn, table, pk, ["canonical_name"],
                                                    wanted).items()}

    @staticmethod
    def _person_id(ids: dict[str, int], name: str | None) -> int | None:
        canon = canonical_name(name)
        return ids.get(canon) if canon is not None else None

    @staticmethod
    def _existing_payouts(conn: Connection, race_id: int) -> set[tuple[Any, ...]]:
        p = db.payouts
        return {tuple(r) for r in conn.execute(
            select(p.c.pool, p.c.bet_type, p.c.combination).where(p.c.race_id == race_id))}

    def save_snapshot(self, race_id: int, snap: RaceSnapshot, payload_id: int | None) -> None:
        after = snap.observed_after_start
        with self.engine.begin() as conn:
            horses = self._resolve(conn, db.horses, "horse_id", ["canonical_name", "birth_year"], {
                (e.horse_canonical, e.birth_year):
                    {"display_name": e.horse_name, "breed": e.breed, "sex_last": e.sex}
                for e in snap.entries})
            jockeys = self._people(conn, db.jockeys, "jockey_id",
                                   [e.rider_name for e in snap.entries])
            trainers = self._people(conn, db.trainers, "trainer_id",
                                    [e.trainer_name for e in snap.entries])
            owners = self._people(conn, db.owners, "owner_id",
                                  [e.owner_name for e in snap.entries])

            re_ = db.race_entries
            last_hash = {r.num_pmu: r.content_hash for r in conn.execute(
                select(re_.c.num_pmu, re_.c.content_hash).where(re_.c.race_id == race_id)
                .order_by(re_.c.observed_at, re_.c.entry_id))}
            horse_ids: dict[int, int] = {}
            new_entries = []
            for e in snap.entries:
                horse_ids[e.num_pmu] = horses[(e.horse_canonical, e.birth_year)]
                h = _entry_hash(e)
                if last_hash.get(e.num_pmu) == h:
                    continue  # aucune modification depuis la dernière observation
                last_hash[e.num_pmu] = h
                new_entries.append(dict(
                    race_id=race_id, num_pmu=e.num_pmu, horse_id=horse_ids[e.num_pmu],
                    jockey_id=self._person_id(jockeys, e.rider_name),
                    trainer_id=self._person_id(trainers, e.trainer_name),
                    owner_id=self._person_id(owners, e.owner_name),
                    age=e.age, sex=e.sex, draw=e.draw, weight_kg=e.weight_kg,
                    weight_carried_kg=e.weight_carried_kg,
                    handicap_distance_m=e.handicap_distance_m, blinkers=e.blinkers,
                    shoes=e.shoes, status=e.status, form_string=e.form_string,
                    career_starts=e.career_starts, career_wins=e.career_wins,
                    career_places=e.career_places, career_earnings=e.career_earnings,
                    observed_at=snap.observed_at, observed_after_start=after,
                    content_hash=h, payload_id=payload_id))
            if new_entries:
                conn.execute(insert(re_), new_entries)

            if after and snap.results:
                rr = db.race_results
                existing = {r.num_pmu: r.result_id for r in conn.execute(
                    select(rr.c.num_pmu, rr.c.result_id).where(rr.c.race_id == race_id))}
                new_results = []
                for r in snap.results:
                    if r.num_pmu not in horse_ids:
                        continue
                    values = {"horse_id": horse_ids[r.num_pmu],
                              "finish_position": r.finish_position, "incident": r.incident,
                              "time_s": r.time_s, "reduction_km_s": r.reduction_km_s,
                              "observed_at": snap.observed_at, "payload_id": payload_id}
                    if r.num_pmu in existing:
                        conn.execute(update(rr).where(rr.c.result_id == existing[r.num_pmu])
                                     .values(**values))
                    else:
                        new_results.append({"race_id": race_id, "num_pmu": r.num_pmu, **values})
                if new_results:
                    conn.execute(insert(rr), new_results)

            if snap.odds:
                os_ = db.odds_snapshots
                last_odds = {(r.num_pmu, r.pool, r.bet_type, r.odds_kind): (r.odds, r.published_at)
                             for r in conn.execute(
                    select(os_.c.num_pmu, os_.c.pool, os_.c.bet_type, os_.c.odds_kind,
                           os_.c.odds, os_.c.published_at).where(os_.c.race_id == race_id)
                    .order_by(os_.c.observed_at, os_.c.odds_id))}
                new_odds = []
                for o in snap.odds:
                    key = (o.num_pmu, o.pool, o.bet_type, o.odds_kind)
                    if last_odds.get(key) == (o.odds, o.published_at):
                        continue
                    last_odds[key] = (o.odds, o.published_at)
                    new_odds.append(dict(
                        race_id=race_id, num_pmu=o.num_pmu, bet_type=o.bet_type, pool=o.pool,
                        odds_kind=o.odds_kind, odds=o.odds, published_at=o.published_at,
                        observed_at=snap.observed_at, payload_id=o.payload_id or payload_id))
                if new_odds:
                    conn.execute(insert(os_), new_odds)

            if snap.payouts:
                self._insert_payouts(conn, race_id, snap.payouts, snap.observed_at)

    def _insert_payouts(self, conn: Connection, race_id: int, payouts: Sequence[PayoutRecord],
                        observed_at: datetime) -> int:
        seen = self._existing_payouts(conn, race_id)
        rows = []
        for p in payouts:
            key = (p.pool, p.bet_type, p.combination)
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(race_id=race_id, pool=p.pool, bet_type=p.bet_type,
                             combination=p.combination, payout_per_eur=p.payout_per_eur,
                             observed_at=observed_at, payload_id=p.payload_id))
        if rows:
            conn.execute(insert(db.payouts), rows)
        return len(rows)

    def save_payouts(self, race_id: int, payouts: Sequence[PayoutRecord],
                     observed_at: datetime) -> int:
        """Ajoute des rapports définitifs (idempotent). Renvoie le nombre de lignes ajoutées."""
        with self.engine.begin() as conn:
            return self._insert_payouts(conn, race_id, payouts, observed_at)

    def purge_post_start_odds(self, before: date) -> int:
        """Supprime les cotes relevées après le départ des courses antérieures à `before`.

        Elles ne servent jamais (exclues de toute décision et de l'apprentissage) et les
        rapports officiels sont conservés à part : place libérée sur la base gratuite.
        """
        o, r = db.odds_snapshots, db.races
        old_races = select(r.c.race_id).where(r.c.race_date < before)
        with self.engine.begin() as conn:
            res = conn.execute(delete(o).where(and_(o.c.odds_kind == "post_start",
                                                    o.c.race_id.in_(old_races))))
        return int(res.rowcount or 0)

    def has_results(self, race_date: date, reunion_num: int, course_num: int) -> bool:
        r, res = db.races, db.race_results
        with self.engine.connect() as conn:
            row = conn.execute(select(res.c.result_id).join(r, r.c.race_id == res.c.race_id)
                               .where(and_(r.c.race_date == race_date,
                                           r.c.reunion_num == reunion_num,
                                           r.c.course_num == course_num)).limit(1)).first()
        return row is not None

    def races_missing_payouts(self, pool: str, start: date, end: date) -> pd.DataFrame:
        r, p = db.races, db.payouts
        has = select(p.c.race_id).where(p.c.pool == pool).distinct()
        return self._df(select(r.c.race_id, r.c.race_date, r.c.reunion_num, r.c.course_num)
                        .where(and_(r.c.result_final.is_(True), r.c.race_date >= start,
                                    r.c.race_date <= end, r.c.race_id.not_in(has)))
                        .order_by(r.c.race_date, r.c.reunion_num, r.c.course_num))

    def save_quality_report(self, report: QualityReport, scope: str = "race") -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(db.data_quality_reports).values(
                scope=scope, scope_ref=report.scope_ref, run_at=datetime.now(UTC),
                status=report.status.value, checks_json=report.to_json()))

    # ---------------------------------------------------------------- lectures
    def _df(self, stmt: Any) -> pd.DataFrame:
        with self.engine.connect() as conn:
            result = conn.execute(stmt)
            return pd.DataFrame(result.mappings().all(), columns=list(result.keys()))

    def counts(self) -> dict[str, int]:
        tables = {"races": db.races, "horses": db.horses, "entries": db.race_entries,
                  "results": db.race_results, "jockeys": db.jockeys, "trainers": db.trainers,
                  "odds": db.odds_snapshots}
        with self.engine.connect() as conn:
            return {k: int(conn.execute(select(func.count()).select_from(t)).scalar_one())
                    for k, t in tables.items()}

    def last_update(self) -> datetime | None:
        with self.engine.connect() as conn:
            value = conn.execute(select(func.max(db.raw_payloads.c.fetched_at))).scalar()
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    def source_statuses(self) -> pd.DataFrame:
        return self._df(select(db.data_sources))

    def recent_runs(self, limit: int = 20) -> pd.DataFrame:
        return self._df(select(db.collection_runs)
                        .order_by(db.collection_runs.c.started_at.desc()).limit(limit))

    def available_dates(self) -> list[date]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(db.races.c.race_date).distinct()
                                .order_by(db.races.c.race_date.desc())).all()
        return [r[0] for r in rows]

    def races_on(self, day: date) -> pd.DataFrame:
        r, rc = db.races, db.racecourses
        latest_q = (select(db.data_quality_reports.c.scope_ref,
                           func.max(db.data_quality_reports.c.report_id).label("rid"))
                    .where(db.data_quality_reports.c.scope == "race")
                    .group_by(db.data_quality_reports.c.scope_ref).subquery())
        stmt = (select(r.c.race_id, r.c.race_date, r.c.reunion_num, r.c.course_num,
                       rc.c.name.label("racecourse"), r.c.label, r.c.discipline,
                       r.c.distance_m, r.c.declared_runners, r.c.scheduled_start_utc,
                       r.c.status, r.c.result_final)
                .join(rc, rc.c.racecourse_id == r.c.racecourse_id, isouter=True)
                .where(r.c.race_date == day)
                .order_by(r.c.reunion_num, r.c.course_num))
        df = self._df(stmt)
        if df.empty:
            return df
        q = self._df(select(db.data_quality_reports.c.scope_ref,
                            db.data_quality_reports.c.status.label("data_status"))
                     .join(latest_q, latest_q.c.rid == db.data_quality_reports.c.report_id))
        df["race_key"] = [f"{d.isoformat()}-R{a}C{b}" for d, a, b in
                          zip(df.race_date, df.reunion_num, df.course_num, strict=True)]
        return df.merge(q, how="left", left_on="race_key", right_on="scope_ref").drop(
            columns=["scope_ref"])

    def entries_as_of(self, race_id: int, as_of: datetime | None = None) -> pd.DataFrame:
        """Dernière version connue de chaque partant à l'instant `as_of` (anti-fuite)."""
        e, h, j, t = db.race_entries, db.horses, db.jockeys, db.trainers
        stmt = (select(e, h.c.display_name.label("horse"), j.c.display_name.label("rider"),
                       t.c.display_name.label("trainer"))
                .join(h, h.c.horse_id == e.c.horse_id)
                .join(j, j.c.jockey_id == e.c.jockey_id, isouter=True)
                .join(t, t.c.trainer_id == e.c.trainer_id, isouter=True)
                .where(e.c.race_id == race_id))
        if as_of is not None:
            stmt = stmt.where(e.c.observed_at <= as_of)
        df = self._df(stmt)
        if df.empty:
            return df
        # drop_duplicates et non groupby().last() : last() combinerait des colonnes de
        # versions différentes en sautant les valeurs nulles.
        df = df.sort_values("observed_at").drop_duplicates("num_pmu", keep="last")
        return df.sort_values("num_pmu").reset_index(drop=True)

    def odds_as_of(self, race_id: int, as_of: datetime,
                   kinds: Sequence[str] = ("reference", "live"),
                   pool: str | None = None) -> pd.DataFrame:
        """Cotes réellement disponibles à `as_of`. Les cotes `post_start` sont toujours exclues.

        Disponibilité (`available_at`) :
        - `reference` : date de publication fournie par la source, si elle précède le
          départ (vérifié le 2026-10-30 : publiée ~20 à 30 min avant). Une cote de
          référence récupérée a posteriori reste ainsi utilisable pour un backtest ;
        - sinon : moment de NOTRE observation.
        """
        o = db.odds_snapshots
        kinds = [k for k in kinds if k != "post_start"]
        with self.engine.connect() as conn:
            start = conn.execute(select(db.races.c.scheduled_start_utc)
                                 .where(db.races.c.race_id == race_id)).scalar()
        cond = and_(o.c.race_id == race_id, o.c.odds_kind.in_(kinds))
        if pool is not None:
            cond = and_(cond, o.c.pool == pool)
        df = self._df(select(o).where(cond))
        if df.empty:
            return df
        df["available_at"] = [
            pub if kind == "reference" and pub is not None and start is not None and pub < start
            else obs
            for kind, pub, obs in zip(df["odds_kind"], df["published_at"], df["observed_at"],
                                      strict=True)
        ]
        df = df[df["available_at"] <= as_of]
        return (df.sort_values("available_at")
                .drop_duplicates(["num_pmu", "pool", "bet_type", "odds_kind"], keep="last")
                .reset_index(drop=True))

    def results_for(self, race_id: int) -> pd.DataFrame:
        return self._df(select(db.race_results).where(db.race_results.c.race_id == race_id)
                        .order_by(db.race_results.c.finish_position))

    def quality_for(self, race_key: str) -> dict[str, Any] | None:
        q = db.data_quality_reports
        with self.engine.connect() as conn:
            row = conn.execute(select(q).where(and_(q.c.scope == "race", q.c.scope_ref == race_key))
                               .order_by(q.c.report_id.desc()).limit(1)).mappings().first()
        return dict(row) if row else None

    def search_horses(self, text: str, limit: int = 50) -> pd.DataFrame:
        canon = canonical_name(text) or ""
        h = db.horses
        return self._df(select(h).where(h.c.canonical_name.contains(canon, autoescape=True))
                        .order_by(h.c.canonical_name).limit(limit))

    def horse_history(self, horse_id: int) -> pd.DataFrame:
        """Historique des courses d'un cheval (dernière version de chaque engagement)."""
        e, r, rc, res, j, t = (db.race_entries, db.races, db.racecourses, db.race_results,
                               db.jockeys, db.trainers)
        stmt = (select(r.c.race_id, r.c.race_date, rc.c.name.label("racecourse"),
                       r.c.discipline, r.c.distance_m, r.c.label, e.c.num_pmu, e.c.status,
                       e.c.draw, e.c.weight_kg, e.c.observed_at,
                       j.c.display_name.label("rider"), t.c.display_name.label("trainer"),
                       res.c.finish_position, res.c.incident)
                .join(r, r.c.race_id == e.c.race_id)
                .join(rc, rc.c.racecourse_id == r.c.racecourse_id, isouter=True)
                .join(j, j.c.jockey_id == e.c.jockey_id, isouter=True)
                .join(t, t.c.trainer_id == e.c.trainer_id, isouter=True)
                .join(res, and_(res.c.race_id == e.c.race_id, res.c.num_pmu == e.c.num_pmu),
                      isouter=True)
                .where(e.c.horse_id == horse_id))
        df = self._df(stmt)
        if df.empty:
            return df
        df = df.sort_values("observed_at").drop_duplicates("race_id", keep="last")
        return df.sort_values("race_date", ascending=False).reset_index(drop=True)
