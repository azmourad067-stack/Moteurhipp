"""Enregistrements normalisés, indépendants du fournisseur."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class RaceRecord:
    race_date: date
    reunion_num: int
    course_num: int
    racecourse_code: str
    racecourse_name: str | None
    country: str | None
    label: str | None
    discipline: str | None
    discipline_raw: str | None
    specialite_raw: str | None
    distance_m: int | None
    corde: str | None
    prize_eur: float | None
    declared_runners: int | None
    sex_condition: str | None
    conditions_text: str | None
    scheduled_start_utc: datetime | None
    status: str | None
    result_final: bool
    going_label: str | None
    penetrometer: str | None

    @property
    def key(self) -> str:
        return f"{self.race_date.isoformat()}-R{self.reunion_num}C{self.course_num}"


@dataclass(frozen=True)
class EntryRecord:
    num_pmu: int
    horse_name: str
    horse_canonical: str
    birth_year: int | None
    breed: str | None
    age: int | None
    sex: str | None
    rider_name: str | None
    trainer_name: str | None
    owner_name: str | None
    draw: int | None
    weight_kg: float | None
    weight_carried_kg: float | None
    handicap_distance_m: int | None
    blinkers: str | None
    shoes: str | None
    status: str | None
    form_string: str | None
    career_starts: int | None
    career_wins: int | None
    career_places: int | None
    career_earnings: float | None

    @property
    def is_runner(self) -> bool:
        return (self.status or "").upper() != "NON_PARTANT"


@dataclass(frozen=True)
class ResultRecord:
    num_pmu: int
    finish_position: int | None
    incident: str | None
    time_s: float | None
    reduction_km_s: float | None


@dataclass(frozen=True)
class OddsRecord:
    num_pmu: int
    bet_type: str
    odds_kind: str
    odds: float
    published_at: datetime | None
    pool: str = "OFFLINE"            # masse de paris : OFFLINE (points de vente) | INTERNET
    payload_id: int | None = None


@dataclass(frozen=True)
class PayoutRecord:
    bet_type: str
    combination: str | None
    payout_per_eur: float | None
    pool: str = "OFFLINE"
    payload_id: int | None = None


@dataclass
class RaceSnapshot:
    """Tout ce qui a été observé pour une course lors d'une collecte."""

    race: RaceRecord
    observed_at: datetime
    entries: list[EntryRecord] = field(default_factory=list)
    results: list[ResultRecord] = field(default_factory=list)
    odds: list[OddsRecord] = field(default_factory=list)
    payouts: list[PayoutRecord] = field(default_factory=list)

    @property
    def observed_after_start(self) -> bool:
        start = self.race.scheduled_start_utc
        return start is not None and self.observed_at >= start
