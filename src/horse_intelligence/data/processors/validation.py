"""Contrôles de qualité d'un instantané de course.

Statuts : `verified` (aucun problème), `partial` (avertissements),
`insufficient` (erreur bloquante — aucune prédiction ne doit être produite).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from horse_intelligence.data.schemas.domain import RaceSnapshot


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class DataStatus(StrEnum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"


STATUS_LABELS = {
    DataStatus.VERIFIED: "Données vérifiées",
    DataStatus.PARTIAL: "Données partielles",
    DataStatus.INSUFFICIENT: "Données insuffisantes",
    DataStatus.UNAVAILABLE: "Analyse indisponible",
}

WEIGHT_RANGE_KG = (40.0, 80.0)
DISTANCE_RANGE_M = (800, 8000)


@dataclass(frozen=True)
class Check:
    code: str
    severity: Severity
    message: str


@dataclass
class QualityReport:
    scope_ref: str
    checks: list[Check]

    @property
    def status(self) -> DataStatus:
        if any(c.severity is Severity.ERROR for c in self.checks):
            return DataStatus.INSUFFICIENT
        if self.checks:
            return DataStatus.PARTIAL
        return DataStatus.VERIFIED

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(c) for c in self.checks]


def validate_snapshot(snap: RaceSnapshot) -> QualityReport:
    checks: list[Check] = []
    race = snap.race

    def add(code: str, sev: Severity, msg: str) -> None:
        checks.append(Check(code, sev, msg))

    if race.scheduled_start_utc is None:
        add("missing_start_time", Severity.ERROR,
            "Heure de départ inconnue : impossible de garantir l'absence de fuite temporelle.")
    if race.distance_m is None:
        add("missing_distance", Severity.WARNING, "Distance absente.")
    elif not DISTANCE_RANGE_M[0] <= race.distance_m <= DISTANCE_RANGE_M[1]:
        add("implausible_distance", Severity.ERROR,
            f"Distance non plausible : {race.distance_m} m.")
    if race.discipline is None:
        add("unknown_discipline", Severity.WARNING,
            f"Discipline non reconnue ({race.specialite_raw or race.discipline_raw}).")

    if not snap.entries:
        add("no_entries", Severity.ERROR, "Aucun partant identifié.")
        return QualityReport(race.key, checks)

    dup = [n for n, c in Counter(e.num_pmu for e in snap.entries).items() if c > 1]
    if dup:
        add("duplicate_numbers", Severity.ERROR, f"Numéros en double : {sorted(dup)}.")

    runners = [e for e in snap.entries if e.is_runner]
    non_runners = len(snap.entries) - len(runners)
    if len(runners) < 2:
        add("too_few_runners", Severity.ERROR, f"Seulement {len(runners)} partant(s).")
    if non_runners:
        add("non_runners", Severity.WARNING, f"{non_runners} non-partant(s) détecté(s).")
    if race.declared_runners is not None and race.declared_runners != len(snap.entries):
        add("runner_count_mismatch", Severity.WARNING,
            f"{len(snap.entries)} participants reçus pour {race.declared_runners} déclarés.")

    missing_rider = [e.num_pmu for e in runners if not e.rider_name]
    if missing_rider:
        add("missing_rider", Severity.WARNING, f"Jockey/driver absent : n° {missing_rider}.")
    missing_trainer = [e.num_pmu for e in runners if not e.trainer_name]
    if missing_trainer:
        add("missing_trainer", Severity.WARNING, f"Entraîneur absent : n° {missing_trainer}.")
    bad_weight = [e.num_pmu for e in runners
                  if any(w is not None and not WEIGHT_RANGE_KG[0] <= w <= WEIGHT_RANGE_KG[1]
                         for w in (e.weight_kg, e.weight_carried_kg))]
    if bad_weight:
        add("implausible_weight", Severity.ERROR,
            f"Poids non plausibles (hypothèse d'unité à revoir ?) : n° {bad_weight}.")

    bad_odds = [o.num_pmu for o in snap.odds if o.odds < 1.0]
    if bad_odds:
        add("invalid_odds", Severity.ERROR, f"Cotes < 1 : n° {bad_odds}.")
    if not any(o.odds_kind in ("reference", "live") for o in snap.odds):
        add("no_pre_race_odds", Severity.WARNING, "Aucune cote disponible avant le départ.")

    if snap.race.result_final and snap.observed_after_start and not snap.payouts:
        add("no_payouts", Severity.WARNING, "Rapports définitifs non disponibles.")

    if snap.results:
        if not snap.observed_after_start:
            add("result_before_start", Severity.ERROR,
                "Résultat observé avant l'heure de départ : donnée incohérente.")
        positions = [r.finish_position for r in snap.results if r.finish_position]
        if any(p > len(snap.entries) for p in positions):
            add("position_out_of_range", Severity.ERROR, "Position d'arrivée hors limites.")
        ties = [p for p, c in Counter(positions).items() if c > 1]
        if ties:
            add("dead_heat", Severity.WARNING, f"Ex æquo aux places {sorted(ties)}.")
        known = {e.num_pmu for e in snap.entries}
        unknown = [r.num_pmu for r in snap.results if r.num_pmu not in known]
        if unknown:
            add("result_unknown_runner", Severity.ERROR, f"Résultat pour n° inconnus {unknown}.")

    return QualityReport(race.key, checks)
