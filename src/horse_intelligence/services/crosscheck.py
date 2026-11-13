"""Recoupement des arrivées PMU avec une source secondaire indépendante."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from horse_intelligence.data.schemas.open_pmu import OpenPmuRace
from horse_intelligence.data.storage.repositories import Repository


@dataclass(frozen=True)
class ArrivalCheck:
    race_key: str
    status: str  # agree | disagree | missing_primary | not_comparable
    primary: list[int]
    secondary: list[int]
    message: str


def compare_arrivals(repo: Repository, day: date, secondary: list[OpenPmuRace]
                     ) -> list[ArrivalCheck]:
    races = repo.races_on(day)
    checks: list[ArrivalCheck] = []
    for sr in secondary:
        rc = sr.reunion_course_nums
        sec = list(sr.arrivee or [])
        if rc is None or not sec:
            checks.append(ArrivalCheck(f"{day}-{sr.reunion_course}", "not_comparable", [], sec,
                                       "R/C ou arrivée absente de la source secondaire."))
            continue
        key = f"{day.isoformat()}-R{rc[0]}C{rc[1]}"
        match = races[(races["reunion_num"] == rc[0]) & (races["course_num"] == rc[1])] \
            if not races.empty else races
        if match.empty:
            checks.append(ArrivalCheck(key, "missing_primary", [], sec,
                                       "Course absente de la base principale."))
            continue
        res = repo.results_for(int(match["race_id"].iloc[0]))
        res = res.dropna(subset=["finish_position"]).sort_values("finish_position")
        primary = [int(n) for n in res["num_pmu"]][: len(sec)]
        if not primary:
            checks.append(ArrivalCheck(key, "missing_primary", [], sec,
                                       "Résultat absent de la base principale."))
        elif primary == sec[: len(primary)]:
            checks.append(ArrivalCheck(key, "agree", primary, sec, "Arrivées concordantes."))
        else:
            checks.append(ArrivalCheck(key, "disagree", primary, sec,
                                       "Arrivées divergentes : vérification manuelle requise."))
    return checks
