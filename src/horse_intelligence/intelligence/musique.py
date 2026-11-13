"""Analyse de la « musique » PMU (résumé des dernières performances).

Format observé : `2a0a8a2a1a1a0a(25)2a1a` — le plus récent d'abord. Chaque jeton est
un résultat (1-9 ; 0 = au-delà de la 9e place ; D = disqualifié, A = arrêté,
T = tombé, R = rétrogradé…) suivi d'une lettre de discipline (a = attelé,
m = monté, p = plat, h = haies, s = steeple, c = cross). `(25)` marque un
changement d'année.

L'audit du 2026-10-30 (intelligence/leakage_audit.py) a établi que la musique
fournie reflète l'état AVANT la course concernée.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN = re.compile(r"(\d|[A-Z])([a-z])")
_YEAR = re.compile(r"\(\d{2}\)")

DISCIPLINE_LETTER = {"trot_attele": "a", "trot_monte": "m", "plat": "p"}
OBSTACLE_LETTERS = {"h", "s", "c"}
UNPLACED_POSITION = 10.0


@dataclass(frozen=True)
class Run:
    position: float | None  # None = incident (disqualifié, arrêté, tombé…)
    discipline: str


def parse_musique(text: object) -> list[Run]:
    if not isinstance(text, str) or not text:  # None / NaN pandas
        return []
    runs: list[Run] = []
    for result, disc in _TOKEN.findall(_YEAR.sub(" ", text)):
        if result.isdigit():
            pos = float(result) if result != "0" else UNPLACED_POSITION
            runs.append(Run(pos, disc))
        else:
            runs.append(Run(None, disc))
    return runs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def musique_features(text: object, discipline: str | None) -> dict[str, float]:
    runs = parse_musique(text)
    last6 = runs[:6]
    # un incident compte comme une place au-delà de la 9e pour les moyennes
    pos6 = [r.position if r.position is not None else UNPLACED_POSITION for r in last6]
    pos3 = pos6[:3]
    prev3 = pos6[3:6]
    letter = DISCIPLINE_LETTER.get(discipline or "")
    if discipline == "obstacle":
        same = [r for r in runs if r.discipline in OBSTACLE_LETTERS]
    else:
        same = [r for r in runs if letter is not None and r.discipline == letter]
    return {
        "mus_n": float(len(runs)),
        "mus_last_pos": pos6[0] if pos6 else math.nan,
        "mus_mean_pos_3": _mean(pos3),
        "mus_mean_pos_6": _mean(pos6),
        "mus_trend": _mean(prev3) - _mean(pos3) if pos3 and prev3 else math.nan,
        "mus_wins_6": float(sum(1 for r in last6 if r.position == 1.0)),
        "mus_top3_6": float(sum(1 for r in last6 if r.position is not None and r.position <= 3)),
        "mus_incidents_6": float(sum(1 for r in last6 if r.position is None)),
        "mus_same_disc_share": len(same) / len(runs) if runs else math.nan,
    }
