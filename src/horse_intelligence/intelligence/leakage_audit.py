"""Audit empirique : les agrégats de la source sont-ils publiés AVANT la course ?

Si la musique ou les cumuls de carrière étaient mis à jour avec la course concernée,
(quasi) tous les gagnants auraient « 1 » comme dernier résultat et au moins une
victoire en carrière. On compare donc gagnants et perdants : un écart modéré est
attendu (les bons chevaux gagnent plus souvent), une quasi-certitude trahirait une fuite.

Constat du 2026-10-30 (52 courses) : 25,5 % des gagnants ont « 1 » en tête de musique
(11 % des perdants) et 11 gagnants sur 51 n'ont aucune victoire en carrière.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

MAX_WINNER_LEADING_ONE = 0.80
MIN_WINNERS = 30
_FIRST = re.compile(r"\s*(?:\(\d{2}\)\s*)?([0-9A-Z])")


@dataclass(frozen=True)
class AggregatesAudit:
    n_winners: int
    winner_leading_one: float
    loser_leading_one: float
    winners_without_career_win: int
    passed: bool
    message: str


def _first_token(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    m = _FIRST.match(text)
    return m.group(1) if m else None


def audit_source_aggregates(frame: pd.DataFrame) -> AggregatesAudit:
    """`frame` : une ligne par partant avec form_string, career_wins, finish_position."""
    df = frame[frame["form_string"].notna() & frame["finish_position"].notna()]
    first = df["form_string"].map(_first_token)
    winners = df["finish_position"] == 1
    losers = df["finish_position"] >= 4
    n_w = int(winners.sum())
    w1 = float((first[winners] == "1").mean()) if n_w else float("nan")
    l1 = float((first[losers] == "1").mean()) if losers.any() else float("nan")
    no_win = int((df.loc[winners, "career_wins"] == 0).sum())
    if n_w < MIN_WINNERS:
        return AggregatesAudit(n_w, w1, l1, no_win, False,
                               f"Échantillon insuffisant ({n_w} gagnants < {MIN_WINNERS}).")
    passed = w1 < MAX_WINNER_LEADING_ONE and no_win > 0
    msg = ("Agrégats publiés avant la course (aucun signe de mise à jour post-course)."
           if passed else "Suspicion de fuite : agrégats probablement mis à jour après course.")
    return AggregatesAudit(n_w, w1, l1, no_win, passed, msg)
