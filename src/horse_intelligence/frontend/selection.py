"""Sélection du jour : le 3e cheval du modèle dans chaque course de trot G.

Règle « 3e du modèle, simple gagnant, trot G » : la seule des règles figées le 10/11/2026
restée positive sur l'année de validation (+6,0 %, IC 95 % [−24 % ; +38 %]). Elle n'est
PAS validée : son intérêt réel est indiscernable du hasard. La page l'applique telle
qu'elle a été testée, sans filtre supplémentaire, pour un suivi à petites mises.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sqlalchemy import select

from horse_intelligence.data.storage import database as db
from horse_intelligence.evaluation.segment_analysis import race_group
from horse_intelligence.frontend.components.cards import notice
from horse_intelligence.frontend.race_analysis import _predict
from horse_intelligence.frontend.state import get_context

PARIS = ZoneInfo("Europe/Paris")
RULE_GROUP = "Trot G"
RULE_RANK = 3  # 3e cheval du modèle
RULE_MODEL = "lgbm_market"


def rule_races(races: pd.DataFrame) -> pd.DataFrame:
    """Courses de la catégorie visée, classées comme dans les tests (libellé + conditions)."""
    if races.empty:
        return races
    groups = [race_group(lab, cond, d) for lab, cond, d in
              zip(races["label"], races["conditions_text"], races["discipline"], strict=True)]
    return races[[g == RULE_GROUP for g in groups]]


def rule_pick(table: pd.DataFrame) -> pd.Series | None:
    """Cheval classé 3e par le modèle (à égalité, l'ordre du tableau départage)."""
    if len(table) < RULE_RANK:
        return None
    return table.sort_values("p_win", ascending=False, kind="stable").iloc[RULE_RANK - 1]


def _races_on(day: date) -> pd.DataFrame:
    r, rc = db.races, db.racecourses
    stmt = (select(r.c.race_id, r.c.reunion_num, r.c.course_num, r.c.label,
                   r.c.conditions_text, r.c.discipline, r.c.distance_m,
                   r.c.scheduled_start_utc, r.c.result_final,
                   rc.c.name.label("racecourse"))
            .join(rc, rc.c.racecourse_id == r.c.racecourse_id, isouter=True)
            .where(r.c.race_date == day).order_by(r.c.scheduled_start_utc))
    with get_context().repo.engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def _paris_time(value: object) -> str:
    if value is None or pd.isna(value):  # type: ignore[call-overload]
        return "—"
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts
    return ts.tz_convert(PARIS).strftime("%H:%M")


def render() -> None:
    st.title("Sélection du jour")
    st.warning("Règle NON validée : « 3e du modèle, simple gagnant, courses de trot G ». "
               "+6 % sur l'année de validation, mais un intervalle de −24 % à +38 % : "
               "impossible de la distinguer du hasard. Petites mises fixes uniquement.")
    repo = get_context().repo
    dates = repo.available_dates()
    if not dates:
        notice("Aucune course en base.")
        return
    today = datetime.now(PARIS).date()
    day = st.date_input("Date", value=today if today in dates else dates[0],
                        min_value=dates[-1], max_value=dates[0], format="DD/MM/YYYY",
                        key="selection_date")
    day = day if isinstance(day, date) else dates[0]
    races = rule_races(_races_on(day))
    if races.empty:
        notice("Aucune course de trot G ce jour-là : rien à jouer.")
        return

    rows = []
    with st.spinner("Calcul des probabilités…"):
        for r in races.to_dict("records"):
            pred = _predict(int(r["race_id"]), day)
            # règle testée avec le modèle « LightGBM + marché » : sans cotes, pas de choix
            tested = pred.primary == RULE_MODEL and not pred.table.empty
            pick = rule_pick(pred.table) if tested else None
            entries = repo.entries_as_of(int(r["race_id"]))
            names = dict(zip(entries["horse_id"], entries["horse"], strict=True)) \
                if not entries.empty else {}
            finish = None
            if pick is not None and bool(r["result_final"]):
                res = repo.results_for(int(r["race_id"]))
                hit = res[res["num_pmu"] == pick["num_pmu"]]
                finish = None if hit.empty else hit["finish_position"].iloc[0]
            rows.append({
                "Départ": _paris_time(r["scheduled_start_utc"]),
                "Course": f"R{r['reunion_num']}C{r['course_num']}",
                "Hippodrome": r["racecourse"] or "?",
                "Distance": f"{r['distance_m'] or '?'} m",
                "N° à jouer": "—" if pick is None else int(pick["num_pmu"]),
                "Cheval": "—" if pick is None else names.get(pick["horse_id"], "?"),
                "P(victoire)": "—" if pick is None else f"{100 * pick['p_win']:.0f} %",
                "Cote réf.": "—" if pick is None or pd.isna(pick["mkt_odds"])
                else f"{pick['mkt_odds']:.1f}",
                "Arrivée": "—" if finish is None or pd.isna(finish) else int(finish),
                "Remarque": "" if pick is not None else (
                    "cotes pas encore publiées : revenir plus tard" if not pred.table.empty
                    else pred.message),
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Heures de Paris. Le classement tient compte des cotes, il peut changer jusqu'au "
        "départ : consultez la page 10 à 20 minutes avant la course (les données sont "
        "rafraîchies toutes les 15 minutes) et jouez le cheval affiché à ce moment, sans "
        "revenir dessus. Les tests ont été réglés aux rapports de la masse Internet du PMU.")
