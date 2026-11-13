"""Courses du jour et détail des partants."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import streamlit as st

from horse_intelligence.frontend.components.cards import notice, status_badge
from horse_intelligence.frontend.state import format_dt, get_context

DISCIPLINE_LABELS = {"trot_attele": "Trot attelé", "trot_monte": "Trot monté",
                     "plat": "Plat", "obstacle": "Obstacle"}


@st.cache_data(ttl=120, show_spinner=False)
def _dates() -> list[date]:
    return get_context().repo.available_dates()


@st.cache_data(ttl=60, show_spinner=False)
def _races(day: date) -> pd.DataFrame:
    return get_context().repo.races_on(day)


def _race_detail(row: pd.Series) -> None:
    repo = get_context().repo
    race_id = int(row["race_id"])
    st.markdown(f"### R{row['reunion_num']}C{row['course_num']} — {row['label'] or ''}")
    st.markdown(status_badge(row.get("data_status")), unsafe_allow_html=True)
    st.caption(f"{row['racecourse'] or '?'} · "
               f"{DISCIPLINE_LABELS.get(row['discipline'], row['discipline'] or '?')} · "
               f"{row['distance_m'] or '?'} m · départ {format_dt(row['scheduled_start_utc'])}")

    quality = repo.quality_for(str(row["race_key"]))
    if quality and quality["checks_json"]:
        with st.expander("Contrôles qualité", expanded=quality["status"] == "insufficient"):
            for c in quality["checks_json"]:
                icon = "⛔" if c["severity"] == "error" else "⚠️"
                st.write(f"{icon} {c['message']}")

    entries = repo.entries_as_of(race_id)
    if entries.empty:
        notice("Partants non disponibles pour cette course.")
        return
    now = datetime.now(UTC)
    odds = repo.odds_as_of(race_id, now, kinds=("reference", "live"))
    table = entries[["num_pmu", "horse", "rider", "trainer", "age", "sex", "draw",
                     "weight_carried_kg", "status", "form_string", "observed_at"]].copy()
    odds_cols: dict[str, str] = {}
    if not odds.empty:
        odds = odds[odds["bet_type"] == "SIMPLE_GAGNANT"]
        for (kind, pool), sub in odds.groupby(["odds_kind", "pool"]):
            col = f"{kind}_{pool}"
            odds_cols[col] = (f"Cote {'réf.' if kind == 'reference' else 'avant départ'} "
                              f"({'Internet' if pool == 'INTERNET' else 'points de vente'})")
            table = table.merge(sub[["num_pmu", "odds"]].rename(columns={"odds": col}),
                                on="num_pmu", how="left")
    if bool(row["result_final"]):
        res = repo.results_for(race_id)
        if not res.empty:
            table = table.merge(res[["num_pmu", "finish_position"]], on="num_pmu", how="left")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "num_pmu": "N°", "horse": "Cheval", "rider": "Jockey / driver", "trainer": "Entraîneur",
        "age": "Âge", "sex": "Sexe", "draw": "Corde", "weight_carried_kg": "Poids porté (kg)",
        "status": "Statut", "form_string": "Musique (source)", "observed_at": "Observé le",
        "finish_position": "Arrivée", **odds_cols})
    st.caption("Cotes simple gagnant. Les cotes relevées après le départ sont exclues de cet "
               "affichage et de toute analyse prédictive ; la cote de référence est datée de sa "
               "publication (environ 20 à 30 min avant le départ). La musique est celle publiée "
               "par la source au moment de l'observation.")


def render() -> None:
    st.title("Courses du jour")
    dates = _dates()
    if not dates:
        notice("Aucune course en base. Lancez une collecte : python scripts/refresh_data.py")
        return
    today = date.today()
    default = today if today in dates else dates[0]
    day = st.date_input("Date", value=default, min_value=dates[-1], max_value=dates[0],
                        format="DD/MM/YYYY")
    races = _races(day)
    if races.empty:
        notice("Aucune course collectée pour cette date.")
        return

    c1, c2, c3, c4 = st.columns(4)
    hippos = sorted(races["racecourse"].dropna().unique())
    sel_h = c1.multiselect("Hippodrome", hippos)
    discs = sorted(races["discipline"].dropna().unique())
    sel_d = c2.multiselect("Discipline", discs,
                           format_func=lambda d: DISCIPLINE_LABELS.get(d, d))
    reunions = sorted(races["reunion_num"].unique())
    sel_r = c3.multiselect("Réunion", reunions, format_func=lambda r: f"R{r}")
    max_runners = int(races["declared_runners"].fillna(0).max() or 0)
    min_runners = c4.number_input("Partants min.", min_value=0, max_value=max(max_runners, 1),
                                  value=0)

    view = races
    if sel_h:
        view = view[view["racecourse"].isin(sel_h)]
    if sel_d:
        view = view[view["discipline"].isin(sel_d)]
    if sel_r:
        view = view[view["reunion_num"].isin(sel_r)]
    if min_runners:
        view = view[view["declared_runners"].fillna(0) >= min_runners]

    st.dataframe(
        view[["reunion_num", "course_num", "racecourse", "label", "discipline", "distance_m",
              "declared_runners", "scheduled_start_utc", "status", "data_status"]],
        hide_index=True, width="stretch",
        column_config={"reunion_num": "R", "course_num": "C", "racecourse": "Hippodrome",
                       "label": "Course", "discipline": "Discipline", "distance_m": "Distance",
                       "declared_runners": "Partants", "scheduled_start_utc": "Départ (UTC)",
                       "status": "Statut", "data_status": "Données"})
    if view.empty:
        return
    labels = {int(r["race_id"]): f"R{r['reunion_num']}C{r['course_num']} — {r['label'] or ''}"
              for r in view.to_dict("records")}
    chosen = st.selectbox("Course", list(labels), format_func=labels.__getitem__)
    _race_detail(view[view["race_id"] == chosen].iloc[0])
