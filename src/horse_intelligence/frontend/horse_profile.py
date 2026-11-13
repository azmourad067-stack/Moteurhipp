"""Fiche cheval."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from horse_intelligence.frontend.components.cards import EMERALD, metric_card, notice
from horse_intelligence.frontend.state import get_context


def render() -> None:
    st.title("Fiche cheval")
    repo = get_context().repo
    query = st.text_input("Rechercher un cheval", placeholder="Nom du cheval")
    if not query.strip():
        notice("Saisissez au moins une partie du nom.")
        return
    found = repo.search_horses(query)
    if found.empty:
        notice("Aucun cheval correspondant en base.")
        return
    options = {int(r["horse_id"]): f"{r['display_name']} ({r['birth_year'] or 'année ?'})"
               for r in found.to_dict("records")}
    horse_id = st.selectbox("Cheval", list(options), format_func=options.__getitem__)
    hist = repo.horse_history(horse_id)
    if hist.empty:
        notice("Aucune course enregistrée pour ce cheval.")
        return

    runs = hist[hist["status"].fillna("").str.upper() != "NON_PARTANT"]
    finished = runs.dropna(subset=["finish_position"])
    wins = int((finished["finish_position"] == 1).sum())
    places = int((finished["finish_position"] <= 3).sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Courses en base", str(len(runs)))
    with c2:
        metric_card("Avec résultat", str(len(finished)))
    with c3:
        metric_card("Victoires", str(wins))
    with c4:
        metric_card("Top 3", str(places))
    st.caption("Statistiques calculées uniquement sur les courses présentes dans notre base, "
               "pas sur la carrière complète.")

    if not finished.empty:
        chron = finished.sort_values("race_date")
        fig = go.Figure(go.Scatter(x=chron["race_date"], y=chron["finish_position"],
                                   mode="lines+markers", line={"color": EMERALD}))
        fig.update_yaxes(autorange="reversed", title="Place")
        fig.update_layout(height=280, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(hist[["race_date", "racecourse", "discipline", "distance_m", "label",
                       "num_pmu", "rider", "trainer", "draw", "weight_kg", "status",
                       "finish_position", "incident"]],
                 hide_index=True, width="stretch")
