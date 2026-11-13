"""Tableau de bord."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from horse_intelligence.frontend.components.cards import metric_card, notice
from horse_intelligence.frontend.state import format_dt, get_context
from horse_intelligence.models.registry import latest_model
from horse_intelligence.monitoring.data_monitor import collection_health
from horse_intelligence.monitoring.model_monitor import live_tracking
from horse_intelligence.services.training import latest_report


@st.cache_data(ttl=60, show_spinner=False)
def _counts() -> dict[str, int]:
    return get_context().repo.counts()


@st.cache_data(ttl=60, show_spinner=False)
def _last_update() -> datetime | None:
    return get_context().repo.last_update()


@st.cache_data(ttl=60, show_spinner=False)
def _sources() -> pd.DataFrame:
    return get_context().repo.source_statuses()


@st.cache_data(ttl=60, show_spinner=False)
def _runs() -> pd.DataFrame:
    return get_context().repo.recent_runs(10)


@st.cache_data(ttl=300, show_spinner=False)
def _live() -> dict[str, Any]:
    return live_tracking(get_context().repo.engine)


@st.cache_data(ttl=60, show_spinner=False)
def _health() -> dict[str, Any]:
    return collection_health(get_context().repo.engine)


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def render() -> None:
    st.title("Tableau de bord")
    ctx = get_context()
    counts = _counts()

    if ctx.settings.using_default_database:
        notice("Base locale SQLite (éphémère sur Streamlit Cloud). Configurez DATABASE_URL "
               "pour une base persistante — voir README.")

    cols = st.columns(4)
    with cols[0]:
        metric_card("Courses en base", _fmt(counts["races"]))
    with cols[1]:
        metric_card("Chevaux étudiés", _fmt(counts["horses"]))
    with cols[2]:
        metric_card("Résultats officiels", _fmt(counts["results"]))
    with cols[3]:
        metric_card("Dernière collecte", format_dt(_last_update()))

    st.write("")
    engine = ctx.repo.engine
    rep = latest_report(engine)
    model = latest_model(engine, "lgbm_market")
    live = _live()
    health = _health()
    cols = st.columns(4)
    with cols[0]:
        metric_card("Modèle principal",
                    f"{model['status']}" if model else "aucun",
                    f"LightGBM + marché v{model['version']}" if model else
                    "lancez scripts/train_models.py")
    with cols[1]:
        if rep and "lgbm_market" in rep.get("vs_reference", {}):
            d = rep["vs_reference"]["lgbm_market"]
            metric_card("Validation glissante", f"{d['mean_diff']:+.4f}",
                        f"Δ log-loss vs marché [{d['ci_low']:+.3f} ; {d['ci_high']:+.3f}]")
        else:
            metric_card("Validation glissante", "—", "aucune évaluation")
    with cols[2]:
        vm = live.get("vs_market")
        metric_card("Conditions réelles", f"{live['races']} courses",
                    f"Δ log-loss vs marché {vm['mean_diff']:+.4f}" if vm and live["races"]
                    else "prédictions figées avant départ")
    with cols[3]:
        metric_card("Collecte", "en retard" if health["stale"] else "à jour",
                    health["message"])

    st.subheader("État des sources")
    sources = _sources()
    if not sources.empty:
        st.dataframe(
            sources[["source_key", "name", "enabled", "status_audit", "last_status",
                     "last_checked_at", "last_message"]],
            hide_index=True, width="stretch",
            column_config={"source_key": "Clé", "name": "Source", "enabled": "Activée",
                           "status_audit": "Audit", "last_status": "Dernier état",
                           "last_checked_at": "Vérifiée le", "last_message": "Message"},
        )
    st.caption("Tester les sources : page Configuration.")

    st.subheader("Dernières collectes")
    runs = _runs()
    if runs.empty:
        notice("Aucune collecte enregistrée. Lancez : python scripts/refresh_data.py")
    else:
        st.dataframe(runs, hide_index=True, width="stretch")
