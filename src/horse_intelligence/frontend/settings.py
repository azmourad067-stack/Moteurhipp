"""Configuration (lecture seule en v1) et test des sources."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from horse_intelligence.data.collectors.registry import check_all
from horse_intelligence.data.processors.normalization import PMU_UNITS
from horse_intelligence.frontend.components.cards import notice
from horse_intelligence.frontend.state import get_context
from horse_intelligence.monitoring.logging_config import redact


def render() -> None:
    st.title("Configuration")
    ctx = get_context()
    notice("Configuration en lecture seule : modifiez config/data_sources.yaml et les "
           "secrets (DATABASE_URL, HI_DATA_DIR). Aucune exécution de script depuis l'interface.")

    st.subheader("Sources de données")
    rows = [{"Clé": s.key, "Source": s.name, "Activée": s.enabled, "Gratuite": s.free,
             "Audit": s.status_audit, "Intervalle min. (s)": s.min_interval_s,
             "Délai d'attente (s)": s.timeout_s, "Tentatives": s.max_retries,
             "Conditions": s.terms_note.strip()} for s in ctx.settings.sources.values()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    if st.button("Tester la disponibilité des sources"):
        with st.spinner("Interrogation des sources…"):
            reports = check_all(ctx.collectors)
        if not ctx.settings.read_only:
            for r in reports:
                ctx.repo.record_health(r)
        st.cache_data.clear()
        st.dataframe(pd.DataFrame([{"Source": r.source, "État": r.status.value,
                                    "Message": r.message,
                                    "Latence (ms)": round(r.latency_ms) if r.latency_ms else None}
                                   for r in reports]), hide_index=True, width="stretch")

    st.subheader("Stockage")
    st.code(redact(ctx.settings.database_url), language=None)

    st.subheader("Unités des champs PMU (vérifiées le 30/10/2026)")
    st.dataframe(pd.DataFrame(list(PMU_UNITS.items()), columns=["Champ", "Unité"]),
                 hide_index=True, width="stretch")
