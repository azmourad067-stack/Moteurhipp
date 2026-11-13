"""Ressources partagées entre pages, mises en cache par Streamlit."""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from horse_intelligence.services.context import AppContext, build_context


@st.cache_resource(show_spinner=False)
def get_context() -> AppContext:
    return build_context()


def format_dt(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.astimezone(UTC).strftime("%d/%m/%Y %H:%M UTC")
