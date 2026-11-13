"""Composants visuels : style global, cartes, badges de statut."""

from __future__ import annotations

import html

import streamlit as st

from horse_intelligence.data.processors.validation import STATUS_LABELS, DataStatus

EMERALD = "#10B981"
ELECTRIC = "#3B82F6"
AMBER = "#F59E0B"
RED = "#EF4444"
MUTED = "#8B98A5"

_STATUS_COLORS = {
    DataStatus.VERIFIED: EMERALD,
    DataStatus.PARTIAL: AMBER,
    DataStatus.INSUFFICIENT: RED,
    DataStatus.UNAVAILABLE: MUTED,
}

_CSS = f"""
<style>
.hi-card {{ background:#18202A; border:1px solid #243041; border-radius:12px;
           padding:16px 18px; height:100%; }}
.hi-card .label {{ color:{MUTED}; font-size:0.8rem; text-transform:uppercase;
                  letter-spacing:.04em; }}
.hi-card .value {{ color:#E6EDF3; font-size:1.6rem; font-weight:600; margin-top:4px; }}
.hi-card .hint {{ color:{MUTED}; font-size:0.8rem; margin-top:4px; }}
.hi-badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem;
            font-weight:600; color:#0F141A; }}
.hi-notice {{ border-left:3px solid {ELECTRIC}; background:#141B24; padding:10px 14px;
             border-radius:6px; color:#C9D4DF; }}
</style>
"""


def inject_style() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def metric_card(label: str, value: str, hint: str = "") -> None:
    hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
    st.markdown(
        f'<div class="hi-card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def status_badge(status: str | None) -> str:
    try:
        s = DataStatus(status) if status else DataStatus.UNAVAILABLE
    except ValueError:
        s = DataStatus.UNAVAILABLE
    return (f'<span class="hi-badge" style="background:{_STATUS_COLORS[s]}">'
            f"{html.escape(STATUS_LABELS[s])}</span>")


def notice(text: str) -> None:
    st.markdown(f'<div class="hi-notice">{html.escape(text)}</div>', unsafe_allow_html=True)
