"""Laboratoire IA : hypothèses découvertes, acteurs, anomalies et dérive."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from horse_intelligence.frontend.components.cards import EMERALD, metric_card, notice
from horse_intelligence.frontend.state import get_context
from horse_intelligence.intelligence.explainability import FEATURE_LABELS
from horse_intelligence.services.discovery import latest_lab_report

STATUS_ORDER = ["validée hors échantillon", "en cours de validation", "non confirmée",
                "invalidée", "hypothèse détectée"]


def _patterns(rep: dict[str, Any]) -> None:
    st.caption(f"Découverte : {rep['discovery_period'][0]} → {rep['discovery_period'][1]} · "
               f"confirmation : {rep['confirmation_period'][0]} → "
               f"{rep['confirmation_period'][1]}")
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Sous-groupes testés", str(rep["n_tests"]),
                    "tous comptés dans la correction des tests multiples")
    with c2:
        metric_card("Retenus (FDR 10 %)", str(len(rep["patterns"])),
                    "puis re-testés sur la période de confirmation")
    if not rep["patterns"]:
        notice("Aucun sous-groupe ne s'écarte significativement des cotes après correction "
               "des tests multiples. Résultat attendu sur un historique court : les cotes "
               "intègrent déjà l'information disponible.")
        return
    t = pd.DataFrame(rep["patterns"])
    t["status"] = pd.Categorical(t["status"], STATUS_ORDER, ordered=True)
    view = pd.DataFrame({
        "Statut": t["status"], "Configuration": t["label"], "Partants": t["n"],
        "A/E découverte": t["ae"].round(2),
        "IC 90 %": [f"[{a:.2f} ; {b:.2f}]" for a, b in zip(t["ae_low"], t["ae_high"],
                                                          strict=True)],
        "q-value": t["q_value"].round(3),
        "A/E confirmation": t.get("oos_ae", pd.Series(dtype=float)).round(2),
        "Stabilité hebdo": t["stability"].round(2),
    }).sort_values("Statut")
    st.dataframe(view, hide_index=True, width="stretch")
    st.caption("A/E = victoires observées / victoires attendues par les cotes. A/E > 1 : le "
               "marché sous-estime ce profil. Association statistique, jamais une cause ; "
               "pas une stratégie rentable démontrée.")


def _actors(rep: dict[str, Any]) -> None:
    for title, label in (("jockeys", "Jockeys / drivers"), ("trainers", "Entraîneurs")):
        a = rep["actors"][title]
        st.markdown(f"#### {label}")
        notice(a["dispersion"])
        p = a["persistence"]
        st.caption(f"Persistance d'une période à l'autre : {p['message']} "
                   f"(ρ = {p['spearman']:.2f}, p = {p['p_value']:.3f}, "
                   f"{p['n_entities']} acteurs communs)")
        t = pd.DataFrame(a["table"])
        if t.empty:
            continue
        st.dataframe(pd.DataFrame({
            "Nom": t["name"], "Courses": t["runs"], "Victoires": t["actual"].astype(int),
            "Attendues (cotes)": t["expected"].round(1), "A/E brut": t["ae_raw"].round(2),
            "A/E ajusté": t["ae_post"].round(2),
            "IC crédible 90 %": [f"[{x:.2f} ; {y:.2f}]" for x, y in
                                 zip(t["ae_low"], t["ae_high"], strict=True)],
        }).head(25), hide_index=True, width="stretch")
    st.caption("A/E ajusté : estimation bayésienne (Gamma-Poisson) dont le lissage est appris "
               "sur les données. Un A/E brut élevé sur peu de courses est ramené vers 1.")


def _anomalies(rep: dict[str, Any]) -> None:
    a = rep["anomalies"]
    t = a["test"]
    notice(f"{t['message']} A/E des 5 % de partants les plus atypiques (période de "
           f"confirmation) : {t['ae']:.2f} [{t['ae_low']:.2f} ; {t['ae_high']:.2f}], "
           f"p = {t['p_value']:.3f}, n = {t['n']}.")
    ex = pd.DataFrame(a["examples"])
    if ex.empty:
        return
    ex["Écarts principaux"] = ex["deviations"].map(
        lambda d: ", ".join(f"{FEATURE_LABELS.get(f, f)} ({v:+.1f})" for f, v in d))
    st.dataframe(ex.rename(columns={"date": "Date", "horse": "Cheval", "num_pmu": "N°",
                                    "score": "Atypicité", "won": "Gagnant"})
                 [["Date", "Cheval", "N°", "Atypicité", "Gagnant", "Écarts principaux"]]
                 .round(3), hide_index=True, width="stretch")
    st.caption("Écarts en nombre d'écarts interquartiles par rapport à la médiane.")


def _drift(rep: dict[str, Any]) -> None:
    psi = pd.DataFrame(rep["drift"]["psi"])
    psi["Variable"] = psi["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    st.dataframe(psi[["Variable", "psi", "niveau"]].rename(
        columns={"psi": "PSI", "niveau": "Niveau"}).round(3), hide_index=True, width="stretch")
    st.caption("PSI entre la période de découverte et les 7 derniers jours. Les variables "
               "d'historique interne évoluent mécaniquement tant que la base se construit.")
    w = pd.DataFrame(rep["drift"]["weekly_market"])
    fig = go.Figure(go.Scatter(x=w["week"], y=w["log_loss"], mode="lines+markers",
                               line={"color": EMERALD}))
    fig.update_layout(height=260, template="plotly_dark", yaxis_title="Log-loss du marché",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, width="stretch")


def render() -> None:
    st.title("Laboratoire IA")
    rep = latest_lab_report(get_context().repo.engine)
    if rep is None:
        notice("Aucune analyse disponible. Lancez : python scripts/run_discovery.py")
        return
    for w in rep.get("warnings", []):
        st.warning(w)
    tabs = st.tabs(["Patterns", "Jockeys & entraîneurs", "Anomalies", "Dérive"])
    with tabs[0]:
        _patterns(rep)
    with tabs[1]:
        _actors(rep)
    with tabs[2]:
        _anomalies(rep)
    with tabs[3]:
        _drift(rep)
