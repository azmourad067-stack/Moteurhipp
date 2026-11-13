"""Résultats de la validation glissante (walk-forward)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from horse_intelligence.frontend.components.cards import EMERALD, metric_card, notice
from horse_intelligence.frontend.state import get_context
from horse_intelligence.services.training import REFERENCE_MODEL, latest_report
from horse_intelligence.services.validation import latest_validation

LABELS = {"uniform": "Uniforme", "market": "Marché brut", REFERENCE_MODEL: "Marché calibré",
          "form_logit": "Logit conditionnel (forme)", "lgbm_form": "LightGBM (forme)",
          "lgbm_market": "LightGBM + marché"}


def render() -> None:
    st.title("Backtesting")
    rep = latest_report(get_context().repo.engine)
    if rep is None:
        notice("Aucune évaluation disponible. Lancez : python scripts/train_models.py")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Période", f"{rep['period'][0]} → {rep['period'][1]}")
    with c2:
        metric_card("Courses", str(rep["n_races"]), f"{rep['n_races_with_market']} avec cotes")
    with c3:
        metric_card("Plis walk-forward", str(len(rep["folds"])))
    with c4:
        audit = rep["source_aggregates_audit"]
        metric_card("Audit des fuites", "réussi" if audit["passed"] else "échec",
                    f"{audit['n_winners']} gagnants contrôlés")
    for w in rep.get("warnings", []):
        st.warning(w)

    st.subheader("Qualité probabiliste (critère principal)")
    m = pd.DataFrame(rep["metrics_market_races"]).T
    m.index = [LABELS.get(i, i) for i in m.index]
    st.dataframe(m[["n_races", "log_loss", "brier", "ece", "top1", "winner_in_top3", "auc"]]
                 .rename(columns={"n_races": "Courses", "log_loss": "Log-loss", "brier": "Brier",
                                  "ece": "ECE", "top1": "Top-1", "winner_in_top3": "Top-3",
                                  "auc": "AUC"}).round(4), width="stretch")
    st.caption("Même échantillon de courses (avec cotes de référence) pour tous les modèles. "
               "Log-loss et Brier : plus bas = meilleur.")

    st.subheader("Comparaison au marché calibré")
    d = pd.DataFrame(rep["vs_reference"]).T
    d.index = [LABELS.get(i, i) for i in d.index]
    d["Verdict"] = np.where(d["ci_high"].astype(float) < 0, "meilleur (IC < 0)",
                            np.where(d["ci_low"].astype(float) > 0, "moins bon (IC > 0)",
                                     "non significatif"))
    st.dataframe(d.rename(columns={"mean_diff": "Δ log-loss", "ci_low": "IC bas",
                                   "ci_high": "IC haut", "n_races": "Courses"}).round(4),
                 width="stretch")

    st.subheader("Calibration")
    options = list(rep["calibration"])
    default = options.index("lgbm_market") if "lgbm_market" in options else 0
    choice = st.selectbox("Modèle", options, index=default,
                          format_func=lambda k: LABELS.get(k, k))
    cal = pd.DataFrame(rep["calibration"][choice])
    fig = go.Figure()
    fig.add_scatter(x=[0, cal["p_mean"].max()], y=[0, cal["p_mean"].max()], mode="lines",
                    line={"dash": "dot", "color": "#8B98A5"}, name="Calibration parfaite")
    fig.add_scatter(x=cal["p_mean"], y=cal["y_rate"], mode="lines+markers",
                    line={"color": EMERALD}, name="Observé")
    fig.update_layout(height=320, template="plotly_dark", xaxis_title="Probabilité prédite",
                      yaxis_title="Fréquence de victoire", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin={"l": 10, "r": 10, "t": 10, "b": 10})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Variables les plus utilisées (LightGBM forme)")
    _importance(rep)
    render_validation()


def _importance(rep: dict[str, Any]) -> None:
    st.bar_chart(pd.Series(rep["feature_importance"]).sort_values())
    st.caption("Importance = gain cumulé dans les arbres : indique l'usage d'une variable, "
               "pas un lien de causalité.")


def _sim_row(label: str, r: dict[str, Any] | None) -> dict[str, Any]:
    if not r:
        return {"": label, "Paris": 0}
    lo, hi = r["roi_ci"]
    return {"": label, "Paris": r["n_bets"], "Misé (€)": round(r["staked"], 1),
            "ROI": f"{100 * r['roi']:+.1f} %",
            "IC 95 % du ROI": f"[{100 * lo:+.1f} ; {100 * hi:+.1f}] %",
            "Réussite": f"{100 * r['hit_rate']:.0f} %",
            "Drawdown max (€)": round(r["max_drawdown"], 1),
            "Pire série": r["longest_losing_streak"]}


def render_validation() -> None:
    val = latest_validation(get_context().repo.engine)
    st.header("Validation (phase 5)")
    if val is None:
        notice("Aucune validation disponible. Lancez : python scripts/run_backtest.py")
        return
    st.subheader("Rentabilité théorique — simulation de paris")
    notice("Décision : probabilités hors échantillon et cote de référence publiée avant le "
           "départ. Règlement : rapport définitif officiel (mise de 1 €). La règle est choisie "
           "sur la période de sélection puis appliquée telle quelle au test : seul le test "
           "mesure une performance. Une rentabilité passée ne garantit rien.")
    st.caption(f"{val['races']} courses avec cotes, dont {val['races_with_payouts']} avec "
               "rapports définitifs collectés.")
    refs = pd.DataFrame([_sim_row(r["strategy"]["name"], r)
                         for r in val["references_all_period"]])
    st.markdown("**Repères sans modèle** (toute la période) : ils mesurent le coût du "
                "prélèvement du pari mutuel.")
    st.dataframe(refs, hide_index=True, width="stretch")
    for model, b in val["betting"].items():
        st.markdown(f"**{LABELS.get(model, model)}** — sélection {b['selection_period'][0]} → "
                    f"{b['selection_period'][1]}, test {b['test_period'][0]} → "
                    f"{b['test_period'][1]} ({b['n_strategies_tried']} règles essayées)")
        if b["chosen"]:
            st.caption(f"Règle retenue : {b['chosen']['name']}")
            st.dataframe(pd.DataFrame([_sim_row("Sélection", b["chosen_on_selection"]),
                                       _sim_row("TEST", b["test"])]),
                         hide_index=True, width="stretch")
            if b["test"] and b["test"]["equity"]:
                eq = pd.DataFrame(b["test"]["equity"])
                eq["cumul"] = (eq["return"] - eq["stake"]).cumsum()
                fig = go.Figure(go.Scatter(y=eq["cumul"], mode="lines",
                                           line={"color": EMERALD}))
                fig.update_layout(height=220, template="plotly_dark",
                                  yaxis_title="Gain cumulé (€)", xaxis_title="Paris",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  margin={"l": 10, "r": 10, "t": 10, "b": 10})
                st.plotly_chart(fig, width="stretch")
        notice(b["message"])

    st.subheader("Résultats par segment (LightGBM + marché vs marché calibré)")
    seg = pd.DataFrame(val["segments"])
    if not seg.empty:
        seg["Verdict"] = np.where(seg["ci_high"] < 0, "meilleur",
                                  np.where(seg["ci_low"] > 0, "moins bon", "non significatif"))
        st.dataframe(seg.rename(columns={
            "dimension": "Dimension", "segment": "Segment", "races": "Courses",
            "log_loss_model": "Log-loss modèle", "log_loss_reference": "Log-loss marché",
            "diff": "Δ", "ci_low": "IC bas", "ci_high": "IC haut",
            "top1_model": "Top-1 modèle", "top1_reference": "Top-1 marché"}).round(4),
            hide_index=True, width="stretch")
        st.caption(val["stability"]["message"] + " Segments de moins de 30 courses masqués.")

    st.subheader("Robustesse")
    rob = val["robustness"]
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Écart entre graines", f"{rob['seed_spread']:.4f}",
                    f"log-loss, {len(rob['seed_log_losses'])} apprentissages")
    with c2:
        metric_card(f"{int(100 * rob['masked_share'])} % de données masquées",
                    f"{rob['degradation_masked']:+.4f}", "variation de log-loss")
    with c3:
        metric_card("Probabilités valides", "oui" if rob["probabilities_valid"] else "NON",
                    "somme à 100 % malgré les manques")

    st.subheader("Audit des fuites de données")
    for c in val["leakage"]:
        icon = "✅" if c["passed"] else "⛔"
        st.markdown(f"{icon} **{c['name']}** — {c['detail']}")
