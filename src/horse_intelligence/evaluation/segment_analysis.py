"""Analyse détaillée par segment : où le modèle apporte-t-il de l'information, et que
rapporterait le simple gagnant / placé sur chaque rang du modèle ?

Dimensions : discipline, nombre de partants, distance, réunion, groupe (catégorie de la
course, déduite du libellé et des conditions) et rang du modèle. Analyse DESCRIPTIVE sur
les prédictions hors échantillon ; plusieurs centaines de cases sont examinées, d'où :
- une p-valeur par case (ROI > 0) corrigée par Benjamini-Hochberg (q-valeur) ;
- le ROI de chaque moitié de la période : une piste n'est « robuste » que si elle est
  positive sur les deux moitiés ET significative après correction.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation.betting_simulator import payout_lookup

MIN_BETS = 60
RANK_LABELS = ("1", "2", "3", "4", "5", "6+")

_TROT_CLASS = re.compile(r"\bCourse ([A-HR])\b")
_GROUP = re.compile(r"\bGroupe?\s*(III|II|I)\b|\bGR\.?\s*(III|II|I)\b", re.IGNORECASE)


def race_group(label: object, conditions: object, discipline: object) -> str:
    """Catégorie de la course à partir du libellé et des conditions (texte PMU)."""
    text = f"{label or ''} {conditions or ''}"
    # le groupe d'une course figure dans son libellé ou en tête de ses conditions ; plus
    # loin, les conditions citent des groupes pour définir les engagés (« n'ayant pas
    # gagné un Groupe I »), ce qui ne dit rien de la course elle-même
    m = _GROUP.search(f"{label or ''} {str(conditions or '')[:40]}")
    if m:
        return f"Groupe {(m.group(1) or m.group(2)).upper()}"
    if re.search(r"\bListed\b|\(L\)", text, re.IGNORECASE):
        return "Listed"
    if str(discipline or "").startswith("trot"):
        c = _TROT_CLASS.search(text)
        return f"Trot {c.group(1)}" if c else "Trot (autre)"
    low = text.lower()
    if "handicap" in low or "valeur" in low:
        return "Handicap"
    if "réclamer" in low or "reclamer" in low:
        return "Réclamer"
    return "Galop (autre)"


def _bins(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=edges, labels=labels, right=True).astype(object)


def add_segments(frame: pd.DataFrame, model_col: str) -> pd.DataFrame:
    f = frame.copy()
    f["partants"] = _bins(f["n_runners"], [0, 8, 12, 16, 99], ["≤ 8", "9-12", "13-16", "≥ 17"])
    f["distance"] = _bins(f["distance_m"], [0, 1600, 2100, 2700, 9999],
                          ["≤ 1600 m", "1601-2100 m", "2101-2700 m", "> 2700 m"])
    f["réunion"] = np.select([f["reunion_num"] == 1, f["reunion_num"] == 2,
                              f["reunion_num"] <= 4], ["R1", "R2", "R3-R4"], "R5+")
    f["groupe"] = [race_group(lab, cond, d) for lab, cond, d in
                   zip(f["label"], f["conditions_text"], f["discipline"], strict=True)]
    f["discipline"] = f["discipline"].astype(object).fillna("?")
    rank = f.groupby("race_id")[model_col].rank(ascending=False, method="first")
    f["rang_modèle"] = np.where(rank >= 6, "6+", rank.astype("Int64").astype(str))
    return f


DIMENSIONS = ("discipline", "partants", "distance", "réunion", "groupe")


def _p_value(profit: np.ndarray) -> float:
    """p-valeur unilatérale de « ROI > 0 » (approximation normale du profit moyen)."""
    n = len(profit)
    sd = float(profit.std(ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0:
        return 1.0
    z = float(profit.mean()) / (sd / math.sqrt(n))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def segment_table(frame: pd.DataFrame, payouts: pd.DataFrame, model_col: str,
                  selection_share: float = 0.5) -> pd.DataFrame:
    """Une ligne par (dimension, segment, rang du modèle, pari)."""
    lookup = payout_lookup(payouts, "INTERNET")
    paid = {k[0] for k in lookup if k[1] == "SIMPLE_GAGNANT"}
    f = add_segments(frame[frame["race_id"].isin(paid)], model_col)
    for bet in ("SIMPLE_GAGNANT", "SIMPLE_PLACE"):
        f[bet] = [lookup.get((int(r), bet, str(int(n))), 0.0) - 1.0
                  for r, n in zip(f["race_id"], f["num_pmu"], strict=True)]
    dates = sorted(f["race_date"].unique())
    cut = dates[int(len(dates) * selection_share)]
    rows: list[dict[str, Any]] = []
    for dim in (*DIMENSIONS, "toutes"):
        groups = f.groupby(dim) if dim != "toutes" else [("toutes", f)]
        for seg, g in groups:
            for rank in RANK_LABELS:
                sub = g[g["rang_modèle"] == rank]
                for bet in ("SIMPLE_GAGNANT", "SIMPLE_PLACE"):
                    profit = sub[bet].to_numpy()
                    if len(profit) < MIN_BETS:
                        continue
                    first = sub.loc[sub["race_date"] < cut, bet].to_numpy()
                    second = sub.loc[sub["race_date"] >= cut, bet].to_numpy()
                    rows.append({
                        "dimension": dim, "segment": str(seg), "rang": rank,
                        "pari": "gagnant" if bet == "SIMPLE_GAGNANT" else "placé",
                        "paris": len(profit), "réussite": float((profit > -1).mean()),
                        "roi": float(profit.mean()),
                        "roi_1re_moitié": float(first.mean()) if len(first) else math.nan,
                        "roi_2nde_moitié": float(second.mean()) if len(second) else math.nan,
                        "p": _p_value(profit)})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["q"] = _bh(table["p"].to_numpy())
        table["robuste"] = ((table["q"] < 0.10) & (table["roi_1re_moitié"] > 0)
                            & (table["roi_2nde_moitié"] > 0))
    return table


def model_value_by_segment(frame: pd.DataFrame, model_col: str,
                           market_col: str) -> pd.DataFrame:
    """Gain de log-loss du modèle sur le marché calibré, par segment (négatif = mieux)."""
    f = add_segments(frame, model_col)
    won = f[f["won"] == 1]
    won = won.assign(diff=-np.log(won[model_col].clip(1e-9))
                     + np.log(won[market_col].clip(1e-9)))
    rows = []
    for dim in (*DIMENSIONS, "toutes"):
        groups = won.groupby(dim) if dim != "toutes" else [("toutes", won)]
        for seg, g in groups:
            d = g["diff"].to_numpy()
            if len(d) < 30:
                continue
            se = float(d.std(ddof=1) / math.sqrt(len(d)))
            rows.append({"dimension": dim, "segment": str(seg), "courses": len(d),
                         "gain_logloss": float(d.mean()),
                         "ic_bas": float(d.mean()) - 1.96 * se,
                         "ic_haut": float(d.mean()) + 1.96 * se})
    return pd.DataFrame(rows)
