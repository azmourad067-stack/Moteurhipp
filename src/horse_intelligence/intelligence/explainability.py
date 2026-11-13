"""Explication des estimations : contributions SHAP natives de LightGBM.

Les probabilités étant calculées course par course, seules les DIFFÉRENCES entre
partants comptent : chaque contribution est donc exprimée relativement à la moyenne
de la course. Une contribution décrit le fonctionnement du modèle, jamais une cause.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from horse_intelligence.models.lightgbm_model import LgbmRaceModel

FEATURE_LABELS = {
    "age": "âge", "draw": "numéro de corde", "draw_rel": "position à la corde",
    "weight_carried_kg": "poids porté", "weight_vs_field": "poids par rapport au lot",
    "handicap_extra_m": "recul (distance supplémentaire)", "n_runners": "nombre de partants",
    "distance_m": "distance", "log_prize": "allocation de la course",
    "mus_n": "nombre de courses récentes", "mus_last_pos": "dernière place",
    "mus_mean_pos_3": "forme sur 3 courses", "mus_mean_pos_6": "forme sur 6 courses",
    "mus_trend": "progression récente", "mus_wins_6": "victoires récentes",
    "mus_top3_6": "places récentes", "mus_incidents_6": "incidents récents",
    "mus_same_disc_share": "habitude de la discipline", "career_starts": "expérience",
    "career_win_rate": "taux de victoire en carrière",
    "career_place_rate": "taux de place en carrière",
    "log_earnings_per_start": "gains par course", "horse_db_runs": "courses suivies",
    "horse_days_since_db": "repos depuis la dernière course",
    "horse_db_mean_pct": "classements récents (base interne)",
    "jockey_win_rate": "réussite du jockey/driver", "jockey_db_runs": "activité du jockey",
    "trainer_win_rate": "réussite de l'entraîneur", "trainer_db_runs": "activité de l'entraîneur",
    "pair_db_runs": "association cheval-jockey", "rider_change": "changement de monte",
    "discipline": "discipline", "racecourse": "hippodrome", "sex": "sexe",
    "blinkers": "œillères", "shoes": "ferrure",
    "mus_mean_pos_6_rel": "forme relative au lot",
    "career_win_rate_rel": "réussite relative au lot",
    "jockey_win_rate_rel": "jockey relatif au lot",
    "trainer_win_rate_rel": "entraîneur relatif au lot",
    "log_earnings_per_start_rel": "gains relatifs au lot",
    "weight_carried_kg_rel": "poids relatif au lot",
}


def relative_contributions(model: LgbmRaceModel, frame: pd.DataFrame) -> pd.DataFrame:
    contrib = model.booster.predict(model._matrix(frame), pred_contrib=True)
    cols = [*model.booster.feature_name(), "_bias"]
    c = pd.DataFrame(np.asarray(contrib), columns=cols, index=frame.index).drop(columns="_bias")
    return c - c.groupby(frame["race_id"]).transform("mean")


def strengths_weaknesses(contrib: pd.Series, k: int = 3, min_abs: float = 0.05
                         ) -> tuple[list[str], list[str]]:
    s = contrib[contrib.abs() >= min_abs].sort_values()
    label = FEATURE_LABELS.get
    strengths = [label(f, f) for f in s[s > 0].index[::-1][:k]]
    weaknesses = [label(f, f) for f in s[s < 0].index[:k]]
    return strengths, weaknesses
