"""Métriques d'évaluation, séparées selon les trois objectifs du projet.

- performance prédictive : top-1, gagnant dans le top-3 prédit, AUC ;
- qualité probabiliste (critère principal) : log-loss par course, Brier, calibration ;
- valeur économique : traitée séparément (evaluation/betting_simulator.py, phase 5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from horse_intelligence.models.ranking import paid_places

EPS = 1e-12


@dataclass(frozen=True)
class Metrics:
    n_races: int
    n_runners: int
    log_loss: float          # moyenne de -log p(gagnant) par course
    brier: float             # Brier multi-classes moyen par course
    top1: float              # gagnant = favori du modèle
    winner_in_top3: float
    auc: float               # AUC partant par partant (gagne / ne gagne pas)
    ece: float               # erreur de calibration (10 classes de probabilité)

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def eval_frame(frame: pd.DataFrame, proba: npt.ArrayLike) -> pd.DataFrame:
    """Restreint aux courses à gagnant unique et attache les probabilités."""
    f = frame[["race_id", "race_date", "num_pmu", "won", "finish_position", "n_runners"]].copy()
    f["p"] = np.asarray(proba, dtype=np.float64)
    winners = f.groupby("race_id")["won"].transform("sum")
    return f[(winners == 1) & f.groupby("race_id")["p"].transform(lambda s: s.notna().all())]


def race_log_losses(f: pd.DataFrame) -> pd.Series:
    w = f[f["won"] == 1].set_index("race_id")["p"]
    return -np.log(w.clip(lower=EPS))


def auc_score(y: npt.ArrayLike, s: npt.ArrayLike) -> float:
    """AUC par la statistique de Mann-Whitney (rangs moyens en cas d'égalité)."""
    y_arr = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y_arr.sum()), int((~y_arr).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(s, dtype=np.float64)).rank(method="average").to_numpy()
    return float((ranks[y_arr].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration_table(p: npt.ArrayLike, y: npt.ArrayLike, bins: int = 10) -> pd.DataFrame:
    """Classes de probabilité à effectifs égaux : prédit moyen vs fréquence observée."""
    df = pd.DataFrame({"p": np.asarray(p, dtype=np.float64), "y": np.asarray(y, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=bins, labels=False)
    t = df.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size"))
    return t.reset_index(drop=True)


def compute_metrics(frame: pd.DataFrame, proba: npt.ArrayLike) -> Metrics:
    f = eval_frame(frame, proba)
    ll = race_log_losses(f)
    brier = ((f["p"] - f["won"]) ** 2).groupby(f["race_id"]).sum()
    rank = f.groupby("race_id")["p"].rank(ascending=False, method="first")
    won = f["won"] == 1
    cal = calibration_table(f["p"], f["won"]) if len(f) >= 10 else None
    ece = float((abs(cal["p_mean"] - cal["y_rate"]) * cal["n"]).sum() / cal["n"].sum()) \
        if cal is not None else float("nan")
    return Metrics(
        n_races=int(f["race_id"].nunique()), n_runners=len(f),
        log_loss=float(ll.mean()), brier=float(brier.mean()),
        top1=float((rank[won] == 1).mean()), winner_in_top3=float((rank[won] <= 3).mean()),
        auc=auc_score(won, f["p"]), ece=ece,
    )


def paired_bootstrap(frame: pd.DataFrame, proba_a: npt.ArrayLike, proba_b: npt.ArrayLike,
                     n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Différence de log-loss (A - B) par course, IC bootstrap à 95 % (rééchantillonnage
    des courses). Négatif = A meilleur que B."""
    fa = eval_frame(frame, proba_a)
    fb = eval_frame(frame, proba_b)
    common = fa["race_id"].unique()
    common = np.intersect1d(common, fb["race_id"].unique())
    la = race_log_losses(fa[fa["race_id"].isin(common)]).sort_index()
    lb = race_log_losses(fb[fb["race_id"].isin(common)]).sort_index()
    d = (la - lb).to_numpy()
    if d.size == 0:
        return {"n_races": 0, "mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return {"n_races": int(d.size), "mean_diff": float(d.mean()),
            "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975))}


def check_paid_places_rule(payouts: pd.DataFrame, runners: pd.Series) -> pd.DataFrame:
    """Compare le nombre de combinaisons payées au simple placé à la règle appliquée.

    `payouts` : colonnes race_id, bet_type, combination ; `runners` : partants DÉCLARÉS
    par course (la règle s'applique aux déclarés, non-partants compris).
    """
    sp = payouts[payouts["bet_type"] == "SIMPLE_PLACE"]
    paid = sp.groupby("race_id")["combination"].nunique().rename("paid_observed")
    t = pd.concat([paid, runners.rename("n_runners")], axis=1, join="inner")
    t["paid_rule"] = t["n_runners"].map(paid_places)
    # un ex æquo peut ajouter une combinaison : on signale seulement les écarts
    t["consistent"] = t["paid_observed"] >= t["paid_rule"]
    return t
