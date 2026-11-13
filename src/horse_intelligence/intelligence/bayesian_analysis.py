"""Performance des acteurs relativement au marché (modèle Gamma-Poisson hiérarchique).

Pour un jockey, un entraîneur ou un cheval : A = victoires observées,
E = victoires attendues par le marché (somme des probabilités implicites).
Modèle : A ~ Poisson(λ·E), λ ~ Gamma(a, a) (moyenne 1 : le marché est juste en moyenne).
L'hyperparamètre a est estimé par maximum de vraisemblance marginale (binomiale
négative) : la force du lissage est apprise, pas choisie. Postérieur : Gamma(a+A, a+E).

λ > 1 : l'acteur gagne plus souvent que ne l'anticipent les cotes — une ASSOCIATION
statistique, jamais une relation causale, et pas une garantie de rentabilité.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_EXPECTED = 1.0
A_MAX = 1e4


@dataclass(frozen=True)
class PriorFit:
    a: float
    n_entities: int
    dispersion_p: float = float("nan")  # test du rapport de vraisemblance vs Poisson pur


def fit_prior(actual: np.ndarray, expected: np.ndarray) -> PriorFit:
    from scipy.optimize import minimize_scalar
    from scipy.special import gammaln
    from scipy.stats import chi2, poisson

    A = np.asarray(actual, dtype=float)
    E = np.asarray(expected, dtype=float)

    def nll(log_a: float) -> float:
        a = np.exp(log_a)
        ll = (gammaln(a + A) - gammaln(a) - gammaln(A + 1)
              + a * np.log(a / (a + E)) + A * np.log(E / (a + E)))
        return -float(ll.sum())

    res = minimize_scalar(nll, bounds=(np.log(0.5), np.log(A_MAX)), method="bounded")
    # H0 : aucune dispersion (A ~ Poisson(E)). Paramètre à la frontière : la statistique
    # suit un mélange 50/50 de χ²(0) et χ²(1).
    ll_poisson = float(poisson.logpmf(A, E).sum())
    lr = max(0.0, 2 * (-float(res.fun) - ll_poisson))
    p = 0.5 * float(chi2.sf(lr, 1)) if lr > 0 else 1.0
    return PriorFit(float(np.exp(res.x)), len(A), p)


def dispersion_message(prior: PriorFit) -> str:
    if not prior.dispersion_p < 0.05:
        return ("Aucune dispersion détectable : les écarts entre acteurs s'expliquent par le "
                f"hasard (p = {prior.dispersion_p:.2f}) ; les cotes intègrent déjà leur "
                "réussite.")
    return (f"Dispersion détectée (p = {prior.dispersion_p:.3f}) : écart-type relatif des "
            f"performances ≈ {100 / np.sqrt(prior.a):.0f} % autour de l'attente du marché.")


def actor_table(frame: pd.DataFrame, key: str, prior: PriorFit | None = None,
                cred: float = 0.90) -> tuple[pd.DataFrame, PriorFit]:
    """`frame` : partants avec résultat et probabilité de marché (`won`, `mkt_prob`)."""
    from scipy.stats import gamma

    f = frame.dropna(subset=[key, "mkt_prob", "won"])
    g = f.groupby(key).agg(runs=("won", "size"), actual=("won", "sum"),
                           expected=("mkt_prob", "sum")).reset_index()
    g = g[g["expected"] >= MIN_EXPECTED]
    prior = prior or fit_prior(g["actual"].to_numpy(), g["expected"].to_numpy())
    shape = prior.a + g["actual"]
    rate = prior.a + g["expected"]
    g["ae_raw"] = g["actual"] / g["expected"]
    g["ae_post"] = shape / rate
    alpha = (1 - cred) / 2
    g["ae_low"] = gamma.ppf(alpha, shape, scale=1 / rate)
    g["ae_high"] = gamma.ppf(1 - alpha, shape, scale=1 / rate)
    g["p_above_1"] = gamma.sf(1.0, shape, scale=1 / rate)
    return (g.sort_values(["ae_post", "expected"], ascending=False).reset_index(drop=True),
            prior)


@dataclass(frozen=True)
class Persistence:
    n_entities: int
    spearman: float
    p_value: float
    message: str


def persistence_test(discovery: pd.DataFrame, confirmation: pd.DataFrame, key: str
                     ) -> Persistence:
    """La surperformance estimée sur une période se retrouve-t-elle sur la suivante ?

    Un seul test global (corrélation de rangs entre λ postérieur de la période de
    découverte et résidu standardisé (A−E)/√E de la période de confirmation) : il évite
    la multiplication des tests acteur par acteur.
    """
    from scipy.stats import spearmanr

    disc, _ = actor_table(discovery, key)
    conf = confirmation.dropna(subset=[key, "mkt_prob", "won"]).groupby(key).agg(
        a2=("won", "sum"), e2=("mkt_prob", "sum")).reset_index()
    m = disc.merge(conf[conf["e2"] >= MIN_EXPECTED], on=key)
    if len(m) < 20:
        return Persistence(len(m), float("nan"), float("nan"),
                           "Trop peu d'acteurs communs aux deux périodes pour conclure.")
    resid = (m["a2"] - m["e2"]) / np.sqrt(m["e2"])
    rho, p = spearmanr(m["ae_post"], resid, alternative="greater")
    msg = ("Surperformance persistante hors échantillon (p < 0,05)." if p < 0.05 else
           "Pas de persistance démontrée hors échantillon.")
    return Persistence(len(m), float(rho), float(p), msg)
