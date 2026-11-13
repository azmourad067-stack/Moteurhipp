"""Cœur probabiliste : probabilités par course.

Une course est un événement de classement. Chaque partant reçoit un score s_i ;
P(i gagne) = exp(s_i) / Σ_j exp(s_j) — les probabilités somment à 1 dans chaque
course par construction (logit conditionnel / Plackett-Luce).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def group_codes(race_ids: Sequence[object] | npt.ArrayLike) -> tuple[IntArray, int]:
    _, codes = np.unique(np.asarray(race_ids), return_inverse=True)
    codes = codes.astype(np.int64)
    return codes, int(codes.max()) + 1 if codes.size else 0


def group_softmax(scores: npt.ArrayLike, codes: IntArray, n_groups: int) -> FloatArray:
    s = np.asarray(scores, dtype=np.float64)
    gmax = np.full(n_groups, -np.inf)
    np.maximum.at(gmax, codes, s)
    e = np.exp(s - gmax[codes])
    denom = np.bincount(codes, weights=e, minlength=n_groups)
    return np.asarray(e / denom[codes], dtype=np.float64)


def normalize_within_race(values: npt.ArrayLike, codes: IntArray, n_groups: int) -> FloatArray:
    v = np.asarray(values, dtype=np.float64)
    total = np.bincount(codes, weights=v, minlength=n_groups)
    return np.asarray(v / total[codes], dtype=np.float64)


def single_winner_races(winner: npt.ArrayLike, codes: IntArray,
                        n_groups: int) -> npt.NDArray[np.bool_]:
    """Masque des lignes appartenant à une course avec exactement un gagnant."""
    w = np.asarray(winner, dtype=np.float64)
    per_race = np.bincount(codes, weights=w, minlength=n_groups)
    return np.asarray(per_race[codes] == 1, dtype=np.bool_)


@dataclass
class ConditionalLogit:
    """Logit conditionnel (multinomial par course) avec pénalité L2.

    Les variables sont centrées-réduites à l'apprentissage ; les valeurs manquantes
    sont remplacées par la moyenne d'apprentissage (0 après réduction), ce qui
    revient à ne pas différencier les partants sur une information absente.
    """

    l2: float = 1.0
    coef_: FloatArray = field(default_factory=lambda: np.zeros(0))
    mean_: FloatArray = field(default_factory=lambda: np.zeros(0))
    scale_: FloatArray = field(default_factory=lambda: np.ones(0))
    feature_names: list[str] = field(default_factory=list)

    def _transform(self, X: npt.ArrayLike) -> FloatArray:
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return np.asarray(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)

    def fit(self, X: npt.ArrayLike, race_ids: npt.ArrayLike, winner: npt.ArrayLike,
            feature_names: list[str] | None = None) -> ConditionalLogit:
        from scipy.optimize import minimize  # dépendance d'entraînement uniquement

        X = np.asarray(X, dtype=np.float64)
        codes, n = group_codes(race_ids)
        keep = single_winner_races(winner, codes, n)
        X, y = X[keep], np.asarray(winner, dtype=np.float64)[keep]
        codes, n = group_codes(np.asarray(race_ids)[keep])
        if n == 0:
            raise ValueError("aucune course exploitable (un gagnant unique requis)")
        with warnings.catch_warnings():  # colonne entièrement vide : moyenne 0, échelle 1
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean_ = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)
        self.scale_ = np.where((sd > 1e-12) & ~np.isnan(sd), sd, 1.0)
        Z = self._transform(X)
        self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]

        def loss_grad(beta: FloatArray) -> tuple[float, FloatArray]:
            s = Z @ beta
            p = group_softmax(s, codes, n)
            ll = -float(np.sum(y * np.log(np.clip(p, 1e-300, None))))
            grad = Z.T @ (p - y) + self.l2 * beta
            return ll + 0.5 * self.l2 * float(beta @ beta), grad

        res = minimize(loss_grad, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
        self.coef_ = np.asarray(res.x, dtype=np.float64)
        return self

    def scores(self, X: npt.ArrayLike) -> FloatArray:
        return np.asarray(self._transform(X) @ self.coef_, dtype=np.float64)

    def predict_proba(self, X: npt.ArrayLike, race_ids: npt.ArrayLike) -> FloatArray:
        codes, n = group_codes(race_ids)
        return group_softmax(self.scores(X), codes, n)

    def to_dict(self) -> dict[str, object]:
        return {"l2": self.l2, "coef": self.coef_.tolist(), "mean": self.mean_.tolist(),
                "scale": self.scale_.tolist(), "features": self.feature_names}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ConditionalLogit:
        m = cls(l2=float(d["l2"]))  # type: ignore[arg-type]
        m.coef_ = np.asarray(d["coef"], dtype=np.float64)
        m.mean_ = np.asarray(d["mean"], dtype=np.float64)
        m.scale_ = np.asarray(d["scale"], dtype=np.float64)
        m.feature_names = list(d["features"])  # type: ignore[call-overload]
        return m


def paid_places(n_runners: int) -> int:
    """Nombre de places payées au simple placé PMU selon le nombre de partants.

    Règle appliquée : moins de 4 partants → pas de placé ; 4 à 7 → 2 places ;
    8 et plus → 3 places. Le nombre à fournir est celui des partants DÉCLARÉS :
    vérifié sur les rapports du 29/10/2026 (88 combinaisons course × masse), où des courses
    à 8 déclarés dont 1 non-partant ont payé 3 places
    (evaluation/metrics.py : `check_paid_places_rule`).
    """
    if n_runners < 4:
        return 0
    return 2 if n_runners < 8 else 3


def place_probabilities(p_win: npt.ArrayLike, n_places: int,
                        discounts: Sequence[float] = (1.0, 1.0, 1.0)) -> FloatArray:
    """P(terminer dans les `n_places` premiers), calcul exact.

    Modèle de Harville généralisé (Lo, Bacon-Shone & Busche) : la probabilité d'être
    k-ième parmi les restants est proportionnelle à p_i ** discounts[k-1].
    Avec discounts = (1, 1, 1) on retrouve Harville, connu pour surestimer les
    chances de place des outsiders ; les exposants se calibrent sur l'historique.
    """
    p = np.asarray(p_win, dtype=np.float64)
    n = p.size
    out = np.zeros(n)
    if n_places <= 0 or n == 0:
        return out
    if n_places >= n:
        return np.ones(n)
    strengths = [np.power(np.clip(p, 1e-12, None), discounts[min(k, len(discounts) - 1)])
                 for k in range(n_places)]
    if n_places <= 3:
        return _place_vectorized(strengths, n_places)
    for order in permutations(range(n), n_places):
        prob = 1.0
        used: list[int] = []
        for k, i in enumerate(order):
            s = strengths[k]
            remaining = s.sum() - s[used].sum()
            prob *= s[i] / remaining
            used.append(i)
        for i in order:
            out[i] += prob
    return out


def _place_vectorized(strengths: list[FloatArray], n_places: int) -> FloatArray:
    """Même calcul exact que l'énumération, sous forme matricielle (n ≤ ~30)."""
    s1 = strengths[0]
    n = s1.size
    first = s1 / s1.sum()
    out = first.copy()
    if n_places == 1:
        return out
    s2 = strengths[1]
    eye = np.eye(n, dtype=bool)
    # m2[j, k] = P(k deuxième | j premier)
    m2 = np.where(eye, 0.0, s2[None, :] / (s2.sum() - s2[:, None]))
    second = first @ m2
    out += second
    if n_places == 2:
        return out
    s3 = strengths[2]
    # t[j, k, i] = P(j 1er) · P(k 2e | j) · P(i 3e | j, k)
    denom = s3.sum() - s3[:, None] - s3[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond3 = s3[None, None, :] / denom[:, :, None]
    i_idx = np.arange(n)
    valid = ((i_idx[None, None, :] != i_idx[:, None, None])
             & (i_idx[None, None, :] != i_idx[None, :, None])
             & ~eye[:, :, None])
    t = np.where(valid, first[:, None, None] * m2[:, :, None] * cond3, 0.0)
    out += t.sum(axis=(0, 1))
    return out
