"""Réseau de neurones de classement par course (expérience « deep learning », phase 4).

Même schéma sans fuite que `LgbmRaceModel`, pour une comparaison à armes égales :
1. un réseau apprend un score « fondamental » par partant sur la partie ancienne de
   l'apprentissage, avec une perte softmax par course (logit conditionnel) ;
   - variables numériques standardisées (+ indicateurs de valeur manquante),
   - profils appris (embeddings) du cheval, du jockey, de l'entraîneur et des variables
     catégorielles : ce que le gradient boosting ne sait pas représenter ;
   - arrêt anticipé sur les dernières dates de cette partie ancienne ;
2. le score est calibré par course sur la partie récente de l'apprentissage, combiné à
   log(p_marché) dans la variante « + marché » (information AJOUTÉE au marché).

Expérimental : n'est pas enregistré ni utilisé par l'application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.models.ranking import (
    ConditionalLogit,
    FloatArray,
    group_codes,
    single_winner_races,
)

ENTITIES = ["horse_id", "jockey_id", "trainer_id"]
MIN_ENTITY_RUNS = 3      # en dessous : profil « inconnu » partagé (évite le surapprentissage)


@dataclass
class NeuralRaceModel:
    numeric: list[str]
    categorical: list[str]
    entities: list[str] = field(default_factory=lambda: list(ENTITIES))
    with_market: bool = False
    hidden: int = 64
    entity_dim: int = 8
    cat_dim: int = 4
    dropout: float = 0.2
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_races: int = 256
    max_epochs: int = 40
    patience: int = 3
    valid_share: float = 0.1
    seed: int = 42
    net: Any = None
    calibrator: ConditionalLogit | None = None
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    missing_flags: list[str] = field(default_factory=list)
    vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    epochs_run: int = 0

    # --- préparation des entrées (paramètres appris sur l'apprentissage seulement) ---
    def _fit_preprocessing(self, base: pd.DataFrame) -> None:
        num = base[self.numeric].astype("float64")
        self.medians = {str(k): float(v) for k, v in num.median().fillna(0.0).items()}
        filled = num.fillna(self.medians)
        self.means = {str(k): float(v) for k, v in filled.mean().items()}
        self.scales = {str(k): float(v) for k, v in
                       filled.std().replace(0.0, 1.0).fillna(1.0).items()}
        self.missing_flags = [c for c in self.numeric if num[c].isna().any()]
        self.vocab = {}
        for c in self.categorical:
            values = base[c].dropna().astype(str).unique()
            self.vocab[c] = {v: i + 1 for i, v in enumerate(sorted(values))}
        for c in self.entities:
            counts = base[c].dropna().astype("int64").value_counts()
            kept = sorted(counts[counts >= MIN_ENTITY_RUNS].index)
            self.vocab[c] = {str(v): i + 1 for i, v in enumerate(kept)}

    def _inputs(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        num = frame[self.numeric].astype("float64")
        flags = num[self.missing_flags].isna().astype("float64").to_numpy()
        z = ((num.fillna(self.medians) - pd.Series(self.means)) / pd.Series(self.scales))
        x_num = np.hstack([z.to_numpy(), flags]).astype(np.float32)
        cats = []
        for c in self.categorical:
            cats.append(frame[c].astype(object).map(
                lambda v, m=self.vocab[c]: m.get(str(v), 0) if pd.notna(v) else 0))
        for c in self.entities:
            cats.append(frame[c].map(
                lambda v, m=self.vocab[c]: m.get(str(int(v)), 0) if pd.notna(v) else 0))
        x_cat = np.stack([np.asarray(s, dtype=np.int64) for s in cats], axis=1)
        return x_num, x_cat

    def _build(self, n_num: int) -> Any:
        import torch
        from torch import nn

        cards = [len(self.vocab[c]) + 1 for c in [*self.categorical, *self.entities]]
        dims = [self.cat_dim] * len(self.categorical) + [self.entity_dim] * len(self.entities)

        class Net(nn.Module):
            def __init__(self, hidden: int, dropout: float) -> None:
                super().__init__()
                self.emb = nn.ModuleList(nn.Embedding(k, d) for k, d in zip(cards, dims,
                                                                             strict=True))
                width = n_num + sum(dims)
                self.mlp = nn.Sequential(
                    nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

            def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
                parts = [x_num, *(e(x_cat[:, i]) for i, e in enumerate(self.emb))]
                return self.mlp(torch.cat(parts, dim=1)).squeeze(1)

        return Net(self.hidden, self.dropout)

    # --- apprentissage du score -------------------------------------------------
    @staticmethod
    def _race_index(race_ids: np.ndarray) -> np.ndarray:
        """Matrice (courses × partants max) des indices de lignes, -1 en remplissage."""
        codes, n = group_codes(race_ids)
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes, minlength=n)
        width = int(counts.max()) if n else 0
        idx = np.full((n, width), -1, dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pos = np.arange(len(order)) - np.repeat(starts, counts)
        idx[codes[order], pos] = order
        return idx

    def _race_tensors(self, frame: pd.DataFrame) -> tuple[Any, ...]:
        import torch

        codes, n = group_codes(frame["race_id"].to_numpy())
        keep = single_winner_races(frame["won"].to_numpy(), codes, n)
        sub = frame[keep]
        x_num, x_cat = self._inputs(sub)
        idx = self._race_index(sub["race_id"].to_numpy())
        won = sub["won"].to_numpy()
        target = np.array([int(np.argmax(won[row[row >= 0]])) for row in idx])
        return (torch.from_numpy(x_num), torch.from_numpy(x_cat), torch.from_numpy(idx),
                torch.from_numpy(target))

    @staticmethod
    def _race_loss(net: Any, x_num: Any, x_cat: Any, idx: Any, target: Any) -> Any:
        import torch

        mask = idx >= 0
        rows = idx.clamp(min=0)
        flat = rows[mask]
        scores = torch.full(idx.shape, float("-inf"))
        scores[mask] = net(x_num[flat], x_cat[flat])
        return torch.nn.functional.cross_entropy(scores, target)

    def _fit_net(self, base: pd.DataFrame) -> None:
        import torch

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dates = sorted(base["race_date"].unique())
        cut = dates[max(1, int(len(dates) * (1 - self.valid_share)))] if len(dates) > 1 \
            else dates[-1]
        train, valid = base[base["race_date"] < cut], base[base["race_date"] >= cut]
        if train.empty or valid.empty:
            train = valid = base
        tr, va = self._race_tensors(train), self._race_tensors(valid)
        self.net = self._build(tr[0].shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        best, best_state, stale = float("inf"), None, 0
        n_races = tr[2].shape[0]
        for epoch in range(self.max_epochs):
            self.net.train()
            for batch in np.array_split(rng.permutation(n_races),
                                        max(1, n_races // self.batch_races)):
                b = torch.from_numpy(batch)
                opt.zero_grad()
                loss = self._race_loss(self.net, tr[0], tr[1], tr[2][b], tr[3][b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(self._race_loss(self.net, *va))
            self.epochs_run = epoch + 1
            if v < best - 1e-4:
                best, stale = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def raw_score(self, frame: pd.DataFrame) -> FloatArray:
        import torch

        if self.net is None:
            raise RuntimeError("réseau non entraîné")
        x_num, x_cat = self._inputs(frame)
        self.net.eval()
        with torch.no_grad():
            s = self.net(torch.from_numpy(x_num), torch.from_numpy(x_cat))
        return np.asarray(s.numpy(), dtype=np.float64)

    # --- calibration par course, comme LgbmRaceModel ----------------------------
    def _calib_inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = {"nn_score": self.raw_score(frame)}
        if self.with_market:
            cols["mkt_log_prob"] = frame["mkt_log_prob"].to_numpy()
        return pd.DataFrame(cols, index=frame.index)

    def fit_score(self, base: pd.DataFrame) -> NeuralRaceModel:
        self._fit_preprocessing(base)
        self._fit_net(base)
        return self

    def fit_calibration(self, calib: pd.DataFrame) -> NeuralRaceModel:
        Z = self._calib_inputs(calib)
        self.calibrator = ConditionalLogit(l2=1e-3).fit(Z, calib["race_id"], calib["won"],
                                                        list(Z.columns))
        return self

    def fit(self, base: pd.DataFrame, calib: pd.DataFrame) -> NeuralRaceModel:
        return self.fit_score(base).fit_calibration(calib)

    def predict(self, frame: pd.DataFrame) -> FloatArray:
        if self.calibrator is None:
            raise RuntimeError("modèle non entraîné")
        return self.calibrator.predict_proba(self._calib_inputs(frame), frame["race_id"])
