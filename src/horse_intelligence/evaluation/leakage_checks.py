"""Audit des fuites de données exécuté sur la base réelle (complète les tests unitaires)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from horse_intelligence.evaluation.metrics import auc_score
from horse_intelligence.intelligence.feature_engineering import (
    DECISION_OFFSET,
    FORM_FEATURES,
    build_dataset,
)
from horse_intelligence.intelligence.leakage_audit import audit_source_aggregates

SUSPICIOUS_AUC = 0.90
HISTORY_COLS = ["jockey_win_rate", "jockey_db_runs", "trainer_win_rate", "trainer_db_runs",
                "horse_db_runs", "horse_days_since_db", "horse_db_mean_pct", "pair_db_runs",
                "rider_change"]


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _entry_predecision_counts(raw: dict[str, pd.DataFrame]) -> tuple[int, int]:
    """Return (entries without a pre-decision snapshot, identifiable entries)."""
    races = raw["races"][["race_id", "scheduled_start_utc"]].copy()
    races["decision_time"] = (
        pd.to_datetime(races["scheduled_start_utc"], utc=True, errors="coerce")
        - DECISION_OFFSET
    )
    races = races.dropna(subset=["decision_time"])
    entries = raw["entries"][["race_id", "num_pmu", "observed_at"]].copy()
    entries = entries.merge(
        races[["race_id", "decision_time"]],
        on="race_id",
        how="inner",
        validate="many_to_one",
    )
    identities = entries.dropna(subset=["race_id", "num_pmu"])[
        ["race_id", "num_pmu"]
    ].drop_duplicates()
    if identities.empty:
        return 0, 0

    entries["observed_at"] = pd.to_datetime(
        entries["observed_at"], utc=True, errors="coerce"
    )
    predecision = entries.loc[
        entries["observed_at"].notna()
        & entries["decision_time"].notna()
        & (entries["observed_at"] <= entries["decision_time"]),
        ["race_id", "num_pmu"],
    ].drop_duplicates()
    coverage = identities.merge(
        predecision,
        on=["race_id", "num_pmu"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    without_snapshot = int((coverage["_merge"] == "left_only").sum())
    return without_snapshot, len(identities)


def run_leakage_checks(raw: dict[str, pd.DataFrame], seed: int = 0) -> list[dict[str, Any]]:
    checks: list[Check] = []
    ds = build_dataset(raw)
    f = ds.frame

    # 1. cotes utilisées publiées avant l'instant de décision
    o = raw["odds"].merge(raw["races"][["race_id", "scheduled_start_utc"]], on="race_id")
    used = f.dropna(subset=["mkt_prob"])[["race_id", "num_pmu", "mkt_pool"]]
    o = o.merge(used, left_on=["race_id", "num_pmu", "pool"],
                right_on=["race_id", "num_pmu", "mkt_pool"])
    decision = o["scheduled_start_utc"] - DECISION_OFFSET
    eligible = o[o["published_at"] <= decision]
    ok = len(eligible) > 0 and eligible.groupby(["race_id", "num_pmu"]).size().size == len(used)
    checks.append(Check("Cotes antérieures à la décision", bool(ok),
                        f"{len(used)} cotes de marché utilisées, toutes avec une publication "
                        f"antérieure à départ − {int(DECISION_OFFSET.total_seconds() // 60)} min"
                        if ok else "Des cotes utilisées n'ont pas de publication antérieure."))

    # 2. permutation du futur sur les données réelles
    dates = sorted(f["race_date"].unique())
    cut = dates[len(dates) // 2]
    altered = {k: v.copy() for k, v in raw.items()}
    res = altered["results"].merge(altered["races"][["race_id", "race_date"]], on="race_id")
    mask = (res["race_date"] > cut).to_numpy()
    rng = np.random.default_rng(seed)
    altered["results"].loc[mask, "finish_position"] = rng.permutation(
        altered["results"].loc[mask, "finish_position"].to_numpy())
    g = build_dataset(altered).frame
    key = ["race_id", "num_pmu"]
    a = f[f["race_date"] <= cut].set_index(key)[HISTORY_COLS].sort_index()
    b = g[g["race_date"] <= cut].set_index(key)[HISTORY_COLS].sort_index()
    changed = [c for c in HISTORY_COLS
               if not (a[c].eq(b[c]) | (a[c].isna() & b[c].isna())).all()]
    checks.append(Check("Permutation des résultats futurs", not changed,
                        f"Variables d'historique jusqu'au {cut} inchangées après permutation "
                        "des résultats ultérieurs" if not changed else
                        "Des variables passées dépendent de résultats futurs : "
                        + ", ".join(changed)))

    # 3. aucune variable ne prédit « trop bien » le résultat à elle seule
    lab = f[f["won"].notna()]
    worst = ("", 0.5)
    for c in FORM_FEATURES:
        s = pd.to_numeric(lab[c], errors="coerce")
        ok_rows = s.notna()
        if ok_rows.sum() < 100:
            continue
        auc = auc_score(lab.loc[ok_rows, "won"] == 1, s[ok_rows])
        auc = max(auc, 1 - auc)
        if auc > worst[1]:
            worst = (c, auc)
    checks.append(Check("Aucune variable trop prédictive", worst[1] < SUSPICIOUS_AUC,
                        f"AUC maximale d'une variable seule : {worst[1]:.3f} ({worst[0]}) ; "
                        f"seuil d'alerte {SUSPICIOUS_AUC}"))

    # 4. agrégats de la source publiés avant la course
    audit = audit_source_aggregates(f)
    checks.append(Check("Musique et cumuls avant course", audit.passed, audit.message))

    # 5. Couverture calculée sur les brutes : f ne contient que les lignes valides
    # pré-décision et ne permet donc pas de mesurer les observations manquantes.
    without_snapshot, total_entries = _entry_predecision_counts(raw)
    share = without_snapshot / total_entries if total_entries else float("nan")
    detail = (
        f"{without_snapshot}/{total_entries} engagements sans snapshot observé avant "
        f"la décision ({100 * share:.0f} %) ; les courses concernées sont exclues"
        if total_entries else "Aucun engagement identifiable dans les données brutes."
    )
    checks.append(Check("Couverture des engagements à la décision", without_snapshot == 0, detail))
    return [asdict(c) for c in checks]
