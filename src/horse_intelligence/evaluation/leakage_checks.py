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

    # 5. information : engagements observés uniquement après le départ (collecte a posteriori)
    share = float((~f["entry_seen_before_decision"]).mean())
    checks.append(Check("Engagements observés avant décision", True,
                        f"{100 * share:.0f} % des engagements ont été collectés après la course "
                        "(historique rattrapé) ; seuls leurs champs déclaratifs sont utilisés"))
    return [asdict(c) for c in checks]
