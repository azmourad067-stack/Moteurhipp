"""Analyse détaillée par segment (discipline, partants, distance, réunion, groupe, rang).

Usage : python scripts/analyse_segments.py [--out DOSSIER]
Lecture seule. Écrit les tableaux complets en CSV dans DOSSIER (défaut : data_store/analyses).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_segment_analysis


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data_store/analyses"))
    args = parser.parse_args()
    configure_logging()
    rep = run_segment_analysis(build_context().repo.engine)
    args.out.mkdir(parents=True, exist_ok=True)
    roi, value = rep["roi"], rep["valeur_modele"]
    roi.to_csv(args.out / "roi_par_segment.csv", index=False)
    value.to_csv(args.out / "valeur_modele_par_segment.csv", index=False)
    per = rep["periode"].iloc[0]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(f"Prédictions hors échantillon {per['debut']} → {per['fin']}")
    print("\n=== Apport du modèle sur le marché calibré, par segment "
          "(gain de log-loss, négatif = le modèle fait mieux) ===")
    print(value.round(4).to_string(index=False))
    print(f"\n=== Simple gagnant / placé par rang du modèle : {len(roi)} cases examinées ===")
    cols = ["dimension", "segment", "rang", "pari", "paris", "réussite", "roi",
            "roi_1re_moitié", "roi_2nde_moitié", "q", "robuste"]
    best = roi.sort_values("roi", ascending=False).head(25)[cols]
    print("\n25 meilleures cases (avant tout contrôle) :")
    print(best.round(3).to_string(index=False))
    both = roi[(roi["roi_1re_moitié"] > 0) & (roi["roi_2nde_moitié"] > 0)]
    print(f"\nCases positives sur les deux moitiés : {len(both)} / {len(roi)}")
    print(both[cols].round(3).to_string(index=False) if len(both) else "  aucune")
    print("\n=== Tableau complet (rangs 1 à 5) ===")
    full = roi[roi["rang"] != "6+"].sort_values(["dimension", "segment", "pari", "rang"])
    print(full[cols].round(3).to_string(index=False))
    robust = roi[roi["robuste"]]
    print(f"\nCases robustes (q < 0,10 et positives sur les deux moitiés) : {len(robust)}")
    print(robust[cols].round(3).to_string(index=False) if len(robust) else "  aucune")
    return 0


if __name__ == "__main__":
    sys.exit(main())
