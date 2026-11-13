"""Confronte les règles figées le 10/11/2026 à une période qu'elles n'ont pas vue.

Usage : python scripts/validate_frozen_rules.py --end 2025-11-19 [--start AAAA-MM-JJ]
Critère fixé à l'avance : ROI > 0 avec IC 95 % entièrement au-dessus de zéro.
"""

import argparse
import sys
from datetime import date

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_frozen_rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    configure_logging()
    rep = run_frozen_rules(build_context().repo.engine, start=args.start, end=args.end)
    print(f"Validation {rep['period'][0]} → {rep['period'][1]} (règles figées le 10/11/2026)")
    print(f"{'règle':38s} {'à la découverte':>15s} {'paris':>6s} {'réussite':>9s} "
          f"{'ROI validation':>15s}  IC 95 %            verdict")
    for r in rep["rules"]:
        lo, hi = r["roi_ci"]
        verdict = "CONFIRMÉE" if r["confirmed"] else "écartée"
        print(f"{r['rule']:38s} {r['found_roi']:+15.1%} {r['n_bets']:6d} {r['hit_rate']:9.1%} "
              f"{r['roi']:+15.1%}  [{lo:+.1%} ; {hi:+.1%}]  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
