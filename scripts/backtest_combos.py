"""Simulation des paris combinés (couplé gagnant/placé, trio, 2 sur 4) sur l'historique.

Usage : python scripts/backtest_combos.py
Utilise les prédictions hors échantillon du dernier entraînement (scripts/train_models.py)
et les rapports officiels. Lecture seule.
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_combo_backtest


def _line(r: dict) -> str:  # type: ignore[type-arg]
    lo, hi = r["roi_ci"]
    return (f"{r['strategy']['name']:38s} {r['n_bets']:6d} paris  réussite {r['hit_rate']:5.1%}"
            f"  ROI {r['roi']:+7.1%} [IC 95 % {lo:+.1%} ; {hi:+.1%}]")


def main() -> int:
    configure_logging()
    rep = run_combo_backtest(build_context().repo.engine)
    print(f"Prédictions hors échantillon {rep['period'][0]} → {rep['period'][1]} "
          "(mise de 1 € par course, rapport officiel)")
    for bt, res in rep["bets"].items():
        sel = res["selection"]
        print(f"\n=== {bt} ===")
        print("Toute la période :")
        for r in res["all_period"]:
            print("  " + _line(r))
        print(f"Sélection {sel['selection_period'][0]} → {sel['selection_period'][1]}, "
              f"test {sel['test_period'][0]} → {sel['test_period'][1]} :")
        if sel["chosen_on_selection"]:
            print("  sélection : " + _line(sel["chosen_on_selection"]))
        if sel["test"]:
            print("  TEST      : " + _line(sel["test"]))
        for r in sel["references_test"]:
            print("  repère    : " + _line(r))
        print(f"  → {sel['message']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
