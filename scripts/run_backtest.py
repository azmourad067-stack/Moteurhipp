"""Phase 5 : simulation de paris, robustesse et audit des fuites.

Pré-requis : python scripts/train_models.py (prédictions hors échantillon) et les
rapports définitifs (scripts/backfill_payouts.py).
Usage : python scripts/run_backtest.py
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.validation import run_validation


def _fmt(r: dict) -> str:  # type: ignore[type-arg]
    if r is None:
        return "—"
    lo, hi = r["roi_ci"]
    return (f"{r['n_bets']} paris, ROI {100 * r['roi']:+.1f} % "
            f"[IC 95 % {100 * lo:+.1f} ; {100 * hi:+.1f}], réussite {100 * r['hit_rate']:.0f} %, "
            f"drawdown max {r['max_drawdown']:.1f} €, pire série {r['longest_losing_streak']}")


def main() -> int:
    configure_logging()
    ctx = build_context()
    try:
        rep = run_validation(ctx.repo.engine)
    except ValueError as exc:
        print(f"Validation impossible : {exc}", file=sys.stderr)
        return 2
    print(f"Prédictions hors échantillon {rep['period'][0]} → {rep['period'][1]} : "
          f"{rep['races']} courses avec cotes, {rep['races_with_payouts']} avec rapports")
    print("\nRepères sans modèle (toute la période, mise de 1 €) :")
    for r in rep["references_all_period"]:
        print(f"  {r['strategy']['name']:28s} {_fmt(r)}")
    for model, b in rep["betting"].items():
        print(f"\n[{model}] sélection {b['selection_period'][0]} → {b['selection_period'][1]}, "
              f"test {b['test_period'][0]} → {b['test_period'][1]} "
              f"({b['n_strategies_tried']} règles essayées)")
        if b["chosen"]:
            print(f"  règle retenue : {b['chosen']['name']}")
            print(f"  sélection : {_fmt(b['chosen_on_selection'])}")
            print(f"  TEST      : {_fmt(b['test'])}")
        print(f"  → {b['message']}")
    print(f"\nStabilité : {rep['stability']['message']}")
    rob = rep["robustness"]
    print(f"Robustesse (dernier pli, {rob['races']} courses) : écart entre graines "
          f"{rob['seed_spread']:.4f} ; {int(100 * rob['masked_share'])} % de valeurs masquées → "
          f"log-loss {rob['degradation_masked']:+.4f} ; probabilités valides : "
          f"{rob['probabilities_valid']}")
    print("\nAudit des fuites :")
    for c in rep["leakage"]:
        print(f"  [{'OK' if c['passed'] else 'ÉCHEC'}] {c['name']} — {c['detail']}")
    return 0 if all(c["passed"] for c in rep["leakage"]) else 1


if __name__ == "__main__":
    sys.exit(main())
