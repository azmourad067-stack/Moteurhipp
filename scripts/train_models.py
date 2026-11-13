"""Entraîne et évalue les modèles (validation glissante), puis les enregistre.

Usage : python scripts/train_models.py [--min-train-days 21] [--test-days 7] [--calib-days 7]

Nécessite les dépendances d'entraînement : pip install -r requirements-train.txt
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.training import train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-train-days", type=int, default=21)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--calib-days", type=int, default=7)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()
    try:
        rep = train_and_evaluate(ctx.repo.engine, args.min_train_days, args.test_days,
                                 args.calib_days, register=not args.no_register)
    except ValueError as exc:
        print(f"Entraînement impossible : {exc}", file=sys.stderr)
        return 2
    print(f"Période {rep.period[0]} → {rep.period[1]} : {rep.n_races} courses, "
          f"{rep.n_races_with_market} avec cotes de référence, {len(rep.folds)} plis")
    print(f"Audit des agrégats source : {rep.source_aggregates_audit['message']}")
    print("\nCourses avec marché (même échantillon pour tous les modèles) :")
    print(f"{'modèle':20s} {'courses':>7s} {'logloss':>8s} {'brier':>7s} {'top1':>6s} "
          f"{'top3':>6s} {'AUC':>6s} {'ECE':>6s}")
    for name, m in rep.metrics_market_races.items():
        print(f"{name:20s} {m['n_races']:7.0f} {m['log_loss']:8.4f} {m['brier']:7.4f} "
              f"{m['top1']:6.3f} {m['winner_in_top3']:6.3f} {m['auc']:6.3f} {m['ece']:6.4f}")
    print("\nDifférence de log-loss vs marché calibré (négatif = meilleur), IC 95 % :")
    for name, d in rep.vs_reference.items():
        print(f"  {name:20s} {d['mean_diff']:+.4f}  [{d['ci_low']:+.4f} ; {d['ci_high']:+.4f}]"
              f"  ({d['n_races']} courses)")
    for w in rep.warnings:
        print(f"⚠ {w}")
    for name, errs in rep.errors.items():
        print(f"✖ {name}: {errs[:2]}")
    for fam, info in rep.registered.items():
        print(f"Enregistré : {fam} v{info['version']} → {info['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
