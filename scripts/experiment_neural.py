"""Expérience deep learning : réseau de neurones contre le modèle LightGBM actuel.

Usage : python scripts/experiment_neural.py [--refit-every-days 7]
Nécessite : pip install -r requirements-train.txt -r requirements-dl.txt
Lecture seule : aucun modèle n'est enregistré.
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import (
    CHALLENGERS,
    INCUMBENT,
    run_neural_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit-every-days", type=int, default=7)
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()
    rep = run_neural_experiment(ctx.repo.engine, refit_every_days=args.refit_every_days)
    print(f"Période {rep.period[0]} → {rep.period[1]} : {rep.n_races} courses comparées, "
          f"{rep.n_folds} plis, {rep.seconds:.0f} s")
    print(f"Réseau réappris {len(rep.nn_fits)} fois : "
          + ", ".join(f"{f['trained_until']} ({f['epochs']} époques, {f['seconds']} s)"
                      for f in rep.nn_fits[:3]) + (" …" if len(rep.nn_fits) > 3 else ""))
    print(f"\n{'modèle':16s} {'logloss':>8s} {'brier':>7s} {'top1':>6s} {'top3':>6s} "
          f"{'AUC':>6s} {'ECE':>6s}")
    for name, m in rep.metrics.items():
        print(f"{name:16s} {m['log_loss']:8.4f} {m['brier']:7.4f} {m['top1']:6.3f} "
              f"{m['winner_in_top3']:6.3f} {m['auc']:6.3f} {m['ece']:6.4f}")
    print("\nDifférence de log-loss vs marché calibré (négatif = meilleur), IC 95 % :")
    for name, d in rep.vs_reference.items():
        print(f"  {name:16s} {d['mean_diff']:+.4f}  [{d['ci_low']:+.4f} ; {d['ci_high']:+.4f}]")
    print(f"\nDifférence de log-loss vs {INCUMBENT} (modèle actuel), IC 95 % :")
    for name, d in rep.vs_incumbent.items():
        print(f"  {name:16s} {d['mean_diff']:+.4f}  [{d['ci_low']:+.4f} ; {d['ci_high']:+.4f}]")
    for name, errs in rep.errors.items():
        print(f"✖ {name}: {errs[:2]}")
    verdict = ("SUCCÈS : le deep learning bat le modèle actuel" if rep.success else
               "ÉCHEC : le deep learning ne bat pas significativement le modèle actuel")
    print(f"\nCritère fixé à l'avance ({' ou '.join(CHALLENGERS)} meilleur que {INCUMBENT}, "
          f"IC 95 % < 0) → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
