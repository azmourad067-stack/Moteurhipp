"""Simulation de bankroll de la règle « 3e du modèle, simple gagnant, trot G ».

Usage : python scripts/simulate_bankroll.py [--sims 10000]
Rejoue des milliers d'années possibles à partir des paris réellement joués par la règle
(prédictions hors échantillon, rapports officiels), sous plusieurs hypothèses de
rentabilité. Montants en mises (1 mise = le montant fixe joué à chaque course).
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.experiment import run_bankroll


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=10_000)
    args = parser.parse_args()
    configure_logging()
    rep = run_bankroll(build_context().repo.engine, n_sims=args.sims)
    rp = rep["real_path"]
    print(f"Règle : {rep['rule']}")
    print(f"Historique {rep['period'][0]} → {rep['period'][1]} : {rep['n_bets']} paris "
          f"(≈ {rep['bets_per_year']} par an), réussite {rep['hit_rate']:.1%}, "
          f"ROI {rep['roi']:+.1%}")
    print(f"Rapport gagnant médian {rep['median_winning_payout']:.2f} € pour 1 €, "
          f"meilleur {rep['best_payout']:.2f} €")
    print(f"Parcours réel : gain final {rp['final']:+.1f} mises, creux maximal "
          f"{rp['max_drawdown']:.1f} mises, plus bas {rp['lowest']:+.1f} mises, "
          f"plus longue série de défaites {rp['longest_losing_streak']}")
    for y, r in rep["real_by_year"].items():
        print(f"  {y} : {r['paris']} paris, {r['gain']:+.1f} mises (ROI {r['roi']:+.1%})")
    print()
    print(f"Simulation : {args.sims} années de {rep['bets_per_year']} paris, en mises")
    for s in rep["scenarios"]:
        fq, st, dd = s.final_quantiles, s.longest_losing_streak, s.max_drawdown
        ruin = "  ".join(f"{b} mises : {p:.0%}" for b, p in s.p_ruin.items())
        print(f"--- {s.scenario} (ROI {s.roi:+.1%})")
        print(f"  gain final : P5 {fq['P5']:+.0f} | P25 {fq['P25']:+.0f} | médiane "
              f"{fq['médiane']:+.0f} | P75 {fq['P75']:+.0f} | P95 {fq['P95']:+.0f}")
        print(f"  probabilité de finir l'année en gain : {s.p_profit:.0%}")
        print(f"  plus longue série de défaites : médiane {st['médiane']:.0f}, "
              f"1 année sur 10 ≥ {st['P90']:.0f}, 1 sur 100 ≥ {st['P99']:.0f}")
        print(f"  creux maximal : médiane {dd['médiane']:.0f}, 1 année sur 10 ≥ "
              f"{dd['P90']:.0f}, 1 sur 100 ≥ {dd['P99']:.0f}")
        print(f"  budget épuisé dans l'année : {ruin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
