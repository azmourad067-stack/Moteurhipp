"""Purge des données inutiles pour rester dans le quota de la base gratuite.

Usage : python scripts/purge_storage.py [--keep-days 2]
Supprime les cotes relevées après le départ des courses de plus de N jours (jamais
utilisées). Les modèles et rapports anciens sont purgés à chaque entraînement.
"""

import argparse
import sys
from datetime import date, timedelta

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-days", type=int, default=2)
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()
    n = ctx.repo.purge_post_start_odds(date.today() - timedelta(days=args.keep_days))
    print(f"{n} cote(s) d'après départ supprimée(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
