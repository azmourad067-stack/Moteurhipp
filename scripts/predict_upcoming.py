"""Enregistre les prédictions des courses qui partent bientôt (test en conditions réelles).

Usage : python scripts/predict_upcoming.py [--window-minutes 75]
À lancer après chaque collecte intrajournalière (tâche planifiée).
"""

import argparse
import sys
from datetime import timedelta

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.prediction.forward import record_upcoming
from horse_intelligence.services.context import build_context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-minutes", type=int, default=75)
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()
    n = record_upcoming(ctx.repo.engine, timedelta(minutes=args.window_minutes))
    print(f"{n} prédiction(s) enregistrée(s) avant départ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
