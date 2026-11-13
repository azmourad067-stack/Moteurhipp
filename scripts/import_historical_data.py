"""Constitution de l'historique (backfill) à faible cadence, avec reprise.

Exemple : python scripts/import_historical_data.py --start 2025-10-30 --end 2026-10-29

Les journées déjà collectées avec succès après leur fin sont ignorées : le script
peut être interrompu et relancé. Le débit est limité par `min_interval_s`
(config/data_sources.yaml).

Rappel anti-fuite : pour une journée passée, les cotes « directes » récupérées
sont classées `post_start` et ne servent jamais aux décisions simulées.
"""

import argparse
import sys
from datetime import date, timedelta

import _bootstrap  # noqa: F401
from horse_intelligence.data.collectors.base import CollectorError
from horse_intelligence.data.collectors.pmu_collector import PmuCollector
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.ingestion import ingest_day


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(1))
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--countries", nargs="+", metavar="PAYS",
                        help="codes pays à collecter (ex. FRA) ; par défaut tous")
    parser.add_argument("--pools", nargs="+", choices=["OFFLINE", "INTERNET"],
                        help="masses de paris à collecter ; par défaut toutes")
    parser.add_argument("--no-payouts", action="store_true",
                        help="ne pas collecter les rapports (collecte 3 fois plus rapide)")
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end doit être postérieure à --start")
    configure_logging()
    ctx = build_context()
    pmu = ctx.collectors["pmu"]
    if not isinstance(pmu, PmuCollector):
        raise SystemExit("Connecteur PMU mal configuré.")

    failures = 0
    day = args.start
    while day <= args.end:
        if ctx.repo.day_completed(day):
            print(f"{day} : déjà collectée, ignorée")
        else:
            try:
                report = ingest_day(day, pmu, ctx.repo, countries=args.countries,
                                    pools=args.pools, fetch_payouts=not args.no_payouts)
                failures = 0
                print(f"{day} : {report.status} — {report.races_seen} courses")
            except CollectorError as exc:
                failures += 1
                print(f"{day} : échec ({exc})", file=sys.stderr)
                if failures >= args.max_consecutive_failures:
                    print("Arrêt : source indisponible de façon répétée.", file=sys.stderr)
                    return 2
        day += timedelta(days=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
