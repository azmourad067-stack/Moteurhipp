"""Collecte du programme, des partants, des cotes et des résultats PMU.

Exemples :
  python scripts/refresh_data.py                  # aujourd'hui
  python scripts/refresh_data.py --date 2026-10-29
  python scripts/refresh_data.py --check-sources  # état des sources uniquement

À lancer plusieurs fois par jour de courses pour historiser l'évolution des cotes
et des non-partants (chaque exécution ajoute un instantané horodaté).
"""

import argparse
import sys
from datetime import date, timedelta

import _bootstrap  # noqa: F401
from horse_intelligence.data.collectors.base import CollectorError
from horse_intelligence.data.collectors.pmu_collector import PmuCollector
from horse_intelligence.data.collectors.registry import check_all
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.ingestion import ingest_day


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--no-payouts", action="store_true")
    parser.add_argument("--countries", nargs="+", metavar="PAYS",
                        help="codes pays à collecter (ex. FRA) ; par défaut tous")
    parser.add_argument("--window-minutes", type=int,
                        help="collecte légère : courses partant dans les N prochaines minutes "
                             "et courses terminées sans résultat")
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()

    if args.check_sources:
        for report in check_all(ctx.collectors):
            ctx.repo.record_health(report)
            print(f"{report.source:14s} {report.status.value:20s} {report.message}")
        return 0

    pmu = ctx.collectors["pmu"]
    if not isinstance(pmu, PmuCollector):
        raise SystemExit("Connecteur PMU mal configuré.")
    try:
        window = timedelta(minutes=args.window_minutes) if args.window_minutes else None
        report = ingest_day(args.date, pmu, ctx.repo, fetch_payouts=not args.no_payouts,
                            countries=args.countries, window=window)
    except CollectorError as exc:
        print(f"Collecte impossible : {exc}", file=sys.stderr)
        return 2
    print(f"{report.day} : {report.status} — {report.races_seen} courses, "
          f"{report.races_ok} exploitables, {len(report.errors)} erreur(s)")
    for err in report.errors[:20]:
        print(f"  - {err['scope']}: {err['message']}")
    return 0 if report.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
