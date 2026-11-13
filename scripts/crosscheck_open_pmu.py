"""Compare les arrivées stockées avec celles de l'API communautaire Open PMU.

Usage : python scripts/crosscheck_open_pmu.py --date 2026-09-29
"""

import argparse
import sys
from datetime import date

import _bootstrap  # noqa: F401
from horse_intelligence.data.collectors.base import CollectorError
from horse_intelligence.data.collectors.secondary_collectors import OpenPmuCollector
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.crosscheck import compare_arrivals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()
    collector = ctx.collectors["open_pmu_api"]
    if not isinstance(collector, OpenPmuCollector):
        raise SystemExit("Connecteur Open PMU mal configuré.")
    try:
        secondary = collector.fetch_arrivees(args.date)
    except CollectorError as exc:
        print(f"Recoupement impossible : {exc}", file=sys.stderr)
        return 2
    for c in compare_arrivals(ctx.repo, args.date, secondary):
        print(f"{c.race_key:22s} {c.status:16s} PMU={c.primary} "
              f"OpenPMU={c.secondary} — {c.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
