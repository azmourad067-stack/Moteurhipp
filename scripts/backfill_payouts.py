"""Rattrape les rapports définitifs (règlement des paris simulés) des courses terminées.

Usage : python scripts/backfill_payouts.py --start 2026-09-20 --end 2026-10-29 [--pool INTERNET]
"""

import argparse
import sys
from datetime import date

import _bootstrap  # noqa: F401
from horse_intelligence.data.collectors.pmu_collector import PmuCollector
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.ingestion import backfill_payouts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--pool", choices=["OFFLINE", "INTERNET"], default="INTERNET")
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()
    pmu = ctx.collectors["pmu"]
    if not isinstance(pmu, PmuCollector):
        raise SystemExit("Connecteur PMU mal configuré.")
    done, absent, errors = backfill_payouts(pmu, ctx.repo, args.start, args.end, args.pool)
    print(f"{done} courses complétées, {absent} sans rapport chez la source, "
          f"{len(errors)} erreur(s)")
    for e in errors[:20]:
        print(f"  - {e}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
