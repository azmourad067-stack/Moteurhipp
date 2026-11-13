"""Exporte les tables principales en Parquet (analyses DuckDB/Polars).

Usage : python scripts/export_parquet.py [--out data_store/parquet]
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from horse_intelligence.data.storage.parquet_manager import export_tables
from horse_intelligence.services.context import build_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ctx = build_context()
    out = args.out or ctx.settings.data_dir / "parquet"
    for name, path in export_tables(ctx.repo.engine, out).items():
        print(f"{name:16s} → {path}")


if __name__ == "__main__":
    main()
