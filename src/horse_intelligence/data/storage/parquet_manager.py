"""Export des tables vers Parquet pour les analyses volumineuses (DuckDB/Polars)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, select

from horse_intelligence.data.storage import database as db

EXPORTABLE = {
    "races": db.races, "race_entries": db.race_entries, "race_results": db.race_results,
    "odds_snapshots": db.odds_snapshots, "horses": db.horses, "jockeys": db.jockeys,
    "trainers": db.trainers, "racecourses": db.racecourses,
}


def export_tables(engine: Engine, out_dir: Path, tables: list[str] | None = None
                  ) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name in tables or list(EXPORTABLE):
        table = EXPORTABLE[name]
        with engine.connect() as conn:
            result = conn.execute(select(table))
            df = pd.DataFrame(result.mappings().all(), columns=list(result.keys()))
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        written[name] = path
    return written
