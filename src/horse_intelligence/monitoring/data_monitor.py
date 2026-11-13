"""Fraîcheur et santé de la collecte."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Engine, func, select

from horse_intelligence.data.storage import database as db

STALE_AFTER = timedelta(hours=26)


def collection_health(engine: Engine, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    runs = db.collection_runs
    with engine.connect() as conn:
        last_ok = conn.execute(select(func.max(runs.c.started_at))
                               .where(runs.c.status.in_(["success", "partial"]))).scalar()
        recent = conn.execute(select(runs.c.status, func.count()).where(
            runs.c.started_at >= now - timedelta(hours=24)).group_by(runs.c.status)).all()
    if isinstance(last_ok, str):
        last_ok = datetime.fromisoformat(last_ok)
    if last_ok is not None and last_ok.tzinfo is None:
        last_ok = last_ok.replace(tzinfo=UTC)
    counts = {str(s): int(n) for s, n in recent}
    stale = last_ok is None or now - last_ok > STALE_AFTER
    return {"last_success": last_ok, "runs_24h": counts, "stale": stale,
            "message": ("Aucune collecte réussie récente : vérifier la tâche planifiée."
                        if stale else "Collecte à jour.")}
