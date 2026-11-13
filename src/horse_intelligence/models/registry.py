"""Registre des modèles : artefacts versionnés (JSON + texte LightGBM, jamais pickle) stockés
en base avec leur empreinte SHA-256, métriques et statut (candidate | active | retired).

Règle de promotion : un modèle ne devient `active` que si, sur la validation
glissante, sa log-loss est meilleure que celle du marché calibré avec un intervalle
de confiance à 95 % entièrement négatif. Sinon il reste `candidate`.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Engine, and_, insert, select, update

from horse_intelligence.data.storage import blob_store
from horse_intelligence.data.storage import database as db


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
                             capture_output=True, text=True, check=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_artifact(engine: Engine, family: str, version: str,
                  payload: dict[str, Any]) -> tuple[str, str]:
    key = f"models/{family}-{version}.json"
    sha = blob_store.put_json(engine, key, payload)
    return f"{blob_store.URI_PREFIX}{key}", sha


def load_artifact(engine: Engine, uri: str,
                  expected_sha256: str | None = None) -> dict[str, Any]:
    if not uri.startswith(blob_store.URI_PREFIX):
        raise ValueError(f"emplacement d'artefact non pris en charge : {uri}")
    body = blob_store.get(engine, uri.removeprefix(blob_store.URI_PREFIX), expected_sha256)
    if body is None:
        raise ValueError(f"artefact introuvable : {uri}")
    data: dict[str, Any] = json.loads(body)
    return data


def register_model(engine: Engine, family: str, version: str, artifact: str, sha256: str,
                   train_start: date, train_end: date, hyperparams: dict[str, Any],
                   metrics: list[dict[str, Any]], promote: bool) -> int:
    now = datetime.now(UTC)
    with engine.begin() as conn:
        res = conn.execute(insert(db.model_versions).values(
            family=family, version=version, train_start=train_start, train_end=train_end,
            hyperparams_json=hyperparams, artifact_uri=artifact, artifact_sha256=sha256,
            git_commit=_git_commit(), status="candidate", created_at=now))
        model_id = int(res.inserted_primary_key[0]) if res.inserted_primary_key else 0
        for m in metrics:
            conn.execute(insert(db.model_metrics).values(model_id=model_id, **m))
        if promote:
            conn.execute(update(db.model_versions).where(and_(
                db.model_versions.c.family == family, db.model_versions.c.status == "active"))
                .values(status="retired"))
            conn.execute(update(db.model_versions).where(db.model_versions.c.model_id == model_id)
                         .values(status="active"))
    return model_id


def latest_model(engine: Engine, family: str | None = None,
                 statuses: tuple[str, ...] = ("active", "candidate")) -> dict[str, Any] | None:
    """Modèle actif en priorité, sinon dernier candidat."""
    mv = db.model_versions
    stmt = select(mv).where(mv.c.status.in_(statuses))
    if family:
        stmt = stmt.where(mv.c.family == family)
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(stmt.order_by(mv.c.created_at.desc())).mappings()]
    if not rows:
        return None
    active = [r for r in rows if r["status"] == "active"]
    return active[0] if active else rows[0]


def model_metrics(engine: Engine, model_id: int) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(
            select(db.model_metrics).where(db.model_metrics.c.model_id == model_id)).mappings()]


def activate_version(engine: Engine, family: str, version: str) -> None:
    """Retour arrière : rend active une version existante (l'actuelle passe « retired »)."""
    mv = db.model_versions
    with engine.begin() as conn:
        target = conn.execute(select(mv.c.model_id, mv.c.artifact_uri).where(and_(
            mv.c.family == family, mv.c.version == version))).first()
        if target is None:
            raise ValueError(f"version introuvable : {family} {version}")
        load_artifact(engine, target.artifact_uri)  # l'artefact doit exister et être intègre
        conn.execute(update(mv).where(and_(mv.c.family == family, mv.c.status == "active"))
                     .values(status="retired"))
        conn.execute(update(mv).where(mv.c.model_id == target.model_id).values(status="active"))


def list_versions(engine: Engine, family: str) -> list[dict[str, Any]]:
    mv = db.model_versions
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(
            select(mv.c.version, mv.c.status, mv.c.created_at, mv.c.train_end)
            .where(mv.c.family == family).order_by(mv.c.created_at.desc())).mappings()]
