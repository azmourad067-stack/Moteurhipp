"""Stockage des artefacts (modèles, rapports) dans la base de données.

La base est la source unique : les tâches planifiées (GitHub Actions) y écrivent, et
l'application Streamlit — dont le disque est éphémère — y lit. Chaque contenu est
accompagné de son empreinte SHA-256, vérifiée à la lecture.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import Engine, delete, insert, select

from horse_intelligence.data.storage import database as db

URI_PREFIX = "db://"


def put(engine: Engine, key: str, content: bytes, content_type: str) -> str:
    sha = hashlib.sha256(content).hexdigest()
    t = db.stored_files
    with engine.begin() as conn:
        conn.execute(delete(t).where(t.c.key == key))
        conn.execute(insert(t).values(key=key, content=content, sha256=sha,
                                      size_bytes=len(content), content_type=content_type,
                                      created_at=datetime.now(UTC)))
    return sha


def get(engine: Engine, key: str, expected_sha256: str | None = None) -> bytes | None:
    t = db.stored_files
    with engine.connect() as conn:
        row = conn.execute(select(t.c.content, t.c.sha256).where(t.c.key == key)).first()
    if row is None:
        return None
    content = bytes(row.content)
    actual = hashlib.sha256(content).hexdigest()
    if actual != row.sha256 or (expected_sha256 and actual != expected_sha256):
        raise ValueError(f"empreinte invalide pour {key} : contenu modifié ou corrompu")
    return content


def put_json(engine: Engine, key: str, obj: Any) -> str:
    body = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()
    return put(engine, key, body, "application/json")


def get_json(engine: Engine, key: str) -> Any:
    body = get(engine, key)
    return None if body is None else json.loads(body)


def put_frame(engine: Engine, key: str, frame: pd.DataFrame) -> str:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return put(engine, key, buf.getvalue(), "application/vnd.apache.parquet")


def get_frame(engine: Engine, key: str) -> pd.DataFrame | None:
    body = get(engine, key)
    return None if body is None else pd.read_parquet(io.BytesIO(body))


def prune(engine: Engine, prefix: str, keep: int, protected: set[str] | None = None) -> int:
    """Supprime les contenus les plus anciens d'un préfixe (fichiers « -latest » et
    clés protégées conservés). Renvoie le nombre de suppressions."""
    t = db.stored_files
    protected = protected or set()
    with engine.begin() as conn:
        rows = conn.execute(select(t.c.key).where(t.c.key.startswith(prefix))
                            .order_by(t.c.created_at.desc())).all()
        candidates = [r.key for r in rows if not r.key.endswith("-latest.json")
                      and not r.key.endswith("-latest.parquet") and r.key not in protected]
        doomed = candidates[keep:]
        if doomed:
            conn.execute(delete(t).where(t.c.key.in_(doomed)))
    return len(doomed)
