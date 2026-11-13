"""Configuration de l'application.

Priorité : variables d'environnement > secrets Streamlit > valeurs par défaut.
Aucune valeur secrète n'a de défaut dans le code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"


def _secret(name: str) -> str | None:
    """Lit une valeur depuis l'environnement puis depuis st.secrets si disponible."""
    value = os.environ.get(name)
    if value:
        return value
    try:  # st.secrets lève une exception s'il n'existe aucun fichier de secrets
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: S110 - absence de secrets = configuration par défaut
        pass
    return None


@dataclass(frozen=True)
class SourceConfig:
    key: str
    name: str
    base_url: str
    kind: str
    enabled: bool
    free: bool
    status_audit: str
    min_interval_s: float = 1.0
    timeout_s: float = 15.0
    max_retries: int = 3
    terms_note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    sources: dict[str, SourceConfig]
    user_agent: str = "horse-intelligence/0.1 (usage personnel; analyse)"
    read_only: bool = False  # application en ligne : aucune écriture en base

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def using_default_database(self) -> bool:
        return _secret("DATABASE_URL") is None

    def source(self, key: str) -> SourceConfig:
        return self.sources[key]


_KNOWN_FIELDS = {
    "name", "base_url", "kind", "enabled", "free", "status_audit",
    "min_interval_s", "timeout_s", "max_retries", "terms_note",
}


def load_sources(path: Path | None = None) -> dict[str, SourceConfig]:
    path = path or CONFIG_DIR / "data_sources.yaml"
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    sources: dict[str, SourceConfig] = {}
    for key, spec in (raw.get("sources") or {}).items():
        known = {k: v for k, v in spec.items() if k in _KNOWN_FIELDS}
        extra = {k: v for k, v in spec.items() if k not in _KNOWN_FIELDS}
        sources[key] = SourceConfig(key=key, extra=extra, **known)
    return sources


def build_settings() -> Settings:
    data_dir = Path(_secret("HI_DATA_DIR") or PROJECT_ROOT / "data_store")
    database_url = _secret("DATABASE_URL") or f"sqlite:///{data_dir / 'horse_intelligence.db'}"
    read_only = (_secret("HI_READ_ONLY") or "").strip().lower() in ("1", "true", "oui", "yes")
    return Settings(data_dir=data_dir, database_url=database_url, sources=load_sources(),
                    read_only=read_only)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return build_settings()
