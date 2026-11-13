"""Assemblage des dépendances (configuration, base, connecteurs)."""

from __future__ import annotations

from dataclasses import dataclass

from horse_intelligence.config.settings import Settings, get_settings
from horse_intelligence.data.collectors.base import BaseCollector
from horse_intelligence.data.collectors.registry import build_collectors
from horse_intelligence.data.storage.database import init_db, make_engine
from horse_intelligence.data.storage.raw_store import RawStore
from horse_intelligence.data.storage.repositories import Repository


@dataclass
class AppContext:
    settings: Settings
    repo: Repository
    collectors: dict[str, BaseCollector]


def build_context(settings: Settings | None = None) -> AppContext:
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)
    repo = Repository(engine, RawStore(settings.raw_dir))
    if not settings.read_only:  # en lecture seule, le schéma est créé par les tâches planifiées
        init_db(engine)
        repo.sync_sources(settings)
    return AppContext(settings, repo, build_collectors(settings))
