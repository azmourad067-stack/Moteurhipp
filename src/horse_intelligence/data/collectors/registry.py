"""Instanciation des connecteurs à partir de la configuration."""

from __future__ import annotations

from horse_intelligence.config.settings import Settings
from horse_intelligence.data.collectors.base import BaseCollector, HealthReport, RawSink
from horse_intelligence.data.collectors.pmu_collector import PmuCollector
from horse_intelligence.data.collectors.secondary_collectors import (
    OpenPmuCollector,
    TurfBzhCollector,
)
from horse_intelligence.data.collectors.weather_collector import WeatherCollector

COLLECTOR_CLASSES: dict[str, type[BaseCollector]] = {
    "pmu": PmuCollector,
    "open_meteo": WeatherCollector,
    "open_pmu_api": OpenPmuCollector,
    "turf_bzh": TurfBzhCollector,
}


def build_collectors(settings: Settings, raw_sink: RawSink | None = None
                     ) -> dict[str, BaseCollector]:
    collectors: dict[str, BaseCollector] = {}
    for key, cls in COLLECTOR_CLASSES.items():
        if key in settings.sources:
            collectors[key] = cls(settings.sources[key], raw_sink=raw_sink,
                                  user_agent=settings.user_agent)
    return collectors


def check_all(collectors: dict[str, BaseCollector]) -> list[HealthReport]:
    return [c.health() for c in collectors.values()]
