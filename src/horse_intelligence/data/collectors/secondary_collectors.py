"""Connecteurs secondaires."""

from __future__ import annotations

import json
from datetime import date

from pydantic import ValidationError

from horse_intelligence.data.collectors.base import (
    BaseCollector,
    ContractError,
    HealthReport,
    NotFoundError,
    NotOperationalCollector,
)
from horse_intelligence.data.schemas.open_pmu import OpenPmuRace, OpenPmuResponse


class OpenPmuCollector(BaseCollector):
    """API communautaire Open PMU — usage limité au recoupement des arrivées.

    Ses cotes ne sont jamais importées (non horodatées), et ses données ne servent
    pas d'historique d'entraînement (une course par jour, champ incomplet).
    """

    key = "open_pmu_api"

    def fetch_arrivees(self, day: date) -> list[OpenPmuRace]:
        url = f"{self.config.base_url.rstrip('/')}/api/arrivees"
        raw = self._fetch("arrivees", url, {"date": day.strftime("%m/%d/%Y")})
        try:
            parsed = OpenPmuResponse.model_validate(json.loads(raw.body))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ContractError("réponse Open PMU inattendue") from exc
        if isinstance(parsed.message, str):
            if parsed.error:
                raise NotFoundError(f"open_pmu_api : {parsed.message}")
            return []
        return parsed.message

    def health(self) -> HealthReport:
        return self._health_from_probe(lambda: self.fetch_arrivees(date(2026, 9, 29)))


class TurfBzhCollector(NotOperationalCollector):
    key = "turf_bzh"
    reason = "Exclu : le projet n'utilise que des API gratuites."
