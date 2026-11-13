"""Connecteur météo Open-Meteo (gratuit, usage non commercial).

Les coordonnées des hippodromes ne sont pas fournies par le PMU : elles doivent être
renseignées dans la table `racecourses` (lat/lon). Sans coordonnées, aucune météo
n'est récupérée — aucune valeur n'est devinée.

Distinction temporelle : une prévision (`forecast`) émise avant la course est
utilisable pour une prédiction ; une observation (`observed`, archive) ne l'est
qu'en analyse a posteriori.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime

from horse_intelligence.data.collectors.base import (
    BaseCollector,
    ContractError,
    HealthReport,
)

HOURLY_VARS = ("temperature_2m", "precipitation", "wind_speed_10m")


@dataclass(frozen=True)
class WeatherPoint:
    valid_time_utc: datetime
    temp_c: float | None
    precip_mm: float | None
    wind_ms: float | None
    kind: str  # "forecast" | "observed"
    issued_at: datetime


class WeatherCollector(BaseCollector):
    key = "open_meteo"

    def _parse(self, body: bytes, kind: str, issued_at: datetime) -> list[WeatherPoint]:
        try:
            payload = json.loads(body)
            hourly = payload["hourly"]
            times = hourly["time"]
            temps = hourly.get("temperature_2m", [None] * len(times))
            precs = hourly.get("precipitation", [None] * len(times))
            winds = hourly.get("wind_speed_10m", [None] * len(times))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ContractError("réponse Open-Meteo inattendue") from exc
        points = []
        for t, temp, prec, wind in zip(times, temps, precs, winds, strict=True):
            points.append(WeatherPoint(
                valid_time_utc=datetime.fromisoformat(t).replace(tzinfo=UTC),
                temp_c=temp, precip_mm=prec,
                # wind_speed_unit=ms demandé explicitement dans la requête
                wind_ms=wind, kind=kind, issued_at=issued_at,
            ))
        return points

    def _params(self, lat: float, lon: float, start: date, end: date) -> dict[str, str]:
        return {
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "hourly": ",".join(HOURLY_VARS), "wind_speed_unit": "ms", "timezone": "UTC",
            "start_date": start.isoformat(), "end_date": end.isoformat(),
        }

    def forecast(self, lat: float, lon: float, day: date) -> list[WeatherPoint]:
        url = f"{self.config.base_url.rstrip('/')}/forecast"
        raw = self._fetch("forecast", url, self._params(lat, lon, day, day))
        return self._parse(raw.body, "forecast", raw.fetched_at)

    def observed(self, lat: float, lon: float, start: date, end: date) -> list[WeatherPoint]:
        archive = str(self.config.extra.get("archive_url", self.config.base_url))
        raw = self._fetch("archive", f"{archive.rstrip('/')}/archive",
                          self._params(lat, lon, start, end))
        return self._parse(raw.body, "observed", raw.fetched_at)

    def health(self) -> HealthReport:
        # Paris, uniquement pour tester la joignabilité du service
        return self._health_from_probe(lambda: self.forecast(48.86, 2.35, date.today()))
