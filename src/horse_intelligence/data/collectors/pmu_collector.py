"""Connecteur PMU turfinfo (source primaire).

Endpoints (contrat vérifié le 2026-10-30, voir data/schemas/pmu.py) :
  /programme/{JJMMAAAA}
  /programme/{JJMMAAAA}/R{r}/C{c}/participants[?specialisation=INTERNET]
  /programme/{JJMMAAAA}/R{r}/C{c}/rapports-definitifs[?specialisation=INTERNET]

Le PMU tient deux masses de paris aux cotes distinctes : OFFLINE (points de vente,
réponse par défaut) et INTERNET. Les courses étrangères n'existent souvent que sur
la masse INTERNET.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from pydantic import TypeAdapter, ValidationError

from horse_intelligence.data.collectors.base import (
    BaseCollector,
    ContractError,
    HealthReport,
    RawResponse,
)
from horse_intelligence.data.schemas.pmu import (
    PmuParticipants,
    PmuProgramme,
    PmuRapportPari,
)

_RAPPORTS = TypeAdapter(list[PmuRapportPari])
POOLS = ("OFFLINE", "INTERNET")


def _pool_params(pool: str) -> dict[str, str]:
    if pool not in POOLS:
        raise ValueError(f"masse de paris inconnue : {pool}")
    return {"specialisation": "INTERNET"} if pool == "INTERNET" else {}


def pmu_date(day: date) -> str:
    return day.strftime("%d%m%Y")


def _decode(raw: RawResponse) -> Any:
    try:
        return json.loads(raw.body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"réponse non JSON sur {raw.endpoint}") from exc


class PmuCollector(BaseCollector):
    key = "pmu"

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}/{path.lstrip('/')}"

    # --- récupération brute -------------------------------------------------
    def fetch_programme_raw(self, day: date) -> RawResponse:
        return self._fetch("programme", self._url(f"programme/{pmu_date(day)}"))

    @property
    def pools(self) -> tuple[str, ...]:
        configured = self.config.extra.get("pools") or POOLS
        return tuple(p for p in configured if p in POOLS)

    def fetch_participants_raw(self, day: date, reunion: int, course: int,
                               pool: str = "OFFLINE") -> RawResponse:
        path = f"programme/{pmu_date(day)}/R{reunion}/C{course}/participants"
        return self._fetch("participants", self._url(path), _pool_params(pool))

    def fetch_rapports_raw(self, day: date, reunion: int, course: int,
                           pool: str = "OFFLINE") -> RawResponse:
        path = f"programme/{pmu_date(day)}/R{reunion}/C{course}/rapports-definitifs"
        return self._fetch("rapports-definitifs", self._url(path), _pool_params(pool))

    # --- décodage selon contrat ---------------------------------------------
    @staticmethod
    def parse_programme(raw: RawResponse) -> PmuProgramme:
        try:
            return PmuProgramme.model_validate(_decode(raw))
        except ValidationError as exc:
            raise ContractError(f"programme : {exc.error_count()} écart(s) de format") from exc

    @staticmethod
    def parse_participants(raw: RawResponse) -> PmuParticipants:
        try:
            return PmuParticipants.model_validate(_decode(raw))
        except ValidationError as exc:
            raise ContractError(f"participants : {exc.error_count()} écart(s) de format") from exc

    @staticmethod
    def parse_rapports(raw: RawResponse) -> list[PmuRapportPari]:
        try:
            return _RAPPORTS.validate_python(_decode(raw))
        except ValidationError as exc:
            raise ContractError(f"rapports : {exc.error_count()} écart(s) de format") from exc

    # --- statut -------------------------------------------------------------
    def health(self) -> HealthReport:
        today = date.today()
        return self._health_from_probe(
            lambda: self.parse_programme(self.fetch_programme_raw(today))
        )
