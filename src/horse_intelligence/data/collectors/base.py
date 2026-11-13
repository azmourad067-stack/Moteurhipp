"""Socle commun des connecteurs : HTTP robuste, statut, traçabilité des réponses brutes."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import requests

from horse_intelligence.config.settings import SourceConfig

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class SourceStatus(StrEnum):
    OPERATIONAL = "opérationnelle"
    DEGRADED = "dégradée"
    UNAVAILABLE = "indisponible"
    NOT_CONFIGURED = "non configurée"
    NOT_OPERATIONAL = "non opérationnelle"


class CollectorError(Exception):
    """Erreur de collecte présentable à l'utilisateur."""


class SourceUnavailableError(CollectorError):
    pass


class NotFoundError(CollectorError):
    """La ressource n'existe pas chez le fournisseur (ex. course sans données)."""


class NotOperationalError(CollectorError):
    """Connecteur présent mais volontairement inactif."""


class ContractError(CollectorError):
    """La réponse ne correspond pas au format attendu (dérive de schéma)."""


@dataclass(frozen=True)
class RawResponse:
    source: str
    endpoint: str
    url: str
    params: dict[str, Any]
    fetched_at: datetime
    http_status: int
    body: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()


@dataclass
class HealthReport:
    source: str
    status: SourceStatus
    checked_at: datetime
    message: str
    latency_ms: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


RawSink = Callable[[RawResponse], None]


class RateLimiter:
    """Impose un délai minimal entre deux requêtes vers un même fournisseur."""

    def __init__(self, min_interval_s: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None:
                remaining = self.min_interval_s - (now - self._last)
                if remaining > 0:
                    self._sleep(remaining)
                    now = self._clock()
            self._last = now


class HttpClient:
    """Client HTTP : délai d'attente, nouvelles tentatives bornées, limitation de débit."""

    def __init__(self, config: SourceConfig, user_agent: str,
                 session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 backoff_base_s: float = 1.0) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self._sleep = sleep
        self.backoff_base_s = backoff_base_s
        self.rate_limiter = RateLimiter(config.min_interval_s, sleep=sleep)

    def get(self, endpoint: str, url: str, params: dict[str, Any] | None = None) -> RawResponse:
        params = params or {}
        attempts = max(1, self.config.max_retries + 1)
        last_error: str = ""
        for attempt in range(1, attempts + 1):
            self.rate_limiter.wait()
            try:
                resp = self.session.get(url, params=params, timeout=self.config.timeout_s,
                                        allow_redirects=False)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}"
                logger.warning("%s: tentative %d/%d échouée (%s)", self.config.key, attempt,
                               attempts, last_error)
            else:
                if resp.status_code == 200:
                    return RawResponse(
                        source=self.config.key, endpoint=endpoint, url=url, params=params,
                        fetched_at=datetime.now(UTC), http_status=200, body=resp.content,
                    )
                if resp.status_code in (204, 404):
                    raise NotFoundError(f"{self.config.key}: ressource absente ({endpoint})")
                if resp.status_code not in RETRYABLE_STATUS:
                    raise SourceUnavailableError(
                        f"{self.config.key}: réponse HTTP {resp.status_code} sur {endpoint}"
                    )
                last_error = f"HTTP {resp.status_code}"
                logger.warning("%s: tentative %d/%d — %s", self.config.key, attempt, attempts,
                               last_error)
            if attempt < attempts:
                self._sleep(self.backoff_base_s * 2 ** (attempt - 1))
        raise SourceUnavailableError(
            f"{self.config.key}: source injoignable après {attempts} tentatives ({last_error})"
        )


class BaseCollector(ABC):
    """Interface commune à tous les fournisseurs."""

    key: str

    def __init__(self, config: SourceConfig, client: HttpClient | None = None,
                 raw_sink: RawSink | None = None, user_agent: str = "horse-intelligence") -> None:
        self.config = config
        self.client = client or HttpClient(config, user_agent)
        self.raw_sink = raw_sink

    def _fetch(self, endpoint: str, url: str, params: dict[str, Any] | None = None) -> RawResponse:
        if not self.config.enabled:
            raise NotOperationalError(f"{self.config.name} est désactivée dans la configuration.")
        raw = self.client.get(endpoint, url, params)
        if self.raw_sink is not None:
            self.raw_sink(raw)
        return raw

    @abstractmethod
    def health(self) -> HealthReport:
        """Vérifie la disponibilité de la source (une requête légère au plus)."""

    def _health_from_probe(self, probe: Callable[[], object]) -> HealthReport:
        now = datetime.now(UTC)
        if not self.config.enabled:
            return HealthReport(self.config.key, SourceStatus.NOT_OPERATIONAL, now,
                                "Source désactivée dans la configuration.")
        start = time.monotonic()
        try:
            probe()
        except NotFoundError as exc:
            return HealthReport(self.config.key, SourceStatus.DEGRADED, now, str(exc))
        except ContractError as exc:
            return HealthReport(self.config.key, SourceStatus.DEGRADED, now,
                                f"Format inattendu : {exc}")
        except CollectorError as exc:
            return HealthReport(self.config.key, SourceStatus.UNAVAILABLE, now, str(exc))
        latency = (time.monotonic() - start) * 1000
        return HealthReport(self.config.key, SourceStatus.OPERATIONAL, now, "Source joignable.",
                            latency_ms=latency)


class NotOperationalCollector(BaseCollector):
    """Connecteur identifié mais non opérationnel tant qu'il n'est pas audité/configuré."""

    reason: str = "Connecteur non opérationnel."

    def health(self) -> HealthReport:
        return HealthReport(self.config.key, SourceStatus.NOT_OPERATIONAL, datetime.now(UTC),
                            self.reason)
