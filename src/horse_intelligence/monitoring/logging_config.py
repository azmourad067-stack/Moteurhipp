"""Journalisation avec masquage des informations sensibles."""

from __future__ import annotations

import logging
import re

_SENSITIVE_PATTERNS = [
    # identifiants dans une URL de connexion : scheme://user:password@host
    (re.compile(r"(\w+://[^:/\s]+:)([^@\s]+)(@)"), r"\1***\3"),
    # paramètres de requête sensibles
    (re.compile(r"((?:api[_-]?key|token|apikey|password|secret)=)[^&\s]+", re.I), r"\1***"),
    # en-têtes d'autorisation
    (re.compile(r"(authorization['\"]?\s*[:=]\s*['\"]?)(bearer\s+)?[^\s'\",]+", re.I), r"\1\2***"),
]


def redact(text: str) -> str:
    for pattern, repl in _SENSITIVE_PATTERNS:
        text = pattern.sub(repl, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        return True


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(f, RedactingFilter) for h in root.handlers for f in h.filters):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    handler.addFilter(RedactingFilter())
    root.addHandler(handler)
    root.setLevel(level)
