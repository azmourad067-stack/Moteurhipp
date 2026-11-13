"""Nettoyage des chaînes et identifiants."""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def canonical_name(name: str | None) -> str | None:
    """Nom canonique pour la résolution d'identité : majuscules, sans accents ni ponctuation.

    « J.-M. Bazire » et « JM BAZIRE » donnent tous deux « J M BAZIRE » / « JM BAZIRE » :
    les initiales collées ou séparées restent distinctes, la résolution fine des alias
    est traitée par la table d'alias (phase ultérieure), jamais par devinette.
    """
    if name is None:
        return None
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c)).upper()
    text = text.replace(".", "")
    text = _NON_ALNUM.sub(" ", text).strip()
    return text or None


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = " ".join(value.split())
    return value or None
