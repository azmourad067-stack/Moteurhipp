"""Crée le schéma de la base et enregistre les sources configurées.

Usage : python scripts/initialize_database.py
"""

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging, redact
from horse_intelligence.services.context import build_context


def main() -> None:
    configure_logging()
    ctx = build_context()
    print(f"Base initialisée : {redact(ctx.settings.database_url)}")
    print(f"Sources enregistrées : {', '.join(ctx.settings.sources)}")


if __name__ == "__main__":
    main()
