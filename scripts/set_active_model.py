"""Retour arrière contrôlé : rend active une version précédente d'un modèle.

Usage :
  python scripts/set_active_model.py --family lgbm_market            # liste les versions
  python scripts/set_active_model.py --family lgbm_market --version 20261030-112415 --yes

L'action est sensible : elle exige --yes et vérifie l'intégrité de l'artefact.
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from horse_intelligence.models.registry import activate_version, list_versions
from horse_intelligence.services.context import build_context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True,
                        choices=["lgbm_market", "lgbm_form", "market_calibrated"])
    parser.add_argument("--version")
    parser.add_argument("--yes", action="store_true", help="confirme l'activation")
    args = parser.parse_args()
    ctx = build_context()
    if not args.version:
        for v in list_versions(ctx.repo.engine, args.family):
            print(f"{v['version']}  {v['status']:10s}  entraîné jusqu'au {v['train_end']}")
        return 0
    if not args.yes:
        print("Action sensible : ajoutez --yes pour confirmer l'activation.", file=sys.stderr)
        return 2
    try:
        activate_version(ctx.repo.engine, args.family, args.version)
    except ValueError as exc:
        print(f"Impossible : {exc}", file=sys.stderr)
        return 1
    print(f"{args.family} {args.version} est désormais actif.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
