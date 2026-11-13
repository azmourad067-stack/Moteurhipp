"""Laboratoire de découverte : patterns, acteurs, anomalies, dérive.

Usage : python scripts/run_discovery.py [--confirm-share 0.4]
Nécessite les dépendances d'entraînement : pip install -r requirements-train.txt
"""

import argparse
import sys

import _bootstrap  # noqa: F401
from horse_intelligence.monitoring.logging_config import configure_logging
from horse_intelligence.services.context import build_context
from horse_intelligence.services.discovery import run_lab


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-share", type=float, default=0.4)
    args = parser.parse_args()
    configure_logging()
    ctx = build_context()
    try:
        rep = run_lab(ctx.repo.engine, args.confirm_share)
    except ValueError as exc:
        print(f"Laboratoire impossible : {exc}", file=sys.stderr)
        return 2
    print(f"Découverte {rep.discovery_period[0]} → {rep.discovery_period[1]} ; "
          f"confirmation {rep.confirmation_period[0]} → {rep.confirmation_period[1]}")
    print(f"{rep.n_tests} sous-groupes testés ; {len(rep.patterns)} retenus après correction "
          "des tests multiples (FDR 10 %)")
    for p in rep.patterns[:15]:
        oos = f"{p['oos_ae']:.2f}" if p.get("oos_ae") is not None else "—"
        print(f"  [{p['status']}] {p['label']}: A/E={p['ae']:.2f} "
              f"[{p['ae_low']:.2f}; {p['ae_high']:.2f}] q={p['q_value']:.3f} "
              f"n={p['n']} | confirmation A/E={oos}")
    for title, a in rep.actors.items():
        print(f"{title}: {a['dispersion']}")
        print(f"  lissage a={a['prior_a']:.1f} ({a['n_entities']} acteurs) — "
              f"{a['persistence']['message']} (rho={a['persistence']['spearman']:.3f}, "
              f"p={a['persistence']['p_value']:.3f})")
    t = rep.anomalies["test"]
    print(f"Anomalies (top 5 %, confirmation) : A/E={t['ae']:.2f} "
          f"[{t['ae_low']:.2f}; {t['ae_high']:.2f}] p={t['p_value']:.3f} — {t['message']}")
    strong = [d for d in rep.drift["psi"] if d["niveau"] in ("à surveiller", "dérive forte")]
    print(f"Dérive : {len(strong)} variable(s) à surveiller ou en dérive forte "
          f"({', '.join(d['feature'] for d in strong) or 'aucune'})")
    for w in rep.warnings:
        print(f"⚠ {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
