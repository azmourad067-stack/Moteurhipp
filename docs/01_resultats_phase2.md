# Phase 2 — premiers résultats (2026-10-30)

> Résultats **préliminaires**, sur un historique court (40 jours). Ils décrivent
> l'état du moteur et ne démontrent aucune capacité prédictive ou économique.

## Données

- Période : 20/09/2026 → 29/10/2026, courses **françaises** uniquement.
- 1 510 courses avec résultat ; 1 263 avec des cotes de référence publiées avant
  l'instant de décision (départ − 10 min), masse Internet.
- Audit des agrégats de la source : 1 459 gagnants contrôlés, **aucun signe de mise à
  jour post-course** de la musique ni des cumuls de carrière.
- Règle des places payées vérifiée sur les rapports réels : elle dépend des partants
  **déclarés** (8 déclarés dont 1 non-partant → 3 places).

## Protocole

Validation glissante : 3 plis de 7 jours testés (608 courses avec marché), chacun
appris sur tous les jours antérieurs. Les 7 derniers jours d'apprentissage servent à
calibrer et à combiner les modèles. Tous les modèles sont évalués sur le **même** échantillon.

## Résultats hors échantillon (608 courses)

| Modèle | Log-loss | Brier | ECE | Top-1 | Gagnant dans le top-3 | Δ log-loss vs marché calibré [IC 95 %] |
|---|---|---|---|---|---|---|
| Uniforme | 2,424 | 0,907 | 0,001 | 6,7 % | 26,6 % | +0,362 [+0,295 ; +0,428] |
| Marché brut | 2,061 | 0,829 | 0,008 | 27,3 % | 59,9 % | −0,001 [−0,005 ; +0,003] |
| **Marché calibré (référence)** | 2,062 | 0,830 | 0,010 | 27,3 % | 59,9 % | — |
| Logit conditionnel (forme) | 2,248 | 0,878 | 0,015 | 21,4 % | 50,8 % | +0,185 [+0,130 ; +0,241] |
| LightGBM (forme) | 2,235 | 0,871 | 0,009 | 21,7 % | 52,3 % | +0,173 [+0,115 ; +0,230] |
| LightGBM + marché | 2,060 | 0,831 | 0,009 | 27,1 % | 58,2 % | −0,002 [−0,017 ; +0,013] |

## Lecture

1. **Le marché reste la référence.** Les modèles fondés sur la seule forme sont nettement
   moins bons que les cotes, mais bien meilleurs que le hasard : ils captent un vrai
   signal, que le marché intègre déjà.
2. **LightGBM + marché n'apporte pas d'amélioration démontrée** : l'écart est nul à la
   précision près (IC à cheval sur 0). Il reste donc **candidat** et l'interface ne signale
   aucune « valeur ».
3. Le marché calibré ne fait pas mieux que le marché brut sur cette période : la
   correction du biais favori/outsider demande davantage de données.
4. Variables à faible couverture : l'historique interne (jours depuis la dernière
   course, forme dans notre base) ne couvre que 40 jours. Il prendra du poids avec
   12 mois de collecte.

## Prochaines étapes

- Étendre l'historique à 12 mois (collecte longue, hors de cet environnement).
- Collecter des cotes intermédiaires (H-60, H-30, H-5) à partir de maintenant.
- Phase 3 : incertitude par bootstrap, détection d'anomalies, découverte de sous-groupes
  avec validation hors échantillon.
