# Phase 5 — validation : rentabilité théorique, robustesse, fuites (2026-10-30)

> Prédictions **hors échantillon** du 11/10 au 29/10/2026 (608 courses françaises avec
> cote de référence, toutes avec rapports définitifs officiels collectés).

## Protocole de simulation

- **Décision** : probabilités issues de la validation glissante (modèle appris sans la
  course) et cote de référence publiée avant l'instant de décision (départ − 10 min).
- **Règlement** : rapport définitif officiel du pari mutuel (masse Internet), mise de 1 €.
- **Choix de la règle** : 39 règles (seuils d'espérance, de probabilité, de cote ;
  simple gagnant ou placé) évaluées sur la période de **sélection** (11/10 → 19/10) ;
  la meilleure est appliquée telle quelle au **test** (20/10 → 29/10). Seul le test compte.

## Résultats

| Stratégie | Paris | ROI | IC 95 % | Réussite | Pire série |
|---|---|---|---|---|---|
| Repère : favori du marché, gagnant (toute la période) | 608 | −15,5 % | [−28,1 ; −1,8] | 27 % | 13 |
| Repère : favori du marché, placé (toute la période) | 608 | −8,0 % | [−15,5 ; −0,2] | 55 % | 6 |
| LightGBM + marché — sélection | 46 | +9,1 % | [−72,8 ; +126,7] | 15 % | 14 |
| **LightGBM + marché — TEST** | 161 | **−36,0 %** | [−68,4 ; +3,9] | 11 % | 60 |
| LightGBM forme — sélection | 279 | +27,0 % | [−38,1 ; +107,7] | 8 % | 29 |
| **LightGBM forme — TEST** | 261 | **−17,7 %** | [−49,2 ; +19,2] | 9 % | 46 |
| Marché calibré, placé — sélection | 211 | +5,0 % | [−5,4 ; +16,2] | 69 % | 3 |
| **Marché calibré, placé — TEST** | 184 | **−12,6 %** | [−23,8 ; −1,2] | 56 % | 6 |

**Aucune stratégie n'est rentable hors échantillon.** Les ROI positifs de la période de
sélection, avec des intervalles très larges, sont le produit du choix de la meilleure
règle parmi 39 : un cas d'école de surapprentissage, que le protocole met en évidence.
Les repères montrent le coût structurel du pari mutuel : parier sur chaque favori perd
environ 15 % en simple gagnant et 8 % en simple placé.

## Robustesse

| Contrôle | Résultat |
|---|---|
| Stabilité par période | LightGBM + marché meilleur que le marché calibré sur 2 des 3 périodes (écarts non significatifs) |
| Aléa d'apprentissage (3 graines) | Écart de log-loss de 0,0035 |
| 20 % de valeurs masquées | Log-loss −0,0037 (pas de dégradation) ; probabilités toujours valides (somme à 100 %) |
| Segments | Discipline, hippodrome, distance, taille du champ : aucun segment significativement meilleur |

## Audit des fuites (base réelle)

| Contrôle | Résultat |
|---|---|
| Cotes antérieures à la décision | ✅ 14 996 cotes, toutes publiées avant départ − 10 min |
| Permutation des résultats futurs | ✅ Aucune variable passée ne change |
| Variable trop prédictive | ✅ AUC maximale d'une variable seule : 0,63 (seuil d'alerte 0,90) |
| Musique et cumuls de la source | ✅ Publiés avant la course |
| Engagements | ℹ️ 100 % collectés a posteriori (historique rattrapé) : seuls les champs déclaratifs sont utilisés |

## Conclusion

Les trois objectifs sont désormais mesurés séparément, et aucun ne montre d'avantage :
- **performance prédictive** : bien au-dessus du hasard, au niveau du marché ;
- **qualité probabiliste** : bonne calibration, sans amélioration significative
  par rapport aux cotes ;
- **valeur économique** : aucune règle rentable hors échantillon.

Le système ne produit donc aucun signal de pari, et c'est le comportement attendu
d'un outil honnête. Ces conclusions portent sur 19 jours de test : elles seront
réévaluées avec 12 mois d'historique et des cotes intermédiaires.
