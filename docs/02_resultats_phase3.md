# Phase 3 — résultats du Laboratoire IA (2026-10-30)

> Même historique qu'en phase 2 : 40 jours de courses françaises (1 510 courses).
> Découverte : 20/09 → 13/10/2026 ; confirmation : 14/10 → 29/10/2026.

| Analyse | Résultat |
|---|---|
| Sous-groupes testés contre les cotes | 468 (conditions simples et paires) |
| Retenus après correction des tests multiples (FDR 10 %) | **0** |
| Jockeys / drivers (248) | Aucune dispersion de réussite au-delà du hasard une fois les cotes prises en compte (test du rapport de vraisemblance, p = 1,00) ; pas de persistance d'une période à l'autre (ρ = 0,12, p = 0,12) |
| Entraîneurs (335) | Idem (p = 1,00 ; ρ = 0,07, p = 0,23) |
| 5 % de partants les plus atypiques (Isolation Forest) | A/E = 0,83 [0,63 ; 1,08], p = 0,16 : pas d'écart significatif aux cotes |
| Dérive | Seule la distance des courses change notablement (programme saisonnier) ; les variables d'historique interne évoluent mécaniquement avec la collecte |

## Lecture

Sur 40 jours, **le marché PMU intègre déjà toute l'information testée** : forme,
monte, entraîneur, corde, changement de driver, profils atypiques. Ce résultat
négatif est informatif. Il confirme que le protocole ne fabrique pas de faux
signaux : ses tests sur données simulées montrent qu'il retrouve un effet réellement
injecté et qu'il ne valide rien sur du bruit pur.

Avec 40 jours, la puissance statistique reste limitée : un effet de l'ordre de
10 % sur l'A/E d'un sous-groupe ne serait pas détectable. L'historique de 12 mois
et les cotes intermédiaires (H-60/H-30/H-5) sont les prochains leviers.

## Garde-fous vérifiés par les tests

- Un sous-groupe sur lequel le marché se trompe vraiment (effet injecté ×1,6) est
  détecté puis **validé hors échantillon** ; sur des données sans effet, rien n'est validé.
- Le test de dispersion bayésien produit moins de 10 % de faux positifs sous
  l'hypothèse nulle (200 simulations) et détecte une vraie dispersion (p < 0,001).
- Le test de persistance retrouve des jockeys réellement sous-estimés par le marché.
- Les contributions explicatives somment à zéro dans chaque course (seules les
  différences entre partants comptent).
