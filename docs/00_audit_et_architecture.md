# Horse Intelligence — Audit de faisabilité et architecture proposée

> Statut : **proposition à valider** avant tout développement (phase 0).
> Date : 2026-10-30.
> Aucune ligne de code applicatif n'a encore été écrite : ce document fixe le
> cahier des charges technique, les hypothèses à vérifier et les décisions
> attendues du porteur de projet (section 10).

---

## 1. Résumé exécutif

| Question | Réponse courte |
|---|---|
| Le projet est-il faisable ? | Oui, à condition d'accepter que **la valeur vient d'abord de la base de données construite dans le temps**, pas des algorithmes. |
| Principal risque | **Données** : pas d'API officielle documentée et licenciée ; cotes horodatées avant départ indisponibles en historique profond. |
| Deuxième risque | **Fuite de données** : plusieurs champs PMU (musique, gains, cotes « directes ») sont recalculés après coup. Ils doivent être horodatés à la collecte. |
| Référence à battre | **Le marché** (probabilités implicites des cotes, marge retirée). Battre un modèle naïf est facile ; battre le marché en log-loss est l'objectif réel et peut échouer. |
| Modèle cœur | Modèle **logit conditionnel par course** (softmax intra-course) sur gradient boosting, puis combinaison bayésienne/stacking avec le marché (approche de type Benter). |
| Streamlit Community Cloud | Utilisable **uniquement pour l'inférence et la consultation**. Collecte planifiée et entraînement : GitHub Actions + stockage externe. PyTorch exclu du runtime Streamlit. |
| Deep learning | Prévu, mais **conditionné** à un volume minimal (≈ 50 000 partants historisés) et à un gain hors échantillon démontré face au GBM. |

---

## 2. Audit des sources de données

### 2.1. Test d'accès réalisé

Depuis l'environnement de développement actuel, **les quatre hôtes ont été bloqués
par la politique réseau du conteneur** (proxy 403 / DNS refusé) :

- `online.turfinfo.api.pmu.fr`
- `open-pmu-api.vercel.app`
- `www.turf.bzh`
- `www.pmu.fr`

Conséquence : **aucune affirmation ci-dessous sur le format des réponses n'a été
vérifiée en direct dans cette session.** Les éléments marqués *(à vérifier)*
proviennent de la connaissance publique de ces services et seront confirmés par
des tests de contrat automatisés dès que l'accès réseau sera ouvert (voir §10, D1).

### 2.2. Évaluation par source

| Source | Nature | Contenu attendu | Auth | Statut juridique | Fiabilité | Verdict |
|---|---|---|---|---|---|---|
| **PMU turfinfo** `/rest/client/1` | API JSON non documentée publiquement, utilisée par les frontaux PMU | Programme du jour (`/programme/{JJMMAAAA}`), partants (`/…/R{n}/C{n}/participants`), performances détaillées, rapports définitifs, combinaisons | Aucune connue *(à vérifier)* | **Pas de licence de réutilisation publiée connue.** Usage personnel/analytique à faible fréquence ; pas de redistribution. CGU pmu.fr à lire. | Élevée sur le fond (source primaire), mais **contrat non garanti** : champs pouvant changer sans préavis | **Source primaire**, connecteur avec tests de contrat et détection de dérive de schéma |
| **Open PMU API** (vercel) | Projet communautaire : API devant une base PostgreSQL privée | ~1 course/jour (Quinté+ probable) 2004–2026, arrivée, chevaux classés, 3 cotes non horodatées | Aucune | MIT sur le code seulement ; aucun droit sur les données | Mainteneur unique, mises à jour manuelles | **Recoupement des arrivées uniquement** (voir §2.5) |
| **Turf.bzh** | Service tiers avec documentation d'API | Historiques, statistiques *(à vérifier)* | Clé API probable, offre potentiellement **payante** *(à vérifier)* | CGU propres à lire | Inconnue | **À auditer** ; connecteur livré « non opérationnel jusqu'à configuration » |
| **Open-Meteo** (proposé) | API météo ouverte, archive historique | Température, précipitations, vent, horaires ; archive depuis 1940 | Aucune (usage non commercial) | CC BY 4.0, non commercial gratuit | Bonne | **Recommandée** pour la météo, rattachée aux coordonnées des hippodromes |
| **France Galop / LeTrot** | Sites officiels des sociétés mères | Fiches chevaux, pedigree, terrain officiel | Pas d'API publique connue | Scraping à proscrire sans autorisation | Élevée | **Hors périmètre v1** ; à étudier par demande formelle |
| **Recherche documentaire (RAG)** | Presse spécialisée, communiqués | Changements tardifs, contexte | Clé API LLM (**payante**) + éventuel moteur de recherche | Données non fiables par défaut | Variable | **Phase 3**, hors du chemin critique des prédictions |

### 2.3. Champs PMU attendus et risque de fuite

Champs habituellement présents dans la réponse `participants` *(à vérifier)* et
leur classement temporel :

| Champ (nom attendu) | Signification | Disponible avant départ ? | Traitement |
|---|---|---|---|
| `numPmu`, `nom`, `age`, `sexe`, `race` | Identité | Oui | Clé naturelle cheval = nom normalisé + âge/sexe, puis identifiant interne |
| `driver` / `jockey`, `entraineur`, `proprietaire` | Acteurs | Oui (peut changer jusqu'au départ) | Historiser chaque changement |
| `placeCorde`, `handicapPoids`, `handicapDistance` | Conditions | Oui | Feature directe |
| `oeilleres`, `deferre` | Équipement / ferrure (trot) | Oui, modifiable tard | Historiser |
| `statut` (`PARTANT` / `NON_PARTANT`) | Non-partants | Évolue jusqu'au départ | Snapshot horodaté obligatoire |
| `musique` | Résumé des derniers résultats | **Ambigu** : une requête faite *après* la course peut renvoyer une musique mise à jour | **Interdit en backtest** sauf snapshot pré-départ ; on reconstruit la forme depuis nos propres résultats |
| `nombreCourses`, `nombreVictoires`, `gainsParticipant` | Cumuls carrière | **Ambigu** (même problème) | Idem : recalculés en interne « as-of » |
| `dernierRapportReference` | Cote de référence (matin) | Oui, si horodatage confirmé | Candidat principal pour le backtest historique |
| `dernierRapportDirect` | Dernière cote observée | **Non en historique** : c'est souvent la cote finale | **Exclu** des features de backtest ; utilisé seulement s'il a été collecté par nous avant l'heure de décision |
| `ordreArrivee`, `tempsObtenu`, `reductionKilometrique` | Résultat | **Non** | Cible / historique des courses passées uniquement |

**Règle structurante** : toute la reconstruction de forme (fenêtres court/moyen/long
terme, régularité, repos, progression) est calculée **à partir de notre propre
table de résultats**, filtrée par `date_course < date_course_cible`, et non à
partir des agrégats fournis par la source.

### 2.4. Le problème des cotes historiques

C'est la contrainte la plus importante pour la partie « valeur » :

1. Les cotes intermédiaires (H-60, H-30, H-5) **ne sont pas disponibles en
   historique** ; elles doivent être **collectées par nous à partir du lancement**.
2. Pour l'historique antérieur, seule une cote dont l'horodatage est démontré
   (cote de référence du matin, si confirmée) peut servir aux simulations.
3. Le PMU est un **pari mutuel** : le rapport final dépend des mises après la
   décision et inclut un prélèvement (de l'ordre de 15 à 30 % selon le pari,
   *à vérifier par type de pari*). Toute espérance calculée sur une cote
   antérieure au départ est **une estimation**, pas un gain réalisable.

Conséquence : la détection de valeur en conditions réalistes ne pourra être
évaluée sérieusement qu'après **plusieurs mois de collecte de snapshots**.
Le document le dira explicitement dans l'interface.

### 2.5. Audit du code d'Open PMU API (2026-10-30)

Source : https://github.com/open-pmu-api/open-pmu-api (dernier commit du 2026-10-06).
Audit du code et de l'historique Git, en lecture seule ; l'API elle-même reste
inaccessible depuis l'environnement de développement.

| Point | Constat |
|---|---|
| Nature | Fonction serverless Vercel (TypeScript) devant une base PostgreSQL privée. Un seul endpoint : `GET /api/arrivees?date=MM/DD/YYYY` (ou `hippo=`, `prix=`, en recherche `LIKE`). |
| Couverture | Le dernier export JSON public de l'historique Git (juillet 2026) contient **8 196 courses de 2004 à 2026, soit exactement une par jour** (365/an, ~16 partants en moyenne) : vraisemblablement la course Quinté+ du jour, **pas le programme complet** (~15 000 courses/an). |
| Détail des chevaux | `arrivee_details` ne décrit que les **chevaux classés** (5 sur 16 dans l'exemple du README). Dans l'export de juillet 2026, il est **vide pour 99 % des courses**. Les chevaux non classés manquent : biais de sélection si on l'utilise pour entraîner un modèle. |
| Cotes | `cotes` = trois valeurs **sans horodatage ni définition** (ni moment, ni type). Il est impossible de savoir si la cote finale en fait partie : **inutilisables pour une décision simulée**. |
| Temps | `heure_depart` en heure locale sans fuseau ; seule la date est fiable. |
| Identité | Historique des chevaux relié **par le nom seul** (risque d'homonymes). |
| Mises à jour | Lots manuels irréguliers (ex. « ajout des données du 07/07/2026 au 18/08/2026 »). Données annoncées jusqu'au 29/09/2026. |
| Droits | Licence MIT **sur le code seulement** ; le README précise qu'elle « ne confère aucun droit particulier sur les données ». Provenance des données non documentée. Pas de limite de débit publiée ni de garantie de disponibilité. Mainteneur unique. |

**Décision** : source **secondaire de recoupement des arrivées uniquement**
(`scripts/crosscheck_open_pmu.py`). Ni ses cotes ni ses statistiques dérivées
ne sont importées, et elle ne sert pas d'historique d'entraînement. Le connecteur
est implémenté et testé sur des réponses simulées conformes au contrat publié
(README et `types.ts`). Il reste à le valider contre l'API réelle.


### 2.6. Vérification du contrat PMU contre l'API réelle (2026-10-30)

Une fois l'accès réseau ouvert, le programme, les participants et les rapports
du 29/10/2026 ont été comparés au contrat (52 courses, 7 réunions : France,
Espagne, États-Unis, Chili ; trot attelé, trot monté, obstacle, plat).

| Point | Résultat |
|---|---|
| Noms de champs | Conformes au contrat. `placeCorde` est absent au trot (normal). Champ supplémentaire utile : `poidsConditionMonte` (poids réellement porté après décharge). |
| `handicapPoids`, `poidsConditionMonte` | Hectogrammes (580 → 58,0 kg) — **hypothèse confirmée** |
| `heureDepart`, `dateRapport` | Epoch en millisecondes UTC — **confirmé** |
| `tempsObtenu`, `reductionKilometrique` | Millisecondes (72500 → 1'12"5) — désormais convertis en secondes |
| `gainsCarriere`, `dividendePourUnEuro` | **Centimes d'euro** (620 → 6,20 €). Le code les prenait pour des euros : **corrigé** |
| Gains des chevaux étrangers | En **monnaie locale** (ex. Chili : jusqu'à ~95 M « € »). Comme la devise n'est pas garantie, ces gains ne sont pas stockés pour les courses hors France. |
| Types de pari Internet | Préfixés `E_` (`E_SIMPLE_GAGNANT`) : normalisés, la masse étant enregistrée à part |
| `dernierRapportReference` | Publiée **20 à 32 min avant le départ**, avec son horodatage. Elle reste donc utilisable en backtest à sa date de publication, même récupérée après la course. |
| `dernierRapportDirect` | Horodatée **après** le départ pour une course terminée (= cote finale) → classée `post_start`, jamais utilisée pour une décision |
| Non-partants | `statut = NON_PARTANT` et `incident = NON_PARTANT` ; exclus des résultats |
| Disqualifiés | `ordreArrivee` absent, `incident = DISQUALIFIE_POUR_ALLURE_IRREGULIERE` |
| **Masses de paris** | Deux masses aux cotes **distinctes** : points de vente (réponse par défaut) et Internet (`?specialisation=INTERNET`). Exemple R1C1 : cote de référence du n°3 = 4,9 en points de vente, 3,6 sur Internet. Les courses étrangères n'ont souvent de rapports que sur Internet. Les deux masses sont collectées, et chaque cote ou rapport indique sa masse. |

Open PMU API : l'hébergeur (Vercel) répond **403 Forbidden** à cet environnement,
sans doute une protection anti-robots. Elle n'est **pas contournée** : le
recoupement reste en attente, à tester depuis une autre machine.

---

## 3. Volume et profondeur réalistes

Ordre de grandeur France (à affiner) : ~15 000 à 18 000 courses PMU par an,
~10 à 14 partants par course → **~180 000 à 230 000 lignes partant/an**.

| Historique | Lignes partant | Ce qui est raisonnable |
|---|---|---|
| 3 mois | ~50 000 | Baselines, GBM, calibration |
| 1 an | ~200 000 | GBM solide, walk-forward, premières découvertes |
| 3 ans+ | ~600 000 | Séquences (GRU/Transformer), graphe, autoencodeurs |

Stockage : quelques centaines de Mo en Parquet compressé pour plusieurs années
(hors snapshots de cotes, qui multiplient le volume des cotes par le nombre de
relevés).

Collecte : le backfill historique se fera **à faible cadence** (ex. 1 requête/s
maximum, plage horaire creuse, reprise sur erreur, cache local) pour respecter
le service source.

---

## 4. Schéma de données

### 4.1. Principes

- **Trois couches** : `raw` (réponses JSON brutes, immuables, compressées,
  avec hash) → `staging` (normalisé, typé, validé) → `analytics` (features
  calculées « as-of », Parquet/DuckDB).
- **Bitemporalité** : chaque fait observable porte `observed_at` (moment où
  *nous* l'avons collecté) et, si connu, `published_at` (moment où la source
  l'a publié). Les prédictions et backtests filtrent sur
  `observed_at <= decision_time`.
- **Traçabilité** : toute ligne référence `source_id` + `raw_payload_id`.
- **Pas de valeur inventée** : `NULL` + indicateur de manque ; jamais d'imputation
  silencieuse dans les tables de faits (l'imputation éventuelle se fait dans le
  pipeline de features, documentée et marquée par une variable indicatrice).

### 4.2. Tables (SQLite en prototype, PostgreSQL ensuite, même DDL via SQLAlchemy)

```
data_sources          (source_id, name, base_url, kind, enabled, terms_checked_at, notes)
raw_payloads          (payload_id, source_id, endpoint, params_json, fetched_at,
                       http_status, sha256, storage_uri)           -- corps hors base (Parquet/JSON.gz)

racecourses           (racecourse_id, code_pmu, name, country, lat, lon, track_type)
races                 (race_id, date, reunion_num, course_num, racecourse_id,
                       discipline, specialite, distance_m, corde, prize_eur,
                       conditions_text, declared_runners, scheduled_start_utc,
                       source_id, first_observed_at)
horses                (horse_id, canonical_name, birth_year, sex, breed, sire, dam)
jockeys / trainers / owners (person_id, canonical_name, kind)
entity_aliases        (entity_type, entity_id, alias, source_id)    -- résolution d'identité

race_entries          (entry_id, race_id, horse_id, num_pmu, jockey_id, trainer_id,
                       owner_id, draw, weight_kg, handicap_distance_m,
                       blinkers, shoes, status, observed_at, valid_to)  -- versionné (SCD2)
race_results          (race_id, horse_id, finish_position, status_code, time_s,
                       reduction_km, disqualified, published_at, observed_at)
odds_snapshots        (race_id, horse_id, bet_type, odds, odds_kind
                       ['reference','live','final'], published_at, observed_at)
payouts               (race_id, bet_type, combination, payout_eur, published_at)
weather_conditions    (racecourse_id, valid_time_utc, temp_c, precip_mm, wind_ms,
                       kind ['forecast','observed'], issued_at, observed_at)
racecourse_conditions (race_id, going_label, penetrometer, observed_at)

feature_sets          (feature_set_id, version, code_hash, definition_json, created_at)
model_versions        (model_id, family, version, feature_set_id, train_start, train_end,
                       hyperparams_json, artifact_uri, artifact_sha256, git_commit,
                       status ['candidate','active','retired'], created_at)
model_metrics         (model_id, eval_protocol, period_start, period_end, segment,
                       metric, value, ci_low, ci_high, n_races, n_entries)
predictions           (prediction_id, race_id, model_id, decision_time, created_at,
                       data_status ['verified','partial','insufficient','unavailable'],
                       input_snapshot_hash)
prediction_entries    (prediction_id, horse_id, p_win, p_place, p_win_lo, p_win_hi,
                       rank, model_disagreement, ood_score)
prediction_explanations (prediction_id, horse_id, factor, contribution, direction, text)
detected_patterns     (pattern_id, definition_json, discovered_on_period, n_obs,
                       effect_size, ci_low, ci_high, p_value, q_value_fdr,
                       oos_effect, oos_ci_low, oos_ci_high, status, history_json)
research_documents    (doc_id, url, source_name, published_at, collected_at,
                       reliability, expires_at, content_sha256, summary, structured_json)
data_quality_reports  (report_id, scope, run_at, checks_json, status)
```

Index principaux : `(date, reunion_num, course_num)`, `(horse_id, date)`,
`(race_id, observed_at)` sur les snapshots.

---

## 5. Modèles de référence et modèle cœur

### 5.1. Formulation probabiliste

Une course est un **événement de classement** à `n` partants. On modélise un
score `s_i` par partant et :

- **Victoire** : `P(i gagne) = exp(s_i) / Σ_j exp(s_j)` (logit conditionnel).
  La somme vaut 100 % par construction — pas de renormalisation ad hoc.
- **Classement / places** : modèle **Plackett-Luce** (avec correction de type
  Henery/Stern pour éviter la surestimation des outsiders aux places,
  connue avec Harville), probabilités de place par simulation Monte-Carlo
  vectorisée ou calcul exact pour petits champs. Nombre de places payées
  selon le nombre de partants et le pari (règles PMU paramétrées en config).

### 5.2. Échelle des modèles (chacun doit battre le précédent hors échantillon)

| # | Modèle | Rôle |
|---|---|---|
| B0 | Uniforme `1/n` | Plancher |
| B1 | **Marché** : cotes de référence → probabilités implicites, marge retirée (normalisation proportionnelle et méthode de Shin comparées) | **La vraie référence** |
| B2 | Logit conditionnel régularisé sur ~20 features interprétables (forme, distance, corde, repos, taux acteurs lissés) | Référence statistique transparente |
| M1 | **LightGBM** objectif `lambdarank` / softmax par groupe-course, ou scores binaires puis couche softmax calibrée | Moteur principal non linéaire |
| M2 | XGBoost / CatBoost (CatBoost utile pour catégorielles à forte cardinalité : jockeys, entraîneurs) | Diversité pour l'ensemble |
| M3 | **Combinaison modèle + marché** : logit conditionnel `s = α·s_modèle + β·log(p_marché)` appris en walk-forward | Mesure honnête de l'information *ajoutée* au marché |
| M4 | Modèle bayésien hiérarchique (effets aléatoires cheval/jockey/entraîneur/hippodrome) | Taux lissés + incertitude ; aussi source de features « shrinkées » |
| DL | MLP, GRU sur séquences de courses, Transformer temporel, GNN, autoencodeur | Phase 4, **activés seulement si seuils de données atteints et gain démontré** |

Choix techniques assumés :

- **LightGBM d'abord** (léger, rapide, bon support du ranking) ; XGBoost/CatBoost
  ajoutés seulement s'ils améliorent l'ensemble.
- Bayésien : **PyMC est lourd** ; on démarre par un modèle empirical-Bayes
  (bêta-binomial / mixte) en NumPy/SciPy, PyMC réservé à l'entraînement hors ligne.
- Taux jockey/entraîneur/cheval : toujours **lissés** (shrinkage vers la moyenne
  du groupe) et calculés **strictement avant** la course cible.

### 5.3. Incertitude

- Bootstrap par course (ou ensemble de seeds) → intervalles sur `p_win`.
- Désaccord entre modèles (écart-type des log-scores, entropie croisée).
- Score hors distribution (distance de Mahalanobis / Isolation Forest sur
  l'espace de features ; nouveau cheval sans historique ; hippodrome rare).
- Statut de course : `exploitable` / `incertaine` / `données insuffisantes`.
  Les probabilités sont **arrondies à l'unité de pourcentage** à l'affichage.

---

## 6. Protocoles de validation

### 6.1. Découpage

- Chronologique strict : **train** (périodes anciennes) / **validation**
  (sélection d'hyperparamètres, calibration, stacking) / **test** (gelé,
  évalué une seule fois par version majeure).
- **Walk-forward** : fenêtre d'entraînement expansive, pas mensuel, avec
  **purge** (on exclut les courses entre fin d'entraînement et début de test
  d'un délai d'embargo) pour éviter que des features « as-of » ne chevauchent.
- Groupement par **course** (jamais de partants d'une même course de part et
  d'autre d'un split) et calibration apprise sur la validation uniquement.

### 6.2. Métriques (calculées par segment : discipline, hippodrome, taille de champ, période)

- **Qualité probabiliste (critère principal)** : log-loss multinomiale par course,
  Brier, ECE / courbes de fiabilité, **gain en log-loss vs marché**.
- **Performance prédictive** : Top-1, taux du gagnant dans le top-3 prédit,
  AUC/PR-AUC sur la cible victoire, NDCG.
- **Valeur économique (séparée)** : ROI, nombre de paris, drawdown max, plus
  longue série de pertes, **IC bootstrap par course**, en distinguant cote de
  décision et rapport final.
- Tests de différence entre modèles : bootstrap apparié par course
  (ou test de Diebold-Mariano sur log-loss).

### 6.3. Tests anti-fuite (bloquants en CI)

1. Pour chaque feature d'une course `c`, toute source utilisée a
   `observed_at <= decision_time(c)` et toute course passée utilisée a
   `date < date(c)`.
2. Test de **permutation du futur** : modifier les résultats postérieurs à `c`
   ne doit changer aucune feature de `c`.
3. Les champs listés « interdits en backtest » (§2.3) sont refusés par le
   constructeur de features en mode historique.
4. Reproductibilité : même entrée + même version de modèle ⇒ même sortie
   (hash comparé).

### 6.4. Découverte de patterns (Laboratoire IA)

- Recherche sur période **découverte**, validation sur période **confirmation**
  ultérieure et disjointe, puis suivi « en direct ».
- Effets mesurés **conditionnellement au marché** (résidu vs probabilité
  implicite), sinon on redécouvre ce que les cotes savent déjà.
- Correction **Benjamini-Hochberg** sur toutes les hypothèses testées
  (le nombre testé est enregistré), taille d'effet + IC, effectif minimum.
- Statuts : `détectée → en validation → validée hors échantillon / non confirmée / invalidée`.
- Formulation systématique « associé à », jamais causal.

### 6.5. Promotion d'un modèle

Un candidat ne remplace l'actif que si, sur la même fenêtre walk-forward non vue,
il améliore la log-loss avec un IC bootstrap excluant 0 **et** ne dégrade pas la
calibration. Pas de réentraînement automatique sur quelques courses ; cycle
planifié (ex. mensuel), versionné, avec retour arrière possible.

---

## 7. Limites de Streamlit Community Cloud et réponse architecturale

| Limite (à revérifier dans la doc à jour) | Impact | Réponse |
|---|---|---|
| Mémoire limitée (de l'ordre de 1 à quelques Go) | Pas d'entraînement, pas de gros dataframes | Inférence seule ; requêtes DuckDB sur Parquet filtrées ; modèles LightGBM (quelques Mo) |
| Mise en veille après inactivité, redémarrages | Pas de tâches planifiées fiables | **Collecte et entraînement via GitHub Actions (cron)** |
| Système de fichiers éphémère | SQLite local perdu au redémarrage | Base **PostgreSQL managée** (ex. Supabase/Neon, offre gratuite) + Parquet sur stockage objet ; SQLite seulement en local/dev |
| Taille des dépendances / temps de build | PyTorch CPU ≈ plusieurs centaines de Mo | `requirements.txt` de l'app **sans PyTorch** ; modèles DL exportés en **ONNX** et servis par `onnxruntime` si retenus ; `requirements-train.txt` séparé |
| Taille du dépôt GitHub | Pas de données ni de modèles lourds versionnés | Artefacts sur stockage externe, registre de modèles en base (URI + SHA-256) |
| Pas de secrets dans le code | — | `st.secrets` / variables d'environnement, `.streamlit/secrets.toml` ignoré par Git |

```
                ┌────────────────── GitHub Actions (cron) ──────────────────┐
                │ refresh_data (programme, partants, snapshots de cotes,    │
                │ résultats, météo) → validation → raw + staging            │
                │ train_models (mensuel) → walk-forward → registre          │
                │ run_backtest → rapports                                    │
                └───────────────┬───────────────────────────┬───────────────┘
                                │                           │
                     PostgreSQL managé               Stockage objet (Parquet,
                  (référentiel, prédictions,          payloads bruts, modèles)
                   registre, métriques)                     │
                                └─────────────┬─────────────┘
                                              │ lecture seule (+ prédiction à la demande)
                                   Streamlit Community Cloud
                           (UI, inférence légère, cache st.cache_*)
```

Note : les snapshots de cotes à H-60/H-30/H-5 nécessitent une exécution
planifiée fréquente le jour des courses ; GitHub Actions accepte des cron de
l'ordre de 5 minutes mais **sans garantie de ponctualité**. Si la précision
horaire devient critique, prévoir un petit worker dédié (VPS ou fonction cloud).

---

## 8. Architecture logicielle définitive

Arborescence reprise du cahier des charges, avec les ajustements suivants :

1. **Paquet installable** `horse_intelligence/` sous `src/` (imports stables,
   tests propres) ; `app.py` à la racine pour Streamlit Cloud.
2. **Connecteurs** : interface commune `BaseCollector` (fetch → normalize,
   timeout, retry exponentiel borné, rate limit, journalisation sans secrets,
   `health()`, `freshness()`), un module par fournisseur, plus
   `turfbzh_collector.py` et `open_pmu_collector.py` explicitement marqués
   non opérationnels tant qu'ils ne sont pas configurés/audités.
3. **Contrats de données** : schémas Pydantic pour les réponses brutes
   (détection de dérive de schéma) + validations Pandera/maison sur les tables.
4. **Feature store « as-of »** dans `intelligence/feature_engineering.py`,
   unique point d'accès aux données pour les modèles (garde-fou anti-fuite).
5. `models/ranking.py` (logit conditionnel, Plackett-Luce, places) ajouté :
   c'est la pièce centrale de la cohérence probabiliste.
6. `requirements.txt` (runtime léger) et `requirements-train.txt` (LightGBM,
   CatBoost, Optuna, SHAP, PyTorch…) séparés.
7. Python **3.12** (compatibilité des roues LightGBM/CatBoost/PyTorch).
8. Qualité : Ruff, Mypy (strict sur `data/`, `prediction/`, `evaluation/`),
   Pytest, CI GitHub Actions ; tests de contrat réseau isolés (marqueur
   `network`) pour que la CI ne dépende pas de la disponibilité PMU.

Interface Streamlit (thème sombre anthracite, accents émeraude/bleu électrique) :
Tableau de bord, Courses du jour, Analyse approfondie, Fiche cheval,
Laboratoire IA, Backtesting, Monitoring modèles, Configuration. Chaque page
n'appelle que des services (`prediction/`, `evaluation/`, `data/storage/`) ;
aucune logique métier dans `frontend/`. Actions sensibles (réentraînement,
purge) : non déclenchables depuis l'UI publique en v1, ou derrière
confirmation + secret administrateur.

Hors périmètre v1 (rappel) : aucun pari automatique, aucune connexion à un
compte opérateur. Bandeau jeu responsable et mention « estimations, aucune
garantie de gain ».

---

## 9. Plan de réalisation

| Phase | Livrable testé | Critère de sortie |
|---|---|---|
| **1. Infrastructure** | Squelette du paquet, config, base (DDL + migrations), connecteur PMU (programme, partants, résultats, rapports) avec tests de contrat, couche raw/staging, contrôles qualité, app Streamlit qui démarre en mode dégradé, CI | Collecte d'une journée réelle reproductible ; backfill 3 mois lancé ; CI verte |
| **2. Premier moteur** | Feature store as-of, B0/B1/B2, LightGBM + softmax par course, calibration, places Plackett-Luce/Henery, pages Courses + Analyse | Tests anti-fuite verts ; rapport walk-forward vs marché publié tel quel (même si négatif) |
| **3. Intelligence avancée** | Empirical-Bayes hiérarchique, incertitude/OOD, Isolation Forest, clustering, ruptures (forme), découverte de sous-groupes avec FDR, SHAP, Laboratoire IA, recherche documentaire sourcée (si clé LLM) | Chaque pattern affiché avec n, effet, IC, q-value, statut OOS |
| **4. Deep learning** | GRU/Transformer sur séquences, GNN, autoencodeur — entraînement hors ligne, export ONNX | Conservé seulement si gain significatif dans l'ensemble |
| **5. Validation** | Backtesting complet, simulateur de paris (mise fixe, Kelly fractionné théorique), audit de fuite, tests de robustesse | Rapport sur période test gelée |
| **6. Déploiement** | Optimisation cache, sécurité, monitoring dérive, documentation install/API/déploiement | App en production sur Streamlit Cloud, test fonctionnel validé |

Chaque phase se termine par un commit/PR distinct et un rapport de résultats
honnête dans `artifacts/reports/`.

---

## 10. Décisions attendues avant de commencer

- **D1 — Accès réseau** : l'environnement de développement bloque
  `online.turfinfo.api.pmu.fr`, `open-pmu-api.vercel.app`, `www.turf.bzh`,
  `www.pmu.fr` (et `api.open-meteo.com`, `archive-api.open-meteo.com` seront
  nécessaires). Les ajouter aux domaines autorisés de l'environnement permet de
  valider les formats réels. Sans cela, la phase 1 peut être codée mais pas
  vérifiée contre les vraies réponses.
- **D2 — Cadre d'usage** : usage **personnel/privé** de l'outil (recommandé
  tant que les droits de réutilisation des données PMU ne sont pas clarifiés)
  ou application publique ?
- **D3 — Stockage persistant** : PostgreSQL managé (Supabase ou Neon, offre
  gratuite) + stockage objet (ex. Cloudflare R2) — ou autre préférence ?
- **D4 — Périmètre initial** : commencer par **une discipline** (proposition :
  trot attelé, la plus volumineuse en France, puis plat) plutôt que toutes.
- **D5 — Profondeur du backfill** initial (proposition : 12 mois, à faible cadence).
- **D6 — Services payants** : accepter une clé API LLM (recherche documentaire,
  synthèses) et éventuellement Turf.bzh si l'audit le justifie ? Sinon ces
  modules restent désactivés.

---

## 11. Décisions validées (2026-10-30)

| Décision | Choix |
|---|---|
| D2 — Cadre d'usage | **Usage personnel** |
| D6 — Services payants | **API gratuites uniquement** : Turf.bzh exclu, pas de clé LLM payante (le module de recherche documentaire devra reposer sur des sources gratuites ou rester désactivé) |
| D3 — Stockage | Par défaut : SQLite + Parquet en local, PostgreSQL gratuit (Supabase/Neon) via `DATABASE_URL` pour la persistance |
| D4 — Périmètre | Schéma multi-discipline ; trot attelé prioritaire pour la modélisation |
| D5 — Backfill | 12 mois, à faible cadence |
| D1 — Accès réseau | **Ouvert** le 2026-10-30 pour le PMU et Open-Meteo ; Open PMU API refusée par son hébergeur (403) |

D3 à D5 sont les valeurs proposées, appliquées faute d'indication contraire.
