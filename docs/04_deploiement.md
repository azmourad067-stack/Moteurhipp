# Déploiement (phase 6)

Architecture cible, entièrement gratuite pour un usage personnel :

```
GitHub Actions (tâches planifiées)          Streamlit Community Cloud
  collect  : toutes les 15 min, 10 h-22 h      app.py, lecture seule
  nightly  : journée complète + dimanche :      │
             réentraînement, labo, validation   │
  backfill : historique, à la demande           │
        │ écriture                              │ lecture
        └──────────► PostgreSQL gratuit ◄───────┘
                     (Supabase ou Neon)
                     données, modèles, rapports
```

La base est la **source unique** : les modèles (JSON + texte LightGBM, empreinte SHA-256)
et les rapports y sont stockés. Le disque de Streamlit Cloud et celui des tâches GitHub
sont éphémères.

## 0. État actuel (30/10/2026)

Le projet Supabase **`horse-intelligence`** (région Paris `eu-west-3`, offre gratuite,
identifiant `yblpqtcahraievhblsxw`) est créé. Le schéma et la sécurité y sont appliqués
(`supabase/migrations/`) :
- 25 tables, toutes protégées par RLS ;
- aucun droit pour les rôles publics `anon` et `authenticated` de l'API REST Supabase ;
- un rôle `hi_app` en lecture seule, dont la connexion n'est pas encore activée ;
- aucun avis de sécurité Supabase.

Deux identifiants dédiés existent, et aucun n'est le compte administrateur `postgres` :

| Rôle | Usage | Droits |
|---|---|---|
| `hi_writer` | Secret GitHub `DATABASE_URL` (tâches planifiées) | lecture et écriture sur les tables du projet |
| `hi_app` | Secret Streamlit `DATABASE_URL` (application) | lecture seule |

Leurs mots de passe ont été générés aléatoirement et remis au propriétaire du projet
hors du dépôt. Pour les changer : *SQL Editor* →
`alter role hi_writer with password '…';` (idem pour `hi_app`), puis mettre à jour les
secrets. Chaînes de connexion (hôte vérifié le 10/11/2026 : **aws-1**, pas aws-0 ; en cas de doute, *Connect → Session pooler*) :

```
GitHub    : postgresql+psycopg://hi_writer.yblpqtcahraievhblsxw:<mdp>@aws-1-eu-west-3.pooler.supabase.com:5432/postgres?sslmode=require
Streamlit : postgresql+psycopg://hi_app.yblpqtcahraievhblsxw:<mdp>@aws-1-eu-west-3.pooler.supabase.com:5432/postgres?sslmode=require
```

La branche de travail est fusionnée dans `main`. Passez directement à l'étape 2 (GitHub).

## 1. Créer la base PostgreSQL (si vous repartez de zéro)

**Supabase** (offre gratuite, 500 Mo) : créer un projet, puis *Connect* → copier la chaîne
**Session pooler** (port 5432). La connexion directe n'est accessible qu'en IPv6, que
GitHub Actions ne gère pas. Adapter le préfixe pour SQLAlchemy :

```
postgresql+psycopg://postgres.<ref>:<mot_de_passe>@aws-N-<région> (N = 0 ou 1 selon le projet : voir *Connect → Session pooler*).pooler.supabase.com:5432/postgres?sslmode=require
```

**Neon** (offre gratuite, 0,5 Go) : alternative compatible IPv4, même format
`postgresql+psycopg://…?sslmode=require`.

Le schéma est créé automatiquement à la première exécution d'une tâche (ou localement :
`DATABASE_URL=… python scripts/initialize_database.py`).

### Rôle en lecture seule pour l'application (recommandé)

Appliquez les deux fichiers de `supabase/migrations/` (schéma, puis sécurité RLS), puis activez la connexion du rôle. Sur une autre base PostgreSQL, l'équivalent minimal est :

```sql
create role hi_app login password 'choisir-un-mot-de-passe-long';
grant connect on database postgres to hi_app;
grant usage on schema public to hi_app;
grant select on all tables in schema public to hi_app;
alter default privileges in schema public grant select on tables to hi_app;
```

Sur Supabase via le pooler, l'utilisateur s'écrit `hi_app.<ref>`. Ce mode a été vérifié
sur PostgreSQL 16 : toutes les pages s'affichent, et toute écriture est refusée.

## 2. Configurer GitHub

1. *Settings → Secrets and variables → Actions → New repository secret* :
   `DATABASE_URL` = chaîne de connexion **avec droits d'écriture** (utilisateur `postgres`).
2. Coût en minutes : dépôt **public** = gratuit et illimité ; dépôt **privé** = 2 000 min/mois.
   La collecte est une chaîne permanente (exécutions de ~5 h 40 qui collectent toutes les 15 min
   en journée et se relancent elles-mêmes, car GitHub déclenche mal les planifications) :
   réservé à un dépôt public. La tâche nocturne consomme environ 150 min/mois.
3. Fusionner la branche de travail dans `main` : les tâches planifiées ne s'exécutent que
   depuis la branche par défaut.

## 3. Constituer l'historique

*Actions → backfill → Run workflow*, par tranches de 2 à 3 mois (limite de 6 h par tâche) :

| Tranche | start | end |
|---|---|---|
| 1 | 2025-10-30 | 2026-01-15 |
| 2 | 2026-01-16 | 2026-03-31 |
| 3 | 2026-04-01 | 2026-06-15 |
| 4 | 2026-06-16 | 2026-09-19 |

La reprise est automatique : une journée déjà collectée est ignorée. Ensuite :
*Actions → nightly → Run workflow* avec `retrain = true` pour le premier entraînement.

Volume estimé pour 12 mois : environ 25 000 courses et 300 000 partants, soit moins de
200 Mo. Seules les 3 dernières versions de chaque modèle sont conservées (plus le modèle
actif), avec 5 rapports par type.

## 4. Déployer l'application

1. share.streamlit.io → *Create app* → ce dépôt, branche `main`, fichier `app.py`.
2. *Advanced settings* : Python 3.12 ; *Secrets* :
   ```toml
   DATABASE_URL = "postgresql+psycopg://hi_app.<ref>:<mot_de_passe>@aws-1-eu-west-3.pooler.supabase.com:5432/postgres?sslmode=require"
   HI_READ_ONLY = "1"
   ```
3. Déployer. `packages.txt` installe `libgomp1` (requis par LightGBM).
4. Vérifier les journaux (*Manage app → Logs*), puis un test fonctionnel :
   - **Tableau de bord** : nombre de courses non nul, carte « Collecte » à jour ;
   - **Courses du jour** : partants et cotes de la journée ;
   - **Analyse approfondie** : probabilités avec statut d'incertitude ;
   - **Backtesting** et **Laboratoire IA** : rapports du dernier dimanche.

## 5. Exploitation

| Besoin | Action |
|---|---|
| Collecte en échec | GitHub envoie un courriel ; la carte « Collecte » passe à « en retard » après 26 h |
| Suivi en conditions réelles | Carte « Conditions réelles » : prédictions figées avant départ, comparées au marché une fois réglées |
| Retour à un modèle précédent | `python scripts/set_active_model.py --family lgbm_market` (liste), puis `--version … --yes` |
| Réentraînement manuel | *Actions → nightly → Run workflow*, `retrain = true` |
| Promotion d'un modèle | Automatique uniquement si son gain de log-loss sur le marché calibré est significatif (IC 95 % < 0, au moins 300 courses) |

## Sécurité

- Aucun secret dans le code : `DATABASE_URL` vient des secrets GitHub ou Streamlit.
  `.streamlit/secrets.toml` est ignoré par Git (modèle : `.streamlit/secrets.toml.example`).
- Les identifiants sont masqués dans les journaux.
- L'application ne peut ni écrire en base (rôle en lecture seule) ni lancer de script.
- Les artefacts de modèles sont du JSON (jamais de pickle) dont l'empreinte est vérifiée
  au chargement.
- Les dates saisies dans le workflow `backfill` sont validées avant usage (pas
  d'injection de commande).
