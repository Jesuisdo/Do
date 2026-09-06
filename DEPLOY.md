# Déploiement du collecteur de cotes en direct sur Render

Ce dossier est un déploiement Render **autonome et distinct** du pipeline
local (`../pipeline/`). Il tourne 24h/24 sur les serveurs de Render, pas sur
la machine de Dorian ni dans une session Cowork — c'est ce qui garantit que
la collecte des mouvements de cote ne s'arrête jamais, contrairement à une
tâche planifiée Cowork qui ne s'exécute que si l'application est ouverte.

**Coût** : le plan Postgres "free" et le Cron Job "starter" sont ceux
disponibles au moment de la réda

## Cron Zeturf (piste source tierce) -- ajoute le 06/09/2026

Contexte : l'audit du 05/09/2026 a montre que le goulot d'etranglement
VERT x H-15 vient de courses PMU-ignorees a 0% de couverture de cotes
(Allemagne, Chili, Argentine, Uruguay -- PMU liste ces reunions mais ne
publie jamais de cote dessus). Le POC du 06/09/2026 a confirme que
Zeturf.fr publie de vraies cotes variees pour ces memes courses
(Baden-Baden R4C1, San Isidro R8C11), contrairement a Geny.com (cotes
plates non exploitables hors PMU). Nouveau fichier : collect_odds_zeturf_render.py
(voir son docstring pour la methodologie complete point-in-time).

### Different du collecteur PMU sur un point important

Les cellules de cote sur Zeturf sont vides dans le HTML brut et remplies
par un appel AJAX cote client (potentiellement protege par un jeton
anti-CSRF) -- ce script utilise donc Playwright (deja present dans
requirements.txt) pour charger la page avec un vrai navigateur headless
plutot que d'interroger un point d'acces interne non documente. Consequence :
il faut AJOUTER l'installation du navigateur Chromium au build, ce que
ne fait pas le buildCommand actuel (pip install -r requirements.txt seul
ne suffit pas).

### Bloc render.yaml a ajouter (nouveau Cron Job, meme base que l'existant)

```yaml
  - type: cron
    name: collecteur-cotes-zeturf
    runtime: python
    plan: starter
    schedule: "*/5 * * * *"
    buildCommand: "pip install -r requirements.txt && playwright install --with-deps chromium"
    startCommand: "python3 collect_odds_zeturf_render.py"
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: hippique-cotes-db
          property: connectionString
```

### A verifier toi-meme avant de deployer (je ne peux pas le confirmer depuis cet environnement)

1. render.yaml declare la base `hippique-cotes-db` (plan Render interne),
   mais toutes nos requetes SQL de cette session (courses/partants/cotes_historique,
   137000+ lignes) pointent vers le projet Supabase `jvfrvttfedmabnoldbyp`. Soit
   tu as reconfigure manuellement DATABASE_URL sur le Cron Job existant pour
   pointer vers Supabase (le plus probable), soit ce fichier est desynchronise
   de la config reelle -- dans les deux cas, verifie quelle base est reellement
   active et attache ce nouveau Cron Job A LA MEME.
2. Ce DEPLOY.md documente un depot separe `hippique-cotes-render` (etape 1),
   mais collect_live_odds_render.py et render.yaml existent aussi dans CE
   depot (Jesuisdo/Do). Verifie quel depot est reellement connecte au Cron
   Job Render actif avant de pousser collect_odds_zeturf_render.py -- s'il
   s'agit d'un depot separe, il faut y copier ce fichier aussi (ou repointer
   le Blueprint vers Do).
3. Chromium headless consomme sensiblement plus de RAM/CPU qu'un simple
   appel JSON (cas du collecteur PMU) -- si le plan "starter" sature, passer
   a un plan superieur pour ce Cron Job specifiquement.

### Migration DB deja appliquee

Colonne `source` (text, defaut 'pmu') ajoutee a `cotes_historique` le
06/09/2026 -- les lignes existantes restent `source='pmu'` automatiquement,
aucune migration de donnees necessaire. Les nouvelles lignes Zeturf portent
`source='zeturf'`.
ction de ce guide — vérifie les tarifs
actuels sur render.com/pricing avant de déployer, ils peuvent avoir changé.

## Pourquoi le connecteur Render de Cowork n'a pas fonctionné

Cowork a tenté de s'enregistrer automatiquement auprès du serveur OAuth de
Render (et de Railway, même résultat) — ce mécanisme n'est pas supporté par
ces plateformes actuellement. C'est une limitation de la plateforme
d'intégration, pas de ton compte Render. D'où ce déploiement manuel, qui
contourne complètement le problème puisque tu te connectes directement à
Render depuis ton navigateur.

## Étapes (environ 10 minutes, aucun code à écrire)

### 1. Créer un dépôt GitHub

Sur https://github.com/new, crée un dépôt (public ou privé, peu importe —
Render peut se connecter aux deux une fois ton compte GitHub lié).
Nomme-le par exemple `hippique-cotes-render`.

### 2. Pousser les fichiers de ce dossier

Depuis ton ordinateur, dans un terminal, à l'intérieur de ce dossier
(`deploiement-render/`) :

```bash
git init
git add .
git commit -m "Collecteur de cotes en direct - déploiement initial"
git branch -M main
git remote add origin https://github.com/<ton-compte>/hippique-cotes-render.git
git push -u origin main
```

### 3. Déployer via Blueprint sur Render

1. Va sur https://dashboard.render.com
2. Clique "New +" → "Blueprint"
3. Connecte ton compte GitHub si ce n'est pas déjà fait, sélectionne le
   dépôt `hippique-cotes-render`
4. Render détecte automatiquement `render.yaml` et propose de créer :
   - la base Postgres `hippique-cotes-db`
   - le Cron Job `collecteur-cotes-pmu` (toutes les 4 minutes)
5. Clique "Apply" — c'est tout, les deux ressources se créent et se lient
   automatiquement (`DATABASE_URL` est injectée sans configuration de ta
   part).

### 4. Vérifier que ça tourne

Dans le dashboard Render, onglet du Cron Job → "Logs". Tant que
`fetch_live_snapshot()` n'est pas implémenté (voir ci-dessous), chaque
exécution se terminera proprement avec un message "NotImplementedError"
dans les logs et dans la table `sources_log` — c'est attendu, pas une
panne.

## Ce qu'il reste à faire (une seule fonction)

`collect_live_odds_render.py`, fonction `fetch_live_snapshot()` : doit
identifier et interroger le vrai point d'accès aux cotes en direct de
PMU.fr, puis retourner les données au format décrit dans son docstring.
Cette étape nécessite un accès réseau réel pour inspecter le trafic du site
pendant une réunion de courses — l'environnement qui a préparé ce
déploiement ne peut pas le faire lui-même.

Une fois cette fonction écrite, `git push` suffit : Render redéploie
automatiquement à chaque push sur `main`, donc aucune manipulation
supplémentaire dans le dashboard n'est nécessaire pour les mises à jour
futures.

## Réconciliation avec la base locale

Les `course_id` générés ici suivent exactement la même convention que dans
`../pipeline/ingest_open_pmu_api.py` (date + hippodrome + r/c), pour
pouvoir un jour rapprocher la table `cotes_historique` de Postgres avec la
table `courses`/`partants` de la base SQLite locale. Ce rapprochement se
fera manuellement (export Postgres → import SQLite) une fois qu'il y aura
suffisamment de données des deux côtés pour que ce soit utile — pas une
priorité tant que `fetch_live_snapshot()` n'est pas implémenté.
