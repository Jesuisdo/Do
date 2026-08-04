-- ============================================================================
-- Schéma PostgreSQL — collecteur de cotes en direct (déploiement Render)
-- Version adaptée de pipeline/schema.sql (SQLite) pour un stockage durable
-- côté service hébergé, où le disque local du cron job n'est PAS persistant
-- entre deux exécutions (contrainte Render : les Cron Jobs redémarrent avec
-- un système de fichiers vierge à chaque run — seule une base externe comme
-- Postgres survit d'une exécution à l'autre).
--
-- Ne contient QUE ce dont le collecteur de cotes a besoin pour fonctionner
-- de façon autonome, sans dépendre de la base SQLite locale (celle-ci reste
-- gérée séparément par la tâche planifiée Cowork "collecte-quotidienne-
-- hippique" et vit dans Documents/recherche-hippique/pipeline/). Les deux
-- bases se réconcilient a posteriori via course_id (même convention de
-- nommage : date_hippodrome_r/c) — voir README de ce dossier.
-- ============================================================================

CREATE TABLE IF NOT EXISTS courses (
    course_id       TEXT PRIMARY KEY,
    date_course     TEXT NOT NULL,
    hippodrome      TEXT,
    r_c             TEXT,
    heure_depart    TEXT,
    date_decouverte TEXT NOT NULL   -- horodatage de la première fois où cette course a été vue dans le flux de cotes en direct
);

CREATE TABLE IF NOT EXISTS partants (
    id          SERIAL PRIMARY KEY,
    course_id   TEXT NOT NULL REFERENCES courses(course_id),
    numero      INTEGER NOT NULL,
    nom_cheval  TEXT,
    UNIQUE(course_id, numero)
);

CREATE TABLE IF NOT EXISTS cotes_historique (
    id                  SERIAL PRIMARY KEY,
    course_id           TEXT NOT NULL REFERENCES courses(course_id),
    numero              INTEGER NOT NULL,
    horodatage          TEXT NOT NULL,
    cote                REAL NOT NULL,
    minutes_avant_depart INTEGER
);

CREATE TABLE IF NOT EXISTS sources_log (
    id                      SERIAL PRIMARY KEY,
    source                  TEXT NOT NULL,
    date_execution          TEXT NOT NULL,
    parametres              TEXT,
    nb_courses_recuperees   INTEGER,
    statut                  TEXT,
    message                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_partants_course ON partants(course_id);
CREATE INDEX IF NOT EXISTS idx_cotes_course ON cotes_historique(course_id, numero);
