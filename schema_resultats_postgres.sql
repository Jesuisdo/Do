-- ============================================================================
-- Schéma PostgreSQL — résultats historiques (backfill rétroactif)
-- Distinct des tables courses/partants/cotes_historique utilisées par le
-- collecteur de cotes en direct (celles-ci restent focalisées sur le
-- suivi prospectif). Ici : tout ce qui est récupérable rétroactivement
-- (résultats, musique, gains, etc. — jamais les cotes).
-- ============================================================================

CREATE TABLE IF NOT EXISTS resultats_courses (
    course_id           TEXT PRIMARY KEY,
    date_course          TEXT NOT NULL,
    hippodrome           TEXT,
    r_c                  TEXT,
    discipline           TEXT,
    distance_m           INTEGER,
    prix_nom             TEXT,
    montant_allocation   INTEGER,
    partants_declares    INTEGER,
    heure_depart         TEXT,
    date_collecte        TEXT NOT NULL,
    raw_json             TEXT
);

CREATE TABLE IF NOT EXISTS resultats_partants (
    id                    SERIAL PRIMARY KEY,
    course_id             TEXT NOT NULL REFERENCES resultats_courses(course_id),
    numero                INTEGER NOT NULL,
    nom_cheval            TEXT,
    sexe                  TEXT,
    age                   INTEGER,
    nom_jockey            TEXT,
    nom_entraineur        TEXT,
    musique               TEXT,
    gains                 INTEGER,
    position_arrivee      INTEGER,
    UNIQUE(course_id, numero)
);

CREATE TABLE IF NOT EXISTS resultats_log (
    id                     SERIAL PRIMARY KEY,
    date_execution          TEXT NOT NULL,
    date_cible              TEXT NOT NULL,
    nb_courses_recuperees   INTEGER,
    statut                  TEXT,
    message                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_resultats_partants_course ON resultats_partants(course_id);
CREATE INDEX IF NOT EXISTS idx_resultats_courses_date ON resultats_courses(date_course);
