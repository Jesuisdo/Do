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
    raw_json             TEXT,
    meteo_temperature          INTEGER,
    meteo_force_vent           INTEGER,
    meteo_direction_vent       TEXT,
    meteo_nebulosite           TEXT,
    terrain_intitule           TEXT,
    terrain_valeur_penetrometre REAL,
    corde                       TEXT,
    type_piste                  TEXT,
    categorie_particularite     TEXT,
    condition_age               TEXT,
    condition_sexe              TEXT,
    specialite                  TEXT
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
    -- Champs ajoutés : déjà présents dans le JSON PMU, pas extraits jusqu'ici.
    nom_pere              TEXT,
    nom_mere              TEXT,
    oeilleres             TEXT,
    deferre               TEXT,
    driver_change         BOOLEAN,
    avis_entraineur       TEXT,
    nombre_courses        INTEGER,
    nombre_victoires      INTEGER,
    nombre_places         INTEGER,
    nombre_places_second     INTEGER,
    nombre_places_troisieme  INTEGER,
    gains_victoires          INTEGER,
    gains_place              INTEGER,
    gains_annee_encours      INTEGER,
    gains_annee_precedente   INTEGER,
    handicap_distance        INTEGER,
    temps_obtenu             INTEGER,  -- en centièmes de seconde (format PMU), NULL si non couru/non classé
    reduction_kilometrique   INTEGER,  -- temps au kilomètre, centièmes de seconde
    incident                 TEXT,     -- ex: disqualification, allure irrégulière
    commentaire_apres_course TEXT,     -- commentaire d'analyste post-course
    -- Couverture maximale (même liste que partants, cf schema_postgres.sql) :
    id_cheval                    TEXT,
    nom_pere_mere                 TEXT,
    handicap_valeur                REAL,
    handicap_poids                   INTEGER,
    poids_condition_monte              INTEGER,
    poids_condition_monte_change        BOOLEAN,
    distance_cheval_precedent_libelle    TEXT,
    distance_cheval_precedent_code        INTEGER,
    place_corde                            INTEGER,
    indicateur_inedit                       BOOLEAN,
    jument_pleine                            BOOLEAN,
    race                                      TEXT,
    pays                                      TEXT,
    pays_entrainement                        TEXT,
    proprietaire                             TEXT,
    eleveur                                  TEXT,
    robe                                     TEXT,
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
CREATE INDEX IF NOT EXISTS idx_resultats_partants_id_cheval ON resultats_partants(id_cheval);
CREATE INDEX IF NOT EXISTS idx_resultats_courses_date ON resultats_courses(date_course);
