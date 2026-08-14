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
    date_decouverte TEXT NOT NULL,  -- horodatage de la première fois où cette course a été vue dans le flux de cotes en direct
    meteo_temperature          INTEGER,  -- °C, prévision PMU au niveau de la réunion
    meteo_force_vent           INTEGER,
    meteo_direction_vent       TEXT,
    meteo_nebulosite           TEXT,
    terrain_intitule           TEXT,    -- ex: "Bon", "Souple" — mesure officielle PMU (pénétromètre)
    terrain_valeur_penetrometre REAL,
    -- Champs course ajoutés (couverture maximale) : caractéristiques de course
    -- déjà présentes dans le JSON récupéré, simplement pas extraites jusqu'ici.
    corde                       TEXT,   -- ex: "CORDE_DROITE"
    type_piste                  TEXT,   -- ex: "HERBE"
    categorie_particularite     TEXT,   -- ex: "HANDICAP_DIVISE"
    condition_age               TEXT,
    condition_sexe              TEXT,
    specialite                  TEXT
);

CREATE TABLE IF NOT EXISTS partants (
    id          SERIAL PRIMARY KEY,
    course_id   TEXT NOT NULL REFERENCES courses(course_id),
    numero      INTEGER NOT NULL,
    nom_cheval  TEXT,
    -- Champs ajoutés : déjà présents dans le JSON PMU récupéré pour les cotes,
    -- simplement pas extraits jusqu'ici.
    nom_pere              TEXT,
    nom_mere              TEXT,
    oeilleres              TEXT,
    deferre                 TEXT,     -- ferrage (NULL = ferré normalement)
    driver_change            BOOLEAN,
    avis_entraineur           TEXT,
    nombre_courses             INTEGER,
    nombre_victoires            INTEGER,
    nombre_places                 INTEGER,
    nombre_places_second            INTEGER,
    nombre_places_troisieme           INTEGER,
    gains_victoires                     INTEGER,
    gains_place                          INTEGER,
    gains_annee_encours                   INTEGER,
    gains_annee_precedente                 INTEGER,
    handicap_distance                       INTEGER,
    -- Couverture maximale (demande explicite : capturer tout ce qui est
    -- disponible, même si on ne s'en sert pas encore tout de suite) :
    id_cheval                    TEXT,   -- identifiant PMU stable du cheval, clé pour le suivi longitudinal
    nom_pere_mere                 TEXT,  -- père de la mère (BM sire)
    handicap_valeur                REAL, -- note officielle de performance/valeur
    handicap_poids                   INTEGER, -- poids porté (centièmes de kg)
    poids_condition_monte              INTEGER,
    poids_condition_monte_change        BOOLEAN,
    distance_cheval_precedent_libelle    TEXT,   -- marge à l'arrivée précédente, ex: "1/2 longueur"
    distance_cheval_precedent_code        INTEGER,
    place_corde                            INTEGER, -- position de départ / couloir
    indicateur_inedit                       BOOLEAN, -- cheval qui n'a jamais couru
    jument_pleine                            BOOLEAN,
    race                                      TEXT,  -- race/race du cheval, ex: "PUR-SANG"
    pays                                      TEXT,
    pays_entrainement                        TEXT,
    proprietaire                             TEXT,
    eleveur                                  TEXT,
    robe                                     TEXT,
    UNIQUE(course_id, numero)
);

CREATE TABLE IF NOT EXISTS cotes_historique (
    id                  SERIAL PRIMARY KEY,
    course_id           TEXT NOT NULL REFERENCES courses(course_id),
    numero              INTEGER NOT NULL,
    horodatage          TEXT NOT NULL,
    cote                REAL NOT NULL,
    cote_reference       REAL,       -- cote d'ouverture ("dernierRapportReference" côté PMU) — utile pour PRG-2B / H-16 (mouvement de cote)
    tendance             TEXT,       -- "+"/"-" tel que fourni par le PMU (indicateurTendance)
    favori                BOOLEAN,   -- statut favori tel que calculé par le PMU au moment de la lecture
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
