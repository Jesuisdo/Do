"""
collect_live_odds_render.py — Collecteur de cotes en direct, conçu pour
tourner en Cron Job Render (24h/24, indépendamment de toute session Cowork).

SOURCE (identifiée le 04/08/2026, sans inspection manuelle du navigateur —
uniquement via recherche documentaire de forums communautaires qui
utilisent ce point d'accès depuis 2019/2020, puis vérification directe) :

  https://offline.turfinfo.api.pmu.fr/rest/client/7/programme/DDMMYYYY
      → liste toutes les réunions et courses du jour (numéro de réunion,
        hippodrome, numéro de course, heure de départ, statut).

  https://offline.turfinfo.api.pmu.fr/rest/client/7/programme/DDMMYYYY/R{n}/C{m}/participants
      → détail d'une course : partants, et surtout `dernierRapportDirect.rapport`
        (la cote en direct), `dernierRapportReference.rapport` (cote
        d'ouverture), `indicateurTendance` (calculé par le PMU).

Ce point d'accès n'est PAS documenté officiellement par le PMU — c'est une
convention observée et partagée par la communauté (forums Excel/VBA
notamment) depuis plusieurs années, sans garantie de stabilité. On respecte
un intervalle raisonnable entre appels (voir POLL_INTERVAL_MINUTES dans
render.yaml) et on journalise systématiquement les échecs plutôt que de les
masquer, au cas où le point d'accès change ou devient indisponible.

--------------------------------------------------------------------------
CORRECTIF DU 28/08/2026 (Dorian, piste 4 -- cotes de marché) :

Bug diagnostiqué : `partants` et `cotes_historique` restaient à 0 ligne
alors que `courses` en comptait 51. Cause exacte, confirmée via
`pg_constraint` sur la base Supabase en direct : la table `partants`
déployée ne portait PAS la contrainte UNIQUE(course_id, numero) -- alors
que `schema_postgres.sql` la spécifie déjà. `init_schema()` exécute le
schéma via `CREATE TABLE IF NOT EXISTS`, qui est un no-op sur une table
déjà existante : la correction présente dans le fichier n'avait donc
jamais été appliquée à la base réellement déployée (dérive schéma/code).
Résultat : chaque `INSERT ... ON CONFLICT (course_id, numero)` échouait
systématiquement (Postgres valide la cible du ON CONFLICT contre les
contraintes existantes, indépendamment de la survenue réelle d'un
conflit) -- erreur 42P10, non rattrapée nulle part (l'unique try/except du
fichier ne couvrait que `fetch_live_snapshot()`), ce qui faisait
remonter l'exception hors de `poll_once()`/`main()`, annulait toute la
transaction (y compris la ligne `courses` insérée quelques instants
avant) et empêchait même l'écriture d'un log d'échec dans `sources_log`
(le commit + le log n'intervenaient qu'une seule fois, tout à la fin).
`courses` comptait 51 lignes uniquement parce que certains sondages ne
trouvaient encore aucun partant avec cote en direct (liste vide -> pas
d'INSERT partants -> pas d'erreur -> commit réussi).

Corrections apportées ici :
  1. Contrainte UNIQUE(course_id, numero) réappliquée directement sur la
     base Supabase (migration manuelle, hors de ce script, la table étant
     vide) -- ET filet de sécurité idempotent ajouté dans `init_schema()`
     (`_reparer_derive_schema`) pour que ce type de dérive schéma/code ne
     puisse plus jamais bloquer silencieusement la collecte à l'avenir,
     même si la table venait à être recréée sans cette contrainte.
  2. Le sondage n'est plus une transaction unique "tout ou rien" : chaque
     course (course + ses partants + ses cotes) est maintenant insérée et
     validée (commit) indépendamment. Si une course échoue (donnée
     malformée, contrainte violée, etc.), elle est annulée seule
     (rollback ciblé), journalisée avec le détail de l'erreur, et les
     autres courses du même sondage continuent d'être traitées.
  3. `_log(...)` ne peut plus jamais faire échouer silencieusement tout le
     sondage : il utilise sa propre transaction, protégée par son propre
     try/except.
--------------------------------------------------------------------------
"""
import json
import os
import urllib.request
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
except Exception:  # pragma: no cover - filet de sécurité si tzdata absent
    PARIS_TZ = None

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
SOURCE_NAME = "pmu_live_odds_render"
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema_postgres.sql")

PROGRAMME_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
# Élargi à 720 min (12h, ~ toute une journée de courses PMU) plutôt que 90 min.
# Raison : GitHub Actions gratuit n'honore pas fidèlement le "toutes les 5 min"
# demandé (constaté : intervalles réels de plusieurs heures). Avec une fenêtre
# étroite, une course pouvait démarrer ET se terminer entièrement entre deux
# exécutions du programme, sans jamais être captée. Avec une fenêtre large, on
# capte toute course du jour pas encore terminée (voir STATUTS_TERMINES),
# quel que soit le moment où le programme parvient à s'exécuter.
FENETRE_ANTICIPATION_MIN = 720
FENETRE_RETARD_MIN = 5           # continue de suivre jusqu'à 5 min après l'heure officielle (départs parfois retardés)

STATUTS_TERMINES = {
    "FIN_COURSE", "TERMINEE", "ARRIVEE_DEFINITIVE_COMPLETE",
    "ARRIVEE_DEFINITIVE", "COURSE_TERMINEE", "COURSE_ANNULEE",
}


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL absente de l'environnement. Sur Render, cette variable "
            "est injectée automatiquement si le service est bien lié à la base "
            "'hippique-cotes-db' définie dans render.yaml."
        )
    return psycopg2.connect(DATABASE_URL)


def init_schema(conn):
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        with conn.cursor() as cur:
            cur.execute(f.read())
    conn.commit()
    _reparer_derive_schema(conn)


def _reparer_derive_schema(conn):
    """Filet de sécurité contre la dérive schéma/code : `CREATE TABLE IF NOT
    EXISTS` (ci-dessus) ne modifie JAMAIS une table déjà existante. Si
    `partants` a été créée avant que UNIQUE(course_id, numero) soit ajoutée
    à schema_postgres.sql (c'est exactement ce qui s'est produit et qui a
    fait planter poll_once en silence, cf. correctif du 28/08/2026
    documenté en tête de ce fichier), la contrainte peut manquer sur la
    base déployée malgré un schéma à jour dans le dépôt. On vérifie via
    pg_constraint et on l'ajoute nous-mêmes si nécessaire -- idempotent,
    ne fait rien si la contrainte est déjà présente, ne lève jamais."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT 1 FROM pg_constraint
               WHERE conrelid = 'partants'::regclass
                 AND contype = 'u'
                 AND pg_get_constraintdef(oid) = 'UNIQUE (course_id, numero)'"""
        )
        if cur.fetchone() is None:
            cur.execute(
                "ALTER TABLE partants ADD CONSTRAINT partants_course_id_numero_key "
                "UNIQUE (course_id, numero)"
            )
    conn.commit()


def _http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "recherche-hippique-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _now_paris():
    return datetime.now(PARIS_TZ) if PARIS_TZ else datetime.utcnow()


def _parse_valeur_penetrometre(v):
    """PMU renvoie la valeur du pénétromètre (mesure officielle de l'état du
    terrain) en notation française avec virgule (ex: "3,1"). On la convertit
    en nombre pour pouvoir l'exploiter statistiquement."""
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


def fetch_live_snapshot():
    """Retourne la liste des courses actuellement suivies (départ dans la
    fenêtre d'anticipation), avec leurs partants et cotes en direct.
    Voir le docstring du module pour le format exact et la source."""
    now = _now_paris()
    date_str = now.strftime("%d%m%Y")
    programme = _http_get_json(f"{PROGRAMME_BASE}/{date_str}")

    now_ms = int(now.timestamp() * 1000)
    fenetre_ms = FENETRE_ANTICIPATION_MIN * 60 * 1000
    retard_ms = FENETRE_RETARD_MIN * 60 * 1000

    courses_a_suivre = []
    for reunion in programme.get("programme", {}).get("reunions", []):
        r_num = reunion.get("numOfficiel")
        hippodrome = (reunion.get("hippodrome") or {}).get("libelleCourt", "INCONNU")
        meteo = reunion.get("meteo") or {}
        for course in reunion.get("courses", []):
            c_num = course.get("numExterne") or course.get("numOrdre")
            heure_depart_ms = course.get("heureDepart")
            statut = course.get("statut", "")
            if heure_depart_ms is None or r_num is None or c_num is None:
                continue
            if statut in STATUTS_TERMINES:
                continue
            delta_ms = heure_depart_ms - now_ms
            if -retard_ms <= delta_ms <= fenetre_ms:
                penetrometre = course.get("penetrometre") or {}
                courses_a_suivre.append((r_num, c_num, hippodrome, heure_depart_ms, meteo, penetrometre, course))

    snapshot = []
    for r_num, c_num, hippodrome, heure_depart_ms, meteo, penetrometre, course_info in courses_a_suivre:
        try:
            detail = _http_get_json(f"{PROGRAMME_BASE}/{date_str}/R{r_num}/C{c_num}/participants")
        except Exception:
            continue  # course individuelle indisponible — on continue avec les autres, l'échec sera visible dans les logs globaux

        depart_dt = datetime.fromtimestamp(heure_depart_ms / 1000, tz=PARIS_TZ) if PARIS_TZ else datetime.utcfromtimestamp(heure_depart_ms / 1000)
        course_id = f"{depart_dt.date().isoformat()}_{hippodrome}_R{r_num}C{c_num}".replace(" ", "-")

        partants = []
        for p in detail.get("participants", []):
            direct = p.get("dernierRapportDirect") or {}
            reference = p.get("dernierRapportReference") or {}
            gains = p.get("gainsParticipant") or {}
            cote = direct.get("rapport")
            if cote is None or p.get("numPmu") is None:
                continue
            distance_prec = p.get("distanceChevalPrecedent") or {}
            partants.append({
                "numero": p["numPmu"],
                "nom_cheval": p.get("nom"),
                "cote": cote,
                "cote_reference": reference.get("rapport"),
                "tendance": direct.get("indicateurTendance"),
                "favori": bool(direct.get("favoris", False)),
                "nom_pere": p.get("nomPere"),
                "nom_mere": p.get("nomMere"),
                "oeilleres": p.get("oeilleres"),
                "deferre": p.get("deferre"),
                "driver_change": p.get("driverChange"),
                "avis_entraineur": p.get("avisEntraineur"),
                "nombre_courses": p.get("nombreCourses"),
                "nombre_victoires": p.get("nombreVictoires"),
                "nombre_places": p.get("nombrePlaces"),
                "nombre_places_second": p.get("nombrePlacesSecond"),
                "nombre_places_troisieme": p.get("nombrePlacesTroisieme"),
                "gains_victoires": gains.get("gainsVictoires"),
                "gains_place": gains.get("gainsPlace"),
                "gains_annee_encours": gains.get("gainsAnneeEnCours"),
                "gains_annee_precedente": gains.get("gainsAnneePrecedente"),
                "handicap_distance": p.get("handicapDistance"),
                "id_cheval": p.get("idCheval"),
                "nom_pere_mere": p.get("nomPereMere"),
                "handicap_valeur": p.get("handicapValeur"),
                "handicap_poids": p.get("handicapPoids"),
                "poids_condition_monte": p.get("poidsConditionMonte"),
                "poids_condition_monte_change": p.get("poidsConditionMonteChange"),
                "distance_cheval_precedent_libelle": distance_prec.get("libelleLong"),
                "distance_cheval_precedent_code": distance_prec.get("code"),
                "place_corde": p.get("placeCorde"),
                "indicateur_inedit": p.get("indicateurInedit"),
                "jument_pleine": p.get("jumentPleine"),
                "race": p.get("race"),
                "pays": p.get("pays"),
                "pays_entrainement": p.get("paysEntrainement"),
                "proprietaire": p.get("proprietaire"),
                "eleveur": p.get("eleveur"),
                "robe": (p.get("robe") or {}).get("libelleLong"),
            })

        snapshot.append({
            "course_id": course_id,
            "date_course": depart_dt.date().isoformat(),
            "hippodrome": hippodrome,
            "r_c": f"R{r_num}/C{c_num}",
            "heure_depart": depart_dt.strftime("%H:%M:%S"),
            "partants": partants,
            "meteo_temperature": meteo.get("temperature"),
            "meteo_force_vent": meteo.get("forceVent"),
            "meteo_direction_vent": meteo.get("directionVent"),
            "meteo_nebulosite": meteo.get("nebulositeLibelleCourt"),
            "terrain_intitule": penetrometre.get("intitule"),
            "terrain_valeur_penetrometre": _parse_valeur_penetrometre(penetrometre.get("valeurMesure")),
            "corde": course_info.get("corde"),
            "type_piste": course_info.get("typePiste"),
            "categorie_particularite": course_info.get("categorieParticularite"),
            "condition_age": course_info.get("conditionAge"),
            "condition_sexe": course_info.get("conditionSexe"),
            "specialite": course_info.get("specialite"),
        })

    return snapshot


def minutes_avant_depart(date_course: str, heure_depart: str, now: datetime):
    if not heure_depart:
        return None
    try:
        depart_dt = datetime.strptime(f"{date_course} {heure_depart}", "%Y-%m-%d %H:%M:%S")
        if now.tzinfo is not None:
            depart_dt = depart_dt.replace(tzinfo=now.tzinfo)
    except ValueError:
        return None
    return round((depart_dt - now).total_seconds() / 60)


def _inserer_course(conn, course, now):
    """Insère une course + ses partants + leurs cotes du moment, DANS UNE
    SEULE course (au sens hippique du terme). Appelant responsable du
    commit/rollback -- voir poll_once. Isoler chaque course dans sa propre
    transaction évite qu'une ligne malformée dans UNE course n'annule la
    collecte de toutes les autres courses du même sondage (c'est exactement
    ce qui se produisait avant le correctif du 28/08/2026)."""
    cid = course["course_id"]
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO courses (course_id, date_course, hippodrome, r_c, heure_depart, date_decouverte,
                                    meteo_temperature, meteo_force_vent, meteo_direction_vent, meteo_nebulosite,
                                    terrain_intitule, terrain_valeur_penetrometre,
                                    corde, type_piste, categorie_particularite, condition_age, condition_sexe, specialite)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (course_id) DO NOTHING""",
            (cid, course["date_course"], course["hippodrome"], course["r_c"],
             course["heure_depart"], now.isoformat(),
             course.get("meteo_temperature"), course.get("meteo_force_vent"),
             course.get("meteo_direction_vent"), course.get("meteo_nebulosite"),
             course.get("terrain_intitule"), course.get("terrain_valeur_penetrometre"),
             course.get("corde"), course.get("type_piste"), course.get("categorie_particularite"),
             course.get("condition_age"), course.get("condition_sexe"), course.get("specialite")),
        )
        m_avant = minutes_avant_depart(course["date_course"], course["heure_depart"], now)
        for p in course.get("partants", []):
            cur.execute(
                """INSERT INTO partants
                   (course_id, numero, nom_cheval, nom_pere, nom_mere, oeilleres, deferre,
                    driver_change, avis_entraineur, nombre_courses, nombre_victoires,
                    nombre_places, nombre_places_second, nombre_places_troisieme,
                    gains_victoires, gains_place, gains_annee_encours, gains_annee_precedente,
                    handicap_distance,
                    id_cheval, nom_pere_mere, handicap_valeur, handicap_poids,
                    poids_condition_monte, poids_condition_monte_change,
                    distance_cheval_precedent_libelle, distance_cheval_precedent_code,
                    place_corde, indicateur_inedit, jument_pleine, race, pays,
                    pays_entrainement, proprietaire, eleveur, robe)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                           %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (course_id, numero) DO NOTHING""",
                (cid, p["numero"], p.get("nom_cheval"), p.get("nom_pere"), p.get("nom_mere"),
                 p.get("oeilleres"), p.get("deferre"), p.get("driver_change"), p.get("avis_entraineur"),
                 p.get("nombre_courses"), p.get("nombre_victoires"), p.get("nombre_places"),
                 p.get("nombre_places_second"), p.get("nombre_places_troisieme"),
                 p.get("gains_victoires"), p.get("gains_place"), p.get("gains_annee_encours"),
                 p.get("gains_annee_precedente"), p.get("handicap_distance"),
                 p.get("id_cheval"), p.get("nom_pere_mere"), p.get("handicap_valeur"), p.get("handicap_poids"),
                 p.get("poids_condition_monte"), p.get("poids_condition_monte_change"),
                 p.get("distance_cheval_precedent_libelle"), p.get("distance_cheval_precedent_code"),
                 p.get("place_corde"), p.get("indicateur_inedit"), p.get("jument_pleine"),
                 p.get("race"), p.get("pays"), p.get("pays_entrainement"),
                 p.get("proprietaire"), p.get("eleveur"), p.get("robe")),
            )
            cur.execute(
                """INSERT INTO cotes_historique
                   (course_id, numero, horodatage, cote, cote_reference, tendance, favori, minutes_avant_depart)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (cid, p["numero"], now.isoformat(), p["cote"], p.get("cote_reference"),
                 p.get("tendance"), p.get("favori"), m_avant),
            )


def poll_once(conn, now: datetime):
    try:
        snapshot = fetch_live_snapshot()
    except Exception as e:
        _log(conn, now, 0, "ECHEC", f"fetch_live_snapshot: {type(e).__name__}: {e}")
        return

    n_ok = 0
    n_echec = 0
    erreurs = []
    for course in snapshot:
        try:
            _inserer_course(conn, course, now)
            conn.commit()
            n_ok += 1
        except Exception as e:
            conn.rollback()
            n_echec += 1
            erreurs.append(f"{course.get('course_id')}: {type(e).__name__}: {e}")

    if n_echec == 0:
        statut = "OK"
    elif n_ok > 0:
        statut = "PARTIEL"
    else:
        statut = "ECHEC"
    message = f"{n_ok} course(s) suivie(s), {n_echec} echec(s)"
    if erreurs:
        # bornée pour rester lisible dans sources_log.message
        message += " -- " + " | ".join(erreurs[:5])
    _log(conn, now, n_ok, statut, message)


def _log(conn, now, n, statut, message):
    """Journalise dans sources_log. Ne doit JAMAIS faire remonter une
    exception vers l'appelant : utilise sa propre transaction, indépendante
    de l'état (potentiellement déjà en échec) de la transaction principale."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO sources_log (source, date_execution, parametres, nb_courses_recuperees, statut, message)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                (SOURCE_NAME, now.isoformat(), "", n, statut, message),
            )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def main():
    conn = get_connection()
    init_schema(conn)
    poll_once(conn, _now_paris())
    conn.close()


if __name__ == "__main__":
    main()
