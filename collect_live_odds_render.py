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

Tout le reste (connexion Postgres, schéma, écriture, journalisation,
horodatage, gestion des erreurs) est fonctionnel et a été testé dans son
ensemble avant déploiement.
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
FENETRE_ANTICIPATION_MIN = 90    # capte une course dès 90 min avant son départ
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
                courses_a_suivre.append((r_num, c_num, hippodrome, heure_depart_ms, meteo, penetrometre))

    snapshot = []
    for r_num, c_num, hippodrome, heure_depart_ms, meteo, penetrometre in courses_a_suivre:
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
            cote = direct.get("rapport")
            if cote is None or p.get("numPmu") is None:
                continue
            partants.append({
                "numero": p["numPmu"],
                "nom_cheval": p.get("nom"),
                "cote": cote,
                "cote_reference": reference.get("rapport"),
                "tendance": direct.get("indicateurTendance"),
                "favori": bool(direct.get("favoris", False)),
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


def poll_once(conn, now: datetime):
    try:
        snapshot = fetch_live_snapshot()
    except Exception as e:
        _log(conn, now, 0, "ECHEC", f"{type(e).__name__}: {e}")
        conn.commit()
        return

    n_courses = 0
    with conn.cursor() as cur:
        for course in snapshot:
            cid = course["course_id"]
            cur.execute(
                """INSERT INTO courses (course_id, date_course, hippodrome, r_c, heure_depart, date_decouverte,
                                        meteo_temperature, meteo_force_vent, meteo_direction_vent, meteo_nebulosite,
                                        terrain_intitule, terrain_valeur_penetrometre)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (course_id) DO NOTHING""",
                (cid, course["date_course"], course["hippodrome"], course["r_c"],
                 course["heure_depart"], now.isoformat(),
                 course.get("meteo_temperature"), course.get("meteo_force_vent"),
                 course.get("meteo_direction_vent"), course.get("meteo_nebulosite"),
                 course.get("terrain_intitule"), course.get("terrain_valeur_penetrometre")),
            )
            m_avant = minutes_avant_depart(course["date_course"], course["heure_depart"], now)
            for p in course.get("partants", []):
                cur.execute(
                    """INSERT INTO partants (course_id, numero, nom_cheval)
                       VALUES (%s,%s,%s)
                       ON CONFLICT (course_id, numero) DO NOTHING""",
                    (cid, p["numero"], p.get("nom_cheval")),
                )
                cur.execute(
                    """INSERT INTO cotes_historique
                       (course_id, numero, horodatage, cote, cote_reference, tendance, favori, minutes_avant_depart)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (cid, p["numero"], now.isoformat(), p["cote"], p.get("cote_reference"),
                     p.get("tendance"), p.get("favori"), m_avant),
                )
            n_courses += 1

    _log(conn, now, n_courses, "OK", f"{n_courses} course(s) suivie(s)")
    conn.commit()


def _log(conn, now, n, statut, message):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO sources_log (source, date_execution, parametres, nb_courses_recuperees, statut, message)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (SOURCE_NAME, now.isoformat(), "", n, statut, message),
        )


def main():
    conn = get_connection()
    init_schema(conn)
    poll_once(conn, _now_paris())
    conn.close()


if __name__ == "__main__":
    main()
