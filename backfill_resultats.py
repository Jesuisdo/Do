"""
backfill_resultats.py — Récupération rétroactive des RÉSULTATS de courses
(jamais des cotes, qui ne sont pas récupérables a posteriori — voir
collect_live_odds_render.py pour ce constat).

Utilise la même source que collect_live_odds_render.py
(offline.turfinfo.api.pmu.fr, non officielle mais documentée par la
communauté depuis 2019/2020). Écrit dans des tables SÉPARÉES
(resultats_courses / resultats_partants) de celles utilisées par le
collecteur de cotes en direct, pour ne jamais les mélanger.

Se déclenche manuellement (workflow_dispatch), jamais automatiquement :
un backfill de plusieurs années représente des milliers d'appels et doit
rester une décision consciente, pas une tâche qui tourne toute seule sans
supervision. Voir .github/workflows/backfill-resultats.yml.

Principe de prudence : avance jour par jour, journalise chaque jour
(succès ou échec) dans resultats_log, et peut être relancé sans risque de
doublons (ON CONFLICT DO NOTHING) si interrompu en cours de route.
"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
PROGRAMME_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
SLEEP_BETWEEN_CALLS_SEC = 1.0
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema_resultats_postgres.sql")

STATUTS_A_IGNORER = {"COURSE_ANNULEE"}


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL absente de l'environnement.")
    return psycopg2.connect(DATABASE_URL)


def init_schema(conn):
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        with conn.cursor() as cur:
            cur.execute(f.read())
    conn.commit()


def _http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "recherche-hippique-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_valeur_penetrometre(v):
    """PMU renvoie la valeur du pénétromètre (mesure officielle de l'état du
    terrain) en notation française avec virgule (ex: "3,1")."""
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


def backfill_jour(conn, date_str_ddmmyyyy):
    """Récupère toutes les courses d'un jour donné. Retourne le nombre de
    courses effectivement insérées (avec au moins un partant classé)."""
    try:
        programme = _http_get_json(f"{PROGRAMME_BASE}/{date_str_ddmmyyyy}")
    except Exception as e:
        _log(conn, date_str_ddmmyyyy, 0, "ECHEC", f"programme: {type(e).__name__}: {e}")
        return 0

    n_courses = 0
    with conn.cursor() as cur:
        for reunion in programme.get("programme", {}).get("reunions", []):
            r_num = reunion.get("numOfficiel")
            hippodrome = (reunion.get("hippodrome") or {}).get("libelleCourt", "INCONNU")
            meteo = reunion.get("meteo") or {}
            for course in reunion.get("courses", []):
                c_num = course.get("numExterne") or course.get("numOrdre")
                statut = course.get("statut", "")
                if r_num is None or c_num is None or statut in STATUTS_A_IGNORER:
                    continue

                date_iso = datetime.strptime(date_str_ddmmyyyy, "%d%m%Y").date().isoformat()
                course_id = f"{date_iso}_{hippodrome}_R{r_num}C{c_num}".replace(" ", "-")

                try:
                    detail = _http_get_json(f"{PROGRAMME_BASE}/{date_str_ddmmyyyy}/R{r_num}/C{c_num}/participants")
                except Exception:
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                    continue

                participants = detail.get("participants", [])
                if not participants:
                    time.sleep(SLEEP_BETWEEN_CALLS_SEC)
                    continue

                penetrometre = course.get("penetrometre") or {}
                cur.execute(
                    """INSERT INTO resultats_courses
                       (course_id, date_course, hippodrome, r_c, discipline, distance_m,
                        prix_nom, montant_allocation, partants_declares, heure_depart,
                        date_collecte, raw_json,
                        meteo_temperature, meteo_force_vent, meteo_direction_vent, meteo_nebulosite,
                        terrain_intitule, terrain_valeur_penetrometre,
                        corde, type_piste, categorie_particularite, condition_age, condition_sexe, specialite)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (course_id) DO NOTHING""",
                    (
                        course_id, date_iso, hippodrome, f"R{r_num}/C{c_num}",
                        course.get("discipline"), course.get("distance"), course.get("libelle"),
                        course.get("montantTotalOffert") or course.get("montantPrix"),
                        len(participants), course.get("heureDepart"),
                        datetime.utcnow().isoformat(), json.dumps(course, ensure_ascii=False),
                        meteo.get("temperature"), meteo.get("forceVent"),
                        meteo.get("directionVent"), meteo.get("nebulositeLibelleCourt"),
                        penetrometre.get("intitule"), _parse_valeur_penetrometre(penetrometre.get("valeurMesure")),
                        course.get("corde"), course.get("typePiste"), course.get("categorieParticularite"),
                        course.get("conditionAge"), course.get("conditionSexe"), course.get("specialite"),
                    ),
                )

                for p in participants:
                    if p.get("numPmu") is None:
                        continue
                    gp = p.get("gainsParticipant") or {}
                    commentaire = p.get("commentaireApresCourse") or {}
                    distance_prec = p.get("distanceChevalPrecedent") or {}
                    cur.execute(
                        """INSERT INTO resultats_partants
                           (course_id, numero, nom_cheval, sexe, age, nom_jockey,
                            nom_entraineur, musique, gains, position_arrivee,
                            nom_pere, nom_mere, oeilleres, deferre, driver_change, avis_entraineur,
                            nombre_courses, nombre_victoires, nombre_places, nombre_places_second,
                            nombre_places_troisieme, gains_victoires, gains_place,
                            gains_annee_encours, gains_annee_precedente, handicap_distance,
                            temps_obtenu, reduction_kilometrique, incident, commentaire_apres_course,
                            id_cheval, nom_pere_mere, handicap_valeur, handicap_poids,
                            poids_condition_monte, poids_condition_monte_change,
                            distance_cheval_precedent_libelle, distance_cheval_precedent_code,
                            place_corde, indicateur_inedit, jument_pleine, race, pays,
                            pays_entrainement, proprietaire, eleveur, robe)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                   %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (course_id, numero) DO NOTHING""",
                        (
                            course_id, p["numPmu"], p.get("nom"), p.get("sexe"), p.get("age"),
                            p.get("driver"), p.get("entraineur"), p.get("musique"),
                            gp.get("gainsCarriere"),
                            p.get("ordreArrivee"),
                            p.get("nomPere"), p.get("nomMere"), p.get("oeilleres"), p.get("deferre"),
                            p.get("driverChange"), p.get("avisEntraineur"),
                            p.get("nombreCourses"), p.get("nombreVictoires"), p.get("nombrePlaces"),
                            p.get("nombrePlacesSecond"), p.get("nombrePlacesTroisieme"),
                            gp.get("gainsVictoires"), gp.get("gainsPlace"),
                            gp.get("gainsAnneeEnCours"), gp.get("gainsAnneePrecedente"),
                            p.get("handicapDistance"), p.get("tempsObtenu"), p.get("reductionKilometrique"),
                            p.get("incident"), commentaire.get("texte"),
                            p.get("idCheval"), p.get("nomPereMere"), p.get("handicapValeur"), p.get("handicapPoids"),
                            p.get("poidsConditionMonte"), p.get("poidsConditionMonteChange"),
                            distance_prec.get("libelleLong"), distance_prec.get("code"),
                            p.get("placeCorde"), p.get("indicateurInedit"), p.get("jumentPleine"),
                            p.get("race"), p.get("pays"), p.get("paysEntrainement"),
                            p.get("proprietaire"), p.get("eleveur"), (p.get("robe") or {}).get("libelleLong"),
                        ),
                    )
                n_courses += 1
                time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    conn.commit()
    _log(conn, date_str_ddmmyyyy, n_courses, "OK", f"{n_courses} course(s) avec résultats")
    conn.commit()
    return n_courses


def _log(conn, date_str, n, statut, message):
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO resultats_log (date_execution, date_cible, nb_courses_recuperees, statut, message)
               VALUES (%s,%s,%s,%s,%s)""",
            (datetime.utcnow().isoformat(), date_str, n, statut, message),
        )


def main():
    date_debut_str = os.environ.get("DATE_DEBUT")  # format attendu: YYYY-MM-DD
    date_fin_str = os.environ.get("DATE_FIN")      # format attendu: YYYY-MM-DD

    if not date_debut_str or not date_fin_str:
        # Pas de période fournie = usage "collecte quotidienne" : on récupère
        # uniquement la journée d'hier. C'est ce mode qu'utilise le workflow
        # planifié collecte-resultats-quotidien.yml (voir ce fichier).
        hier = (datetime.utcnow() - timedelta(days=1)).date()
        date_debut_str = date_fin_str = hier.isoformat()
        print(f"Aucune période fournie — mode quotidien, récupération de : {hier.isoformat()}")

    date_debut = datetime.strptime(date_debut_str, "%Y-%m-%d").date()
    date_fin = datetime.strptime(date_fin_str, "%Y-%m-%d").date()
    if date_debut > date_fin:
        print("DATE_DEBUT doit être avant DATE_FIN.")
        sys.exit(1)

    conn = get_connection()
    init_schema(conn)

    total = 0
    d = date_debut
    while d <= date_fin:
        date_str_ddmmyyyy = d.strftime("%d%m%Y")
        n = backfill_jour(conn, date_str_ddmmyyyy)
        total += n
        print(f"{d.isoformat()} : {n} course(s)")
        d += timedelta(days=1)

    print(f"\nTerminé : {total} courses au total sur la période {date_debut} → {date_fin}")
    conn.close()


if __name__ == "__main__":
    main()
