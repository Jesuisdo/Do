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

Résilience (ajoutée le 16/08/2026 après un run planté avec
psycopg2.errors.NumericValueOutOfRange sur une valeur non identifiée avec
certitude dans le JSON du 22/07/2026) :
- chaque course est insérée dans sa PROPRE transaction (commit après
  chaque course, pas seulement en fin de journée) — une course
  problématique ne doit jamais faire perdre les courses déjà traitées
  ce jour-là, ni interrompre le reste du backfill (même principe que la
  discussion du bug de collecte des stats jockeys du 14/08/2026) ;
- tout champ destiné à une colonne PostgreSQL `integer` passe par
  _safe_int(), qui renvoie None (au lieu de laisser planter l'insertion)
  si la valeur brute de l'API PMU dépasse la plage int4
  (-2147483648 à 2147483647) ou n'est pas un entier valide.

Concurrence (ajoutée le 18/08/2026 après le backfill historique massif en
jobs parallèles) : lorsque plusieurs instances de ce script tournent en même
temps (matrix GitHub Actions), l'appel à init_schema() par chaque instance
provoquait un psycopg2.errors.DeadlockDetected (plusieurs sessions
exécutaient le même DDL — CREATE TABLE IF NOT EXISTS — en même temps sur les
mêmes tables). La variable d'environnement optionnelle SKIP_INIT_SCHEMA
permet à un appelant qui SAIT que le schéma existe déjà de sauter cette
vérification. Elle n'est définie par AUCUN des workflows existants
(backfill-resultats.yml, collecte-resultats-quotidien.yml) : leur
comportement est donc strictement inchangé, init_schema() continue de
s'exécuter pour eux exactement comme avant. Seul le backfill massif en
parallèle (backfill-historique-massif.yml) la définit.

Résilience connexion (ajoutée le 19/08/2026 après un incident survenu
pendant le backfill historique massif : sur plusieurs jobs de plusieurs
heures, la connexion Postgres (Supabase) est passée en lecture seule en
cours de route (psycopg2.errors.ReadOnlySqlTransaction), très probablement
un incident transitoire côté infrastructure (bascule/redémarrage) plutôt
qu'un problème de nos données. Deux conséquences avant le correctif :
1) chaque course suivante échouait silencieusement (rattrapée par le
   try/except par course, donc sans planter le script, mais sans plus
   écrire une seule ligne en base non plus) ; 2) le script finissait par
   planter à l'appel _log() en fin de journée (seul point non protégé),
   perdant ainsi tout le travail restant du job. Corrigé par :
- une détection explicite des erreurs de connexion/transaction (au lieu de
  les traiter comme une simple course en échec) qui interrompt le jour en
  cours immédiatement plutôt que de perdre toutes les courses restantes en
  silence ;
- main() retente désormais le jour en échec jusqu'à 4 fois avec une
  reconnexion complète (nouvelle connexion psycopg2) et un backoff
  croissant entre chaque tentative, avant d'abandonner ce jour précis et de
  continuer avec le suivant (jamais tout le script) ;
- _log() est maintenant best-effort : si l'écriture du journal échoue
  elle-même, le message est affiché sur stdout au lieu de laisser
  planter tout le run pour la perte d'une simple ligne de diagnostic.

Migration base PLAT-only (ajoutée le 20/08/2026, voir décision de migration
suite à l'incident de quota/disque Supabase sur l'ancienne base) :
- `raw_json` retiré de l'INSERT resultats_courses : colonne supprimée du
  nouveau schéma (confirmé inutilisée par tout le pipeline, sauvegardée
  séparément avant suppression).
- 7 colonnes marginales retirées de l'INSERT resultats_partants
  (driver_change, poids_condition_monte_change,
  distance_cheval_precedent_code, indicateur_inedit, jument_pleine,
  nombre_places_second, nombre_places_troisieme) : confirmées non lues par
  aucun script actif (analyse_variables_avancee.py, entrainer_et_evaluer.py,
  predict_course.py, predict_toutes_courses_jour.py), colonnes supprimées
  du nouveau schéma. Toutes les autres colonnes sont conservées à
  l'identique. DATABASE_URL doit pointer vers le nouveau projet Supabase
  PLAT-only (secret GitHub à mettre à jour séparément).
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

INT4_MIN = -2147483648
INT4_MAX = 2147483647

# Erreurs indiquant que la connexion/transaction elle-même est cassée
# (par opposition à une donnée PMU inattendue sur une course précise) :
# inutile de continuer à boucler sur les courses restantes du jour, elles
# échoueraient toutes de la même façon jusqu'à reconnexion.
ERREURS_CONNEXION = (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.errors.ReadOnlySqlTransaction)


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


def _safe_int(v):
    """Convertit vers un entier compatible avec une colonne PostgreSQL
    `integer` (int4). Renvoie None plutôt que de laisser planter
    l'INSERT si la valeur est absente, non numérique, ou hors de la plage
    int4 — l'API PMU (non officielle) peut renvoyer des valeurs
    inattendues sans prévenir, mieux vaut perdre un champ que toute la
    course."""
    if v is None:
        return None
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return None
    if iv < INT4_MIN or iv > INT4_MAX:
        return None
    return iv


def backfill_jour(conn, date_str_ddmmyyyy):
    """Récupère toutes les courses d'un jour donné. Retourne le nombre de
    courses effectivement insérées (avec au moins un partant classé).
    Chaque course est commitée individuellement : un problème sur une
    course (donnée PMU inattendue, erreur réseau) ne fait perdre que
    cette course, jamais les autres déjà traitées le même jour.

    Si la connexion/transaction elle-même est cassée (ex: bascule en
    lecture seule côté Supabase), la fonction lève l'exception au lieu de
    continuer à boucler : c'est à l'appelant (main()) de reconnecter et de
    retenter le jour entier."""
    try:
        programme = _http_get_json(f"{PROGRAMME_BASE}/{date_str_ddmmyyyy}")
    except Exception as e:
        _log(conn, date_str_ddmmyyyy, 0, "ECHEC", f"programme: {type(e).__name__}: {e}")
        conn.commit()
        return 0

    n_courses = 0
    n_echecs = 0
    for reunion in programme.get("programme", {}).get("reunions", []):
        r_num = reunion.get("numOfficiel")
        hippodrome = (reunion.get("hippodrome") or {}).get("libelleCourt", "INCONNU")
        meteo = reunion.get("meteo") or {}
        for course in reunion.get("courses", []):
            c_num = course.get("numExterne") or course.get("numOrdre")
            statut = course.get("statut", "")
            if r_num is None or c_num is None or statut in STATUTS_A_IGNORER:
                continue

            # Reconstruction Plan B (21/08/2026) : base PLAT-only, on ignore
            # dès ce point toute course dont ni discipline ni specialite ne
            # vaut "PLAT" (trot ATTELE/MONTE etc.) — évite à la fois d'écrire
            # des données hors-scope dans la nouvelle base et d'appeler
            # l'API /participants pour rien sur ~60% des courses du jour.
            if course.get("discipline") != "PLAT" and course.get("specialite") != "PLAT":
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

            try:
                penetrometre = course.get("penetrometre") or {}
                with conn.cursor() as cur:
                    cur.execute(
                        """INSERT INTO resultats_courses
                           (course_id, date_course, hippodrome, r_c, discipline, distance_m,
                            prix_nom, montant_allocation, partants_declares, heure_depart,
                            date_collecte,
                            meteo_temperature, meteo_force_vent, meteo_direction_vent, meteo_nebulosite,
                            terrain_intitule, terrain_valeur_penetrometre,
                            corde, type_piste, categorie_particularite, condition_age, condition_sexe, specialite)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (course_id) DO NOTHING""",
                        (
                            course_id, date_iso, hippodrome, f"R{r_num}/C{c_num}",
                            course.get("discipline"), _safe_int(course.get("distance")), course.get("libelle"),
                            _safe_int(course.get("montantTotalOffert") or course.get("montantPrix")),
                            len(participants), course.get("heureDepart"),
                            datetime.utcnow().isoformat(),
                            _safe_int(meteo.get("temperature")), _safe_int(meteo.get("forceVent")),
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
                                nom_pere, nom_mere, oeilleres, deferre, avis_entraineur,
                                nombre_courses, nombre_victoires, nombre_places,
                                gains_victoires, gains_place,
                                gains_annee_encours, gains_annee_precedente, handicap_distance,
                                temps_obtenu, reduction_kilometrique, incident, commentaire_apres_course,
                                id_cheval, nom_pere_mere, handicap_valeur, handicap_poids,
                                poids_condition_monte,
                                distance_cheval_precedent_libelle,
                                place_corde, race, pays,
                                pays_entrainement, proprietaire, eleveur, robe)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                       %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                               ON CONFLICT (course_id, numero) DO NOTHING""",
                            (
                                course_id, _safe_int(p["numPmu"]), p.get("nom"), p.get("sexe"), _safe_int(p.get("age")),
                                p.get("driver"), p.get("entraineur"), p.get("musique"),
                                _safe_int(gp.get("gainsCarriere")),
                                _safe_int(p.get("ordreArrivee")),
                                p.get("nomPere"), p.get("nomMere"), p.get("oeilleres"), p.get("deferre"),
                                p.get("avisEntraineur"),
                                _safe_int(p.get("nombreCourses")), _safe_int(p.get("nombreVictoires")), _safe_int(p.get("nombrePlaces")),
                                _safe_int(gp.get("gainsVictoires")), _safe_int(gp.get("gainsPlace")),
                                _safe_int(gp.get("gainsAnneeEnCours")), _safe_int(gp.get("gainsAnneePrecedente")),
                                _safe_int(p.get("handicapDistance")), _safe_int(p.get("tempsObtenu")), _safe_int(p.get("reductionKilometrique")),
                                p.get("incident"), commentaire.get("texte"),
                                p.get("idCheval"), p.get("nomPereMere"), p.get("handicapValeur"), _safe_int(p.get("handicapPoids")),
                                _safe_int(p.get("poidsConditionMonte")),
                                distance_prec.get("libelleLong"),
                                _safe_int(p.get("placeCorde")),
                                p.get("race"), p.get("pays"), p.get("paysEntrainement"),
                                p.get("proprietaire"), p.get("eleveur"), (p.get("robe") or {}).get("libelleLong"),
                            ),
                        )
                conn.commit()
                n_courses += 1
            except ERREURS_CONNEXION as e:
                # Connexion/transaction cassée (ex: bascule en lecture seule
                # côté Supabase) : continuer à boucler sur les courses
                # restantes ne ferait que toutes les perdre en silence
                # jusqu'à la fin de la journée. On abandonne ce jour tout de
                # suite et on laisse main() reconnecter et retenter.
                try:
                    conn.rollback()
                except Exception:
                    pass
                print(f"  [CONNEXION PERDUE] {date_iso} R{r_num}/C{c_num} {hippodrome} : {type(e).__name__}: {e} — abandon du jour, nouvelle tentative programmée.")
                raise
            except Exception as e:
                conn.rollback()
                n_echecs += 1
                print(f"  [ECHEC course] {date_iso} R{r_num}/C{c_num} {hippodrome} : {type(e).__name__}: {e}")

            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    _log(conn, date_str_ddmmyyyy, n_courses, "OK", f"{n_courses} course(s) avec résultats, {n_echecs} échec(s)")
    conn.commit()
    return n_courses


def _log(conn, date_str, n, statut, message):
    """Journalise le résultat du jour. Best-effort (ajouté le 19/08/2026) :
    si l'écriture du log échoue elle-même (ex: connexion en lecture seule),
    on affiche le message sur stdout au lieu de laisser planter tout le
    script pour la perte d'une simple ligne de diagnostic."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO resultats_log (date_execution, date_cible, nb_courses_recuperees, statut, message)
                   VALUES (%s,%s,%s,%s,%s)""",
                (datetime.utcnow().isoformat(), date_str, n, statut, message),
            )
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"  [LOG NON ÉCRIT] {date_str} : {statut} — {message} (cause : {type(e).__name__}: {e})")


def main():
    date_debut_str = os.environ.get("DATE_DEBUT")  # format attendu: YYYY-MM-DD
    date_fin_str = os.environ.get("DATE_FIN")  # format attendu: YYYY-MM-DD

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
    if os.environ.get("SKIP_INIT_SCHEMA", "").lower() == "true":
        print("SKIP_INIT_SCHEMA=true — vérification du schéma ignorée (jobs parallèles, schéma déjà garanti par ailleurs).")
    else:
        init_schema(conn)

    total = 0
    d = date_debut
    while d <= date_fin:
        date_str_ddmmyyyy = d.strftime("%d%m%Y")
        # Résilience connexion (19/08/2026) : on retente le jour en cours
        # jusqu'à 4 fois avec reconnexion complète entre chaque tentative,
        # au lieu de laisser une erreur de connexion (ex: bascule en
        # lecture seule côté Supabase) planter tout le script et perdre
        # tous les jours restants de la tranche. ON CONFLICT DO NOTHING
        # protège contre les doublons sur les courses déjà insérées avant
        # l'incident.
        n = None
        backoffs = (5, 15, 45)
        for tentative in range(1, 5):
            try:
                n = backfill_jour(conn, date_str_ddmmyyyy)
                break
            except Exception as e:
                print(f"  [ECHEC JOUR] {d.isoformat()} tentative {tentative}/4 : {type(e).__name__}: {e}")
                try:
                    conn.close()
                except Exception:
                    pass
                if tentative < 4:
                    time.sleep(backoffs[tentative - 1])
                    conn = get_connection()
        if n is None:
            print(f"  [JOUR ABANDONNÉ] {d.isoformat()} après 4 tentatives — passage au jour suivant (à retenter séparément)."
            n = 0
        total += n
        print(f"{d.isoformat()} : {n} course(s)")
        d += timedelta(days=1)

    print(f"\nTerminé : {total} courses au total sur la période {date_debut} → {date_fin}")
    conn.close()


if __name__ == "__main__":
    main()
