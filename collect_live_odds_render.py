"""
collect_live_odds_render.py — Collecteur de cotes en direct, conçu pour
tourner en Cron Job Render (24h/24, indépendamment de toute session Cowork).

DIFFÉRENCE IMPORTANTE avec pipeline/collect_live_odds.py (la version locale) :
Cette version ne dépend PAS d'une table `courses` déjà peuplée localement
(le cron Render n'a pas accès à la base SQLite qui vit dans le dossier
Documents de Dorian). À la place, elle attend que `fetch_live_snapshot()`
renvoie DIRECTEMENT la liste des courses actuellement suivies par le flux de
cotes PMU en direct, avec leurs partants et cotes — ce qui est de toute façon
la forme naturelle d'un flux de cotes en direct (il faut bien qu'il liste
les courses concernées). Chaque course est identifiée avec la même
convention que le reste du pipeline (date_hippodrome_r/c), pour permettre
une réconciliation ultérieure avec la base SQLite locale alimentée par la
collecte quotidienne des résultats.

CE QUI RESTE À FAIRE AVANT LE PREMIER DÉPLOIEMENT RÉEL
--------------------------------------------------------
`fetch_live_snapshot()` est un point d'intégration volontairement laissé
incomplet, pour la même raison que dans la version locale : PMU.fr ne
publie pas d'API publique documentée pour les cotes en direct, et
l'environnement qui a préparé ce programme n'a pas d'accès réseau sortant
pour la vérifier empiriquement (voir pipeline/README.md). Une fois le point
d'accès identifié (inspection réseau du navigateur pendant une réunion de
courses réelle) :
  1. Implémenter `fetch_live_snapshot()` pour qu'il retourne une liste de
     dicts au format décrit dans son docstring ci-dessous.
  2. `git push` — Render redéploie automatiquement, aucune autre action
     manuelle nécessaire.

Tout le reste (connexion Postgres, schéma, écriture, journalisation,
horodatage, gestion des erreurs) est fonctionnel tel quel.
"""
import os
import sys
import time
from datetime import datetime

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
SOURCE_NAME = "pmu_live_odds_render"
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema_postgres.sql")


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


def fetch_live_snapshot():
    """POINT D'INTÉGRATION À COMPLÉTER.

    Doit retourner une liste de dicts, un par course actuellement suivie
    par le flux de cotes en direct, au format :
        {
            "course_id": "2026-08-04_VINCENNES_R1C3",   # même convention que le reste du pipeline
            "date_course": "2026-08-04",
            "hippodrome": "VINCENNES",
            "r_c": "R1/C3",
            "heure_depart": "20:15:00",
            "partants": [
                {"numero": 1, "nom_cheval": "EXEMPLE", "cote": 4.3},
                ...
            ],
        }
    """
    raise NotImplementedError(
        "Point d'accès aux cotes en direct non encore identifié/vérifié. "
        "Voir la section 'CE QUI RESTE À FAIRE' en tête de ce fichier."
    )


def minutes_avant_depart(date_course: str, heure_depart: str, now: datetime):
    if not heure_depart:
        return None
    try:
        depart_dt = datetime.strptime(f"{date_course} {heure_depart}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return round((depart_dt - now).total_seconds() / 60)


def poll_once(conn, now: datetime):
    try:
        snapshot = fetch_live_snapshot()
    except NotImplementedError as e:
        _log(conn, now, 0, "ECHEC", str(e))
        conn.commit()
        return

    n_courses = 0
    with conn.cursor() as cur:
        for course in snapshot:
            cid = course["course_id"]
            cur.execute(
                """INSERT INTO courses (course_id, date_course, hippodrome, r_c, heure_depart, date_decouverte)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (course_id) DO NOTHING""",
                (cid, course["date_course"], course["hippodrome"], course["r_c"],
                 course["heure_depart"], now.isoformat()),
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
                    """INSERT INTO cotes_historique (course_id, numero, horodatage, cote, minutes_avant_depart)
                       VALUES (%s,%s,%s,%s,%s)""",
                    (cid, p["numero"], now.isoformat(), p["cote"], m_avant),
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
    poll_once(conn, datetime.utcnow())
    conn.close()


if __name__ == "__main__":
    main()
