"""
generate_report.py — Rapport final obligatoire de la reconstruction PLAT
(Plan B). Lecture seule sur la NOUVELLE base uniquement (DATABASE_URL).
N'accède jamais à l'ancienne base.

Produit :
- nombre exact de courses PLAT et de partants PLAT ;
- première et dernière date couverte ;
- nombre de jours couverts / jours attendus ;
- liste des jours manquants (si non vide) ;
- doublons éventuels (course_id / (course_id, numero)) ;
- nombre d'erreurs/retries consignés dans resultats_log ;
- taille réelle de la nouvelle base.

Écrit le rapport sur stdout et, si GITHUB_STEP_SUMMARY est défini,
l'ajoute aussi au résumé du job GitHub Actions.
"""
import os
from datetime import datetime, timedelta

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
DATE_DEBUT = os.environ.get("DATE_DEBUT", "2014-01-02")
DATE_FIN = os.environ.get("DATE_FIN", "2026-08-18")


def main():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL absente.")

    d0 = datetime.strptime(DATE_DEBUT, "%Y-%m-%d").date()
    d1 = datetime.strptime(DATE_FIN, "%Y-%m-%d").date()
    jours_attendus = (d1 - d0).days + 1

    conn = psycopg2.connect(DATABASE_URL)
    lignes = []
    lignes.append("=== RAPPORT FINAL — Reconstruction historique PLAT (Plan B) ===")
    lignes.append(f"Période cible : {d0.isoformat()} -> {d1.isoformat()} ({jours_attendus} jours attendus)")
    lignes.append("Base cible : nouvelle base PLAT uniquement (aucun accès à l'ancienne base).")
    lignes.append("")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM resultats_courses")
        nb_courses = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM resultats_partants")
        nb_partants = cur.fetchone()[0]
        lignes.append(f"Courses PLAT récupérées : {nb_courses}")
        lignes.append(f"Partants PLAT récupérés : {nb_partants}")
        lignes.append("")

        cur.execute("SELECT MIN(date_course), MAX(date_course), COUNT(DISTINCT date_course) FROM resultats_courses")
        date_min, date_max, nb_jours_avec_course = cur.fetchone()
        lignes.append(f"Première date : {date_min}")
        lignes.append(f"Dernière date : {date_max}")
        lignes.append(f"Jours avec au moins une course PLAT : {nb_jours_avec_course}")
        lignes.append("")

        # Jours réellement "traités" (log OK), quel que soit le nombre de
        # courses trouvées ce jour-là (un jour sans réunion PLAT est
        # légitime, pas un trou).
        cur.execute("SELECT DISTINCT date_cible FROM resultats_log WHERE statut = 'OK'")
        ok_dates = set()
        for (date_cible,) in cur.fetchall():
            try:
                ok_dates.add(datetime.strptime(date_cible, "%d%m%Y").date())
            except (ValueError, TypeError):
                continue
        jours_couverts = len(ok_dates)
        lignes.append(f"Jours couverts (traités avec succès) : {jours_couverts} / {jours_attendus}")

        d = d0
        manquants = []
        while d <= d1:
            if d not in ok_dates:
                manquants.append(d.isoformat())
            d += timedelta(days=1)
        lignes.append(f"Jours manquants : {len(manquants)}")
        if manquants:
            apercu = manquants[:30]
            lignes.append("  " + ", ".join(apercu) + (" ..." if len(manquants) > 30 else ""))
        lignes.append("")

        # Doublons — devraient être structurellement impossibles grâce aux
        # clés (course_id) et (course_id, numero), vérifié quand même.
        cur.execute("SELECT COUNT(*) FROM (SELECT course_id FROM resultats_courses GROUP BY course_id HAVING COUNT(*) > 1) t")
        dup_courses = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM (SELECT course_id, numero FROM resultats_partants GROUP BY course_id, numero HAVING COUNT(*) > 1) t")
        dup_partants = cur.fetchone()[0]
        lignes.append(f"Doublons course_id (resultats_courses) : {dup_courses}")
        lignes.append(f"Doublons (course_id, numero) (resultats_partants) : {dup_partants}")
        lignes.append("")

        # Erreurs / retries consignés
        cur.execute("SELECT COUNT(*) FROM resultats_log WHERE statut != 'OK'")
        nb_echecs_log = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM resultats_log")
        nb_log_total = cur.fetchone()[0]
        lignes.append(f"Entrées resultats_log en échec : {nb_echecs_log} / {nb_log_total} entrées au total")
        lignes.append("")

        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        taille = cur.fetchone()[0]
        lignes.append(f"Taille réelle de la nouvelle base : {taille}")

    conn.close()

    rapport = "\n".join(lignes)
    print(rapport)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("```\n" + rapport + "\n```\n")


if __name__ == "__main__":
    main()
