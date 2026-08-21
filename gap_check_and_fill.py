"""
gap_check_and_fill.py — Détection et comblement automatique des jours
manquants après la reconstruction PLAT massive (matrice de 20 jobs).

Un jour est considéré "traité avec succès" s'il existe une ligne
resultats_log avec statut='OK' pour ce jour (peu importe le nombre de
courses PLAT trouvées ce jour-là : un jour sans réunion PLAT est une
donnée légitime, pas un trou). Un jour est "manquant" s'il n'a AUCUNE
ligne OK dans resultats_log — soit parce que le job de sa tranche a
échoué avant de l'atteindre, soit parce que les 4 tentatives internes
du script ont toutes échoué pour ce jour précis.

Ne touche jamais l'ancienne base : DATABASE_URL doit pointer vers la
nouvelle base PLAT (jvfrvttfedmabnoldbyp), comme le script de
reconstruction lui-même. Ce script ne fait que lire/écrire sur la
connexion fournie par DATABASE_URL.

Usage :
    DATABASE_URL=... DATE_DEBUT=2014-01-02 DATE_FIN=2026-08-18 \
    python3 gap_check_and_fill.py
"""
import os
import subprocess
import sys
from datetime import datetime, timedelta

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
DATE_DEBUT = os.environ.get("DATE_DEBUT", "2014-01-02")
DATE_FIN = os.environ.get("DATE_FIN", "2026-08-18")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECONSTRUCTION_SCRIPT = os.path.join(SCRIPT_DIR, "backfill_resultats_plat_reconstruction.py")


def full_range(d0, d1):
    days = []
    d = d0
    while d <= d1:
        days.append(d)
        d += timedelta(days=1)
    return days


def get_ok_days(conn):
    """Renvoie l'ensemble des dates (objets date) ayant au moins une ligne
    resultats_log avec statut='OK'. date_cible est stocké au format
    DDMMYYYY (chaîne) par _log() dans le script de reconstruction."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT date_cible FROM resultats_log WHERE statut = 'OK'")
        rows = cur.fetchall()
    ok = set()
    for (date_cible,) in rows:
        try:
            ok.add(datetime.strptime(date_cible, "%d%m%Y").date())
        except (ValueError, TypeError):
            continue
    return ok


def main():
    if not DATABASE_URL:
        print("DATABASE_URL absente — abandon.", file=sys.stderr)
        sys.exit(1)

    d0 = datetime.strptime(DATE_DEBUT, "%Y-%m-%d").date()
    d1 = datetime.strptime(DATE_FIN, "%Y-%m-%d").date()
    attendu = full_range(d0, d1)

    conn = psycopg2.connect(DATABASE_URL)
    ok_days = get_ok_days(conn)
    conn.close()

    manquants = [d for d in attendu if d not in ok_days]
    print(f"Jours attendus : {len(attendu)} | Jours OK en base : {len(ok_days & set(attendu))} | Jours manquants : {len(manquants)}")

    if not manquants:
        print("Aucun jour manquant — rien à relancer.")
        return

    print(f"Relance individuelle de {len(manquants)} jour(s) manquant(s)...")
    env = dict(os.environ)
    env["SKIP_INIT_SCHEMA"] = "true"
    echecs_persistants = []
    for d in manquants:
        jour_str = d.isoformat()
        env["DATE_DEBUT"] = jour_str
        env["DATE_FIN"] = jour_str
        print(f"  -> relance {jour_str}")
        result = subprocess.run([sys.executable, RECONSTRUCTION_SCRIPT], env=env)
        if result.returncode != 0:
            echecs_persistants.append(jour_str)

    # Deuxième passage de vérification après la relance, pour le rapport
    # final : certains jours peuvent encore manquer si l'API PMU ne
    # répond vraiment pas pour cette date précise (ex: aucune course
    # PLAT n'a jamais existé ce jour côté PMU, ou panne API persistante).
    conn = psycopg2.connect(DATABASE_URL)
    ok_days_apres = get_ok_days(conn)
    conn.close()
    toujours_manquants = [d.isoformat() for d in manquants if d not in ok_days_apres]

    print(f"\nAprès relance : {len(manquants) - len(toujours_manquants)} jour(s) récupéré(s), {len(toujours_manquants)} toujours manquant(s).")
    if toujours_manquants:
        print("Jours toujours manquants après relance (à examiner manuellement) :")
        for j in toujours_manquants:
            print(f"  - {j}")

    # Fichier de sortie pour le job de rapport final
    out_path = os.path.join(SCRIPT_DIR, "jours_manquants_finaux.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(toujours_manquants))
    print(f"\nListe écrite dans {out_path}")


if __name__ == "__main__":
    main()
