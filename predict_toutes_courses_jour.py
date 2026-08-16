"""
predict_toutes_courses_jour.py — Version autonome et quotidienne de
predict_course.py.

Contrairement à predict_course.py (une seule course, déclenché manuellement
avec DATE_CIBLE/R_NUM/C_NUM), ce script fait DEUX choses à chaque exécution,
dans l'ordre :

  1. Réconciliation : relie les prédictions déjà enregistrées à leur résultat
     réel désormais connu (position_arrivee), dès que la collecte quotidienne
     des résultats (6h) les a rendus disponibles. Condition nécessaire pour
     le forward-testing (comparer objectivement pronostic vs réalité).

  2. Génération : récupère le programme PMU du jour, identifie TOUTES les
     courses PLAT (galop) pas encore courues (statut "PROGRAMMEE"), et
     enregistre un pronostic (méthode "combine_v1", même formule déjà
     backtestée que predict_course.py) pour chacune qui n'en a pas déjà un
     (idempotent — peut être relancé sans dupliquer).

Pensé pour tourner une fois par jour via GitHub Actions (voir
.github/workflows/predictions-quotidiennes.yml), après la collecte des
résultats de la veille (6h) — contrairement à la tâche Cowork équivalente
qu'il remplace, ce script tourne sur les serveurs GitHub, indépendamment de
si l'ordinateur/l'app est allumé.

Principe de prudence habituel du projet : un échec isolé (une course, un
appel réseau) ne doit jamais interrompre le traitement des autres — voir la
discussion du bug de collecte des stats jockeys (14/08/2026).

Rappel de prudence méthodologique : ce script calcule un score selon une
méthode déjà backtestée, mais cette méthode n'est PAS encore validée
statistiquement (intervalle de confiance du ROI encore trop large sur
l'échantillon actuel — voir Journal des Hypothèses). Une prédiction ici est
un point de données pour évaluer le modèle dans le temps, pas un conseil de
pari.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
PROGRAMME_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL absente de l'environnement.")
    return psycopg2.connect(DATABASE_URL)


def _http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "recherche-hippique-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _forme_recente(musique: str):
    """Voir predict_course.py : premier caractère de la musique, plus petit =
    meilleur ; non classé/disqualifié/vide = pire cas (99)."""
    if not musique:
        return 99
    first_char = musique[0]
    if first_char.isdigit():
        val = int(first_char)
        return 99 if val == 0 else val
    return 99


def reconcilier_predictions(conn):
    """Relie chaque prédiction déjà enregistrée à son résultat réel dès qu'il
    est connu. Sans risque à relancer (ne touche que les lignes encore NULL)."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE predictions p
            SET position_arrivee = rp.position_arrivee
            FROM resultats_partants rp
            WHERE rp.course_id = p.course_id AND rp.numero = p.numero
              AND p.position_arrivee IS NULL AND rp.position_arrivee IS NOT NULL
        """)
        n = cur.rowcount
    conn.commit()
    return n


def fetch_programme_plat_du_jour(date_ddmmyyyy):
    """Retourne la liste des courses PLAT du jour pas encore courues :
    [(r_num, c_num, hippodrome), ...]. Champs confirmés le 16/08/2026 par
    inspection directe du JSON réel : reunion["numOfficiel"] (numéro de
    réunion), course["numOrdre"] (numéro de course dans la réunion),
    course["specialite"] (discipline), course["statut"] ("PROGRAMMEE" tant
    que la course n'a pas eu lieu)."""
    programme = _http_get_json(f"{PROGRAMME_BASE}/{date_ddmmyyyy}")
    courses = []
    for reunion in programme.get("programme", {}).get("reunions", []):
        r_num = reunion.get("numOfficiel")
        hippodrome = (reunion.get("hippodrome") or {}).get("libelleCourt", "INCONNU")
        if r_num is None:
            continue
        for course in reunion.get("courses", []):
            discipline = course.get("specialite") or course.get("discipline")
            statut = course.get("statut")
            c_num = course.get("numOrdre")
            if discipline == "PLAT" and statut == "PROGRAMMEE" and c_num is not None:
                courses.append((r_num, c_num, hippodrome))
    return courses


def fetch_participants(date_ddmmyyyy, r_num, c_num):
    detail = _http_get_json(f"{PROGRAMME_BASE}/{date_ddmmyyyy}/R{r_num}/C{c_num}/participants")
    return detail.get("participants", [])


def construire_partants(participants):
    """Filtre les non-partants et extrait les champs utiles au score."""
    rows = []
    for p in participants:
        if p.get("statut") not in (None, "PARTANT"):
            continue
        numero = p.get("numPmu")
        if numero is None:
            continue
        gains = (p.get("gainsParticipant") or {}).get("gainsCarriere") or 0
        rows.append({
            "numero": numero,
            "nom_cheval": p.get("nom"),
            "nom_jockey": p.get("driver"),
            "forme": _forme_recente(p.get("musique") or ""),
            "nb_courses": p.get("nombreCourses") or 0,
            "nb_victoires": p.get("nombreVictoires") or 0,
            "gains": gains,
        })
    return rows


def deja_predite(conn, course_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM predictions WHERE course_id = %s AND methode = 'combine_v1' LIMIT 1",
            (course_id,),
        )
        return cur.fetchone() is not None


def calculer_et_enregistrer(conn, rows, course_id, date_course, hippodrome, r_c):
    """Même logique de score que predict_course.py (rang_forme +
    rang_taux_victoire + rang_gains + rang_jockey, plus petit = meilleur)."""
    if len(rows) < 3:
        raise ValueError(f"Trop peu de partants valides ({len(rows)}) pour un score utile.")

    values_sql = ",\n    ".join("(%s,%s,%s,%s,%s,%s,%s)" for _ in rows)
    params = []
    for r in rows:
        params.extend([
            r["numero"], r["nom_cheval"], r["nom_jockey"], r["forme"],
            r["nb_courses"], r["nb_victoires"], r["gains"],
        ])

    query = f"""
    WITH partants AS (
      SELECT * FROM (VALUES
        {values_sql}
      ) AS t(numero, nom_cheval, nom_jockey, forme_norm, nb_courses, nb_victoires, gains)
    ),
    avec_jockey AS (
      SELECT p.*,
        ROUND(p.nb_victoires::numeric / NULLIF(p.nb_courses,0), 4) AS taux_victoire,
        (SELECT s.victoires_pct FROM mapping_intervenants m
         JOIN stats_intervenants s ON s.id_geny = m.id_geny
         WHERE m.nom_pmu = p.nom_jockey AND m.statut_resolution = 'RESOLU'
         ORDER BY s.date_maj DESC LIMIT 1) AS jockey_victoires_pct
      FROM partants p
    ),
    rangs AS (
      SELECT *,
        RANK() OVER (ORDER BY forme_norm ASC) AS rang_forme,
        RANK() OVER (ORDER BY taux_victoire DESC NULLS LAST) AS rang_taux_victoire,
        RANK() OVER (ORDER BY gains DESC) AS rang_gains,
        RANK() OVER (ORDER BY COALESCE(jockey_victoires_pct, -1) DESC) AS rang_jockey
      FROM avec_jockey
    ),
    final AS (
      SELECT *,
        (rang_forme + rang_taux_victoire + rang_gains + rang_jockey) AS score_combine,
        RANK() OVER (ORDER BY (rang_forme + rang_taux_victoire + rang_gains + rang_jockey) ASC) AS rang_predit
      FROM rangs
    )
    INSERT INTO predictions (course_id, date_course, hippodrome, r_c, numero, nom_cheval, nom_jockey,
                              rang_forme, rang_taux_victoire, rang_gains, rang_jockey,
                              score_combine, rang_predit, methode, date_prediction)
    SELECT %s, %s, %s, %s, numero, nom_cheval, nom_jockey,
           rang_forme, rang_taux_victoire, rang_gains, rang_jockey, score_combine, rang_predit,
           'combine_v1', %s
    FROM final
    ON CONFLICT (course_id, numero, methode) DO NOTHING
    RETURNING numero;
    """
    params.extend([course_id, date_course, hippodrome, r_c, datetime.now(timezone.utc).isoformat()])

    with conn.cursor() as cur:
        cur.execute(query, params)
        n = cur.rowcount
    conn.commit()
    return n


def main():
    date_ddmmyyyy = datetime.now(timezone.utc).strftime("%d%m%Y")
    date_iso = datetime.strptime(date_ddmmyyyy, "%d%m%Y").date().isoformat()

    conn = get_connection()

    n_reconcilie = reconcilier_predictions(conn)
    print(f"Réconciliation : {n_reconcilie} prédiction(s) reliée(s) à un résultat réel.")

    try:
        courses = fetch_programme_plat_du_jour(date_ddmmyyyy)
    except Exception as e:
        print(f"[ECHEC] Impossible de récupérer le programme du jour : {type(e).__name__}: {e}")
        conn.close()
        sys.exit(0)

    print(f"{len(courses)} course(s) PLAT non encore courue(s) détectée(s) aujourd'hui.")

    n_ok, n_ignore, n_echec = 0, 0, 0
    for r_num, c_num, hippodrome in courses:
        r_c = f"R{r_num}/C{c_num}"
        course_id = f"{date_iso}_{hippodrome}_R{r_num}C{c_num}".replace(" ", "-")
        try:
            if deja_predite(conn, course_id):
                n_ignore += 1
                continue
            participants = fetch_participants(date_ddmmyyyy, r_num, c_num)
            rows = construire_partants(participants)
            calculer_et_enregistrer(conn, rows, course_id, date_iso, hippodrome, r_c)
            n_ok += 1
            print(f"[OK] {r_c} {hippodrome} — {len(rows)} partant(s) noté(s).")
        except Exception as e:
            n_echec += 1
            print(f"[ECHEC] {r_c} {hippodrome} : {type(e).__name__}: {e}")

    print(f"\nTerminé : {n_ok} nouvelle(s) prédiction(s), {n_ignore} déjà faite(s), {n_echec} échec(s).")
    conn.close()


if __name__ == "__main__":
    main()
