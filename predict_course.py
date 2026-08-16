"""
predict_course.py — Calcule un pronostic pour UNE course PLAT (galop) à venir,
avant son départ, et enregistre le résultat dans la table `predictions` pour
pouvoir le comparer objectivement au résultat réel une fois la course courue
(forward-testing — plus convaincant que le backtesting rétrospectif, voir
discussion du 14/08/2026).

Méthode : "combine_v1", la même formule de somme de rangs déjà backtestée sur
les données historiques (voir PRG-1A / test combiné) :
  - forme récente (dernier résultat de la musique, plus petit = meilleur)
  - taux de victoire en carrière (nombreVictoires / nombreCourses)
  - gains en carrière
  - % de victoires du jockey/driver sur les 12 derniers mois (stats_intervenants,
    via mapping_intervenants — seulement pour les jockeys déjà résolus ; les
    autres reçoivent le pire rang possible sur ce facteur, pas un score neutre,
    pour ne pas les avantager artificiellement par manque de donnée)

Usage (variables d'environnement) :
  DATE_CIBLE   format DDMMYYYY (ex: 15082026)
  R_NUM        numéro de réunion (ex: 1)
  C_NUM        numéro de course (ex: 3)
  DATABASE_URL

Se déclenche manuellement (workflow_dispatch), jamais automatiquement — une
prédiction n'a de sens que pour une course précise qu'on choisit d'analyser.

Rappel de prudence (établi tout au long de ce projet) : ce script calcule un
score selon une méthode déjà backtestée, mais cette méthode n'est PAS encore
validée statistiquement (intervalle de confiance du ROI encore trop large sur
l'échantillon actuel — voir Journal des Hypothèses). Une prédiction ici est un
point de données pour évaluer le modèle dans le temps, pas un conseil de pari.
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
    """Extrait le résultat le plus récent de la musique (chaîne continue,
    résultat le plus récent en premier — voir découverte du format réel du
    champ musique lors du pilote PRG-1A). Retourne un entier : plus petit =
    meilleur. '0' (non classé / au-delà du visible) et toute valeur non
    numérique (D, T, A...) sont traités comme le pire cas plutôt que comme un
    bon rang, pour ne jamais avantager par erreur un cheval disqualifié/tombé."""
    if not musique:
        return 99
    first_char = musique[0]
    if first_char.isdigit():
        val = int(first_char)
        return 99 if val == 0 else val
    return 99


def fetch_participants(date_ddmmyyyy, r_num, c_num):
    detail = _http_get_json(f"{PROGRAMME_BASE}/{date_ddmmyyyy}/R{r_num}/C{c_num}/participants")
    return detail.get("participants", [])


def fetch_hippodrome(date_ddmmyyyy, r_num):
    """Récupère le nom de l'hippodrome depuis le programme du jour, pour
    construire un course_id dans la MÊME convention que le reste du pipeline
    ({date}_{hippodrome}_R{n}C{m}) — condition nécessaire pour pouvoir
    rapprocher automatiquement une prédiction du résultat réel une fois la
    course courue (jointure sur course_id avec resultats_courses)."""
    try:
        programme = _http_get_json(f"{PROGRAMME_BASE}/{date_ddmmyyyy}")
    except Exception:
        return "INCONNU"
    for reunion in programme.get("programme", {}).get("reunions", []):
        if str(reunion.get("numOfficiel")) == str(r_num):
            return (reunion.get("hippodrome") or {}).get("libelleCourt", "INCONNU")
    return "INCONNU"


def construire_partants(participants):
    """Filtre les non-partants et extrait les champs utiles au score."""
    rows = []
    for p in participants:
        if p.get("statut") not in (None, "PARTANT"):
            continue  # exclut NON_PARTANT, etc.
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


def calculer_et_enregistrer(conn, rows, course_id, date_course, hippodrome, r_c):
    """Envoie les partants dans une requête SQL qui calcule les rangs par
    facteur (même logique que le test combiné déjà backtesté) et insère le
    résultat dans `predictions`. Fait tout le calcul côté SQL plutôt qu'en
    Python pour rester rigoureusement identique à la méthode déjà validée."""
    if len(rows) < 3:
        raise ValueError(f"Trop peu de partants valides ({len(rows)}) pour un score utile.")

    values_sql = ",\n    ".join(
        "(%s,%s,%s,%s,%s,%s,%s)" for _ in rows
    )
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
    ON CONFLICT (course_id, numero, methode) DO UPDATE SET
      rang_forme = EXCLUDED.rang_forme,
      rang_taux_victoire = EXCLUDED.rang_taux_victoire,
      rang_gains = EXCLUDED.rang_gains,
      rang_jockey = EXCLUDED.rang_jockey,
      score_combine = EXCLUDED.score_combine,
      rang_predit = EXCLUDED.rang_predit,
      date_prediction = EXCLUDED.date_prediction
    RETURNING numero, nom_cheval, nom_jockey, rang_predit, score_combine;
    """
    params.extend([course_id, date_course, hippodrome, r_c, datetime.now(timezone.utc).isoformat()])

    with conn.cursor() as cur:
        cur.execute(query, params)
        resultats = cur.fetchall()
    conn.commit()
    return sorted(resultats, key=lambda r: r[3])  # tri par rang_predit


def main():
    date_ddmmyyyy = os.environ.get("DATE_CIBLE")
    r_num = os.environ.get("R_NUM")
    c_num = os.environ.get("C_NUM")

    if not date_ddmmyyyy or not r_num or not c_num:
        print("Variables requises : DATE_CIBLE (DDMMYYYY), R_NUM, C_NUM")
        sys.exit(1)

    participants = fetch_participants(date_ddmmyyyy, r_num, c_num)
    if not participants:
        print(f"Aucun partant trouvé pour R{r_num}C{c_num} le {date_ddmmyyyy}.")
        sys.exit(1)

    rows = construire_partants(participants)
    date_iso = datetime.strptime(date_ddmmyyyy, "%d%m%Y").date().isoformat()
    hippodrome = fetch_hippodrome(date_ddmmyyyy, r_num)
    course_id = f"{date_iso}_{hippodrome}_R{r_num}C{c_num}".replace(" ", "-")
    r_c = f"R{r_num}/C{c_num}"

    conn = get_connection()
    try:
        classement = calculer_et_enregistrer(conn, rows, course_id, date_iso, hippodrome, r_c)
    finally:
        conn.close()

    seuil_place = 2 if len(classement) <= 7 else 3

    print(f"\nPronostic {r_c} du {date_iso} — {len(classement)} partant(s) noté(s) :\n")
    for numero, nom_cheval, nom_jockey, rang_predit, score in classement:
        if rang_predit == 1:
            tag = "[GAGNANT]"
        elif rang_predit <= seuil_place:
            tag = "[PLACE]  "
        else:
            tag = "         "
        print(f"  {rang_predit:>2}. {tag} n°{numero:<3} {nom_cheval:<22} ({nom_jockey})  score={score}")


if __name__ == "__main__":
    main()
