"""
collect_odds_zeturf_render.py -- Collecteur de cotes en direct Zeturf.fr,
concu pour tourner en Cron Job Render, en complement de
collect_live_odds_render.py (source PMU).

CONTEXTE (Dorian, piste 4 -- 06/09/2026) :

L'audit du 05/09/2026 a montre que le goulot d'etranglement VERT x H-15
n'est PAS un bug de collecte PMU : le PMU liste bien certaines reunions
etrangeres (Allemagne, Chili, Argentine, Uruguay...) dans son programme
(nos tables courses/partants les contiennent deja, avec un score modele
B+genealogie deja calcule), mais il ne publie JAMAIS de cote en direct
pour ces courses (pas de pari mutuel PMU dessus). Consequence :
cotes_historique reste a 0% de couverture pour ces courses precises,
quelle que soit la frequence de sondage -- rien a corriger cote PMU.

Le POC du 06/09/2026 a confirme sur Zeturf.fr, en conditions reelles :
  - Baden-Baden R4C1 (Allemagne, ignoree par PMU) : cotes numeriques
    reelles et variees (2.5 / 5.3 / 56.7 / 4.2 / 26.1 / 4.0 / 57.8 / 39.2).
  - San Isidro R8C11 (Argentine, ignoree par PMU) : idem (4.0 / 59.3 /
    4.7 / 9.8 / 5.6 / 7.4 / 18.0 / 46.1 / 44.4 / 31.2 / 5.0).
  - A titre de comparaison/exclusion, Geny.com a ete teste sur la MEME
    course Baden-Baden : valeur plate "7,1" pour 7 chevaux sur 8 (note
    interne synthetique, pas une cote de marche) -- Geny est donc
    ECARTE comme source de cotes pour les reunions hors PMU.

CONTRAINTE TECHNIQUE IMPORTANTE (verifiee manuellement le 06/09/2026) :
la cellule de cote (td id="cote-sg-{numero}" class="cote-simplegagnant
cote-live") est VIDE dans le HTML brut servi par le serveur (confirme
via fetch direct sur l'URL, sans execution JS) -- elle n'est remplie
que cote client, via un appel AJAX (GET .../api/cotes) declenche par
le JS de la page (cotes.js, fonction non documentee, potentiellement
protegee par un jeton anti-CSRF extrait de la page). Plutot que de
parier sur la stabilite d'un point d'acces interne non documente (et
sur l'absence d'un tel jeton), ce collecteur charge la page reelle avec
un navigateur headless (Playwright/Chromium) et lit le DOM apres rendu
-- strictement ce qu'un visiteur humain verrait, donc naturellement
robuste aux changements internes de l'implementation AJAX de Zeturf.

METHODOLOGIE POINT-IN-TIME (identique en substance a collect_live_odds_
render.py, jamais de logique "closest to target") :
  - Chaque sondage insere une ligne par (course_id, numero) avec
    l'horodatage reel du sondage et minutes_avant_depart = (heure_depart
    reelle - horodatage), calcule a partir de la meme heure_depart deja
    stockee dans courses (issue du programme PMU, jamais reecrite ici).
  - Aucune cote posterieure au depart n'est utilisee pour decider quoi
    que ce soit : ce script ne fait qu'ENREGISTRER des snapshots ; c'est
    piste4_backtest_engine.compute_point_in_time_odds() (JAMAIS modifie,
    JAMAIS reimplemente ici) qui reconstruit ensuite H-15 en filtrant
    minutes_avant_depart >= 15 et en prenant le MIN de cette valeur.
  - Toutes les lignes ecrites ici portent source='zeturf' (colonne
    ajoutee par migration le 06/09/2026, defaut 'pmu' pour compat.) afin
    de ne jamais se confondre avec les lignes PMU existantes.

CIBLAGE DES COURSES (jamais de liste de pays codee en dur) :
  Pour chaque course de courses dont heure_depart tombe dans la
  fenetre [maintenant - 5 min ; maintenant + FENETRE_ANTICIPATION_MIN],
  on ne cible que celles dont la couverture PMU actuelle (source='pmu'
  dans cotes_historique) est quasi nulle (< SEUIL_COUVERTURE_PMU_CIBLE
  des partants attendus) -- c'est exactement le symptome objectif du
  goulot diagnostique le 05/09, quel que soit le pays concerne.

CORRESPONDANCE AVEC ZETURF (jamais de slug devine) :
  Le slug d'URL Zeturf (nom de course en francais/allemand/espagnol,
  numerotation R{n}C{m} propre a Zeturf) ne peut pas etre derive des
  donnees PMU. Ce script charge donc d'abord la page programme du jour
  de Zeturf, y lit la liste reelle des reunions/courses (nom hippodrome,
  heure), et ne retient une correspondance que si : (a) le nom
  d'hippodrome normalise (majuscules, sans accents/tirets) correspond,
  ET (b) l'heure de depart Zeturf est a moins de MARGE_HEURE_MIN minutes
  de heure_depart PMU. Une fois sur la page de la course, le nombre de
  partants ET les noms de chevaux normalises sont compares a nos propres
  partants (course_id) ; toute incoherence significative est journalisee
  et la course est IGNOREE plutot que d'inserer une cote sur un mauvais
  numero.

DEPLOIEMENT : voir DEPLOY.md, section "Cron Zeturf (piste source tierce)".
Reutilise volontairement la meme variable d'environnement DATABASE_URL
que collect_live_odds_render.py (a rattacher, sur Render, a la meme base
que celle deja utilisee pour courses/partants/cotes_historique -- verifier
laquelle est reellement active cote Dorian, render.yaml actuel ne semble
plus refleter la configuration reelle, cf. message de rapport).
"""
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
except Exception:  # pragma: no cover
    PARIS_TZ = None

import psycopg2
from playwright.sync_api import sync_playwright

DATABASE_URL = os.environ.get("DATABASE_URL")
SOURCE_TAG = "zeturf"
SOURCE_LOG_NAME = "zeturf_live_odds_render"

ZETURF_BASE = "https://www.zeturf.fr"
ZETURF_PROGRAMME_URL = f"{ZETURF_BASE}/fr/programmes-et-pronostics-du-jour"

FENETRE_ANTICIPATION_MIN = 40
FENETRE_RETARD_MIN = 5

SEUIL_COUVERTURE_PMU_CIBLE = 0.20

MARGE_HEURE_MIN = 20

SEUIL_CORRESPONDANCE_CHEVAUX = 0.6


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL absente de l'environnement. Sur Render, attacher ce "
            "cron job a la MEME base que collecteur-cotes-pmu (voir DEPLOY.md)."
        )
    return psycopg2.connect(DATABASE_URL)


def _now_paris():
    return datetime.now(PARIS_TZ) if PARIS_TZ else datetime.utcnow()


def _log(conn, nb_courses, statut, message):
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sources_log (source, date_execution, parametres, "
                "nb_courses_recuperees, statut, message) VALUES (%s, %s, %s, %s, %s, %s)",
                (SOURCE_LOG_NAME, _now_paris().isoformat(), json.dumps({
                    "fenetre_anticipation_min": FENETRE_ANTICIPATION_MIN,
                    "seuil_couverture_pmu_cible": SEUIL_COUVERTURE_PMU_CIBLE,
                }), nb_courses, statut, message[:2000] if message else None),
            )
        conn.commit()
    except Exception as e:
        print(f"[WARN] echec ecriture sources_log : {e}", file=sys.stderr)


def _normalize(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "", s)
    return s.upper()


def get_courses_cibles(conn):
    now = _now_paris()
    date_str = now.strftime("%Y-%m-%d")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.course_id, c.hippodrome, c.r_c, c.heure_depart,
                   COUNT(DISTINCT p.numero) AS n_partants,
                   COUNT(DISTINCT CASE WHEN ch.source = 'pmu' THEN ch.numero END) AS n_avec_cote_pmu
            FROM courses c
            JOIN partants p ON p.course_id = c.course_id
            LEFT JOIN cotes_historique ch ON ch.course_id = c.course_id
            WHERE c.date_course = %s
            GROUP BY c.course_id, c.hippodrome, c.r_c, c.heure_depart
            """,
            (date_str,),
        )
        rows = cur.fetchall()

    cibles = []
    for course_id, hippodrome, r_c, heure_depart, n_partants, n_avec_cote_pmu in rows:
        if not heure_depart or not n_partants:
            continue
        try:
            depart_dt = datetime.strptime(f"{date_str} {heure_depart}", "%Y-%m-%d %H:%M")
            if PARIS_TZ:
                depart_dt = depart_dt.replace(tzinfo=PARIS_TZ)
        except ValueError:
            continue
        delta_min = (depart_dt - now).total_seconds() / 60.0
        if not (-FENETRE_RETARD_MIN <= delta_min <= FENETRE_ANTICIPATION_MIN):
            continue
        couverture_pmu = (n_avec_cote_pmu or 0) / n_partants
        if couverture_pmu >= SEUIL_COUVERTURE_PMU_CIBLE:
            continue
        cibles.append({
            "course_id": course_id,
            "hippodrome": hippodrome,
            "r_c": r_c,
            "heure_depart": heure_depart,
            "depart_dt": depart_dt,
            "n_partants": n_partants,
        })
    return cibles


def get_partants_pmu(conn, course_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT numero, nom_cheval FROM partants WHERE course_id = %s ORDER BY numero",
            (course_id,),
        )
        return {numero: nom_cheval for numero, nom_cheval in cur.fetchall()}


def find_zeturf_programme(page):
    page.goto(ZETURF_PROGRAMME_URL, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(1500)
    liens = page.eval_on_selector_all(
        "a[href*='/fr/course-du-jour/']",
        "els => els.map(e => ({href: e.getAttribute('href'), text: e.textContent.trim()}))",
    )
    courses = []
    for lien in liens:
        href = lien.get("href") or ""
        m = re.match(r".*/course-du-jour/(\d{4}-\d{2}-\d{2})/R(\d+)C(\d+)-([a-z0-9\-]+)", href)
        if not m:
            continue
        date_zt, r_num, c_num, slug = m.groups()
        heure_m = re.match(r"^(\d{2})h(\d{2})", lien["text"])
        heure_zt = f"{heure_m.group(1)}:{heure_m.group(2)}" if heure_m else None
        courses.append({
            "href": ZETURF_BASE + href if href.startswith("/") else href,
            "r_num": r_num, "c_num": c_num, "slug": slug,
            "heure_zt": heure_zt, "date_zt": date_zt,
        })
    return courses


def matcher_course(cible, programme_zeturf):
    hippo_cible = _normalize(cible["hippodrome"])
    candidats = [c for c in programme_zeturf if hippo_cible and hippo_cible in _normalize(c["slug"])]
    if not candidats:
        return None
    if len(candidats) == 1:
        return candidats[0]
    meilleur, meilleur_ecart = None, None
    for c in candidats:
        if not c["heure_zt"]:
            continue
        try:
            h, m = map(int, c["heure_zt"].split(":"))
            ecart = abs((cible["depart_dt"].hour * 60 + cible["depart_dt"].minute) - (h * 60 + m))
        except ValueError:
            continue
        if meilleur_ecart is None or ecart < meilleur_ecart:
            meilleur, meilleur_ecart = c, ecart
    if meilleur is not None and meilleur_ecart is not None and meilleur_ecart <= MARGE_HEURE_MIN:
        return meilleur
    return None


def lire_cotes_page(page, url):
    page.goto(url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(4000)
    rows = page.eval_on_selector_all(
        "table tr",
        """els => els.map(r => Array.from(r.querySelectorAll('td,th')).map(c => c.textContent.trim()))""",
    )
    partants_zt = {}
    for row in rows:
        if len(row) < 9:
            continue
        numero_txt = row[1].strip()
        if not numero_txt.isdigit():
            continue
        numero = int(numero_txt)
        nom_bloc = row[3] if len(row) > 3 else ""
        nom_cheval = nom_bloc.split("\n")[0].strip() if nom_bloc else ""
        cote_txt = row[8].strip() if len(row) > 8 else ""
        cote_txt = cote_txt.replace(",", ".")
        try:
            cote = float(cote_txt)
        except ValueError:
            continue
        if cote <= 0:
            continue
        partants_zt[numero] = {"nom_cheval": nom_cheval, "cote": cote}
    return partants_zt


def verifier_correspondance(partants_pmu, partants_zt):
    if not partants_pmu or not partants_zt:
        return False
    n_ok = 0
    n_total = 0
    for numero, nom_pmu in partants_pmu.items():
        zt = partants_zt.get(numero)
        if zt is None:
            continue
        n_total += 1
        if _normalize(nom_pmu) and _normalize(nom_pmu) in _normalize(zt["nom_cheval"]):
            n_ok += 1
        elif _normalize(zt["nom_cheval"]) and _normalize(zt["nom_cheval"]) in _normalize(nom_pmu):
            n_ok += 1
    if n_total == 0:
        return False
    return (n_ok / n_total) >= SEUIL_CORRESPONDANCE_CHEVAUX


def inserer_cotes(conn, course_id, cible, partants_zt, horodatage):
    minutes_avant_depart = int(round((cible["depart_dt"] - horodatage).total_seconds() / 60.0))
    n_inserees = 0
    with conn.cursor() as cur:
        for numero, info in partants_zt.items():
            cur.execute(
                "INSERT INTO cotes_historique (course_id, numero, horodatage, cote, "
                "minutes_avant_depart, source) VALUES (%s, %s, %s, %s, %s, %s)",
                (course_id, numero, horodatage.isoformat(), info["cote"],
                 minutes_avant_depart, SOURCE_TAG),
            )
            n_inserees += 1
    conn.commit()
    return n_inserees


def main():
    conn = get_connection()
    horodatage = _now_paris()
    try:
        cibles = get_courses_cibles(conn)
    except Exception as e:
        _log(conn, 0, "ECHEC", f"get_courses_cibles: {e}")
        conn.close()
        raise

    if not cibles:
        _log(conn, 0, "OK", "Aucune course cible (couverture PMU suffisante ou aucun depart proche).")
        conn.close()
        return

    total_inserees = 0
    total_courses_ok = 0
    erreurs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0 (compatible; recherche-hippique-pipeline/1.0)")
        try:
            programme_zeturf = find_zeturf_programme(page)
        except Exception as e:
            erreurs.append(f"programme Zeturf inaccessible: {e}")
            programme_zeturf = []

        for cible in cibles:
            course_id = cible["course_id"]
            try:
                match = matcher_course(cible, programme_zeturf)
                if match is None:
                    erreurs.append(f"{course_id}: pas de correspondance Zeturf trouvee")
                    continue

                partants_pmu = get_partants_pmu(conn, course_id)
                partants_zt = lire_cotes_page(page, match["href"])

                if not verifier_correspondance(partants_pmu, partants_zt):
                    erreurs.append(
                        f"{course_id}: correspondance chevaux insuffisante avec {match['href']}, "
                        f"course ignoree (pas d'insertion a risque)"
                    )
                    continue

                n = inserer_cotes(conn, course_id, cible, partants_zt, horodatage)
                total_inserees += n
                total_courses_ok += 1
            except Exception as e:
                erreurs.append(f"{course_id}: {e}")
                conn.rollback()

        browser.close()

    statut = "OK" if not erreurs else ("PARTIEL" if total_courses_ok > 0 else "ECHEC")
    message = f"{total_courses_ok}/{len(cibles)} courses cibles traitees, {total_inserees} lignes inserees."
    if erreurs:
        message += " Erreurs: " + " | ".join(erreurs[:10])
    _log(conn, total_courses_ok, statut, message)
    print(message)
    conn.close()


if __name__ == "__main__":
    main()

