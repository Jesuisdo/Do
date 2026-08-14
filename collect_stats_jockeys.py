"""
collect_stats_jockeys.py — Collecteur de statistiques jockeys/drivers/entraîneurs
(source : Geny.com, complémentaire aux données PMU qui ne fournissent que le nom).

Fonctionnement en deux temps, volontairement séparés :

1. RÉSOLUTION (manuelle, incrémentale) : PMU ne donne les jockeys/drivers/
   entraîneurs que sous forme abrégée ("M.GUYON"), alors que Geny.com identifie
   chaque personne par un identifiant numérique lié à son nom complet
   ("Maxime Guyon" -> jockey n°1011478). Il n'existe pas d'API de recherche
   publique chez Geny pour automatiser cette correspondance : elle est donc
   construite à la main, au fur et à mesure, dans la table
   `mapping_intervenants` (colonnes nom_pmu / id_geny / slug_geny /
   statut_resolution). Ce script ne résout PAS de nouveaux noms tout seul —
   il se contente d'aller chercher les stats de ceux déjà marqués 'RESOLU'.
   Priorité de résolution : jockeys les plus fréquents en GALOP d'abord (voir
   requête de priorisation dans le README de ce dossier).

2. COLLECTE (automatisable, ce script) : pour chaque intervenant résolu,
   récupère sa fiche Geny et en extrait les statistiques agrégées sur 12 mois
   (courses, victoires, places, écarts, rapports moyens). Écrit une nouvelle
   ligne par jour d'exécution dans `stats_intervenants` (historique conservé,
   pas d'écrasement) — pensé pour tourner une fois par semaine (ces stats ne
   bougent pas course par course).

Principe de prudence, comme le reste du pipeline : toute erreur de parsing
est journalisée explicitement (jamais avalée silencieusement) — voir la
discussion sur le bug du dashboard qui affichait "aucune donnée" au lieu
d'une vraie erreur. Un échec de parsing sur UN intervenant ne doit jamais
faire échouer la collecte des autres.
"""
import os
import re
import time
from datetime import datetime, timezone

import psycopg2
import requests

DATABASE_URL = os.environ.get("DATABASE_URL")
GENY_BASE = "https://www.geny.com/jockey"
SLEEP_BETWEEN_CALLS_SEC = 2.0
# En-têtes proches d'un vrai navigateur : un simple "requests.py-ish" User-Agent
# se fait bloquer plus facilement par les protections anti-bot (Cloudflare et
# assimilés) que peuvent avoir des sites comme Geny.com face à des IP de
# datacenter (celles des runners GitHub Actions notamment).
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.geny.com/",
}

# Ancre de fin de bloc : cette phrase suit toujours le bloc de stats "12
# derniers mois" qu'on veut extraire, et permet de délimiter la zone de texte
# à parser sans dépendre de la structure HTML exacte (non vérifiable sans
# accès direct au DOM au moment de l'écriture de ce script — voir README).
ANCRE_FIN_BLOC = "Statistiques réalisées sur les courses des 12 derniers mois"

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t\xa0]+")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL absente de l'environnement.")
    return psycopg2.connect(DATABASE_URL)


def _to_float(s):
    if s is None:
        return None
    try:
        return float(s.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _strip_html(html: str) -> str:
    """Supprime les balises et normalise les espaces, pour retomber sur un
    texte comparable à ce qu'on observe en lecture normale de la page."""
    text = TAG_RE.sub("\n", html)
    text = WS_RE.sub(" ", text)
    return text


def parse_stats_bloc(text: str):
    """Extrait le bloc de statistiques '12 derniers mois' à partir du texte
    nettoyé de la page jockey. Retourne un dict, ou lève ValueError si le
    bloc attendu n'est pas trouvé (plutôt que de retourner des valeurs
    fantômes)."""
    idx_fin = text.find(ANCRE_FIN_BLOC)
    if idx_fin == -1:
        raise ValueError(f"Ancre de fin de bloc introuvable ('{ANCRE_FIN_BLOC}')")
    # On prend une fenêtre raisonnable avant l'ancre plutôt que tout le texte,
    # pour éviter d'attraper par erreur un bloc similaire plus haut/bas.
    bloc = text[max(0, idx_fin - 1500):idx_fin]

    def _search(pattern, cast=int):
        m = re.search(pattern, bloc, re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        return cast(m.group(1))

    nb_courses = _search(r"(\d+)\s*courses")
    victoires = _search(r"(\d+)\s*Victoires\s*\(([\d.,]+)\s*%\)")
    m_v = re.search(r"(\d+)\s*Victoires\s*\(([\d.,]+)\s*%\)", bloc, re.IGNORECASE | re.DOTALL)
    m_23 = re.search(r"(\d+)\s*Places\s*2.\s*et\s*3.\s*\(([\d.,]+)\s*%\)", bloc, re.IGNORECASE | re.DOTALL)
    m_45 = re.search(r"(\d+)\s*Places\s*4.\s*et\s*5.\s*\(([\d.,]+)\s*%\)", bloc, re.IGNORECASE | re.DOTALL)
    ecart_g = _search(r"Ecart\s*gagnant\s*:?\s*(\d+)")
    rapport_g = _search(r"rapport moyen gagnant\s*:?\s*([\d.,]+)\s*€", cast=_to_float)
    ecart_p = _search(r"Ecart\s*plac[ée]\s*:?\s*(\d+)")
    rapport_p = _search(r"Rapport moyen plac[ée]\s*:?\s*([\d.,]+)\s*€", cast=_to_float)

    if nb_courses is None or m_v is None:
        raise ValueError("Champs essentiels (nb_courses / victoires) introuvables dans le bloc extrait")

    return {
        "nb_courses_12mois": nb_courses,
        "victoires": int(m_v.group(1)),
        "victoires_pct": _to_float(m_v.group(2)),
        "places_2_3": int(m_23.group(1)) if m_23 else None,
        "places_2_3_pct": _to_float(m_23.group(2)) if m_23 else None,
        "places_4_5": int(m_45.group(1)) if m_45 else None,
        "places_4_5_pct": _to_float(m_45.group(2)) if m_45 else None,
        "ecart_gagnant": ecart_g,
        "rapport_moyen_gagnant": rapport_g,
        "ecart_place": ecart_p,
        "rapport_moyen_place": rapport_p,
    }


def fetch_stats_intervenant(slug_geny: str):
    url = f"{GENY_BASE}/{slug_geny}"
    resp = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
    if resp.status_code != 200:
        # Diagnostic explicite plutôt qu'un simple raise_for_status() : on veut
        # savoir, dans les logs du prochain run, si c'est un blocage anti-bot
        # (ex: 403, ou 200 mais page de vérification Cloudflare) plutôt que de
        # se contenter d'un message générique.
        raise RuntimeError(
            f"HTTP {resp.status_code} pour {url} — extrait réponse: {resp.text[:200]!r}"
        )
    text = _strip_html(resp.text)
    try:
        return parse_stats_bloc(text)
    except ValueError as e:
        # Idem : si le parsing échoue alors qu'on a bien reçu un 200, c'est
        # probablement que le contenu reçu n'est pas la vraie page (page de
        # vérification, redirection inattendue, structure changée) — on
        # journalise un extrait pour pouvoir diagnostiquer sans avoir à
        # reproduire l'appel manuellement.
        raise ValueError(f"{e} — url finale: {resp.url}, longueur texte: {len(text)}, début: {text[:300]!r}")


def main():
    conn = get_connection()
    today = datetime.now(timezone.utc).date().isoformat()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT nom_pmu, role, id_geny, nom_complet_geny, slug_geny "
            "FROM mapping_intervenants WHERE statut_resolution = 'RESOLU' AND slug_geny IS NOT NULL"
        )
        intervenants = cur.fetchall()

    print(f"{len(intervenants)} intervenant(s) résolu(s) à mettre à jour.")

    n_ok, n_echec = 0, 0
    for nom_pmu, role, id_geny, nom_complet, slug in intervenants:
        try:
            stats = fetch_stats_intervenant(slug)
        except Exception as e:
            print(f"[ECHEC] {nom_pmu} ({nom_complet}) : {type(e).__name__}: {e}")
            n_echec += 1
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)
            continue

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO stats_intervenants
                   (id_geny, nom_complet, role, nb_courses_12mois, victoires, victoires_pct,
                    places_2_3, places_2_3_pct, places_4_5, places_4_5_pct,
                    ecart_gagnant, rapport_moyen_gagnant, ecart_place, rapport_moyen_place, date_maj)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id_geny, date_maj) DO UPDATE SET
                     nb_courses_12mois = EXCLUDED.nb_courses_12mois,
                     victoires = EXCLUDED.victoires,
                     victoires_pct = EXCLUDED.victoires_pct,
                     places_2_3 = EXCLUDED.places_2_3,
                     places_2_3_pct = EXCLUDED.places_2_3_pct,
                     places_4_5 = EXCLUDED.places_4_5,
                     places_4_5_pct = EXCLUDED.places_4_5_pct,
                     ecart_gagnant = EXCLUDED.ecart_gagnant,
                     rapport_moyen_gagnant = EXCLUDED.rapport_moyen_gagnant,
                     ecart_place = EXCLUDED.ecart_place,
                     rapport_moyen_place = EXCLUDED.rapport_moyen_place""",
                (id_geny, nom_complet, role, stats["nb_courses_12mois"], stats["victoires"], stats["victoires_pct"],
                 stats["places_2_3"], stats["places_2_3_pct"], stats["places_4_5"], stats["places_4_5_pct"],
                 stats["ecart_gagnant"], stats["rapport_moyen_gagnant"],
                 stats["ecart_place"], stats["rapport_moyen_place"], today),
            )
        conn.commit()
        n_ok += 1
        print(f"[OK] {nom_pmu} ({nom_complet}) : {stats['victoires']}/{stats['nb_courses_12mois']} "
              f"({stats['victoires_pct']}%)")
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    print(f"\nTerminé : {n_ok} succès, {n_echec} échec(s) sur {len(intervenants)} intervenant(s).")
    conn.close()


if __name__ == "__main__":
    main()
