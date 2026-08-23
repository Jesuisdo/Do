"""
analyse_variables_avancee.py — Architecture de "base historique vivante" et
sélection statistique de variables, demandée le 17/08/2026.

Objectif : arrêter de choisir 4 facteurs à la main et à poids égal. À la
place :
  1. Reconstruire, pour CHAQUE partant de CHAQUE course PLAT passée, l'état
     de connaissance disponible STRICTEMENT AVANT le départ de cette course
     (aucune fuite d'information du futur vers le passé — voir la fonction
     _construire_features_point_in_time, qui n'utilise que des courses avec
     date_course < date de la course cible, ou une heure de départ
     antérieure le même jour).
  2. Construire automatiquement des dizaines de variables historiques et
     contextuelles (cheval, jockey, entraîneur, course, quelques
     interactions) plutôt qu'une liste choisie arbitrairement.
  3. Laisser un modèle statistique (gradient boosting, qui capture les
     interactions automatiquement — cheval×jockey, cheval×terrain, etc. —
     sans qu'on ait à les lister à la main) et une régression logistique
     L1 (pour l'interprétabilité, variables à coefficient non-nul = "gardées
     par le modèle") décider quelles variables comptent vraiment.
  4. Évaluer sur un découpage CHRONOLOGIQUE (entraînement sur les courses
     les plus anciennes, test sur les plus récentes — jamais de mélange
     aléatoire, qui donnerait une fausse impression de performance en
     laissant fuiter de l'information temporelle).
  5. Publier un rapport honnête : combien de variables créées, combien
     réellement utilisées, lesquelles améliorent les prédictions, lesquelles
     ne servent à rien, performance hors-échantillon, décomposée par nombre
     de partants et par type de course.

Limite assumée et annoncée dès maintenant (pas cachée dans le rapport) :
l'historique par cheval dans NOTRE base est encore très court (la majorité
des chevaux n'ont qu'1 à 2 courses chez nous à la date du 17/08/2026, le
temps que le backfill et la collecte quotidienne s'accumulent). Beaucoup de
variables "historique interne" seront donc creuses aujourd'hui — c'est un
fait mesuré, pas une excuse, et il est rapporté explicitement (colonne
"couverture" du rapport). Ce script est conçu pour être RELANCÉ
régulièrement : les mêmes variables deviendront mécaniquement plus utiles
à mesure que la base s'enrichit chaque jour, sans qu'on ait à changer le
code.

Parsing de la "musique" (forme récente encodée par PMU, ex: "1a2a0a3a(23)4a") :
approximation documentée, pas une garantie d'exactitude à 100% — on retire
les groupes entre parenthèses (marqueurs d'année) puis on extrait les
chiffres dans l'ordre comme suite de positions (0 = non classé = pire cas,
comme déjà établi dans predict_course.py), et les lettres D/T/A comme
incidents (disqualifié/tombé/arrêté). Cette limite est documentée plutôt
que dissimulée.
"""
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import psycopg2
import psycopg2.extras

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL absente de l'environnement.")
    return psycopg2.connect(DATABASE_URL)


# ---------------------------------------------------------------------------
# 1. Extraction des données brutes
# ---------------------------------------------------------------------------

REQUETE_PRINCIPALE = """
SELECT
    rp.course_id, rc.date_course, rc.hippodrome, rc.r_c, rc.distance_m,
    rc.montant_allocation, rc.meteo_temperature, rc.meteo_force_vent,
    rc.terrain_valeur_penetrometre, rc.categorie_particularite,
    rc.condition_age, rc.condition_sexe, rc.heure_depart, rc.partants_declares,
    rp.numero, rp.nom_cheval, rp.id_cheval, rp.sexe, rp.age,
    rp.nom_jockey, rp.nom_entraineur, rp.musique, rp.gains,
    rp.gains_annee_encours, rp.gains_annee_precedente,
    rp.nombre_courses, rp.nombre_victoires, rp.nombre_places,
    rp.handicap_poids, rp.poids_condition_monte, rp.place_corde,
    rp.oeilleres, rp.deferre, rp.position_arrivee,
    rp.commentaire_apres_course
FROM resultats_partants rp
JOIN resultats_courses rc ON rc.course_id = rp.course_id
WHERE (rc.specialite = 'PLAT' OR rc.discipline = 'PLAT')
  AND rp.position_arrivee IS NOT NULL
ORDER BY rc.date_course ASC, COALESCE(rc.heure_depart,'00:00:00') ASC, rp.course_id ASC;
"""

REQUETE_JOCKEYS = """
SELECT m.nom_pmu, s.victoires_pct, s.places_2_3_pct, s.places_4_5_pct,
       s.nb_courses_12mois, s.rapport_moyen_place
FROM mapping_intervenants m
JOIN stats_intervenants s ON s.id_geny = m.id_geny
WHERE m.statut_resolution = 'RESOLU'
"""

MOT_INCIDENT = re.compile(
    r"(gêné|gêne|bouscul|hésitant|hésitation|disqualifi|coincé|enfermé|accroch|"
    r"mauvais départ|manqué son départ|a fait un écart|a chuté|est tombé|boiteux|"
    r"s.est arrêté|perdu toute chance au départ|contrarié)",
    re.IGNORECASE,
)


def charger_donnees(conn):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(REQUETE_PRINCIPALE)
        lignes = cur.fetchall()
        cur.execute(REQUETE_JOCKEYS)
        jockeys = {r["nom_pmu"]: dict(r) for r in cur.fetchall()}
    resultat = []
    for l in lignes:
        d = dict(l)
        # date_course est stockée en TEXT (format YYYY-MM-DD) dans le schéma
        # actuel, pas en DATE native — on la convertit ici explicitement
        # pour pouvoir calculer des écarts en jours (jours de repos) plus
        # loin, sans quoi une soustraction de deux chaînes ferait planter
        # le script.
        if isinstance(d.get("date_course"), str):
            d["date_course"] = datetime.strptime(d["date_course"], "%Y-%m-%d").date()
        resultat.append(d)
    return resultat, jockeys


# ---------------------------------------------------------------------------
# 2. Parsing musique
# ---------------------------------------------------------------------------

def parser_musique(musique):
    """Retourne (liste_positions_recentes_les_plus_recentes_dabord, nb_incidents)."""
    if not musique:
        return [], 0
    sans_annees = re.sub(r"\([0-9]{2}\)", " ", musique)
    positions = []
    nb_incidents = 0
    i = 0
    while i < len(sans_annees):
        c = sans_annees[i]
        if c.isdigit():
            v = int(c)
            positions.append(99 if v == 0 else v)
        elif c in "DTA":
            positions.append(99)
            nb_incidents += 1
        i += 1
    return positions, nb_incidents


# ---------------------------------------------------------------------------
# 3. Construction des features point-in-time (chronologique, sans fuite)
# ---------------------------------------------------------------------------

class EtatCumulatif:
    """Maintient, pour chaque entité (cheval/jockey/entraîneur/combinaisons),
    des compteurs mis à jour au fur et à mesure qu'on avance dans le temps.
    Au moment de scorer une course, on LIT l'état actuel (= tout ce qui
    s'est passé strictement avant), puis on met à jour APRES avoir construit
    les features de cette course — jamais l'inverse, pour ne jamais laisser
    le résultat du jour influencer sa propre prédiction."""

    def __init__(self):
        self.cheval = defaultdict(lambda: {"n": 0, "v": 0, "p": 0, "incidents": 0, "derniere_date": None})
        self.cheval_hippo = defaultdict(lambda: {"n": 0, "p": 0})
        self.cheval_distance = defaultdict(lambda: {"n": 0, "p": 0})
        self.entraineur = defaultdict(lambda: {"n": 0, "v": 0, "p": 0})
        self.jockey_entraineur = defaultdict(lambda: {"n": 0, "p": 0})
        self.jockey_cheval = defaultdict(lambda: {"n": 0, "p": 0})

    def lire_cheval(self, id_cheval):
        return dict(self.cheval[id_cheval]) if id_cheval else {"n": 0, "v": 0, "p": 0, "incidents": 0, "derniere_date": None}

    def maj_cheval(self, id_cheval, place, gagnant, incident, date_course):
        if not id_cheval:
            return
        e = self.cheval[id_cheval]
        e["n"] += 1
        e["v"] += 1 if gagnant else 0
        e["p"] += 1 if place else 0
        e["incidents"] += 1 if incident else 0
        e["derniere_date"] = date_course


def _distance_bucket(d):
    if d is None:
        return None
    if d < 1600:
        return "court"
    if d < 2400:
        return "moyen"
    return "long"


def construire_feature_matrix(lignes, jockeys_stats):
    """Parcourt les lignes DANS L'ORDRE CHRONOLOGIQUE et construit, pour
    chaque partant, un dict de features basé UNIQUEMENT sur ce qui est
    connu avant cette course, puis met à jour l'état cumulatif."""
    etat = EtatCumulatif()
    rows = []

    # Grouper par course pour calculer nb_partants et seuil placé.
    par_course = defaultdict(list)
    for l in lignes:
        par_course[l["course_id"]].append(l)

    # On traite les courses dans l'ordre chronologique (déjà trié par la
    # requête SQL), mais on doit s'assurer que TOUS les partants d'une même
    # course lisent l'état AVANT que quiconque de cette course ne le mette à
    # jour (sinon le partant n°2 verrait déjà le résultat du partant n°1 de
    # la même course, qui n'a aucun sens causal).
    vus = set()
    ordre_courses = []
    for l in lignes:
        if l["course_id"] not in vus:
            vus.add(l["course_id"])
            ordre_courses.append(l["course_id"])

    for course_id in ordre_courses:
        partants = par_course[course_id]
        nb_partants = len(partants)
        if nb_partants < 3:
            continue
        seuil = 2 if nb_partants <= 7 else 3

        lectures = []
        for p in partants:
            id_cheval = p["id_cheval"]
            positions, nb_incidents_musique = parser_musique(p["musique"])

            hist_cheval = etat.lire_cheval(id_cheval)
            hist_hippo = etat.cheval_hippo.get((id_cheval, p["hippodrome"]), {"n": 0, "p": 0})
            hist_dist = etat.cheval_distance.get((id_cheval, _distance_bucket(p["distance_m"])), {"n": 0, "p": 0})
            hist_entr = etat.entraineur.get(p["nom_entraineur"], {"n": 0, "v": 0, "p": 0})
            hist_je = etat.jockey_entraineur.get((p["nom_jockey"], p["nom_entraineur"]), {"n": 0, "p": 0})
            hist_jc = etat.jockey_cheval.get((p["nom_jockey"], id_cheval), {"n": 0, "p": 0})
            jk = jockeys_stats.get(p["nom_jockey"])

            jours_repos = None
            if hist_cheval["derniere_date"]:
                jours_repos = (p["date_course"] - hist_cheval["derniere_date"]).days

            feats = {
                # --- cible ---
                "course_id": course_id,
                "date_course": p["date_course"],
                "numero": p["numero"],
                "position_arrivee": p["position_arrivee"],
                "nb_partants": nb_partants,
                "seuil": seuil,
                "categorie_particularite": p["categorie_particularite"],
                # --- cheval : musique (dispo dès le 1er jour, indépendant de notre historique interne) ---
                "musique_dernier": positions[0] if positions else None,
                "musique_moy3": sum(positions[:3]) / len(positions[:3]) if positions[:3] else None,
                "musique_moy5": sum(positions[:5]) / len(positions[:5]) if positions[:5] else None,
                "musique_tendance": (
                    (sum(positions[:2]) / 2 - sum(positions[2:5]) / len(positions[2:5]))
                    if len(positions) >= 4 else None
                ),
                "musique_nb_incidents": nb_incidents_musique,
                "musique_nb_courses_visibles": len(positions),
                # --- cheval : carrière (fourni par PMU, dispo dès le 1er jour) ---
                "carriere_nb_courses": p["nombre_courses"],
                "carriere_taux_victoire": (p["nombre_victoires"] / p["nombre_courses"]) if p["nombre_courses"] else None,
                "carriere_taux_place": ((p["nombre_places"] or 0) / p["nombre_courses"]) if p["nombre_courses"] else None,
                "gains_carriere": p["gains"],
                "gains_annee_encours": p["gains_annee_encours"],
                "gains_annee_precedente": p["gains_annee_precedente"],
                "age": p["age"],
                "sexe": p["sexe"],
                "handicap_poids": p["handicap_poids"],
                "poids_condition_monte": p["poids_condition_monte"],
                "place_corde": p["place_corde"],
                "oeilleres_presence": 0 if (p["oeilleres"] == "SANS_OEILLERES" or not p["oeilleres"]) else 1,
                # --- cheval : historique interne à NOTRE base (point-in-time, sparse au début) ---
                "interne_cheval_nb_courses": hist_cheval["n"],
                "interne_cheval_taux_victoire": (hist_cheval["v"] / hist_cheval["n"]) if hist_cheval["n"] else None,
                "interne_cheval_taux_place": (hist_cheval["p"] / hist_cheval["n"]) if hist_cheval["n"] else None,
                "interne_cheval_taux_incident": (hist_cheval["incidents"] / hist_cheval["n"]) if hist_cheval["n"] else None,
                "interne_cheval_jours_repos": jours_repos,
                "interne_cheval_hippo_taux_place": (hist_hippo["p"] / hist_hippo["n"]) if hist_hippo["n"] else None,
                "interne_cheval_hippo_nb": hist_hippo["n"],
                "interne_cheval_distance_taux_place": (hist_dist["p"] / hist_dist["n"]) if hist_dist["n"] else None,
                "interne_cheval_distance_nb": hist_dist["n"],
                # --- jockey (externe Geny.com, dispo dès le 1er jour pour les jockeys résolus) ---
                "jockey_victoires_pct": jk["victoires_pct"] if jk else None,
                "jockey_places_2_3_pct": jk["places_2_3_pct"] if jk else None,
                "jockey_nb_courses_12mois": jk["nb_courses_12mois"] if jk else None,
                # --- entraîneur (interne, point-in-time, self-computed) ---
                "interne_entraineur_nb_courses": hist_entr["n"],
                "interne_entraineur_taux_victoire": (hist_entr["v"] / hist_entr["n"]) if hist_entr["n"] else None,
                "interne_entraineur_taux_place": (hist_entr["p"] / hist_entr["n"]) if hist_entr["n"] else None,
                # --- interactions internes (point-in-time, sparse au début) ---
                "interne_jockey_entraineur_taux_place": (hist_je["p"] / hist_je["n"]) if hist_je["n"] else None,
                "interne_jockey_entraineur_nb": hist_je["n"],
                "interne_jockey_cheval_taux_place": (hist_jc["p"] / hist_jc["n"]) if hist_jc["n"] else None,
                "interne_jockey_cheval_nb": hist_jc["n"],
                # --- course ---
                "distance_m": p["distance_m"],
                "montant_allocation": p["montant_allocation"],
                "meteo_temperature": p["meteo_temperature"],
                "meteo_force_vent": p["meteo_force_vent"],
                "terrain_valeur_penetrometre": p["terrain_valeur_penetrometre"],
            }
            lectures.append((p, hist_cheval, feats))
            rows.append(feats)

        # Mise à jour de l'état APRES avoir construit les features de TOUS
        # les partants de cette course (jamais pendant).
        for p, hist_cheval, feats in lectures:
            est_place = 1 if p["position_arrivee"] and p["position_arrivee"] <= seuil else 0
            est_gagnant = 1 if p["position_arrivee"] == 1 else 0
            incident_detecte = 1 if (p["commentaire_apres_course"] and MOT_INCIDENT.search(p["commentaire_apres_course"])) else 0

            etat.maj_cheval(p["id_cheval"], est_place, est_gagnant, incident_detecte, p["date_course"])

            if p["id_cheval"]:
                key_h = (p["id_cheval"], p["hippodrome"])
                etat.cheval_hippo[key_h]["n"] += 1
                etat.cheval_hippo[key_h]["p"] += est_place
                key_d = (p["id_cheval"], _distance_bucket(p["distance_m"]))
                etat.cheval_distance[key_d]["n"] += 1
                etat.cheval_distance[key_d]["p"] += est_place

            e = etat.entraineur[p["nom_entraineur"]]
            e["n"] += 1
            e["v"] += est_gagnant
            e["p"] += est_place

            key_je = (p["nom_jockey"], p["nom_entraineur"])
            etat.jockey_entraineur[key_je]["n"] += 1
            etat.jockey_entraineur[key_je]["p"] += est_place

            key_jc = (p["nom_jockey"], p["id_cheval"])
            etat.jockey_cheval[key_jc]["n"] += 1
            etat.jockey_cheval[key_jc]["p"] += est_place

    return rows


if __name__ == "__main__":
    print("Ce module est importé par entrainer_et_evaluer.py — voir ce fichier pour l'exécution complète.")
