# -*- coding: utf-8 -*-
"""
variables_historiques.py — Construction d'un large ensemble de variables
point-in-time (aucune fuite du futur) pour chaque partant, à partir de
l'historique interne accumulé course après course, dans l'ordre
chronologique strict.

Règle non négociable, vérifiée par les tests unitaires de ce module :
pour un partant donné d'une course C, toutes les variables "historique
interne" ne peuvent refléter que des courses dont (date_course, heure_depart)
est STRICTEMENT antérieur à C, ou qui appartiennent à C elle-même mais dont
l'état n'est mis à jour qu'APRÈS que toutes les lectures de C ont eu lieu.

Ce module ne se connecte à aucune base : il prend en entrée une liste de
dicts déjà triée chronologiquement (voir `trier_chronologiquement`) et
retourne une liste de dicts de features, une par partant.
"""
import re
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def trier_chronologiquement(lignes):
    return sorted(
        lignes,
        key=lambda l: (l["date_course"], l.get("heure_depart") or "00:00:00", l["course_id"], l.get("numero") or 0),
    )


def _distance_bucket(d):
    if d is None:
        return None
    if d < 1600:
        return "court"
    if d < 2100:
        return "moyen"
    if d < 2800:
        return "long"
    return "tres_long"


_TERRAIN_KEYWORDS = [
    # (mot-cle, score ordinal 1=tres lourd/collant ... 9=sec, bucket texte)
    ("TRES LOURD", 1, "lourd"), ("TRï¿", 1, "lourd"),
    ("LOURD", 2, "lourd"),
    ("COLLANT", 2, "lourd"),
    ("PROFOND", 3, "souple"),
    ("TRES SOUPLE", 3, "souple"),
    ("SOUPLE", 4, "souple"),  # attrape aussi "BON SOUPLE" si teste apres BON
    ("BON SOUPLE", 4, "souple"),
    ("BON LEGER", 6, "bon"),
    ("BON", 5, "bon"),
    ("TRES LEGER", 8, "leger"),
    ("LEGER", 7, "leger"),
    ("SEC", 9, "leger"),
]


def normaliser_terrain(texte):
    """Retourne (score_ordinal 1-9 ou None, bucket ou None). Robuste aux
    valeurs vides/"Inconnu"/artefacts d'encodage (mojibake) : le score
    ordinal représente l'état du terrain de "lourd" (1) à "sec" (9)."""
    if not texte:
        return None, None
    t = texte.upper().strip()
    if t in ("", "INCONNU", "NULL"):
        return None, None
    # PSF = piste en sable fibré (tout temps), pas comparable à l'échelle gazon
    if "PSF" in t or "FIBRE" in t or "SABLE" in t:
        if "RAPIDE" in t:
            return None, "psf_rapide"
        if "LENT" in t:
            return None, "psf_lente"
        return None, "psf_standard"
    for mot, score, bucket in [("TRES LOURD", 1, "lourd"), ("LOURD", 2, "lourd"), ("COLLANT", 2, "lourd"),
                                ("PROFOND", 3, "souple"), ("TRES SOUPLE", 3, "souple"), ("BON SOUPLE", 4, "souple"),
                                ("SOUPLE", 4, "souple"), ("BON LEGER", 6, "bon"), ("BON", 5, "bon"),
                                ("TRES LEGER", 8, "leger"), ("LEGER", 7, "leger"), ("SEC", 9, "leger")]:
        if mot in t:
            return score, bucket
    return None, "autre"  # artefact d'encodage non reconnu (ex: mojibake) -> categorie "autre", pas perdu


def _corde_bucket(place_corde, nb_partants):
    if place_corde is None or not nb_partants:
        return None
    frac = (place_corde - 1) / max(nb_partants - 1, 1)
    if frac <= 0.34:
        return "interieur"
    if frac <= 0.67:
        return "milieu"
    return "exterieur"


def parser_musique(musique):
    if not musique:
        return [], 0
    sans_annees = re.sub(r"\([0-9]{2}\)", " ", musique)
    positions = []
    nb_incidents = 0
    for c in sans_annees:
        if c.isdigit():
            v = int(c)
            positions.append(99 if v == 0 else v)
        elif c in "DTA":
            positions.append(99)
            nb_incidents += 1
    return positions, nb_incidents


def _repos_bucket(jours):
    if jours is None:
        return "premiere_sortie"
    if jours <= 14:
        return "tres_rapproche"
    if jours <= 30:
        return "normal"
    if jours <= 60:
        return "espace"
    if jours <= 120:
        return "long"
    return "tres_long"


def _moy(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _taux(n_reussi, n_total):
    return (n_reussi / n_total) if n_total else None


# ---------------------------------------------------------------------------
# État cumulatif (point-in-time)
# ---------------------------------------------------------------------------

class Compteur:
    __slots__ = ("n", "v", "p2", "p3", "incidents")

    def __init__(self):
        self.n = 0
        self.v = 0
        self.p2 = 0  # top 2
        self.p3 = 0  # top 3
        self.incidents = 0

    def as_dict(self, prefix):
        return {
            f"{prefix}_nb": self.n,
            f"{prefix}_taux_victoire": _taux(self.v, self.n),
            f"{prefix}_taux_top2": _taux(self.p2, self.n),
            f"{prefix}_taux_top3": _taux(self.p3, self.n),
        }

    def maj(self, position, seuil2=2, seuil3=3):
        self.n += 1
        if position == 1:
            self.v += 1
        if position is not None and position <= seuil2:
            self.p2 += 1
        if position is not None and position <= seuil3:
            self.p3 += 1


class EtatCheval:
    __slots__ = ("positions_recentes", "derniere_date", "poids_recents", "allocations_recentes",
                 "distances_recentes", "compteur_global", "incidents")

    def __init__(self):
        self.positions_recentes = deque(maxlen=20)  # plus récente en tête (appendleft)
        self.derniere_date = None
        self.poids_recents = deque(maxlen=10)
        self.allocations_recentes = deque(maxlen=20)
        self.distances_recentes = deque(maxlen=20)
        self.compteur_global = Compteur()
        self.incidents = 0


class EtatCumulatif:
    def __init__(self):
        self.cheval = defaultdict(EtatCheval)
        self.cheval_distance = defaultdict(Compteur)   # (horse_uid, bucket_distance)
        self.cheval_terrain = defaultdict(Compteur)     # (horse_uid, bucket_terrain)
        self.cheval_hippo = defaultdict(Compteur)        # (horse_uid, hippodrome)
        self.cheval_categorie = defaultdict(Compteur)    # (horse_uid, categorie)
        self.cheval_corde = defaultdict(Compteur)        # (horse_uid, bucket_corde)
        self.jockey = defaultdict(Compteur)
        self.entraineur = defaultdict(Compteur)
        self.jockey_entraineur = defaultdict(Compteur)
        self.jockey_cheval = defaultdict(Compteur)
        self.entraineur_cheval = defaultdict(Compteur)
        self.hippo_distance_corde = defaultdict(Compteur)  # biais de corde (pas specifique a un cheval)
        self.proprietaire = defaultdict(Compteur)
        self.eleveur = defaultdict(Compteur)


# ---------------------------------------------------------------------------
# Construction des features
# ---------------------------------------------------------------------------

def construire_variables(lignes):
    """lignes : liste de dicts déjà triée chronologiquement (voir
    trier_chronologiquement), chaque dict devant contenir au moins :
      course_id, date_course, heure_depart, horse_uid, hippodrome,
      distance_m, terrain_intitule, terrain_valeur_penetrometre,
      categorie_particularite, corde, place_corde, partants_declares,
      montant_allocation, numero, nom_jockey, nom_entraineur, musique,
      gains, gains_annee_encours, gains_annee_precedente, nombre_courses,
      nombre_victoires, nombre_places, age, sexe, handicap_poids,
      poids_condition_monte, oeilleres, deferre, condition_age,
      condition_sexe, type_piste, meteo_temperature, meteo_force_vent,
      position_arrivee.

    Retourne une liste de dicts de features, un par ligne d'entrée, dans le
    même ordre que `lignes`.
    """
    etat = EtatCumulatif()
    par_course = defaultdict(list)
    ordre_courses = []
    vus = set()
    for l in lignes:
        if l["course_id"] not in vus:
            vus.add(l["course_id"])
            ordre_courses.append(l["course_id"])
        par_course[l["course_id"]].append(l)

    resultats_par_index = {}
    index_global = {id(l): i for i, l in enumerate(lignes)}

    for course_id in ordre_courses:
        partants = par_course[course_id]
        nb_partants_reel = len(partants)
        # seuil de "place" utilise la meme convention que le modele precedent
        # (comparabilite directe des resultats) : top2 si <=7 partants
        # arrivants, top3 sinon. Documente comme simplification (le vrai
        # bareme PMU depend du nombre de partants DECLARES et de la
        # categorie, avec parfois un top4 sur les grands handicaps).
        seuil = 2 if nb_partants_reel <= 7 else 3

        lignes_features_temp = []
        for p in partants:
            horse_uid = p.get("horse_uid")
            ec = etat.cheval[horse_uid] if horse_uid else EtatCheval()

            positions, nb_incidents_musique = parser_musique(p.get("musique"))

            jours_repos = None
            if ec.derniere_date is not None:
                jours_repos = (p["date_course"] - ec.derniere_date).days

            terrain_score, terrain_bucket = normaliser_terrain(p.get("terrain_intitule"))
            dist_bucket = _distance_bucket(p.get("distance_m"))
            corde_bucket_relatif = _corde_bucket(p.get("place_corde"), p.get("partants_declares") or nb_partants_reel)

            hist_dist = etat.cheval_distance.get((horse_uid, dist_bucket), Compteur())
            hist_terrain = etat.cheval_terrain.get((horse_uid, terrain_bucket), Compteur())
            hist_hippo = etat.cheval_hippo.get((horse_uid, p.get("hippodrome")), Compteur())
            hist_cat = etat.cheval_categorie.get((horse_uid, p.get("categorie_particularite")), Compteur())
            hist_corde = etat.cheval_corde.get((horse_uid, corde_bucket_relatif), Compteur())
            hist_jockey = etat.jockey.get(p.get("nom_jockey"), Compteur())
            hist_entraineur = etat.entraineur.get(p.get("nom_entraineur"), Compteur())
            hist_je = etat.jockey_entraineur.get((p.get("nom_jockey"), p.get("nom_entraineur")), Compteur())
            hist_jc = etat.jockey_cheval.get((p.get("nom_jockey"), horse_uid), Compteur())
            hist_ec = etat.entraineur_cheval.get((p.get("nom_entraineur"), horse_uid), Compteur())
            hist_biais_corde = etat.hippo_distance_corde.get(
                (p.get("hippodrome"), dist_bucket, corde_bucket_relatif), Compteur())
            hist_proprietaire = etat.proprietaire.get(p.get("proprietaire"), Compteur())
            hist_eleveur = etat.eleveur.get(p.get("eleveur"), Compteur())

            positions_5 = list(ec.positions_recentes)[:5]
            positions_10 = list(ec.positions_recentes)[:10]
            positions_20 = list(ec.positions_recentes)[:20]

            poids_moyen_carriere = _moy(ec.poids_recents)
            allocation_moyenne_carriere = _moy(ec.allocations_recentes)
            distance_moyenne_carriere = _moy(ec.distances_recentes)

            feats = {
                "course_id": course_id,
                "date_course": p["date_course"],
                "numero": p.get("numero"),
                "horse_uid": horse_uid,
                "position_arrivee": p.get("position_arrivee"),
                "nb_partants_reel": nb_partants_reel,
                "partants_declares": p.get("partants_declares") or nb_partants_reel,
                "seuil": seuil,
                "hippodrome": p.get("hippodrome"),
                "categorie_particularite": p.get("categorie_particularite"),
                "condition_age": p.get("condition_age"),
                "condition_sexe": p.get("condition_sexe"),
                "type_piste": p.get("type_piste"),

                # --- musique (independante de notre historique interne) ---
                "musique_dernier": positions[0] if positions else None,
                "musique_moy3": _moy(positions[:3]),
                "musique_moy5": _moy(positions[:5]),
                "musique_tendance": (
                    (_moy(positions[:2]) - _moy(positions[2:5])) if len(positions) >= 4 else None
                ),
                "musique_nb_incidents": nb_incidents_musique,
                "musique_nb_courses_visibles": len(positions),

                # --- carriere globale fournie par PMU (dispo des le 1er jour) ---
                "carriere_nb_courses": p.get("nombre_courses"),
                "carriere_taux_victoire": _taux(p.get("nombre_victoires"), p.get("nombre_courses")),
                "carriere_taux_place": _taux(p.get("nombre_places"), p.get("nombre_courses")),
                "gains_carriere": p.get("gains"),
                "gains_annee_encours": p.get("gains_annee_encours"),
                "gains_annee_precedente": p.get("gains_annee_precedente"),

                # --- signalement / poids / equipement ---
                "age": p.get("age"),
                "sexe": p.get("sexe"),
                "handicap_poids": p.get("handicap_poids"),
                "poids_condition_monte": p.get("poids_condition_monte"),
                "poids_delta_vs_carriere": (
                    (p["handicap_poids"] - poids_moyen_carriere)
                    if p.get("handicap_poids") is not None and poids_moyen_carriere is not None else None
                ),
                "oeilleres_presence": 0 if (p.get("oeilleres") in (None, "SANS_OEILLERES")) else 1,
                "deferre_present": 0 if (p.get("deferre") in (None, "", "NON DEFERRE")) else 1,
                "place_corde": p.get("place_corde"),
                "corde_bucket_relatif": corde_bucket_relatif,

                # --- course (contexte, connu avant le depart) ---
                "distance_m": p.get("distance_m"),
                "distance_bucket": dist_bucket,
                "distance_delta_vs_carriere": (
                    (p["distance_m"] - distance_moyenne_carriere)
                    if p.get("distance_m") is not None and distance_moyenne_carriere is not None else None
                ),
                "montant_allocation": p.get("montant_allocation"),
                "allocation_delta_vs_carriere": (
                    (p["montant_allocation"] - allocation_moyenne_carriere)
                    if p.get("montant_allocation") is not None and allocation_moyenne_carriere is not None else None
                ),
                "meteo_temperature": p.get("meteo_temperature"),
                "meteo_force_vent": p.get("meteo_force_vent"),
                "terrain_valeur_penetrometre": p.get("terrain_valeur_penetrometre"),
                "terrain_score_ordinal": terrain_score,
                "terrain_bucket": terrain_bucket,

                # --- repos ---
                "jours_repos": jours_repos,
                "repos_bucket": _repos_bucket(jours_repos),

                # --- forme recente interne (fenetres 5/10/20, point-in-time) ---
                "forme_n_courses_internes": ec.compteur_global.n,
                "forme_taux_victoire_carriere_interne": _taux(ec.compteur_global.v, ec.compteur_global.n),
                "forme_taux_place_carriere_interne": _taux(ec.compteur_global.p3, ec.compteur_global.n),
                "forme_moy_position_5": _moy(positions_5),
                "forme_moy_position_10": _moy(positions_10),
                "forme_moy_position_20": _moy(positions_20),
                "forme_meilleure_position_5": min(positions_5) if positions_5 else None,
                "forme_ecart_type_position_5": (
                    (sum((x - _moy(positions_5)) ** 2 for x in positions_5) / len(positions_5)) ** 0.5
                    if len(positions_5) >= 2 else None
                ),
                "forme_tendance_5_vs_10": (
                    (_moy(positions_5) - _moy(positions_10[5:10]))
                    if len(positions_10) >= 8 else None
                ),
                "forme_nb_courses_disponibles": len(ec.positions_recentes),
                "forme_a_au_moins_5": 1 if len(ec.positions_recentes) >= 5 else 0,
                "forme_a_au_moins_10": 1 if len(ec.positions_recentes) >= 10 else 0,
                "forme_a_au_moins_20": 1 if len(ec.positions_recentes) >= 20 else 0,

                # --- historique interne par distance / terrain / hippodrome / categorie / corde ---
                **hist_dist.as_dict("interne_distance"),
                **hist_terrain.as_dict("interne_terrain"),
                **hist_hippo.as_dict("interne_hippo"),
                **hist_cat.as_dict("interne_categorie"),
                **hist_corde.as_dict("interne_corde_relative"),

                # --- jockey / entraineur (internes, point-in-time) ---
                **hist_jockey.as_dict("interne_jockey"),
                **hist_entraineur.as_dict("interne_entraineur"),

                # --- interactions ---
                **hist_je.as_dict("interne_jockey_entraineur"),
                **hist_jc.as_dict("interne_jockey_cheval"),
                **hist_ec.as_dict("interne_entraineur_cheval"),

                # --- biais de corde a cet hippodrome/distance (pas specifique au cheval) ---
                **hist_biais_corde.as_dict("biais_corde_hippo_distance"),

                # --- proprietaire / eleveur (internes, point-in-time — proxy
                # d'un "niveau d'ecurie" sans exploser la cardinalite via un
                # one-hot direct, cf. traitement jockey/entraineur) ---
                **hist_proprietaire.as_dict("interne_proprietaire"),
                **hist_eleveur.as_dict("interne_eleveur"),

                # --- autres attributs pre-course disponibles dans la base ---
                "race_cheval": p.get("race"),
                "pays_cheval": p.get("pays"),
                "entraine_a_letranger": (
                    1 if (p.get("pays_entrainement") and p.get("pays_entrainement").upper() not in ("FRANCE", "FR"))
                    else (0 if p.get("pays_entrainement") else None)
                ),
                "robe": p.get("robe"),
            }
            lignes_features_temp.append((p, feats))

        # --- niveau des adversaires : calcule APRES avoir construit les
        # features individuelles de TOUS les partants de cette course, a
        # partir de leurs stats historiques deja connues avant la course
        # (aucune fuite : on n'utilise pas le resultat du jour). ---
        taux_victoire_champ = [
            f["carriere_taux_victoire"] for _, f in lignes_features_temp if f["carriere_taux_victoire"] is not None
        ]
        moy_champ = _moy(taux_victoire_champ)
        for p, f in lignes_features_temp:
            tv = f["carriere_taux_victoire"]
            f["niveau_moyen_adversaires"] = (
                _moy([x for x in taux_victoire_champ if x != tv]) if len(taux_victoire_champ) > 1 else moy_champ
            )
            f["ecart_vs_niveau_moyen_champ"] = (tv - moy_champ) if (tv is not None and moy_champ is not None) else None
            # rang "sur le papier" (1 = meilleur taux de victoire carriere du champ)
            if tv is not None:
                f["rang_papier_taux_victoire"] = 1 + sum(1 for x in taux_victoire_champ if x > tv)
            else:
                f["rang_papier_taux_victoire"] = None

        for p, f in lignes_features_temp:
            resultats_par_index[index_global[id(p)]] = f

        # --- mise a jour de l'etat, APRES avoir lu/construit toutes les
        # features de TOUS les partants de cette course (jamais pendant). ---
        for p, f in lignes_features_temp:
            horse_uid = p.get("horse_uid")
            position = p.get("position_arrivee")
            est_place3 = position is not None and position <= 3
            est_place2 = position is not None and position <= 2

            if horse_uid:
                ec = etat.cheval[horse_uid]
                ec.compteur_global.maj(position)
                ec.positions_recentes.appendleft(position if position is not None else 99)
                ec.derniere_date = p["date_course"]
                if p.get("handicap_poids") is not None:
                    ec.poids_recents.append(p["handicap_poids"])
                if p.get("montant_allocation") is not None:
                    ec.allocations_recentes.append(p["montant_allocation"])
                if p.get("distance_m") is not None:
                    ec.distances_recentes.append(p["distance_m"])

                dist_bucket = f["distance_bucket"]
                etat.cheval_distance[(horse_uid, dist_bucket)].maj(position)
                terrain_bucket = f["terrain_bucket"]
                etat.cheval_terrain[(horse_uid, terrain_bucket)].maj(position)
                etat.cheval_hippo[(horse_uid, p.get("hippodrome"))].maj(position)
                etat.cheval_categorie[(horse_uid, p.get("categorie_particularite"))].maj(position)
                etat.cheval_corde[(horse_uid, f["corde_bucket_relatif"])].maj(position)

            etat.jockey[p.get("nom_jockey")].maj(position)
            etat.entraineur[p.get("nom_entraineur")].maj(position)
            etat.jockey_entraineur[(p.get("nom_jockey"), p.get("nom_entraineur"))].maj(position)
            etat.jockey_cheval[(p.get("nom_jockey"), horse_uid)].maj(position)
            etat.entraineur_cheval[(p.get("nom_entraineur"), horse_uid)].maj(position)
            etat.hippo_distance_corde[(p.get("hippodrome"), f["distance_bucket"], f["corde_bucket_relatif"])].maj(position)
            etat.proprietaire[p.get("proprietaire")].maj(position)
            etat.eleveur[p.get("eleveur")].maj(position)

    return [resultats_par_index[i] for i in range(len(lignes))]
