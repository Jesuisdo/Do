# -*- coding: utf-8 -*-
"""
variables_genealogie.py -- Construction de variables de genealogie
point-in-time (piste 3, demandee par Dorian le 28/08/2026 apres avoir
retenu B/lambdarank_graded comme meilleur candidat de la piste 1 et
constate que 71,6% des courses restent "toujours ratees" quelle que soit
la formulation de l'objectif de ranking -- indice que le probleme n'est
plus la fonction de perte mais l'information disponible).

Meme discipline stricte que variables_historiques.py : ce module ne se
connecte a aucune base, prend en entree une liste de dicts DEJA TRIEE
chronologiquement (voir variables_historiques.trier_chronologiquement) et
retourne une liste de dicts de features, une par ligne, dans le meme
ordre. Pour un partant d'une course C, toutes les statistiques de
genealogie ne refletent que des courses dont (date_course, heure_depart)
est strictement anterieur a C -- l'etat n'est mis a jour qu'APRES avoir lu
TOUS les partants de C (jamais pendant), exactement comme
variables_historiques.construire_variables.

Principe (demande explicite de Dorian, 28/08/2026) : ne PAS se contenter
d'encoder nom_pere/nom_mere comme categories (deja tente implicitement par
la cardinalite capee de preparer_matrice, sans grand interet). A la place,
construire des statistiques d'aptitude a la course -- performance
historique du pere, de la mere, aptitude distance, aptitude terrain,
interactions pere x distance / pere x terrain -- avec :
  (a) le nombre d'observations utilise pour chaque statistique (volume) ;
  (b) un lissage bayesien (shrinkage) vers une moyenne globale, POUR
      NE JAMAIS laisser un pere/mere a faible volume produire un taux
      brut bruite (ex. 1 seul poulain gagnant sur 1 course -> taux brut
      100%, absurde) ;
  (c) une structure HIERARCHIQUE pour les interactions : pere x distance
      et pere x terrain ne sont pas lisses directement vers la moyenne
      globale, mais vers la statistique MARGINALE deja lissee du pere
      (idem pour la mere). Ceci isole un signal "ce pere est-il
      specifiquement meilleur sur cette distance/ce terrain que sa
      moyenne generale" plutot que de re-capter la simple qualite
      generale du pere (deja dans la statistique marginale).

Consequence directe de (b)+(c) : aucun seuil arbitraire de volume n'est
necessaire pour decider d'inclure ou d'exclure une interaction (ex.
mere x distance, tres majoritairement a faible volume -- mediane de 4
courses par mere sur l'historique complet, cf. audit Supabase du
28/08/2026). Une interaction a volume quasi nul converge simplement,
mecaniquement, vers sa statistique parente (deja fiable) -- exactement le
comportement souhaite ("ne conserve les interactions que si le volume
statistique est suffisant"), sans regle de decision separee qui
introduirait elle-meme un choix arbitraire.
"""
from collections import defaultdict

from variables_historiques import Compteur, _distance_bucket, normaliser_terrain

# --- forces de lissage (pseudo-observations), choisies A PRIORI a partir
# de la distribution reelle du nombre de courses par pere/mere (audit
# Supabase du 28/08/2026 : pere median=11, p90=307, max=8550, 6905
# distincts ; mere mediane=4, p90=26, max=432, 92641 distinctes) --
# JAMAIS ajustees sur validation (pas de fuite de choix d'hyperparametre).
K_MARGINAL = 20
K_INTERACTION = 15

# Constantes de demarrage (utilisees UNIQUEMENT tant qu'aucune course
# n'a encore ete vue, donc que les compteurs globaux point-in-time sont
# encore a zero -- impact negligeable, quelques lignes au tout debut de
# l'historique 2014 sur plusieurs centaines de milliers de lignes).
PRIOR_VICTOIRE_DEMARRAGE = 0.10
PRIOR_PLACE_DEMARRAGE = 0.30


def _shrink(n, taux_observe, prior, k):
    """Lissage bayesien standard : (n*taux + k*prior) / (n+k). A n=0 (ou
    taux_observe indisponible), retourne integralement le prior -- aucune
    information -> aucune deviation par rapport au prior."""
    if not n or taux_observe is None:
        return prior
    return (n * taux_observe + k * prior) / (n + k)


class _EtatGenealogie:
    __slots__ = ("global_", "pere", "mere", "pere_distance", "pere_terrain",
                 "mere_distance", "mere_terrain")

    def __init__(self):
        self.global_ = Compteur()
        self.pere = defaultdict(Compteur)
        self.mere = defaultdict(Compteur)
        self.pere_distance = defaultdict(Compteur)  # (nom_pere, distance_bucket)
        self.pere_terrain = defaultdict(Compteur)   # (nom_pere, terrain_bucket)
        self.mere_distance = defaultdict(Compteur)  # (nom_mere, distance_bucket)
        self.mere_terrain = defaultdict(Compteur)   # (nom_mere, terrain_bucket)


def construire_variables_genealogie(lignes_triees):
    """lignes_triees : meme liste de dicts (deja triee chronologiquement)
    que celle passee a variables_historiques.construire_variables, chaque
    dict devant contenir au moins : course_id, date_course, heure_depart,
    nom_pere, nom_mere, distance_m, terrain_intitule, position_arrivee.

    Retourne une liste de dicts de features de genealogie, une par ligne
    d'entree, dans le meme ordre -- a concatener (par position, pas par
    cle) aux features de variables_historiques.construire_variables."""
    etat = _EtatGenealogie()
    par_course = defaultdict(list)
    ordre_courses = []
    vus = set()
    for l in lignes_triees:
        if l["course_id"] not in vus:
            vus.add(l["course_id"])
            ordre_courses.append(l["course_id"])
        par_course[l["course_id"]].append(l)

    index_global = {id(l): i for i, l in enumerate(lignes_triees)}
    resultats_par_index = {}

    for course_id in ordre_courses:
        partants = par_course[course_id]
        lignes_features_temp = []

        for p in partants:
            nom_pere = p.get("nom_pere")
            nom_mere = p.get("nom_mere")
            dist_bucket = _distance_bucket(p.get("distance_m"))
            _, terrain_bucket = normaliser_terrain(p.get("terrain_intitule"))

            prior_v = _taux_ou_defaut(etat.global_.v, etat.global_.n, PRIOR_VICTOIRE_DEMARRAGE)
            prior_p = _taux_ou_defaut(etat.global_.p3, etat.global_.n, PRIOR_PLACE_DEMARRAGE)

            hp = etat.pere.get(nom_pere, Compteur())
            hm = etat.mere.get(nom_mere, Compteur())

            pere_v_shrunk = _shrink(hp.n, _taux(hp.v, hp.n), prior_v, K_MARGINAL)
            pere_p_shrunk = _shrink(hp.n, _taux(hp.p3, hp.n), prior_p, K_MARGINAL)
            mere_v_shrunk = _shrink(hm.n, _taux(hm.v, hm.n), prior_v, K_MARGINAL)
            mere_p_shrunk = _shrink(hm.n, _taux(hm.p3, hm.n), prior_p, K_MARGINAL)

            hpd = etat.pere_distance.get((nom_pere, dist_bucket), Compteur())
            hpt = etat.pere_terrain.get((nom_pere, terrain_bucket), Compteur())
            hmd = etat.mere_distance.get((nom_mere, dist_bucket), Compteur())
            hmt = etat.mere_terrain.get((nom_mere, terrain_bucket), Compteur())

            w_pere = hp.n / (hp.n + K_MARGINAL) if (hp.n + K_MARGINAL) else 0.0
            w_mere = hm.n / (hm.n + K_MARGINAL) if (hm.n + K_MARGINAL) else 0.0
            if (w_pere + w_mere) > 0:
                lignee_v = (w_pere * pere_v_shrunk + w_mere * mere_v_shrunk) / (w_pere + w_mere)
                lignee_p = (w_pere * pere_p_shrunk + w_mere * mere_p_shrunk) / (w_pere + w_mere)
            else:
                lignee_v, lignee_p = prior_v, prior_p

            feats = {
                "genea_pere_n": hp.n,
                "genea_pere_taux_victoire_shrunk": pere_v_shrunk,
                "genea_pere_taux_place_shrunk": pere_p_shrunk,
                "genea_mere_n": hm.n,
                "genea_mere_taux_victoire_shrunk": mere_v_shrunk,
                "genea_mere_taux_place_shrunk": mere_p_shrunk,
                "genea_pere_distance_n": hpd.n,
                "genea_pere_distance_taux_victoire_shrunk": _shrink(hpd.n, _taux(hpd.v, hpd.n), pere_v_shrunk, K_INTERACTION),
                "genea_pere_distance_taux_place_shrunk": _shrink(hpd.n, _taux(hpd.p3, hpd.n), pere_p_shrunk, K_INTERACTION),
                "genea_pere_terrain_n": hpt.n,
                "genea_pere_terrain_taux_victoire_shrunk": _shrink(hpt.n, _taux(hpt.v, hpt.n), pere_v_shrunk, K_INTERACTION),
                "genea_pere_terrain_taux_place_shrunk": _shrink(hpt.n, _taux(hpt.p3, hpt.n), pere_p_shrunk, K_INTERACTION),
                "genea_mere_distance_n": hmd.n,
                "genea_mere_distance_taux_victoire_shrunk": _shrink(hmd.n, _taux(hmd.v, hmd.n), mere_v_shrunk, K_INTERACTION),
                "genea_mere_distance_taux_place_shrunk": _shrink(hmd.n, _taux(hmd.p3, hmd.n), mere_p_shrunk, K_INTERACTION),
                "genea_mere_terrain_n": hmt.n,
                "genea_mere_terrain_taux_victoire_shrunk": _shrink(hmt.n, _taux(hmt.v, hmt.n), mere_v_shrunk, K_INTERACTION),
                "genea_mere_terrain_taux_place_shrunk": _shrink(hmt.n, _taux(hmt.p3, hmt.n), mere_p_shrunk, K_INTERACTION),
                "genea_lignee_combinee_n": hp.n + hm.n,
                "genea_lignee_combinee_taux_victoire_shrunk": lignee_v,
                "genea_lignee_combinee_taux_place_shrunk": lignee_p,
            }
            lignes_features_temp.append((p, dist_bucket, terrain_bucket, feats))

        for p, _, _, f in lignes_features_temp:
            resultats_par_index[index_global[id(p)]] = f

        # --- mise a jour de l'etat, APRES avoir lu tous les partants de
        # cette course (jamais pendant) -- meme discipline que
        # variables_historiques.construire_variables. ---
        for p, dist_bucket, terrain_bucket, _ in lignes_features_temp:
            position = p.get("position_arrivee")
            nom_pere = p.get("nom_pere")
            nom_mere = p.get("nom_mere")
            etat.global_.maj(position)
            etat.pere[nom_pere].maj(position)
            etat.mere[nom_mere].maj(position)
            etat.pere_distance[(nom_pere, dist_bucket)].maj(position)
            etat.pere_terrain[(nom_pere, terrain_bucket)].maj(position)
            etat.mere_distance[(nom_mere, dist_bucket)].maj(position)
            etat.mere_terrain[(nom_mere, terrain_bucket)].maj(position)

    return [resultats_par_index[i] for i in range(len(lignes_triees))]


def _taux(n_reussi, n_total):
    return (n_reussi / n_total) if n_total else None


def _taux_ou_defaut(n_reussi, n_total, defaut):
    t = _taux(n_reussi, n_total)
    return t if t is not None else defaut


# Liste explicite des colonnes produites (pour variables_config-like
# reference et pour la matrice X_..._geneal cote entrainer_v3_phase1_*).
COLONNES_GENEALOGIE = [
    "genea_pere_n", "genea_pere_taux_victoire_shrunk", "genea_pere_taux_place_shrunk",
    "genea_mere_n", "genea_mere_taux_victoire_shrunk", "genea_mere_taux_place_shrunk",
    "genea_pere_distance_n", "genea_pere_distance_taux_victoire_shrunk", "genea_pere_distance_taux_place_shrunk",
    "genea_pere_terrain_n", "genea_pere_terrain_taux_victoire_shrunk", "genea_pere_terrain_taux_place_shrunk",
    "genea_mere_distance_n", "genea_mere_distance_taux_victoire_shrunk", "genea_mere_distance_taux_place_shrunk",
    "genea_mere_terrain_n", "genea_mere_terrain_taux_victoire_shrunk", "genea_mere_terrain_taux_place_shrunk",
    "genea_lignee_combinee_n", "genea_lignee_combinee_taux_victoire_shrunk", "genea_lignee_combinee_taux_place_shrunk",
]
