# -*- coding: utf-8 -*-
"""
variables_equipement.py -- Construction de variables de changement
d'equipement point-in-time (piste 3, demandee par Dorian le 28/08/2026
apres l'abandon de la piste 2 -- vitesse chronometrique : temps_obtenu et
reduction_kilometrique sont 100% NULL sur les 978 471 lignes de
resultats_partants, aucune autre table ne contient de chrono).

Avant construction, audit Supabase du 28/08/2026 sur les DEUX candidats
d'equipement demandes par Dorian :
  - oeilleres (resultats_partants)  : 978 471/978 471 renseigne (100%),
    3 modalites propres : SANS_OEILLERES (75,4%), OEILLERES_CLASSIQUE
    (14,2%), OEILLERES_AUSTRALIENNES (10,4%) -- exploitable.
  - deferre (resultats_partants)    : 99/978 471 renseigne (0,01%), une
    seule modalite jamais observee (DEFERRE_POSTERIEURS) -- collecte
    quasiment inexistante, pas un phenomene hippique rare (le deferrage
    concerne reellement plusieurs % des partants en France). Verifie
    egalement dans `partants` (table "en direct") : 0 ligne au total (le
    meme bug de collecte deja diagnostique le 28/08/2026 sur les cotes --
    contrainte de cle manquante cassant les ON CONFLICT -- non corrige ce
    tour-ci, comme demande). Aucune autre table de la base ne contient de
    champ de deferrage exploitable.
  => DECISION : l'axe deferrage est abandonne DEFINITIVEMENT pour cette
     piste. Seul l'axe oeilleres est construit ci-dessous.

Meme discipline stricte que variables_historiques.py et
variables_genealogie.py : ce module ne se connecte a aucune base, prend en
entree une liste de dicts DEJA TRIEE chronologiquement et retourne une
liste de dicts de features, une par ligne, dans le meme ordre. Pour un
partant d'une course C, tout ce qui decrit un changement ou un historique
de performance par configuration ne reflete que des courses strictement
anterieures a C -- l'etat n'est mis a jour qu'APRES avoir lu tous les
partants de C (jamais pendant). La seule information de C elle-meme
utilisee est la configuration d'oeilleres DECLAREE pour la course du jour
(rp.oeilleres) : c'est une donnee connue avant le depart (comme
oeilleres_presence deja utilise dans les variables v3), donc aucune fuite.

Choix de conception (mecanisme unique, pas de 50 features arbitraires) :
tout est derive de DEUX etats par cheval, mis a jour course apres course :
  (1) la derniere configuration connue (pour detecter tout type de
      transition : pose, retrait, retour, classique<->australienne) et le
      nombre de courses consecutives sous la configuration en cours
      (stabilite recente) ;
  (2) un compteur victoire/place PAR configuration (SANS / CLASSIQUE /
      AUSTRALIENNE), lisse par shrinkage bayesien vers le taux global du
      cheval lui-meme (hierarchie a un niveau : la performance
      "sous cette config" converge vers la performance "generale du
      cheval" quand le volume sous cette config precise est faible --
      memes principes que le shrinkage pere/mere de variables_genealogie.py).
Aucun seuil arbitraire de volume minimal n'est code en dur dans les
features (le shrinkage l'absorbe deja) -- en revanche les compteurs de
volume (equip_n_courses_meme_config, equip_n_courses_depuis_dernier_
changement) sont exposes explicitement pour que l'analyse en phase 2
puisse verifier qu'un effet fort ne repose pas sur un sous-echantillon
minuscule, comme demande.
"""
from collections import defaultdict

SANS = "SANS_OEILLERES"
CLASSIQUE = "OEILLERES_CLASSIQUE"
AUSTRALIENNE = "OEILLERES_AUSTRALIENNES"
_CONFIGS_CONNUES = (SANS, CLASSIQUE, AUSTRALIENNE)

# Force de lissage (pseudo-observations) pour les taux de performance par
# configuration. Choisie a partir de la distribution reelle du nombre de
# courses par cheval (audit Supabase du 28/08/2026, vue id_cheval seul --
# sous-estime la vraie carriere resolue par horse_uid : mediane=2, p90=8,
# max=32). Avec un partage supplementaire par configuration d'oeilleres, le
# volume par (cheval, config) est structurellement encore plus faible ->
# lissage relativement fort necessaire. Jamais ajustee sur validation.
K_CONFIG = 8
PRIOR_VICTOIRE_DEMARRAGE = 0.10
PRIOR_PLACE_DEMARRAGE = 0.30


def _taux(n_reussi, n_total):
    return (n_reussi / n_total) if n_total else None


def _taux_ou_defaut(n_reussi, n_total, defaut):
    t = _taux(n_reussi, n_total)
    return t if t is not None else defaut


def _shrink(n, taux_observe, prior, k):
    if not n or taux_observe is None:
        return prior
    return (n * taux_observe + k * prior) / (n + k)


class _CompteurSimple:
    __slots__ = ("n", "v", "p3")

    def __init__(self):
        self.n = 0
        self.v = 0
        self.p3 = 0

    def maj(self, position, seuil3=3):
        self.n += 1
        if position == 1:
            self.v += 1
        if position is not None and position <= seuil3:
            self.p3 += 1


class _EtatCheval:
    __slots__ = ("dernier_config", "a_deja_porte_oeilleres", "n_depuis_changement",
                 "global_", "par_config")

    def __init__(self):
        self.dernier_config = None
        self.a_deja_porte_oeilleres = False
        self.n_depuis_changement = 0
        self.global_ = _CompteurSimple()
        self.par_config = defaultdict(_CompteurSimple)


def _normaliser_config(valeur):
    """SANS/CLASSIQUE/AUSTRALIENNE si reconnue, sinon None (valeur absente
    ou inattendue -- traitee comme "inconnue", jamais assimilee a SANS,
    pour ne pas fabriquer un faux "retrait" ou une fausse "premiere pose")."""
    return valeur if valeur in _CONFIGS_CONNUES else None


def construire_variables_equipement(lignes_triees):
    """lignes_triees : meme liste de dicts (deja triee chronologiquement)
    que celle passee a variables_historiques.construire_variables, chaque
    dict devant contenir au moins : course_id, horse_uid, oeilleres,
    position_arrivee.

    Retourne une liste de dicts de features, une par ligne d'entree, dans
    le meme ordre -- a concatener (par position) aux features de
    variables_historiques.construire_variables."""
    etat_par_cheval = defaultdict(_EtatCheval)
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
            horse_uid = p.get("horse_uid")
            config_actuel = _normaliser_config(p.get("oeilleres"))
            ec = etat_par_cheval[horse_uid] if horse_uid else _EtatCheval()

            dernier = ec.dernier_config
            a_historique = dernier is not None

            if a_historique and config_actuel is not None:
                changement = int(config_actuel != dernier)
                premiere_pose = int(config_actuel != SANS and dernier == SANS and not ec.a_deja_porte_oeilleres)
                retour_absence = int(config_actuel != SANS and dernier == SANS and ec.a_deja_porte_oeilleres)
                retrait = int(config_actuel == SANS and dernier != SANS)
                classique_vers_australienne = int(dernier == CLASSIQUE and config_actuel == AUSTRALIENNE)
                australienne_vers_classique = int(dernier == AUSTRALIENNE and config_actuel == CLASSIQUE)
                n_depuis_changement = ec.n_depuis_changement
            else:
                changement = premiere_pose = retour_absence = retrait = None
                classique_vers_australienne = australienne_vers_classique = None
                n_depuis_changement = None

            prior_v = _taux_ou_defaut(ec.global_.v, ec.global_.n, PRIOR_VICTOIRE_DEMARRAGE)
            prior_p = _taux_ou_defaut(ec.global_.p3, ec.global_.n, PRIOR_PLACE_DEMARRAGE)

            if config_actuel is not None:
                hc = ec.par_config.get(config_actuel, _CompteurSimple())
                n_meme_config = hc.n
                taux_v_meme_config = _shrink(hc.n, _taux(hc.v, hc.n), prior_v, K_CONFIG)
                taux_p_meme_config = _shrink(hc.n, _taux(hc.p3, hc.n), prior_p, K_CONFIG)
                delta_v_vs_global = (taux_v_meme_config - prior_v) if ec.global_.n else None
                delta_p_vs_global = (taux_p_meme_config - prior_p) if ec.global_.n else None
            else:
                n_meme_config = None
                taux_v_meme_config = taux_p_meme_config = None
                delta_v_vs_global = delta_p_vs_global = None

            feats = {
                "equip_changement_generique": changement,
                "equip_premiere_pose_oeilleres": premiere_pose,
                "equip_retour_apres_absence": retour_absence,
                "equip_retrait_oeilleres": retrait,
                "equip_classique_vers_australienne": classique_vers_australienne,
                "equip_australienne_vers_classique": australienne_vers_classique,
                "equip_n_courses_depuis_dernier_changement": n_depuis_changement,
                "equip_n_courses_meme_config": n_meme_config,
                "equip_taux_victoire_meme_config_shrunk": taux_v_meme_config,
                "equip_taux_place_meme_config_shrunk": taux_p_meme_config,
                "equip_delta_taux_victoire_config_vs_global": delta_v_vs_global,
                "equip_delta_taux_place_config_vs_global": delta_p_vs_global,
            }
            lignes_features_temp.append((p, config_actuel, feats))

        for p, _, f in lignes_features_temp:
            resultats_par_index[index_global[id(p)]] = f

        # --- mise a jour de l'etat, APRES avoir lu tous les partants de
        # cette course (jamais pendant) -- meme discipline que
        # variables_historiques.construire_variables / variables_genealogie. ---
        for p, config_actuel, _ in lignes_features_temp:
            horse_uid = p.get("horse_uid")
            if not horse_uid:
                continue
            ec = etat_par_cheval[horse_uid]
            position = p.get("position_arrivee")
            if config_actuel is not None:
                if ec.dernier_config is not None and config_actuel != ec.dernier_config:
                    ec.n_depuis_changement = 1  # cette course demarre une nouvelle serie
                else:
                    ec.n_depuis_changement += 1  # continue la serie (ou la demarre a 1 si 1ere donnee connue)
                if config_actuel != SANS:
                    ec.a_deja_porte_oeilleres = True
                ec.par_config[config_actuel].maj(position)
                ec.dernier_config = config_actuel
            ec.global_.maj(position)

    return [resultats_par_index[i] for i in range(len(lignes_triees))]


# Liste explicite des colonnes produites (pour entrainer_v3_phase1_equipement.py).
COLONNES_EQUIPEMENT = [
    "equip_changement_generique",
    "equip_premiere_pose_oeilleres",
    "equip_retour_apres_absence",
    "equip_retrait_oeilleres",
    "equip_classique_vers_australienne",
    "equip_australienne_vers_classique",
    "equip_n_courses_depuis_dernier_changement",
    "equip_n_courses_meme_config",
    "equip_taux_victoire_meme_config_shrunk",
    "equip_taux_place_meme_config_shrunk",
    "equip_delta_taux_victoire_config_vs_global",
    "equip_delta_taux_place_config_vs_global",
]
