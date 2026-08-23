# -*- coding: utf-8 -*-
"""
identite_chevaux.py — Résolution d'identité robuste des chevaux, sans
supposer que le nom seul est un identifiant fiable.

Contexte (mesuré sur la base PLAT au 23/08/2026) :
  - id_cheval (identifiant stable fourni par PMU) : 0% de couverture
    2014-2024, 48.5% en 2025, 100% en 2026. Inutilisable seul comme clé
    d'identité sur la majorité de l'historique.
  - nom_pere / nom_mere : ~99.99% de couverture sur toute la période.

Stratégie retenue :
  1. Quand id_cheval est présent ET cohérent à travers toutes les lignes
     d'un même (nom, père, mère), on lui fait confiance en priorité absolue
     (c'est la source la plus fiable).
  2. Sinon, on regroupe les lignes par nom normalisé, puis on les scinde en
     "clusters" (chevaux distincts) au sein d'un même nom si l'un des
     signaux suivants CONTREDIT une identité commune :
       - père renseigné et différent sur les deux lignes,
       - mère renseignée et différente sur les deux lignes,
       - sexe incompatible (FEMELLE vs MALE/HONGRE — un hongre est un mâle
         castré, donc une transition MALE -> HONGRE est normale et n'est
         PAS un conflit ; FEMELLE ne peut jamais devenir MALE/HONGRE ni
         l'inverse),
       - trajectoire d'âge impossible (année de naissance implicite
         incompatible de plus d'un an entre les deux lignes).
  3. Quand aucun signal ne contredit une identité commune, on fusionne (une
     ligne sans père/mère renseigné, ou un nom porté par un seul cheval
     dans nos données, reste un seul cluster).
  4. Chaque cluster obtient un horse_uid stable. Les cas où un même nom se
     scinde en plusieurs clusters (collision de nom) sont comptés et un
     échantillon est conservé pour audit humain.

Limites documentées (pas cachées) :
  - Ceci reste une heuristique, pas une vérité terrain. Une paire de vrais
    jumeaux issus des mêmes parents, courant sous des noms différents,
    n'est bien sûr pas concernée (ce sont des lignes de noms différents).
    Le cas non couvert par construction : deux chevaux distincts partageant
    EXACTEMENT le même nom ET les mêmes père/mère (frères/sœurs complets
    homonymes) ne seraient pas séparés par les règres 1-2 seules ; la
    vérification de trajectoire d'âge (né la même année vs pas) est le
    seul filet de sécurité dans ce cas précis, et il est imparfait.
  - L'année de naissance est estimée par (année_course - âge), ce qui est
    une approximation : l'âge officiel des chevaux de course change au
    1er janvier en France, pas au jour anniversaire réel.
"""
import re
import unicodedata
from collections import defaultdict


SEXES_MALES = {"MALES", "HONGRES"}
SEXES_FEMELLES = {"FEMELLES"}


def normaliser_nom(nom):
    if not nom:
        return None
    n = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode("ascii")
    n = n.upper().strip()
    n = re.sub(r"\s+", " ", n)
    return n or None


def sexe_compatible(s1, s2):
    if not s1 or not s2:
        return True  # information manquante : ne contredit rien
    if s1 == s2:
        return True
    if s1 in SEXES_MALES and s2 in SEXES_MALES:
        return True  # transition MALE -> HONGRE (castration), normale
    return False  # FEMELLE incompatible avec MALE/HONGRE, dans un sens ou l'autre


def annee_naissance_estimee(annee_course, age):
    if age is None:
        return None
    try:
        return int(annee_course) - int(age)
    except (TypeError, ValueError):
        return None


def age_compatible(annee_naissance_a, annee_naissance_b, tolerance=1):
    if annee_naissance_a is None or annee_naissance_b is None:
        return True
    return abs(annee_naissance_a - annee_naissance_b) <= tolerance


class Cluster:
    __slots__ = ("pere", "mere", "sexes_vus", "annee_naissance", "id_chevaux_vus", "n_lignes", "indices")

    def __init__(self, pere, mere, sexe, annee_naissance, id_cheval):
        self.pere = pere
        self.mere = mere
        self.sexes_vus = {sexe} if sexe else set()
        self.annee_naissance = annee_naissance
        self.id_chevaux_vus = {id_cheval} if id_cheval else set()
        self.n_lignes = 1
        self.indices = []

    def compatible(self, pere, mere, sexe, annee_naissance, id_cheval):
        # id_cheval fiable : priorité absolue si les deux sont renseignés.
        if id_cheval and self.id_chevaux_vus:
            if id_cheval not in self.id_chevaux_vus:
                return False
            return True  # id_cheval confirme l'identité, on ignore le reste
        if pere and self.pere and pere != self.pere:
            return False
        if mere and self.mere and mere != self.mere:
            return False
        for s_vu in self.sexes_vus:
            if not sexe_compatible(s_vu, sexe):
                return False
        if not age_compatible(self.annee_naissance, annee_naissance):
            return False
        return True

    def fusionner(self, pere, mere, sexe, annee_naissance, id_cheval):
        if pere and not self.pere:
            self.pere = pere
        if mere and not self.mere:
            self.mere = mere
        if sexe:
            self.sexes_vus.add(sexe)
        if annee_naissance is not None:
            # on garde la moyenne pondérée arrondie comme meilleure estimation
            if self.annee_naissance is None:
                self.annee_naissance = annee_naissance
            else:
                self.annee_naissance = round(
                    (self.annee_naissance * self.n_lignes + annee_naissance) / (self.n_lignes + 1)
                )
        if id_cheval:
            self.id_chevaux_vus.add(id_cheval)
        self.n_lignes += 1


def resoudre_identite_chevaux(lignes):
    """lignes : itérable de dicts avec au moins nom_cheval, nom_pere,
    nom_mere, sexe, age, date_course (datetime.date ou objet avec .year),
    id_cheval (peut être None).

    Retourne (horse_uid_par_index, rapport) où horse_uid_par_index est une
    liste parallèle à `lignes` donnant le horse_uid de chaque ligne, et
    rapport est un dict de statistiques + échantillon de cas ambigus.
    """
    par_nom = defaultdict(list)
    for i, l in enumerate(lignes):
        nom_norm = normaliser_nom(l.get("nom_cheval"))
        par_nom[nom_norm].append(i)

    horse_uid = [None] * len(lignes)
    noms_avec_collision = []
    total_clusters = 0
    total_noms = 0

    for nom_norm, indices in par_nom.items():
        if nom_norm is None:
            for i in indices:
                horse_uid[i] = None
            continue
        total_noms += 1
        # trier par date pour un passage chronologique stable
        indices_tries = sorted(indices, key=lambda i: (lignes[i].get("date_course") or 0))
        clusters = []
        for i in indices_tries:
            l = lignes[i]
            pere = normaliser_nom(l.get("nom_pere"))
            mere = normaliser_nom(l.get("nom_mere"))
            sexe = l.get("sexe")
            date_c = l.get("date_course")
            annee = getattr(date_c, "year", None)
            annee_naissance = annee_naissance_estimee(annee, l.get("age")) if annee else None
            id_cheval = l.get("id_cheval")

            candidats = [c for c in clusters if c.compatible(pere, mere, sexe, annee_naissance, id_cheval)]
            if candidats:
                # si plusieurs clusters compatibles (rare), prendre celui
                # avec le plus de lignes (le plus "établi")
                c = max(candidats, key=lambda c: c.n_lignes)
                c.fusionner(pere, mere, sexe, annee_naissance, id_cheval)
                c.indices.append(i)
            else:
                c = Cluster(pere, mere, sexe, annee_naissance, id_cheval)
                c.indices.append(i)
                clusters.append(c)

        total_clusters += len(clusters)
        if len(clusters) > 1:
            noms_avec_collision.append({
                "nom": nom_norm,
                "n_clusters": len(clusters),
                "n_lignes_total": len(indices),
                "detail": [
                    {"n_lignes": c.n_lignes, "pere": c.pere, "mere": c.mere,
                     "annee_naissance_est": c.annee_naissance}
                    for c in clusters
                ],
            })

        for rang, c in enumerate(clusters):
            uid = f"{nom_norm}::{rang}"
            for i in c.indices:
                horse_uid[i] = uid

    rapport = {
        "n_lignes": len(lignes),
        "n_noms_distincts": total_noms,
        "n_chevaux_distincts_resolus": total_clusters,
        "n_noms_avec_collision": len(noms_avec_collision),
        "pct_noms_avec_collision": round(100 * len(noms_avec_collision) / total_noms, 3) if total_noms else 0.0,
        "echantillon_collisions": noms_avec_collision[:25],
    }
    return horse_uid, rapport
