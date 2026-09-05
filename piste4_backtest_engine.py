"""
Moteur de backtest ROI — Piste 4 (B+généalogie -> VERT -> Top-5 -> marché H-15).

STATUT (05/09/2026, mise à jour) : les fonctions ci-dessous sont de vraies
implémentations (pas des stubs), sauf run_backtest() qui reste verrouillé.
Rien ici n'a été exécuté sur des données réelles de courses -- seulement un
auto-test sur données synthétiques fabriquées (voir _selftest en bas de
fichier), pour vérifier l'absence de bug avant commit.

Ajout du 05/09/2026 (demande explicite de Dorian, "préparer le moteur pour
être opérationnel dès que le volume de marché sera suffisant") : le pipeline
complet B+généalogie -> VERT -> Top-5 -> marché H-15 -> Top-1 marché ->
résultat -> ROI est maintenant implémenté de bout en bout
(top1_marche_h15_dans_top5_vert_strategy), avec sa comparaison au modèle
seul (top1_modele_strategy, compare_modele_seul_vs_marche_h15) et le
rendement cumulé dans le temps (compute_rendement_cumule).

Ajout du 05/09/2026 (deuxième demande, validation explicite de Dorian des 4
architectures conceptuelles décrites dans piste4_preparation_marche_decision.md,
chantier 3) : les 4 architectures sont maintenant implémentées comme
stratégies de SÉLECTION (architecture_a_confirmation, architecture_b_arbitre,
architecture_c_score_combine, architecture_d_filtrage_incertitude), avec
leurs paramètres libres FIGÉS ci-dessous à des valeurs simples et rondes,
choisies AVANT tout accès à un résultat de ROI, précisément pour éviter le
surapprentissage :
  - SEUIL_COTE_ARBITRE_STRATEGIE_B = 2.0 (le favori marché ne remplace le
    pick modèle, en cas de désaccord, que si sa cote H-15 est au moins 2.0 --
    valeur ronde, aucune recherche de valeur optimale).
  - POIDS_MODELE_STRATEGIE_C = 0.5 (pondération strictement égale entre la
    probabilité implicite du modèle et celle du marché dans le score
    combiné -- le choix le plus neutre possible, pas de grid-search).
  - SEUIL_ECART_COTE_TOP5_RELATIF_STRATEGIE_D = 0.10 (10% -- en dessous de ce
    seuil, l'écart de cote entre le favori et le deuxième choix du marché
    est jugé trop faible, le marché est considéré indécis, on ne joue pas).
Ces trois constantes sont les SEULS paramètres libres des 4 architectures.
Elles ne doivent pas être modifiées après avoir vu un résultat de ROI sur
données réelles -- toute révision devra être justifiée AVANT d'y toucher à
nouveau, et documentée comme telle (avec la raison), jamais en cherchant
la valeur qui améliore le ROI observé.

Ces 6 stratégies de SÉLECTION (quel cheval jouer) sont désormais enregistrées
dans SELECTION_STRATEGIES. En revanche, BETTING_RULES (quelle cote minimale
GLOBALE accepter pour placer un pari, quelle mise) reste VOLONTAIREMENT VIDE :
Dorian n'a validé que les paramètres INTERNES aux 4 architectures ci-dessus,
pas de règle de mise générale -- décision distincte, encore à prendre
séparément avant le lancement du gros backtest. Sans entrée dans
BETTING_RULES, run_backtest() continue de refuser de s'exécuter.

Ne pas exécuter run_backtest() sans :
  - un volume de courses avec cote H-15 exploitable jugé suffisant par Dorian
    (objectif 150-200 courses, ~51/76 le 05/09/2026, accumulation automatique
    en cours),
  - une règle de mise explicitement choisie par Dorian (BETTING_RULES reste
    vide exprès -- les stratégies de SÉLECTION, elles, sont prêtes, voir
    ci-dessus).

Ne modifie jamais :
- collect_live_odds_render.py / .github/workflows/collecte-cotes.yml
- les hyperparamètres LightGBM de B+généalogie (ce moteur ne réentraîne rien)
- le filtre VERT (SEUIL_VERT_FIGE, reproduit tel quel depuis piste7)
Ne construit jamais de modèle combiné B+généalogie+marché (le "score combiné"
de l'architecture C est une combinaison de PROBABILITÉS post-hoc au moment du
pari, pas un réentraînement -- B+généalogie n'est jamais retouché).

FILTRE VERT -- SOURCE DE VÉRITÉ (résolu le 04/09/2026 par inspection du dépôt) :
La classification VERT/ORANGE/ROUGE n'est PAS recalculée ici à neuf. Elle est
reproduite à l'identique depuis le filtre déjà figé et utilisé en production
dans piste7_phase2_confiance_directe.py (calibration initiale, 01/09/2026) et
repris tel quel dans piste7_diagnostic_marge_top1_vert.py et
piste7_approche3_specialise_vert.py :
  SEUIL_VERT_FIGE = 0.5848 sur somme_top3_proba, où somme_top3_proba est la
  somme des 3 plus hautes pseudo-probabilités (softmax de score_geneal, alias
  score_modele dans l'export de test_marche_forward_29082026.py -- même
  modèle B+généalogie, même entraînement, seule la colonne change de nom
  d'un script à l'autre) calculées sur TOUT le champ de la course (pas
  seulement le Top-5). Le Top-5 lui-même reste le classement brut
  (rang_geneal / rang_modele <= 5), inchangé par le filtre VERT.
Ce moteur ne modifie AUCUN script de production pour obtenir ce champ : le
score_modele est déjà présent dans l'export scores_modele_{date}.csv, donc
somme_top3_proba et le niveau VERT sont recalculables en aval, sans toucher
à test_marche_forward_29082026.py ni à v3_lib.py.
Remarque pour Dorian : le seuil 0.5848 est dupliqué ici tel quel (avec
attribution) plutôt que factorisé dans v3_lib.py, pour ne pas toucher à un
module partagé par les scripts d'entraînement sans ta validation. Si tu veux
une source unique partagée, c'est un refactor séparé à valider explicitement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constante figée -- reprise à l'identique de piste7 (NE PAS RETOUCHER ICI
# SANS RETOUCHER LA SOURCE -- ce n'est pas un paramètre de ce moteur).
# ---------------------------------------------------------------------------
SEUIL_VERT_FIGE = 0.5848

# ---------------------------------------------------------------------------
# Paramètres libres des 4 architectures de décision (chantier 3, VALIDÉS et
# FIGÉS par Dorian le 05/09/2026, AVANT tout résultat de ROI -- voir
# docstring du module pour la justification de chaque valeur). Ne pas
# modifier après avoir vu un ROI sur données réelles.
# ---------------------------------------------------------------------------
SEUIL_COTE_ARBITRE_STRATEGIE_B = 2.0
POIDS_MODELE_STRATEGIE_C = 0.5
SEUIL_ECART_COTE_TOP5_RELATIF_STRATEGIE_D = 0.10

# ---------------------------------------------------------------------------
# 1. Chargement des prédictions du modèle (sortie de test_marche_forward_29082026.py)
# ---------------------------------------------------------------------------

def load_model_scores(csv_path: str) -> pd.DataFrame:
    """Charge un fichier scores_modele_{date}.csv déjà récupéré sur disque
    (typiquement extrait des marqueurs ===CSV_SCORES_START/END=== du Job
    Summary GitHub Actions, comme fait pour piste4_marche_journalier.csv).

    Colonnes attendues : course_id, numero, position_arrivee, est_gagnant,
    cible_place, rang_modele, score_modele, categorie_particularite.

    LIMITE CONNUE (à signaler, pas à corriger seul) : ces CSV par cheval ne
    sont PAS persistés dans le dépôt au-delà de l'exécution qui les produit
    (seul l'agrégat par course entre dans piste4_marche_journalier.csv). Pour
    un backtest rétroactif sur plusieurs jours, il faudra soit les avoir
    sauvegardés au moment de chaque run, soit décider avec Dorian d'ajouter
    une persistance dédiée -- décision hors périmètre de ce moteur.
    """
    df = pd.read_csv(csv_path)
    colonnes_requises = {
        "course_id", "numero", "position_arrivee", "est_gagnant",
        "cible_place", "rang_modele", "score_modele", "categorie_particularite",
    }
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans {csv_path} : {sorted(manquantes)}")
    return df

# ---------------------------------------------------------------------------
# 2. Filtre VERT -- reproduit à l'identique depuis piste7 (voir docstring module)
# ---------------------------------------------------------------------------

def calculer_somme_top3_proba(scores: pd.DataFrame, colonne_score: str = "score_modele") -> pd.DataFrame:
    """Ajoute la colonne somme_top3_proba : softmax de `colonne_score` calculé
    sur TOUT le champ de chaque course_id, puis somme des 3 plus hautes
    probabilités. Reproduction exacte de calculer_somme_top3_proba() dans
    piste7_diagnostic_marge_top1_vert.py (softmax stabilisé par soustraction
    du max, pas d'autre transformation)."""
    d = scores.copy()
    out = np.empty(len(d), dtype=float)
    for _, idx in d.groupby("course_id").groups.items():
        pos = d.index.get_indexer(idx)
        s = d.loc[idx, colonne_score].to_numpy(dtype=float)
        z = s - s.max()
        proba = np.exp(z) / np.exp(z).sum()
        somme_top3 = float(np.sort(proba)[::-1][:3].sum())
        out[pos] = somme_top3
    d["somme_top3_proba"] = out
    return d

def filter_vert(scores: pd.DataFrame, colonne_score: str = "score_modele") -> pd.DataFrame:
    """Classe chaque course VERT / NON-VERT selon le seuil figé de piste7, et
    retourne UNIQUEMENT les lignes des courses VERT (toutes leurs lignes,
    pas juste le Top-5 -- rank_top5() s'en charge séparément).

    Ajoute les colonnes somme_top3_proba, niveau_vert (bool),
    seuil_vert_utilise (traçabilité, toujours 0.5848 ici)."""
    d = calculer_somme_top3_proba(scores, colonne_score=colonne_score)
    d["niveau_vert"] = d["somme_top3_proba"] >= SEUIL_VERT_FIGE
    d["seuil_vert_utilise"] = SEUIL_VERT_FIGE
    return d[d["niveau_vert"]].copy()

# ---------------------------------------------------------------------------
# 3. Top-5 parmi les VERT
# ---------------------------------------------------------------------------

def rank_top5(scores_vert: pd.DataFrame) -> pd.DataFrame:
    """Garde au plus 5 chevaux par course_id, classement BRUT B+généalogie
    (rang_modele croissant, inchangé par le filtre VERT). Si moins de 5
    VERT disponibles pour une course, retourne tous les partants de cette
    course (le filtre VERT s'applique à la COURSE, pas au cheval -- une
    fois la course retenue, on garde son Top-5 tel quel)."""
    if scores_vert.empty:
        return scores_vert.copy()
    return (
        scores_vert[scores_vert["rang_modele"] <= 5]
        .sort_values(["course_id", "rang_modele"])
        .reset_index(drop=True)
    )

# ---------------------------------------------------------------------------
# 4. Cote H-15 point-in-time stricte
# ---------------------------------------------------------------------------

def compute_point_in_time_odds(cotes_brutes: pd.DataFrame, window_minutes: int = 15) -> pd.DataFrame:
    """Reçoit un DataFrame brut (course_id, numero, minutes_avant_depart,
    cote) tel que renvoyé par une requête SELECT simple sur cotes_historique
    (même requête que charger_cotes_marche() dans
    test_marche_forward_29082026.py -- aucune logique de fenêtre en SQL),
    et applique la méthodologie point-in-time STRICTE validée le 30/08/2026 :

    - snapshot interdit si minutes_avant_depart < window_minutes (jamais une
      cote captée après l'instant simulé) ;
    - snapshot négatif (minutes_avant_depart < 0, cote captée après le
      départ réel) toujours exclu, quelle que soit la fenêtre ;
    - parmi les snapshots valides, on garde le PLUS PETIT minutes_avant_depart
      (= la mise à jour la plus récente disponible au moment ou avant
      H-{window_minutes}) ;
    - jamais de logique "closest to target" (ORDER BY abs(ecart)) ;
    - dédoublonnage par (course_id, numero) AVANT tout calcul de rang.
    """
    if window_minutes < 0:
        raise ValueError("window_minutes ne peut pas être négatif.")
    d = cotes_brutes.copy()
    d = d[(d["minutes_avant_depart"] >= window_minutes) & (d["minutes_avant_depart"] >= 0)]
    if d.empty:
        return pd.DataFrame(columns=["course_id", "numero", "minutes_avant_depart", "cote"])
    d = d.sort_values("minutes_avant_depart", ascending=True)
    d = d.drop_duplicates(["course_id", "numero"], keep="first")
    return d[["course_id", "numero", "minutes_avant_depart", "cote"]].reset_index(drop=True)

def check_couverture(cotes_pit: pd.DataFrame, resultats: pd.DataFrame, seuil_pct: float = 70.0) -> pd.DataFrame:
    """Retourne, par course_id, le % de partants (ceux ayant un résultat
    dans `resultats`) couverts par une cote point-in-time valide, et si la
    course franchit le seuil minimal (70% par défaut, jamais de valeur
    estimée en dessous)."""
    n_partants = resultats.groupby("course_id")["numero"].nunique().rename("n_partants_avec_resultat")
    n_avec_cote = cotes_pit.groupby("course_id")["numero"].nunique().rename("n_avec_cote_pit")
    cov = pd.concat([n_partants, n_avec_cote], axis=1).fillna(0)
    cov["couverture_pct"] = 100 * cov["n_avec_cote_pit"] / cov["n_partants_avec_resultat"]
    cov["exploitable"] = cov["couverture_pct"] >= seuil_pct
    return cov.reset_index()

def rank_odds(cotes_pit: pd.DataFrame) -> pd.DataFrame:
    """Rang de cote par course, calculé sur les cotes DÉJÀ dédupliquées par
    cheval (jamais sur les lignes brutes -- sinon le rang explose)."""
    d = cotes_pit.copy()
    d["rang_cote"] = d.groupby("course_id")["cote"].rank(method="min", ascending=True)
    return d

# ---------------------------------------------------------------------------
# 5. Stratégies de sélection (registre extensible)
# ---------------------------------------------------------------------------

@dataclass
class RaceContext:
    course_id: str
    top5: pd.DataFrame          # Top-5 VERT + cote_h15 + variables marché (5b) si dispo
    faible_confiance: bool
    handicap: bool
    date: Optional[str] = None  # requis uniquement pour compute_rendement_cumule()

SelectionStrategy = Callable[[RaceContext], Optional[int]]  # renvoie un numéro de cheval ou None

def top1_modele_strategy(ctx: RaceContext) -> Optional[int]:
    """B+généalogie SEUL : le pick #1 du modèle (rang_modele == 1) parmi le
    Top-5 VERT. Sert de référence pour la comparaison "modèle seul" demandée
    par Dorian (05/09/2026) -- n'utilise jamais la cote de marché."""
    row = ctx.top5[ctx.top5["rang_modele"] == 1]
    return int(row["numero"].iloc[0]) if not row.empty else None

def top1_marche_h15_dans_top5_vert_strategy(ctx: RaceContext) -> Optional[int]:
    """MARCHÉ SEUL (restreint au Top-5 VERT) : parmi le Top-5 VERT déjà
    déterminé par B+généalogie (rang_modele <= 5, course VERT), on
    sélectionne le cheval avec la cote de marché H-15 la plus basse (favori
    du marché RESTREINT au Top-5 VERT, jamais sur le champ entier -- le
    filtre modèle reste la première étape, le marché ne fait qu'arbitrer
    ENTRE les chevaux déjà retenus par le modèle). Sert à la fois de
    référence "marché seul" et de pipeline "modèle+marché" (05/09/2026) dans
    la grande comparaison demandée par Dorian -- c'est la même stratégie,
    les deux angles de lecture sont légitimes puisque le marché est de toute
    façon restreint au Top-5 du modèle. Retourne None si aucun cheval du
    Top-5 n'a de cote H-15 exploitable (jamais d'estimation)."""
    d = ctx.top5.dropna(subset=["cote_h15"]) if "cote_h15" in ctx.top5.columns else ctx.top5.iloc[0:0]
    if d.empty:
        return None
    return int(d.loc[d["cote_h15"].idxmin(), "numero"])

# ---------------------------------------------------------------------------
# 5b. Variables de marché H-15 sans fuite (chantier 1, spécifié le 05/09/2026
# dans piste4_preparation_marche_decision.md, implémenté le même jour après
# validation de Dorian). Toutes calculées UNIQUEMENT à partir de cote_h15
# (déjà point-in-time strict, voir section 4) et des colonnes déjà présentes
# dans le Top-5 VERT -- aucune source de données supplémentaire, aucune
# fuite temporelle possible au-delà de ce que compute_point_in_time_odds()
# garantit déjà.
#
# LIMITE ASSUMÉE (documentée, pas corrigée ici) : ecart_cote_top5,
# dispersion_cotes_top5 et proba_implicite_marche sont calculées sur le
# Top-5 VERT uniquement, jamais sur le champ entier de la course -- cohérent
# avec le fait que toutes les stratégies de décision de ce moteur restent
# elles-mêmes restreintes au Top-5 VERT (le marché n'arbitre jamais en
# dehors du Top-5 déjà retenu par le modèle). Une version "champ entier"
# demanderait de faire transiter les cotes de tous les partants (pas
# seulement le Top-5) jusqu'à RaceContext, ce qui n'est pas câblé
# aujourd'hui -- hors périmètre de cette préparation, à discuter séparément
# si Dorian le juge utile après le gros backtest.
# ---------------------------------------------------------------------------

def _softmax(valeurs: np.ndarray) -> np.ndarray:
    z = valeurs - valeurs.max()
    e = np.exp(z)
    return e / e.sum()

def add_market_features(top5_avec_cote_h15: pd.DataFrame) -> pd.DataFrame:
    """Ajoute au Top-5 VERT (déjà fusionné avec cote_h15, voir _selftest pour
    l'exemple de fusion) les colonnes suivantes, calculées par groupby
    course_id :
      - rang_cote_h15 : rang croissant de cote_h15 parmi les chevaux du
        Top-5 ayant une cote H-15 valide (NaN sinon).
      - favori_h15 : booléen, rang_cote_h15 == 1.
      - proba_modele_top5 : softmax de score_modele RESTREINT au Top-5 (voir
        limite ci-dessus -- distinct de somme_top3_proba qui est calculée sur
        le champ entier pour le filtre VERT).
      - proba_implicite_marche : 1/cote_h15 normalisée sur les chevaux du
        Top-5 ayant une cote H-15 valide (NaN pour les autres).
      - ecart_cote_top5_abs / ecart_cote_top5_relatif : écart entre la cote
        H-15 la plus basse et la deuxième plus basse du Top-5 (valeur
        répétée sur toutes les lignes de la course -- NaN si moins de 2
        chevaux du Top-5 ont une cote H-15 valide).
      - dispersion_cotes_top5 : coefficient de variation (écart-type / moyenne)
        des cote_h15 valides du Top-5 (valeur répétée par course -- NaN si
        moins de 2 valeurs valides).
      - accord_desaccord_modele_marche : "accord_total" si le pick #1 modèle
        (rang_modele==1) est aussi le favori marché (favori_h15==1),
        "desaccord_partiel" si le favori marché est un autre cheval du
        Top-5, None/NaN si aucun favori marché n'est déterminable (pas de
        cote H-15 exploitable dans le Top-5).
    """
    d = top5_avec_cote_h15.copy()
    if "cote_h15" not in d.columns:
        d["cote_h15"] = np.nan

    d["rang_cote_h15"] = d.groupby("course_id")["cote_h15"].rank(method="min", ascending=True)
    d["favori_h15"] = d["rang_cote_h15"] == 1.0

    proba_modele = np.full(len(d), np.nan)
    proba_marche = np.full(len(d), np.nan)
    ecart_abs = np.full(len(d), np.nan)
    ecart_rel = np.full(len(d), np.nan)
    dispersion = np.full(len(d), np.nan)
    accord = np.array([None] * len(d), dtype=object)

    for course_id, idx in d.groupby("course_id").groups.items():
        pos = d.index.get_indexer(idx)
        bloc = d.loc[idx]

        s = bloc["score_modele"].to_numpy(dtype=float)
        proba_modele[pos] = _softmax(s)

        cotes_valides = bloc["cote_h15"].dropna()
        if len(cotes_valides) >= 1:
            inv = 1.0 / cotes_valides.to_numpy(dtype=float)
            proba_norm = inv / inv.sum()
            for i, (label_idx, p) in enumerate(zip(cotes_valides.index, proba_norm)):
                proba_marche[d.index.get_indexer([label_idx])[0]] = p

        if len(cotes_valides) >= 2:
            cotes_triees = cotes_valides.sort_values()
            e_abs = float(cotes_triees.iloc[1] - cotes_triees.iloc[0])
            e_rel = float(e_abs / cotes_triees.iloc[0]) if cotes_triees.iloc[0] != 0 else np.nan
            ecart_abs[pos] = e_abs
            ecart_rel[pos] = e_rel
            moyenne = float(cotes_valides.mean())
            ecart_type = float(cotes_valides.std(ddof=0))
            dispersion[pos] = ecart_type / moyenne if moyenne != 0 else np.nan

        favori_rows = bloc[bloc["favori_h15"]] if "favori_h15" in bloc.columns else bloc.iloc[0:0]
        modele_rows = bloc[bloc["rang_modele"] == 1]
        if favori_rows.empty or modele_rows.empty:
            valeur_accord = None
        elif int(favori_rows["numero"].iloc[0]) == int(modele_rows["numero"].iloc[0]):
            valeur_accord = "accord_total"
        else:
            valeur_accord = "desaccord_partiel"
        accord[pos] = valeur_accord

    d["proba_modele_top5"] = proba_modele
    d["proba_implicite_marche"] = proba_marche
    d["ecart_cote_top5_abs"] = ecart_abs
    d["ecart_cote_top5_relatif"] = ecart_rel
    d["dispersion_cotes_top5"] = dispersion
    d["accord_desaccord_modele_marche"] = accord
    return d

# ---------------------------------------------------------------------------
# 5c. Architectures de décision (chantier 3) -- VALIDÉES et paramètres FIGÉS
# par Dorian le 05/09/2026 (voir constantes en tête de fichier). Aucune
# testée sur données réelles ni sur les 51-76 courses H-15 déjà accumulées --
# implémentation préparatoire uniquement, exercée ci-dessous seulement sur
# données synthétiques (_selftest).
# ---------------------------------------------------------------------------

def architecture_a_confirmation_strategy(ctx: RaceContext) -> Optional[int]:
    """A. Marché comme confirmation : parie le pick #1 du modèle UNIQUEMENT
    si le marché le confirme (accord_desaccord_modele_marche ==
    'accord_total'). Sinon, aucun pari. Aucun paramètre libre."""
    if "accord_desaccord_modele_marche" not in ctx.top5.columns:
        return None
    accord = ctx.top5["accord_desaccord_modele_marche"].iloc[0] if not ctx.top5.empty else None
    if accord != "accord_total":
        return None
    row = ctx.top5[ctx.top5["rang_modele"] == 1]
    return int(row["numero"].iloc[0]) if not row.empty else None

def architecture_b_arbitre_strategy(ctx: RaceContext) -> Optional[int]:
    """B. Marché comme arbitre en cas de désaccord : si accord_total, parie
    le pick modèle. Si desaccord_partiel, bascule sur le favori marché SI sa
    cote H-15 est >= SEUIL_COTE_ARBITRE_STRATEGIE_B (figé, voir tête de
    fichier) ; sinon aucun pari. Si l'accord n'est pas déterminable (pas de
    cote exploitable), aucun pari."""
    if ctx.top5.empty or "accord_desaccord_modele_marche" not in ctx.top5.columns:
        return None
    accord = ctx.top5["accord_desaccord_modele_marche"].iloc[0]
    if accord == "accord_total":
        row = ctx.top5[ctx.top5["rang_modele"] == 1]
        return int(row["numero"].iloc[0]) if not row.empty else None
    if accord == "desaccord_partiel":
        favori = ctx.top5[ctx.top5["favori_h15"]] if "favori_h15" in ctx.top5.columns else ctx.top5.iloc[0:0]
        if favori.empty:
            return None
        cote_favori = float(favori["cote_h15"].iloc[0])
        if cote_favori >= SEUIL_COTE_ARBITRE_STRATEGIE_B:
            return int(favori["numero"].iloc[0])
        return None
    return None

def architecture_c_score_combine_strategy(ctx: RaceContext) -> Optional[int]:
    """C. Score modèle/marché combiné : score = POIDS_MODELE_STRATEGIE_C *
    proba_modele_top5 + (1 - POIDS_MODELE_STRATEGIE_C) * proba_implicite_marche
    (poids figé, voir tête de fichier), calculé UNIQUEMENT sur les chevaux du
    Top-5 ayant une cote H-15 valide (impossible de comparer équitablement un
    cheval sans donnée marché) -- retient le score combiné maximal. Retourne
    None si aucun cheval du Top-5 n'a de cote H-15 valide."""
    if ctx.top5.empty or "proba_modele_top5" not in ctx.top5.columns:
        return None
    d = ctx.top5.dropna(subset=["cote_h15", "proba_implicite_marche"])
    if d.empty:
        return None
    score = (POIDS_MODELE_STRATEGIE_C * d["proba_modele_top5"]
             + (1 - POIDS_MODELE_STRATEGIE_C) * d["proba_implicite_marche"])
    return int(d.loc[score.idxmax(), "numero"])

def architecture_d_filtrage_incertitude_strategy(ctx: RaceContext) -> Optional[int]:
    """D. Filtrage des situations trop incertaines : le cheval joué reste
    TOUJOURS le pick #1 du modèle (cette architecture ne change jamais le
    choix du cheval), mais aucun pari n'est placé si le marché est jugé trop
    indécis -- ecart_cote_top5_relatif < SEUIL_ECART_COTE_TOP5_RELATIF_STRATEGIE_D
    (figé, voir tête de fichier), OU si cet écart n'est pas déterminable
    (moins de 2 cotes H-15 valides dans le Top-5 -- absence de signal marché
    traitée par prudence comme une incertitude, pas comme une confirmation)."""
    if ctx.top5.empty:
        return None
    row = ctx.top5[ctx.top5["rang_modele"] == 1]
    if row.empty:
        return None
    ecart_rel = ctx.top5["ecart_cote_top5_relatif"].iloc[0] if "ecart_cote_top5_relatif" in ctx.top5.columns else np.nan
    if pd.isna(ecart_rel) or ecart_rel < SEUIL_ECART_COTE_TOP5_RELATIF_STRATEGIE_D:
        return None
    return int(row["numero"].iloc[0])

SELECTION_STRATEGIES: dict[str, SelectionStrategy] = {
    "top1_modele": top1_modele_strategy,
    "top1_marche_h15_dans_top5_vert": top1_marche_h15_dans_top5_vert_strategy,
    "architecture_a_confirmation": architecture_a_confirmation_strategy,
    "architecture_b_arbitre": architecture_b_arbitre_strategy,
    "architecture_c_score_combine": architecture_c_score_combine_strategy,
    "architecture_d_filtrage_incertitude": architecture_d_filtrage_incertitude_strategy,
    # Les 6 stratégies ci-dessus sont VALIDÉES par Dorian (05/09/2026) pour
    # préparer le futur "backtest décisionnel complet" (modèle seul / marché
    # seul / modèle+marché / 4 architectures). Aucune n'est CHOISIE comme
    # stratégie de pari finale -- run_backtest() reste verrouillé
    # indépendamment de leur présence ici. Ne pas ajouter d'autre stratégie
    # sans décision explicite de Dorian (consigne réaffirmée le 05/09/2026 :
    # "aucun nouveau chantier exploratoire avant ce backtest").
}

# ---------------------------------------------------------------------------
# 6. Règles de mise / cote minimale acceptable (registre extensible)
# ---------------------------------------------------------------------------

BettingRule = Callable[[float], bool]  # cote_h15 -> pari autorisé ?

BETTING_RULES: dict[str, BettingRule] = {
    # "cote_min_2_0": lambda cote: cote >= 2.0,
    # Volontairement vide. Dorian a figé le 05/09/2026 les paramètres
    # INTERNES des 4 architectures (voir tête de fichier), mais pas de règle
    # de mise GLOBALE (cote minimale pour parier + montant de la mise) --
    # décision distincte, encore à prendre séparément avant le gros backtest.
}

# ---------------------------------------------------------------------------
# 7. Décision de pari + résultat + gain
# ---------------------------------------------------------------------------

def decide_and_settle_bet(
    ctx: RaceContext,
    strategy: SelectionStrategy,
    rule: BettingRule,
    mise: float,
    resultats: pd.DataFrame,
) -> dict:
    """Applique stratégie + règle de mise, joint le résultat réel, calcule le
    gain. Ne place PAS de pari (pari_place=0) si aucun cheval n'est
    sélectionné ou si sa cote H-15 est sous le seuil -- la course reste
    comptabilisée comme évaluée pour ne pas biaiser le taux de "skip".

    `resultats` : DataFrame course_id, numero, position_arrivee, est_gagnant.
    """
    ligne = {
        "course_id": ctx.course_id,
        "date": ctx.date,
        "faible_confiance": ctx.faible_confiance,
        "handicap": ctx.handicap,
        "cheval_selectionne": None,
        "cote_h15_selectionne": None,
        "pari_place": 0,
        "mise": 0.0,
        "gagnant": 0,
        "gain_brut": 0.0,
        "resultat_net": 0.0,
    }

    numero = strategy(ctx)
    if numero is None:
        return ligne
    ligne["cheval_selectionne"] = numero

    cheval_row = ctx.top5[ctx.top5["numero"] == numero]
    if cheval_row.empty or "cote_h15" not in cheval_row or pd.isna(cheval_row["cote_h15"].iloc[0]):
        # Sélectionné par la stratégie mais pas de cote H-15 exploitable : pas de pari.
        return ligne
    cote = float(cheval_row["cote_h15"].iloc[0])
    ligne["cote_h15_selectionne"] = cote

    if not rule(cote):
        return ligne  # sous la cote minimale acceptable : pas de pari

    ligne["pari_place"] = 1
    ligne["mise"] = mise

    res = resultats[(resultats["course_id"] == ctx.course_id) & (resultats["numero"] == numero)]
    gagnant = bool(res["est_gagnant"].iloc[0]) if not res.empty else False
    ligne["gagnant"] = int(gagnant)
    if gagnant:
        ligne["gain_brut"] = mise * (cote - 1.0)
        ligne["resultat_net"] = ligne["gain_brut"]
    else:
        ligne["gain_brut"] = 0.0
        ligne["resultat_net"] = -mise

    return ligne

# ---------------------------------------------------------------------------
# 7b. Simulation d'une stratégie sur un ensemble de courses (préparation --
# ne PAS appeler sur données réelles avant décision explicite de Dorian ;
# utilisé pour l'instant uniquement par _selftest() sur données synthétiques)
# ---------------------------------------------------------------------------

def simuler_strategie(
    contexts: list[RaceContext],
    strategy: SelectionStrategy,
    rule: BettingRule,
    mise: float,
    resultats: pd.DataFrame,
) -> pd.DataFrame:
    """Applique decide_and_settle_bet() à une liste de RaceContext (une par
    course) et retourne le DataFrame `bets` complet. Pure orchestration --
    aucune logique de décision ici, tout est dans strategy/rule/decide_and_settle_bet.
    Ne choisit ni la stratégie ni la règle : les deux sont des paramètres
    explicites fournis par l'appelant (jamais de valeur par défaut choisie
    ici)."""
    lignes = [decide_and_settle_bet(ctx, strategy, rule, mise, resultats) for ctx in contexts]
    return pd.DataFrame(lignes)

# ---------------------------------------------------------------------------
# 8. Agrégation : ROI, drawdown, taux de réussite
# ---------------------------------------------------------------------------

def compute_metrics(bets: pd.DataFrame) -> dict:
    """Calcule, sur l'ensemble des paris ÉVALUÉS (une ligne par course, que le
    pari ait été placé ou non) :
    - n_courses_evaluees, n_paris_places
    - roi_pct = gain net total / mise totale (sur les paris placés)
    - taux_reussite = paris gagnants / paris placés
    - drawdown_max_pct sur la courbe de bankroll cumulée (ordre chronologique
      d'entrée du DataFrame -- l'appelant doit trier par date avant d'appeler)
    """
    n_courses_evaluees = len(bets)
    places = bets[bets["pari_place"] == 1]
    n_paris_places = len(places)
    if n_paris_places == 0:
        return {
            "n_courses_evaluees": n_courses_evaluees, "n_paris_places": 0,
            "n_paris_gagnants": 0, "taux_reussite_pct": None,
            "mise_totale": 0.0, "gain_total": 0.0, "roi_pct": None,
            "drawdown_max_pct": None, "cote_moyenne": None,
            "taux_courses_jouees_pct": round(100 * 0 / n_courses_evaluees, 1) if n_courses_evaluees else None,
        }
    mise_totale = float(places["mise"].sum())
    gain_total = float(places["resultat_net"].sum())
    n_paris_gagnants = int(places["gagnant"].sum())
    cumul = places["resultat_net"].cumsum()
    pic = cumul.cummax()
    drawdown = cumul - pic
    # drawdown_max_pct rapporté à la mise totale engagée jusque-là (pas à une
    # bankroll de départ arbitraire, qui n'est pas encore définie)
    drawdown_max_pct = float(-drawdown.min() / mise_totale * 100) if mise_totale > 0 else None
    cote_moyenne = float(places["cote_h15_selectionne"].mean()) if "cote_h15_selectionne" in places.columns else None
    return {
        "n_courses_evaluees": n_courses_evaluees,
        "n_paris_places": n_paris_places,
        "n_paris_gagnants": n_paris_gagnants,
        "taux_reussite_pct": round(100 * n_paris_gagnants / n_paris_places, 1),
        "mise_totale": mise_totale,
        "gain_total": round(gain_total, 2),
        "roi_pct": round(100 * gain_total / mise_totale, 1) if mise_totale > 0 else None,
        "drawdown_max_pct": round(drawdown_max_pct, 1) if drawdown_max_pct is not None else None,
        "cote_moyenne": round(cote_moyenne, 2) if cote_moyenne is not None else None,
        "taux_courses_jouees_pct": round(100 * n_paris_places / n_courses_evaluees, 1) if n_courses_evaluees else None,
    }

def compute_metrics_par_segment(bets: pd.DataFrame) -> dict:
    """compute_metrics() décliné global / faible_confiance / handicap."""
    return {
        "global": compute_metrics(bets),
        "faible_confiance": compute_metrics(bets[bets["faible_confiance"]]),
        "handicap": compute_metrics(bets[bets["handicap"]]),
    }

def compute_rendement_cumule(bets: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Rendement cumulé dans le temps (demandé par Dorian, 05/09/2026), en
    plus des métriques globales de compute_metrics(). Trie les paris PLACÉS
    par date croissante (date obligatoire -- lève une erreur explicite si
    `date_col` est absente ou contient des valeurs manquantes, plutôt que de
    trier dans un ordre arbitraire qui fausserait le drawdown/la courbe).
    Retourne une ligne par pari placé, dans l'ordre chronologique :
    date, mise_cumulee, gain_cumule, roi_cumule_pct, n_paris_cumule,
    n_gagnants_cumule, winrate_cumule_pct."""
    places = bets[bets["pari_place"] == 1].copy()
    if places.empty:
        return pd.DataFrame(columns=[
            date_col, "mise_cumulee", "gain_cumule", "roi_cumule_pct",
            "n_paris_cumule", "n_gagnants_cumule", "winrate_cumule_pct",
        ])
    if date_col not in places.columns or places[date_col].isna().any():
        raise ValueError(
            f"compute_rendement_cumule requiert une colonne '{date_col}' "
            "renseignée pour CHAQUE pari placé (tri chronologique obligatoire) "
            "-- renseigner RaceContext.date en amont."
        )
    places = places.sort_values(date_col, kind="stable").reset_index(drop=True)
    places["mise_cumulee"] = places["mise"].cumsum()
    places["gain_cumule"] = places["resultat_net"].cumsum()
    places["roi_cumule_pct"] = round(100 * places["gain_cumule"] / places["mise_cumulee"], 1)
    places["n_paris_cumule"] = np.arange(1, len(places) + 1)
    places["n_gagnants_cumule"] = places["gagnant"].cumsum()
    places["winrate_cumule_pct"] = round(100 * places["n_gagnants_cumule"] / places["n_paris_cumule"], 1)
    return places[[date_col, "mise_cumulee", "gain_cumule", "roi_cumule_pct",
                   "n_paris_cumule", "n_gagnants_cumule", "winrate_cumule_pct"]]

# ---------------------------------------------------------------------------
# 8b. Stabilité par sous-période (demandé par Dorian, 05/09/2026, pour le
# gros backtest décisionnel complet). Découpage en N tranches CHRONOLOGIQUES
# de taille égale (nombre de courses évaluées, pas de calendrier fixe) --
# choix délibéré pour rester robuste quelle que soit la durée réelle de la
# période accumulée (un découpage par mois calendaire produirait un nombre
# de tranches imprévisible et non comparable d'un run à l'autre).
# ---------------------------------------------------------------------------

def compute_stabilite_par_sous_periode(bets: pd.DataFrame, date_col: str = "date", n_periodes: int = 4) -> pd.DataFrame:
    """Découpe les courses ÉVALUÉES (pas seulement les paris placés, pour que
    chaque tranche corresponde à une période calendaire cohérente) en
    n_periodes tranches chronologiques de taille égale (à un near-egal près),
    calcule compute_metrics() sur chacune. Retourne une ligne par période :
    periode (1..n_periodes), date_debut, date_fin, puis les champs de
    compute_metrics(). Lève une erreur explicite si date_col est absente ou
    incomplète (même contrainte que compute_rendement_cumule)."""
    if date_col not in bets.columns or bets[date_col].isna().any():
        raise ValueError(
            f"compute_stabilite_par_sous_periode requiert une colonne '{date_col}' "
            "renseignée pour CHAQUE course évaluée."
        )
    d = bets.sort_values(date_col, kind="stable").reset_index(drop=True)
    n = len(d)
    if n == 0:
        return pd.DataFrame()
    tranches = np.array_split(np.arange(n), min(n_periodes, n))
    lignes = []
    for i, idx in enumerate(tranches, start=1):
        bloc = d.iloc[idx]
        m = compute_metrics(bloc)
        lignes.append({
            "periode": i,
            "date_debut": bloc[date_col].iloc[0],
            "date_fin": bloc[date_col].iloc[-1],
            **m,
        })
    return pd.DataFrame(lignes)

# ---------------------------------------------------------------------------
# 9. Comparaison à des stratégies de référence
# ---------------------------------------------------------------------------

BASELINE_STRATEGIES = [
    "flat_top1_modele",          # B+généalogie seul, sans marché
    "flat_favori_marche",        # favori du marché seul, sans modèle
    "flat_aleatoire_top5_vert",  # borne basse
]

def compare_to_baselines(resultats_par_strategie: dict[str, dict]) -> pd.DataFrame:
    """Assemble un tableau comparatif à partir de dicts {strategie_id:
    compute_metrics(...)}. Ne calcule rien de nouveau -- pure mise en forme,
    pour comparer une stratégie testée aux baselines de BASELINE_STRATEGIES."""
    lignes = []
    for strategie_id, m in resultats_par_strategie.items():
        lignes.append({"strategie_id": strategie_id, **m})
    return pd.DataFrame(lignes)

def compare_modele_seul_vs_marche_h15(
    contexts: list[RaceContext],
    rule: BettingRule,
    mise: float,
    resultats: pd.DataFrame,
) -> pd.DataFrame:
    """Comparaison explicitement demandée par Dorian (05/09/2026) :
    'top1_modele' (B+généalogie seul, sans marché) vs
    'top1_marche_h15_dans_top5_vert' (pipeline complet B+généalogie -> VERT ->
    Top-5 -> marché H-15 -> Top-1 marché), sur le MÊME ensemble de courses et
    la MÊME règle de mise/mise (fournies par l'appelant -- ce module ne
    choisit toujours aucune règle de mise par défaut). Retourne un tableau à
    une ligne par stratégie (compute_metrics), prêt pour compute_rendement_cumule
    en complément si l'appelant a besoin de la courbe dans le temps."""
    strategies_comparees = {
        k: v for k, v in SELECTION_STRATEGIES.items()
        if k in ("top1_modele", "top1_marche_h15_dans_top5_vert")
    }
    resultats_par_strategie = {
        strategie_id: compute_metrics(simuler_strategie(contexts, fn, rule, mise, resultats))
        for strategie_id, fn in strategies_comparees.items()
    }
    return compare_to_baselines(resultats_par_strategie)

def compare_toutes_strategies(
    contexts: list[RaceContext],
    rule: BettingRule,
    mise: float,
    resultats: pd.DataFrame,
    n_periodes: int = 4,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Brique prête pour LE gros backtest décisionnel complet demandé par
    Dorian (05/09/2026) : compare, sur le MÊME ensemble de courses et la
    MÊME règle de mise/mise, les 6 stratégies de SELECTION_STRATEGIES --
    modèle seul (top1_modele), marché seul restreint au Top-5 VERT
    (top1_marche_h15_dans_top5_vert, qui sert aussi de pipeline modèle+marché,
    voir sa docstring), et les 4 architectures validées (A/B/C/D). Ne choisit
    ni la règle de mise ni la mise -- toujours fournies par l'appelant.

    Retourne (tableau_comparatif, stabilite_par_strategie) où
    tableau_comparatif a une ligne par stratégie (compute_metrics) et
    stabilite_par_strategie associe à chaque strategie_id le résultat de
    compute_stabilite_par_sous_periode() pour cette stratégie."""
    resultats_par_strategie: dict[str, dict] = {}
    stabilite_par_strategie: dict[str, pd.DataFrame] = {}
    for strategie_id, fn in SELECTION_STRATEGIES.items():
        bets = simuler_strategie(contexts, fn, rule, mise, resultats)
        resultats_par_strategie[strategie_id] = compute_metrics(bets)
        stabilite_par_strategie[strategie_id] = compute_stabilite_par_sous_periode(
            bets, n_periodes=n_periodes)
    tableau_comparatif = compare_to_baselines(resultats_par_strategie)
    return tableau_comparatif, stabilite_par_strategie

# ---------------------------------------------------------------------------
# 10. Point d'entrée — verrouillé
# ---------------------------------------------------------------------------

def run_backtest(date_range: tuple[str, str], strategie_id: str, regle_mise_id: str) -> pd.DataFrame:
    """Verrouillé quel que soit l'état des registres. Depuis le 05/09/2026,
    SELECTION_STRATEGIES contient 6 entrées (top1_modele,
    top1_marche_h15_dans_top5_vert, architecture_a_confirmation,
    architecture_b_arbitre, architecture_c_score_combine,
    architecture_d_filtrage_incertitude) pour permettre la comparaison
    complète hors de run_backtest (voir compare_toutes_strategies, à
    appeler directement une fois le volume suffisant). BETTING_RULES reste
    vide : c'est la condition bloquante principale ici. Ce verrou est
    INDÉPENDANT du contenu des registres -- ne pas le lever sans décision
    explicite de Dorian (volume ET règle de mise globale)."""
    if strategie_id not in SELECTION_STRATEGIES or regle_mise_id not in BETTING_RULES:
        raise RuntimeError(
            "Backtest désactivé : pas de règle de mise enregistrée dans "
            "BETTING_RULES (et/ou stratégie de sélection inconnue). Ne pas "
            "en ajouter sans décision explicite de Dorian, et ne pas exécuter "
            "avant volume de données suffisant."
        )
    raise RuntimeError("Backtest désactivé pour l'instant, quelle que soit la stratégie demandée.")

# ---------------------------------------------------------------------------
# Auto-test sur données SYNTHÉTIQUES (pas de données de courses réelles) --
# vérifie juste l'absence de bug dans le code ci-dessus. Ne pas interpréter
# les chiffres produits ici comme un résultat de performance.
# ---------------------------------------------------------------------------

def _selftest() -> None:
    scores = pd.DataFrame({
        "course_id": ["C1"] * 4 + ["C2"] * 4 + ["C3"] * 4,
        "numero": [1, 2, 3, 4] * 3,
        "position_arrivee": [1, 2, 3, 4, 2, 1, 3, 4, 1, 3, 2, 4],
        "est_gagnant": [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        "cible_place": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        "rang_modele": [1, 2, 3, 4] * 3,
        "score_modele": [3.0, 1.0, 0.2, -1.0, 0.6, 0.5, 0.4, 0.3, 2.0, 1.9, 0.1, -0.5],
        "categorie_particularite": (
            ["", "", "", "", "HANDICAP", "HANDICAP", "HANDICAP", "HANDICAP", "", "", "", ""]
        ),
    })

    vert = filter_vert(scores)
    assert "somme_top3_proba" in vert.columns
    assert set(vert["course_id"].unique()) <= {"C1", "C2", "C3"}
    # C1 a un score#1 tres detache -> somme_top3_proba haute -> VERT attendu
    assert "C1" in set(vert["course_id"].unique()), "C1 devrait être VERT (favori très détaché)"

    top5 = rank_top5(vert)
    assert (top5.groupby("course_id").size() <= 5).all()

    cotes_brutes = pd.DataFrame({
        "course_id": ["C1", "C1", "C1", "C1", "C2", "C2", "C3", "C3", "C3", "C3"],
        "numero": [1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
        "minutes_avant_depart": [20, 10, 20, 5, 30, 12, 20, 10, 20, 10],
        "cote": [2.5, 2.2, 5.0, 4.8, 3.0, 2.8, 1.8, 1.7, 1.85, 1.82],
    })
    pit = compute_point_in_time_odds(cotes_brutes, window_minutes=15)
    # pour (C1,1) : minutes>=15 -> seule la ligne a 20 qualifie -> cote 2.5 attendue
    row = pit[(pit["course_id"] == "C1") & (pit["numero"] == 1)]
    assert not row.empty and float(row["cote"].iloc[0]) == 2.5
    # aucune ligne (C2,1) : minutes 30 et 12 -> seule 30 qualifie (>=15) -> cote 3.0
    row2 = pit[(pit["course_id"] == "C2") & (pit["numero"] == 1)]
    assert not row2.empty and float(row2["cote"].iloc[0]) == 3.0

    resultats = scores[["course_id", "numero", "position_arrivee", "est_gagnant"]]
    cov = check_couverture(pit, resultats, seuil_pct=70.0)
    assert "couverture_pct" in cov.columns

    top5 = top5.merge(pit.rename(columns={"cote": "cote_h15"})[["course_id", "numero", "cote_h15"]],
                       on=["course_id", "numero"], how="left")

    # Nouveau (05/09/2026) : variables de marché (chantier 1) + 4 architectures (chantier 3).
    top5 = add_market_features(top5)
    for col in ("rang_cote_h15", "favori_h15", "proba_modele_top5", "proba_implicite_marche",
                "ecart_cote_top5_abs", "ecart_cote_top5_relatif", "dispersion_cotes_top5",
                "accord_desaccord_modele_marche"):
        assert col in top5.columns, f"colonne marché manquante : {col}"
    # C3 : deux cotes valides (1, 2), écart faible attendu (1.8 vs 1.85 environ)
    c3 = top5[top5["course_id"] == "C3"]
    assert c3["ecart_cote_top5_relatif"].notna().any()

    def regle_cote_min_1_5(cote: float) -> bool:
        return cote >= 1.5

    dates = {"C1": "2026-08-30", "C2": "2026-08-31", "C3": "2026-09-01"}
    contexts = []
    for course_id, grp in top5.groupby("course_id"):
        contexts.append(RaceContext(
            course_id=course_id, top5=grp,
            faible_confiance=False,
            handicap=bool((grp["categorie_particularite"] == "HANDICAP").any()),
            date=dates[course_id],
        ))

    bets = simuler_strategie(contexts, top1_modele_strategy, regle_cote_min_1_5, mise=10.0, resultats=resultats)
    m = compute_metrics(bets)
    assert m["n_courses_evaluees"] == bets.shape[0]
    assert "cote_moyenne" in m and "taux_courses_jouees_pct" in m

    assert set(SELECTION_STRATEGIES) == {
        "top1_modele", "top1_marche_h15_dans_top5_vert",
        "architecture_a_confirmation", "architecture_b_arbitre",
        "architecture_c_score_combine", "architecture_d_filtrage_incertitude",
    }
    bets_marche = simuler_strategie(
        contexts, top1_marche_h15_dans_top5_vert_strategy, regle_cote_min_1_5, mise=10.0, resultats=resultats)
    assert bets_marche.shape[0] == bets.shape[0]

    comparaison = compare_modele_seul_vs_marche_h15(contexts, regle_cote_min_1_5, mise=10.0, resultats=resultats)
    assert set(comparaison["strategie_id"]) == {"top1_modele", "top1_marche_h15_dans_top5_vert"}

    # Nouveau (05/09/2026) : les 4 architectures ne doivent jamais planter,
    # même sur des contextes sans cote (None attendu proprement, pas d'exception).
    for strat_id in ("architecture_a_confirmation", "architecture_b_arbitre",
                      "architecture_c_score_combine", "architecture_d_filtrage_incertitude"):
        fn = SELECTION_STRATEGIES[strat_id]
        bets_archi = simuler_strategie(contexts, fn, regle_cote_min_1_5, mise=10.0, resultats=resultats)
        assert bets_archi.shape[0] == len(contexts)
        _ = compute_metrics(bets_archi)  # ne doit jamais lever

    tableau_complet, stabilite = compare_toutes_strategies(
        contexts, regle_cote_min_1_5, mise=10.0, resultats=resultats, n_periodes=2)
    assert set(tableau_complet["strategie_id"]) == set(SELECTION_STRATEGIES)
    assert set(stabilite) == set(SELECTION_STRATEGIES)

    rendement = compute_rendement_cumule(bets_marche)
    if not rendement.empty:
        assert list(rendement["n_paris_cumule"]) == list(range(1, len(rendement) + 1))
        assert (rendement["mise_cumulee"].diff().dropna() >= 0).all()

    stab_modele = compute_stabilite_par_sous_periode(bets, n_periodes=2)
    assert "periode" in stab_modele.columns

    print("Auto-test piste4_backtest_engine.py : OK (données synthétiques, aucune donnée de course réelle).")
    print("Métriques top1_modele (synthétiques, sans signification) :", m)
    print("Comparaison modele seul vs marche H-15 dans Top-5 VERT (synthétique) :")
    print(comparaison.to_string(index=False))
    print("Comparaison complète des 6 stratégies (synthétique, chantier 3) :")
    print(tableau_complet.to_string(index=False))
    print("Rendement cumulé (synthétique) :")
    print(rendement.to_string(index=False))
    print("Stabilité par sous-période, top1_modele (synthétique) :")
    print(stab_modele.to_string(index=False))

if __name__ == "__main__":
    _selftest()
