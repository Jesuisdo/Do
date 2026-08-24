# -*- coding: utf-8 -*-
"""
entrainer_et_evaluer_v3.py — Analyse d'erreurs du modele GBM v2 (47,1% en
multi-pick mais seulement 24,3% de gagnants trouves sur le meilleur pick) et
test d'ameliorations CIBLEES pour ameliorer specifiquement le taux de
gagnants du meilleur pick. Feu vert donne le 24/08/2026, suite au premier
rapport v2 du 23/08/2026.

Ce script NE part PAS d'un nouveau catalogue de variables. Il :
  1. Reconstruit EXACTEMENT le meme pipeline que v2 (memes 109 variables,
     meme resolution d'identite, meme decoupage chronologique strict
     train/validation/test) pour obtenir les predictions ligne-par-ligne du
     modele v2 (non sauvegardees lors du run precedent) et pouvoir analyser
     ses erreurs precisement.
  2. Analyse les erreurs du modele v2 : rang donne par le modele au vrai
     gagnant (top1/2/3/5/au-dela), ecart de probabilite entre le gagnant
     reel et le pick choisi, profils compares des gagnants trouves vs
     rates, performance du pick gagnant par segment (partants, handicap,
     distance, terrain).
  3. Teste DEUX ameliorations ciblees, choisies a partir des variables deja
     identifiees par v2 comme porteuses de signal (voir permutation
     importance du rapport v2) et des categories explicitement demandees
     (ecart d'allocation vs carriere, musique, repos, corde, interactions
     jockey/entraineur) :
       a) Variables "relatives au champ" (rang et z-score intra-course) pour
          12 variables cibles deja a fort signal — PAS un nouveau catalogue
          de 300 variables, un enrichissement cible de 24 colonnes.
       b) Un modele GBM entraine directement sur la cible "gagnant" (au lieu
          de la cible "place" utilisee jusqu'ici) pour le classement du
          meilleur pick — hypothese : optimiser directement pour "qui
          gagne" plutot que pour "qui est place" devrait mieux servir la
          metrique qui interesse (taux de gagnants du meilleur pick).
  4. Conserve EXACTEMENT le meme protocole hors echantillon que v2 (meme
     split chronologique, meme jeu de TEST jamais touche avant le rapport
     final) pour permettre une comparaison directe et honnete.

Aucune cote, aucune donnee de marche. Aucune donnee posterieure a la course
ne peut influencer sa prediction (meme garantie point-in-time que v2).
"""
import os
import sys
import json
import gc
import random
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

try:
    import psycopg2
    import psycopg2.extras
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.inspection import permutation_importance
    DEPENDANCES_LOURDES_DISPONIBLES = True
except ImportError:
    # Meme remarque que v2 : psycopg2/scikit-learn ne sont pas installables
    # dans l'environnement de developpement local de ce projet (proxy
    # sortant restreint) — ce script est concu pour tourner via GitHub
    # Actions. En local, seule la logique pandas pure est testable — voir
    # test_entrainer_et_evaluer_v3.py.
    DEPENDANCES_LOURDES_DISPONIBLES = False

from identite_chevaux import resoudre_identite_chevaux
from variables_historiques import construire_variables, trier_chronologiquement
from variables_config import VARIABLES_NUMERIQUES, VARIABLES_CATEGORIELLES

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 300)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATABASE_URL = os.environ.get("DATABASE_URL")
CAP_CARDINALITE_CATEGORIELLE = 20
SOUS_ECHANTILLON_PERMUTATION = 30_000    # reduit vs v2 (60000) : le run v2 a
                                          # montre que la permutation etait le
                                          # poste le plus couteux en temps ;
                                          # ce run-ci entraine 3 modeles GBM
                                          # au lieu d'1, budget de temps
                                          # reaffecte en consequence.

# --- meilleurs hyperparametres GBM retenus par v2 (deja choisis sur
# VALIDATION lors du run precedent) : reutilises tels quels pour reproduire
# le modele v2 sans repeter la recherche de grille, afin d'obtenir ses
# predictions ligne par ligne (non sauvegardees a l'epoque) necessaires a
# l'analyse d'erreurs ci-dessous. ---
MEILLEURS_PARAMS_GBM_V2 = {
    "max_depth": 5, "max_iter": 200, "learning_rate": 0.05,
    "l2_regularization": 1.0, "min_samples_leaf": 40,
}

# --- variables cibles pour l'enrichissement "relatif au champ" (v3) ---
# Choisies a partir de deux sources convergentes : (a) le top de
# l'importance par permutation du rapport v2 (allocation_delta_vs_carriere,
# musique_dernier, jours_repos, place_corde, interne_entraineur_taux_top3,
# musique_moy3, interne_jockey_taux_top3, gains_annee_encours,
# montant_allocation, carriere_taux_place) et (b) les categories
# explicitement demandees (ecart d'allocation vs carriere, musique, repos,
# corde, interactions jockey/entraineur — completees ici par
# jockey/cheval et entraineur/cheval). Pour chacune, on ajoute son RANG et
# son Z-SCORE intra-course (calcules uniquement a partir de valeurs deja
# connues avant la course, comme "niveau_moyen_adversaires" et
# "rang_papier_taux_victoire" le faisaient deja dans v2 — aucune fuite).
# 12 variables x 2 = 24 nouvelles colonnes : un enrichissement CIBLE, pas un
# nouveau catalogue de variables.
VARIABLES_RELATIVES_CIBLES = [
    "allocation_delta_vs_carriere",
    "montant_allocation",
    "musique_dernier",
    "musique_moy3",
    "jours_repos",
    "place_corde",
    "interne_jockey_entraineur_taux_victoire",
    "interne_jockey_taux_top3",
    "interne_entraineur_taux_top3",
    "interne_jockey_cheval_taux_victoire",
    "interne_entraineur_cheval_taux_victoire",
    "carriere_taux_place",
]
VARIABLES_NUMERIQUES_V3 = (
    VARIABLES_NUMERIQUES
    + [f"{v}_rang_course" for v in VARIABLES_RELATIVES_CIBLES]
    + [f"{v}_z_course" for v in VARIABLES_RELATIVES_CIBLES]
)

# --- variables utilisees pour comparer les profils "gagnant trouve" vs
# "gagnant rate" (melange de variables brutes et de leurs versions
# relatives, pour voir si la version relative discrimine mieux) ---
VARIABLES_PROFIL_GAGNANTS = [
    "allocation_delta_vs_carriere", "allocation_delta_vs_carriere_rang_course",
    "musique_dernier", "musique_dernier_rang_course",
    "musique_moy3", "rang_papier_taux_victoire", "niveau_moyen_adversaires",
    "jours_repos", "jours_repos_rang_course",
    "place_corde", "place_corde_rang_course",
    "interne_jockey_entraineur_taux_victoire", "interne_jockey_taux_top3",
    "interne_entraineur_taux_top3", "carriere_taux_victoire",
    "carriere_taux_place", "age", "nb_partants_reel", "gains_annee_encours",
]

REQUETE = """
SELECT
    rp.course_id, rc.date_course, rc.heure_depart, rc.hippodrome,
    rc.distance_m, rc.montant_allocation, rc.meteo_temperature,
    rc.meteo_force_vent, rc.terrain_intitule, rc.terrain_valeur_penetrometre,
    rc.corde, rc.type_piste, rc.categorie_particularite, rc.condition_age,
    rc.condition_sexe, rc.partants_declares, rc.specialite, rc.discipline,
    rp.numero, rp.nom_cheval, rp.id_cheval, rp.nom_pere, rp.nom_mere,
    rp.sexe, rp.age, rp.nom_jockey, rp.nom_entraineur, rp.musique,
    rp.gains, rp.gains_annee_encours, rp.gains_annee_precedente,
    rp.nombre_courses, rp.nombre_victoires, rp.nombre_places,
    rp.handicap_poids, rp.poids_condition_monte, rp.place_corde,
    rp.oeilleres, rp.deferre, rp.position_arrivee, rp.race, rp.pays,
    rp.pays_entrainement, rp.proprietaire, rp.eleveur, rp.robe
FROM resultats_partants rp
JOIN resultats_courses rc ON rc.course_id = rp.course_id
WHERE (rc.specialite = 'PLAT' OR rc.discipline = 'PLAT')
ORDER BY rc.date_course ASC, COALESCE(rc.heure_depart,'00:00:00') ASC, rp.course_id ASC, rp.numero ASC;
"""


def log(msg):
    print(msg, flush=True)


def charger_donnees_brutes():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL absente de l'environnement.")
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(REQUETE)
        lignes = [dict(r) for r in cur.fetchall()]
    conn.close()
    for l in lignes:
        if isinstance(l.get("date_course"), str):
            l["date_course"] = datetime.strptime(l["date_course"], "%Y-%m-%d").date()
    return lignes


def bucket_partants(n):
    if n <= 7:
        return "petit (<=7)"
    if n <= 12:
        return "moyen (8-12)"
    return "grand (13+)"


def calculer_baseline_combine_v1(df):
    """Identique a v2 : reproduit la regle de production pour comparaison."""
    d = df.copy()
    d["forme_norm"] = d["musique_dernier"].fillna(99)
    d["rang_forme"] = d.groupby("course_id")["forme_norm"].rank(method="min", ascending=True)
    d["rang_taux_victoire"] = d.groupby("course_id")["carriere_taux_victoire"].rank(method="min", ascending=False, na_option="bottom")
    d["rang_gains"] = d.groupby("course_id")["gains_carriere"].rank(method="min", ascending=False, na_option="bottom")
    jvp = d["interne_jockey_taux_victoire"].fillna(-1)
    d["rang_jockey"] = jvp.groupby(d["course_id"]).rank(method="min", ascending=False)
    d["score4"] = d["rang_forme"] + d["rang_taux_victoire"] + d["rang_gains"] + d["rang_jockey"]
    d["rang_predit"] = d.groupby("course_id")["score4"].rank(method="min", ascending=True)
    return d["rang_predit"]


def taux_reussite_place(df, colonne_rang_predit):
    d = df.copy()
    d["est_pick"] = d[colonne_rang_predit] <= d["seuil"]
    d["est_reussi"] = d["est_pick"] & (d["position_arrivee"] <= d["seuil"])
    essais = int(d["est_pick"].sum())
    reussis = int(d["est_reussi"].sum())
    pct = round(100 * reussis / essais, 1) if essais else float("nan")
    return essais, reussis, pct


def taux_reussite_top1(df, colonne_rang_predit, cible_col):
    d = df[df[colonne_rang_predit] == 1].copy()
    n_courses = len(d)
    n_reussis = int((d[cible_col] == 1).sum()) if cible_col == "est_gagnant" else int((d["position_arrivee"] <= d["seuil"]).sum())
    pct = round(100 * n_reussis / n_courses, 1) if n_courses else float("nan")
    return n_courses, n_reussis, pct


def preparer_matrice(df, variables_numeriques, colonnes_dummies_reference=None):
    """Version generalisee de preparer_matrice (v2) : accepte la liste de
    variables numeriques en parametre, pour pouvoir construire soit la
    matrice v2 (109 variables), soit la matrice v3 enrichie (109+24), a
    partir du meme DataFrame source."""
    cat_capee = {}
    for col in VARIABLES_CATEGORIELLES:
        valeurs = df[col].fillna("INCONNU").astype(str)
        if colonnes_dummies_reference is None:
            top = valeurs.value_counts().head(CAP_CARDINALITE_CATEGORIELLE).index
        else:
            top = None
        cat_capee[col] = valeurs if top is None else valeurs.where(valeurs.isin(top), "AUTRE")
    cat_df = pd.concat(
        [pd.get_dummies(cat_capee[col], prefix=col) for col in VARIABLES_CATEGORIELLES], axis=1
    )
    num_df = df[variables_numeriques].reset_index(drop=True).astype("float32")
    X = pd.concat([num_df, cat_df.reset_index(drop=True)], axis=1)
    if colonnes_dummies_reference is not None:
        X = X.reindex(columns=colonnes_dummies_reference, fill_value=0)
    return X


def ajouter_variables_relatives(df, variables_cibles):
    """Ajoute, pour chaque variable de `variables_cibles`, son RANG (1 =
    valeur la plus faible du champ) et son Z-SCORE intra-course. Ne calcule
    rien a partir du resultat de la course : uniquement a partir de valeurs
    deja point-in-time (memes garanties que niveau_moyen_adversaires /
    rang_papier_taux_victoire dans variables_historiques.py)."""
    df = df.copy()
    g = df.groupby("course_id")
    for var in variables_cibles:
        rang_col = f"{var}_rang_course"
        z_col = f"{var}_z_course"
        df[rang_col] = g[var].rank(method="min", ascending=True)
        moyenne = g[var].transform("mean")
        ecart_type = g[var].transform("std")
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (df[var] - moyenne) / ecart_type
        df[z_col] = z.replace([np.inf, -np.inf], np.nan)
    return df


def rang_distribution_gagnant(df_test, rang_col):
    """Pour chaque course, quel rang le modele a-t-il donne au VRAI
    gagnant ? Retourne (stats: dict, gagnants: DataFrame une ligne/course)."""
    gagnants = df_test[df_test["position_arrivee"] == 1].copy()
    gagnants["rang_du_gagnant"] = gagnants[rang_col]
    n = len(gagnants)
    top1 = int((gagnants["rang_du_gagnant"] == 1).sum())
    top2 = int((gagnants["rang_du_gagnant"] == 2).sum())
    top3 = int((gagnants["rang_du_gagnant"] == 3).sum())
    top4_5 = int(gagnants["rang_du_gagnant"].between(4, 5).sum())
    au_dela = int((gagnants["rang_du_gagnant"] > 5).sum())
    stats = {
        "n_courses": n,
        "top1": top1, "top1_pct": round(100 * top1 / n, 1) if n else float("nan"),
        "cumul_top2": top1 + top2, "cumul_top2_pct": round(100 * (top1 + top2) / n, 1) if n else float("nan"),
        "cumul_top3": top1 + top2 + top3, "cumul_top3_pct": round(100 * (top1 + top2 + top3) / n, 1) if n else float("nan"),
        "cumul_top5": top1 + top2 + top3 + top4_5, "cumul_top5_pct": round(100 * (top1 + top2 + top3 + top4_5) / n, 1) if n else float("nan"),
        "au_dela_de_5": au_dela, "au_dela_de_5_pct": round(100 * au_dela / n, 1) if n else float("nan"),
    }
    return stats, gagnants


def ecart_probabilite_gagnant(gagnants, df_test, rang_col, proba_col):
    """Ecart entre la probabilite du pick choisi par le modele (rang==1) et
    celle attribuee au vrai gagnant, dans les courses ou le gagnant n'est
    PAS le pick #1. Retourne (stats: dict, gagnants enrichi)."""
    picks = df_test[df_test[rang_col] == 1][["course_id", proba_col]].rename(columns={proba_col: "proba_pick"})
    g = gagnants.merge(picks, on="course_id", how="left")
    g["ecart_proba"] = g["proba_pick"] - g[proba_col]
    rates = g[g["rang_du_gagnant"] > 1]
    n = len(rates)
    stats = {
        "n_rates": n,
        "moyenne": round(float(rates["ecart_proba"].mean()), 4) if n else float("nan"),
        "mediane": round(float(rates["ecart_proba"].median()), 4) if n else float("nan"),
        "p90": round(float(rates["ecart_proba"].quantile(0.9)), 4) if n else float("nan"),
        "quasi_trouve_le_0_02": int((rates["ecart_proba"] <= 0.02).sum()) if n else 0,
        "quasi_trouve_pct": round(100 * int((rates["ecart_proba"] <= 0.02).sum()) / n, 1) if n else float("nan"),
    }
    return stats, g


def profils_gagnants(gagnants, variables):
    """Compare gagnants 'trouves' (rang_du_gagnant==1) vs 'rates' (>1) sur
    chaque variable. Retourne une liste triee par |d de Cohen| decroissant :
    [(var, moyenne_trouve, moyenne_rate, d_cohen), ...]."""
    trouve = gagnants[gagnants["rang_du_gagnant"] == 1]
    rate = gagnants[gagnants["rang_du_gagnant"] > 1]
    resultats = []
    for var in variables:
        if var not in gagnants.columns:
            continue
        m_t, m_r = trouve[var].mean(), rate[var].mean()
        s_t, s_r = trouve[var].std(), rate[var].std()
        n_t, n_r = int(trouve[var].notna().sum()), int(rate[var].notna().sum())
        if n_t > 1 and n_r > 1:
            pooled = np.sqrt(((n_t - 1) * s_t ** 2 + (n_r - 1) * s_r ** 2) / max(n_t + n_r - 2, 1))
            d = (m_t - m_r) / pooled if pooled and pooled > 0 else np.nan
        else:
            d = np.nan
        resultats.append((var, m_t, m_r, d))
    resultats.sort(key=lambda x: abs(x[3]) if pd.notna(x[3]) else -1, reverse=True)
    return resultats


def performance_segments_gagnant(df_test, rang_col):
    """Taux de reussite top-1 gagnant (methodologie B) par segment :
    nombre de partants, handicap ou non, distance, terrain. Retourne un
    dict de listes [(segment, n_courses, n_reussis, pct), ...]."""
    d = df_test.copy()
    d["groupe_partants"] = d["nb_partants_reel"].apply(bucket_partants)
    d["est_handicap"] = d["categorie_particularite"].fillna("").str.contains("HANDICAP")
    resultats = {"partants": [], "handicap": [], "distance": [], "terrain": []}
    for groupe, sous_df in d.groupby("groupe_partants"):
        n_c, n_r, pct = taux_reussite_top1(sous_df, rang_col, "est_gagnant")
        resultats["partants"].append((groupe, n_c, n_r, pct))
    for est_h, sous_df in d.groupby("est_handicap"):
        n_c, n_r, pct = taux_reussite_top1(sous_df, rang_col, "est_gagnant")
        resultats["handicap"].append(("HANDICAP" if est_h else "NON HANDICAP", n_c, n_r, pct))
    for dist_b, sous_df in d.groupby(d["distance_bucket"].fillna("INCONNU")):
        if sous_df["course_id"].nunique() < 100:
            continue
        n_c, n_r, pct = taux_reussite_top1(sous_df, rang_col, "est_gagnant")
        resultats["distance"].append((dist_b, n_c, n_r, pct))
    for terr_b, sous_df in d.groupby(d["terrain_bucket"].fillna("INCONNU")):
        if sous_df["course_id"].nunique() < 100:
            continue
        n_c, n_r, pct = taux_reussite_top1(sous_df, rang_col, "est_gagnant")
        resultats["terrain"].append((terr_b, n_c, n_r, pct))
    return resultats


def log_analyse_erreurs(df_test, rang_col, proba_col, label):
    """Enchaine et journalise l'analyse d'erreurs complete pour un modele
    donne (identifie par `label`). Retourne le DataFrame gagnants enrichi,
    au cas ou il faille le reutiliser."""
    stats_rang, gagnants = rang_distribution_gagnant(df_test, rang_col)
    log(f"\n-- Rang donne par le modele au gagnant reel, sur {stats_rang['n_courses']} courses ({label}) --")
    log(f"  Gagnant = pick #1 (trouve) : {stats_rang['top1']}/{stats_rang['n_courses']} = {stats_rang['top1_pct']}%")
    log(f"  Gagnant dans le top 2 : {stats_rang['cumul_top2']}/{stats_rang['n_courses']} = {stats_rang['cumul_top2_pct']}%")
    log(f"  Gagnant dans le top 3 : {stats_rang['cumul_top3']}/{stats_rang['n_courses']} = {stats_rang['cumul_top3_pct']}%")
    log(f"  Gagnant dans le top 5 : {stats_rang['cumul_top5']}/{stats_rang['n_courses']} = {stats_rang['cumul_top5_pct']}%")
    log(f"  Gagnant au-dela du top 5 : {stats_rang['au_dela_de_5']}/{stats_rang['n_courses']} = {stats_rang['au_dela_de_5_pct']}%")

    stats_ecart, gagnants = ecart_probabilite_gagnant(gagnants, df_test, rang_col, proba_col)
    log(f"\n  Ecart de probabilite (pick choisi moins gagnant reel), courses ou le gagnant n'est PAS le pick #1 (n={stats_ecart['n_rates']}) :")
    log(f"    Moyenne={stats_ecart['moyenne']}  Mediane={stats_ecart['mediane']}  P90={stats_ecart['p90']}")
    log(f"    Cas 'quasi-trouve' (ecart <= 0.02) : {stats_ecart['quasi_trouve_le_0_02']}/{stats_ecart['n_rates']} = "
        f"{stats_ecart['quasi_trouve_pct']}% — le modele hesitait entre 2 chevaux tres proches.")

    profils = profils_gagnants(gagnants, VARIABLES_PROFIL_GAGNANTS)
    log(f"\n  Profils compares : gagnants trouves vs gagnants rates ({label}) — tries par ecart standardise (d de Cohen) :")
    for var, m_t, m_r, d in profils:
        marqueur = "  <-- ecart notable" if pd.notna(d) and abs(d) >= 0.2 else ""
        if pd.notna(d):
            log(f"    {var:45s} trouve={m_t:.3f} rate={m_r:.3f} d={d:+.3f}{marqueur}")
        else:
            log(f"    {var:45s} trouve={m_t} rate={m_r} d=NA")

    segs = performance_segments_gagnant(df_test, rang_col)
    log(f"\n  Performance top-1 gagnant par segment ({label}) :")
    log("    -- Par nombre de partants --")
    for seg, n_c, n_r, pct in segs["partants"]:
        log(f"      {seg:15s} {n_r}/{n_c} = {pct}%")
    log("    -- Handicap vs non-handicap --")
    for seg, n_c, n_r, pct in segs["handicap"]:
        log(f"      {seg:15s} {n_r}/{n_c} = {pct}%")
    log("    -- Par distance --")
    for seg, n_c, n_r, pct in segs["distance"]:
        log(f"      {str(seg):15s} {n_r}/{n_c} = {pct}%")
    log("    -- Par terrain --")
    for seg, n_c, n_r, pct in segs["terrain"]:
        log(f"      {str(seg):15s} {n_r}/{n_c} = {pct}%")

    return stats_rang, stats_ecart, profils, segs


def entrainer_gbm_avec_grille(X_train, y_train, X_val, y_val, grille, label):
    """Recherche d'hyperparametres sur VALIDATION (identique au protocole
    v2). Retourne (meilleurs_params, meilleure_auc_val)."""
    meilleurs_params, meilleure_auc_val = None, -1
    for params in grille:
        m = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **params)
        m.fit(X_train, y_train)
        try:
            auc_val = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        except ValueError:
            auc_val = -1
        log(f"  [{label}] {params} -> AUC validation = {round(auc_val, 4)}")
        if auc_val > meilleure_auc_val:
            meilleure_auc_val, meilleurs_params = auc_val, params
        del m
    return meilleurs_params, meilleure_auc_val


def main():
    if not DEPENDANCES_LOURDES_DISPONIBLES:
        raise RuntimeError(
            "psycopg2 et/ou scikit-learn ne sont pas installes. Ce script doit tourner dans "
            "l'environnement GitHub Actions du workflow dedie, pas en local."
        )
    log("=" * 100)
    log("ANALYSE D'ERREURS DU MODELE v2 + AMELIORATIONS CIBLEES POUR LE TAUX DE GAGNANTS — 24/08/2026")
    log("=" * 100)

    log("\n[1/9] Chargement des donnees brutes depuis Supabase (PLAT, toutes lignes)...")
    lignes = charger_donnees_brutes()
    log(f"  {len(lignes)} lignes partant/course brutes chargees.")

    log("\n[2/9] Resolution d'identite des chevaux (identique a v2)...")
    horse_uids, rapport_identite = resoudre_identite_chevaux(lignes)
    for l, uid in zip(lignes, horse_uids):
        l["horse_uid"] = uid
    log(f"  {rapport_identite['n_chevaux_distincts_resolus']} chevaux distincts resolus.")

    log("\n[3/9] Construction des 109 variables point-in-time (identique a v2)...")
    lignes_triees = trier_chronologiquement(lignes)
    features = construire_variables(lignes_triees)
    df = pd.DataFrame(features)
    del lignes, lignes_triees, features, horse_uids
    gc.collect()

    df = df[df["position_arrivee"].notna()].copy()
    df = df[df["nb_partants_reel"] >= 3].reset_index(drop=True)
    df["est_gagnant"] = (df["position_arrivee"] == 1).astype(int)
    df["cible_place"] = (df["position_arrivee"] <= df["seuil"]).astype(int)
    log(f"  {len(df)} lignes, {df['course_id'].nunique()} courses apres filtrage.")

    log("\n[4/9] Ajout des 24 variables relatives au champ (rang + z-score intra-course, "
        f"sur les {len(VARIABLES_RELATIVES_CIBLES)} variables cibles a fort signal)...")
    df = ajouter_variables_relatives(df, VARIABLES_RELATIVES_CIBLES)

    # --- decoupage chronologique STRICT, identique a v2 : 70/15/15 ---
    df = df.sort_values(["date_course", "course_id"]).reset_index(drop=True)
    courses_ordre = df["course_id"].drop_duplicates().tolist()
    n = len(courses_ordre)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    courses_train = set(courses_ordre[:n_train])
    courses_val = set(courses_ordre[n_train:n_val])
    courses_test = set(courses_ordre[n_val:])
    df_train = df[df["course_id"].isin(courses_train)].reset_index(drop=True)
    df_val = df[df["course_id"].isin(courses_val)].reset_index(drop=True)
    df_test = df[df["course_id"].isin(courses_test)].reset_index(drop=True)
    del df
    gc.collect()

    log("\n[5/9] Decoupage chronologique strict (identique a v2, meme jeu de TEST) :")
    log(f"  TRAIN : {len(df_train)} lignes / {df_train['course_id'].nunique()} courses")
    log(f"  VALIDATION : {len(df_val)} lignes / {df_val['course_id'].nunique()} courses")
    log(f"  TEST : {len(df_test)} lignes / {df_test['course_id'].nunique()} courses "
        f"({df_test['date_course'].min()} -> {df_test['date_course'].max()})")

    y_train_place = df_train["cible_place"].values
    y_val_place = df_val["cible_place"].values
    y_test_place = df_test["cible_place"].values
    y_train_gagnant = df_train["est_gagnant"].values
    y_val_gagnant = df_val["est_gagnant"].values

    # --- matrice enrichie v3 (109 + 24 = 133 numeriques + categorielles) —
    # construite UNE SEULE FOIS ; la matrice v2 (198 colonnes) en est un
    # sous-ensemble par simple selection de colonnes (pas besoin de la
    # reconstruire depuis df, economie de temps/memoire). ---
    X_train_v3 = preparer_matrice(df_train, VARIABLES_NUMERIQUES_V3)
    colonnes_v3 = X_train_v3.columns
    X_val_v3 = preparer_matrice(df_val, VARIABLES_NUMERIQUES_V3, colonnes_dummies_reference=colonnes_v3)
    X_test_v3 = preparer_matrice(df_test, VARIABLES_NUMERIQUES_V3, colonnes_dummies_reference=colonnes_v3)

    colonnes_relatives = [f"{v}_rang_course" for v in VARIABLES_RELATIVES_CIBLES] + \
                         [f"{v}_z_course" for v in VARIABLES_RELATIVES_CIBLES]
    colonnes_v2 = [c for c in colonnes_v3 if c not in colonnes_relatives]
    log(f"\nMatrice v3 (enrichie) : {X_train_v3.shape[1]} colonnes. Matrice v2 (sous-ensemble) : {len(colonnes_v2)} colonnes.")

    del df_train, df_val
    gc.collect()

    # --- colonne temoin aleatoire, ajoutee une fois sur la matrice complete ---
    rng = np.random.RandomState(RANDOM_SEED)
    X_train_v3 = X_train_v3.copy()
    X_val_v3 = X_val_v3.copy()
    X_test_v3 = X_test_v3.copy()
    X_train_v3["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_train_v3))
    X_val_v3["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_val_v3))
    X_test_v3["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_test_v3))
    colonnes_v2_ctrl = colonnes_v2 + ["temoin_aleatoire"]

    # =========================================================================
    # [6/9] MODELE v2 (reproduit a l'identique) — pour disposer des
    # predictions ligne par ligne necessaires a l'analyse d'erreurs.
    # =========================================================================
    log("\n[6/9] Reproduction du modele v2 (109 variables, hyperparametres deja choisis "
        f"sur validation lors du run precedent : {MEILLEURS_PARAMS_GBM_V2})...")
    gbm_v2 = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **MEILLEURS_PARAMS_GBM_V2)
    gbm_v2.fit(X_train_v3[colonnes_v2_ctrl], y_train_place)
    try:
        auc_val_v2 = round(roc_auc_score(y_val_place, gbm_v2.predict_proba(X_val_v3[colonnes_v2_ctrl])[:, 1]), 4)
        log(f"  AUC validation (verification de coherence avec le rapport v2, attendu ~0.7026) : {auc_val_v2}")
    except ValueError:
        pass
    X_trainval_v2 = pd.concat([X_train_v3[colonnes_v2_ctrl], X_val_v3[colonnes_v2_ctrl]], axis=0).reset_index(drop=True)
    y_trainval_place = np.concatenate([y_train_place, y_val_place])
    gbm_v2_final = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **MEILLEURS_PARAMS_GBM_V2)
    gbm_v2_final.fit(X_trainval_v2, y_trainval_place)
    proba_v2_test = gbm_v2_final.predict_proba(X_test_v3[colonnes_v2_ctrl])[:, 1]
    del gbm_v2, X_trainval_v2
    gc.collect()

    df_test = df_test.reset_index(drop=True)
    df_test["proba_v2"] = proba_v2_test
    df_test["rang_v2"] = df_test.groupby("course_id")["proba_v2"].rank(method="min", ascending=False)

    log("\n" + "=" * 100)
    log("=== ANALYSE D'ERREURS DU MODELE v2 (rappel : 47,1% multi-pick / 24,3% gagnant top-1) ===")
    log("=" * 100)
    log_analyse_erreurs(df_test, "rang_v2", "proba_v2", "GBM v2 (cible=place)")

    # =========================================================================
    # [7/9] MODELE v3-place : memes variables enrichies (+24), meme cible
    # "place", recherche d'hyperparametres identique a v2.
    # =========================================================================
    log("\n[7/9] Entrainement GBM v3 sur variables enrichies, cible='place' (recherche d'hyperparametres)...")
    grille_gbm = [
        {"max_depth": 4, "max_iter": 150, "learning_rate": 0.05, "l2_regularization": 1.0, "min_samples_leaf": 25},
        {"max_depth": 5, "max_iter": 200, "learning_rate": 0.05, "l2_regularization": 1.0, "min_samples_leaf": 40},
        {"max_depth": 3, "max_iter": 250, "learning_rate": 0.03, "l2_regularization": 2.0, "min_samples_leaf": 60},
    ]
    params_place, auc_place = entrainer_gbm_avec_grille(
        X_train_v3, y_train_place, X_val_v3, y_val_place, grille_gbm, "v3-place")
    log(f"  Meilleurs hyperparametres (v3-place) : {params_place} (AUC validation={round(auc_place,4)})")
    X_trainval_v3 = pd.concat([X_train_v3, X_val_v3], axis=0).reset_index(drop=True)
    y_trainval_place_v3 = np.concatenate([y_train_place, y_val_place])
    gbm_v3_place = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **params_place)
    gbm_v3_place.fit(X_trainval_v3, y_trainval_place_v3)
    proba_v3_place_test = gbm_v3_place.predict_proba(X_test_v3)[:, 1]
    df_test["proba_v3_place"] = proba_v3_place_test
    df_test["rang_v3_place"] = df_test.groupby("course_id")["proba_v3_place"].rank(method="min", ascending=False)

    log("\n" + "=" * 100)
    log("=== ANALYSE D'ERREURS DU MODELE v3-place (variables enrichies, cible='place') ===")
    log("=" * 100)
    log_analyse_erreurs(df_test, "rang_v3_place", "proba_v3_place", "GBM v3-place")

    # --- importance par permutation : les 24 nouvelles variables ont-elles
    # un vrai signal, ou sont-elles sous le bruit ? ---
    log("\nCalcul de l'importance par permutation (GBM v3-place, sous-echantillon de TEST)...")
    idx_perm = X_test_v3.sample(min(len(X_test_v3), SOUS_ECHANTILLON_PERMUTATION), random_state=RANDOM_SEED).index
    perm = permutation_importance(
        gbm_v3_place, X_test_v3.loc[idx_perm], y_test_place[idx_perm],
        scoring="roc_auc", n_repeats=3, random_state=RANDOM_SEED, n_jobs=1,
    )
    importances = pd.Series(perm.importances_mean, index=X_test_v3.columns).sort_values(ascending=False)
    seuil_bruit = importances.get("temoin_aleatoire", 0.0)
    log(f"  Bruit de reference (temoin aleatoire) = {round(seuil_bruit, 5)}")
    log("\n  Les 24 nouvelles variables relatives, classees par importance de permutation :")
    for var in colonnes_relatives:
        if var in importances.index:
            imp = importances[var]
            marqueur = "  <-- AU-DESSUS du bruit" if imp > seuil_bruit else "  <-- sous le bruit"
            log(f"    {var:50s} importance={imp:.5f}{marqueur}")
    n_relatives_utiles = sum(1 for v in colonnes_relatives if v in importances.index and importances[v] > seuil_bruit)
    log(f"\n  Bilan : {n_relatives_utiles}/{len(colonnes_relatives)} des nouvelles variables relatives "
        f"depassent le bruit de reference.")

    del X_trainval_v3, gbm_v3_place
    gc.collect()

    # =========================================================================
    # [8/9] MODELE v3-gagnant : memes variables enrichies, mais entraine
    # DIRECTEMENT sur la cible "gagnant" — hypothese testee : optimiser pour
    # "qui gagne" plutot que pour "qui est place" sert mieux le taux de
    # gagnants du meilleur pick.
    # =========================================================================
    log("\n[8/9] Entrainement GBM v3 sur variables enrichies, cible='gagnant' (recherche d'hyperparametres)...")
    params_gagnant, auc_gagnant = entrainer_gbm_avec_grille(
        X_train_v3, y_train_gagnant, X_val_v3, y_val_gagnant, grille_gbm, "v3-gagnant")
    log(f"  Meilleurs hyperparametres (v3-gagnant) : {params_gagnant} (AUC validation={round(auc_gagnant,4)})")
    X_trainval_v3b = pd.concat([X_train_v3, X_val_v3], axis=0).reset_index(drop=True)
    y_trainval_gagnant = np.concatenate([y_train_gagnant, y_val_gagnant])
    gbm_v3_gagnant = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **params_gagnant)
    gbm_v3_gagnant.fit(X_trainval_v3b, y_trainval_gagnant)
    proba_v3_gagnant_test = gbm_v3_gagnant.predict_proba(X_test_v3)[:, 1]
    df_test["proba_v3_gagnant"] = proba_v3_gagnant_test
    df_test["rang_v3_gagnant"] = df_test.groupby("course_id")["proba_v3_gagnant"].rank(method="min", ascending=False)
    del X_trainval_v3b, gbm_v3_gagnant, X_train_v3, X_val_v3, X_test_v3
    gc.collect()

    log("\n" + "=" * 100)
    log("=== ANALYSE D'ERREURS DU MODELE v3-gagnant (variables enrichies, cible='gagnant') ===")
    log("=" * 100)
    log_analyse_erreurs(df_test, "rang_v3_gagnant", "proba_v3_gagnant", "GBM v3-gagnant")

    # =========================================================================
    # [9/9] COMPARAISON FINALE — memes 3 methodologies que v2, meme jeu TEST.
    # =========================================================================
    df_test["rang_predit_baseline"] = calculer_baseline_combine_v1(df_test)

    log("\n" + "=" * 100)
    log("=== COMPARAISON FINALE (meme protocole hors echantillon que v2) ===")
    log("=" * 100)

    modeles = [
        ("Baseline combine_v1", "rang_predit_baseline", "proba_v2"),  # proba non utilisee pour baseline
        ("GBM v2 (cible=place, 109 variables)", "rang_v2", "proba_v2"),
        ("GBM v3-place (cible=place, 133 variables)", "rang_v3_place", "proba_v3_place"),
        ("GBM v3-gagnant (cible=gagnant, 133 variables)", "rang_v3_gagnant", "proba_v3_gagnant"),
    ]

    log("\n-- Methodologie A : taux de reussite multi-picks (identique au calcul du 41,1% historique) --")
    resultats_A = {}
    for nom, rang_col, _ in modeles:
        essais, reussis, pct = taux_reussite_place(df_test, rang_col)
        resultats_A[nom] = pct
        log(f"  {nom:50s} {reussis}/{essais} = {pct}%")

    log("\n-- Methodologie B : le MEILLEUR pick du modele par course (top-1), GAGNANT — la metrique cible de ce run --")
    resultats_B = {}
    for nom, rang_col, _ in modeles:
        n_courses, n_reussis, pct = taux_reussite_top1(df_test, rang_col, "est_gagnant")
        resultats_B[nom] = pct
        log(f"  {nom:50s} {n_reussis}/{n_courses} = {pct}%")

    log("\n-- Methodologie C : le MEILLEUR pick du modele par course (top-1), PLACE --")
    resultats_C = {}
    for nom, rang_col, _ in modeles:
        n_courses, n_reussis, pct = taux_reussite_top1(df_test, rang_col, "cible_place")
        resultats_C[nom] = pct
        log(f"  {nom:50s} {n_reussis}/{n_courses} = {pct}%")

    try:
        for nom, _, proba_col in modeles[1:]:
            auc = round(roc_auc_score(y_test_place, df_test[proba_col]), 4)
            ll = round(log_loss(y_test_place, df_test[proba_col]), 4)
            log(f"  AUC/logloss (cible=place) {nom:50s} AUC={auc} logloss={ll}")
    except ValueError as e:
        log(f"  AUC/log-loss non calculables : {e}")

    log("\n" + "=" * 100)
    log("=== RESUME ===")
    log("=" * 100)
    log(f"Rappel v2 (rapport du 23/08/2026) : multi-pick 47,1% | top-1 gagnant 24,3% | top-1 place 54,2%")
    log(f"Ce run (memes lignes de TEST, recalcule pour l'analyse d'erreurs) :")
    for nom, _, _ in modeles:
        log(f"  {nom:50s} multi-pick={resultats_A[nom]}%  top1-gagnant={resultats_B[nom]}%  top1-place={resultats_C[nom]}%")
    meilleur = max(modeles[1:], key=lambda t: resultats_B[t[0]])
    log(f"\nMeilleur taux de gagnants du meilleur pick : {meilleur[0]} avec {resultats_B[meilleur[0]]}% "
        f"(vs {resultats_B['GBM v2 (cible=place, 109 variables)']}% pour v2).")
    log("Ce rapport est le resultat REEL, non ajuste. Objectif de ce run : ameliorer le taux de gagnants "
        "du meilleur pick, pas seulement la metrique multi-pick globale — voir Methodologie B ci-dessus.")


if __name__ == "__main__":
    main()
