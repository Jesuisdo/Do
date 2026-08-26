# -*- coding: utf-8 -*-
"""
v3_lib.py — Module partage entre entrainer_v3_phase1.py (etapes 1-6 :
chargement, construction des variables, split, reproduction du modele v2)
et entrainer_v3_phase2.py (etapes 7-9 : entrainement des modeles v3,
comparaison finale). Contient TOUTE la logique commune (constantes,
fonctions pandas pures, wrapper GBM) pour eviter la duplication et
permettre de tester la logique une seule fois.

Historique : entrainer_et_evaluer_v3.py (version monolithique) plantait a
l'etape 7/9 apres ~8 minutes de calcul reel (chargement + construction des
variables + reproduction du modele v2 + analyse d'erreurs v2, DEJA REUSSIS)
a cause d'une colonne degeneree (voir colonnes_degenerees ci-dessous). Pour
qu'une erreur future dans les etapes 7-9 n'oblige plus a refaire les etapes
1-6, le pipeline est desormais scinde en deux scripts relies par un
checkpoint (voir entrainer_v3_phase1.py / entrainer_v3_phase2.py et
analyse-erreurs-v3.yml : job phase1 -> artefact -> job phase2).
"""
import os
import random

import numpy as np
import pandas as pd

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_DISPONIBLE = True
except ImportError:
    # psycopg2 n'est necessaire qu'a la phase 1 (chargement Supabase). La
    # phase 2 et le test smoke n'en ont pas besoin et ne l'installent pas
    # (bug identifie le 24/08/2026 : un seul try/except combine psycopg2 ET
    # scikit-learn faisait que l'absence de psycopg2 seul empechait aussi le
    # binding de HistGradientBoostingClassifier dans ce module -> NameError
    # dans entrainer_gbm_avec_grille en phase 2, meme quand scikit-learn
    # etait bien installe. Les deux dependances sont maintenant isolees.)
    PSYCOPG2_DISPONIBLE = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.inspection import permutation_importance
    SKLEARN_DISPONIBLE = True
except ImportError:
    # scikit-learn n'est pas installable dans l'environnement de
    # developpement local de ce projet (proxy sortant restreint) â ce module
    # est concu pour tourner via GitHub Actions. En local, seule la logique
    # pandas pure est testable â voir test_v3_lib.py.
    SKLEARN_DISPONIBLE = False

# Conserve pour compatibilite : la phase 1 a besoin des DEUX dependances
# (chargement Supabase + reproduction du modele v2). La phase 2 et le test
# smoke ne dependent, eux, que de SKLEARN_DISPONIBLE (voir plus haut).
DEPENDANCES_LOURDES_DISPONIBLES = PSYCOPG2_DISPONIBLE and SKLEARN_DISPONIBLE

from datetime import datetime

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 300)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATABASE_URL = os.environ.get("DATABASE_URL")
CAP_CARDINALITE_CATEGORIELLE = 20
SOUS_ECHANTILLON_PERMUTATION = 30_000

# --- meilleurs hyperparametres GBM retenus par v2 (deja choisis sur
# VALIDATION lors du run precedent) : reutilises tels quels pour reproduire
# le modele v2 sans repeter la recherche de grille. ---
MEILLEURS_PARAMS_GBM_V2 = {
    "max_depth": 5, "max_iter": 200, "learning_rate": 0.05,
    "l2_regularization": 1.0, "min_samples_leaf": 40,
}

GRILLE_GBM = [
    {"max_depth": 4, "max_iter": 150, "learning_rate": 0.05, "l2_regularization": 1.0, "min_samples_leaf": 25},
    {"max_depth": 5, "max_iter": 200, "learning_rate": 0.05, "l2_regularization": 1.0, "min_samples_leaf": 40},
    {"max_depth": 3, "max_iter": 250, "learning_rate": 0.03, "l2_regularization": 2.0, "min_samples_leaf": 60},
]

# --- variables cibles pour l'enrichissement "relatif au champ" (v3) ---
# Choisies a partir de deux sources convergentes : (a) le top de
# l'importance par permutation du rapport v2 et (b) les categories
# explicitement demandees par Dorian (ecart d'allocation vs carriere,
# musique, repos, corde, interactions jockey/entraineur â completees ici
# par jockey/cheval et entraineur/cheval). Pour chacune, on ajoute son RANG
# et son Z-SCORE intra-course (calcules uniquement a partir de valeurs deja
# connues avant la course â aucune fuite).
VARIABLES_RELATIVES_CIBLES = [
    "allocation_delta_vs_carriere",
    # "montant_allocation" retire (bug identifie le 24/08/2026) : c'est le
    # montant total de la course, IDENTIQUE pour tous les partants d'une
    # meme course -> ecart-type intra-course nul -> z-score = NaN partout
    # et rang constant = 1 partout (colonne degeneree, faisait planter le
    # binning HistGradientBoosting). "allocation_delta_vs_carriere"
    # ci-dessus est la variable runner-level deja demandee par Dorian
    # (ecart vs carriere), donc rien n'est perdu.
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


def variables_numeriques_v3(variables_numeriques_base):
    return (
        list(variables_numeriques_base)
        + [f"{v}_rang_course" for v in VARIABLES_RELATIVES_CIBLES]
        + [f"{v}_z_course" for v in VARIABLES_RELATIVES_CIBLES]
    )


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


def preparer_matrice(df, variables_numeriques, variables_categorielles, colonnes_dummies_reference=None):
    """Version generalisee de preparer_matrice (v2) : accepte la liste de
    variables numeriques ET categorielles en parametre, pour pouvoir
    construire soit la matrice v2 (109 variables), soit la matrice v3
    enrichie (109+22), a partir du meme DataFrame source."""
    cat_capee = {}
    for col in variables_categorielles:
        valeurs = df[col].fillna("INCONNU").astype(str)
        if colonnes_dummies_reference is None:
            top = valeurs.value_counts().head(CAP_CARDINALITE_CATEGORIELLE).index
        else:
            top = None
        cat_capee[col] = valeurs if top is None else valeurs.where(valeurs.isin(top), "AUTRE")
    cat_df = pd.concat(
        [pd.get_dummies(cat_capee[col], prefix=col) for col in variables_categorielles], axis=1
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


def colonnes_degenerees(X_train, colonnes_candidates):
    """Filet de securite generique : une colonne avec < 2 valeurs distinctes
    non-manquantes (constante, ou entierement NaN) fait planter le binning
    de HistGradientBoosting (numpy.lib.stride_tricks.sliding_window_view
    exige au moins 2 valeurs). Controle sur TRAIN uniquement (pas de fuite
    train/val/test). Retourne la liste des colonnes a exclure."""
    return [c for c in colonnes_candidates if X_train[c].nunique(dropna=True) < 2]


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
    """Taux de reussite top-1 gagnant (methodologie B) par segment : nombre
    de partants, handicap ou non, distance, terrain. Retourne un dict de
    listes [(segment, n_courses, n_reussis, pct), ...]."""
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
    donne (identifie par `label`). Retourne les objets de stats bruts, au
    cas ou il faille les reutiliser."""
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
        f"{stats_ecart['quasi_trouve_pct']}% â le modele hesitait entre 2 chevaux tres proches.")

    profils = profils_gagnants(gagnants, VARIABLES_PROFIL_GAGNANTS)
    log(f"\n  Profils compares : gagnants trouves vs gagnants rates ({label}) â tries par ecart standardise (d de Cohen) :")
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


# =============================================================================
# ANALYSE D'ERREURS APPROFONDIE (demandee par Dorian le 25/08/2026, APRES le
# run production v3 qui a montre AUC en hausse mais top-1 gagnant stagnant a
# ~24,4%). But : comprendre POURQUOI le modele se trompe, PAS l'ameliorer.
# Aucune de ces fonctions ne selectionne de variable ni ne modifie un modele
# â lecture seule sur des predictions et des variables deja point-in-time.
# =============================================================================

# Variables demandees explicitement par Dorian pour le profil des erreurs :
# musique/forme, repos, corde, age, poids, niveau des adversaires, plus des
# proxys "information disponible avant la course" (carriere/historique
# interne) pour trancher entre probleme de modele et probleme de donnees.
VARIABLES_DIAGNOSTIC_ERREURS = [
    "musique_dernier", "musique_moy3", "musique_tendance",
    "jours_repos", "place_corde",
    "age", "handicap_poids", "poids_condition_monte", "poids_delta_vs_carriere",
    "niveau_moyen_adversaires", "ecart_vs_niveau_moyen_champ", "rang_papier_taux_victoire",
    "carriere_nb_courses", "carriere_taux_victoire", "carriere_taux_place", "gains_carriere",
    "forme_nb_courses_disponibles", "forme_moy_position_5", "forme_moy_position_10",
    "forme_ecart_type_position_5", "forme_tendance_5_vs_10",
    "distance_m", "terrain_valeur_penetrometre",
    "allocation_delta_vs_carriere",
]


def bucket_rang_3(rang):
    """Regroupement en 3 buckets utilise pour toute l'analyse d'erreurs :
    top1 (trouve), top2_a_5 (presque trouve), hors_top5 (rate largement)."""
    if pd.isna(rang):
        return "inconnu"
    if rang == 1:
        return "top1"
    if 2 <= rang <= 5:
        return "top2_a_5"
    return "hors_top5"


def ajouter_flags_diagnostic(df_test):
    """Ajoute au DataFrame les colonnes derivees necessaires a l'analyse
    d'erreurs (aucune n'utilise le resultat de la course) :
      - bucket_partants : petit/moyen/grand champ
      - est_handicap : booleen
      - est_debutant : peu/pas de courses en carriere officielle OU aucun
        historique interne exploitable (proxy "information manquante avant
        la course")
      - identite_possible_fragmentee : carriere officielle connue (>3
        courses) MAIS historique interne vide -> le lien horse_uid n'a
        probablement pas retrouve les courses precedentes de ce cheval
        (proxy "probleme d'identification", cf. identite_chevaux.py que ne
        renvoie qu'un rapport agrege, pas de flag par ligne)."""
    d = df_test.copy()
    d["bucket_partants"] = d["nb_partants_reel"].apply(bucket_partants)
    d["est_handicap"] = d["categorie_particularite"].fillna("").str.contains("HANDICAP")
    carriere = d["carriere_nb_courses"].fillna(0) if "carriere_nb_courses" in d.columns else pd.Series(0, index=d.index)
    forme_dispo = d["forme_nb_courses_disponibles"].fillna(0) if "forme_nb_courses_disponibles" in d.columns else pd.Series(0, index=d.index)
    d["est_debutant"] = (carriere <= 2) | (forme_dispo == 0)
    d["identite_possible_fragmentee"] = (carriere > 3) & (forme_dispo == 0)
    return d


def profils_par_bucket_rang(gagnants, variables):
    """Version a 3 groupes de profils_gagnants : compare top1 / top2_a_5 /
    hors_top5 (au lieu du seul trouve/rate binaire), avec un d de Cohen de
    chaque groupe vs top1 (le groupe de reference "ce qui caracterise un
    gagnant que le modele repere bien"). gagnants doit deja avoir
    'rang_du_gagnant' (voir rang_distribution_gagnant)."""
    g = gagnants.copy()
    g["bucket_rang"] = g["rang_du_gagnant"].apply(bucket_rang_3)
    resultats = []
    for var in variables:
        if var not in g.columns:
            continue
        stats_par_bucket = {}
        for b in ["top1", "top2_a_5", "hors_top5"]:
            sous = g.loc[g["bucket_rang"] == b, var]
            stats_par_bucket[b] = (sous.mean(), sous.std(), int(sous.notna().sum()))
        row = {"variable": var}
        for b in ["top1", "top2_a_5", "hors_top5"]:
            m, _, n = stats_par_bucket[b]
            row[f"{b}_moyenne"] = round(float(m), 3) if pd.notna(m) else None
            row[f"{b}_n"] = n
        m1, s1, n1 = stats_par_bucket["top1"]
        for b in ["top2_a_5", "hors_top5"]:
            mb, sb, nb = stats_par_bucket[b]
            if n1 > 1 and nb > 1 and pd.notna(m1) and pd.notna(mb) and pd.notna(s1) and pd.notna(sb):
                pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (nb - 1) * sb ** 2) / max(n1 + nb - 2, 1))
                d = (m1 - mb) / pooled if pooled and pooled > 0 else np.nan
            else:
                d = np.nan
            row[f"d_top1_vs_{b}"] = round(float(d), 3) if pd.notna(d) else None
        resultats.append(row)
    out = pd.DataFrame(resultats)
    if not out.empty:
        out["abs_d_max"] = out[["d_top1_vs_top2_a_5", "d_top1_vs_hors_top5"]].abs().max(axis=1)
        out = out.sort_values("abs_d_max", ascending=False).drop(columns=["abs_d_max"])
    return out


def comparer_gagnant_vs_pick_modele(df_test, rang_col, proba_col, variables):
    """Pour les courses ou le pick #1 du modele n'est PAS le vrai gagnant,
    met en regard les valeurs de `variables` pour le gagnant reel et pour
    le cheval que le modele a prefere a tort. Repond directement a : 'quand
    le gagnant n'est pas trouve, qu'est-ce qui a fait pencher le modele
    vers un autre cheval ?'. Retourne (DataFrame comparatif, n_courses_ratees)."""
    colonnes = ["course_id"] + [v for v in variables if v in df_test.columns] + [proba_col]
    gagnants = df_test.loc[df_test["position_arrivee"] == 1, colonnes].copy()
    picks = df_test.loc[df_test[rang_col] == 1, colonnes].copy()
    fusion = gagnants.merge(picks, on="course_id", how="inner", suffixes=("_gagnant", "_pick_modele"))
    meme_cheval = fusion[f"{proba_col}_gagnant"] == fusion[f"{proba_col}_pick_modele"]
    rates = fusion[~meme_cheval]
    n = len(rates)
    resultats = []
    for var in variables:
        col_g, col_p = f"{var}_gagnant", f"{var}_pick_modele"
        if col_g not in rates.columns or col_p not in rates.columns:
            continue
        m_g, m_p = rates[col_g].mean(), rates[col_p].mean()
        resultats.append({
            "variable": var,
            "moyenne_vrai_gagnant": round(float(m_g), 3) if pd.notna(m_g) else None,
            "moyenne_pick_modele_a_tort": round(float(m_p), 3) if pd.notna(m_p) else None,
            "ecart": round(float(m_p - m_g), 3) if pd.notna(m_g) and pd.notna(m_p) else None,
        })
    out = pd.DataFrame(resultats)
    if not out.empty:
        out = out.reindex(out["ecart"].abs().sort_values(ascending=False, na_position="last").index)
    return out, n


def distribution_ecart_probabilite_buckets(gagnants_avec_ecart):
    """Repartit les courses ratees (gagnant pas pick #1) en tranches
    d'ecart de probabilite (pick_modele - gagnant), globalement, avec pour
    chaque tranche la probabilite moyenne attribuee au pick #1 fautif (une
    proba de pick eleve = modele confiant a tort = plutot un probleme de
    modele ; une proba de pick faible et diffuse = plusieurs chevaux
    proches = plutot une limite d'information)."""
    rates = gagnants_avec_ecart[gagnants_avec_ecart["rang_du_gagnant"] > 1].copy()
    n = len(rates)
    if n == 0:
        return []
    bornes = [0, 0.01, 0.05, 0.10, 0.20, 0.40, 1.01]
    labels = ["<=0.01", "0.01-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", ">0.40"]
    rates["tranche"] = pd.cut(rates["ecart_proba"], bins=bornes, labels=labels, include_lowest=True)
    resultats = []
    for tranche in labels:
        sous = rates[rates["tranche"] == tranche]
        if len(sous) == 0:
            continue
        resultats.append({
            "tranche_ecart": tranche,
            "n_courses": len(sous),
            "pct_des_courses_ratees": round(100 * len(sous) / n, 1),
            "proba_moyenne_du_pick_fautif": round(float(sous["proba_pick"].mean()), 3),
        })
    return resultats


def taux_gagnant_par_segments(df_test, rang_col, segment_cols, n_min=100):
    """Generalisation de performance_segments_gagnant a une ou plusieurs
    colonnes de segmentation deja presentes dans df_test (categorielles ou
    discretisees au prealable, ex. bucket_partants). Retourne un DataFrame
    trie par taux top-1 gagnant CROISSANT (les segments les plus difficiles
    en premier), limite aux segments avec au moins n_min courses pour
    eviter le bruit d'echantillon."""
    d = df_test.copy()
    resultats = []
    for cles, sous_df in d.groupby(segment_cols, observed=True):
        n_courses, n_reussis, pct = taux_reussite_top1(sous_df, rang_col, "est_gagnant")
        if n_courses < n_min:
            continue
        cles_tuple = cles if isinstance(cles, tuple) else (cles,)
        cols = segment_cols if isinstance(segment_cols, list) else [segment_cols]
        ligne = {col: val for col, val in zip(cols, cles_tuple)}
        ligne.update({"n_courses": n_courses, "n_reussis": n_reussis, "pct_top1_gagnant": pct})
        resultats.append(ligne)
    out = pd.DataFrame(resultats)
    if not out.empty:
        out = out.sort_values("pct_top1_gagnant")
    return out


def taux_top1_par_groupe_binaire(gagnants, colonne_bool):
    """Taux top-1 gagnant (parmi les VRAIS gagnants) selon une colonne
    booleenne (ex. est_debutant, identite_possible_fragmentee). Sert a
    isoler l'effet d'un sous-groupe precis sur la performance globale."""
    resultats = []
    for val, sous in gagnants.groupby(colonne_bool):
        n = len(sous)
        top1 = int((sous["rang_du_gagnant"] == 1).sum())
        resultats.append({
            colonne_bool: bool(val), "n_gagnants": n, "n_top1": top1,
            "pct_top1": round(100 * top1 / n, 1) if n else None,
        })
    return resultats


# =============================================================================
# PROTOCOLE D'EVALUATION PERMANENT -- benchmark reel vs benchmark donnees
# propres (demande par Dorian le 26/08/2026, apres le rejet du blend (piste
# 1) et de la calibration handicaps/grands champs (piste 3), pour s'assurer
# que toute future experimentation soit jugee sur des donnees fiables et
# non sur des anomalies de collecte/scraping deguisees en erreurs modele).
#
# Deux benchmarks, a afficher COTE A COTE dans chaque futur rapport :
#
#   - BENCHMARK REEL (reference principale) : toutes les courses avec un
#     resultat exploitable, y COMPRIS les courses avec NON_PARTANT, ARRETE,
#     TOMBE, DISTANCE, RESTE_AU_POTEAU, DEROBE ou DISQUALIFIE -- ce sont des
#     evenements hippiques officiels normaux, pas des anomalies. Seules les
#     courses SANS AUCUNE position d'arrivee (aucune verite terrain
#     exploitable, quelle que soit la raison) en sont necessairement
#     absentes -- impossible de juger une prediction sans resultat.
#
#   - BENCHMARK DONNEES PROPRES : la MEME population que le benchmark reel,
#     moins 3 motifs d'exclusion strictement objectifs et factuels, valides
#     explicitement par Dorian le 26/08/2026 :
#       1. sans_resultat      : course sans aucune position d'arrivee pour
#          aucun partant (40 courses sur l'historique complet au
#          26/08/2026). Exclue des DEUX benchmarks (aucune verite terrain).
#       2. triple_ex_aequo    : au moins 3 partants partagent exactement la
#          meme position d'arrivee au sein d'une course -- mathematiquement
#          quasi impossible comme vrai resultat, quasi certainement une
#          erreur de saisie/scraping (15 courses).
#       3. statut_inconnu     : la course a par ailleurs un resultat, mais
#          au moins un partant n'a NI position d'arrivee NI incident
#          officiel enregistre (NON_PARTANT/ARRETE/TOMBE/DISTANCE/
#          RESTE_AU_POTEAU/DEROBE/DISQUALIFIE) -- statut reellement
#          indetermine, donnee manquante plutot qu'evenement hippique
#          explicable (4658 courses).
#
#   IMPORTANT (decision explicite de Dorian, 26/08/2026) : les courses avec
#   EXACTEMENT 2 partants a egalite sur une meme position (2278 courses)
#   restent dans les DEUX benchmarks -- pas de preuve qu'il s'agisse d'une
#   anomalie plutot que d'un vrai dead-heat ou d'une convention de
#   classement des chevaux distances.
#
#   Regle absolue : aucune course n'est jamais retiree parce que son
#   resultat est surprenant, imprevisible ou defavorable au modele. Seules
#   des anomalies factuelles et objectives (definies ci-dessus) sont
#   retirees, et uniquement du benchmark propre (sauf sans_resultat, retire
#   des deux faute de verite terrain a comparer).
#
# La classification (course_id -> motif_exclusion) a ete calculee UNE FOIS
# par requete SQL directe sur Supabase (table resultats_partants, colonnes
# position_arrivee + incident) le 26/08/2026, verifiee (totaux et
# repartition par motif confirmes independamment), puis exportee dans
# exclusions_benchmark_propre.csv (4713 lignes : course_id, motif_exclusion
# -- seules les courses EXCLUES y figurent ; l'absence d'une course dans ce
# fichier signifie qu'elle est incluse dans les deux benchmarks). Ce
# fichier est charge en lecture seule par les scripts d'evaluation, SANS
# jamais se reconnecter a Supabase -- exactement comme le checkpoint-v3.
#
# A REGENERER PERIODIQUEMENT (nouvelle requete SQL + nouvel export) au fur
# et a mesure que de nouvelles courses sont collectees, pour que les
# benchmarks restent a jour. Ce n'est PAS automatique.
# =============================================================================

EXCLUSIONS_BENCHMARK_PATH = "exclusions_benchmark_propre.csv"

NOMS_MOTIFS_EXCLUSION_BENCHMARK = {
    "sans_resultat": "Course sans aucun resultat (exclue des DEUX benchmarks -- aucune verite terrain)",
    "triple_ex_aequo": "3+ partants a la meme position d'arrivee (exclue du benchmark propre)",
    "statut_inconnu": "Partant sans position ET sans incident officiel connu (exclue du benchmark propre)",
}


def charger_exclusions_benchmark(chemin=EXCLUSIONS_BENCHMARK_PATH):
    """Charge la table d'exclusions (course_id -> motif_exclusion), deja
    calculee et verifiee (voir docstring de section ci-dessus). Lecture
    seule d'un fichier CSV statique versionne -- ne se connecte JAMAIS a
    Supabase depuis un script d'evaluation."""
    return pd.read_csv(chemin, dtype={"course_id": str, "motif_exclusion": str})


def appliquer_benchmarks(df, exclusions):
    """Ajoute a `df` (doit contenir une colonne course_id) les colonnes :
      - motif_exclusion_benchmark : motif si la course est exclue de l'un
        des deux benchmarks, sinon NaN.
      - est_benchmark_reel : True sauf si motif == 'sans_resultat'.
      - est_benchmark_propre : True seulement si aucun motif d'exclusion.
    Ne retire AUCUNE ligne -- c'est a l'appelant de filtrer selon le
    benchmark voulu (ex. df[df['est_benchmark_propre']])."""
    motifs = exclusions.set_index("course_id")["motif_exclusion"]
    motif_course = df["course_id"].map(motifs)
    d = df.copy()
    d["motif_exclusion_benchmark"] = motif_course
    d["est_benchmark_reel"] = motif_course != "sans_resultat"
    d["est_benchmark_propre"] = motif_course.isna()
    return d


def rapport_double_benchmark(df_avec_benchmarks, label=""):
    """Journalise, pour un DataFrame deja enrichi par appliquer_benchmarks(),
    le nombre exact de courses dans le benchmark reel et dans le benchmark
    donnees propres, cote a cote, avec le detail des motifs d'exclusion --
    A APPELER SYSTEMATIQUEMENT dans chaque futur rapport (demande de
    Dorian, 26/08/2026). Retourne un dict {n_total, n_benchmark_reel,
    n_benchmark_propre} pour reutilisation eventuelle."""
    d = df_avec_benchmarks
    n_total = d["course_id"].nunique()
    n_reel = d.loc[d["est_benchmark_reel"], "course_id"].nunique()
    n_propre = d.loc[d["est_benchmark_propre"], "course_id"].nunique()
    log("\n" + "=" * 100)
    titre = "PROTOCOLE D'EVALUATION -- benchmark reel vs benchmark donnees propres"
    log(f"=== {titre}{(' -- ' + label) if label else ''} ===")
    log("=" * 100)
    log(f"  Population consideree ici : {n_total} courses")
    log(f"  BENCHMARK REEL (reference principale)  : {n_reel} courses "
        f"({round(100 * n_reel / n_total, 1) if n_total else 0}% de la population)")
    log(f"  BENCHMARK DONNEES PROPRES               : {n_propre} courses "
        f"({round(100 * n_propre / n_total, 1) if n_total else 0}% de la population)")
    motifs_presents = d.drop_duplicates("course_id")["motif_exclusion_benchmark"].value_counts(dropna=True)
    if motifs_presents.empty:
        log("  Aucune exclusion dans cette population (les courses sans_resultat, si presentes dans")
        log("  l'univers de depart, ont deja disparu du DataFrame evalue -- comportement attendu).")
    else:
        log("\n  Motifs d'exclusion (une course comptee une seule fois par motif) :")
        for motif, n in motifs_presents.items():
            log(f"    {NOMS_MOTIFS_EXCLUSION_BENCHMARK.get(motif, motif):75s} : {n} courses")
    return {"n_total": n_total, "n_benchmark_reel": n_reel, "n_benchmark_propre": n_propre}
