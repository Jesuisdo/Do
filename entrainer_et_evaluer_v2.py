# -*- coding: utf-8 -*-
"""
entrainer_et_evaluer_v2.py — Premier vrai test du modèle sportif pur sur
données hors échantillon. Feu vert donné le 23/08/2026.

Ce script :
  1. Charge TOUT l'historique PLAT (resultats_courses/resultats_partants).
  2. Résout l'identité de chaque cheval de façon robuste (voir
     identite_chevaux.py) — le nom seul n'est PAS supposé fiable avant 2025
     (id_cheval PMU absent à 0% avant 2025, cf. rapport base du 23/08/2026).
  3. Construit un large ensemble de variables historiques STRICTEMENT
     point-in-time (voir variables_historiques.py — testé unitairement,
     aucune variable ne peut refléter une information postérieure à la
     course qu'elle décrit).
  4. AUCUNE cote, AUCUNE donnée de marché n'entre dans ce jeu de variables.
  5. Découpe TRAIN / VALIDATION / TEST strictement chronologique (jamais de
     mélange aléatoire). Les hyperparamètres sont choisis sur VALIDATION.
     Le jeu TEST n'est touché QU'UNE SEULE FOIS, à la toute fin, pour le
     rapport final — jamais pour choisir quoi que ce soit.
  6. Laisse un gradient boosting (interactions automatiques) et une
     régression logistique L1 (sélection de variables explicite) décider
     quelles variables comptent, plutôt que de partir d'une liste de 43
     variables choisies à la main.
  7. Publie un rapport honnête, y compris si le résultat est décevant.

Convention de la cible "placé" (identique au modèle précédent pour
permettre une comparaison directe) : top 2 si <=7 partants arrivants, top 3
sinon. Ce n'est PAS exactement le barème PMU officiel (qui dépend aussi de
la catégorie et peut monter à 4 sur les gros handicaps) — simplification
déjà présente dans le modèle précédent, reconduite ici pour comparabilité,
et documentée dans le rapport (section "problèmes de données").
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.inspection import permutation_importance
    DEPENDANCES_LOURDES_DISPONIBLES = True
except ImportError:
    # psycopg2/scikit-learn ne sont pas installables dans l'environnement de
    # developpement local de ce projet (proxy sortant restreint) — ce
    # script est concu pour tourner via GitHub Actions, ou ces paquets
    # s'installent librement. En local, seule la logique pandas pure
    # (preparer_matrice, calculer_baseline_combine_v1, taux_reussite_*) est
    # testable — voir test_entrainer_et_evaluer_v2.py.
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
SOUS_ECHANTILLON_L1 = 150_000       # cf. section "methode" du rapport
SOUS_ECHANTILLON_PERMUTATION = 60_000

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
    """Reproduit EXACTEMENT la methode combine_v1 en production (somme de
    rangs a poids egal sur 4 facteurs) pour comparaison honnete sur le MEME
    jeu de test que le nouveau modele."""
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
    """Methodologie IDENTIQUE au modele precedent (comparabilite directe
    avec le 41.1% historique) : parmi les picks classes <= seuil par le
    modele, quelle proportion finit reellement <= seuil ?"""
    d = df.copy()
    d["est_pick"] = d[colonne_rang_predit] <= d["seuil"]
    d["est_reussi"] = d["est_pick"] & (d["position_arrivee"] <= d["seuil"])
    essais = int(d["est_pick"].sum())
    reussis = int(d["est_reussi"].sum())
    pct = round(100 * reussis / essais, 1) if essais else float("nan")
    return essais, reussis, pct


def taux_reussite_top1(df, colonne_rang_predit, cible_col):
    """Metrique complementaire, plus intuitive : si on ne retient QUE le
    meilleur pick du modele par course, a quelle frequence est-il reellement
    dans la cible (gagnant, ou place) ?"""
    d = df[df[colonne_rang_predit] == 1].copy()
    n_courses = len(d)
    n_reussis = int((d[cible_col] == 1).sum()) if cible_col == "est_gagnant" else int((d["position_arrivee"] <= d["seuil"]).sum())
    pct = round(100 * n_reussis / n_courses, 1) if n_courses else float("nan")
    return n_courses, n_reussis, pct


def preparer_matrice(df, colonnes_dummies_reference=None):
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
    # float32 (pas float64) pour la partie numerique : divise par ~2 la memoire
    # des grandes matrices X_*, sans consequence sur la precision des AUC/rangs.
    num_df = df[VARIABLES_NUMERIQUES].reset_index(drop=True).astype("float32")
    X = pd.concat([num_df, cat_df.reset_index(drop=True)], axis=1)
    if colonnes_dummies_reference is not None:
        X = X.reindex(columns=colonnes_dummies_reference, fill_value=0)
    return X


def main():
    if not DEPENDANCES_LOURDES_DISPONIBLES:
        raise RuntimeError(
            "psycopg2 et/ou scikit-learn ne sont pas installes. Ce script doit tourner dans "
            "l'environnement GitHub Actions du workflow dedie, pas en local."
        )
    log("=" * 100)
    log("PREMIER TEST DU MODELE SPORTIF PUR SUR DONNEES HORS ECHANTILLON — 23/08/2026")
    log("=" * 100)

    log("\n[1/8] Chargement des donnees brutes depuis Supabase (PLAT, toutes lignes)...")
    lignes = charger_donnees_brutes()
    log(f"  {len(lignes)} lignes partant/course brutes chargees (avant filtrage resultat connu).")
    n_sans_position = sum(1 for l in lignes if l.get("position_arrivee") is None)
    log(f"  dont {n_sans_position} sans position d'arrivee connue (DNF/disqualifie/non partant) "
        f"— {round(100*n_sans_position/len(lignes),1)}% — exclues du jeu d'entrainement/test ci-dessous.")

    log("\n[2/8] Resolution d'identite des chevaux (id_cheval quand fiable, sinon nom+peremere+trajectoire d'age)...")
    horse_uids, rapport_identite = resoudre_identite_chevaux(lignes)
    for l, uid in zip(lignes, horse_uids):
        l["horse_uid"] = uid
    log(f"  {rapport_identite['n_noms_distincts']} noms distincts -> "
        f"{rapport_identite['n_chevaux_distincts_resolus']} chevaux distincts resolus.")
    log(f"  {rapport_identite['n_noms_avec_collision']} noms ({rapport_identite['pct_noms_avec_collision']}%) "
        f"portes par PLUSIEURS chevaux differents dans nos donnees (collision detectee et separee).")

    # calcule maintenant (pendant que 'lignes' brutes existent encore) ce dont la
    # section 9/10 du rapport aura besoin plus tard, pour pouvoir liberer 'lignes'
    # juste apres — cf. note memoire ci-dessous.
    id_cheval_present = sum(1 for l in lignes if l.get("id_cheval"))
    n_lignes_brutes = len(lignes)

    log("\n[3/8] Construction des variables point-in-time (aucune fuite du futur, teste unitairement)...")
    lignes_triees = trier_chronologiquement(lignes)
    features = construire_variables(lignes_triees)
    df = pd.DataFrame(features)
    log(f"  {len(df)} lignes de features construites, {df['course_id'].nunique()} courses.")

    # Memoire : 'lignes'/'lignes_triees'/'features' sont des listes de ~1M dicts
    # Python (tres couteux en RAM, largement plus que les DataFrame equivalents).
    # Elles ne servent plus a rien une fois 'df' construit (tout ce qui est
    # necessaire au rapport en a deja ete extrait ci-dessus) : on les libere
    # explicitement pour rester sous la limite memoire du runner GitHub Actions.
    del lignes, lignes_triees, features, horse_uids
    gc.collect()

    df = df[df["position_arrivee"].notna()].copy()
    df = df[df["nb_partants_reel"] >= 3].reset_index(drop=True)
    log(f"  Apres filtrage (resultat connu, >=3 partants arrivants) : "
        f"{len(df)} lignes, {df['course_id'].nunique()} courses.")

    df["est_gagnant"] = (df["position_arrivee"] == 1).astype(int)
    df["cible_place"] = (df["position_arrivee"] <= df["seuil"]).astype(int)

    # --- decoupage chronologique STRICT : 70% train / 15% validation / 15% test ---
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

    log("\n[4/8] Decoupage chronologique strict (jamais de melange aleatoire) :")
    log(f"  TRAIN : {len(df_train)} lignes / {df_train['course_id'].nunique()} courses "
        f"({df_train['date_course'].min()} -> {df_train['date_course'].max()})")
    log(f"  VALIDATION : {len(df_val)} lignes / {df_val['course_id'].nunique()} courses "
        f"({df_val['date_course'].min()} -> {df_val['date_course'].max()}) — sert UNIQUEMENT a choisir les hyperparametres")
    log(f"  TEST (hors echantillon, jamais touche avant le rapport final) : {len(df_test)} lignes / "
        f"{df_test['course_id'].nunique()} courses ({df_test['date_course'].min()} -> {df_test['date_course'].max()})")

    nb_variables_creees = len(VARIABLES_NUMERIQUES) + len(VARIABLES_CATEGORIELLES)
    log(f"\n=== 1. VARIABLES CREEES : {nb_variables_creees} ===")
    log(f"({len(VARIABLES_NUMERIQUES)} numeriques + {len(VARIABLES_CATEGORIELLES)} categorielles, "
        f"contre 43 dans le modele precedent)")

    couverture = (df[VARIABLES_NUMERIQUES].notna().mean() * 100).round(1).sort_values(ascending=False)
    log("\nCouverture (part de lignes non-nulles) par variable numerique :")
    for nom, pct in couverture.items():
        log(f"  {nom:50s} {pct:5.1f}%")
    variables_peu_couvertes = couverture[couverture < 1.0].index.tolist()
    nb_utilisables = len(VARIABLES_NUMERIQUES) - len(variables_peu_couvertes) + len(VARIABLES_CATEGORIELLES)
    log(f"\n=== 2. VARIABLES REELLEMENT UTILISABLES (couverture >= 1%) : {nb_utilisables}/{nb_variables_creees} ===")
    if variables_peu_couvertes:
        log(f"Quasi-vides (<1% de couverture, exclues de fait) : {variables_peu_couvertes}")
    else:
        log("Aucune variable numerique n'est quasi-vide.")

    # 'df' (non-decoupe) ne sert plus a rien une fois df_train/df_val/df_test
    # construits et la couverture mesuree — le liberer avant de construire les
    # matrices d'entrainement (autre gros poste memoire).
    del df
    gc.collect()

    # --- matrices ---
    X_train = preparer_matrice(df_train)
    colonnes_ref = X_train.columns
    X_val = preparer_matrice(df_val, colonnes_dummies_reference=colonnes_ref)
    X_test = preparer_matrice(df_test, colonnes_dummies_reference=colonnes_ref)
    y_train_place = df_train["cible_place"].values
    y_val_place = df_val["cible_place"].values
    y_test_place = df_test["cible_place"].values

    # df_train/df_val (DataFrame "large", ~118 colonnes brutes) ne servent plus
    # a rien une fois les matrices X_train/X_val et les cibles y_* extraites —
    # df_test, plus petit, est conserve car reutilise section 5 a 8.
    del df_train, df_val
    gc.collect()

    log(f"\nMatrice finale (apres expansion categorielle, cap {CAP_CARDINALITE_CATEGORIELLE}+AUTRE) : {X_train.shape[1]} colonnes.")

    # --- colonne temoin aleatoire : sert a distinguer "vraie signal" de "bruit" ---
    rng = np.random.RandomState(RANDOM_SEED)
    X_train_ctrl = X_train.copy()
    X_val_ctrl = X_val.copy()
    X_test_ctrl = X_test.copy()
    X_train_ctrl["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_train_ctrl))
    X_val_ctrl["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_val_ctrl))
    X_test_ctrl["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_test_ctrl))

    # =========================================================================
    # MODELE 1 : Gradient Boosting — recherche d'hyperparametres sur VALIDATION
    # =========================================================================
    log("\n[5/8] Entrainement Gradient Boosting (recherche d'hyperparametres sur VALIDATION uniquement)...")
    grille_gbm = [
        {"max_depth": 4, "max_iter": 150, "learning_rate": 0.05, "l2_regularization": 1.0, "min_samples_leaf": 25},
        {"max_depth": 5, "max_iter": 200, "learning_rate": 0.05, "l2_regularization": 1.0, "min_samples_leaf": 40},
        {"max_depth": 3, "max_iter": 250, "learning_rate": 0.03, "l2_regularization": 2.0, "min_samples_leaf": 60},
    ]
    meilleur_gbm, meilleurs_params, meilleure_auc_val = None, None, -1
    for params in grille_gbm:
        m = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **params)
        m.fit(X_train_ctrl, y_train_place)
        try:
            auc_val = roc_auc_score(y_val_place, m.predict_proba(X_val_ctrl)[:, 1])
        except ValueError:
            auc_val = -1
        log(f"  {params} -> AUC validation = {round(auc_val, 4)}")
        if auc_val > meilleure_auc_val:
            meilleure_auc_val, meilleurs_params, meilleur_gbm = auc_val, params, m
    log(f"  Meilleurs hyperparametres retenus (sur validation) : {meilleurs_params}")

    log("  Refit final sur TRAIN+VALIDATION avec les meilleurs hyperparametres...")
    X_trainval_ctrl = pd.concat([X_train_ctrl, X_val_ctrl], axis=0).reset_index(drop=True)
    y_trainval_place = np.concatenate([y_train_place, y_val_place])
    gbm_final = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **meilleurs_params)
    gbm_final.fit(X_trainval_ctrl, y_trainval_place)
    proba_gbm_test = gbm_final.predict_proba(X_test_ctrl)[:, 1]
    # X_trainval_ctrl (fusion temporaire train+validation) et les 3 modeles GBM
    # non retenus de la grille ne servent plus a rien — liberes avant la section
    # L1 (qui construit elle-meme plusieurs grandes matrices supplementaires).
    del X_trainval_ctrl, X_train_ctrl, X_val_ctrl, meilleur_gbm
    gc.collect()

    # =========================================================================
    # MODELE 2 : Regression logistique L1 — C choisi sur VALIDATION (pas sur TEST,
    # correction d'une faiblesse methodologique du script precedent qui
    # choisissait C en regardant l'AUC du jeu de TEST lui-meme).
    # =========================================================================
    log("\n[6/8] Entrainement regression logistique L1 (selection de variables explicite)...")
    log(f"  Par souci de temps de calcul, le C est choisi sur un sous-echantillon "
        f"stratifie de {SOUS_ECHANTILLON_L1} lignes de TRAIN (voir 'problemes de donnees' "
        f"dans le rapport pour la justification de ce compromis).")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    # Chaque _imp est immediatement consomme par le scaler puis libere : evite
    # de garder 6 grandes matrices (3 imputees + 3 mises a l'echelle) vivantes
    # simultanement. float32 (au lieu du float64 par defaut de sklearn) divise
    # par ~2 la memoire de chacune.
    X_train_imp = imputer.fit_transform(X_train).astype("float32")
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train.columns, dtype="float32")
    del X_train_imp
    X_val_imp = imputer.transform(X_val).astype("float32")
    X_val_sc = pd.DataFrame(scaler.transform(X_val_imp), columns=X_train.columns, dtype="float32")
    del X_val_imp
    X_test_imp = imputer.transform(X_test).astype("float32")
    X_test_sc = pd.DataFrame(scaler.transform(X_test_imp), columns=X_train.columns, dtype="float32")
    del X_test_imp
    gc.collect()

    if len(X_train_sc) > SOUS_ECHANTILLON_L1:
        idx_sous = pd.Series(y_train_place).groupby(y_train_place).apply(
            lambda s: s.sample(min(len(s), SOUS_ECHANTILLON_L1 // 2), random_state=RANDOM_SEED)
        ).index.get_level_values(1)
        X_train_sc_l1 = X_train_sc.iloc[idx_sous]
        y_train_l1 = y_train_place[idx_sous]
    else:
        X_train_sc_l1, y_train_l1 = X_train_sc, y_train_place

    meilleure_logreg, meilleur_c, meilleure_auc_val_lr = None, None, -1
    for C in [0.003, 0.01, 0.03, 0.1, 0.3]:
        lr = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=3000, random_state=RANDOM_SEED)
        lr.fit(X_train_sc_l1, y_train_l1)
        try:
            auc_val = roc_auc_score(y_val_place, lr.predict_proba(X_val_sc)[:, 1])
        except ValueError:
            auc_val = -1
        log(f"  C={C} -> AUC validation = {round(auc_val, 4)}")
        if auc_val > meilleure_auc_val_lr:
            meilleure_auc_val_lr, meilleur_c, meilleure_logreg = auc_val, C, lr
    log(f"  Meilleur C retenu (sur validation) : {meilleur_c}")
    proba_lr_test = meilleure_logreg.predict_proba(X_test_sc)[:, 1]

    coefs = pd.Series(meilleure_logreg.coef_[0], index=X_train.columns)
    selectionnees_l1 = coefs[coefs.abs() > 1e-6].sort_values(key=abs, ascending=False)
    inutiles_l1 = coefs[coefs.abs() <= 1e-6].index.tolist()
    log(f"\n=== 3. VARIABLES SELECTIONNEES (regression L1, coefficient non-nul) : "
        f"{len(selectionnees_l1)}/{len(coefs)} ===")
    for nom, c in selectionnees_l1.head(40).items():
        log(f"  {nom:50s} coef={c:+.4f}")
    if len(selectionnees_l1) > 40:
        log(f"  ... et {len(selectionnees_l1) - 40} autres.")

    # X_train_sc/X_val_sc/X_train_sc_l1 (matrices mises a l'echelle pour la
    # regression L1) ne servent plus a rien une fois le meilleur modele choisi.
    del X_train_sc, X_val_sc, X_test_sc, X_train_sc_l1, X_train, X_val, X_test
    gc.collect()

    # =========================================================================
    # Importance par permutation (GBM final, sur un sous-echantillon de TEST
    # pour rester dans un temps de calcul raisonnable — les metriques de
    # PERFORMANCE elles-memes, plus haut, restent calculees sur le TEST complet)
    # =========================================================================
    log("\nCalcul de l'importance par permutation (GBM, sous-echantillon de TEST)...")
    idx_perm = X_test_ctrl.sample(min(len(X_test_ctrl), SOUS_ECHANTILLON_PERMUTATION), random_state=RANDOM_SEED).index
    # n_jobs=1 (pas -1) : evite que joblib duplique la matrice/le modele en
    # memoire dans plusieurs processus paralleles — risque d'OOM sur le runner
    # GitHub Actions, pour un gain de temps marginal ici (echantillon deja reduit).
    perm = permutation_importance(
        gbm_final, X_test_ctrl.loc[idx_perm], y_test_place[idx_perm],
        scoring="roc_auc", n_repeats=5, random_state=RANDOM_SEED, n_jobs=1,
    )
    importances = pd.Series(perm.importances_mean, index=X_test_ctrl.columns).sort_values(ascending=False)
    seuil_bruit = importances.get("temoin_aleatoire", 0.0)
    log(f"  Importance de la colonne TEMOIN ALEATOIRE (bruit pur) = {round(seuil_bruit, 5)} "
        f"— toute variable en dessous de ce seuil n'est pas distinguable du hasard.")
    log("\n  Top 25 variables par importance de permutation (GBM) :")
    for nom, imp in importances.head(25).items():
        marqueur = "  <-- BRUIT DE REFERENCE" if nom == "temoin_aleatoire" else ""
        log(f"    {nom:50s} importance={imp:.5f}{marqueur}")

    inutiles_permutation = importances[importances <= seuil_bruit].index.tolist()
    inutiles_permutation = [v for v in inutiles_permutation if v != "temoin_aleatoire"]
    inutiles_final = sorted(set(inutiles_l1) & set(inutiles_permutation))
    log(f"\n=== 4. VARIABLES INUTILES (coefficient L1 nul ET importance permutation <= bruit) : "
        f"{len(inutiles_final)}/{len(coefs)} ===")
    log(", ".join(inutiles_final) if inutiles_final else "(aucune par ce double critere)")
    log(f"  (pour reference, seules L1: {len(inutiles_l1)} a coefficient nul ; "
        f"seules permutation: {len(inutiles_permutation)} sous le bruit)")

    # =========================================================================
    # 5. PERFORMANCE HORS ECHANTILLON — TEST touche ICI, une seule fois.
    # =========================================================================
    df_test = df_test.reset_index(drop=True)
    df_test["rang_predit_baseline"] = calculer_baseline_combine_v1(df_test)
    df_test["proba_gbm"] = proba_gbm_test
    df_test["proba_lr"] = proba_lr_test
    df_test["rang_predit_gbm"] = df_test.groupby("course_id")["proba_gbm"].rank(method="min", ascending=False)
    df_test["rang_predit_lr"] = df_test.groupby("course_id")["proba_lr"].rank(method="min", ascending=False)

    log("\n" + "=" * 100)
    log("=== 5. PERFORMANCE HORS ECHANTILLON (jeu TEST, jamais utilise avant maintenant) ===")
    log("=" * 100)
    log("\n-- Methodologie A : taux de reussite multi-picks (IDENTIQUE au calcul du 41.1% historique) --")
    resultats_A = {}
    for nom, col in [("Baseline combine_v1 (regle de production, recalculee sur ce nouveau test set)", "rang_predit_baseline"),
                      ("Gradient Boosting (nouvelles variables)", "rang_predit_gbm"),
                      ("Regression logistique L1 (nouvelles variables)", "rang_predit_lr")]:
        essais, reussis, pct = taux_reussite_place(df_test, col)
        resultats_A[nom] = pct
        log(f"  {nom:65s} {reussis}/{essais} = {pct}%")

    log("\n-- Methodologie B : le MEILLEUR pick du modele par course (top-1), gagnant --")
    for nom, col in [("Baseline combine_v1", "rang_predit_baseline"),
                      ("Gradient Boosting", "rang_predit_gbm"),
                      ("Regression logistique L1", "rang_predit_lr")]:
        n_courses, n_reussis, pct = taux_reussite_top1(df_test, col, "est_gagnant")
        log(f"  {nom:35s} gagnant trouve {n_reussis}/{n_courses} fois = {pct}%")

    log("\n-- Methodologie C : le MEILLEUR pick du modele par course (top-1), place --")
    for nom, col in [("Baseline combine_v1", "rang_predit_baseline"),
                      ("Gradient Boosting", "rang_predit_gbm"),
                      ("Regression logistique L1", "rang_predit_lr")]:
        n_courses, n_reussis, pct = taux_reussite_top1(df_test, col, "cible_place")
        log(f"  {nom:35s} place {n_reussis}/{n_courses} fois = {pct}%")

    try:
        auc_gbm = round(roc_auc_score(y_test_place, proba_gbm_test), 4)
        ll_gbm = round(log_loss(y_test_place, proba_gbm_test), 4)
        auc_lr = round(roc_auc_score(y_test_place, proba_lr_test), 4)
        ll_lr = round(log_loss(y_test_place, proba_lr_test), 4)
        log(f"\n  AUC / log-loss hors-echantillon — GBM: AUC={auc_gbm} logloss={ll_gbm} | "
            f"LR-L1: AUC={auc_lr} logloss={ll_lr}")
    except ValueError as e:
        log(f"  AUC/log-loss non calculables : {e}")

    log("\n=== 6. COMPARAISON AVEC LE PRECEDENT 41,1% ===")
    log("  Le 41.1% historique (rapport du 17/08/2026) a ete mesure sur un jeu de test BEAUCOUP plus "
        "petit (base encore tres jeune a l'epoque). Pour une comparaison honnete, le baseline combine_v1 "
        "a ete recalcule ICI, sur EXACTEMENT le meme jeu de test que le nouveau modele (voir ligne "
        "'Baseline combine_v1' ci-dessus) — c'est ce chiffre-la, pas le 41.1% brut, qui constitue la "
        "comparaison rigoureuse.")
    log(f"  Baseline combine_v1 sur ce nouveau test : {resultats_A.get('Baseline combine_v1 (regle de production, recalculee sur ce nouveau test set)')}%")
    log(f"  Nouveau modele Gradient Boosting : {resultats_A.get('Gradient Boosting (nouvelles variables)')}%")
    log(f"  Nouveau modele Regression L1 : {resultats_A.get('Regression logistique L1 (nouvelles variables)')}%")

    log("\n=== 7. PERFORMANCE PAR NOMBRE DE PARTANTS ===")
    df_test["groupe_partants"] = df_test["nb_partants_reel"].apply(bucket_partants)
    for groupe, sous_df in df_test.groupby("groupe_partants"):
        log(f"\n  -- {groupe} ({sous_df['course_id'].nunique()} courses) --")
        for nom, col in [("baseline", "rang_predit_baseline"), ("gbm", "rang_predit_gbm"), ("lr_l1", "rang_predit_lr")]:
            essais, reussis, pct = taux_reussite_place(sous_df, col)
            log(f"    {nom:12s} {reussis}/{essais} = {pct}%")

    log("\n=== 8. PERFORMANCE PAR TYPE DE COURSE ===")
    log("  Toutes les courses sont PLAT (perimetre du projet). Decoupage par categorie/particularite et par type de piste :")
    for var, label in [("categorie_particularite", "categorie_particularite"), ("type_piste", "type_piste")]:
        log(f"\n  -- Par {label} --")
        df_test["_grp"] = df_test[var].fillna("INCONNU")
        cats_frequentes = [c for c, n in Counter(df_test["_grp"]).items() if n >= 200]
        if not cats_frequentes:
            log("    Aucune categorie n'a assez de lignes de test (>=200) pour un resultat lisible.")
        for cat in cats_frequentes:
            sous_df = df_test[df_test["_grp"] == cat]
            log(f"\n    -- {cat} ({sous_df['course_id'].nunique()} courses, {len(sous_df)} lignes) --")
            for nom, col in [("baseline", "rang_predit_baseline"), ("gbm", "rang_predit_gbm")]:
                essais, reussis, pct = taux_reussite_place(sous_df, col)
                log(f"      {nom:12s} {reussis}/{essais} = {pct}%")

    log("\n=== 9. COUVERTURE ET QUALITE DE L'IDENTIFICATION DES CHEVAUX ===")
    log(f"  Lignes traitees : {rapport_identite['n_lignes']}")
    log(f"  Noms distincts : {rapport_identite['n_noms_distincts']}")
    log(f"  Chevaux distincts resolus : {rapport_identite['n_chevaux_distincts_resolus']}")
    log(f"  Noms portes par plusieurs chevaux differents (collision detectee et separee) : "
        f"{rapport_identite['n_noms_avec_collision']} ({rapport_identite['pct_noms_avec_collision']}%)")
    log("  Echantillon de collisions detectees (jusqu'a 25) :")
    for c in rapport_identite["echantillon_collisions"][:15]:
        log(f"    {c['nom']}: {c['n_clusters']} chevaux distincts sur {c['n_lignes_total']} lignes -> {c['detail']}")
    log(f"  id_cheval PMU present sur {id_cheval_present}/{n_lignes_brutes} lignes "
        f"({round(100*id_cheval_present/n_lignes_brutes,1)}%) — utilise en priorite quand disponible et coherent, "
        "heuristique nom+pere+mere+trajectoire d'age utilisee pour le reste (voir identite_chevaux.py).")

    log("\n=== 10. PROBLEMES DE DONNEES RENCONTRES (liste honnete) ===")
    log("  - id_cheval PMU absent a 0% avant 2025, 48.5% en 2025, 100% en 2026 : l'identite des chevaux "
        "avant 2025 repose sur une heuristique nom+parents+trajectoire d'age (voir section 9), pas sur "
        "un identifiant garanti.")
    log("  - mapping_intervenants / stats_intervenants (statistiques jockeys externes Geny.com) sont "
        "VIDES sur la nouvelle base PLAT (non re-scrapees depuis la reconstruction). Ce script ne "
        "depend donc PAS de ces tables : il recalcule des statistiques jockey/entraineur/proprietaire/"
        "eleveur en interne, point-in-time, a partir de resultats_partants uniquement.")
    log(f"  - {round(100*n_sans_position/n_lignes_brutes,1)}% des lignes partant n'ont pas de position "
        "d'arrivee connue (non partant, disqualifie, etc.) et sont exclues du jeu d'entrainement/test "
        "ET de la construction de l'historique interne — legere sous-estimation possible des compteurs "
        "'nombre de courses' internes.")
    log("  - terrain_intitule manque sur ~11.5% des courses, type_piste sur ~37%, poids_condition_monte "
        "sur ~79% (cf. rapport base du 23/08/2026) — le gradient boosting gere le NaN nativement, la "
        "regression L1 impute par la mediane (peut diluer legerement le signal de ces variables).")
    log("  - parsing de la 'musique' PMU : heuristique documentee (positions extraites dans l'ordre, "
        "0=non classe, D/T/A=incident), pas une garantie d'exactitude a 100%.")
    log("  - selection de variables L1 : par contrainte de temps de calcul, le C de regularisation est "
        f"choisi sur un sous-echantillon de {SOUS_ECHANTILLON_L1} lignes de train (stratifie), pas sur "
        "le train complet. Le modele GBM (resultat principal rapporte) est lui entraine sur l'integralite "
        "du train+validation, sans sous-echantillonnage.")
    log(f"  - l'importance par permutation est calculee sur un sous-echantillon de {SOUS_ECHANTILLON_PERMUTATION} "
        "lignes de test (les METRIQUES DE PERFORMANCE elles-memes restent calculees sur le test complet).")
    log("  - seuil de 'place' simplifie (top2/top3 selon nombre de partants arrivants) — ne reproduit pas "
        "exactement le bareme PMU officiel (categorie, partants declares vs arrivants, cas a 4 places).")
    log("  - un seul seed aleatoire (42) a ete utilise : les pourcentages ci-dessus ont une variance "
        "d'echantillonnage non quantifiee ici (pas de repetition multi-seed dans ce premier test).")

    log("\n" + "=" * 100)
    log("=== RESUME ===")
    log("=" * 100)
    log(f"Variables creees : {nb_variables_creees} (vs 43 precedemment)")
    log(f"Variables selectionnees (L1) : {len(selectionnees_l1)}/{len(coefs)}")
    log(f"Variables inutiles (L1 nul ET sous le bruit) : {len(inutiles_final)}/{len(coefs)}")
    log(f"Taux de reussite (methodologie 41.1%) — baseline recalculee : "
        f"{resultats_A.get('Baseline combine_v1 (regle de production, recalculee sur ce nouveau test set)')}% | "
        f"GBM : {resultats_A.get('Gradient Boosting (nouvelles variables)')}% | "
        f"LR-L1 : {resultats_A.get('Regression logistique L1 (nouvelles variables)')}%")
    log("Ce rapport est le resultat REEL, non ajuste. Voir section 10 pour les limites methodologiques "
        "assumees.")


if __name__ == "__main__":
    main()
