# -*- coding: utf-8 -*-
"""
entrainer_v3_phase1_ranking.py -- COPIE STRICTE de entrainer_v3_phase1.py pour la
piste 1 (objectif d'entrainement oriente classement intra-course), avec UNIQUEMENT
deux ajouts (aucune ligne existante modifiee ou supprimee, aucun changement de
feature ni de split) : conservation de df_val complet et de course_id_train,
necessaires pour entrainer/evaluer un modele de ranking groupe par course_id.
Voir entrainer_v3_phase2_ranking.py pour la suite. Checkpoint ecrit sous un nom
dedie (checkpoint_v3_phase1_ranking.pkl) pour ne jamais interferer avec le
pipeline v3 de reference.
"""
import gc
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
except ImportError:
    pass

from identite_chevaux import resoudre_identite_chevaux
from variables_historiques import construire_variables, trier_chronologiquement
from variables_config import VARIABLES_NUMERIQUES, VARIABLES_CATEGORIELLES

CHECKPOINT_PATH = "checkpoint_v3_phase1_ranking.pkl"  # copie dediee piste 1 (ranking), voir bas de fichier


def main():
    if not lib.DEPENDANCES_LOURDES_DISPONIBLES:
        raise RuntimeError(
            "psycopg2 et/ou scikit-learn ne sont pas installes. Ce script doit tourner dans "
            "l'environnement GitHub Actions du workflow dedie, pas en local."
        )
    lib.log("=" * 100)
    lib.log("PHASE 1/2 â CHARGEMENT, VARIABLES, SPLIT, MODELE v2 + ANALYSE D'ERREURS â 24/08/2026")
    lib.log("=" * 100)

    lib.log("\n[1/6] Chargement des donnees brutes depuis Supabase (PLAT, toutes lignes)...")
    lignes = lib.charger_donnees_brutes()
    lib.log(f"  {len(lignes)} lignes partant/course brutes chargees.")

    lib.log("\n[2/6] Resolution d'identite des chevaux (identique a v2)...")
    horse_uids, rapport_identite = resoudre_identite_chevaux(lignes)
    for l, uid in zip(lignes, horse_uids):
        l["horse_uid"] = uid
    lib.log(f"  {rapport_identite['n_chevaux_distincts_resolus']} chevaux distincts resolus.")

    lib.log("\n[3/6] Construction des 109 variables point-in-time (identique a v2)...")
    lignes_triees = trier_chronologiquement(lignes)
    features = construire_variables(lignes_triees)
    df = pd.DataFrame(features)
    del lignes, lignes_triees, features, horse_uids
    gc.collect()

    df = df[df["position_arrivee"].notna()].copy()
    df = df[df["nb_partants_reel"] >= 3].reset_index(drop=True)
    df["est_gagnant"] = (df["position_arrivee"] == 1).astype(int)
    df["cible_place"] = (df["position_arrivee"] <= df["seuil"]).astype(int)
    lib.log(f"  {len(df)} lignes, {df['course_id'].nunique()} courses apres filtrage.")

    lib.log(f"\n[4/6] Ajout des {2*len(lib.VARIABLES_RELATIVES_CIBLES)} variables relatives au champ (rang + z-score "
            f"intra-course, sur les {len(lib.VARIABLES_RELATIVES_CIBLES)} variables cibles a fort signal)...")
    df = lib.ajouter_variables_relatives(df, lib.VARIABLES_RELATIVES_CIBLES)

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

    lib.log("\n[5/6] Decoupage chronologique strict (identique a v2, meme jeu de TEST) :")
    lib.log(f"  TRAIN : {len(df_train)} lignes / {df_train['course_id'].nunique()} courses")
    lib.log(f"  VALIDATION : {len(df_val)} lignes / {df_val['course_id'].nunique()} courses")
    lib.log(f"  TEST : {len(df_test)} lignes / {df_test['course_id'].nunique()} courses "
            f"({df_test['date_course'].min()} -> {df_test['date_course'].max()})")

    y_train_place = df_train["cible_place"].values
    y_val_place = df_val["cible_place"].values
    y_test_place = df_test["cible_place"].values
    y_train_gagnant = df_train["est_gagnant"].values
    y_val_gagnant = df_val["est_gagnant"].values

    variables_numeriques_v3 = lib.variables_numeriques_v3(VARIABLES_NUMERIQUES)
    X_train_v3 = lib.preparer_matrice(df_train, variables_numeriques_v3, VARIABLES_CATEGORIELLES)
    colonnes_v3 = X_train_v3.columns
    X_val_v3 = lib.preparer_matrice(df_val, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)
    X_test_v3 = lib.preparer_matrice(df_test, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)

    colonnes_relatives = [f"{v}_rang_course" for v in lib.VARIABLES_RELATIVES_CIBLES] + \
                         [f"{v}_z_course" for v in lib.VARIABLES_RELATIVES_CIBLES]
    colonnes_v2 = [c for c in colonnes_v3 if c not in colonnes_relatives]
    lib.log(f"\nMatrice v3 (enrichie) : {X_train_v3.shape[1]} colonnes. Matrice v2 (sous-ensemble) : {len(colonnes_v2)} colonnes.")

    df_val_complet_ranking = df_val.copy()  # ajout piste 1 (ranking) : conserve les colonnes completes de VALIDATION (course_id, position_arrivee, est_gagnant, cible_place, segments...), supprimees dans le checkpoint v3 standard
    course_id_train_ranking = df_train["course_id"].reset_index(drop=True)  # ajout piste 1 (ranking) : necessaire pour regrouper les lignes de X_train_v3 par course_id (objectif ranking)
    del df_train, df_val
    gc.collect()

    # --- colonne temoin aleatoire, ajoutee une fois sur la matrice complete ---
    rng = np.random.RandomState(lib.RANDOM_SEED)
    X_train_v3 = X_train_v3.copy()
    X_val_v3 = X_val_v3.copy()
    X_test_v3 = X_test_v3.copy()
    X_train_v3["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_train_v3))
    X_val_v3["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_val_v3))
    X_test_v3["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_test_v3))
    colonnes_v2_ctrl = colonnes_v2 + ["temoin_aleatoire"]

    # --- filet de securite generique contre les colonnes degenerees (voir
    # v3_lib.colonnes_degenerees) : controle sur TRAIN uniquement, applique
    # de facon identique a train/val/test. ---
    colonnes_numeriques_v3 = [c for c in colonnes_v3 if c in variables_numeriques_v3] + ["temoin_aleatoire"]
    degenerees = lib.colonnes_degenerees(X_train_v3, colonnes_numeriques_v3)
    if degenerees:
        lib.log(f"\n  ATTENTION : {len(degenerees)} colonne(s) degeneree(s) (< 2 valeurs distinctes sur TRAIN, "
                f"exclue(s) du modele v3) : {degenerees}")
    else:
        lib.log("\n  Aucune colonne degeneree detectee sur TRAIN (filet de securite : rien a exclure).")
    colonnes_v3_filtrees = [c for c in colonnes_v3 if c not in degenerees] + ["temoin_aleatoire"]
    X_train_v3 = X_train_v3[colonnes_v3_filtrees]
    X_val_v3 = X_val_v3[colonnes_v3_filtrees]
    X_test_v3 = X_test_v3[colonnes_v3_filtrees]
    # les colonnes relatives degenerees ne doivent plus etre proposees a la
    # phase 2 (permutation importance, etc.)
    colonnes_relatives = [c for c in colonnes_relatives if c not in degenerees]
    # bug identifie le 25/08/2026 (run production #6) : colonnes_v2_ctrl a
    # ete calcule AVANT le filet de securite, a partir de colonnes_v3 non
    # filtre. Si une colonne "de base" (hors variables relatives v3, ex.
    # entraine_a_letranger, degeneree sur les donnees reelles) est exclue
    # ci-dessus, X_train_v3 ne la contient plus -> KeyError au fit du modele
    # v2. On applique le meme filtrage, pour rester coherent avec
    # colonnes_v3_filtrees (aucun changement de variable ni de methodologie :
    # c'est le meme filet de securite deja valide, applique de facon
    # uniforme).
    colonnes_v2_ctrl = [c for c in colonnes_v2_ctrl if c not in degenerees]

    # =========================================================================
    # [6/6] MODELE v2 (reproduit a l'identique) â pour disposer des
    # predictions ligne par ligne necessaires a l'analyse d'erreurs.
    # =========================================================================
    lib.log("\n[6/6] Reproduction du modele v2 (109 variables, hyperparametres deja choisis "
            f"sur validation lors du run precedent : {lib.MEILLEURS_PARAMS_GBM_V2})...")
    gbm_v2 = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **lib.MEILLEURS_PARAMS_GBM_V2)
    gbm_v2.fit(X_train_v3[colonnes_v2_ctrl], y_train_place)
    try:
        auc_val_v2 = round(roc_auc_score(y_val_place, gbm_v2.predict_proba(X_val_v3[colonnes_v2_ctrl])[:, 1]), 4)
        lib.log(f"  AUC validation (verification de coherence avec le rapport v2, attendu ~0.7026) : {auc_val_v2}")
    except ValueError:
        pass
    X_trainval_v2 = pd.concat([X_train_v3[colonnes_v2_ctrl], X_val_v3[colonnes_v2_ctrl]], axis=0).reset_index(drop=True)
    y_trainval_place = np.concatenate([y_train_place, y_val_place])
    gbm_v2_final = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **lib.MEILLEURS_PARAMS_GBM_V2)
    gbm_v2_final.fit(X_trainval_v2, y_trainval_place)
    proba_v2_test = gbm_v2_final.predict_proba(X_test_v3[colonnes_v2_ctrl])[:, 1]
    del gbm_v2, gbm_v2_final, X_trainval_v2
    gc.collect()

    df_test = df_test.reset_index(drop=True)
    df_test["proba_v2"] = proba_v2_test
    df_test["rang_v2"] = df_test.groupby("course_id")["proba_v2"].rank(method="min", ascending=False)

    lib.log("\n" + "=" * 100)
    lib.log("=== ANALYSE D'ERREURS DU MODELE v2 (rappel : 47,1% multi-pick / 24,3% gagnant top-1) ===")
    lib.log("=" * 100)
    lib.log_analyse_erreurs(df_test, "rang_v2", "proba_v2", "GBM v2 (cible=place)")

    # =========================================================================
    # CHECKPOINT â tout ce dont la phase 2 a besoin, sans jamais retoucher a
    # Supabase ni reconstruire les variables.
    # =========================================================================
    checkpoint = {
        "X_train_v3": X_train_v3,
        "X_val_v3": X_val_v3,
        "X_test_v3": X_test_v3,
        "y_train_place": y_train_place,
        "y_val_place": y_val_place,
        "y_test_place": y_test_place,
        "y_train_gagnant": y_train_gagnant,
        "y_val_gagnant": y_val_gagnant,
        "df_test": df_test,
        "colonnes_relatives": colonnes_relatives,
        "df_val": df_val_complet_ranking,
        "course_id_train": course_id_train_ranking,
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    lib.log(f"\nCheckpoint sauvegarde : {CHECKPOINT_PATH} "
            f"(X_train_v3={X_train_v3.shape}, X_val_v3={X_val_v3.shape}, X_test_v3={X_test_v3.shape}, "
            f"df_test={df_test.shape}, colonnes_relatives={len(colonnes_relatives)})")
    lib.log("\nPHASE 1 TERMINEE â la phase 2 peut reprendre directement depuis ce checkpoint, "
            "sans recharger ni reconstruire les donnees.")


if __name__ == "__main__":
    main()
