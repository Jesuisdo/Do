# -*- coding: utf-8 -*-
"""
piste6_phase1_split4.py -- PHASE 1/2 de la piste 6 (architecture a deux
etages : selection Top-5 puis affinage), demandee par Dorian le 30/08/2026.

Copie STRICTE de entrainer_v3_phase1_genealogie.py (chargement, resolution
d'identite, 109+22 variables v3 point-in-time, AUCUNE modification) -- le
SEUL changement est le decoupage chronologique, etendu de 70/15/15 a
70/15/7.5/7.5 (TRAIN / VALIDATION / TEST A / TEST B), pour permettre le
protocole approuve par Dorian : etape 1 (B+genealogie) entrainee sur TRAIN
exactement comme avant (aucun changement de son propre entrainement), etape
2 (affinage) et seuils de confiance calibres sur VALIDATION, puis confirmes
sans aucun ajustement sur TEST A puis TEST B (tous deux strictement
posterieurs a TRAIN+VALIDATION, jamais vus avant le calcul final).

Ne se connecte a Supabase QU'EN LECTURE. N'ecrit rien en base.
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
from variables_genealogie import construire_variables_genealogie, COLONNES_GENEALOGIE

CHECKPOINT_PATH = "checkpoint_piste6_split4.pkl"


def main():
    if not lib.DEPENDANCES_LOURDES_DISPONIBLES:
        raise RuntimeError(
            "psycopg2 et/ou scikit-learn ne sont pas installes. Ce script doit tourner dans "
            "l'environnement GitHub Actions du workflow dedie, pas en local.")
    lib.log("=" * 100)
    lib.log("PISTE 6 -- PHASE 1/2 -- MEMES VARIABLES QUE PISTE 3, DECOUPAGE 70/15/7.5/7.5 -- 30/08/2026")
    lib.log("=" * 100)

    lib.log("\n[1/7] Chargement des donnees brutes depuis Supabase (PLAT, toutes lignes)...")
    lignes = lib.charger_donnees_brutes()
    lib.log(f"   {len(lignes)} lignes partant/course brutes chargees.")

    lib.log("\n[2/7] Resolution d'identite des chevaux (identique a v2/v3)...")
    horse_uids, rapport_identite = resoudre_identite_chevaux(lignes)
    for l, uid in zip(lignes, horse_uids):
        l["horse_uid"] = uid
    lib.log(f"   {rapport_identite['n_chevaux_distincts_resolus']} chevaux distincts resolus.")

    lib.log("\n[3/7] Construction des 109 variables point-in-time v3 (identique a v2/v3, AUCUN changement)...")
    lignes_triees = trier_chronologiquement(lignes)
    features = construire_variables(lignes_triees)
    df = pd.DataFrame(features)

    lib.log("\n[4/7] Construction des 20 variables de genealogie point-in-time (identique a piste 3)...")
    features_geneal = construire_variables_genealogie(lignes_triees)
    df_geneal = pd.DataFrame(features_geneal)
    assert len(df_geneal) == len(df), "desalignement genealogie vs variables v3"
    for col in COLONNES_GENEALOGIE:
        df[col] = df_geneal[col].values
    del lignes, lignes_triees, features, features_geneal, df_geneal, horse_uids
    gc.collect()

    df = df[df["position_arrivee"].notna()].copy()
    df = df[df["nb_partants_reel"] >= 3].reset_index(drop=True)
    df["est_gagnant"] = (df["position_arrivee"] == 1).astype(int)
    df["cible_place"] = (df["position_arrivee"] <= df["seuil"]).astype(int)
    lib.log(f"   {len(df)} lignes, {df['course_id'].nunique()} courses apres filtrage.")

    lib.log(f"\n[5/7] Ajout des {2*len(lib.VARIABLES_RELATIVES_CIBLES)} variables relatives au champ v3 "
            f"(rang + z-score intra-course, inchange)...")
    df = lib.ajouter_variables_relatives(df, lib.VARIABLES_RELATIVES_CIBLES)

    # --- decoupage chronologique STRICT, 70/15/7.5/7.5 (approuve par Dorian
    # le 30/08/2026 -- TRAIN/VALIDATION identiques aux 70/15 habituels, les
    # 15% restants (jamais construits jusqu'ici) sont desormais coupes en
    # deux : TEST A (premiere moitie chronologique) puis TEST B (derniere,
    # la plus recente). Aucun chevauchement, aucun retour en arriere.) ---
    df = df.sort_values(["date_course", "course_id"]).reset_index(drop=True)
    courses_ordre = df["course_id"].drop_duplicates().tolist()
    n = len(courses_ordre)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    n_testA = int(n * 0.925)
    courses_train = set(courses_ordre[:n_train])
    courses_val = set(courses_ordre[n_train:n_val])
    courses_testA = set(courses_ordre[n_val:n_testA])
    courses_testB = set(courses_ordre[n_testA:])
    df_train = df[df["course_id"].isin(courses_train)].reset_index(drop=True)
    df_val = df[df["course_id"].isin(courses_val)].reset_index(drop=True)
    df_testA = df[df["course_id"].isin(courses_testA)].reset_index(drop=True)
    df_testB = df[df["course_id"].isin(courses_testB)].reset_index(drop=True)
    del df
    gc.collect()

    lib.log("\n[6/7] Decoupage chronologique strict (70/15/7.5/7.5) :")
    lib.log(f"   TRAIN      : {len(df_train)} lignes / {df_train['course_id'].nunique()} courses")
    lib.log(f"   VALIDATION : {len(df_val)} lignes / {df_val['course_id'].nunique()} courses")
    lib.log(f"   TEST A     : {len(df_testA)} lignes / {df_testA['course_id'].nunique()} courses")
    lib.log(f"   TEST B     : {len(df_testB)} lignes / {df_testB['course_id'].nunique()} courses")

    y_train_place = df_train["cible_place"].values
    y_val_place = df_val["cible_place"].values
    y_testA_place = df_testA["cible_place"].values
    y_testB_place = df_testB["cible_place"].values
    y_train_gagnant = df_train["est_gagnant"].values
    y_val_gagnant = df_val["est_gagnant"].values
    y_testA_gagnant = df_testA["est_gagnant"].values
    y_testB_gagnant = df_testB["est_gagnant"].values

    variables_numeriques_v3 = lib.variables_numeriques_v3(VARIABLES_NUMERIQUES)
    X_train_v3 = lib.preparer_matrice(df_train, variables_numeriques_v3, VARIABLES_CATEGORIELLES)
    colonnes_v3 = X_train_v3.columns
    X_val_v3 = lib.preparer_matrice(df_val, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)
    X_testA_v3 = lib.preparer_matrice(df_testA, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)
    X_testB_v3 = lib.preparer_matrice(df_testB, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)

    colonnes_numeriques_v3 = [c for c in colonnes_v3 if c in variables_numeriques_v3]
    degenerees_v3 = lib.colonnes_degenerees(X_train_v3, colonnes_numeriques_v3)
    if degenerees_v3:
        lib.log(f"\n   ATTENTION : {len(degenerees_v3)} colonne(s) v3 degeneree(s) sur TRAIN, exclue(s) : {degenerees_v3}")
        colonnes_v3_filtrees = [c for c in colonnes_v3 if c not in degenerees_v3]
        X_train_v3 = X_train_v3[colonnes_v3_filtrees]
        X_val_v3 = X_val_v3[colonnes_v3_filtrees]
        X_testA_v3 = X_testA_v3[colonnes_v3_filtrees]
        X_testB_v3 = X_testB_v3[colonnes_v3_filtrees]

    X_train_geneal_seul = df_train[COLONNES_GENEALOGIE].reset_index(drop=True).astype("float32")
    X_val_geneal_seul = df_val[COLONNES_GENEALOGIE].reset_index(drop=True).astype("float32")
    X_testA_geneal_seul = df_testA[COLONNES_GENEALOGIE].reset_index(drop=True).astype("float32")
    X_testB_geneal_seul = df_testB[COLONNES_GENEALOGIE].reset_index(drop=True).astype("float32")
    degenerees_geneal = lib.colonnes_degenerees(X_train_geneal_seul, COLONNES_GENEALOGIE)
    if degenerees_geneal:
        lib.log(f"   ATTENTION : {len(degenerees_geneal)} colonne(s) genealogie degeneree(s) sur TRAIN, exclue(s) : {degenerees_geneal}")
        colonnes_geneal_filtrees = [c for c in COLONNES_GENEALOGIE if c not in degenerees_geneal]
        X_train_geneal_seul = X_train_geneal_seul[colonnes_geneal_filtrees]
        X_val_geneal_seul = X_val_geneal_seul[colonnes_geneal_filtrees]
        X_testA_geneal_seul = X_testA_geneal_seul[colonnes_geneal_filtrees]
        X_testB_geneal_seul = X_testB_geneal_seul[colonnes_geneal_filtrees]
    else:
        colonnes_geneal_filtrees = list(COLONNES_GENEALOGIE)

    X_train_v3_geneal = pd.concat([X_train_v3.reset_index(drop=True), X_train_geneal_seul.reset_index(drop=True)], axis=1)
    X_val_v3_geneal = pd.concat([X_val_v3.reset_index(drop=True), X_val_geneal_seul.reset_index(drop=True)], axis=1)
    X_testA_v3_geneal = pd.concat([X_testA_v3.reset_index(drop=True), X_testA_geneal_seul.reset_index(drop=True)], axis=1)
    X_testB_v3_geneal = pd.concat([X_testB_v3.reset_index(drop=True), X_testB_geneal_seul.reset_index(drop=True)], axis=1)

    lib.log(f"\n   Matrice v3+genealogie : {X_train_v3_geneal.shape[1]} colonnes "
            f"({len(colonnes_geneal_filtrees)} variables de genealogie).")

    course_id_train = df_train["course_id"].reset_index(drop=True)

    del X_train_geneal_seul, X_val_geneal_seul, X_testA_geneal_seul, X_testB_geneal_seul
    gc.collect()

    checkpoint = {
        "X_train_v3_geneal": X_train_v3_geneal,
        "X_val_v3_geneal": X_val_v3_geneal,
        "X_testA_v3_geneal": X_testA_v3_geneal,
        "X_testB_v3_geneal": X_testB_v3_geneal,
        "y_train_place": y_train_place, "y_val_place": y_val_place,
        "y_testA_place": y_testA_place, "y_testB_place": y_testB_place,
        "y_train_gagnant": y_train_gagnant, "y_val_gagnant": y_val_gagnant,
        "y_testA_gagnant": y_testA_gagnant, "y_testB_gagnant": y_testB_gagnant,
        "df_val": df_val.copy(), "df_testA": df_testA.copy(), "df_testB": df_testB.copy(),
        "course_id_train": course_id_train,
        "colonnes_genealogie": colonnes_geneal_filtrees,
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    lib.log(f"\n[7/7] Checkpoint sauvegarde : {CHECKPOINT_PATH}")
    lib.log("\nPHASE 1 (piste 6) TERMINEE -- la phase 2 peut entrainer l'architecture a deux etages.")


if __name__ == "__main__":
    main()
