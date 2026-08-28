# -*- coding: utf-8 -*-
"""
entrainer_v3_phase1_equipement.py -- PHASE 1/2 de la piste 3 (changement
d'equipement du jour), demandee par Dorian le 28/08/2026 apres l'abandon
de la piste 2 (vitesse chronometrique -- donnees 100% absentes en base) et
l'audit qui a montre que seul l'axe oeilleres est exploitable (deferre
quasi jamais renseigne, 0,01% de couverture, et absent aussi de la table
`partants` en direct, entierement vide -- voir variables_equipement.py
pour le detail de cet audit).

Copie STRICTE de entrainer_v3_phase1_genealogie.py (meme chargement, meme
resolution d'identite, memes 109+22 variables v3 point-in-time inchangees,
meme decoupage chronologique 70/15/15) -- AUCUNE modification de la
construction des variables v3 existantes, pour que le baseline
"v3-gagnant" et le candidat "B" restent exactement reproductibles a
l'identique des runs precedents. Le SEUL ajout est la construction, sur
les MEMES lignes deja triees chronologiquement, des 12 variables
d'equipement point-in-time (voir variables_equipement.py).

Ne se connecte a Supabase QU'EN LECTURE (meme REQUETE PLAT-only que
v3_lib.py -- rp.oeilleres y est deja selectionne, aucune colonne
supplementaire necessaire). N'ecrit rien en base.
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
from variables_equipement import construire_variables_equipement, COLONNES_EQUIPEMENT

CHECKPOINT_PATH = "checkpoint_v3_phase1_equipement.pkl"


def main():
    if not lib.DEPENDANCES_LOURDES_DISPONIBLES:
        raise RuntimeError(
            "psycopg2 et/ou scikit-learn ne sont pas installes. Ce script doit tourner dans "
            "l'environnement GitHub Actions du workflow dedie, pas en local."
        )
    lib.log("=" * 100)
    lib.log("PISTE 3 -- EQUIPEMENT -- PHASE 1/2 -- CHARGEMENT, VARIABLES v3 + OEILLERES POINT-IN-TIME, SPLIT -- 28/08/2026")
    lib.log("=" * 100)

    lib.log("\n[1/7] Chargement des donnees brutes depuis Supabase (PLAT, toutes lignes, dont oeilleres)...")
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

    lib.log("\n[4/7] Construction des 12 variables d'equipement (oeilleres) point-in-time (piste 3, NOUVEAU)...")
    features_equip = construire_variables_equipement(lignes_triees)
    df_equip = pd.DataFrame(features_equip)
    assert len(df_equip) == len(df), "desalignement equipement vs variables v3 -- meme liste triee attendue"
    for col in COLONNES_EQUIPEMENT:
        df[col] = df_equip[col].values
    del lignes, lignes_triees, features, features_equip, df_equip, horse_uids
    gc.collect()

    df = df[df["position_arrivee"].notna()].copy()
    df = df[df["nb_partants_reel"] >= 3].reset_index(drop=True)
    df["est_gagnant"] = (df["position_arrivee"] == 1).astype(int)
    df["cible_place"] = (df["position_arrivee"] <= df["seuil"]).astype(int)
    lib.log(f"   {len(df)} lignes, {df['course_id'].nunique()} courses apres filtrage.")

    lib.log(f"\n[5/7] Ajout des {2*len(lib.VARIABLES_RELATIVES_CIBLES)} variables relatives au champ v3 "
            f"(rang + z-score intra-course, inchange)...")
    df = lib.ajouter_variables_relatives(df, lib.VARIABLES_RELATIVES_CIBLES)

    # --- decoupage chronologique STRICT, identique a v2/v3/genealogie : 70/15/15 ---
    df = df.sort_values(["date_course", "course_id"]).reset_index(drop=True)
    courses_ordre = df["course_id"].drop_duplicates().tolist()
    n = len(courses_ordre)
    n_train = int(n * 0.70)
    n_val = int(n * 0.85)
    courses_train = set(courses_ordre[:n_train])
    courses_val = set(courses_ordre[n_train:n_val])
    df_train = df[df["course_id"].isin(courses_train)].reset_index(drop=True)
    df_val = df[df["course_id"].isin(courses_val)].reset_index(drop=True)
    # (le jeu TEST n'est pas construit ici : validation uniquement, comme
    #  demande -- aucun TEST A/B lance.)
    del df
    gc.collect()

    lib.log("\n[6/7] Decoupage chronologique strict (identique a v2/v3/genealogie, memes bornes de courses) :")
    lib.log(f"   TRAIN      : {len(df_train)} lignes / {df_train['course_id'].nunique()} courses")
    lib.log(f"   VALIDATION : {len(df_val)} lignes / {df_val['course_id'].nunique()} courses")

    y_train_place = df_train["cible_place"].values
    y_val_place = df_val["cible_place"].values
    y_train_gagnant = df_train["est_gagnant"].values
    y_val_gagnant = df_val["est_gagnant"].values

    variables_numeriques_v3 = lib.variables_numeriques_v3(VARIABLES_NUMERIQUES)
    X_train_v3 = lib.preparer_matrice(df_train, variables_numeriques_v3, VARIABLES_CATEGORIELLES)
    colonnes_v3 = X_train_v3.columns
    X_val_v3 = lib.preparer_matrice(df_val, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)

    # --- filet de securite generique contre les colonnes degenerees (v3
    # + equipement ensemble, controle sur TRAIN uniquement) ---
    colonnes_numeriques_v3 = [c for c in colonnes_v3 if c in variables_numeriques_v3]
    degenerees_v3 = lib.colonnes_degenerees(X_train_v3, colonnes_numeriques_v3)
    if degenerees_v3:
        lib.log(f"\n   ATTENTION : {len(degenerees_v3)} colonne(s) v3 degeneree(s) sur TRAIN, exclue(s) : {degenerees_v3}")
    colonnes_v3_filtrees = [c for c in colonnes_v3 if c not in degenerees_v3]
    X_train_v3 = X_train_v3[colonnes_v3_filtrees]
    X_val_v3 = X_val_v3[colonnes_v3_filtrees]

    X_train_equip_seul = df_train[COLONNES_EQUIPEMENT].reset_index(drop=True).astype("float32")
    X_val_equip_seul = df_val[COLONNES_EQUIPEMENT].reset_index(drop=True).astype("float32")
    degenerees_equip = lib.colonnes_degenerees(X_train_equip_seul, COLONNES_EQUIPEMENT)
    if degenerees_equip:
        lib.log(f"   ATTENTION : {len(degenerees_equip)} colonne(s) equipement degeneree(s) sur TRAIN, exclue(s) : {degenerees_equip}")
    colonnes_equip_filtrees = [c for c in COLONNES_EQUIPEMENT if c not in degenerees_equip]
    X_train_equip_seul = X_train_equip_seul[colonnes_equip_filtrees]
    X_val_equip_seul = X_val_equip_seul[colonnes_equip_filtrees]

    X_train_v3_equip = pd.concat(
        [X_train_v3.reset_index(drop=True), X_train_equip_seul.reset_index(drop=True)], axis=1)
    X_val_v3_equip = pd.concat(
        [X_val_v3.reset_index(drop=True), X_val_equip_seul.reset_index(drop=True)], axis=1)

    colonnes_relatives = [c for c in ([f"{v}_rang_course" for v in lib.VARIABLES_RELATIVES_CIBLES] +
                                       [f"{v}_z_course" for v in lib.VARIABLES_RELATIVES_CIBLES]) if c not in degenerees_v3]

    lib.log(f"\n   Matrice v3 (baseline/B) : {X_train_v3.shape[1]} colonnes.")
    lib.log(f"   Matrice v3+equipement (B+equipement) : {X_train_v3_equip.shape[1]} colonnes "
            f"({len(colonnes_equip_filtrees)} nouvelles variables d'equipement).")

    # --- couverture reelle des nouvelles features (demandee explicitement
    # par Dorian) : calculee ici sur l'ensemble TRAIN+VAL, avant tout
    # entrainement -- lecture seule. ---
    lib.log("\n   Couverture des nouvelles variables d'equipement (TRAIN+VALIDATION, avant filtrage degenerescence) :")
    df_couverture = pd.concat([df_train[COLONNES_EQUIPEMENT], df_val[COLONNES_EQUIPEMENT]], axis=0)
    n_total_couverture = len(df_couverture)
    for col in COLONNES_EQUIPEMENT:
        n_non_null = int(df_couverture[col].notna().sum())
        pct = round(100 * n_non_null / n_total_couverture, 1) if n_total_couverture else float("nan")
        if col.startswith("equip_") and set(df_couverture[col].dropna().unique().tolist()) <= {0, 1, 0.0, 1.0}:
            n_positifs = int((df_couverture[col] == 1).sum())
            lib.log(f"     {col:48s} couverture={pct:>5}% ({n_non_null}/{n_total_couverture})  n_positifs={n_positifs}")
        else:
            lib.log(f"     {col:48s} couverture={pct:>5}% ({n_non_null}/{n_total_couverture})")

    # --- groupes pour l'objectif de ranking (B/lambdarank_graded), meme
    # convention que entrainer_v3_phase1_genealogie.py : contiguite
    # garantie par le tri ["date_course", "course_id"] ci-dessus. ---
    course_id_train = df_train["course_id"].reset_index(drop=True)
    df_val_complet = df_val.copy()

    del X_train_equip_seul, X_val_equip_seul
    gc.collect()

    checkpoint = {
        "X_train_v3": X_train_v3,
        "X_val_v3": X_val_v3,
        "X_train_v3_equip": X_train_v3_equip,
        "X_val_v3_equip": X_val_v3_equip,
        "y_train_place": y_train_place,
        "y_val_place": y_val_place,
        "y_train_gagnant": y_train_gagnant,
        "y_val_gagnant": y_val_gagnant,
        "df_val": df_val_complet,
        "course_id_train": course_id_train,
        "colonnes_relatives": colonnes_relatives,
        "colonnes_equipement": colonnes_equip_filtrees,
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    lib.log(f"\n[7/7] Checkpoint sauvegarde : {CHECKPOINT_PATH} "
            f"(X_train_v3={X_train_v3.shape}, X_train_v3_equip={X_train_v3_equip.shape}, "
            f"X_val_v3={X_val_v3.shape}, df_val={df_val_complet.shape})")
    lib.log("\nPHASE 1 (piste 3, equipement) TERMINEE -- la phase 2 peut entrainer v3-gagnant, B et B+equipement depuis ce checkpoint.")


if __name__ == "__main__":
    main()
