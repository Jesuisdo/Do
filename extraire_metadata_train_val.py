# -*- coding: utf-8 -*-
"""
extraire_metadata_train_val.py -- Etape preparatoire a la piste n3 (modele
specifique handicaps / grands champs), demandee par Dorian le 25/08/2026.

PROBLEME : le checkpoint-v3 (run production 32871825116) ne contient QUE
les matrices numeriques (X_train_v3/X_val_v3) et les cibles (y_*) pour
TRAIN et VALIDATION -- la metadonnee necessaire pour segmenter par
"handicap" ou "grand champ (13+ partants)" (categorie_particularite,
nb_partants_reel) n'a ete sauvegardee QUE pour TEST (df_test), car ce sont
des colonnes meta (variables_config.COLONNES_META), pas des variables
predictives directement exploitables sous cette forme.

CE SCRIPT NE RE-ENTRAINE AUCUN MODELE ET NE MODIFIE RIEN. Il reconstruit
EXACTEMENT le meme pipeline que entrainer_v3_phase1.py (etapes 1-5 :
chargement Supabase, resolution d'identite, construction des 109+22
variables, decoupage chronologique strict identique), puis VERIFIE que les
matrices reconstruites sont RIGOUREUSEMENT IDENTIQUES (egalite exacte,
pas juste memes dimensions) aux matrices deja persistees dans le
checkpoint du run production 32871825116, avant d'extraire et de
sauvegarder la seule metadonnee manquante (course_id, date_course,
nb_partants_reel, categorie_particularite) pour TRAIN et VALIDATION.

Si la verification d'egalite echoue (ex. de nouvelles donnees auraient ete
ajoutees a Supabase depuis le run production, ou un backfill historique
aurait modifie des courses anterieures), le script s'arrete EN ERREUR
plutot que de continuer sur des donnees potentiellement differentes -- il
ne faut SURTOUT PAS que la piste n3 utilise un decoupage train/val/test
different de celui deja valide par Dorian ("utilise exactement les memes
donnees historiques").
"""
import gc
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

from identite_chevaux import resoudre_identite_chevaux
from variables_historiques import construire_variables, trier_chronologiquement
from variables_config import VARIABLES_NUMERIQUES, VARIABLES_CATEGORIELLES

CHECKPOINT_EXISTANT_PATH = "checkpoint_v3_phase1.pkl"  # deja telecharge par le workflow (run 32871825116)
METADATA_OUT_PATH = "metadata_train_val_v3.pkl"


def verifier_egalite(nom, nouveau, ancien):
    """Compare deux objets (DataFrame ou array) et journalise le resultat.
    Ne leve jamais d'exception : une comparaison impossible (colonnes
    manquantes, shape differente) est traitee comme un echec (DIFFERENT),
    pas comme un crash, pour que le rapport reste lisible."""
    try:
        if isinstance(nouveau, pd.DataFrame):
            identique = bool(nouveau.reset_index(drop=True).equals(ancien.reset_index(drop=True)))
        else:
            identique = bool(np.array_equal(np.asarray(nouveau), np.asarray(ancien)))
    except Exception as e:
        identique = False
        lib.log(f"  Verification {nom} : ECHEC DE COMPARAISON ({type(e).__name__}: {e})")
        return False
    shape_nouveau = getattr(nouveau, "shape", len(nouveau))
    shape_ancien = getattr(ancien, "shape", len(ancien))
    lib.log(f"  Verification {nom} : {'IDENTIQUE' if identique else 'DIFFERENT !!'} "
            f"(nouveau shape={shape_nouveau}, ancien shape={shape_ancien})")
    return identique


def main():
    if not lib.DEPENDANCES_LOURDES_DISPONIBLES:
        raise RuntimeError(
            "psycopg2 et/ou scikit-learn ne sont pas installes. Ce script doit tourner dans "
            "l'environnement GitHub Actions du workflow dedie, pas en local."
        )

    lib.log("=" * 100)
    lib.log("EXTRACTION METADATA TRAIN/VALIDATION -- preparation piste n3 (Dorian, 25/08/2026)")
    lib.log("Reconstruction du MEME pipeline que entrainer_v3_phase1.py, avec verification d'egalite")
    lib.log("stricte avant d'extraire la seule metadonnee manquante (course_id, date_course,")
    lib.log("nb_partants_reel, categorie_particularite) pour TRAIN et VALIDATION.")
    lib.log("Aucun modele n'est entraine ici. Aucune donnee n'est modifiee.")
    lib.log("=" * 100)

    lib.log("\nChargement de l'ancien checkpoint (run production 32871825116) pour comparaison...")
    with open(CHECKPOINT_EXISTANT_PATH, "rb") as f:
        ancien = pickle.load(f)

    lib.log("\n[1/5] Chargement des donnees brutes depuis Supabase (meme requete que phase1, inchangee)...")
    lignes = lib.charger_donnees_brutes()
    lib.log(f"  {len(lignes)} lignes brutes chargees.")

    lib.log("\n[2/5] Resolution d'identite des chevaux (identique a phase1)...")
    horse_uids, rapport_identite = resoudre_identite_chevaux(lignes)
    for l, uid in zip(lignes, horse_uids):
        l["horse_uid"] = uid
    lib.log(f"  {rapport_identite['n_chevaux_distincts_resolus']} chevaux distincts resolus.")

    lib.log("\n[3/5] Construction des 109 variables point-in-time (identique a phase1)...")
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

    lib.log("\n[4/5] Ajout des variables relatives au champ (identique a phase1)...")
    df = lib.ajouter_variables_relatives(df, lib.VARIABLES_RELATIVES_CIBLES)

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

    lib.log(f"\n[5/5] Decoupage chronologique (identique a phase1) : "
            f"TRAIN={len(df_train)} lignes/{df_train['course_id'].nunique()} courses, "
            f"VALIDATION={len(df_val)} lignes/{df_val['course_id'].nunique()} courses, "
            f"TEST={len(df_test)} lignes/{df_test['course_id'].nunique()} courses.")

    variables_numeriques_v3 = lib.variables_numeriques_v3(VARIABLES_NUMERIQUES)
    X_train_v3 = lib.preparer_matrice(df_train, variables_numeriques_v3, VARIABLES_CATEGORIELLES)
    colonnes_v3 = X_train_v3.columns
    X_val_v3 = lib.preparer_matrice(df_val, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)
    X_test_v3 = lib.preparer_matrice(df_test, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)

    # temoin_aleatoire : ajoute avant le controle des degenerees, identique a phase1.
    # Sa valeur exacte n'a aucune importance (colonne de bruit) -- elle sera exclue
    # de la comparaison d'egalite ci-dessous.
    rng = np.random.RandomState(lib.RANDOM_SEED)
    X_train_v3 = X_train_v3.copy()
    X_val_v3 = X_val_v3.copy()
    X_test_v3 = X_test_v3.copy()
    X_train_v3["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_train_v3))
    X_val_v3["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_val_v3))
    X_test_v3["temoin_aleatoire"] = rng.uniform(0, 1, size=len(X_test_v3))

    colonnes_numeriques_v3 = [c for c in colonnes_v3 if c in variables_numeriques_v3] + ["temoin_aleatoire"]
    degenerees = lib.colonnes_degenerees(X_train_v3, colonnes_numeriques_v3)
    if degenerees:
        lib.log(f"\n  {len(degenerees)} colonne(s) degeneree(s) detectee(s) (identique a phase1 attendu) : {degenerees}")
    colonnes_v3_filtrees = [c for c in colonnes_v3 if c not in degenerees] + ["temoin_aleatoire"]
    X_train_v3 = X_train_v3[colonnes_v3_filtrees]
    X_val_v3 = X_val_v3[colonnes_v3_filtrees]
    X_test_v3 = X_test_v3[colonnes_v3_filtrees]

    y_train_place = df_train["cible_place"].values
    y_val_place = df_val["cible_place"].values
    y_train_gagnant = df_train["est_gagnant"].values
    y_val_gagnant = df_val["est_gagnant"].values

    lib.log("\n" + "=" * 100)
    lib.log("=== VERIFICATION D'EGALITE STRICTE avec le checkpoint existant (run 32871825116) ===")
    lib.log("=" * 100)

    # temoin_aleatoire exclu : colonne de bruit non deterministe (sa valeur exacte
    # n'a aucune importance), seules les VRAIES variables sont comparees.
    cols_a_comparer = [c for c in colonnes_v3_filtrees if c != "temoin_aleatoire"]
    checks = [
        verifier_egalite("X_train_v3 (hors temoin_aleatoire)", X_train_v3[cols_a_comparer], ancien["X_train_v3"][cols_a_comparer]),
        verifier_egalite("X_val_v3 (hors temoin_aleatoire)", X_val_v3[cols_a_comparer], ancien["X_val_v3"][cols_a_comparer]),
        verifier_egalite("X_test_v3 (hors temoin_aleatoire)", X_test_v3[cols_a_comparer], ancien["X_test_v3"][cols_a_comparer]),
        verifier_egalite("y_train_place", y_train_place, ancien["y_train_place"]),
        verifier_egalite("y_val_place", y_val_place, ancien["y_val_place"]),
        verifier_egalite("y_train_gagnant", y_train_gagnant, ancien["y_train_gagnant"]),
        verifier_egalite("y_val_gagnant", y_val_gagnant, ancien["y_val_gagnant"]),
    ]

    if not all(checks):
        raise RuntimeError(
            "ARRET : les donnees reconstruites NE SONT PAS identiques au checkpoint existant "
            "(run production 32871825116). Cela signifie que Supabase a change depuis ce run "
            "(nouvelles donnees ajoutees, backfill historique, etc.). Poursuivre romprait la "
            "garantie 'memes donnees historiques, meme decoupage' explicitement demandee par "
            "Dorian pour la piste n3. NE PAS relancer la piste n3 sans clarifier d'abord ce qui "
            "a change entre les deux runs."
        )
    lib.log("\n  TOUTES LES VERIFICATIONS PASSENT : les donnees reconstruites sont rigoureusement identiques")
    lib.log("  au checkpoint existant. On peut donc extraire la metadonnee manquante en toute confiance --")
    lib.log("  meme decoupage, memes lignes, aucune donnee supplementaire ni differente.")

    meta_train = df_train[["course_id", "date_course", "nb_partants_reel", "categorie_particularite"]].reset_index(drop=True)
    meta_val = df_val[["course_id", "date_course", "nb_partants_reel", "categorie_particularite"]].reset_index(drop=True)

    with open(METADATA_OUT_PATH, "wb") as f:
        pickle.dump({"meta_train": meta_train, "meta_val": meta_val}, f, protocol=pickle.HIGHEST_PROTOCOL)
    lib.log(f"\nMetadata sauvegardee : {METADATA_OUT_PATH} (meta_train={meta_train.shape}, meta_val={meta_val.shape})")
    lib.log("\nEXTRACTION TERMINEE -- pret pour piste3_handicap_grandschamps.py.")


if __name__ == "__main__":
    main()
