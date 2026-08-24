# -*- coding: utf-8 -*-
"""
test_pipeline_v3_resumable.py â Test "petit echantillon synthetique" du
pipeline v3 decoupe en 2 phases, demande par Dorian le 24/08/2026 AVANT de
relancer le gros run reel. Ne se connecte PAS a Supabase (donnees 100%
synthetiques, generees en local). Necessite scikit-learn reel (contrairement
a test_entrainer_et_evaluer_v3.py qui ne teste que la logique pandas pure) â
tourne donc via le workflow GitHub Actions test-pipeline-v3-resumable.yml,
pas en local.

Verifie explicitement les 4 points demandes :
  1. L'artefact (checkpoint) est correctement cree.
  2. Il peut etre restaure (round-trip pickle, egalite des donnees).
  3. Le pipeline reprend correctement a l'etape 7 (entrainement GBM sur les
     donnees restaurees, sans reconstruire quoi que ce soit).
  4. Une variable constante (degeneree) ne fait plus planter le modele â
     AVEC un contre-exemple negatif qui prouve que sans le filet de
     securite, ca plante bien (donc que le fix adresse le vrai bug, pas une
     coincidence).
"""
import pickle
import tempfile
import os

import numpy as np
import pandas as pd

import v3_lib as lib

from sklearn.ensemble import HistGradientBoostingClassifier

from variables_config import VARIABLES_NUMERIQUES, VARIABLES_CATEGORIELLES

ECHECS = []


def verifie(condition, message):
    statut = "OK" if condition else "ECHEC"
    print(f"  [{statut}] {message}", flush=True)
    if not condition:
        ECHECS.append(message)


def _df_synthetique(n_courses=40, seed=0, avec_variable_globalement_constante=True):
    rng = np.random.RandomState(seed)
    lignes = []
    for c in range(n_courses):
        nb_partants = rng.randint(4, 16)
        seuil = 2 if nb_partants <= 7 else 3
        for num in range(1, nb_partants + 1):
            row = {v: rng.uniform(0, 1) if rng.random() > 0.3 else np.nan for v in VARIABLES_NUMERIQUES}
            for v in VARIABLES_CATEGORIELLES:
                row[v] = rng.choice(["A", "B", "C", None])
            row["course_id"] = f"C{c}"
            row["numero"] = num
            row["nb_partants_reel"] = nb_partants
            row["seuil"] = seuil
            row["musique_dernier"] = rng.randint(1, 15)
            row["carriere_taux_victoire"] = rng.uniform(0, 1)
            row["gains_carriere"] = rng.uniform(0, 100000)
            row["interne_jockey_taux_victoire"] = rng.uniform(0, 1)
            row["categorie_particularite"] = rng.choice(["HANDICAP", "", None])
            row["distance_bucket"] = rng.choice(["court", "moyen", "long"])
            row["terrain_bucket"] = rng.choice(["bon", "souple", "lourd"])
            row["date_course"] = "2026-01-01"
            row["position_arrivee"] = None
            lignes.append(row)
    df = pd.DataFrame(lignes)
    for c, grp in df.groupby("course_id"):
        ordre = rng.permutation(len(grp)) + 1
        df.loc[grp.index, "position_arrivee"] = ordre
    df["cible_place"] = (df["position_arrivee"] <= df["seuil"]).astype(int)
    df["est_gagnant"] = (df["position_arrivee"] == 1).astype(int)
    if avec_variable_globalement_constante:
        # Simule le bug reel (montant_allocation) : une variable dont la
        # valeur est identique pour TOUTES les lignes -> degeneree.
        df["variable_test_degeneree"] = 42.0
    df = lib.ajouter_variables_relatives(df, lib.VARIABLES_RELATIVES_CIBLES)
    return df


def main():
    print("=" * 100, flush=True)
    print("TEST PIPELINE v3 RESUMABLE â donnees synthetiques, pas de connexion Supabase", flush=True)
    print("=" * 100, flush=True)

    df = _df_synthetique(n_courses=60, seed=1)
    n_train = int(df["course_id"].nunique() * 0.7)
    n_val = int(df["course_id"].nunique() * 0.85)
    courses = df["course_id"].unique()
    df_train = df[df["course_id"].isin(courses[:n_train])].reset_index(drop=True)
    df_val = df[df["course_id"].isin(courses[n_train:n_val])].reset_index(drop=True)
    df_test = df[df["course_id"].isin(courses[n_val:])].reset_index(drop=True)

    variables_numeriques_v3 = lib.variables_numeriques_v3(VARIABLES_NUMERIQUES) + ["variable_test_degeneree"]

    print("\n[Etape phase-1-equivalente] Construction des matrices + filtrage des colonnes degenerees...", flush=True)
    X_train = lib.preparer_matrice(df_train, variables_numeriques_v3, VARIABLES_CATEGORIELLES)
    colonnes_ref = X_train.columns
    X_val = lib.preparer_matrice(df_val, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_ref)
    X_test = lib.preparer_matrice(df_test, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_ref)

    # --- CONTRE-EXEMPLE NEGATIF : sans le filet de securite, la colonne
    # degeneree fait bien planter le binning HistGradientBoosting (preuve
    # que le fix adresse le vrai bug, pas une coincidence). ---
    print("\n[Contre-exemple negatif] Verification que SANS filtrage, la colonne degeneree plante bien...", flush=True)
    a_plante_sans_filtre = False
    try:
        HistGradientBoostingClassifier(random_state=0, max_iter=10).fit(
            X_train, df_train["cible_place"].values
        )
    except ValueError as e:
        a_plante_sans_filtre = True
        print(f"    (plantage attendu obtenu : {type(e).__name__}: {str(e)[:80]}...)", flush=True)
    verifie(a_plante_sans_filtre, "[TEST 0 - controle negatif] Sans filet de securite, la colonne degeneree fait "
                                   "bien planter le modele (confirme que le fix cible le vrai bug)")

    degenerees = lib.colonnes_degenerees(X_train, [c for c in colonnes_ref if c in variables_numeriques_v3])
    verifie("variable_test_degeneree" in degenerees,
            "[TEST 4a] La variable globalement constante est bien detectee comme degeneree")
    colonnes_filtrees = [c for c in colonnes_ref if c not in degenerees]
    X_train = X_train[colonnes_filtrees]
    X_val = X_val[colonnes_filtrees]
    X_test = X_test[colonnes_filtrees]

    a_plante_avec_filtre = False
    try:
        HistGradientBoostingClassifier(random_state=0, max_iter=10).fit(
            X_train, df_train["cible_place"].values
        )
    except ValueError:
        a_plante_avec_filtre = True
    verifie(not a_plante_avec_filtre,
            "[TEST 4b] APRES filtrage, le modele s'entraine sans planter (variable constante neutralisee)")

    checkpoint = {
        "X_train_v3": X_train, "X_val_v3": X_val, "X_test_v3": X_test,
        "y_train_place": df_train["cible_place"].values,
        "y_val_place": df_val["cible_place"].values,
        "y_test_place": df_test["cible_place"].values,
        "y_train_gagnant": df_train["est_gagnant"].values,
        "y_val_gagnant": df_val["est_gagnant"].values,
        "df_test": df_test,
        "colonnes_relatives": [f"{v}_rang_course" for v in lib.VARIABLES_RELATIVES_CIBLES],
    }

    print("\n[TEST 1] Creation de l'artefact (checkpoint pickle)...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        chemin = os.path.join(tmpdir, "checkpoint_test.pkl")
        with open(chemin, "wb") as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        taille = os.path.getsize(chemin)
        verifie(os.path.exists(chemin) and taille > 0, f"[TEST 1] Artefact cree sur disque ({taille} octets)")

        print("\n[TEST 2] Restauration de l'artefact...", flush=True)
        with open(chemin, "rb") as f:
            checkpoint_restaure = pickle.load(f)
        verifie(
            checkpoint_restaure["X_train_v3"].shape == X_train.shape
            and list(checkpoint_restaure["X_train_v3"].columns) == list(X_train.columns)
            and checkpoint_restaure["X_train_v3"].equals(X_train)
            and np.array_equal(checkpoint_restaure["y_train_place"], checkpoint["y_train_place"])
            and checkpoint_restaure["df_test"].equals(df_test),
            "[TEST 2] Artefact restaure correctement (memes formes, memes colonnes, memes valeurs)"
        )

    print("\n[TEST 3] Reprise a l'etape 7 depuis l'artefact restaure (entrainement GBM + analyse d'erreurs)...", flush=True)
    X_train_r = checkpoint_restaure["X_train_v3"]
    X_val_r = checkpoint_restaure["X_val_v3"]
    X_test_r = checkpoint_restaure["X_test_v3"]
    y_train_r = checkpoint_restaure["y_train_place"]
    y_val_r = checkpoint_restaure["y_val_place"]
    df_test_r = checkpoint_restaure["df_test"]

    reprise_ok = True
    try:
        params, auc = lib.entrainer_gbm_avec_grille(
            X_train_r, y_train_r, X_val_r, y_val_r,
            [{"max_depth": 3, "max_iter": 30, "learning_rate": 0.1, "l2_regularization": 1.0, "min_samples_leaf": 5}],
            "v3-place-test",
        )
        gbm = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
        gbm.fit(X_train_r, y_train_r)
        proba = gbm.predict_proba(X_test_r)[:, 1]
        df_test_r = df_test_r.copy()
        df_test_r["proba_test"] = proba
        df_test_r["rang_test"] = df_test_r.groupby("course_id")["proba_test"].rank(method="min", ascending=False)
        lib.log_analyse_erreurs(df_test_r, "rang_test", "proba_test", "GBM test-reprise")
    except Exception as e:
        reprise_ok = False
        print(f"    ERREUR pendant la reprise : {type(e).__name__}: {e}", flush=True)
    verifie(reprise_ok, "[TEST 3] Le pipeline reprend correctement a l'etape 7 depuis le checkpoint restaure "
                         "(entrainement + analyse d'erreurs completes, sans reconstruire les donnees)")

    print("\n" + "=" * 100, flush=True)
    if ECHECS:
        print(f"ECHEC : {len(ECHECS)} verification(s) ont echoue :", flush=True)
        for e in ECHECS:
            print(f"  - {e}", flush=True)
        raise SystemExit(1)
    else:
        print("TOUTES LES VERIFICATIONS PASSENT â le pipeline resumable est valide sur echantillon synthetique.", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
