# -*- coding: utf-8 -*-
"""
entrainer_v3_phase2.py â PHASE 2/2 du pipeline v3 (etapes 7-9) : entraine
les deux modeles v3 (variables enrichies, cible='place' puis cible=
'gagnant'), leurs analyses d'erreurs, l'importance par permutation, et la
comparaison finale avec le modele v2 et la baseline de production.

NE SE CONNECTE PAS A SUPABASE. Charge uniquement le checkpoint produit par
entrainer_v3_phase1.py (checkpoint_v3_phase1.pkl, telecharge en artefact
GitHub Actions par le workflow avant ce script). Si cette phase plante pour
une raison quelconque, on peut la relancer seule (le job GitHub Actions
"phase2" peut etre re-execute independamment via "Re-run failed jobs")
sans jamais avoir a refaire le chargement/la construction des variables de
la phase 1.
"""
import gc
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.inspection import permutation_importance
except ImportError:
    pass

CHECKPOINT_PATH = "checkpoint_v3_phase1.pkl"


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError(
            "scikit-learn n'est pas installe. Ce script doit tourner dans "
            "l'environnement GitHub Actions du workflow dedie, pas en local."
        )
    lib.log("=" * 100)
    lib.log("PHASE 2/2 â MODELES v3 (place, gagnant) + COMPARAISON FINALE â 24/08/2026")
    lib.log("=" * 100)

    lib.log(f"\nChargement du checkpoint {CHECKPOINT_PATH} (produit par la phase 1, aucune reconnexion "
            "a Supabase)...")
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    X_train_v3 = checkpoint["X_train_v3"]
    X_val_v3 = checkpoint["X_val_v3"]
    X_test_v3 = checkpoint["X_test_v3"]
    y_train_place = checkpoint["y_train_place"]
    y_val_place = checkpoint["y_val_place"]
    y_test_place = checkpoint["y_test_place"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    y_val_gagnant = checkpoint["y_val_gagnant"]
    df_test = checkpoint["df_test"]
    colonnes_relatives = checkpoint["colonnes_relatives"]
    lib.log(f"  Checkpoint charge : X_train_v3={X_train_v3.shape}, X_val_v3={X_val_v3.shape}, "
            f"X_test_v3={X_test_v3.shape}, df_test={df_test.shape}. Reprise directe a l'etape 7/9.")

    # =========================================================================
    # [7/9] MODELE v3-place : memes variables enrichies, meme cible "place",
    # recherche d'hyperparametres identique a v2.
    # =========================================================================
    lib.log("\n[7/9] Entrainement GBM v3 sur variables enrichies, cible='place' (recherche d'hyperparametres)...")
    params_place, auc_place = lib.entrainer_gbm_avec_grille(
        X_train_v3, y_train_place, X_val_v3, y_val_place, lib.GRILLE_GBM, "v3-place")
    lib.log(f"  Meilleurs hyperparametres (v3-place) : {params_place} (AUC validation={round(auc_place,4)})")
    X_trainval_v3 = pd.concat([X_train_v3, X_val_v3], axis=0).reset_index(drop=True)
    y_trainval_place_v3 = np.concatenate([y_train_place, y_val_place])
    gbm_v3_place = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_place)
    gbm_v3_place.fit(X_trainval_v3, y_trainval_place_v3)
    proba_v3_place_test = gbm_v3_place.predict_proba(X_test_v3)[:, 1]
    df_test["proba_v3_place"] = proba_v3_place_test
    df_test["rang_v3_place"] = df_test.groupby("course_id")["proba_v3_place"].rank(method="min", ascending=False)

    lib.log("\n" + "=" * 100)
    lib.log("=== ANALYSE D'ERREURS DU MODELE v3-place (variables enrichies, cible='place') ===")
    lib.log("=" * 100)
    lib.log_analyse_erreurs(df_test, "rang_v3_place", "proba_v3_place", "GBM v3-place")

    # --- importance par permutation : les variables relatives ont-elles un
    # vrai signal, ou sont-elles sous le bruit ? ---
    lib.log("\nCalcul de l'importance par permutation (GBM v3-place, sous-echantillon de TEST)...")
    idx_perm = X_test_v3.sample(min(len(X_test_v3), lib.SOUS_ECHANTILLON_PERMUTATION), random_state=lib.RANDOM_SEED).index
    perm = permutation_importance(
        gbm_v3_place, X_test_v3.loc[idx_perm], y_test_place[idx_perm],
        scoring="roc_auc", n_repeats=3, random_state=lib.RANDOM_SEED, n_jobs=1,
    )
    importances = pd.Series(perm.importances_mean, index=X_test_v3.columns).sort_values(ascending=False)
    seuil_bruit = importances.get("temoin_aleatoire", 0.0)
    lib.log(f"  Bruit de reference (temoin aleatoire) = {round(seuil_bruit, 5)}")
    lib.log(f"\n  Les {len(colonnes_relatives)} nouvelles variables relatives, classees par importance de permutation :")
    for var in colonnes_relatives:
        if var in importances.index:
            imp = importances[var]
            marqueur = "  <-- AU-DESSUS du bruit" if imp > seuil_bruit else "  <-- sous le bruit"
            lib.log(f"    {var:50s} importance={imp:.5f}{marqueur}")
    n_relatives_utiles = sum(1 for v in colonnes_relatives if v in importances.index and importances[v] > seuil_bruit)
    lib.log(f"\n  Bilan : {n_relatives_utiles}/{len(colonnes_relatives)} des nouvelles variables relatives "
            f"depassent le bruit de reference.")

    del X_trainval_v3, gbm_v3_place
    gc.collect()

    # =========================================================================
    # [8/9] MODELE v3-gagnant : memes variables enrichies, mais entraine
    # DIRECTEMENT sur la cible "gagnant".
    # =========================================================================
    lib.log("\n[8/9] Entrainement GBM v3 sur variables enrichies, cible='gagnant' (recherche d'hyperparametres)...")
    params_gagnant, auc_gagnant = lib.entrainer_gbm_avec_grille(
        X_train_v3, y_train_gagnant, X_val_v3, y_val_gagnant, lib.GRILLE_GBM, "v3-gagnant")
    lib.log(f"  Meilleurs hyperparametres (v3-gagnant) : {params_gagnant} (AUC validation={round(auc_gagnant,4)})")
    X_trainval_v3b = pd.concat([X_train_v3, X_val_v3], axis=0).reset_index(drop=True)
    y_trainval_gagnant = np.concatenate([y_train_gagnant, y_val_gagnant])
    gbm_v3_gagnant = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_gagnant)
    gbm_v3_gagnant.fit(X_trainval_v3b, y_trainval_gagnant)
    proba_v3_gagnant_test = gbm_v3_gagnant.predict_proba(X_test_v3)[:, 1]
    df_test["proba_v3_gagnant"] = proba_v3_gagnant_test
    df_test["rang_v3_gagnant"] = df_test.groupby("course_id")["proba_v3_gagnant"].rank(method="min", ascending=False)
    del X_trainval_v3b, gbm_v3_gagnant, X_train_v3, X_val_v3, X_test_v3
    gc.collect()

    lib.log("\n" + "=" * 100)
    lib.log("=== ANALYSE D'ERREURS DU MODELE v3-gagnant (variables enrichies, cible='gagnant') ===")
    lib.log("=" * 100)
    lib.log_analyse_erreurs(df_test, "rang_v3_gagnant", "proba_v3_gagnant", "GBM v3-gagnant")

    # =========================================================================
    # [9/9] COMPARAISON FINALE â memes 3 methodologies que v2, meme jeu TEST.
    # =========================================================================
    df_test["rang_predit_baseline"] = lib.calculer_baseline_combine_v1(df_test)

    lib.log("\n" + "=" * 100)
    lib.log("=== COMPARAISON FINALE (meme protocole hors echantillon que v2) ===")
    lib.log("=" * 100)

    modeles = [
        ("Baseline combine_v1", "rang_predit_baseline", "proba_v2"),
        ("GBM v2 (cible=place, 109 variables)", "rang_v2", "proba_v2"),
        ("GBM v3-place (cible=place, variables enrichies)", "rang_v3_place", "proba_v3_place"),
        ("GBM v3-gagnant (cible=gagnant, variables enrichies)", "rang_v3_gagnant", "proba_v3_gagnant"),
    ]

    lib.log("\n-- Methodologie A : taux de reussite multi-picks (identique au calcul du 41,1% historique) --")
    resultats_A = {}
    for nom, rang_col, _ in modeles:
        essais, reussis, pct = lib.taux_reussite_place(df_test, rang_col)
        resultats_A[nom] = pct
        lib.log(f"  {nom:50s} {reussis}/{essais} = {pct}%")

    lib.log("\n-- Methodologie B : le MEILLEUR pick du modele par course (top-1), GAGNANT â la metrique cible de ce run --")
    resultats_B = {}
    for nom, rang_col, _ in modeles:
        n_courses, n_reussis, pct = lib.taux_reussite_top1(df_test, rang_col, "est_gagnant")
        resultats_B[nom] = pct
        lib.log(f"  {nom:50s} {n_reussis}/{n_courses} = {pct}%")

    lib.log("\n-- Methodologie C : le MEILLEUR pick du modele par course (top-1), PLACE --")
    resultats_C = {}
    for nom, rang_col, _ in modeles:
        n_courses, n_reussis, pct = lib.taux_reussite_top1(df_test, rang_col, "cible_place")
        resultats_C[nom] = pct
        lib.log(f"  {nom:50s} {n_reussis}/{n_courses} = {pct}%")

    try:
        for nom, _, proba_col in modeles[1:]:
            auc = round(roc_auc_score(y_test_place, df_test[proba_col]), 4)
            ll = round(log_loss(y_test_place, df_test[proba_col]), 4)
            lib.log(f"  AUC/logloss (cible=place) {nom:50s} AUC={auc} logloss={ll}")
    except ValueError as e:
        lib.log(f"  AUC/log-loss non calculables : {e}")

    lib.log("\n" + "=" * 100)
    lib.log("=== RESUME ===")
    lib.log("=" * 100)
    lib.log("Rappel v2 (rapport du 23/08/2026) : multi-pick 47,1% | top-1 gagnant 24,3% | top-1 place 54,2%")
    lib.log("Ce run (memes lignes de TEST, recalcule pour l'analyse d'erreurs) :")
    for nom, _, _ in modeles:
        lib.log(f"  {nom:50s} multi-pick={resultats_A[nom]}%  top1-gagnant={resultats_B[nom]}%  top1-place={resultats_C[nom]}%")
    meilleur = max(modeles[1:], key=lambda t: resultats_B[t[0]])
    lib.log(f"\nMeilleur taux de gagnants du meilleur pick : {meilleur[0]} avec {resultats_B[meilleur[0]]}% "
            f"(vs {resultats_B['GBM v2 (cible=place, 109 variables)']}% pour v2).")
    lib.log("Ce rapport est le resultat REEL, non ajuste. Objectif de ce run : ameliorer le taux de gagnants "
            "du meilleur pick, pas seulement la metrique multi-pick globale â voir Methodologie B ci-dessus.")


if __name__ == "__main__":
    main()
