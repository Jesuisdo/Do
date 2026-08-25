# -*- coding: utf-8 -*-
"""
analyse_erreurs_v3_diagnostic.py â Analyse d'erreurs APPROFONDIE demandee
par Dorian le 25/08/2026, apres le run production v3 (run #7, id
32871825116) : AUC en hausse mais top-1 gagnant stagnant a ~24,4%.

REGLES STRICTES DE CETTE ANALYSE (rappelees explicitement par Dorian) :
  - Lecture seule : AUCUNE modification du modele.
  - AUCUNE selection de nouvelle variable a partir du test (pas de fuite/
    p-hacking). Les variables analysees ci-dessous sont TOUTES deja
    utilisees par le modele v3 (VARIABLES_DIAGNOSTIC_ERREURS, v3_lib.py) â
    on ne fait qu'observer comment le modele s'est comporte vis-a-vis
    d'elles, on n'en ajoute aucune.
  - But unique : comprendre POURQUOI le modele se trompe (modele / donnees
    / identification des chevaux / information manquante avant la course),
    et lister ensuite des pistes d'amelioration (dans le rapport de chat,
    pas dans ce script) classees par potentiel et niveau de preuve.

NE SE CONNECTE PAS A SUPABASE. Reutilise le checkpoint DEJA PRODUIT par le
run production v3 (checkpoint-v3, run 32871825116, telecharge en artefact
cross-run par le workflow avant ce script) : aucune reconstruction de
variables, aucun nouveau chargement de donnees. Reproduit uniquement les
predictions v3-place / v3-gagnant (jamais persistees par entrainer_v3_
phase2.py) en relancant EXACTEMENT le meme code d'entrainement (meme grille
GRILLE_GBM, meme RANDOM_SEED=42) â reproduction deterministe, pas un
nouvel entrainement different.
"""
import gc
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    pass

CHECKPOINT_PATH = "checkpoint_v3_phase1.pkl"
OUTPUT_PATH = "df_test_diagnostic_v3.pkl"


def reproduire_predictions_v3(checkpoint):
    """Reproduction deterministe et FIDELE du code de entrainer_v3_phase2.py
    (etapes 7 et 8) : meme grille, meme seed, memes donnees. Ne constitue
    pas un nouvel entrainement : c'est le meme modele que celui deja evalue
    et rapporte a Dorian, seulement pas encore persiste en predictions."""
    X_train_v3 = checkpoint["X_train_v3"]
    X_val_v3 = checkpoint["X_val_v3"]
    X_test_v3 = checkpoint["X_test_v3"]
    y_train_place = checkpoint["y_train_place"]
    y_val_place = checkpoint["y_val_place"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    y_val_gagnant = checkpoint["y_val_gagnant"]
    df_test = checkpoint["df_test"].copy()

    lib.log("\n[Reproduction] GBM v3-place (identique a entrainer_v3_phase2.py, etape 7/9)...")
    params_place, auc_place = lib.entrainer_gbm_avec_grille(
        X_train_v3, y_train_place, X_val_v3, y_val_place, lib.GRILLE_GBM, "v3-place")
    lib.log(f"  Hyperparametres retrouves (v3-place) : {params_place} (AUC validation={round(auc_place,4)})")
    X_trainval = pd.concat([X_train_v3, X_val_v3], axis=0).reset_index(drop=True)
    y_trainval_place = np.concatenate([y_train_place, y_val_place])
    gbm_place = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_place)
    gbm_place.fit(X_trainval, y_trainval_place)
    df_test["proba_v3_place"] = gbm_place.predict_proba(X_test_v3)[:, 1]
    df_test["rang_v3_place"] = df_test.groupby("course_id")["proba_v3_place"].rank(method="min", ascending=False)
    del X_trainval, gbm_place
    gc.collect()

    lib.log("\n[Reproduction] GBM v3-gagnant (identique a entrainer_v3_phase2.py, etape 8/9)...")
    params_gagnant, auc_gagnant = lib.entrainer_gbm_avec_grille(
        X_train_v3, y_train_gagnant, X_val_v3, y_val_gagnant, lib.GRILLE_GBM, "v3-gagnant")
    lib.log(f"  Hyperparametres retrouves (v3-gagnant) : {params_gagnant} (AUC validation={round(auc_gagnant,4)})")
    X_trainval2 = pd.concat([X_train_v3, X_val_v3], axis=0).reset_index(drop=True)
    y_trainval_gagnant = np.concatenate([y_train_gagnant, y_val_gagnant])
    gbm_gagnant = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_gagnant)
    gbm_gagnant.fit(X_trainval2, y_trainval_gagnant)
    df_test["proba_v3_gagnant"] = gbm_gagnant.predict_proba(X_test_v3)[:, 1]
    df_test["rang_v3_gagnant"] = df_test.groupby("course_id")["proba_v3_gagnant"].rank(method="min", ascending=False)
    del X_trainval2, gbm_gagnant, X_train_v3, X_val_v3, X_test_v3
    gc.collect()

    return df_test


def rapport_pour_modele(df_test, rang_col, proba_col, label):
    lib.log("\n" + "#" * 100)
    lib.log(f"### ANALYSE D'ERREURS APPROFONDIE â MODELE {label}")
    lib.log("#" * 100)

    stats_rang, gagnants = lib.rang_distribution_gagnant(df_test, rang_col)
    lib.log(f"\n[A] Rang donne au gagnant reel (n={stats_rang['n_courses']} courses) :")
    lib.log(f"    top1={stats_rang['top1_pct']}%  top2={stats_rang['cumul_top2_pct']}%  "
            f"top3={stats_rang['cumul_top3_pct']}%  top5={stats_rang['cumul_top5_pct']}%  "
            f"hors_top5={stats_rang['au_dela_de_5_pct']}%")

    stats_ecart, gagnants = lib.ecart_probabilite_gagnant(gagnants, df_test, rang_col, proba_col)
    lib.log(f"\n[B] Ecart de probabilite pick-modele vs gagnant reel (courses ratees, n={stats_ecart['n_rates']}) :")
    lib.log(f"    moyenne={stats_ecart['moyenne']}  mediane={stats_ecart['mediane']}  p90={stats_ecart['p90']}  "
            f"quasi-trouve(<=0.02)={stats_ecart['quasi_trouve_pct']}%")

    lib.log("\n[B2] Repartition des courses ratees par tranche d'ecart de probabilite "
            "(+ confiance moyenne du pick fautif dans chaque tranche) :")
    for t in lib.distribution_ecart_probabilite_buckets(gagnants):
        lib.log(f"    {t['tranche_ecart']:12s} {t['n_courses']:5d} courses ({t['pct_des_courses_ratees']}%)  "
                f"proba_moyenne_pick_fautif={t['proba_moyenne_du_pick_fautif']}")

    lib.log("\n[C] Profils (moyennes) : top1 (trouve) vs top2-a-5 (presque) vs hors-top5 (rate largement) â "
            "trie par ecart standardise (d de Cohen) le plus marquant :")
    profils3 = lib.profils_par_bucket_rang(gagnants, lib.VARIABLES_DIAGNOSTIC_ERREURS)
    for _, row in profils3.iterrows():
        lib.log(f"    {row['variable']:40s} top1={row['top1_moyenne']}  top2_a_5={row['top2_a_5_moyenne']}  "
                f"hors_top5={row['hors_top5_moyenne']}  d(top1 vs top2_a_5)={row['d_top1_vs_top2_a_5']}  "
                f"d(top1 vs hors_top5)={row['d_top1_vs_hors_top5']}")

    lib.log("\n[D] Quand le modele se trompe de favori : vrai gagnant vs cheval choisi a tort par le modele "
            "(moyennes sur les courses ratees) :")
    comparatif, n_rates = lib.comparer_gagnant_vs_pick_modele(df_test, rang_col, proba_col, lib.VARIABLES_DIAGNOSTIC_ERREURS)
    lib.log(f"    ({n_rates} courses ou le pick #1 n'est pas le vrai gagnant)")
    for _, row in comparatif.iterrows():
        lib.log(f"    {row['variable']:40s} vrai_gagnant={row['moyenne_vrai_gagnant']}  "
                f"pick_modele_a_tort={row['moyenne_pick_modele_a_tort']}  ecart={row['ecart']}")

    lib.log("\n[E] Taux top-1 gagnant par nombre de partants (petit/moyen/grand champ) :")
    seg = lib.taux_gagnant_par_segments(df_test, rang_col, "bucket_partants", n_min=50)
    for _, r in seg.iterrows():
        lib.log(f"    {r['bucket_partants']:15s} {r['n_reussis']}/{r['n_courses']} = {r['pct_top1_gagnant']}%")

    lib.log("\n[F] Taux top-1 gagnant handicap vs non-handicap :")
    seg = lib.taux_gagnant_par_segments(df_test, rang_col, "est_handicap", n_min=50)
    for _, r in seg.iterrows():
        lib.log(f"    {'HANDICAP' if r['est_handicap'] else 'NON HANDICAP':15s} {r['n_reussis']}/{r['n_courses']} = {r['pct_top1_gagnant']}%")

    lib.log("\n[G] Taux top-1 gagnant par distance / terrain (>=100 courses) :")
    for col in ["distance_bucket", "terrain_bucket"]:
        seg = lib.taux_gagnant_par_segments(df_test, rang_col, col, n_min=100)
        for _, r in seg.iterrows():
            lib.log(f"    {col}={str(r[col]):15s} {r['n_reussis']}/{r['n_courses']} = {r['pct_top1_gagnant']}%")

    lib.log("\n[H] Croisement partants x handicap (>=50 courses) â recherche de segments particulierement difficiles :")
    seg = lib.taux_gagnant_par_segments(df_test, rang_col, ["bucket_partants", "est_handicap"], n_min=50)
    for _, r in seg.iterrows():
        lib.log(f"    partants={r['bucket_partants']:15s} handicap={r['est_handicap']!s:6s} "
                f"{r['n_reussis']}/{r['n_courses']} = {r['pct_top1_gagnant']}%")

    lib.log("\n[I] Croisement distance x terrain (>=80 courses) :")
    seg = lib.taux_gagnant_par_segments(df_test, rang_col, ["distance_bucket", "terrain_bucket"], n_min=80)
    for _, r in seg.iterrows():
        lib.log(f"    distance={str(r['distance_bucket']):10s} terrain={str(r['terrain_bucket']):10s} "
                f"{r['n_reussis']}/{r['n_courses']} = {r['pct_top1_gagnant']}%")

    lib.log("\n[J] Debutants / historique interne quasi-vide vs chevaux experimentes (proxy 'information "
            "manquante avant la course') â taux top-1 PARMI LES VRAIS GAGNANTS :")
    for label_flag in ["est_debutant", "identite_possible_fragmentee"]:
        for r in lib.taux_top1_par_groupe_binaire(gagnants, label_flag):
            lib.log(f"    {label_flag}={r[label_flag]!s:6s} n_gagnants={r['n_gagnants']:5d} "
                    f"top1={r['n_top1']:4d} = {r['pct_top1']}%")

    lib.log("\n[K] Segments les plus difficiles toutes vues confondues (le tableau [E]-[I] deja trie croissant "
            "identifie les segments a faible pct_top1_gagnant ci-dessus ; voir aussi [H]/[I] pour les croisements).")

    return stats_rang, stats_ecart, profils3, comparatif


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn n'est pas installe. Ce script doit tourner via le workflow GitHub Actions dedie.")

    lib.log("=" * 100)
    lib.log("ANALYSE D'ERREURS APPROFONDIE v3 â demande de Dorian le 25/08/2026")
    lib.log("Lecture seule : aucune modification du modele, aucune nouvelle variable selectionnee.")
    lib.log("=" * 100)

    lib.log(f"\nChargement du checkpoint {CHECKPOINT_PATH} (reutilise du run production 32871825116, "
            "aucune reconnexion a Supabase, aucune reconstruction de variables)...")
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    lib.log(f"  Checkpoint charge : df_test={checkpoint['df_test'].shape}")

    df_test = reproduire_predictions_v3(checkpoint)
    del checkpoint
    gc.collect()

    df_test = lib.ajouter_flags_diagnostic(df_test)
    lib.log(f"\nFlags diagnostic ajoutes (bucket_partants, est_handicap, est_debutant, "
            f"identite_possible_fragmentee). {int(df_test['est_debutant'].sum())} lignes 'debutant' sur "
            f"{len(df_test)} ({round(100*df_test['est_debutant'].mean(),1)}%).")

    for rang_col, proba_col, label in [
        ("rang_v2", "proba_v2", "v2 (109 variables, benchmark)"),
        ("rang_v3_gagnant", "proba_v3_gagnant", "v3-gagnant (variables enrichies, cible=gagnant â le modele cible de cette analyse)"),
    ]:
        rapport_pour_modele(df_test, rang_col, proba_col, label)

    lib.log("\n" + "=" * 100)
    lib.log("=== FIN DE L'ANALYSE D'ERREURS â voir sections [A] a [J] ci-dessus pour chaque modele ===")
    lib.log("Rappel : ce script est purement diagnostique. Aucun modele n'a ete modifie, aucune nouvelle")
    lib.log("variable n'a ete selectionnee. La synthese des pistes d'amelioration (classees par potentiel")
    lib.log("et niveau de preuve) est produite separement, a partir de ce rapport.")
    lib.log("=" * 100)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({"df_test_diagnostic": df_test}, f, protocol=pickle.HIGHEST_PROTOCOL)
    lib.log(f"\ndf_test enrichi (proba/rang v2+v3, flags diagnostic) sauvegarde dans {OUTPUT_PATH} "
            "pour reutilisation future eventuelle (aucune action requise).")


if __name__ == "__main__":
    main()
