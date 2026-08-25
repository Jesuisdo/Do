# -*- coding: utf-8 -*-
"""
blend_v2_v3gagnant.py Ã¢ Etape 1 du plan valide par Dorian le 25/08/2026,
suite a l'analyse d'erreurs approfondie (constat : 5,8 a 8,4% des courses
ratees sont des "quasi-trouve", ecart de probabilite <= 0.02 -> hypothese
qu'un blend v2 (cible=place) + v3-gagnant (cible=gagnant) puisse faire
basculer certaines de ces courses proches).

PROTOCOLE STRICT (rappele par Dorian) :
  - Poids testes : 25/75, 50/50, 75/25 (v2/v3-gagnant).
  - Le poids retenu est choisi UNIQUEMENT a partir de VALIDATION (log-loss
    + AUC contre la cible 'gagnant'), JAMAIS a partir du TEST.
  - Aucun reentrainement des modeles sur le TEST. v2 et v3-gagnant sont
    reproduits a l'identique (memes hyperparametres, meme grille, meme
    RANDOM_SEED=42) : ce n'est pas une nouvelle selection de variable.
  - Le TEST est lui-meme scinde chronologiquement en deux : TEST_A (les
    ~60% de courses les plus anciennes du TEST) sert a la decision/lecture
    principale ; TEST_B (les ~40% les plus recentes, jamais vues lors du
    choix du poids) sert de confirmation finale hors-echantillon
    supplementaire. Le blend n'est retenu que s'il tient sur les DEUX.

NE SE CONNECTE PAS A SUPABASE. Reutilise le meme checkpoint-v3 (run
production 32871825116) que le diagnostic precedent.

Limite methodologique assumee et signalee dans le rapport : proba_v2 est
calibree sur la cible 'place' et proba_v3_gagnant sur la cible 'gagnant' Ã¢
deux echelles differentes (base rates differents). Le blend lineaire
demande par Dorian est fait tel quel ; ce n'est pas une combinaison
'apples-to-apples' parfaite, et c'est explicitement note dans le rapport
final pour que la decision soit prise en connaissance de cause.
"""
import gc
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss
except ImportError:
    pass

CHECKPOINT_PATH = "checkpoint_v3_phase1.pkl"
POIDS_CANDIDATS = [0.25, 0.50, 0.75]  # poids attribue a v2 ; le reste va a v3-gagnant
FRACTION_TEST_A = 0.60  # part chronologique du TEST utilisee pour la decision/lecture principale


def entrainer_v2_train_only_et_trainval(X_train_v3, X_val_v3, X_test_v3, colonnes_v2_ctrl, y_train_place, y_val_place):
    """Reproduit EXACTEMENT le modele v2 (memes hyperparametres deja
    choisis, MEILLEURS_PARAMS_GBM_V2) : un modele TRAIN-only (pour obtenir
    des predictions propres sur VALIDATION, comme deja fait dans
    entrainer_v3_phase1.py pour la verification de coherence AUC) et un
    modele TRAIN+VAL (pour les predictions finales sur TEST, identique au
    checkpoint d'origine)."""
    gbm_train_only = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **lib.MEILLEURS_PARAMS_GBM_V2)
    gbm_train_only.fit(X_train_v3[colonnes_v2_ctrl], y_train_place)
    proba_val = gbm_train_only.predict_proba(X_val_v3[colonnes_v2_ctrl])[:, 1]
    del gbm_train_only
    gc.collect()

    X_trainval = pd.concat([X_train_v3[colonnes_v2_ctrl], X_val_v3[colonnes_v2_ctrl]], axis=0).reset_index(drop=True)
    y_trainval = np.concatenate([y_train_place, y_val_place])
    gbm_final = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **lib.MEILLEURS_PARAMS_GBM_V2)
    gbm_final.fit(X_trainval, y_trainval)
    proba_test = gbm_final.predict_proba(X_test_v3[colonnes_v2_ctrl])[:, 1]
    del gbm_final, X_trainval
    gc.collect()
    return proba_val, proba_test


def entrainer_v3gagnant_train_only_et_trainval(X_train_v3, X_val_v3, X_test_v3, y_train_gagnant, y_val_gagnant):
    """Meme grille de recherche d'hyperparametres que d'habitude
    (GRILLE_GBM, choisie sur VALIDATION), puis meme logique TRAIN-only /
    TRAIN+VAL que pour v2 ci-dessus, pour obtenir des predictions propres
    sur VALIDATION et sur TEST avec le meme modele (config gagnante de la
    grille), sans aucune fuite."""
    params, auc_val = lib.entrainer_gbm_avec_grille(
        X_train_v3, y_train_gagnant, X_val_v3, y_val_gagnant, lib.GRILLE_GBM, "v3-gagnant")
    lib.log(f"  Hyperparametres retrouves (v3-gagnant) : {params} (AUC validation={round(auc_val,4)})")

    gbm_train_only = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    gbm_train_only.fit(X_train_v3, y_train_gagnant)
    proba_val = gbm_train_only.predict_proba(X_val_v3)[:, 1]
    del gbm_train_only
    gc.collect()

    X_trainval = pd.concat([X_train_v3, X_val_v3], axis=0).reset_index(drop=True)
    y_trainval = np.concatenate([y_train_gagnant, y_val_gagnant])
    gbm_final = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    gbm_final.fit(X_trainval, y_trainval)
    proba_test = gbm_final.predict_proba(X_test_v3)[:, 1]
    del gbm_final, X_trainval
    gc.collect()
    return proba_val, proba_test


def choisir_poids_sur_validation(proba_v2_val, proba_v3g_val, y_val_gagnant, y_val_place):
    """Choix du poids UNIQUEMENT sur VALIDATION : log-loss contre la cible
    'gagnant' (objectif final de Dorian) comme critere principal, AUC
    (gagnant et place) comme confirmation. Retourne le meilleur poids et le
    tableau complet pour transparence."""
    lib.log("\n[Choix du poids sur VALIDATION uniquement -- jamais sur TEST]")
    resultats = []
    for w in POIDS_CANDIDATS:
        blend_val = w * proba_v2_val + (1 - w) * proba_v3g_val
        ll_gagnant = log_loss(y_val_gagnant, blend_val)
        auc_gagnant = roc_auc_score(y_val_gagnant, blend_val)
        auc_place = roc_auc_score(y_val_place, blend_val)
        resultats.append({"poids_v2": w, "poids_v3g": round(1 - w, 2), "logloss_gagnant": round(ll_gagnant, 5),
                           "auc_gagnant": round(auc_gagnant, 4), "auc_place": round(auc_place, 4)})
        lib.log(f"  v2={w:.2f} / v3-gagnant={1-w:.2f}  logloss(gagnant)={round(ll_gagnant,5)}  "
                f"AUC(gagnant)={round(auc_gagnant,4)}  AUC(place)={round(auc_place,4)}")
    # reference : v2 seul et v3-gagnant seul, pour situer les blends
    for label, p in [("v2 seul", proba_v2_val), ("v3-gagnant seul", proba_v3g_val)]:
        ll = log_loss(y_val_gagnant, p)
        auc_g = roc_auc_score(y_val_gagnant, p)
        lib.log(f"  (reference) {label:20s}  logloss(gagnant)={round(ll,5)}  AUC(gagnant)={round(auc_g,4)}")
    meilleur = min(resultats, key=lambda r: r["logloss_gagnant"])
    lib.log(f"\n  Poids retenu (meilleur logloss sur VALIDATION, cible=gagnant) : "
            f"v2={meilleur['poids_v2']} / v3-gagnant={meilleur['poids_v3g']}")
    return meilleur["poids_v2"], resultats


def rapport_comparatif(df, rang_cols_par_modele, proba_cols_par_modele, label_periode):
    """rang_cols_par_modele / proba_cols_par_modele : dict {nom_modele: colonne}.
    Produit, pour chaque modele, exactement les metriques demandees par
    Dorian sur `df` (deja restreint a la periode voulue) : top-1 gagnant,
    top2/3/5, taux de place (top-1 et multi-pick), AUC, par nombre de
    partants, handicap/non-handicap."""
    lib.log("\n" + "=" * 100)
    lib.log(f"=== COMPARATIF v2 / v3-gagnant / blend Ã¢ {label_periode} (n={df['course_id'].nunique()} courses) ===")
    lib.log("=" * 100)
    for nom, rang_col in rang_cols_par_modele.items():
        proba_col = proba_cols_par_modele[nom]
        stats_rang, _ = lib.rang_distribution_gagnant(df, rang_col)
        n_c, n_r, pct_top1_place = lib.taux_reussite_top1(df, rang_col, "cible_place")
        essais, reussis, pct_multipick = lib.taux_reussite_place(df, rang_col)
        try:
            auc = round(roc_auc_score(df["cible_place"], df[proba_col]), 4)
        except ValueError:
            auc = None
        lib.log(f"\n-- {nom} --")
        lib.log(f"  top1 gagnant={stats_rang['top1_pct']}%  top2={stats_rang['cumul_top2_pct']}%  "
                f"top3={stats_rang['cumul_top3_pct']}%  top5={stats_rang['cumul_top5_pct']}%  "
                f"hors_top5={stats_rang['au_dela_de_5_pct']}%")
        lib.log(f"  taux place (top-1 pick)={pct_top1_place}%  taux place (multi-pick)={pct_multipick}%  AUC(place)={auc}")
        seg_partants = lib.taux_gagnant_par_segments(df, rang_col, "bucket_partants", n_min=20)
        for _, r in seg_partants.iterrows():
            lib.log(f"    partants={r['bucket_partants']:15s} {r['n_reussis']}/{r['n_courses']} = {r['pct_top1_gagnant']}%")
        seg_handicap = lib.taux_gagnant_par_segments(df, rang_col, "est_handicap", n_min=20)
        for _, r in seg_handicap.iterrows():
            lib.log(f"    {'HANDICAP' if r['est_handicap'] else 'NON HANDICAP':15s} {r['n_reussis']}/{r['n_courses']} = {r['pct_top1_gagnant']}%")


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn n'est pas installe. Ce script doit tourner via le workflow GitHub Actions dedie.")

    lib.log("=" * 100)
    lib.log("ETAPE 1 -- BLEND v2 + v3-gagnant -- demande de Dorian le 25/08/2026")
    lib.log("Poids choisi uniquement sur VALIDATION. Confirmation sur un TEST scinde en 2 periodes chronologiques.")
    lib.log("=" * 100)

    lib.log(f"\nChargement du checkpoint {CHECKPOINT_PATH} (reutilise du run production 32871825116)...")
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    X_train_v3 = checkpoint["X_train_v3"]
    X_val_v3 = checkpoint["X_val_v3"]
    X_test_v3 = checkpoint["X_test_v3"]
    y_train_place = checkpoint["y_train_place"]
    y_val_place = checkpoint["y_val_place"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    y_val_gagnant = checkpoint["y_val_gagnant"]
    df_test = checkpoint["df_test"].copy()
    colonnes_relatives = checkpoint["colonnes_relatives"]
    colonnes_v2_ctrl = [c for c in X_train_v3.columns if c not in colonnes_relatives]
    lib.log(f"  Checkpoint charge : df_test={df_test.shape}. Colonnes v2 reconstruites : {len(colonnes_v2_ctrl)} "
            f"(= colonnes v3 moins les {len(colonnes_relatives)} variables relatives -- deduction mecanique, "
            "aucune nouvelle selection de variable).")

    lib.log("\n[1/3] Reproduction de v2 (train-only pour VALIDATION + train+val pour TEST)...")
    proba_v2_val, proba_v2_test = entrainer_v2_train_only_et_trainval(
        X_train_v3, X_val_v3, X_test_v3, colonnes_v2_ctrl, y_train_place, y_val_place)
    # controle de fidelite : la reproduction TEST doit correspondre a proba_v2 deja present dans le checkpoint
    ecart_max = float(np.max(np.abs(proba_v2_test - df_test["proba_v2"].values)))
    lib.log(f"  Controle de fidelite (reproduction vs proba_v2 deja persistee dans le checkpoint) : "
            f"ecart absolu max = {ecart_max:.6f} (doit etre ~0, meme modele, meme seed).")

    lib.log("\n[2/3] Reproduction de v3-gagnant (train-only pour VALIDATION + train+val pour TEST)...")
    proba_v3g_val, proba_v3g_test = entrainer_v3gagnant_train_only_et_trainval(
        X_train_v3, X_val_v3, X_test_v3, y_train_gagnant, y_val_gagnant)
    del X_train_v3, X_val_v3, X_test_v3
    gc.collect()

    lib.log("\n[3/3] Choix du poids de blend sur VALIDATION, puis application figee sur TEST...")
    meilleur_poids_v2, tableau_poids = choisir_poids_sur_validation(proba_v2_val, proba_v3g_val, y_val_gagnant, y_val_place)

    df_test["proba_v2"] = proba_v2_test  # reecrit avec la reproduction (identique, cf. controle de fidelite ci-dessus)
    df_test["rang_v2"] = df_test.groupby("course_id")["proba_v2"].rank(method="min", ascending=False)
    df_test["proba_v3_gagnant"] = proba_v3g_test
    df_test["rang_v3_gagnant"] = df_test.groupby("course_id")["proba_v3_gagnant"].rank(method="min", ascending=False)
    for w in POIDS_CANDIDATS:
        col_p = f"proba_blend_{int(w*100)}"
        col_r = f"rang_blend_{int(w*100)}"
        df_test[col_p] = w * df_test["proba_v2"] + (1 - w) * df_test["proba_v3_gagnant"]
        df_test[col_r] = df_test.groupby("course_id")[col_p].rank(method="min", ascending=False)
    col_p_final = f"proba_blend_{int(meilleur_poids_v2*100)}"
    col_r_final = f"rang_blend_{int(meilleur_poids_v2*100)}"

    df_test = lib.ajouter_flags_diagnostic(df_test)

    # --- scission chronologique de TEST : TEST_A (decision) / TEST_B (confirmation finale, jamais vue avant) ---
    courses_ordre = df_test.sort_values(["date_course", "course_id"])["course_id"].drop_duplicates().tolist()
    n = len(courses_ordre)
    n_a = int(n * FRACTION_TEST_A)
    courses_A = set(courses_ordre[:n_a])
    courses_B = set(courses_ordre[n_a:])
    df_test_A = df_test[df_test["course_id"].isin(courses_A)].reset_index(drop=True)
    df_test_B = df_test[df_test["course_id"].isin(courses_B)].reset_index(drop=True)
    lib.log(f"\nScission chronologique du TEST : TEST_A={df_test_A['course_id'].nunique()} courses "
            f"({df_test_A['date_course'].min()} -> {df_test_A['date_course'].max()}), "
            f"TEST_B={df_test_B['course_id'].nunique()} courses "
            f"({df_test_B['date_course'].min()} -> {df_test_B['date_course'].max()}, jamais utilise pour choisir le poids).")

    modeles = {
        "GBM v2 (cible=place)": ("rang_v2", "proba_v2"),
        "GBM v3-gagnant (cible=gagnant)": ("rang_v3_gagnant", "proba_v3_gagnant"),
        f"Blend v2={meilleur_poids_v2}/v3-gagnant={round(1-meilleur_poids_v2,2)} (poids choisi sur VALIDATION)": (col_r_final, col_p_final),
    }
    rang_cols = {k: v[0] for k, v in modeles.items()}
    proba_cols = {k: v[1] for k, v in modeles.items()}

    rapport_comparatif(df_test_A, rang_cols, proba_cols, "TEST_A (decision, plus ancien)")
    rapport_comparatif(df_test_B, rang_cols, proba_cols, "TEST_B (confirmation finale hors-echantillon, plus recent)")

    # --- pour reference/transparence : les 3 poids candidats, sur TEST_A uniquement (lecture, pas decision) ---
    lib.log("\n" + "=" * 100)
    lib.log("=== POUR INFORMATION : les 3 poids candidats sur TEST_A (n'a pas servi a choisir le poids) ===")
    lib.log("=" * 100)
    rang_cols_poids = {f"Blend v2={w}/v3-gagnant={round(1-w,2)}": f"rang_blend_{int(w*100)}" for w in POIDS_CANDIDATS}
    proba_cols_poids = {f"Blend v2={w}/v3-gagnant={round(1-w,2)}": f"proba_blend_{int(w*100)}" for w in POIDS_CANDIDATS}
    rapport_comparatif(df_test_A, rang_cols_poids, proba_cols_poids, "TEST_A -- comparaison des 3 poids candidats")

    lib.log("\n" + "=" * 100)
    lib.log("=== FIN -- rappel : poids choisi uniquement sur VALIDATION, TEST_B jamais vu avant cette lecture finale ===")
    lib.log("Limite assumee : proba_v2 (cible=place) et proba_v3_gagnant (cible=gagnant) sont sur deux echelles")
    lib.log("differentes (base rates differents) ; le blend lineaire est fait tel que demande, mais ce n'est pas")
    lib.log("une combinaison parfaitement homogene -- a garder en tete en lisant les resultats ci-dessus.")
    lib.log("=" * 100)


if __name__ == "__main__":
    main()
