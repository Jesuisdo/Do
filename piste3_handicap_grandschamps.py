# -*- coding: utf-8 -*-
"""
piste3_handicap_grandschamps.py -- Piste n3 (modele/calibration specifique
handicaps et grands champs), feu vert donne par Dorian le 25/08/2026 apres
rejet du blend v2+v3-gagnant (piste n1, gain non confirme).

PROTOCOLE STRICT (rappele par Dorian) :
  - Approche CONDITIONNELLE : le modele general (v3-gagnant, le meilleur des
    deux modeles generaux deja valides) reste la reference. Un modele
    specialise n'est utilise sur un segment QUE s'il demontre une
    amelioration reelle et confirmee sur ce segment precis.
  - 3 segments testes SEPAREMENT : handicaps ; grands champs (13+ partants) ;
    handicaps ET grands champs (intersection).
  - Memes donnees historiques, meme decoupage chronologique strict (memes
    TRAIN/VALIDATION/TEST que le pipeline v3 de production, run 32871825116).
  - Le choix "on utilise le modele specialise ou pas", ses hyperparametres,
    et le seuil de decision sont determines UNIQUEMENT sur VALIDATION (jamais
    sur TEST). Si le filtre VALIDATION rejette un segment, ce segment n'est
    JAMAIS evalue sur TEST (TEST reste vierge pour ce segment).
  - Seuils PRE-ENREGISTRES (fixes AVANT de voir le moindre resultat, pour ne
    pas les ajuster a posteriori pour forcer un "gain") :
      * SEUIL_MIN_COURSES_VALIDATION = 300 courses minimum dans le segment
        sur VALIDATION pour meme tenter un modele specialise (sinon : trop
        peu de donnees pour distinguer un vrai signal du bruit).
      * Le modele specialise n'est retenu comme candidat (a tester ensuite
        sur TEST) QUE s'il bat le modele general v3-gagnant a la fois sur le
        log-loss ET sur l'AUC (cible=gagnant), sur VALIDATION restreinte au
        segment. Pas de marge arbitraire ajoutee : le vrai garde-fou contre
        le bruit est l'etape suivante.
  - Meme apres avoir passe le filtre VALIDATION, le modele specialise n'est
    declare "retenu" pour un segment QUE s'il bat le modele general v3-gagnant
    sur le top-1 gagnant a la fois sur TEST_A (decision) ET sur TEST_B
    (confirmation independante, jamais vue avant cette lecture finale). Un
    gain qui n'apparait que sur TEST_A est traite comme du bruit et rejete.
  - TEST_A / TEST_B : meme scission chronologique 60/40 que le script blend
    (blend_v2_v3gagnant.py), pour rester coherent avec l'analyse precedente.
  - Aucune cote, aucune donnee de marche. Le systeme de mise n'est pas touche.

NE SE CONNECTE PAS A SUPABASE. Reutilise le checkpoint-v3 (run production
32871825116) et la metadata TRAIN/VALIDATION extraite et verifiee par
extraire_metadata_train_val.py (deja confirmee identique au checkpoint).
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
METADATA_PATH = "metadata_train_val_v3.pkl"

SEUIL_MIN_COURSES_VALIDATION = 300  # pre-enregistre avant tout resultat
FRACTION_TEST_A = 0.60  # identique a blend_v2_v3gagnant.py, pour rester coherent


def entrainer_modele_general(X_train, X_val, X_test, y_train, y_val, params, label):
    """Reproduit un modele general (deja calibre) avec des hyperparametres
    FIXES (pas de recherche de grille) : un fit TRAIN-only (predictions
    honnetes sur VALIDATION) et un fit TRAIN+VAL (predictions finales sur
    TEST). Utilise pour reproduire v2 (place) a l'identique."""
    gbm_train_only = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    gbm_train_only.fit(X_train, y_train)
    proba_val = gbm_train_only.predict_proba(X_val)[:, 1]
    del gbm_train_only
    gc.collect()

    X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_trainval = np.concatenate([y_train, y_val])
    gbm_final = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    gbm_final.fit(X_trainval, y_trainval)
    proba_test = gbm_final.predict_proba(X_test)[:, 1]
    del gbm_final, X_trainval
    gc.collect()
    return proba_val, proba_test


def entrainer_avec_grille_train_only_et_trainval(X_train, y_train, X_val, y_val, X_test, grille, label):
    """Recherche d'hyperparametres sur VALIDATION (lib.entrainer_gbm_avec_grille,
    identique au protocole deja utilise pour v3-gagnant), puis meme logique
    TRAIN-only / TRAIN+VAL que ci-dessus, avec les hyperparametres retenus.
    Retourne (params, proba_val, proba_test) -- proba_test est calcule
    seulement si X_test est fourni (None sinon, pour ne pas toucher TEST
    inutilement quand on n'a pas encore franchi le filtre VALIDATION)."""
    params, auc_val = lib.entrainer_gbm_avec_grille(X_train, y_train, X_val, y_val, grille, label)
    lib.log(f"  [{label}] Hyperparametres retenus (VALIDATION uniquement) : {params} (AUC validation={round(auc_val,4)})")

    gbm_train_only = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    gbm_train_only.fit(X_train, y_train)
    proba_val = gbm_train_only.predict_proba(X_val)[:, 1]
    del gbm_train_only
    gc.collect()

    proba_test = None
    if X_test is not None:
        X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_trainval = np.concatenate([y_train, y_val])
        gbm_final = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
        gbm_final.fit(X_trainval, y_trainval)
        proba_test = gbm_final.predict_proba(X_test)[:, 1]
        del gbm_final, X_trainval
        gc.collect()
    return params, proba_val, proba_test


def evaluer_modeles(df, modeles, label_periode):
    """modeles : dict {nom_modele: (rang_col, proba_col)}. Calcule et
    journalise, sur `df` (deja restreint a la periode/segment voulu), pour
    chaque modele : top-1 gagnant, top2/3/5, taux de place (top-1 pick et
    multi-pick), AUC(place), log-loss(gagnant), performance par nombre de
    partants et handicap/non-handicap (avec garde-fou n_min pour eviter de
    conclure sur un sous-segment trop petit)."""
    lib.log("\n" + "=" * 100)
    lib.log(f"=== {label_periode} (n={df['course_id'].nunique()} courses, {len(df)} lignes) ===")
    lib.log("=" * 100)
    for nom, (rang_col, proba_col) in modeles.items():
        stats_rang, _ = lib.rang_distribution_gagnant(df, rang_col)
        n_c, n_r, pct_top1_place = lib.taux_reussite_top1(df, rang_col, "cible_place")
        essais, reussis, pct_multipick = lib.taux_reussite_place(df, rang_col)
        try:
            auc = round(roc_auc_score(df["cible_place"], df[proba_col]), 4)
        except ValueError:
            auc = None
        try:
            ll = round(log_loss(df["est_gagnant"], df[proba_col]), 5)
        except ValueError:
            ll = None
        lib.log(f"\n-- {nom} --")
        lib.log(f"  top1 gagnant={stats_rang['top1_pct']}%  top2={stats_rang['cumul_top2_pct']}%  "
                f"top3={stats_rang['cumul_top3_pct']}%  top5={stats_rang['cumul_top5_pct']}%  "
                f"hors_top5={stats_rang['au_dela_de_5_pct']}%")
        lib.log(f"  taux place (top-1 pick)={pct_top1_place}%  taux place (multi-pick)={pct_multipick}%  "
                f"AUC(place)={auc}  log-loss(gagnant)={ll}")
        seg_partants = lib.taux_gagnant_par_segments(df, rang_col, "bucket_partants", n_min=100)
        for _, r in seg_partants.iterrows():
            lib.log(f"    partants={r['bucket_partants']:15s} {r['n_reussis']}/{r['n_courses']} = {r['pct_top1_gagnant']}%")
        seg_handicap = lib.taux_gagnant_par_segments(df, rang_col, "est_handicap", n_min=100)
        for _, r in seg_handicap.iterrows():
            lib.log(f"    {'HANDICAP' if r['est_handicap'] else 'NON HANDICAP':15s} {r['n_reussis']}/{r['n_courses']} = {r['pct_top1_gagnant']}%")


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn n'est pas installe. Ce script doit tourner via le workflow GitHub Actions dedie.")

    lib.log("=" * 100)
    lib.log("PISTE 3 -- MODELE CONDITIONNEL HANDICAPS / GRANDS CHAMPS -- feu vert Dorian le 25/08/2026")
    lib.log("Approche conditionnelle : le modele general (v3-gagnant) reste la reference. Un modele")
    lib.log("specialise n'est utilise sur un segment que s'il demontre un gain reel ET confirme.")
    lib.log(f"Seuil pre-enregistre : minimum {SEUIL_MIN_COURSES_VALIDATION} courses sur VALIDATION pour tenter un segment.")
    lib.log("=" * 100)

    lib.log("\nChargement du checkpoint et de la metadata TRAIN/VALIDATION (deja verifiee identique)...")
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

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

    meta_train = metadata["meta_train"].reset_index(drop=True)
    meta_val = metadata["meta_val"].reset_index(drop=True)
    lib.log(f"  Checkpoint : df_test={df_test.shape}. Metadata : meta_train={meta_train.shape}, meta_val={meta_val.shape}.")

    # =========================================================================
    # MODELES GENERAUX (reference) -- reproduits a l'identique, memes
    # hyperparametres/protocole que dans blend_v2_v3gagnant.py.
    # =========================================================================
    lib.log("\n[1/4] Reproduction du modele general v2 (cible=place)...")
    proba_v2_val, proba_v2_test = entrainer_modele_general(
        X_train_v3[colonnes_v2_ctrl], X_val_v3[colonnes_v2_ctrl], X_test_v3[colonnes_v2_ctrl],
        y_train_place, y_val_place, lib.MEILLEURS_PARAMS_GBM_V2, "v2")
    ecart_max = float(np.max(np.abs(proba_v2_test - df_test["proba_v2"].values)))
    lib.log(f"  Controle de fidelite (vs proba_v2 deja persistee) : ecart absolu max = {ecart_max:.6f} (doit etre ~0).")

    lib.log("\n[2/4] Reproduction du modele general v3-gagnant (cible=gagnant, LA reference principale)...")
    params_v3g, proba_v3g_val, proba_v3g_test = entrainer_avec_grille_train_only_et_trainval(
        X_train_v3, y_train_gagnant, X_val_v3, y_val_gagnant, X_test_v3, lib.GRILLE_GBM, "v3-gagnant")

    df_test["proba_v2"] = proba_v2_test
    df_test["rang_v2"] = df_test.groupby("course_id")["proba_v2"].rank(method="min", ascending=False)
    df_test["proba_v3_gagnant"] = proba_v3g_test
    df_test["rang_v3_gagnant"] = df_test.groupby("course_id")["proba_v3_gagnant"].rank(method="min", ascending=False)
    df_test = lib.ajouter_flags_diagnostic(df_test)

    # --- meme scission chronologique TEST_A/TEST_B que blend_v2_v3gagnant.py ---
    courses_ordre = df_test.sort_values(["date_course", "course_id"])["course_id"].drop_duplicates().tolist()
    n = len(courses_ordre)
    n_a = int(n * FRACTION_TEST_A)
    courses_A = set(courses_ordre[:n_a])
    courses_B = set(courses_ordre[n_a:])
    lib.log(f"\n  TEST_A={len(courses_A)} courses, TEST_B={len(courses_B)} courses (identique au split du blend).")

    # =========================================================================
    # SEGMENTS : construction des masques TRAIN / VALIDATION (via metadata,
    # jamais persistee avant) / TEST (via df_test, deja enrichi).
    # =========================================================================
    est_handicap_train = meta_train["categorie_particularite"].fillna("").str.contains("HANDICAP")
    est_handicap_val = meta_val["categorie_particularite"].fillna("").str.contains("HANDICAP")
    grand_train = meta_train["nb_partants_reel"] >= 13
    grand_val = meta_val["nb_partants_reel"] >= 13

    segments = {
        "handicap": {
            "train": est_handicap_train.values, "val": est_handicap_val.values,
            "test": df_test["est_handicap"].values,
        },
        "grand_champ_13plus": {
            "train": grand_train.values, "val": grand_val.values,
            "test": (df_test["bucket_partants"] == "grand (13+)").values,
        },
        "handicap_et_grand_champ": {
            "train": (est_handicap_train & grand_train).values, "val": (est_handicap_val & grand_val).values,
            "test": (df_test["est_handicap"] & (df_test["bucket_partants"] == "grand (13+)")).values,
        },
    }

    verdicts = {}

    for i, (nom_segment, masques) in enumerate(segments.items(), start=3):
        lib.log("\n" + "#" * 100)
        lib.log(f"# SEGMENT {i-2}/3 : {nom_segment}")
        lib.log("#" * 100)

        m_train, m_val, m_test = masques["train"], masques["val"], masques["test"]
        n_courses_train = meta_train.loc[m_train, "course_id"].nunique()
        n_courses_val = meta_val.loc[m_val, "course_id"].nunique()
        n_courses_test = df_test.loc[m_test, "course_id"].nunique()
        lib.log(f"\n[Taille d'echantillon] TRAIN={m_train.sum()} lignes/{n_courses_train} courses, "
                f"VALIDATION={m_val.sum()} lignes/{n_courses_val} courses, "
                f"TEST={m_test.sum()} lignes/{n_courses_test} courses "
                f"({round(100*n_courses_test/df_test['course_id'].nunique(),1)}% du TEST total).")

        if n_courses_val < SEUIL_MIN_COURSES_VALIDATION:
            lib.log(f"\n  REJET IMMEDIAT : {n_courses_val} courses sur VALIDATION < seuil pre-enregistre "
                    f"({SEUIL_MIN_COURSES_VALIDATION}). Echantillon trop petit pour distinguer un vrai signal "
                    f"du bruit. Le modele general (v3-gagnant) reste la reference pour ce segment. "
                    f"TEST non touche pour ce segment.")
            verdicts[nom_segment] = "REJETE (echantillon VALIDATION insuffisant)"
            continue

        lib.log(f"\n[Entrainement du modele specialise '{nom_segment}'] "
                f"grille d'hyperparametres identique (GRILLE_GBM), sur TRAIN/VALIDATION restreints au segment...")
        params_special, proba_special_val, _ = entrainer_avec_grille_train_only_et_trainval(
            X_train_v3[m_train], y_train_gagnant[m_train], X_val_v3[m_val], y_val_gagnant[m_val],
            None, lib.GRILLE_GBM, f"specialise-{nom_segment}")

        y_val_gagnant_segment = y_val_gagnant[m_val]
        y_val_place_segment = y_val_place[m_val]
        proba_v3g_val_segment = proba_v3g_val[m_val]

        ll_special = log_loss(y_val_gagnant_segment, proba_special_val)
        ll_general = log_loss(y_val_gagnant_segment, proba_v3g_val_segment)
        auc_special = roc_auc_score(y_val_gagnant_segment, proba_special_val)
        auc_general = roc_auc_score(y_val_gagnant_segment, proba_v3g_val_segment)
        lib.log(f"\n[Decision sur VALIDATION uniquement -- jamais sur TEST]")
        lib.log(f"  Specialise '{nom_segment}' : logloss(gagnant)={round(ll_special,5)}  AUC(gagnant)={round(auc_special,4)}")
        lib.log(f"  General v3-gagnant (meme sous-ensemble)  : logloss(gagnant)={round(ll_general,5)}  AUC(gagnant)={round(auc_general,4)}")

        retenu_validation = (ll_special < ll_general) and (auc_special >= auc_general)
        if not retenu_validation:
            lib.log(f"\n  REJET : le modele specialise ne bat PAS le modele general sur VALIDATION "
                    f"(logloss et/ou AUC). Le modele general (v3-gagnant) reste la reference pour ce segment. "
                    f"TEST non touche pour ce segment.")
            verdicts[nom_segment] = "REJETE (ne bat pas le general sur VALIDATION)"
            continue

        lib.log(f"\n  CANDIDAT RETENU sur VALIDATION : le modele specialise bat le general sur ce segment. "
                f"Confirmation maintenant sur TEST_A (decision) puis TEST_B (confirmation independante)...")

        # --- entrainement final TRAIN+VAL-segment, SEULEMENT maintenant que le filtre VALIDATION est passe ---
        X_trainval_segment = pd.concat([X_train_v3[m_train], X_val_v3[m_val]], axis=0).reset_index(drop=True)
        y_trainval_segment = np.concatenate([y_train_gagnant[m_train], y_val_gagnant[m_val]])
        gbm_final_special = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_special)
        gbm_final_special.fit(X_trainval_segment, y_trainval_segment)
        proba_special_test_segment = gbm_final_special.predict_proba(X_test_v3[m_test])[:, 1]
        del gbm_final_special, X_trainval_segment
        gc.collect()

        df_segment = df_test[m_test].copy()
        df_segment["proba_special"] = proba_special_test_segment
        df_segment["rang_special"] = df_segment.groupby("course_id")["proba_special"].rank(method="min", ascending=False)

        df_segment_A = df_segment[df_segment["course_id"].isin(courses_A)].reset_index(drop=True)
        df_segment_B = df_segment[df_segment["course_id"].isin(courses_B)].reset_index(drop=True)

        modeles = {
            "GBM v2 (general, cible=place)": ("rang_v2", "proba_v2"),
            "GBM v3-gagnant (general, cible=gagnant -- REFERENCE)": ("rang_v3_gagnant", "proba_v3_gagnant"),
            f"Modele specialise '{nom_segment}'": ("rang_special", "proba_special"),
        }
        evaluer_modeles(df_segment_A, modeles, f"TEST_A -- segment {nom_segment} (decision)")
        evaluer_modeles(df_segment_B, modeles, f"TEST_B -- segment {nom_segment} (confirmation independante)")

        top1_v3g_A = lib.rang_distribution_gagnant(df_segment_A, "rang_v3_gagnant")[0]["top1_pct"]
        top1_special_A = lib.rang_distribution_gagnant(df_segment_A, "rang_special")[0]["top1_pct"]
        top1_v3g_B = lib.rang_distribution_gagnant(df_segment_B, "rang_v3_gagnant")[0]["top1_pct"]
        top1_special_B = lib.rang_distribution_gagnant(df_segment_B, "rang_special")[0]["top1_pct"]

        gagne_A = top1_special_A > top1_v3g_A
        gagne_B = top1_special_B > top1_v3g_B
        lib.log(f"\n[Verdict final -- segment {nom_segment}]")
        lib.log(f"  Top-1 gagnant -- specialise vs v3-gagnant (reference) : "
                f"TEST_A {top1_special_A}% vs {top1_v3g_A}% ({'gagne' if gagne_A else 'ne gagne pas'}) | "
                f"TEST_B {top1_special_B}% vs {top1_v3g_B}% ({'gagne' if gagne_B else 'ne gagne pas'})")
        if gagne_A and gagne_B:
            verdicts[nom_segment] = f"RETENU -- gain confirme sur TEST_A ET TEST_B ({top1_special_A}%/{top1_special_B}% vs {top1_v3g_A}%/{top1_v3g_B}% pour v3-gagnant)"
            lib.log("  --> GAIN CONFIRME sur les DEUX periodes. Modele specialise RETENU pour ce segment.")
        else:
            verdicts[nom_segment] = "REJETE (gain non confirme sur les deux periodes TEST_A et TEST_B)"
            lib.log("  --> Gain NON confirme sur les deux periodes (au moins une periode ne montre pas de gain). "
                    "Modele specialise REJETE malgre le passage du filtre VALIDATION -- le modele general reste "
                    "la reference pour ce segment.")

    lib.log("\n" + "=" * 100)
    lib.log("=== RESUME FINAL -- decision par segment (approche conditionnelle) ===")
    lib.log("=" * 100)
    for nom_segment, verdict in verdicts.items():
        lib.log(f"  {nom_segment:30s} : {verdict}")
    lib.log("\nRappel : pour tout segment REJETE, le modele general (v3-gagnant) reste seul utilise -- ")
    lib.log("aucun modele specialise n'est mis en production pour ce segment.")
    lib.log("=" * 100)


if __name__ == "__main__":
    main()
