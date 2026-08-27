# -*- coding: utf-8 -*-
"""
entrainer_v3_phase2_ranking.py -- PISTE 1 (objectif d'entrainement oriente
classement intra-course), demandee par Dorian le 27/08/2026 APRES validation
et verrouillage du protocole double-benchmark.

Protocole exact demande :
  1. Memes variables v3 (aucun changement de feature).
  2. Memes splits chronologiques (checkpoint_v3_phase1_ranking.pkl, copie
     stricte de la phase 1 v3, voir entrainer_v3_phase1_ranking.py).
  3. Un seul objectif de ranking teste (LightGBM, objectif "lambdarank",
     groupe par course_id) -- pas de multiplication d'essais.
  4. Aucun changement de feature en meme temps que le changement d'objectif :
     le seul delta entre le modele "actuel" recalcule ici et le modele
     ranking est la fonction de perte d'entrainement.
  5. Metriques sur VALIDATION uniquement : top-1, top-3, top-5, NDCG@3/5,
     MRR, AUC en secondaire.
  6. Comparaison systematique au modele v3 actuel (meme grille d'hyper-
     parametres GRILLE_GBM, meme protocole "fit TRAIN, evalue VALIDATION"
     que la recherche d'hyperparametres deja utilisee en production).
  7. TEST A et TEST B ne sont PAS lances ici -- uniquement si ce rapport de
     validation montre un gain convaincant et robuste, sur decision de
     Dorian.
  8. Benchmark reel et benchmark donnees propres affiches cote a cote,
     comme prevu par le protocole permanent (v3_lib.rapport_double_benchmark).

NE SE CONNECTE PAS A SUPABASE. Charge uniquement checkpoint_v3_phase1_ranking.pkl
(produit par entrainer_v3_phase1_ranking.py, telecharge en artefact GitHub
Actions par le workflow dedie).
"""
import itertools
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
except ImportError:
    pass

try:
    import lightgbm as lgb
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    LIGHTGBM_DISPONIBLE = False

CHECKPOINT_PATH = "checkpoint_v3_phase1_ranking.pkl"


def groupes_consecutifs(course_id_iterable):
    """Tailles des groupes de lignes CONSECUTIVES partageant le meme
    course_id (format attendu par LightGBM pour un objectif de ranking).
    Repose sur le tri ["date_course", "course_id"] deja applique en phase 1 :
    les lignes d'une meme course sont deja contigues, jamais entrelacees."""
    return [len(list(g)) for _, g in itertools.groupby(list(course_id_iterable))]


def ndcg_gagnant_at_k(df, rang_col, k, cible_col="est_gagnant"):
    """NDCG@k specialise au cas 'un seul document pertinent par course' (le
    vrai gagnant) : IDCG = 1 (gagnant idealement au rang 1), donc
    NDCG@k = moyenne sur les courses de 1/log2(rang_du_gagnant + 1) si ce
    rang est <= k, sinon 0."""
    d = df[df[cible_col] == 1]
    n = len(d)
    if n == 0:
        return float("nan")
    gains = d[rang_col].apply(lambda r: 1.0 / np.log2(r + 1) if r <= k else 0.0)
    return round(float(gains.mean()), 4)


def mrr_gagnant(df, rang_col, cible_col="est_gagnant"):
    """Mean Reciprocal Rank : moyenne de 1/rang_du_gagnant sur les courses."""
    d = df[df[cible_col] == 1]
    n = len(d)
    if n == 0:
        return float("nan")
    return round(float((1.0 / d[rang_col]).mean()), 4)


def calculer_toutes_metriques(df, rang_col, proba_ou_score_col, y_vrai, label):
    """Calcule le jeu complet de metriques demande par Dorian pour un
    modele donne, sur la population `df` deja fournie (benchmark reel OU
    benchmark propre, filtre par l'appelant)."""
    stats_rang, _ = lib.rang_distribution_gagnant(df, rang_col)
    ndcg3 = ndcg_gagnant_at_k(df, rang_col, 3)
    ndcg5 = ndcg_gagnant_at_k(df, rang_col, 5)
    mrr = mrr_gagnant(df, rang_col)
    try:
        auc = round(roc_auc_score(y_vrai, df[proba_ou_score_col]), 4)
    except ValueError as e:
        auc = None
        lib.log(f"   [{label}] AUC non calculable : {e}")
    lib.log(f"\n   -- {label} (n={stats_rang['n_courses']} courses) --")
    lib.log(f"      top-1  : {stats_rang['top1_pct']}%")
    lib.log(f"      top-3  : {stats_rang['cumul_top3_pct']}%")
    lib.log(f"      top-5  : {stats_rang['cumul_top5_pct']}%")
    lib.log(f"      NDCG@3 : {ndcg3}")
    lib.log(f"      NDCG@5 : {ndcg5}")
    lib.log(f"      MRR    : {mrr}")
    lib.log(f"      AUC (secondaire) : {auc}")
    return {
        "n_courses": stats_rang["n_courses"],
        "top1_pct": stats_rang["top1_pct"],
        "top3_pct": stats_rang["cumul_top3_pct"],
        "top5_pct": stats_rang["cumul_top5_pct"],
        "ndcg3": ndcg3,
        "ndcg5": ndcg5,
        "mrr": mrr,
        "auc": auc,
    }


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError(
            "scikit-learn n'est pas installe. Ce script doit tourner dans "
            "l'environnement GitHub Actions du workflow dedie, pas en local."
        )
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError(
            "lightgbm n'est pas installe (necessaire pour l'objectif de ranking "
            "'lambdarank'). Verifier l'etape 'pip install' du workflow."
        )

    lib.log("=" * 100)
    lib.log("PISTE 1 -- OBJECTIF DE RANKING INTRA-COURSE (validation uniquement) -- 27/08/2026")
    lib.log("=" * 100)

    lib.log(f"\nChargement du checkpoint {CHECKPOINT_PATH} (produit par "
             "entrainer_v3_phase1_ranking.py, copie stricte de la phase 1 v3, "
             "aucune reconnexion a Supabase)...")
    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    X_train_v3 = checkpoint["X_train_v3"]
    X_val_v3 = checkpoint["X_val_v3"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    y_val_gagnant = checkpoint["y_val_gagnant"]
    df_val = checkpoint["df_val"].reset_index(drop=True)
    course_id_train = checkpoint["course_id_train"]
    lib.log(f"   Checkpoint charge : X_train_v3={X_train_v3.shape}, X_val_v3={X_val_v3.shape}, "
             f"df_val={df_val.shape}. TEST (X_test_v3/df_test) volontairement non charge/non "
             "utilise a ce stade -- protocole de Dorian : validation d'abord, TEST A ensuite "
             "seulement si le gain est confirme.")

    assert len(X_train_v3) == len(course_id_train), "course_id_train desaligne de X_train_v3"
    assert len(X_val_v3) == len(df_val), "df_val desaligne de X_val_v3"

    groups_train = groupes_consecutifs(course_id_train)
    groups_val = groupes_consecutifs(df_val["course_id"])
    assert sum(groups_train) == len(X_train_v3), "somme des groupes TRAIN != nb lignes X_train_v3"
    assert sum(groups_val) == len(X_val_v3), "somme des groupes VALIDATION != nb lignes X_val_v3"
    assert len(groups_train) == course_id_train.nunique(), (
        "des lignes d'une meme course ne sont pas contigues en TRAIN -- le tri amont a du changer"
    )
    assert len(groups_val) == df_val["course_id"].nunique(), (
        "des lignes d'une meme course ne sont pas contigues en VALIDATION -- le tri amont a du changer"
    )
    lib.log(f"\n   Groupes ranking : {len(groups_train)} courses en TRAIN, {len(groups_val)} courses en VALIDATION "
             f"(tailles verifiees coherentes avec X_train_v3/X_val_v3).")

    # =========================================================================
    # MODELE "ACTUEL" (objectif pointwise, meme protocole que la production
    # v3-gagnant) -- recalcule ICI sur fit TRAIN / eval VALIDATION, pour une
    # comparaison strictement apples-to-apples avec le modele ranking
    # ci-dessous (meme checkpoint, meme run, aucun ecart de donnees possible).
    # =========================================================================
    lib.log("\n[BASELINE] Modele v3-gagnant actuel (objectif pointwise, HistGradientBoosting) "
             "-- recherche d'hyperparametres identique a la production (GRILLE_GBM, fit TRAIN / eval VALIDATION)...")
    params_baseline, auc_baseline_grille = lib.entrainer_gbm_avec_grille(
        X_train_v3, y_train_gagnant, X_val_v3, y_val_gagnant, lib.GRILLE_GBM, "v3-gagnant-baseline (piste1)")
    lib.log(f"   Meilleurs hyperparametres : {params_baseline} (AUC validation={round(auc_baseline_grille, 4)})")
    modele_baseline = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_baseline)
    modele_baseline.fit(X_train_v3, y_train_gagnant)
    df_val["proba_baseline"] = modele_baseline.predict_proba(X_val_v3)[:, 1]
    df_val["rang_baseline"] = df_val.groupby("course_id")["proba_baseline"].rank(method="min", ascending=False)

    # =========================================================================
    # MODELE RANKING (piste 1) -- objectif "lambdarank" (pairwise/listwise),
    # groupe par course_id. UNE SEULE variante testee, complexite d'arbre
    # alignee sur les hyperparametres v3 deja valides (pas de grille
    # supplementaire) ; le nombre d'arbres est choisi par early stopping sur
    # le NDCG de VALIDATION (partie standard et necessaire de cet objectif,
    # pas une multiplication d'essais).
    # =========================================================================
    lib.log("\n[RANKING] Modele piste 1 (LightGBM, objectif='lambdarank', groupe=course_id) "
             "-- variante unique, complexite alignee sur les hyperparametres v3 (max_depth=5, "
             "learning_rate=0.05, min_samples_leaf=40, l2=1.0), early stopping sur NDCG de VALIDATION...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=5,
        learning_rate=0.05,
        min_child_samples=40,
        reg_lambda=1.0,
        n_estimators=500,
        random_state=lib.RANDOM_SEED,
        verbosity=-1,
    )
    ranker.fit(
        X_train_v3, y_train_gagnant,
        group=groups_train,
        eval_set=[(X_val_v3, y_val_gagnant)],
        eval_group=[groups_val],
        eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=True), lgb.log_evaluation(period=50)],
    )
    lib.log(f"   Nombre d'arbres retenu (early stopping sur NDCG@5 validation) : {ranker.best_iteration_}")
    df_val["score_ranking"] = ranker.predict(X_val_v3)
    df_val["rang_ranking"] = df_val.groupby("course_id")["score_ranking"].rank(method="min", ascending=False)

    # =========================================================================
    # DOUBLE BENCHMARK (protocole permanent verrouille le 26/08/2026) --
    # benchmark reel (reference principale) et benchmark donnees propres,
    # affiches cote a cote, sur la population VALIDATION uniquement.
    # =========================================================================
    exclusions = lib.charger_exclusions_benchmark()
    df_val = lib.appliquer_benchmarks(df_val, exclusions)
    rapport_pop = lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 1 (ranking)")

    df_val_reel = df_val[df_val["est_benchmark_reel"]]
    df_val_propre = df_val[df_val["est_benchmark_propre"]]

    lib.log("\n" + "=" * 100)
    lib.log("=== RESULTATS -- BENCHMARK REEL (reference principale) ===")
    lib.log("=" * 100)
    m_baseline_reel = calculer_toutes_metriques(
        df_val_reel, "rang_baseline", "proba_baseline", df_val_reel["est_gagnant"], "v3-gagnant actuel (pointwise)")
    m_ranking_reel = calculer_toutes_metriques(
        df_val_reel, "rang_ranking", "score_ranking", df_val_reel["est_gagnant"], "piste 1 (ranking lambdarank)")

    lib.log("\n" + "=" * 100)
    lib.log("=== RESULTATS -- BENCHMARK DONNEES PROPRES ===")
    lib.log("=" * 100)
    m_baseline_propre = calculer_toutes_metriques(
        df_val_propre, "rang_baseline", "proba_baseline", df_val_propre["est_gagnant"], "v3-gagnant actuel (pointwise)")
    m_ranking_propre = calculer_toutes_metriques(
        df_val_propre, "rang_ranking", "score_ranking", df_val_propre["est_gagnant"], "piste 1 (ranking lambdarank)")

    # =========================================================================
    # RESUME COMPARATIF FINAL
    # =========================================================================
    lib.log("\n" + "=" * 100)
    lib.log("=== RESUME COMPARATIF -- v3-gagnant actuel vs piste 1 (ranking) -- VALIDATION uniquement ===")
    lib.log("=" * 100)
    lib.log(f"Population : {rapport_pop['n_total']} courses VALIDATION au total "
             f"({rapport_pop['n_benchmark_reel']} benchmark reel, {rapport_pop['n_benchmark_propre']} benchmark propre).")

    def ligne(nom, m):
        lib.log(f"   {nom:38s} top1={m['top1_pct']:>5}%  top3={m['top3_pct']:>5}%  top5={m['top5_pct']:>5}%  "
                 f"NDCG@3={m['ndcg3']}  NDCG@5={m['ndcg5']}  MRR={m['mrr']}  AUC={m['auc']}")

    lib.log("\n-- Benchmark REEL --")
    ligne("v3-gagnant actuel (pointwise)", m_baseline_reel)
    ligne("Piste 1 (ranking lambdarank)", m_ranking_reel)
    lib.log(f"   Delta top-1 (ranking - actuel) : {round(m_ranking_reel['top1_pct'] - m_baseline_reel['top1_pct'], 1)} pt")
    lib.log(f"   Delta top-5 (ranking - actuel) : {round(m_ranking_reel['top5_pct'] - m_baseline_reel['top5_pct'], 1)} pt")

    lib.log("\n-- Benchmark PROPRE --")
    ligne("v3-gagnant actuel (pointwise)", m_baseline_propre)
    ligne("Piste 1 (ranking lambdarank)", m_ranking_propre)
    lib.log(f"   Delta top-1 (ranking - actuel) : {round(m_ranking_propre['top1_pct'] - m_baseline_propre['top1_pct'], 1)} pt")
    lib.log(f"   Delta top-5 (ranking - actuel) : {round(m_ranking_propre['top5_pct'] - m_baseline_propre['top5_pct'], 1)} pt")

    lib.log("\nCe rapport est le resultat REEL de VALIDATION, non ajuste. Aucun TEST A ni TEST B "
             "n'a ete lance dans ce run -- decision d'y passer laissee a Dorian, sur la base de ce "
             "resultat de validation (gain robuste sur top-5, attention particuliere au top-1, "
             "stabilite entre les deux benchmarks).")


if __name__ == "__main__":
    main()
