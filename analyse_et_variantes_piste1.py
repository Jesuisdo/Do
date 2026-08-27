# -*- coding: utf-8 -*-
"""
analyse_et_variantes_piste1.py -- Analyse d'erreurs approfondie du premier
LambdaRank (piste 1) + test de variantes ciblees, demande par Dorian le
27/08/2026 apres un premier gain juge trop faible (+0,6/+0,7 pt top-1,
+0,4 pt top-5, run GitHub Actions #33099229947).

NE SE CONNECTE PAS A SUPABASE, NE RECONSTRUIT AUCUNE VARIABLE. Reutilise
tel quel le checkpoint_v3_phase1_ranking.pkl du run #1 (telecharge en
artefact CROSS-RUN -- memes features, memes splits, aucun risque de
derive de donnees entre l'analyse et les variantes testees ici).

PARTIE 1 -- reproduit le baseline v3-gagnant (pointwise) et le LambdaRank
original (memes hyperparametres que le run #1), pour disposer des
predictions ligne par ligne necessaires a l'analyse d'erreurs (le run #1
n'imprimait que des metriques agregees, sans les sauvegarder).

PARTIE 2 -- analyse d'erreurs approfondie (baseline vs LambdaRank original) :
  2a. confusion top-1 (corrigee / degradee / inchangee) par taille de champ,
      handicap, distance, terrain, age, sexe, type de piste ;
  2b. profil des chevaux qui montent/descendent le plus dans le classement
      (top et bottom decile de variation de rang), sur les variables de
      diagnostic deja definies (v3_lib.VARIABLES_DIAGNOSTIC_ERREURS) et les
      variables relatives v3 (rang/z intra-course) ;
  2c. dans les courses "toujours ratees" (gagnant pas en pick #1 ni avant ni
      apres), le rang donne au gagnant s'ameliore-t-il quand meme ? ;
  2d. diagnostic de la baisse d'AUC : deplacement coherent (AUC recalculee
      sur un score normalise intra-course) ou vraie perte de pouvoir
      discriminant.

PARTIE 3 -- 3 variantes de ranking, choisies pour couvrir les 3 hypotheses
les plus directement testables a partir de l'analyse ci-dessus, toutes a
partir des SEULES donnees deja dans le checkpoint (aucune nouvelle feature) :
  - lambdarank_trunc5 : LambdaRank, lambdarank_truncation_level=5 (le
    gradient de la perte n'est plus calcule que sur les 5 premieres
    positions du classement, au lieu de 20 par defaut) -- pondere
    directement le haut du classement, la ou porte l'objectif de Dorian ;
  - lambdarank_graded : LambdaRank, relevance a 3 niveaux (gagnant=2,
    place-non-gagnant=1, sinon 0) au lieu du label binaire actuel --
    construite uniquement a partir de y_train_gagnant/y_train_place deja
    dans le checkpoint (aucune nouvelle donnee), pour donner au modele un
    signal plus fin sur les chevaux "proches" (2e/3e) plutot que de tout
    traiter comme un seul bloc de negatifs ;
  - xgboost_pairwise : formulation pairwise differente (RankNet classique,
    XGBoost rank:pairwise, ponderation uniforme des paires) en contraste
    du pairwise pondere par le gain NDCG de LambdaRank.

Toutes les variantes : memes features (X_train_v3/X_val_v3 inchanges),
memes splits, validation interne uniquement, comparaison systematique a
v3-gagnant ET au LambdaRank original, double benchmark (reel / propre).
AUCUN TEST A lance ici.
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

try:
    import xgboost as xgb
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False

CHECKPOINT_PATH = "checkpoint_v3_phase1_ranking.pkl"


def groupes_consecutifs(course_id_iterable):
    return [len(list(g)) for _, g in itertools.groupby(list(course_id_iterable))]


def ndcg_gagnant_at_k(df, rang_col, k, cible_col="est_gagnant"):
    d = df[df[cible_col] == 1]
    n = len(d)
    if n == 0:
        return float("nan")
    gains = d[rang_col].apply(lambda r: 1.0 / np.log2(r + 1) if r <= k else 0.0)
    return round(float(gains.mean()), 4)


def mrr_gagnant(df, rang_col, cible_col="est_gagnant"):
    d = df[df[cible_col] == 1]
    n = len(d)
    if n == 0:
        return float("nan")
    return round(float((1.0 / d[rang_col]).mean()), 4)


def calculer_toutes_metriques(df, rang_col, proba_ou_score_col, y_vrai, label):
    stats_rang, _ = lib.rang_distribution_gagnant(df, rang_col)
    ndcg3 = ndcg_gagnant_at_k(df, rang_col, 3)
    ndcg5 = ndcg_gagnant_at_k(df, rang_col, 5)
    mrr = mrr_gagnant(df, rang_col)
    try:
        auc = round(roc_auc_score(y_vrai, df[proba_ou_score_col]), 4)
    except ValueError as e:
        auc = None
        lib.log(f"   [{label}] AUC non calculable : {e}")
    lib.log(f"   {label:38s} top1={stats_rang['top1_pct']:>5}%  top3={stats_rang['cumul_top3_pct']:>5}%  "
             f"top5={stats_rang['cumul_top5_pct']:>5}%  NDCG@3={ndcg3}  NDCG@5={ndcg5}  MRR={mrr}  AUC={auc}")
    return {
        "n_courses": stats_rang["n_courses"], "top1_pct": stats_rang["top1_pct"],
        "top3_pct": stats_rang["cumul_top3_pct"], "top5_pct": stats_rang["cumul_top5_pct"],
        "ndcg3": ndcg3, "ndcg5": ndcg5, "mrr": mrr, "auc": auc,
    }


def profils_par_groupe(df, groupe_col, groupes_labels, variables):
    """Compare des variables entre 2 groupes nommes, triees par |d de Cohen|
    decroissant entre groupes_labels[0] et groupes_labels[-1]."""
    resultats = []
    for var in variables:
        if var not in df.columns:
            continue
        stats = {}
        for g in groupes_labels:
            sous = df.loc[df[groupe_col] == g, var]
            stats[g] = (sous.mean(), sous.std(), int(sous.notna().sum()))
        m1, s1, n1 = stats[groupes_labels[0]]
        m2, s2, n2 = stats[groupes_labels[-1]]
        if n1 > 1 and n2 > 1 and pd.notna(m1) and pd.notna(m2) and pd.notna(s1) and pd.notna(s2):
            pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / max(n1 + n2 - 2, 1))
            d = (m1 - m2) / pooled if pooled and pooled > 0 else np.nan
        else:
            d = np.nan
        row = {"variable": var, "d_cohen": round(float(d), 3) if pd.notna(d) else None}
        for g in groupes_labels:
            m, _, n = stats[g]
            row[f"{g}_moyenne"] = round(float(m), 3) if pd.notna(m) else None
            row[f"{g}_n"] = n
        resultats.append(row)
    out = pd.DataFrame(resultats)
    if not out.empty:
        out = out.reindex(out["d_cohen"].abs().sort_values(ascending=False, na_position="last").index)
    return out


def entrainer_lambdarank(X_train, y_train, groups_train, X_val, y_val_eval, groups_val, label, extra_params=None):
    params = dict(
        objective="lambdarank", metric="ndcg", boosting_type="gbdt",
        num_leaves=31, max_depth=5, learning_rate=0.05, min_child_samples=40,
        reg_lambda=1.0, n_estimators=500, random_state=lib.RANDOM_SEED, verbosity=-1,
    )
    if extra_params:
        params.update(extra_params)
    modele = lgb.LGBMRanker(**params)
    modele.fit(
        X_train, y_train, group=groups_train,
        eval_set=[(X_val, y_val_eval)], eval_group=[groups_val], eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)],
    )
    lib.log(f"   [{label}] parametres additionnels={extra_params}  arbres retenus={modele.best_iteration_}")
    return modele


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn non installe.")
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError("lightgbm non installe.")
    if not XGBOOST_DISPONIBLE:
        raise RuntimeError("xgboost non installe (necessaire pour la variante xgboost_pairwise).")

    lib.log("=" * 100)
    lib.log("PISTE 1 -- ANALYSE D'ERREURS DU LAMBDARANK ORIGINAL + VARIANTES CIBLEES -- 27/08/2026")
    lib.log("=" * 100)

    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    X_train_v3 = checkpoint["X_train_v3"]
    X_val_v3 = checkpoint["X_val_v3"]
    y_train_place = checkpoint["y_train_place"]
    y_val_place = checkpoint["y_val_place"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    y_val_gagnant = checkpoint["y_val_gagnant"]
    df_val = checkpoint["df_val"].reset_index(drop=True)
    course_id_train = checkpoint["course_id_train"]
    lib.log(f"   Checkpoint (run #1, cross-run) charge : X_train_v3={X_train_v3.shape}, X_val_v3={X_val_v3.shape}, "
             f"df_val={df_val.shape}. Memes features, memes splits que le run #1 -- aucune reconstruction.")

    groups_train = groupes_consecutifs(course_id_train)
    groups_val = groupes_consecutifs(df_val["course_id"])
    assert sum(groups_train) == len(X_train_v3) and sum(groups_val) == len(X_val_v3)
    assert len(groups_train) == course_id_train.nunique() and len(groups_val) == df_val["course_id"].nunique()

    # =========================================================================
    # PARTIE 1 -- reproduction du baseline et du LambdaRank original
    # =========================================================================
    lib.log("\n[1/3] Reproduction du baseline v3-gagnant (pointwise, memes hyperparametres que le run #1)...")
    params_baseline, auc_baseline_grille = lib.entrainer_gbm_avec_grille(
        X_train_v3, y_train_gagnant, X_val_v3, y_val_gagnant, lib.GRILLE_GBM, "v3-gagnant-baseline")
    modele_baseline = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_baseline)
    modele_baseline.fit(X_train_v3, y_train_gagnant)
    df_val["proba_baseline"] = modele_baseline.predict_proba(X_val_v3)[:, 1]
    df_val["rang_baseline"] = df_val.groupby("course_id")["proba_baseline"].rank(method="min", ascending=False)

    lib.log("\n   Reproduction du LambdaRank original (memes hyperparametres que le run #1)...")
    modele_original = entrainer_lambdarank(
        X_train_v3, y_train_gagnant, groups_train, X_val_v3, y_val_gagnant, groups_val, "lambdarank_original")
    df_val["score_ranking"] = modele_original.predict(X_val_v3)
    df_val["rang_ranking"] = df_val.groupby("course_id")["score_ranking"].rank(method="min", ascending=False)

    # =========================================================================
    # PARTIE 2 -- ANALYSE D'ERREURS APPROFONDIE (baseline vs LambdaRank original)
    # =========================================================================
    lib.log("\n" + "=" * 100)
    lib.log("[2/3] ANALYSE D'ERREURS APPROFONDIE -- baseline (pointwise) vs LambdaRank original")
    lib.log("=" * 100)

    df_val["nb_partants_reel_int"] = df_val["nb_partants_reel"]
    df_val["bucket_partants"] = df_val["nb_partants_reel"].apply(lib.bucket_partants)

    # -- 2a. confusion top-1 par segment --------------------------------------
    gagnants = df_val[df_val["est_gagnant"] == 1].copy()
    gagnants["ok_baseline"] = gagnants["rang_baseline"] == 1
    gagnants["ok_ranking"] = gagnants["rang_ranking"] == 1

    def categorie_flip(row):
        if row["ok_baseline"] and row["ok_ranking"]:
            return "les_deux_ok"
        if row["ok_baseline"] and not row["ok_ranking"]:
            return "degradee_par_ranking"
        if not row["ok_baseline"] and row["ok_ranking"]:
            return "corrigee_par_ranking"
        return "toujours_ratee"

    gagnants["categorie_flip"] = gagnants.apply(categorie_flip, axis=1)
    lib.log(f"\n2a. Confusion top-1 globale (n={len(gagnants)} courses VALIDATION) :")
    for cat, n in gagnants["categorie_flip"].value_counts().items():
        lib.log(f"    {cat:25s} {n:6d}  ({round(100*n/len(gagnants),1)}%)")

    def confusion_par_segment(col):
        if col not in gagnants.columns:
            return
        lib.log(f"\n    -- par {col} --")
        for seg, sous in gagnants.groupby(gagnants[col].fillna("INCONNU")):
            if len(sous) < 100:
                continue
            n = len(sous)
            n_corr = int((sous["categorie_flip"] == "corrigee_par_ranking").sum())
            n_deg = int((sous["categorie_flip"] == "degradee_par_ranking").sum())
            n_deux = int((sous["categorie_flip"] == "les_deux_ok").sum())
            top1_baseline = round(100 * (n_deux + n_deg) / n, 1)
            top1_ranking = round(100 * (n_deux + n_corr) / n, 1)
            lib.log(f"    {str(seg):20s} n={n:5d}  top1_baseline={top1_baseline:>5}%  top1_ranking={top1_ranking:>5}%  "
                     f"delta={round(top1_ranking-top1_baseline,1):+5}pt  (corrigees={n_corr}, degradees={n_deg})")

    for col in ["bucket_partants", "categorie_particularite", "distance_bucket", "terrain_bucket",
                "condition_age", "condition_sexe", "type_piste"]:
        confusion_par_segment(col)

    # -- 2b. profil des chevaux qui montent/descendent le plus ---------------
    df_val["delta_rang"] = df_val["rang_baseline"] - df_val["rang_ranking"]
    q90 = df_val["delta_rang"].quantile(0.90)
    q10 = df_val["delta_rang"].quantile(0.10)
    df_val["groupe_mouvement"] = "stable"
    df_val.loc[df_val["delta_rang"] >= q90, "groupe_mouvement"] = "monte_fortement"
    df_val.loc[df_val["delta_rang"] <= q10, "groupe_mouvement"] = "descend_fortement"
    n_monte = int((df_val["groupe_mouvement"] == "monte_fortement").sum())
    n_descend = int((df_val["groupe_mouvement"] == "descend_fortement").sum())
    pct_gagnant_monte = round(100 * df_val.loc[df_val["groupe_mouvement"] == "monte_fortement", "est_gagnant"].mean(), 2)
    pct_gagnant_descend = round(100 * df_val.loc[df_val["groupe_mouvement"] == "descend_fortement", "est_gagnant"].mean(), 2)
    lib.log(f"\n2b. Chevaux qui montent/descendent fortement dans le classement (decile extreme de delta_rang) :")
    lib.log(f"    monte_fortement (n={n_monte}) : {pct_gagnant_monte}% sont le vrai gagnant de leur course")
    lib.log(f"    descend_fortement (n={n_descend}) : {pct_gagnant_descend}% sont le vrai gagnant de leur course")

    variables_relatives = [f"{v}_rang_course" for v in lib.VARIABLES_RELATIVES_CIBLES] + \
        [f"{v}_z_course" for v in lib.VARIABLES_RELATIVES_CIBLES]
    variables_profil = lib.VARIABLES_DIAGNOSTIC_ERREURS + [v for v in variables_relatives if v not in lib.VARIABLES_DIAGNOSTIC_ERREURS]
    profils_mouvement = profils_par_groupe(df_val, "groupe_mouvement", ["monte_fortement", "descend_fortement"], variables_profil)
    lib.log("\n    Profil (monte_fortement vs descend_fortement), triees par |d de Cohen| :")
    for _, row in profils_mouvement.head(15).iterrows():
        marqueur = " <-- ecart notable" if row["d_cohen"] is not None and abs(row["d_cohen"]) >= 0.2 else ""
        lib.log(f"    {row['variable']:45s} monte={row['monte_fortement_moyenne']}  descend={row['descend_fortement_moyenne']}  "
                 f"d={row['d_cohen']}{marqueur}")

    # -- 2c. courses toujours ratees : le rang du gagnant progresse-t-il ? ---
    toujours_ratees = gagnants[gagnants["categorie_flip"] == "toujours_ratee"]
    if len(toujours_ratees) > 0:
        rang_moyen_baseline = round(float(toujours_ratees["rang_baseline"].mean()), 2)
        rang_moyen_ranking = round(float(toujours_ratees["rang_ranking"].mean()), 2)
        rang_median_baseline = round(float(toujours_ratees["rang_baseline"].median()), 2)
        rang_median_ranking = round(float(toujours_ratees["rang_ranking"].median()), 2)
        pct_ameliore = round(100 * float((toujours_ratees["rang_ranking"] < toujours_ratees["rang_baseline"]).mean()), 1)
        pct_degrade = round(100 * float((toujours_ratees["rang_ranking"] > toujours_ratees["rang_baseline"]).mean()), 1)
        lib.log(f"\n2c. Courses 'toujours ratees' (gagnant pas en pick #1, ni avant ni apres, n={len(toujours_ratees)}) :")
        lib.log(f"    Rang moyen du gagnant : baseline={rang_moyen_baseline}  ranking={rang_moyen_ranking}")
        lib.log(f"    Rang median du gagnant : baseline={rang_median_baseline}  ranking={rang_median_ranking}")
        lib.log(f"    Part des courses ou le rang du gagnant s'ameliore quand meme : {pct_ameliore}% "
                 f"(se degrade : {pct_degrade}%)")

    # -- 2d. diagnostic de la baisse d'AUC : deplacement coherent ou reel ? --
    df_val["score_ranking_norm"] = (df_val["nb_partants_reel"] - df_val["rang_ranking"] + 1) / df_val["nb_partants_reel"]
    df_val["proba_baseline_norm"] = (df_val["nb_partants_reel"] - df_val["rang_baseline"] + 1) / df_val["nb_partants_reel"]
    auc_ranking_brut = round(roc_auc_score(y_val_gagnant, df_val["score_ranking"]), 4)
    auc_ranking_norm = round(roc_auc_score(y_val_gagnant, df_val["score_ranking_norm"]), 4)
    auc_baseline_brut = round(roc_auc_score(y_val_gagnant, df_val["proba_baseline"]), 4)
    auc_baseline_norm = round(roc_auc_score(y_val_gagnant, df_val["proba_baseline_norm"]), 4)
    lib.log(f"\n2d. Diagnostic de la baisse d'AUC (score brut vs score normalise intra-course [0,1] par taille de champ) :")
    lib.log(f"    Baseline   : AUC(score brut)={auc_baseline_brut}   AUC(score normalise intra-course)={auc_baseline_norm}")
    lib.log(f"    LambdaRank : AUC(score brut)={auc_ranking_brut}   AUC(score normalise intra-course)={auc_ranking_norm}")
    lib.log("    Lecture : si AUC(normalise) du LambdaRank remonte pres du niveau baseline, la baisse d'AUC brute "
             "vient d'une echelle de score non comparable ENTRE courses (LambdaRank n'optimise que l'ordre INTRA-course, "
             "pas la magnitude absolue du score) -- pas d'une perte reelle de pouvoir de classement DANS chaque course.")

    # =========================================================================
    # PARTIE 3 -- VARIANTES CIBLEES
    # =========================================================================
    lib.log("\n" + "=" * 100)
    lib.log("[3/3] VARIANTES DE RANKING CIBLEES (memes features, memes splits, validation uniquement)")
    lib.log("=" * 100)

    resultats_variantes = {}

    # --- Variante A : lambdarank_trunc5 -------------------------------------
    lib.log("\n-- Variante A : lambdarank_trunc5 (lambdarank_truncation_level=5, le gradient ne porte que sur le top-5) --")
    modele_trunc5 = entrainer_lambdarank(
        X_train_v3, y_train_gagnant, groups_train, X_val_v3, y_val_gagnant, groups_val,
        "lambdarank_trunc5", extra_params={"lambdarank_truncation_level": 5})
    df_val["score_trunc5"] = modele_trunc5.predict(X_val_v3)
    df_val["rang_trunc5"] = df_val.groupby("course_id")["score_trunc5"].rank(method="min", ascending=False)

    # --- Variante B : lambdarank_graded (relevance 3 niveaux) ---------------
    lib.log("\n-- Variante B : lambdarank_graded (relevance gagnant=2, place-non-gagnant=1, sinon 0) --")
    y_train_graded = np.where(y_train_gagnant == 1, 2, np.where(y_train_place == 1, 1, 0)).astype(int)
    y_val_graded = np.where(y_val_gagnant == 1, 2, np.where(y_val_place == 1, 1, 0)).astype(int)
    modele_graded = entrainer_lambdarank(
        X_train_v3, y_train_graded, groups_train, X_val_v3, y_val_graded, groups_val, "lambdarank_graded")
    df_val["score_graded"] = modele_graded.predict(X_val_v3)
    df_val["rang_graded"] = df_val.groupby("course_id")["score_graded"].rank(method="min", ascending=False)

    # --- Variante C : xgboost_pairwise (RankNet classique) -------------------
    lib.log("\n-- Variante C : xgboost_pairwise (rank:pairwise, ponderation uniforme des paires) --")
    modele_xgb = xgb.XGBRanker(
        objective="rank:pairwise", n_estimators=500, max_depth=5, learning_rate=0.05,
        reg_lambda=1.0, min_child_weight=40, random_state=lib.RANDOM_SEED,
        early_stopping_rounds=30, eval_metric="ndcg@5",
    )
    modele_xgb.fit(
        X_train_v3, y_train_gagnant, group=groups_train,
        eval_set=[(X_val_v3, y_val_gagnant)], eval_group=[groups_val], verbose=False,
    )
    lib.log(f"   [xgboost_pairwise] arbres retenus={modele_xgb.best_iteration}")
    df_val["score_xgb"] = modele_xgb.predict(X_val_v3)
    df_val["rang_xgb"] = df_val.groupby("course_id")["score_xgb"].rank(method="min", ascending=False)

    # =========================================================================
    # DOUBLE BENCHMARK -- comparaison finale de TOUTES les variantes
    # =========================================================================
    exclusions = lib.charger_exclusions_benchmark()
    df_val = lib.appliquer_benchmarks(df_val, exclusions)
    rapport_pop = lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 1, variantes")

    df_val_reel = df_val[df_val["est_benchmark_reel"]]
    df_val_propre = df_val[df_val["est_benchmark_propre"]]

    modeles = [
        ("v3-gagnant actuel (pointwise)", "rang_baseline", "proba_baseline"),
        ("LambdaRank original", "rang_ranking", "score_ranking"),
        ("Variante A: lambdarank_trunc5", "rang_trunc5", "score_trunc5"),
        ("Variante B: lambdarank_graded", "rang_graded", "score_graded"),
        ("Variante C: xgboost_pairwise", "rang_xgb", "score_xgb"),
    ]

    lib.log("\n" + "=" * 100)
    lib.log("=== RESULTATS -- BENCHMARK REEL (reference principale) ===")
    lib.log("=" * 100)
    res_reel = {}
    for nom, rang_col, proba_col in modeles:
        res_reel[nom] = calculer_toutes_metriques(df_val_reel, rang_col, proba_col, df_val_reel["est_gagnant"], nom)

    lib.log("\n" + "=" * 100)
    lib.log("=== RESULTATS -- BENCHMARK DONNEES PROPRES ===")
    lib.log("=" * 100)
    res_propre = {}
    for nom, rang_col, proba_col in modeles:
        res_propre[nom] = calculer_toutes_metriques(df_val_propre, rang_col, proba_col, df_val_propre["est_gagnant"], nom)

    lib.log("\n" + "=" * 100)
    lib.log("=== RESUME COMPARATIF FINAL -- toutes variantes -- VALIDATION uniquement ===")
    lib.log("=" * 100)
    lib.log(f"Population : {rapport_pop['n_total']} courses ({rapport_pop['n_benchmark_reel']} benchmark reel, "
             f"{rapport_pop['n_benchmark_propre']} benchmark propre).")

    def ligne(nom, m):
        lib.log(f"   {nom:34s} top1={m['top1_pct']:>5}%  top3={m['top3_pct']:>5}%  top5={m['top5_pct']:>5}%  "
                 f"NDCG@3={m['ndcg3']}  NDCG@5={m['ndcg5']}  MRR={m['mrr']}  AUC={m['auc']}")

    lib.log("\n-- Benchmark REEL --")
    for nom, _, _ in modeles:
        ligne(nom, res_reel[nom])
    ref = res_reel["v3-gagnant actuel (pointwise)"]
    for nom, _, _ in modeles[1:]:
        m = res_reel[nom]
        lib.log(f"   Delta vs v3-gagnant actuel -- {nom:30s} top1={round(m['top1_pct']-ref['top1_pct'],1):+.1f}pt  "
                 f"top5={round(m['top5_pct']-ref['top5_pct'],1):+.1f}pt")

    lib.log("\n-- Benchmark PROPRE --")
    for nom, _, _ in modeles:
        ligne(nom, res_propre[nom])
    ref_p = res_propre["v3-gagnant actuel (pointwise)"]
    for nom, _, _ in modeles[1:]:
        m = res_propre[nom]
        lib.log(f"   Delta vs v3-gagnant actuel -- {nom:30s} top1={round(m['top1_pct']-ref_p['top1_pct'],1):+.1f}pt  "
                 f"top5={round(m['top5_pct']-ref_p['top5_pct'],1):+.1f}pt")

    lib.log("\nCe rapport est le resultat REEL de VALIDATION, non ajuste. Toujours aucun TEST A ni TEST B lance -- "
             "decision laissee a Dorian sur la base de l'analyse d'erreurs et de la comparaison des variantes ci-dessus.")


if __name__ == "__main__":
    main()
