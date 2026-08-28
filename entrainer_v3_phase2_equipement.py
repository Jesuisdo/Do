# -*- coding: utf-8 -*-
"""
entrainer_v3_phase2_equipement.py -- PHASE 2/2 de la piste 3 (changement
d'equipement du jour -- oeilleres), demandee par Dorian le 28/08/2026.
Entraine, depuis le checkpoint produit par entrainer_v3_phase1_equipement.py,
TROIS modeles sur les MEMES splits train/validation :
  1. v3-gagnant (pointwise, memes hyperparametres/grille que le run de
     reference) -- reproduit pour la comparaison a 3 termes ;
  2. B = lambdarank_graded SUR LES SEULES VARIABLES v3 (aucun equipement)
     -- reproduit a l'identique du run piste 1 retenu par Dorian ;
  3. B + equipement = lambdarank_graded, MEMES hyperparametres que B, sur
     v3 + les 12 variables d'equipement (oeilleres) point-in-time.

Applique le double benchmark (reel / donnees propres) puis l'analyse
demandee explicitement par Dorian le 28/08/2026 :
  (a) top-1/top-3/top-5/NDCG@3/NDCG@5/MRR/AUC, benchmark reel + propre ;
  (b) HANDICAP vs NON HANDICAP ;
  (c) petits/moyens/grands champs ;
  (d) courses "toujours ratees" (noyau dur des 71,3%, ni v3-gagnant ni B
      en pick #1) : combien deviennent pick #1 avec B+equipement,
      evolution du rang moyen/median du vrai gagnant ;
  (e) courses ou B se trompe mais B+equipement corrige : taux de
      correction + profil Cohen's d (vrai gagnant corrige vs pick fautif
      de B) sur les 12 variables d'equipement ;
  (f) couverture reelle de chaque nouvelle feature sur VALIDATION, et
      Top-1/Top-5 restreints aux VRAIS GAGNANTS qui ont eux-memes ce
      flag d'equipement a 1 (repond a "Top-1 et Top-5 sur ces
      sous-groupes"), avec la taille d'echantillon n toujours affichee a
      cote du pourcentage et un avertissement explicite si n est trop
      petit pour conclure -- demande explicite de Dorian ("verifie que
      son effet ne vient pas simplement d'un sous-echantillon minuscule").

AUCUN TEST A/B lance ici (validation uniquement, comme demande).
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

CHECKPOINT_PATH = "checkpoint_v3_phase1_equipement.pkl"
SEUIL_N_MIN_FIABLE = 30  # en-dessous, un pourcentage est signale comme peu fiable


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
    lib.log(f"   {label:32s} top1={stats_rang['top1_pct']:>5}%  top3={stats_rang['cumul_top3_pct']:>5}%  "
            f"top5={stats_rang['cumul_top5_pct']:>5}%  NDCG@3={ndcg3}  NDCG@5={ndcg5}  MRR={mrr}  AUC={auc}")
    return {
        "n_courses": stats_rang["n_courses"], "top1_pct": stats_rang["top1_pct"],
        "top3_pct": stats_rang["cumul_top3_pct"], "top5_pct": stats_rang["cumul_top5_pct"],
        "ndcg3": ndcg3, "ndcg5": ndcg5, "mrr": mrr, "auc": auc,
    }


def entrainer_lambdarank(X_train, y_train, groups_train, X_val, y_val_eval, groups_val, label):
    params = dict(
        objective="lambdarank", metric="ndcg", boosting_type="gbdt",
        num_leaves=31, max_depth=5, learning_rate=0.05, min_child_samples=40,
        reg_lambda=1.0, n_estimators=500, random_state=lib.RANDOM_SEED, verbosity=-1,
    )
    modele = lgb.LGBMRanker(**params)
    modele.fit(
        X_train, y_train, group=groups_train,
        eval_set=[(X_val, y_val_eval)], eval_group=[groups_val], eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)],
    )
    lib.log(f"   [{label}] arbres retenus={modele.best_iteration_}")
    return modele


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


def top1_par_masque(df, rang_col, masque, label_segment):
    d = df[masque]
    n = len(d)
    if n == 0:
        return None
    n_gagnants = int((d["est_gagnant"] == 1).sum())
    n_reussis = int(((d["est_gagnant"] == 1) & (d[rang_col] == 1)).sum())
    pct = round(100 * n_reussis / n_gagnants, 1) if n_gagnants else float("nan")
    return {"segment": label_segment, "n_courses": n_gagnants, "n_reussis": n_reussis, "pct_top1": pct}


def topk_sur_gagnants_filtres(gagnants, rang_col, masque, k, label):
    """Sur le sous-ensemble de VRAIS GAGNANTS pour lesquels `masque` est
    vrai (typiquement : le vrai gagnant a lui-meme declenche ce flag
    d'equipement le jour de sa victoire), quelle fraction est classee
    dans le top-k par le modele ? Retourne aussi n (taille de
    l'echantillon) et un avertissement explicite si n < SEUIL_N_MIN_FIABLE
    -- demande explicite de Dorian de verifier qu'un effet fort n'est pas
    un artefact de petit echantillon."""
    sous = gagnants[masque]
    n = len(sous)
    if n == 0:
        return {"label": label, "n": 0, "topk": k, "pct": None, "fiable": False}
    n_topk = int((sous[rang_col] <= k).sum())
    pct = round(100 * n_topk / n, 1)
    return {"label": label, "n": n, "topk": k, "pct": pct, "fiable": n >= SEUIL_N_MIN_FIABLE}


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn non installe.")
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError("lightgbm non installe.")

    lib.log("=" * 100)
    lib.log("PISTE 3 -- EQUIPEMENT -- PHASE 2/2 -- v3-gagnant vs B (lambdarank_graded) vs B+equipement -- 28/08/2026")
    lib.log("=" * 100)

    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    X_train_v3 = checkpoint["X_train_v3"]
    X_val_v3 = checkpoint["X_val_v3"]
    X_train_v3_equip = checkpoint["X_train_v3_equip"]
    X_val_v3_equip = checkpoint["X_val_v3_equip"]
    y_train_place = checkpoint["y_train_place"]
    y_val_place = checkpoint["y_val_place"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    y_val_gagnant = checkpoint["y_val_gagnant"]
    df_val = checkpoint["df_val"].reset_index(drop=True)
    course_id_train = checkpoint["course_id_train"]
    colonnes_equipement = checkpoint["colonnes_equipement"]

    lib.log(f"\n   Checkpoint charge : X_train_v3={X_train_v3.shape}, X_train_v3_equip={X_train_v3_equip.shape}, "
            f"X_val_v3={X_val_v3.shape}, df_val={df_val.shape}. Memes features/splits que la reference -- aucune reconstruction.")

    groups_train = groupes_consecutifs(course_id_train)
    groups_val = groupes_consecutifs(df_val["course_id"])
    assert sum(groups_train) == len(X_train_v3) and sum(groups_val) == len(X_val_v3)

    # =========================================================================
    # [1/4] v3-gagnant (pointwise, reproduction a l'identique)
    # =========================================================================
    lib.log("\n[1/4] Reproduction du baseline v3-gagnant (pointwise)...")
    params_baseline, _ = lib.entrainer_gbm_avec_grille(
        X_train_v3, y_train_gagnant, X_val_v3, y_val_gagnant, lib.GRILLE_GBM, "v3-gagnant-baseline")
    modele_baseline = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_baseline)
    modele_baseline.fit(X_train_v3, y_train_gagnant)
    df_val["proba_v3gagnant"] = modele_baseline.predict_proba(X_val_v3)[:, 1]
    df_val["rang_v3gagnant"] = df_val.groupby("course_id")["proba_v3gagnant"].rank(method="min", ascending=False)

    # =========================================================================
    # [2/4] B = lambdarank_graded, v3 seul (reproduction du candidat retenu)
    # =========================================================================
    lib.log("\n[2/4] Reproduction de B (lambdarank_graded, v3 seul, memes hyperparametres que le run piste 1)...")
    y_train_graded = np.where(y_train_gagnant == 1, 2, np.where(y_train_place == 1, 1, 0)).astype(int)
    modele_B = entrainer_lambdarank(X_train_v3, y_train_graded, groups_train, X_val_v3, y_val_gagnant, groups_val, "B (v3 seul)")
    df_val["score_B"] = modele_B.predict(X_val_v3)
    df_val["rang_B"] = df_val.groupby("course_id")["score_B"].rank(method="min", ascending=False)

    # =========================================================================
    # [3/4] B + equipement -- MEMES hyperparametres, MEMES labels graded,
    # MEMES groupes -- seule la matrice de features change (12 colonnes en plus).
    # =========================================================================
    lib.log(f"\n[3/4] B + equipement (lambdarank_graded, v3 + {len(colonnes_equipement)} variables d'equipement)...")
    modele_B_equip = entrainer_lambdarank(X_train_v3_equip, y_train_graded, groups_train, X_val_v3_equip, y_val_gagnant, groups_val, "B+equipement")
    df_val["score_equip"] = modele_B_equip.predict(X_val_v3_equip)
    df_val["rang_equip"] = df_val.groupby("course_id")["score_equip"].rank(method="min", ascending=False)

    # =========================================================================
    # [4/4] DOUBLE BENCHMARK + comparaison a 3 termes
    # =========================================================================
    exclusions = lib.charger_exclusions_benchmark()
    df_val = lib.appliquer_benchmarks(df_val, exclusions)
    rapport_pop = lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 3, equipement (oeilleres)")

    df_val_reel = df_val[df_val["est_benchmark_reel"]]
    df_val_propre = df_val[df_val["est_benchmark_propre"]]

    modeles = [
        ("v3-gagnant actuel (pointwise)", "rang_v3gagnant", "proba_v3gagnant"),
        ("B (lambdarank_graded, v3 seul)", "rang_B", "score_B"),
        ("B + equipement", "rang_equip", "score_equip"),
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
    lib.log("=== DELTAS -- B+equipement vs B (le candidat de reference actuel) ===")
    lib.log("=" * 100)
    for label_bench, res in [("REEL", res_reel), ("PROPRE", res_propre)]:
        b, be = res["B (lambdarank_graded, v3 seul)"], res["B + equipement"]
        lib.log(f"   [{label_bench}] top1={round(be['top1_pct']-b['top1_pct'],1):+}pt  top5={round(be['top5_pct']-b['top5_pct'],1):+}pt  "
                f"NDCG@5={round(be['ndcg5']-b['ndcg5'],4):+}  MRR={round(be['mrr']-b['mrr'],4):+}  AUC={round((be['auc'] or 0)-(b['auc'] or 0),4):+}")

    # =========================================================================
    # ANALYSE D'ERREURS APPROFONDIE -- demandee par Dorian le 28/08/2026
    # =========================================================================
    lib.log("\n" + "=" * 100)
    lib.log("=== ANALYSE D'ERREURS APPROFONDIE (benchmark reel) -- v3-gagnant / B / B+equipement ===")
    lib.log("=" * 100)

    d = df_val_reel.copy()
    d["bucket_partants"] = d["nb_partants_reel"].apply(lib.bucket_partants)
    d["est_handicap"] = d["categorie_particularite"].fillna("").str.contains("HANDICAP")

    lib.log("\n-- (a) top-1 / top-3 / top-5 / NDCG / MRR / AUC global -- deja ci-dessus (tableau des 3 modeles) --")

    lib.log("\n-- (b) HANDICAP en particulier --")
    for est_h, label_h in [(True, "HANDICAP"), (False, "NON HANDICAP")]:
        masque = d["est_handicap"] == est_h
        for nom, rang_col, _ in modeles:
            r = top1_par_masque(d, rang_col, masque, label_h)
            if r:
                lib.log(f"   {label_h:15s} {nom:32s} n={r['n_courses']:>6}  top1={r['pct_top1']}%")

    lib.log("\n-- (c) Petits / moyens / grands champs --")
    for bucket in ["petit (<=7)", "moyen (8-12)", "grand (13+)"]:
        masque = d["bucket_partants"] == bucket
        for nom, rang_col, _ in modeles:
            r = top1_par_masque(d, rang_col, masque, bucket)
            if r:
                lib.log(f"   {bucket:15s} {nom:32s} n={r['n_courses']:>6}  top1={r['pct_top1']}%")

    lib.log("\n-- (d) Courses 'toujours ratees' par v3-gagnant ET B (noyau dur des 71,3%, ni l'un ni l'autre en pick #1) --")
    gagnants = d[d["est_gagnant"] == 1].copy()
    toujours_ratees = gagnants[(gagnants["rang_v3gagnant"] != 1) & (gagnants["rang_B"] != 1)]
    n_tr = len(toujours_ratees)
    if n_tr:
        rang_moyen_B = round(float(toujours_ratees["rang_B"].mean()), 2)
        rang_moyen_equip = round(float(toujours_ratees["rang_equip"].mean()), 2)
        rang_median_B = round(float(toujours_ratees["rang_B"].median()), 2)
        rang_median_equip = round(float(toujours_ratees["rang_equip"].median()), 2)
        pct_devient_top1 = round(100 * float((toujours_ratees["rang_equip"] == 1).mean()), 1)
        pct_ameliore = round(100 * float((toujours_ratees["rang_equip"] < toujours_ratees["rang_B"]).mean()), 1)
        pct_degrade = round(100 * float((toujours_ratees["rang_equip"] > toujours_ratees["rang_B"]).mean()), 1)
        lib.log(f"   n={n_tr} courses toujours ratees (sur {len(gagnants)} courses VALIDATION benchmark reel, "
                f"{round(100*n_tr/len(gagnants),1)}%).")
        lib.log(f"   Rang moyen du gagnant   : B={rang_moyen_B}  B+equipement={rang_moyen_equip}")
        lib.log(f"   Rang median du gagnant  : B={rang_median_B}  B+equipement={rang_median_equip}")
        lib.log(f"   Part de ces courses ou B+equipement devient pick #1 (course reellement 'debloquee') : {pct_devient_top1}%")
        lib.log(f"   Part ou le rang s'ameliore (meme sans devenir #1) : {pct_ameliore}%  (se degrade : {pct_degrade}%)")
    else:
        lib.log("   Aucune course 'toujours ratee' trouvee (inattendu).")

    lib.log("\n-- (e) Courses ou B se trompe mais B+equipement corrige (pick #1 devient le vrai gagnant) --")
    ratees_B = gagnants[gagnants["rang_B"] != 1].copy()
    corrigees_equip = ratees_B[ratees_B["rang_equip"] == 1]
    n_corrigees = len(corrigees_equip)
    n_ratees_B = len(ratees_B)
    lib.log(f"   Sur {n_ratees_B} courses ou B se trompe, B+equipement corrige (trouve le gagnant) dans "
            f"{n_corrigees} cas ({round(100*n_corrigees/n_ratees_B,1) if n_ratees_B else 0}%).")

    if n_corrigees > 0:
        picks_fautifs_B = d[(d["course_id"].isin(corrigees_equip["course_id"])) & (d["rang_B"] == 1)].copy()
        vrais_gagnants = corrigees_equip.copy()
        vrais_gagnants["role"] = "vrai_gagnant_corrige"
        picks_fautifs_B["role"] = "pick_fautif_de_B"
        comparaison = pd.concat([vrais_gagnants, picks_fautifs_B], axis=0, ignore_index=True)
        profils = profils_par_groupe(comparaison, "role", ["vrai_gagnant_corrige", "pick_fautif_de_B"], colonnes_equipement)
        lib.log(f"\n   Profil equipement : vrai gagnant (corrige par B+equipement, n={n_corrigees}) vs pick fautif de B, "
                "memes courses, tries par |d de Cohen| :")
        for _, row in profils.iterrows():
            marqueur = " <-- ecart notable" if row["d_cohen"] is not None and abs(row["d_cohen"]) >= 0.2 else ""
            lib.log(f"      {row['variable']:48s} gagnant_corrige={row['vrai_gagnant_corrige_moyenne']}  "
                    f"pick_fautif_B={row['pick_fautif_de_B_moyenne']}  d={row['d_cohen']}{marqueur}")
        if n_corrigees < SEUIL_N_MIN_FIABLE:
            lib.log(f"   ATTENTION : n={n_corrigees} < {SEUIL_N_MIN_FIABLE} -- ce profil repose sur un tres petit "
                    f"echantillon, a interpreter avec prudence (demande explicite de verification de Dorian).")

    lib.log("\n-- (f) Couverture reelle des nouvelles features sur VALIDATION (benchmark reel) --")
    n_val_reel = len(d)
    for col in colonnes_equipement:
        n_non_null = int(d[col].notna().sum())
        pct_cov = round(100 * n_non_null / n_val_reel, 1) if n_val_reel else float("nan")
        vals = d[col].dropna().unique().tolist()
        if set(vals) <= {0, 1, 0.0, 1.0}:
            n_pos = int((d[col] == 1).sum())
            lib.log(f"   {col:48s} couverture={pct_cov:>5}% ({n_non_null}/{n_val_reel})  n_positifs={n_pos}")
        else:
            lib.log(f"   {col:48s} couverture={pct_cov:>5}% ({n_non_null}/{n_val_reel})")

    lib.log("\n-- (f, suite) Top-1 / Top-5 restreints aux VRAIS GAGNANTS presentant eux-memes chaque flag "
            "d'equipement (n toujours affiche -- verification anti-petit-echantillon) --")
    flags_binaires = [
        "equip_premiere_pose_oeilleres", "equip_retour_apres_absence", "equip_retrait_oeilleres",
        "equip_classique_vers_australienne", "equip_australienne_vers_classique", "equip_changement_generique",
    ]
    for flag in flags_binaires:
        if flag not in gagnants.columns:
            continue
        masque = gagnants[flag] == 1
        for k in (1, 5):
            for nom, rang_col, _ in modeles:
                r = topk_sur_gagnants_filtres(gagnants, rang_col, masque, k, nom)
                if r["n"] == 0:
                    continue
                avert = "" if r["fiable"] else "  <-- n TROP PETIT, non concluant"
                lib.log(f"   {flag:38s} top{k}  {nom:32s} n={r['n']:>4}  pct={r['pct']}%{avert}")

    lib.log("\n" + "=" * 100)
    lib.log("=== RESUME FINAL -- ce rapport est le resultat REEL de VALIDATION, non ajuste. ===")
    lib.log("=== AUCUN TEST A ni TEST B lance -- decision laissee a Dorian. ===")
    lib.log("=" * 100)


if __name__ == "__main__":
    main()
