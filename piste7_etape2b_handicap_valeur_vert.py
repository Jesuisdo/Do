# -*- coding: utf-8 -*-
"""
piste7_etape2b_handicap_valeur_vert.py -- test handicap_valeur (piste 2 de
la methodologie du 01/09/2026), UNIQUEMENT sur les courses VERT (filtre de
confiance deja fige et valide -- ne pas y toucher). Fait suite au rejet de
la piste "forme recente des adversaires" (etape 2, gain +0.4pt Top-1 mais
non attribuable aux nouvelles variables selon l'importance par permutation
-- rejetee). Dorian a explicitement demande de tester handicap_valeur
independamment, ce rejet ne prejugeant pas de son resultat.

Point architectural INCHANGE (identique a l'etape 2) : le B+genealogie
utilise pour determiner les courses VERT/ORANGE/ROUGE N'EST PAS MODIFIE --
memes 240 variables, meme entrainement sur TRAIN, memes seuils figes
(0.5848 / 0.4662). handicap_valeur n'est utilisee QUE par des modeles
candidats SEPARES, entraines et evalues UNIQUEMENT au sein des courses
deja classees VERT par le filtre inchange, VAL_FIT/VAL_CALIB, aucun TEST A/B.

Particularite de cette experience : handicap_valeur n'est PAS chargee par
la requete SQL de v3_lib.py (REQUETE) -- absente a 100% du pipeline
d'entrainement actuel (audit du 01/09/2026, inventaire livre a Dorian).
Ce script la recupere via une requete SQL LECTURE SEULE ciblee,
SEPAREMENT, uniquement pour les partants des courses deja classees VERT
(aucun impact sur TRAIN, aucune modification de v3_lib.REQUETE ni du
pipeline de production).

Deux candidats INDEPENDANTS, comme demande par Dorian :
- Candidat A : score_geneal, rang_geneal, handicap_valeur (brute).
- Candidat B : score_geneal, rang_geneal, handicap_valeur_rang_course,
handicap_valeur_z_course (rang + z-score intra-course, mecanisme
generique deja en place, calcule sur le champ ENTIER des courses VERT).
Chaque candidat est entraine et evalue separement (memes VAL_FIT/VAL_CALIB
pour les deux, pour rester comparables). Si le brut (A) fonctionne deja,
Dorian a demande de ne pas complexifier avec B.
"""
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    import psycopg2
    PSYCOPG2_DISPONIBLE = True
except ImportError:
    PSYCOPG2_DISPONIBLE = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.inspection import permutation_importance
except ImportError:
    pass

try:
    import lightgbm as lgb
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    LIGHTGBM_DISPONIBLE = False

CHECKPOINT_PATH = "checkpoint_piste6_split4.pkl"
K_SHORTLIST = 5

# Seuil FIGE de la piste 7 -- NE PAS RETOUCHER.
SEUIL_VERT_FIGE = 0.5848

FEATURES_CANDIDAT_A = ["score_geneal", "rang_geneal", "handicap_valeur"]
FEATURES_CANDIDAT_B = ["score_geneal", "rang_geneal", "handicap_valeur_rang_course", "handicap_valeur_z_course"]
VARIABLES_NOUVELLES_A = ["handicap_valeur"]
VARIABLES_NOUVELLES_B = ["handicap_valeur_rang_course", "handicap_valeur_z_course"]


def groupes_consecutifs(course_id_iterable):
    import itertools
    return [len(list(g)) for _, g in itertools.groupby(list(course_id_iterable))]


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
    lib.log(f"    [{label}] arbres retenus={modele.best_iteration_}")
    return modele


def calculer_somme_top3_proba(d):
    lignes = []
    for course_id, groupe in d.groupby("course_id", sort=False):
        scores = groupe["score_geneal"].values.astype(float)
        z = scores - scores.max()
        exp_z = np.exp(z)
        proba = exp_z / exp_z.sum()
        proba_tries = np.sort(proba)[::-1]
        somme_top3_proba = float(proba_tries[:3].sum())
        for idx in groupe.index:
            lignes.append((idx, somme_top3_proba))
    df_ind = pd.DataFrame(lignes, columns=["_idx", "somme_top3_proba"]).set_index("_idx")
    return d.join(df_ind)


def sous_split_chronologique(df, frac_fit):
    courses_ordre = df["course_id"].drop_duplicates().tolist()
    n = len(courses_ordre)
    n_fit = int(n * frac_fit)
    courses_fit = set(courses_ordre[:n_fit])
    courses_calib = set(courses_ordre[n_fit:])
    return courses_fit, courses_calib


def charger_handicap_valeur(course_ids):
    """Requete SQL LECTURE SEULE, ciblee sur les seules courses VERT
    (aucun impact sur TRAIN ni sur v3_lib.REQUETE, aucune ecriture)."""
    if not PSYCOPG2_DISPONIBLE:
        raise RuntimeError("psycopg2 non installe.")
    conn = psycopg2.connect(lib.DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT course_id, numero, handicap_valeur FROM resultats_partants WHERE course_id = ANY(%s)",
            (list(course_ids),),
        )
        rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["course_id", "numero", "handicap_valeur"])


def construire_par_course_candidat(df_vert_bloc, rang_candidat_series, suffixe):
    d = df_vert_bloc.copy()
    d[f"rang_candidat_{suffixe}"] = rang_candidat_series.reindex(d.index)
    d[f"rang_effectif_{suffixe}"] = d[f"rang_candidat_{suffixe}"].fillna(d["rang_geneal"])

    gagnants = d[d["est_gagnant"] == 1]
    rang_baseline = gagnants.groupby("course_id")["rang_geneal"].min()
    rang_candidat = gagnants.groupby("course_id")[f"rang_effectif_{suffixe}"].min()

    par_course = d.drop_duplicates("course_id").set_index("course_id").copy()
    par_course["rang_final_baseline"] = par_course.index.map(rang_baseline)
    par_course[f"rang_final_{suffixe}"] = par_course.index.map(rang_candidat)
    return par_course.reset_index()


def metriques(df_courses, colonne_rang, label):
    n = len(df_courses)
    if n == 0:
        return {"label": label, "n_courses": 0, "top1_pct": None, "top3_pct": None, "top5_pct": None}
    top1 = round(100 * float((df_courses[colonne_rang] == 1).mean()), 1)
    top3 = round(100 * float((df_courses[colonne_rang] <= 3).mean()), 1)
    top5 = round(100 * float((df_courses[colonne_rang] <= 5).mean()), 1)
    return {"label": label, "n_courses": n, "top1_pct": top1, "top3_pct": top3, "top5_pct": top5}


def entrainer_evaluer_candidat(df_vert_fit_sl, df_vert_calib_sl, df_vert_calib, features, variables_nouvelles, suffixe, nom_candidat):
    lib.log(f"\n{'='*100}")
    lib.log(f"=== CANDIDAT {nom_candidat} -- variables : {features} ===")
    lib.log(f"{'='*100}")

    X_fit = df_vert_fit_sl[features].astype(float).to_numpy()
    y_fit = df_vert_fit_sl["est_gagnant"].astype(int).to_numpy()
    X_calib = df_vert_calib_sl[features].astype(float).to_numpy()
    y_calib = df_vert_calib_sl["est_gagnant"].astype(int).to_numpy()
    lib.log(f"  Matrice candidate {nom_candidat} : {len(features)} variables -- VAL_FIT={X_fit.shape}, VAL_CALIB={X_calib.shape}.")

    params, _ = lib.entrainer_gbm_avec_grille(
        X_fit, y_fit, X_calib, y_calib, lib.GRILLE_GBM, f"handicap-valeur-{suffixe}")
    modele = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    modele.fit(X_fit, y_fit)

    proba_calib = modele.predict_proba(X_calib)[:, 1]
    auc_final = round(float(roc_auc_score(y_calib, proba_calib)), 4)
    logloss_final = round(float(log_loss(y_calib, np.clip(proba_calib, 1e-6, 1 - 1e-6))), 4)
    lib.log(f"\n  >>> AUC candidat {nom_candidat} sur VAL_CALIB : {auc_final}   log-loss : {logloss_final}")

    imp = permutation_importance(
        modele, X_calib, y_calib, scoring="roc_auc", n_repeats=20, random_state=lib.RANDOM_SEED)
    ordre = np.argsort(imp.importances_mean)[::-1]
    lib.log(f"\n  Importance des variables (permutation, AUC, VAL_CALIB, 20 repetitions) -- candidat {nom_candidat} :")
    for i in ordre:
        nom = features[i]
        marqueur = " <-- nouvelle variable" if nom in variables_nouvelles else " (baseline B+genealogie)"
        lib.log(f"    {nom:35s} delta_AUC_moyen={round(float(imp.importances_mean[i]), 4):+.4f} "
                f"(+/-{round(float(imp.importances_std[i]), 4)}){marqueur}")

    proba_series = pd.Series(proba_calib, index=df_vert_calib_sl.index)
    rang_candidat_calib = proba_series.groupby(df_vert_calib_sl["course_id"]).rank(method="min", ascending=False)
    par_course_calib = construire_par_course_candidat(df_vert_calib, rang_candidat_calib, suffixe)

    lib.log(f"\n  -- COMPARAISON classement B+genealogie BRUT vs CANDIDAT {nom_candidat} -- VAL_CALIB, courses VERT --")
    resultats = {}
    for nom_bench, colonne_filtre in [("REEL", None), ("PROPRE", "est_benchmark_propre")]:
        sous = par_course_calib if colonne_filtre is None else par_course_calib[par_course_calib[colonne_filtre]]
        m_baseline = metriques(sous, "rang_final_baseline", "REFERENCE")
        m_candidat = metriques(sous, f"rang_final_{suffixe}", "NOUVEAU")
        resultats[nom_bench] = (m_baseline, m_candidat)
        lib.log(f"\n    -- benchmark {nom_bench} -- n courses VERT (VAL_CALIB) = {m_baseline['n_courses']} --")
        if m_baseline["top1_pct"] is not None:
            lib.log(f"       REFERENCE (BRUT B+genealogie) : top1={m_baseline['top1_pct']}% top3={m_baseline['top3_pct']}% top5={m_baseline['top5_pct']}%")
            lib.log(f"       NOUVEAU (candidat {nom_candidat})    : top1={m_candidat['top1_pct']}% top3={m_candidat['top3_pct']}% top5={m_candidat['top5_pct']}%")
            delta_top1 = round(m_candidat["top1_pct"] - m_baseline["top1_pct"], 1)
            delta_top3 = round(m_candidat["top3_pct"] - m_baseline["top3_pct"], 1)
            delta_top5 = round(m_candidat["top5_pct"] - m_baseline["top5_pct"], 1)
            lib.log(f"       DELTA : top1={delta_top1:+}pt top3={delta_top3:+}pt top5={delta_top5:+}pt "
                    f"(top5 doit rester strictement identique par construction)")
        else:
            lib.log("       n/a (aucune course)")

    return {"auc": auc_final, "logloss": logloss_final, "resultats": resultats, "importance": imp, "features": features}


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn non installe.")
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError("lightgbm non installe.")
    if not PSYCOPG2_DISPONIBLE:
        raise RuntimeError("psycopg2 non installe.")

    lib.log("=" * 100)
    lib.log("PISTE 7 -- TEST HANDICAP_VALEUR, COURSES VERT UNIQUEMENT -- 01/09/2026")
    lib.log("=== AUCUN TEST A / TEST B charge ni utilise dans cette phase (decision interdite dessus). ===")
    lib.log("=== TRAIN et B+genealogie INCHANGES -- memes 240 variables, meme entrainement, memes seuils figes. ===")
    lib.log("=== Piste 'forme recente des adversaires' deja rejetee independamment -- ceci est un test separe. ===")
    lib.log("=" * 100)

    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    X_train_v3_geneal = checkpoint["X_train_v3_geneal"]
    X_val_v3_geneal = checkpoint["X_val_v3_geneal"]
    y_train_place = checkpoint["y_train_place"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    df_val = checkpoint["df_val"].reset_index(drop=True)
    course_id_train = checkpoint["course_id_train"]

    lib.log(f"\n  Checkpoint charge : TRAIN={X_train_v3_geneal.shape}, VAL={X_val_v3_geneal.shape} "
            f"(240 variables deja existantes, TRAIN inchange).")

    lib.log("\n[1/9] Entrainement B+genealogie (memes hyperparametres que tous les runs precedents)...")
    groups_train = groupes_consecutifs(course_id_train)
    y_train_graded = np.where(y_train_gagnant == 1, 2, np.where(y_train_place == 1, 1, 0)).astype(int)
    groups_val = groupes_consecutifs(df_val["course_id"])
    modele_geneal = entrainer_lambdarank(
        X_train_v3_geneal, y_train_graded, groups_train, X_val_v3_geneal, checkpoint["y_val_gagnant"],
        groups_val, "B+genealogie")
    df_val["score_geneal"] = modele_geneal.predict(X_val_v3_geneal)
    df_val["rang_geneal"] = df_val.groupby("course_id")["score_geneal"].rank(method="min", ascending=False)

    exclusions = lib.charger_exclusions_benchmark()
    df_val = lib.appliquer_benchmarks(df_val, exclusions)
    lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 7 test handicap_valeur")
    df_val_reel = df_val[df_val["est_benchmark_reel"]].copy()

    lib.log("\n[2/9] Identification des courses VERT (seuil fige, somme_top3_proba >= "
            f"{SEUIL_VERT_FIGE}, filtre INCHANGE)...")
    df_val_reel = calculer_somme_top3_proba(df_val_reel)
    par_course_ind = df_val_reel.drop_duplicates("course_id")[["course_id", "somme_top3_proba"]]
    n_val_total = len(par_course_ind)
    courses_vert = set(par_course_ind.loc[par_course_ind["somme_top3_proba"] >= SEUIL_VERT_FIGE, "course_id"])
    lib.log(f"  {len(courses_vert)}/{n_val_total} courses VALIDATION (benchmark reel) classees VERT "
            f"({round(100*len(courses_vert)/n_val_total,1)}%).")

    df_vert = df_val_reel[df_val_reel["course_id"].isin(courses_vert)].copy()
    df_vert["dans_shortlist"] = df_vert["rang_geneal"] <= K_SHORTLIST

    lib.log("\n[3/9] Chargement de handicap_valeur (requete SQL ciblee, lecture seule, "
            "uniquement pour les courses VERT -- aucun impact sur TRAIN ni sur v3_lib.REQUETE)...")
    df_hv = charger_handicap_valeur(courses_vert)
    lib.log(f"  {len(df_hv)} lignes chargees pour {df_hv['course_id'].nunique()} courses VERT.")
    df_vert = df_vert.merge(df_hv, on=["course_id", "numero"], how="left")
    n_dispo = int(df_vert["handicap_valeur"].notna().sum())
    lib.log(f"  Couverture handicap_valeur sur le champ entier des courses VERT : "
            f"{round(100*n_dispo/len(df_vert),1)}% ({n_dispo}/{len(df_vert)})")

    lib.log("\n[4/9] Construction handicap_valeur_rang_course / _z_course (mecanisme generique, champ entier)...")
    df_vert = lib.ajouter_variables_relatives(df_vert, ["handicap_valeur"])
    for col in ["handicap_valeur_rang_course", "handicap_valeur_z_course"]:
        n_c = int(df_vert[col].notna().sum())
        lib.log(f"    {col:35s} couverture={round(100*n_c/len(df_vert),1)}% ({n_c}/{len(df_vert)})")

    courses_fit, courses_calib = sous_split_chronologique(df_vert.drop_duplicates("course_id"), 0.80)
    df_vert_fit = df_vert[df_vert["course_id"].isin(courses_fit)]
    df_vert_calib = df_vert[df_vert["course_id"].isin(courses_calib)]
    lib.log(f"\n[5/9] Sous-decoupage VERT : VAL_FIT={len(courses_fit)} courses (entraine les modeles candidats), "
            f"VAL_CALIB={len(courses_calib)} courses (decide seul si le signal est reel). "
            f"MEME decoupage pour les deux candidats.")

    df_vert_fit_sl = df_vert_fit[df_vert_fit["dans_shortlist"]].copy()
    df_vert_calib_sl = df_vert_calib[df_vert_calib["dans_shortlist"]].copy()

    lib.log("\n[6/9] Candidat A -- handicap_valeur brute...")
    resultat_a = entrainer_evaluer_candidat(
        df_vert_fit_sl, df_vert_calib_sl, df_vert_calib,
        FEATURES_CANDIDAT_A, VARIABLES_NOUVELLES_A, "A", "A (handicap_valeur brute)")

    lib.log("\n[7/9] Candidat B -- handicap_valeur_rang_course + handicap_valeur_z_course...")
    resultat_b = entrainer_evaluer_candidat(
        df_vert_fit_sl, df_vert_calib_sl, df_vert_calib,
        FEATURES_CANDIDAT_B, VARIABLES_NOUVELLES_B, "B", "B (handicap_valeur rang/z-score)")

    lib.log("\n[8/9] " + "=" * 96)
    lib.log("=== BLOC RESUME -- format demande par Dorian, candidats A et B ===")
    lib.log("=" * 100)
    for nom_candidat, resultat in [("A (brute)", resultat_a), ("B (rang/z-score)", resultat_b)]:
        lib.log(f"\n  --- Candidat {nom_candidat} -- AUC={resultat['auc']} log-loss={resultat['logloss']} ---")
        for nom_bench in ("REEL", "PROPRE"):
            m_baseline, m_candidat = resultat["resultats"][nom_bench]
            lib.log(f"\n  [{nom_bench}] REFERENCE -> NOUVEAU (n={m_baseline['n_courses']})")
            if m_baseline["top1_pct"] is not None:
                lib.log(f"  Top-1 : {m_baseline['top1_pct']}% -> {m_candidat['top1_pct']}%")
                lib.log(f"  Top-3 : {m_baseline['top3_pct']}% -> {m_candidat['top3_pct']}%")
                lib.log(f"  Top-5 : {m_baseline['top5_pct']}% -> {m_candidat['top5_pct']}%")
                lib.log(f"  Delta : top1={round(m_candidat['top1_pct']-m_baseline['top1_pct'],1):+}pt "
                        f"top3={round(m_candidat['top3_pct']-m_baseline['top3_pct'],1):+}pt "
                        f"top5={round(m_candidat['top5_pct']-m_baseline['top5_pct'],1):+}pt")
            else:
                lib.log("  n/a (aucune course)")

    lib.log("\n" + "=" * 100)
    lib.log("=== [9/9] FIN -- rappel : reference deja validee sur VERT ~34-36% top1, ~73-74% top3, ~91-92% top5. ===")
    lib.log("=== Decision (a appliquer honnetement, sans cherry-picking) : gain Top-1 doit etre reellement ===")
    lib.log("=== attribuable a handicap_valeur (cf. importance par permutation), sans degradation ===")
    lib.log("=== significative de Top-3/Top-5. Si aucun signal robuste, rejeter definitivement la piste. ===")
    lib.log("=" * 100)

    lib.log("\n===CSV_METRIQUES_HANDICAP_VALEUR_START===")
    lignes_csv = ["candidat,benchmark,label,n_courses,top1_pct,top3_pct,top5_pct"]
    for nom_candidat, resultat in [("A", resultat_a), ("B", resultat_b)]:
        for nom_bench in ("REEL", "PROPRE"):
            m_baseline, m_candidat = resultat["resultats"][nom_bench]
            lignes_csv.append(f"{nom_candidat},{nom_bench},REFERENCE,{m_baseline['n_courses']},{m_baseline['top1_pct']},{m_baseline['top3_pct']},{m_baseline['top5_pct']}")
            lignes_csv.append(f"{nom_candidat},{nom_bench},NOUVEAU,{m_candidat['n_courses']},{m_candidat['top1_pct']},{m_candidat['top3_pct']},{m_candidat['top5_pct']}")
    for ligne in lignes_csv:
        lib.log(ligne)
    lib.log("===CSV_METRIQUES_HANDICAP_VALEUR_END===")

    lib.log(f"\n===AUC_CANDIDAT_A=== {resultat_a['auc']}")
    lib.log(f"===AUC_CANDIDAT_B=== {resultat_b['auc']}")


if __name__ == "__main__":
    main()
