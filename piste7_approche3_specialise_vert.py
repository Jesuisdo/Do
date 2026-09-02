# -*- coding: utf-8 -*-
"""
piste7_approche3_specialise_vert.py -- approche 3 de la recherche
methodologique Top-1 (feu vert donne par Dorian le 02/09/2026, apres rejet
definitif de l'approche 1 -- marge #1/#2 non concentree, non monotone).

Architecture STRICTE demandee :
  B+genealogie ACTUEL ET FIGE
  -> filtre VERT ACTUEL ET FIGE (seuil 0.5848, INCHANGE)
  -> modele specialise VERT (NOUVEAU, SEPARE)
  -> classement Top-5 interne (le Top-5 lui-meme n'est JAMAIS modifie ;
     seul l'ordre a l'interieur des 5 chevaux deja retenus peut changer).

Changement de protocole APPROUVE par Dorian (jusqu'ici jamais fait dans ce
projet) : B+genealogie (deja entraine, FIGE, memes hyperparametres) est
applique en PREDICTION SEULE sur TRAIN pour identifier les courses VERT de
TRAIN (~18600 courses attendues). Ceci n'entraine JAMAIS B+genealogie sur
autre chose que TRAIN (aucun changement a son propre entrainement) et ne
modifie ni le filtre VERT ni ses seuils. Le modele specialise, lui, est
entraine UNIQUEMENT sur les 5 chevaux du shortlist VERT-TRAIN (~93000
lignes), avec les 240 variables deja existantes (v3 + genealogie),
AUCUNE nouvelle variable, AUCUNE cote.

Protocole d'evaluation :
  - Entrainement du modele specialise : shortlist VERT-TRAIN (~18600
    courses, 5 chevaux/course).
  - Selection des hyperparametres (regularisation) : shortlist VERT de
    VAL_FIT (sous-decoupage chronologique 80/20 de VALIDATION VERT,
    IDENTIQUE au decoupage utilise dans tous les runs precedents de la
    piste 7) -- reutilise lib.GRILLE_GBM et lib.entrainer_gbm_avec_grille,
    memes 3 configurations que pour tout candidat GBM de ce projet.
  - Decision finale, SEULE population jamais retouchee : shortlist VERT de
    VAL_CALIB (les 20% les plus recents de VALIDATION VERT).
  - Benchmark REEL et benchmark PROPRE, tous deux rapportes.
  - AUCUN TEST A / TEST B charge ni utilise.

Le modele specialise ne sert QU'A reordonner les 5 chevaux VERT deja
retenus par B+genealogie -- il ne peut jamais faire gagner une course dont
le vrai gagnant n'etait pas dans le Top-5 d'origine (dans ce cas, son rang
effectif retombe sur le rang B+genealogie d'origine, exactement comme dans
piste7_etape2b_handicap_valeur_vert.py).

Critere de reussite fixe par Dorian AVANT le run (aucune optimisation a
posteriori) : gain Top-1 clair et suffisamment important, stable entre
REEL et PROPRE, sans degradation significative de Top-3/Top-5. Si le
resultat est negatif ou marginal, ce script ne cherche PAS de variante :
il rapporte le constat tel quel.
"""
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

try:
    import lightgbm as lgb
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    LIGHTGBM_DISPONIBLE = False

CHECKPOINT_PATH = "checkpoint_piste6_split4.pkl"
K_SHORTLIST = 5

# Seuil FIGE de la piste 7 -- NE PAS RETOUCHER.
SEUIL_VERT_FIGE = 0.5848


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
        proba_triees = np.sort(proba)[::-1]
        somme_top3_proba = float(proba_triees[:3].sum())
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


def construire_par_course_candidat(df_vert_bloc, rang_candidat_series, suffixe):
    """Identique au mecanisme de piste7_etape2b : la ou le modele
    specialise n'a pas de rang calcule (gagnant hors shortlist), le rang
    effectif retombe sur rang_geneal -- le modele specialise ne peut donc
    jamais degrader une course qu'il ne touche pas."""
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


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn non installe.")
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError("lightgbm non installe.")

    lib.log("=" * 100)
    lib.log("PISTE 7 -- APPROCHE 3 -- MODELE SPECIALISE VERT (VERT-TRAIN) -- 02/09/2026")
    lib.log("=== AUCUN TEST A / TEST B charge ni utilise. AUCUNE COTE. AUCUNE NOUVELLE VARIABLE. ===")
    lib.log("=== B+genealogie et le filtre VERT (seuil 0.5848) sont INCHANGES -- meme entrainement, ===")
    lib.log("=== memes hyperparametres, meme seuil fige. Le modele specialise est un modele SEPARE ===")
    lib.log("=== qui ne fait que reordonner les 5 chevaux deja retenus -- le Top-5 n'est jamais modifie. ===")
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

    lib.log("\n[1/10] Entrainement B+genealogie (memes hyperparametres que tous les runs precedents, FIGE)...")
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
    lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 7 approche 3 (specialise VERT)")
    df_val_reel = df_val[df_val["est_benchmark_reel"]].copy()

    lib.log("\n[2/10] Identification des courses VERT de VALIDATION (seuil fige, somme_top3_proba >= "
            f"{SEUIL_VERT_FIGE}, filtre INCHANGE)...")
    df_val_reel = calculer_somme_top3_proba(df_val_reel)
    par_course_val_ind = df_val_reel.drop_duplicates("course_id")[["course_id", "somme_top3_proba"]]
    n_val_total = len(par_course_val_ind)
    courses_vert_val = set(par_course_val_ind.loc[par_course_val_ind["somme_top3_proba"] >= SEUIL_VERT_FIGE, "course_id"])
    lib.log(f"  {len(courses_vert_val)}/{n_val_total} courses VALIDATION (benchmark reel) classees VERT "
            f"({round(100*len(courses_vert_val)/n_val_total,1)}%).")

    df_vert_val = df_val_reel[df_val_reel["course_id"].isin(courses_vert_val)].copy()
    df_vert_val["dans_shortlist"] = df_vert_val["rang_geneal"] <= K_SHORTLIST

    lib.log("\n[3/10] === CHANGEMENT DE PROTOCOLE APPROUVE : identification des courses VERT de TRAIN "
            "(B+genealogie applique en PREDICTION SEULE sur TRAIN, aucun reentrainement) ===")
    score_geneal_train = modele_geneal.predict(X_train_v3_geneal)
    train_scores = pd.DataFrame(
        {"course_id": course_id_train.values, "score_geneal": score_geneal_train},
        index=X_train_v3_geneal.index,
    )
    train_scores["rang_geneal"] = train_scores.groupby("course_id")["score_geneal"].rank(method="min", ascending=False)
    train_scores = calculer_somme_top3_proba(train_scores)
    par_course_train_ind = train_scores.drop_duplicates("course_id")[["course_id", "somme_top3_proba"]]
    n_train_total = len(par_course_train_ind)
    courses_vert_train = set(par_course_train_ind.loc[par_course_train_ind["somme_top3_proba"] >= SEUIL_VERT_FIGE, "course_id"])
    lib.log(f"  {len(courses_vert_train)}/{n_train_total} courses TRAIN classees VERT "
            f"({round(100*len(courses_vert_train)/n_train_total,1)}%) -- population d'entrainement du modele specialise.")

    lib.log("\n[4/10] Construction du shortlist VERT-TRAIN (5 chevaux/course, sert UNIQUEMENT a "
            "l'entrainement du modele specialise -- B+genealogie et le filtre VERT restent inchanges)...")
    shortlist_train_mask = train_scores["course_id"].isin(courses_vert_train) & (train_scores["rang_geneal"] <= K_SHORTLIST)
    shortlist_train = train_scores[shortlist_train_mask]
    lib.log(f"  Shortlist VERT-TRAIN : {len(shortlist_train)} lignes pour "
            f"{shortlist_train['course_id'].nunique()} courses VERT-TRAIN.")

    X_train_spec = X_train_v3_geneal.loc[shortlist_train.index]
    y_train_spec = np.asarray(y_train_gagnant)[shortlist_train.index.values]
    lib.log(f"  Matrice d'entrainement du modele specialise : {X_train_spec.shape} "
            f"(240 variables existantes, objectif = est_gagnant binaire).")

    lib.log("\n[5/10] Sous-decoupage VERT de VALIDATION : VAL_FIT (selection des hyperparametres), "
            "VAL_CALIB (seule population jamais retouchee, decision finale)...")
    courses_fit_val, courses_calib_val = sous_split_chronologique(df_vert_val.drop_duplicates("course_id"), 0.80)
    df_vert_val_fit = df_vert_val[df_vert_val["course_id"].isin(courses_fit_val)]
    df_vert_val_calib = df_vert_val[df_vert_val["course_id"].isin(courses_calib_val)]
    lib.log(f"  VAL_FIT={len(courses_fit_val)} courses VERT, VAL_CALIB={len(courses_calib_val)} courses VERT.")

    df_vert_val_fit_sl = df_vert_val_fit[df_vert_val_fit["dans_shortlist"]].copy()
    df_vert_val_calib_sl = df_vert_val_calib[df_vert_val_calib["dans_shortlist"]].copy()

    X_valfit_spec = X_val_v3_geneal.loc[df_vert_val_fit_sl.index]
    y_valfit_spec = df_vert_val_fit_sl["est_gagnant"].astype(int).values
    X_calib_spec = X_val_v3_geneal.loc[df_vert_val_calib_sl.index]
    y_calib_spec = df_vert_val_calib_sl["est_gagnant"].astype(int).values
    lib.log(f"  Matrice VAL_FIT (selection hyperparametres) : {X_valfit_spec.shape}. "
            f"Matrice VAL_CALIB (decision finale) : {X_calib_spec.shape}.")

    lib.log("\n[6/10] Selection des hyperparametres (regularisation) du modele specialise -- "
            "entraine sur VERT-TRAIN, evalue sur VAL_FIT VERT (meme grille GBM que tout le projet)...")
    params, auc_val_fit = lib.entrainer_gbm_avec_grille(
        X_train_spec, y_train_spec, X_valfit_spec, y_valfit_spec, lib.GRILLE_GBM, "specialise-vert")
    lib.log(f"  >>> Hyperparametres retenus : {params} (AUC VAL_FIT={round(auc_val_fit,4)})")

    lib.log("\n[7/10] Entrainement final du modele specialise sur VERT-TRAIN (parametres figes ci-dessus)...")
    modele_spec = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    modele_spec.fit(X_train_spec, y_train_spec)

    proba_calib = modele_spec.predict_proba(X_calib_spec)[:, 1]
    auc_final = round(float(roc_auc_score(y_calib_spec, proba_calib)), 4)
    logloss_final = round(float(log_loss(y_calib_spec, np.clip(proba_calib, 1e-6, 1 - 1e-6))), 4)
    auc_score_geneal_seul = round(float(roc_auc_score(y_calib_spec, df_vert_val_calib_sl["score_geneal"].values)), 4)
    lib.log(f"\n  >>> AUC modele specialise sur VAL_CALIB : {auc_final}   log-loss : {logloss_final}")
    lib.log(f"  >>> AUC de score_geneal seul (B+genealogie brut) sur la MEME population VAL_CALIB : "
            f"{auc_score_geneal_seul}  (verification d'attribution : le modele specialise doit depasser "
            f"cette reference pour que son gain lui soit reellement imputable)")

    lib.log("\n[8/10] Importance des variables (permutation, AUC, VAL_CALIB, 20 repetitions, top 20 sur "
            f"{X_calib_spec.shape[1]} variables)...")
    imp = permutation_importance(
        modele_spec, X_calib_spec, y_calib_spec, scoring="roc_auc", n_repeats=20, random_state=lib.RANDOM_SEED)
    ordre = np.argsort(imp.importances_mean)[::-1]
    colonnes = list(X_calib_spec.columns)
    for rang_i, i in enumerate(ordre[:20], start=1):
        nom = colonnes[i]
        lib.log(f"    {rang_i:2d}. {nom:40s} delta_AUC_moyen={round(float(imp.importances_mean[i]), 4):+.4f} "
                f"(+/-{round(float(imp.importances_std[i]), 4)})")

    proba_series = pd.Series(proba_calib, index=df_vert_val_calib_sl.index)
    rang_spec_calib = proba_series.groupby(df_vert_val_calib_sl["course_id"]).rank(method="min", ascending=False)
    par_course_calib = construire_par_course_candidat(df_vert_val_calib, rang_spec_calib, "spec")

    lib.log("\n[9/10] " + "=" * 96)
    lib.log("=== COMPARAISON REFERENCE (B+genealogie brut) vs MODELE SPECIALISE VERT -- VAL_CALIB ===")
    lib.log("=" * 100)
    resultats = {}
    for nom_bench, colonne_filtre in [("REEL", None), ("PROPRE", "est_benchmark_propre")]:
        sous = par_course_calib if colonne_filtre is None else par_course_calib[par_course_calib[colonne_filtre]]
        m_baseline = metriques(sous, "rang_final_baseline", "REFERENCE")
        m_candidat = metriques(sous, "rang_final_spec", "SPECIALISE VERT")
        resultats[nom_bench] = (m_baseline, m_candidat)
        lib.log(f"\n  -- benchmark {nom_bench} -- n courses VERT (VAL_CALIB) = {m_baseline['n_courses']} --")
        if m_baseline["top1_pct"] is not None:
            lib.log(f"     REFERENCE (B+genealogie brut)  : top1={m_baseline['top1_pct']}% top3={m_baseline['top3_pct']}% top5={m_baseline['top5_pct']}%")
            lib.log(f"     SPECIALISE VERT                : top1={m_candidat['top1_pct']}% top3={m_candidat['top3_pct']}% top5={m_candidat['top5_pct']}%")
            delta_top1 = round(m_candidat["top1_pct"] - m_baseline["top1_pct"], 1)
            delta_top3 = round(m_candidat["top3_pct"] - m_baseline["top3_pct"], 1)
            delta_top5 = round(m_candidat["top5_pct"] - m_baseline["top5_pct"], 1)
            lib.log(f"     DELTA : top1={delta_top1:+}pt top3={delta_top3:+}pt top5={delta_top5:+}pt")
        else:
            lib.log("     n/a (aucune course)")

    m_baseline_reel, m_candidat_reel = resultats["REEL"]
    m_baseline_propre, m_candidat_propre = resultats["PROPRE"]
    delta_top1_reel = round(m_candidat_reel["top1_pct"] - m_baseline_reel["top1_pct"], 1) if m_baseline_reel["top1_pct"] is not None else None
    delta_top1_propre = round(m_candidat_propre["top1_pct"] - m_baseline_propre["top1_pct"], 1) if m_baseline_propre["top1_pct"] is not None else None
    delta_top3_reel = round(m_candidat_reel["top3_pct"] - m_baseline_reel["top3_pct"], 1) if m_baseline_reel["top1_pct"] is not None else None
    delta_top3_propre = round(m_candidat_propre["top3_pct"] - m_baseline_propre["top3_pct"], 1) if m_baseline_propre["top1_pct"] is not None else None
    delta_top5_reel = round(m_candidat_reel["top5_pct"] - m_baseline_reel["top5_pct"], 1) if m_baseline_reel["top1_pct"] is not None else None
    delta_top5_propre = round(m_candidat_propre["top5_pct"] - m_baseline_propre["top5_pct"], 1) if m_baseline_propre["top1_pct"] is not None else None

    lib.log("\n[10/10] " + "=" * 92)
    lib.log("=== CONCLUSION -- approche 3, modele specialise VERT -- constat honnete, aucune optimisation "
            "a posteriori ===")
    lib.log("=" * 100)
    lib.log(f"\n  Gain Top-1 REEL   : {delta_top1_reel:+}pt   Gain Top-1 PROPRE : {delta_top1_propre:+}pt")
    lib.log(f"  Gain Top-3 REEL   : {delta_top3_reel:+}pt   Gain Top-3 PROPRE : {delta_top3_propre:+}pt")
    lib.log(f"  Gain Top-5 REEL   : {delta_top5_reel:+}pt   Gain Top-5 PROPRE : {delta_top5_propre:+}pt")
    lib.log(f"  AUC modele specialise (VAL_CALIB) : {auc_final}   AUC score_geneal seul (meme population) : {auc_score_geneal_seul}")
    lib.log("\n  Lecture (a interpreter honnetement par Dorian, aucun seuil de succes impose ici -- "
            "le critere de reussite a ete fixe par Dorian AVANT ce run) :")
    lib.log("  - Gain Top-1 attendu : clair, stable entre REEL et PROPRE, sans degradation Top-3/Top-5.")
    lib.log("  - Si le gain est negatif, marginal, ou instable entre REEL/PROPRE, ou si Top-3/Top-5 se "
            "degrade significativement : rejeter le candidat, ne pas chercher de variante.")

    lib.log("\n===CSV_METRIQUES_APPROCHE3_START===")
    lignes_csv = ["benchmark,label,n_courses,top1_pct,top3_pct,top5_pct"]
    for nom_bench in ("REEL", "PROPRE"):
        m_baseline, m_candidat = resultats[nom_bench]
        lignes_csv.append(f"{nom_bench},REFERENCE,{m_baseline['n_courses']},{m_baseline['top1_pct']},{m_baseline['top3_pct']},{m_baseline['top5_pct']}")
        lignes_csv.append(f"{nom_bench},SPECIALISE_VERT,{m_candidat['n_courses']},{m_candidat['top1_pct']},{m_candidat['top3_pct']},{m_candidat['top5_pct']}")
    for ligne in lignes_csv:
        lib.log(ligne)
    lib.log("===CSV_METRIQUES_APPROCHE3_END===")

    lib.log(f"\n===AUC_SPECIALISE_VERT=== {auc_final}")
    lib.log(f"===AUC_SCORE_GENEAL_SEUL=== {auc_score_geneal_seul}")
    lib.log(f"===HYPERPARAMS_RETENUS=== {params}")
    lib.log(f"===N_COURSES_VERT_TRAIN=== {len(courses_vert_train)}")
    lib.log(f"===N_COURSES_VERT_TRAIN_TOTAL=== {n_train_total}")
    lib.log(f"===DELTA_TOP1_REEL=== {delta_top1_reel}")
    lib.log(f"===DELTA_TOP1_PROPRE=== {delta_top1_propre}")
    lib.log(f"===DELTA_TOP3_REEL=== {delta_top3_reel}")
    lib.log(f"===DELTA_TOP3_PROPRE=== {delta_top3_propre}")
    lib.log(f"===DELTA_TOP5_REEL=== {delta_top5_reel}")
    lib.log(f"===DELTA_TOP5_PROPRE=== {delta_top5_propre}")


if __name__ == "__main__":
    main()
