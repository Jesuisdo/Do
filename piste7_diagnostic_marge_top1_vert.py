# -*- coding: utf-8 -*-
"""
piste7_diagnostic_marge_top1_vert.py -- DIAGNOSTIC PUR (approche 1, etape 1/2),
UNIQUEMENT sur les courses VERT de VAL_CALIB. AUCUN MODELE, AUCUN
ENTRAINEMENT DE CANDIDAT, AUCUNE NOUVELLE VARIABLE AJOUTEE AU PIPELINE,
AUCUNE CORRECTION, AUCUNE COTE, AUCUN TEST A/B.

Objectif : verifier l'hypothese "les erreurs Top-1 VERT sont-elles
concentrees dans les courses ou le #1 et le #2 ont des scores tres
proches ?"

Point architectural INCHANGE : B+genealogie (240 variables, memes
hyperparametres, meme entrainement sur TRAIN) et le filtre VERT/ORANGE/ROUGE
(seuil fige 0.5848) sont recalcules a l'identique -- c'est la reference
figee necessaire pour identifier les courses VERT et leur classement actuel,
pas un nouveau modele. Aucun classifieur candidat n'est entraine dans ce
script (contrairement a piste7_etape2 et piste7_etape2b).

Methode objective pour les seuils de marge (aucune optimisation a posteriori) :
les quartiles (et deciles pour le controle de monotonie) de l'ecart de
probabilite (softmax) entre le #1 et le #2 sont calcules UNIQUEMENT sur
VAL_FIT (80% des courses VERT, chronologiquement les plus anciennes), puis
appliques tels quels, sans aucune modification, a VAL_CALIB (20% restants)
pour l'analyse. Les seuils ne sont jamais recalcules ou ajustes apres avoir
vu les resultats sur VAL_CALIB.
"""
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    from scipy.stats import spearmanr
    SCIPY_DISPONIBLE = True
except ImportError:
    SCIPY_DISPONIBLE = False

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


def calculer_marge_top1_top2(d):
    """Ecart de probabilite (softmax sur score_geneal, champ entier de la
    course) entre le #1 et le #2 du classement -- indicateur diagnostique
    pur, jamais ajoute a aucune matrice de modele."""
    lignes = []
    for course_id, groupe in d.groupby("course_id", sort=False):
        scores = groupe["score_geneal"].values.astype(float)
        z = scores - scores.max()
        exp_z = np.exp(z)
        proba = exp_z / exp_z.sum()
        proba_triees = np.sort(proba)[::-1]
        if len(proba_triees) >= 2:
            marge = float(proba_triees[0] - proba_triees[1])
        else:
            marge = np.nan
        for idx in groupe.index:
            lignes.append((idx, marge))
    df_ind = pd.DataFrame(lignes, columns=["_idx", "marge_top1_top2"]).set_index("_idx")
    return d.join(df_ind)


def sous_split_chronologique(df, frac_fit):
    courses_ordre = df["course_id"].drop_duplicates().tolist()
    n = len(courses_ordre)
    n_fit = int(n * frac_fit)
    courses_fit = set(courses_ordre[:n_fit])
    courses_calib = set(courses_ordre[n_fit:])
    return courses_fit, courses_calib


def construire_table_par_course(df_vert_bloc):
    """Une ligne par course VERT : rang du gagnant selon le classement
    ACTUEL (rang_geneal, non modifie), marge #1/#2, indicateur benchmark
    propre."""
    d = df_vert_bloc.copy()
    gagnants = d[d["est_gagnant"] == 1]
    rang_gagnant = gagnants.groupby("course_id")["rang_geneal"].min()

    par_course = d.drop_duplicates("course_id").set_index("course_id").copy()
    par_course["rang_gagnant"] = par_course.index.map(rang_gagnant)
    return par_course.reset_index()


def zone_par_seuils(marge, q1, q2, q3):
    if pd.isna(marge):
        return None
    if marge <= q1:
        return "1_tres_faible"
    if marge <= q2:
        return "2_faible"
    if marge <= q3:
        return "3_moyenne"
    return "4_forte"


def zone_par_deciles(marge, seuils_deciles):
    if pd.isna(marge):
        return None
    for i, s in enumerate(seuils_deciles):
        if marge <= s:
            return i + 1
    return len(seuils_deciles) + 1


def metriques_zone(sous_df):
    n = len(sous_df)
    if n == 0:
        return {"n_courses": 0, "top1_pct": None, "top3_pct": None, "top5_pct": None, "n_erreurs_top1": 0}
    top1 = round(100 * float((sous_df["rang_gagnant"] == 1).mean()), 1)
    top3 = round(100 * float((sous_df["rang_gagnant"] <= 3).mean()), 1)
    top5 = round(100 * float((sous_df["rang_gagnant"] <= 5).mean()), 1)
    n_erreurs = int((sous_df["rang_gagnant"] != 1).sum())
    return {"n_courses": n, "top1_pct": top1, "top3_pct": top3, "top5_pct": top5, "n_erreurs_top1": n_erreurs}


def analyser_benchmark(par_course_calib, nom_benchmark, colonne_filtre, q1, q2, q3, seuils_deciles):
    lib.log(f"\n{'=' * 100}")
    lib.log(f"=== DIAGNOSTIC MARGE #1/#2 -- benchmark {nom_benchmark} -- VERT VAL_CALIB ===")
    lib.log(f"{'=' * 100}")

    sous = par_course_calib if colonne_filtre is None else par_course_calib[par_course_calib[colonne_filtre]]
    n_total = len(sous)
    n_erreurs_total = int((sous["rang_gagnant"] != 1).sum())
    lib.log(f"\nPopulation : {n_total} courses VERT (VAL_CALIB, benchmark {nom_benchmark}).")
    lib.log(f"Erreurs Top-1 totales (le #1 n'a pas gagne) : {n_erreurs_total}/{n_total} "
            f"({round(100*n_erreurs_total/n_total,1) if n_total else 0}%).")

    lib.log(f"\n-- [1] Distribution de la marge #1/#2 (probabilite softmax) sur {nom_benchmark} --")
    desc = sous["marge_top1_top2"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    for k in ["count", "mean", "std", "min", "10%", "25%", "50%", "75%", "90%", "max"]:
        if k in desc.index:
            lib.log(f"    {k:8s} = {round(float(desc[k]), 4)}")

    lib.log(f"\n-- Seuils de zone (quartiles FIGES, calcules sur VAL_FIT uniquement, jamais retouches) --")
    lib.log(f"    Q1={round(q1,4)}  Q2(mediane)={round(q2,4)}  Q3={round(q3,4)}")

    sous = sous.copy()
    sous["zone"] = sous["marge_top1_top2"].apply(lambda m: zone_par_seuils(m, q1, q2, q3))

    lib.log(f"\n-- [2] et [3] Top-1/Top-3/Top-5 par zone de marge -- {nom_benchmark} --")
    lib.log(f"{'zone':16s} {'n_courses':>10s} {'top1_pct':>9s} {'top3_pct':>9s} {'top5_pct':>9s} "
            f"{'n_erreurs':>10s} {'pct_des_erreurs':>16s}")
    lignes_csv = []
    resultat_zones = {}
    for zone in ["1_tres_faible", "2_faible", "3_moyenne", "4_forte"]:
        m = metriques_zone(sous[sous["zone"] == zone])
        pct_erreurs = round(100 * m["n_erreurs_top1"] / n_erreurs_total, 1) if n_erreurs_total else None
        resultat_zones[zone] = {**m, "pct_erreurs": pct_erreurs}
        lib.log(f"{zone:16s} {m['n_courses']:>10d} {str(m['top1_pct'])+'%':>9s} {str(m['top3_pct'])+'%':>9s} "
                f"{str(m['top5_pct'])+'%':>9s} {m['n_erreurs_top1']:>10d} {str(pct_erreurs)+'%':>16s}")
        lignes_csv.append((nom_benchmark, zone, m["n_courses"], m["top1_pct"], m["top3_pct"], m["top5_pct"],
                            m["n_erreurs_top1"], pct_erreurs))

    lib.log(f"\n-- [4] Concentration des erreurs Top-1 -- {nom_benchmark} --")
    lib.log(f"    Zone 'tres faible' + 'faible' regroupent "
            f"{round((resultat_zones['1_tres_faible']['n_erreurs_top1']+resultat_zones['2_faible']['n_erreurs_top1'])*100/n_erreurs_total,1) if n_erreurs_total else 0}% "
            f"des erreurs Top-1 (pour {round((resultat_zones['1_tres_faible']['n_courses']+resultat_zones['2_faible']['n_courses'])*100/n_total,1) if n_total else 0}% des courses).")

    lib.log(f"\n-- [5] Controle de monotonie (deciles FIGES, calcules sur VAL_FIT) -- {nom_benchmark} --")
    sous["decile"] = sous["marge_top1_top2"].apply(lambda m: zone_par_deciles(m, seuils_deciles))
    top1_par_decile = []
    lib.log(f"{'decile':8s} {'n_courses':>10s} {'top1_pct':>9s}")
    for d_idx in range(1, 11):
        sd = sous[sous["decile"] == d_idx]
        if len(sd) == 0:
            continue
        t1 = round(100 * float((sd["rang_gagnant"] == 1).mean()), 1)
        top1_par_decile.append((d_idx, len(sd), t1))
        lib.log(f"{d_idx:8d} {len(sd):>10d} {str(t1)+'%':>9s}")

    n_inversions = sum(1 for i in range(1, len(top1_par_decile))
                        if top1_par_decile[i][2] < top1_par_decile[i - 1][2])
    if SCIPY_DISPONIBLE and len(sous) > 2:
        rho, pval = spearmanr(sous["marge_top1_top2"], (sous["rang_gagnant"] == 1).astype(int))
        lib.log(f"\n    Correlation de Spearman (marge continue vs succes Top-1) : rho={round(float(rho),4)} "
                f"(p={round(float(pval),4)})")
    else:
        rho, pval = None, None
        lib.log("\n    scipy indisponible -- correlation de Spearman non calculee.")
    lib.log(f"    Nombre d'inversions dans la sequence des 10 deciles (Top-1% qui redescend "
            f"vs le decile precedent) : {n_inversions}/9 transitions "
            f"({'monotone strict' if n_inversions == 0 else 'non strictement monotone'}).")

    return {
        "n_total": n_total, "n_erreurs_total": n_erreurs_total,
        "zones": resultat_zones, "spearman_rho": rho, "spearman_p": pval,
        "n_inversions_deciles": n_inversions, "lignes_csv": lignes_csv,
    }


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn non installe.")
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError("lightgbm non installe.")

    lib.log("=" * 100)
    lib.log("PISTE 7 -- DIAGNOSTIC MARGE #1/#2, COURSES VERT DE VAL_CALIB UNIQUEMENT -- 01/09/2026")
    lib.log("=== AUCUN MODELE CANDIDAT ENTRAINE, AUCUNE NOUVELLE VARIABLE, AUCUNE CORRECTION. ===")
    lib.log("=== AUCUNE COTE. AUCUN TEST A/B charge ni utilise. ===")
    lib.log("=== TRAIN et B+genealogie INCHANGES -- memes 240 variables, meme entrainement, memes seuils figes. ===")
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

    lib.log("\n[1/6] Entrainement B+genealogie (memes hyperparametres que tous les runs precedents)...")
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
    lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 7 diagnostic marge #1/#2")
    df_val_reel = df_val[df_val["est_benchmark_reel"]].copy()

    lib.log("\n[2/6] Identification des courses VERT (seuil fige, somme_top3_proba >= "
            f"{SEUIL_VERT_FIGE}, filtre INCHANGE)...")
    df_val_reel = calculer_somme_top3_proba(df_val_reel)
    par_course_ind = df_val_reel.drop_duplicates("course_id")[["course_id", "somme_top3_proba"]]
    n_val_total = len(par_course_ind)
    courses_vert = set(par_course_ind.loc[par_course_ind["somme_top3_proba"] >= SEUIL_VERT_FIGE, "course_id"])
    lib.log(f"  {len(courses_vert)}/{n_val_total} courses VALIDATION (benchmark reel) classees VERT "
            f"({round(100*len(courses_vert)/n_val_total,1)}%).")

    df_vert = df_val_reel[df_val_reel["course_id"].isin(courses_vert)].copy()
    df_vert = calculer_marge_top1_top2(df_vert)

    lib.log("\n[3/6] Sous-decoupage VERT : VAL_FIT (calcule les seuils, fige), "
            "VAL_CALIB (seul a etre analyse)...")
    courses_fit, courses_calib = sous_split_chronologique(df_vert.drop_duplicates("course_id"), 0.80)
    df_vert_fit = df_vert[df_vert["course_id"].isin(courses_fit)]
    df_vert_calib = df_vert[df_vert["course_id"].isin(courses_calib)]
    lib.log(f"  VAL_FIT={len(courses_fit)} courses (calcule les quartiles/deciles, FIGES ensuite), "
            f"VAL_CALIB={len(courses_calib)} courses (seule population analysee).")

    lib.log("\n[4/6] Calcul des seuils objectifs (quartiles + deciles) sur VAL_FIT UNIQUEMENT...")
    marges_fit = df_vert_fit.drop_duplicates("course_id")["marge_top1_top2"].dropna()
    q1 = float(marges_fit.quantile(0.25))
    q2 = float(marges_fit.quantile(0.50))
    q3 = float(marges_fit.quantile(0.75))
    seuils_deciles = [float(marges_fit.quantile(p / 10)) for p in range(1, 10)]
    lib.log(f"  Quartiles (VAL_FIT, n={len(marges_fit)}) : Q1={round(q1,4)} Q2={round(q2,4)} Q3={round(q3,4)}")
    lib.log(f"  Deciles (VAL_FIT) : {[round(s,4) for s in seuils_deciles]}")
    lib.log("  Ces seuils sont maintenant FIGES et appliques tels quels a VAL_CALIB ci-dessous "
            "(aucun ajustement possible apres ce point).")

    par_course_calib = construire_table_par_course(df_vert_calib)
    par_course_calib = par_course_calib.merge(
        df_vert_calib.drop_duplicates("course_id")[["course_id", "marge_top1_top2", "est_benchmark_propre"]],
        on="course_id", how="left")

    lib.log("\n[5/6] Analyse -- benchmark REEL puis benchmark PROPRE...")
    resultat_reel = analyser_benchmark(par_course_calib, "REEL", None, q1, q2, q3, seuils_deciles)
    resultat_propre = analyser_benchmark(par_course_calib, "PROPRE", "est_benchmark_propre", q1, q2, q3, seuils_deciles)

    lib.log("\n[6/6] " + "=" * 96)
    lib.log("=== CONCLUSION -- diagnostic marge #1/#2, VERT VAL_CALIB ===")
    lib.log("=" * 100)

    part_faibles_reel = round(
        (resultat_reel["zones"]["1_tres_faible"]["n_erreurs_top1"] + resultat_reel["zones"]["2_faible"]["n_erreurs_top1"])
        * 100 / resultat_reel["n_erreurs_total"], 1) if resultat_reel["n_erreurs_total"] else 0
    part_faibles_propre = round(
        (resultat_propre["zones"]["1_tres_faible"]["n_erreurs_top1"] + resultat_propre["zones"]["2_faible"]["n_erreurs_top1"])
        * 100 / resultat_propre["n_erreurs_total"], 1) if resultat_propre["n_erreurs_total"] else 0

    top1_tres_faible_reel = resultat_reel["zones"]["1_tres_faible"]["top1_pct"]
    top1_forte_reel = resultat_reel["zones"]["4_forte"]["top1_pct"]

    lib.log(f"\n  REEL   : {part_faibles_reel}% des erreurs Top-1 dans les zones tres faible+faible. "
            f"Top-1 zone tres faible={top1_tres_faible_reel}% vs zone forte={top1_forte_reel}%. "
            f"Inversions deciles={resultat_reel['n_inversions_deciles']}/9. "
            f"Spearman rho={resultat_reel['spearman_rho']}.")
    lib.log(f"  PROPRE : {part_faibles_propre}% des erreurs Top-1 dans les zones tres faible+faible. "
            f"Top-1 zone tres faible={resultat_propre['zones']['1_tres_faible']['top1_pct']}% vs "
            f"zone forte={resultat_propre['zones']['4_forte']['top1_pct']}%. "
            f"Inversions deciles={resultat_propre['n_inversions_deciles']}/9. "
            f"Spearman rho={resultat_propre['spearman_rho']}.")

    lib.log("\n  Lecture (a interpreter honnetement par Dorian, aucun seuil de decision impose ici) :")
    lib.log("  - Si les erreurs sont tres majoritairement concentrees dans les zones faibles ET la relation")
    lib.log("    est globalement monotone (peu d'inversions, rho negatif significatif) -> hypothese confirmee")
    lib.log("    (option A) : construire une correction ciblee sur la zone d'ambiguite.")
    lib.log("  - Si les erreurs sont dispersees sur toutes les zones (y compris zone forte) et/ou la relation")
    lib.log("    n'est pas monotone -> hypothese non confirmee (option B) : abandonner cette piste, passer a")
    lib.log("    l'approche 3 (modele specialise VERT entraine sur VERT-TRAIN, plus de volume).")

    lib.log("\n===CSV_DIAGNOSTIC_MARGE_START===")
    lib.log("benchmark,zone,n_courses,top1_pct,top3_pct,top5_pct,n_erreurs_top1,pct_des_erreurs")
    for ligne in resultat_reel["lignes_csv"] + resultat_propre["lignes_csv"]:
        lib.log(",".join(str(x) for x in ligne))
    lib.log("===CSV_DIAGNOSTIC_MARGE_END===")

    lib.log(f"\n===SPEARMAN_REEL=== {resultat_reel['spearman_rho']}")
    lib.log(f"===SPEARMAN_PROPRE=== {resultat_propre['spearman_rho']}")
    lib.log(f"===INVERSIONS_REEL=== {resultat_reel['n_inversions_deciles']}")
    lib.log(f"===INVERSIONS_PROPRE=== {resultat_propre['n_inversions_deciles']}")
    lib.log(f"===PART_ERREURS_ZONES_FAIBLES_REEL=== {part_faibles_reel}")
    lib.log(f"===PART_ERREURS_ZONES_FAIBLES_PROPRE=== {part_faibles_propre}")


if __name__ == "__main__":
    main()
