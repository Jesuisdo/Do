# -*- coding: utf-8 -*-
"""
piste6_phase2_architecture_2etages.py -- PHASE 2/2 de la piste 6, demandee
par Dorian le 30/08/2026 : architecture a deux etages (selection Top-5 puis
affinage) + filtre de confiance calibre en VALIDATION et confirme, sans
aucun ajustement, sur TEST A puis TEST B.

ETAPE 1 (selection) : B+genealogie, EXACTEMENT les memes hyperparametres/
donnees que tous les runs precedents (aucune modification -- entraine sur
TRAIN, 70%). Pour chaque course de VALIDATION / TEST A / TEST B, retient les
5 chevaux les mieux classes (rang_geneal <= 5) comme "short-list".

ETAPE 2 (affinage) : classifieur pointwise (meme famille que le baseline
v3-gagnant -- HistGradientBoostingClassifier + grille, proba calibree pour
AUC/log-loss), entraine UNIQUEMENT sur les lignes de la short-list issues
d'une sous-partie de VALIDATION ("VAL_FIT", 80% chronologique des courses de
VALIDATION), et UNIQUEMENT sur les courses ou le vrai gagnant fait partie
de la short-list (sinon la ligne est structurellement non gagnable pour
l'etape 2 et diluerait l'apprentissage).

CALIBRAGE DU FILTRE DE CONFIANCE : la courbe complete (100/90/75/60/50/40/
30/20/10%) est calculee UNIQUEMENT sur "VAL_CALIB" (les 20% chronologiques
restants de VALIDATION, jamais vus par l'etape 2 pendant son entrainement --
evite un calibrage optimiste). Les seuils (des VALEURS de l'indicateur de
confiance, pas des pourcentages) sont figes a partir de cette courbe, puis
appliques TELS QUELS, sans aucun ajustement, a TEST A puis TEST B.

IMPORTANT (demande explicite de Dorian) : les courses ou l'etape 1 ne
contient pas le vrai gagnant dans sa short-list restent comptees comme des
echecs (top1=top3=top5=0) dans TOUTES les metriques de bout en bout -- rien
n'est retire du denominateur. Double benchmark (reel/propre) applique
identiquement a VALIDATION, TEST A et TEST B. Aucune cote/marche utilisee.
"""
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, log_loss
except ImportError:
    pass

try:
    import lightgbm as lgb
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    LIGHTGBM_DISPONIBLE = False

CHECKPOINT_PATH = "checkpoint_piste6_split4.pkl"
PALIERS_COUVERTURE = [100, 90, 75, 60, 50, 40, 30, 20, 10]
K_SHORTLIST = 5


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
    lib.log(f"   [{label}] arbres retenus={modele.best_iteration_}")
    return modele


def preparer_benchmark(df, exclusions, label):
    df = lib.appliquer_benchmarks(df, exclusions)
    lib.rapport_double_benchmark(df, label=label)
    return df


def sous_split_chronologique(df, frac_fit):
    """Coupe chronologiquement un dataframe (deja trie par date_course) en
    deux, par COURSE, sans melange -- meme logique que le split principal."""
    courses_ordre = df["course_id"].drop_duplicates().tolist()
    n = len(courses_ordre)
    n_fit = int(n * frac_fit)
    courses_fit = set(courses_ordre[:n_fit])
    courses_calib = set(courses_ordre[n_fit:])
    return df["course_id"].isin(courses_fit).values, df["course_id"].isin(courses_calib).values


def construire_shortlist(df, k=K_SHORTLIST):
    """Ajoute une colonne 'dans_shortlist' (rang_geneal <= k) et
    'gagnant_dans_shortlist' (par course, le vrai gagnant est-il dedans)."""
    df = df.copy()
    df["dans_shortlist"] = df["rang_geneal"] <= k
    gagnant_present = (
        df[df["est_gagnant"] == 1].groupby("course_id")["dans_shortlist"].any()
    )
    df["gagnant_dans_shortlist"] = df["course_id"].map(gagnant_present).fillna(False)
    return df


def indicateurs_confiance_shortlist(d):
    """Sur les lignes de la short-list UNIQUEMENT (5 par course), calcule
    par course : ecart de proba #1-#2 (etape 2) et entropie sur les 5."""
    lignes = []
    for course_id, groupe in d.groupby("course_id", sort=False):
        p = np.sort(groupe["proba_etape2"].values.astype(float))[::-1]
        p_norm = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
        ecart_1_2 = float(p[0] - p[1]) if len(p) >= 2 else np.nan
        entropie = float(-(p_norm * np.log(np.clip(p_norm, 1e-12, None))).sum())
        for idx in groupe.index:
            lignes.append((idx, ecart_1_2, entropie))
    df_ind = pd.DataFrame(lignes, columns=["_idx", "ecart_proba_1_2", "entropie_shortlist"]).set_index("_idx")
    return d.join(df_ind)


def metriques_bout_en_bout(df_courses, label):
    """df_courses : une ligne par course (deja restreinte au bloc/benchmark
    voulu), avec rang_final (rang du gagnant apres etape 2, ou NaN/>5 si
    l'etape 1 a deja rate -- dans ce cas rang_final doit valoir un nombre
    > 5 pour que top1/top3/top5 valent 0, jamais retire du denominateur)."""
    n = len(df_courses)
    if n == 0:
        return {"label": label, "n_courses": 0}
    top1 = round(100 * float((df_courses["rang_final"] == 1).mean()), 1)
    top3 = round(100 * float((df_courses["rang_final"] <= 3).mean()), 1)
    top5 = round(100 * float((df_courses["rang_final"] <= 5).mean()), 1)
    return {"label": label, "n_courses": n, "top1_pct": top1, "top3_pct": top3, "top5_pct": top5}


def courbe_couverture_avec_n(df_courses, indicateur, label):
    """Courbe couverture -> top1/top3/top5 + n_courses EXPLICITE a chaque
    palier (demande explicite de Dorian : ne jamais afficher un % sans le n)."""
    df_tri = df_courses.sort_values(indicateur, ascending=True, na_position="last").reset_index(drop=True)
    n_total = len(df_tri)
    lignes = []
    for pct in PALIERS_COUVERTURE:
        n_sel = max(1, int(round(n_total * pct / 100)))
        sous = df_tri.iloc[:n_sel]
        n = len(sous)
        lignes.append({
            "indicateur": label, "couverture_visee_pct": pct, "n_courses": n,
            "seuil_indicateur": round(float(sous[indicateur].iloc[-1]), 4) if n else None,
            "top1_pct": round(100 * float((sous["rang_final"] == 1).mean()), 1),
            "top3_pct": round(100 * float((sous["rang_final"] <= 3).mean()), 1),
            "top5_pct": round(100 * float((sous["rang_final"] <= 5).mean()), 1),
        })
    return pd.DataFrame(lignes)


def appliquer_seuils_figes(df_courses, indicateur, seuils, label):
    """seuils = dict {"vert": valeur_max, "orange": valeur_max} -- indicateur
    est l'entropie (plus bas = plus confiant), donc vert = indicateur <=
    seuils['vert'], orange = <= seuils['orange'], rouge = le reste."""
    d = df_courses.copy()
    d["niveau"] = np.where(
        d[indicateur] <= seuils["vert"], "VERT",
        np.where(d[indicateur] <= seuils["orange"], "ORANGE", "ROUGE"))
    lignes = []
    for niveau in ["VERT", "ORANGE", "ROUGE"]:
        sous = d[d["niveau"] == niveau]
        n = len(sous)
        lignes.append({
            "bloc": label, "niveau": niveau, "n_courses": n,
            "pct_du_bloc": round(100 * n / len(d), 1) if len(d) else None,
            "top1_pct": round(100 * float((sous["rang_final"] == 1).mean()), 1) if n else None,
            "top3_pct": round(100 * float((sous["rang_final"] <= 3).mean()), 1) if n else None,
            "top5_pct": round(100 * float((sous["rang_final"] <= 5).mean()), 1) if n else None,
        })
    return pd.DataFrame(lignes)


def auc_logloss_shortlist(df_shortlist, label):
    y = df_shortlist["est_gagnant"].values
    p = np.clip(df_shortlist["proba_etape2"].values.astype(float), 1e-6, 1 - 1e-6)
    try:
        auc = round(float(roc_auc_score(y, p)), 4)
    except ValueError as e:
        auc = None
        lib.log(f"   [{label}] AUC non calculable : {e}")
    try:
        ll = round(float(log_loss(y, p)), 4)
    except ValueError as e:
        ll = None
        lib.log(f"   [{label}] log-loss non calculable : {e}")
    lib.log(f"   [{label}] n_lignes_shortlist={len(df_shortlist)} AUC={auc} log-loss={ll}")
    return {"label": label, "n_lignes": len(df_shortlist), "auc": auc, "log_loss": ll}


def ventilation(df_courses, colonne, valeurs_labels, label_bloc):
    lignes = []
    for valeur, lib_valeur in valeurs_labels:
        sous = df_courses[df_courses[colonne] == valeur]
        n = len(sous)
        if n == 0:
            continue
        lignes.append({
            "bloc": label_bloc, "segment": lib_valeur, "n_courses": n,
            "top1_pct": round(100 * float((sous["rang_final"] == 1).mean()), 1),
            "top3_pct": round(100 * float((sous["rang_final"] <= 3).mean()), 1),
            "top5_pct": round(100 * float((sous["rang_final"] <= 5).mean()), 1),
        })
    return pd.DataFrame(lignes)


def pipeline_etage2_predire(modele, X_bloc, df_bloc):
    """Applique le classifieur etape 2 aux lignes de la short-list d'un
    bloc, calcule rang_final (rang du gagnant selon la proba etape 2 PARMI
    la short-list ; si le gagnant n'est pas dans la short-list, rang_final
    = 99 -- echec de bout en bout, jamais retire du denominateur)."""
    df = df_bloc.copy()
    mask_sl = df["dans_shortlist"].values
    df["proba_etape2"] = np.nan
    if mask_sl.sum() > 0:
        df.loc[mask_sl, "proba_etape2"] = modele.predict_proba(X_bloc[mask_sl])[:, 1]
    df["rang_etape2_intra_sl"] = (
        df[df["dans_shortlist"]].groupby("course_id")["proba_etape2"]
        .rank(method="min", ascending=False)
    )
    # groupby(...).min() plutot que set_index direct : en cas de dead-heat
    # (2+ partants a egalite sur la position 1 -- pattern reel conserve dans
    # les deux benchmarks, cf. docstring double-benchmark de v3_lib.py), il
    # peut y avoir plusieurs lignes "gagnant" pour une meme course. On garde
    # le MEILLEUR rang parmi les gagnants ex-aequo (si le modele a bien
    # classe l'un des deux en #1, c'est une reussite) et on evite le crash
    # pandas.errors.InvalidIndexError (index course_id non-unique) observe
    # lors du premier run (30/08/2026).
    rang_gagnant = (
        df[(df["dans_shortlist"]) & (df["est_gagnant"] == 1)]
        .groupby("course_id")["rang_etape2_intra_sl"].min()
    )
    par_course = df.drop_duplicates("course_id").set_index("course_id")
    par_course["rang_final"] = par_course.index.map(rang_gagnant).fillna(99).astype(int)
    par_course["gagnant_dans_shortlist"] = par_course["gagnant_dans_shortlist"].astype(bool)
    return df, par_course.reset_index()


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn non installe.")
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError("lightgbm non installe.")

    lib.log("=" * 100)
    lib.log("PISTE 6 -- PHASE 2/2 -- ARCHITECTURE 2 ETAGES (TOP-5 PUIS AFFINAGE) + FILTRE DE CONFIANCE -- 30/08/2026")
    lib.log("=" * 100)

    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    X_train_v3_geneal = checkpoint["X_train_v3_geneal"]
    X_val_v3_geneal = checkpoint["X_val_v3_geneal"]
    X_testA_v3_geneal = checkpoint["X_testA_v3_geneal"]
    X_testB_v3_geneal = checkpoint["X_testB_v3_geneal"]
    y_train_place = checkpoint["y_train_place"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    df_val = checkpoint["df_val"].reset_index(drop=True)
    df_testA = checkpoint["df_testA"].reset_index(drop=True)
    df_testB = checkpoint["df_testB"].reset_index(drop=True)
    course_id_train = checkpoint["course_id_train"]

    lib.log(f"\n   Checkpoint charge : TRAIN={X_train_v3_geneal.shape}, VAL={X_val_v3_geneal.shape}, "
            f"TEST A={X_testA_v3_geneal.shape}, TEST B={X_testB_v3_geneal.shape}.")

    # =========================================================================
    # ETAPE 1 -- B+genealogie, AUCUN changement (entraine sur TRAIN, 70%)
    # =========================================================================
    lib.log("\n[ETAPE 1] Entrainement B+genealogie (memes hyperparametres que tous les runs precedents)...")
    groups_train = groupes_consecutifs(course_id_train)
    y_train_graded = np.where(y_train_gagnant == 1, 2, np.where(y_train_place == 1, 1, 0)).astype(int)

    groups_val = groupes_consecutifs(df_val["course_id"])
    modele_etage1 = entrainer_lambdarank(
        X_train_v3_geneal, y_train_graded, groups_train, X_val_v3_geneal, checkpoint["y_val_gagnant"],
        groups_val, "B+genealogie (etape 1)")

    for nom_bloc, df_bloc, X_bloc in [("val", df_val, X_val_v3_geneal),
                                       ("testA", df_testA, X_testA_v3_geneal),
                                       ("testB", df_testB, X_testB_v3_geneal)]:
        df_bloc["score_geneal"] = modele_etage1.predict(X_bloc)
        df_bloc["rang_geneal"] = df_bloc.groupby("course_id")["score_geneal"].rank(method="min", ascending=False)

    exclusions = lib.charger_exclusions_benchmark()
    df_val = preparer_benchmark(df_val, exclusions, "VALIDATION -- piste 6")
    df_testA = preparer_benchmark(df_testA, exclusions, "TEST A -- piste 6")
    df_testB = preparer_benchmark(df_testB, exclusions, "TEST B -- piste 6")

    df_val_reel = df_val[df_val["est_benchmark_reel"]].copy()
    df_testA_reel = df_testA[df_testA["est_benchmark_reel"]].copy()
    df_testB_reel = df_testB[df_testB["est_benchmark_reel"]].copy()

    for nom, d in [("VALIDATION", df_val_reel), ("TEST A", df_testA_reel), ("TEST B", df_testB_reel)]:
        stats_rang, _ = lib.rang_distribution_gagnant(d, "rang_geneal")
        lib.log(f"   [ETAPE 1 seule, reference, {nom}] n={stats_rang['n_courses']} "
                f"top1={stats_rang['top1_pct']}% top3={stats_rang['cumul_top3_pct']}% "
                f"top5={stats_rang['cumul_top5_pct']}%")

    # Short-list Top-5 sur chaque bloc (benchmark reel uniquement)
    df_val_reel = construire_shortlist(df_val_reel)
    df_testA_reel = construire_shortlist(df_testA_reel)
    df_testB_reel = construire_shortlist(df_testB_reel)

    for nom, d in [("VALIDATION", df_val_reel), ("TEST A", df_testA_reel), ("TEST B", df_testB_reel)]:
        n_courses = d["course_id"].nunique()
        n_gagnable = d.drop_duplicates("course_id")["gagnant_dans_shortlist"].sum()
        lib.log(f"   [ETAPE 1 -> short-list Top-5, {nom}] {n_gagnable}/{n_courses} courses "
                f"({round(100*n_gagnable/n_courses,1)}%) ont le vrai gagnant dans la short-list.")

    # =========================================================================
    # Sous-decoupage interne de VALIDATION : VAL_FIT (80%, entraine etape 2)
    # / VAL_CALIB (20%, calibre les seuils -- jamais vu par etape 2 pendant
    # son entrainement, pour eviter un calibrage optimiste)
    # =========================================================================
    mask_fit, mask_calib = sous_split_chronologique(df_val_reel.drop_duplicates("course_id"), 0.80)
    courses_fit = set(df_val_reel.drop_duplicates("course_id").loc[mask_fit, "course_id"])
    courses_calib = set(df_val_reel.drop_duplicates("course_id").loc[mask_calib, "course_id"])
    df_val_fit = df_val_reel[df_val_reel["course_id"].isin(courses_fit)].copy()
    df_val_calib = df_val_reel[df_val_reel["course_id"].isin(courses_calib)].copy()
    lib.log(f"\n   Sous-decoupage VALIDATION : VAL_FIT={df_val_fit['course_id'].nunique()} courses "
            f"(entraine etape 2), VAL_CALIB={df_val_calib['course_id'].nunique()} courses "
            f"(calibre les seuils, jamais vu par etape 2 a l'entrainement).")

    # =========================================================================
    # ETAPE 2 -- classifieur pointwise, entraine UNIQUEMENT sur les lignes de
    # la short-list de VAL_FIT, UNIQUEMENT sur les courses gagnables.
    # =========================================================================
    lib.log("\n[ETAPE 2] Entrainement du classifieur d'affinage (short-list Top-5, VAL_FIT uniquement)...")

    def sous_matrice(df_source, X_source_full, index_source, mask_lignes):
        pos = index_source.get_indexer(df_source.index[mask_lignes])
        return X_source_full.iloc[pos].reset_index(drop=True)

    idx_val = df_val.index
    mask_fit_gagnable = (df_val_fit["dans_shortlist"]) & (df_val_fit["gagnant_dans_shortlist"])
    mask_calib_gagnable_pour_grille = (df_val_calib["dans_shortlist"]) & (df_val_calib["gagnant_dans_shortlist"])

    X_fit = sous_matrice(df_val_fit, X_val_v3_geneal, idx_val, mask_fit_gagnable.values)
    y_fit = df_val_fit.loc[mask_fit_gagnable, "est_gagnant"].values
    X_grille_eval = sous_matrice(df_val_calib, X_val_v3_geneal, idx_val, mask_calib_gagnable_pour_grille.values)
    y_grille_eval = df_val_calib.loc[mask_calib_gagnable_pour_grille, "est_gagnant"].values

    lib.log(f"   Lignes d'entrainement etape 2 (short-list, courses gagnables, VAL_FIT) : {len(X_fit)}")

    params_etage2, _ = lib.entrainer_gbm_avec_grille(
        X_fit, y_fit, X_grille_eval, y_grille_eval, lib.GRILLE_GBM, "etape2-affinage")
    modele_etage2 = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_etage2)
    modele_etage2.fit(X_fit, y_fit)

    # Application de l'etape 2 a VAL_CALIB (calibrage), TEST A, TEST B (confirmation)
    X_val_calib_full = sous_matrice(df_val_calib, X_val_v3_geneal, idx_val, np.ones(len(df_val_calib), dtype=bool))
    lignes_val_calib, courses_val_calib = pipeline_etage2_predire(modele_etage2, X_val_calib_full, df_val_calib)

    idx_testA = df_testA.index
    X_testA_full = sous_matrice(df_testA_reel, X_testA_v3_geneal, idx_testA, np.ones(len(df_testA_reel), dtype=bool))
    lignes_testA, courses_testA = pipeline_etage2_predire(modele_etage2, X_testA_full, df_testA_reel)

    idx_testB = df_testB.index
    X_testB_full = sous_matrice(df_testB_reel, X_testB_v3_geneal, idx_testB, np.ones(len(df_testB_reel), dtype=bool))
    lignes_testB, courses_testB = pipeline_etage2_predire(modele_etage2, X_testB_full, df_testB_reel)

    lib.log("\n" + "=" * 100)
    lib.log("=== METRIQUES DE BOUT EN BOUT (etape1 -> etape2), echec etape1 = rang_final 99, jamais retire ===")
    lib.log("=" * 100)
    for nom, courses_bloc in [("VAL_CALIB", courses_val_calib), ("TEST A", courses_testA), ("TEST B", courses_testB)]:
        m = metriques_bout_en_bout(courses_bloc, nom)
        lib.log(f"   [{nom}] n={m['n_courses']} top1={m.get('top1_pct')}% top3={m.get('top3_pct')}% "
                f"top5={m.get('top5_pct')}%")

    lib.log("\n" + "=" * 100)
    lib.log("=== AUC / LOG-LOSS sur les lignes de la short-list (gagnant + 4 autres, toutes courses) ===")
    lib.log("=" * 100)
    for nom, d in [("VAL_CALIB", lignes_val_calib), ("TEST A", lignes_testA), ("TEST B", lignes_testB)]:
        sl = d[d["dans_shortlist"]].dropna(subset=["proba_etape2"])
        auc_logloss_shortlist(sl, nom)

    # =========================================================================
    # COURBE DE CONFIANCE -- calibree UNIQUEMENT sur VAL_CALIB
    # =========================================================================
    lib.log("\n" + "=" * 100)
    lib.log("=== COURBE COUVERTURE/CONFIANCE -- calibree UNIQUEMENT sur VAL_CALIB (jamais vu par etape 2) ===")
    lib.log("=" * 100)
    sl_val_calib = lignes_val_calib[lignes_val_calib["dans_shortlist"]].dropna(subset=["proba_etape2"]).copy()
    sl_val_calib = indicateurs_confiance_shortlist(sl_val_calib)
    ind_par_course = sl_val_calib.drop_duplicates("course_id")[["course_id", "entropie_shortlist", "ecart_proba_1_2"]]
    courses_val_calib_ind = courses_val_calib.merge(ind_par_course, on="course_id", how="left")

    courbe_val_calib = courbe_couverture_avec_n(courses_val_calib_ind, "entropie_shortlist", "entropie_shortlist (VAL_CALIB)")
    for _, row in courbe_val_calib.iterrows():
        lib.log(f"   couverture_visee={row['couverture_visee_pct']:>3}% n_courses={row['n_courses']:>5} "
                f"seuil_entropie<={row['seuil_indicateur']} -> top1={row['top1_pct']}% top3={row['top3_pct']}% "
                f"top5={row['top5_pct']}%")

    # Seuils fige a partir de la courbe VAL_CALIB : on prend les paliers 30%
    # (-> VERT) et 60% (-> ORANGE) comme repere de depart, MAIS le seuil
    # retenu est bien une VALEUR d'entropie (celle observee a ce palier),
    # pas un pourcentage -- donc la couverture reelle sur TEST A/TEST B sera
    # ce qu'elle sera, jamais forcee.
    seuil_vert = float(courbe_val_calib.loc[courbe_val_calib["couverture_visee_pct"] == 30, "seuil_indicateur"].iloc[0])
    seuil_orange = float(courbe_val_calib.loc[courbe_val_calib["couverture_visee_pct"] == 60, "seuil_indicateur"].iloc[0])
    seuils_figes = {"vert": seuil_vert, "orange": seuil_orange}
    lib.log(f"\n   SEUILS FIGES (a partir de VAL_CALIB uniquement, jamais retouches ensuite) : "
            f"VERT si entropie<={seuil_vert}, ORANGE si entropie<={seuil_orange}, sinon ROUGE.")

    lib.log("\n" + "=" * 100)
    lib.log("=== APPLICATION DES SEUILS FIGES, SANS AUCUN AJUSTEMENT -- TEST A puis TEST B ===")
    lib.log("=" * 100)

    lignes_testA_sl = lignes_testA[lignes_testA["dans_shortlist"]].dropna(subset=["proba_etape2"]).copy()
    lignes_testA_sl = indicateurs_confiance_shortlist(lignes_testA_sl)
    ind_testA = lignes_testA_sl.drop_duplicates("course_id")[["course_id", "entropie_shortlist"]]
    courses_testA_ind = courses_testA.merge(ind_testA, on="course_id", how="left")
    resultat_testA = appliquer_seuils_figes(courses_testA_ind, "entropie_shortlist", seuils_figes, "TEST A")
    for _, row in resultat_testA.iterrows():
        lib.log(f"   [TEST A] {row['niveau']:6s} n={row['n_courses']:>5} ({row['pct_du_bloc']}% du bloc) "
                f"top1={row['top1_pct']}% top3={row['top3_pct']}% top5={row['top5_pct']}%")

    lignes_testB_sl = lignes_testB[lignes_testB["dans_shortlist"]].dropna(subset=["proba_etape2"]).copy()
    lignes_testB_sl = indicateurs_confiance_shortlist(lignes_testB_sl)
    ind_testB = lignes_testB_sl.drop_duplicates("course_id")[["course_id", "entropie_shortlist"]]
    courses_testB_ind = courses_testB.merge(ind_testB, on="course_id", how="left")
    resultat_testB = appliquer_seuils_figes(courses_testB_ind, "entropie_shortlist", seuils_figes, "TEST B")
    for _, row in resultat_testB.iterrows():
        lib.log(f"   [TEST B] {row['niveau']:6s} n={row['n_courses']:>5} ({row['pct_du_bloc']}% du bloc) "
                f"top1={row['top1_pct']}% top3={row['top3_pct']}% top5={row['top5_pct']}%")

    # =========================================================================
    # Ventilation nb_partants / handicap, sur TEST A et TEST B (bout en bout)
    # =========================================================================
    lib.log("\n" + "=" * 100)
    lib.log("=== VENTILATION nb_partants / handicap (bout en bout, TEST A + TEST B) ===")
    lib.log("=" * 100)
    for nom, courses_bloc, df_source in [("TEST A", courses_testA, df_testA_reel), ("TEST B", courses_testB, df_testB_reel)]:
        meta = df_source.drop_duplicates("course_id")[["course_id", "nb_partants_reel", "categorie_particularite"]]
        cb = courses_bloc.merge(meta, on="course_id", how="left")
        cb["bucket_partants"] = cb["nb_partants_reel"].apply(lib.bucket_partants)
        cb["est_handicap"] = cb["categorie_particularite"].fillna("").str.contains("HANDICAP")
        vp = ventilation(cb, "bucket_partants",
                          [("petit (<=7)", "petit (<=7)"), ("moyen (8-12)", "moyen (8-12)"), ("grand (13+)", "grand (13+)")],
                          nom)
        for _, row in vp.iterrows():
            lib.log(f"   [{nom}] {row['segment']:15s} n={row['n_courses']:>5} top1={row['top1_pct']}% "
                    f"top3={row['top3_pct']}% top5={row['top5_pct']}%")
        vh = ventilation(cb, "est_handicap", [(True, "HANDICAP"), (False, "NON HANDICAP")], nom)
        for _, row in vh.iterrows():
            lib.log(f"   [{nom}] {row['segment']:15s} n={row['n_courses']:>5} top1={row['top1_pct']}% "
                    f"top3={row['top3_pct']}% top5={row['top5_pct']}%")

    lib.log("\n" + "=" * 100)
    lib.log("=== FIN -- resultats REELS de TEST A / TEST B, non ajustes. Seuils figes sur VAL_CALIB uniquement. ===")
    lib.log("=" * 100)

    lib.log("\n===CSV_COURBE_VALCALIB_START===")
    for ligne in courbe_val_calib.to_csv(index=False).splitlines():
        lib.log(ligne)
    lib.log("===CSV_COURBE_VALCALIB_END===")

    lib.log("\n===CSV_SEUILS_TESTA_START===")
    for ligne in resultat_testA.to_csv(index=False).splitlines():
        lib.log(ligne)
    lib.log("===CSV_SEUILS_TESTA_END===")

    lib.log("\n===CSV_SEUILS_TESTB_START===")
    for ligne in resultat_testB.to_csv(index=False).splitlines():
        lib.log(ligne)
    lib.log("===CSV_SEUILS_TESTB_END===")


if __name__ == "__main__":
    main()
