# -*- coding: utf-8 -*-
"""
piste7_approche2_classification5.py -- approche 2 de la recherche
methodologique Top-1 (feu vert donne par Dorian le 03/09/2026, apres rejet
definitif de l'approche 3 -- le volume supplementaire de VERT-TRAIN ne
resout pas le probleme : Top-1 -2.8pt reel / -2.6pt propre malgre ~93000
lignes d'entrainement).

Architecture STRICTE demandee :
  B+genealogie ACTUEL ET FIGE
  -> filtre VERT ACTUEL ET FIGE (seuil 0.5848, INCHANGE)
  -> Top-5 ACTUEL (jamais modifie)
  -> classification conjointe des 5 candidats ("lequel des 5 est le
     gagnant ?"), avec des variables RELATIVES entre les 5 candidats de la
     meme course (transformation des 240 variables existantes -- AUCUNE
     nouvelle donnee externe)
  -> choix du Top-1 final (reordonnancement interne uniquement).

Contraintes explicites de Dorian, toutes respectees ici :
  - B+genealogie et le filtre VERT (seuils 0.5848 / 0.4662) : INCHANGES.
  - Le Top-5 reste EXACTEMENT celui de la reference (meme mecanisme de
    repli sur rang_geneal que piste7_etape2b et piste7_approche3 : le
    candidat ne peut jamais faire gagner une course dont le vrai gagnant
    n'etait pas dans le Top-5 d'origine).
  - Aucune cote utilisee.
  - Aucune variable "handicap_valeur" utilisee (verifie explicitement ci-
    dessous : cette variable n'existe pas dans les 240 colonnes v3 +
    genealogie du checkpoint standard -- assertion de securite incluse).
  - Aucune nouvelle donnee externe : les "variables relatives" sont de
    pures transformations arithmetiques (valeur du candidat moins moyenne
    des 4 AUTRES candidats de la meme course) des 240 variables deja
    existantes -- pas une source d'information nouvelle.
  - Pas de pairwise refait : la cible est une classification BINAIRE
    directe par ligne (candidat, course) -- "ce candidat est-il le
    gagnant ? oui/non" -- pas une fonction de perte par paire comme
    B+genealogie (lambdarank).
  - Un seul candidat, un seul protocole, un seul verdict : AUCUNE variante
    n'est entrainee ici (pas de version "sans variables relatives" a cote
    -- l'attribution du gain aux variables relatives est verifiee par
    l'importance par permutation du modele unique retenu, pas par un
    deuxieme modele).

Construction de la cible (verifiee explicitement dans ce script) :
  - Une ligne par (course, candidat du Top-5 VERT).
  - Cible = 1 si ce candidat est le vrai gagnant de la course, 0 sinon.
  - Assertion stricte : jamais plus d'un gagnant par groupe de 5 candidats.
  - Courses ou le vrai gagnant est HORS Top-5 : ces courses sont EXCLUES de
    l'ENTRAINEMENT (aucune cible positive possible dans le groupe -> pas de
    signal exploitable, seulement du bruit) mais restent presentes dans
    l'EVALUATION finale (VAL_CALIB), exactement comme pour la reference et
    pour l'approche 3, afin de garder une population de decision identique
    et directement comparable a tous les runs precedents de la piste 7.
  - Selection des hyperparametres (VAL_FIT) : evaluee sur la population
    VERT VAL_FIT COMPLETE (non filtree), pour rester coherente avec la
    population reellement deployee et avec la methodologie de l'approche 3.

Separation TRAIN / VALIDATION strictement chronologique : reprise a
l'identique du checkpoint piste6_phase1_split4.py (TRAIN=passe, VALIDATION
=futur proche, aucun melange), et du sous-decoupage chronologique VAL_FIT /
VAL_CALIB deja utilise dans tous les runs precedents de la piste 7.

Critere de reussite fixe par Dorian AVANT le run (aucune optimisation a
posteriori) : gain Top-1 clair et suffisamment important, stable entre
REEL et PROPRE, sans degradation significative de Top-3/Top-5. Si le
resultat est negatif ou marginal, ce script ne cherche PAS de variante :
il rapporte le constat tel quel, un seul verdict.
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

# Variable explicitement interdite par Dorian pour cette approche.
VARIABLE_INTERDITE = "handicap_valeur"


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


def construire_features_relatives(X, course_ids):
    """Pour chacune des variables existantes, ajoute une colonne '<var>__rel'
    = valeur du candidat moins la moyenne des AUTRES candidats de la meme
    course (comparaison directe des forces/faiblesses au sein des 5
    candidats, demandee explicitement par Dorian). Pure transformation
    arithmetique des variables deja existantes -- AUCUNE nouvelle donnee."""
    colonnes = list(X.columns)
    tmp = X.copy()
    tmp["_course_id_tmp"] = np.asarray(course_ids)
    sommes = tmp.groupby("_course_id_tmp")[colonnes].transform("sum")
    effectifs = tmp.groupby("_course_id_tmp")[colonnes[0]].transform("count")
    moyennes_autres = (sommes - tmp[colonnes]).div((effectifs - 1).clip(lower=1), axis=0)
    rel = tmp[colonnes] - moyennes_autres
    rel.columns = [f"{c}__rel" for c in colonnes]
    rel.index = X.index
    return pd.concat([X, rel], axis=1)


def filtrer_groupes_avec_gagnant(X_rel, y, course_ids):
    """Ne garde, pour l'ENTRAINEMENT uniquement, que les lignes des courses
    ou le vrai gagnant fait partie du shortlist VERT (somme(y)==1 dans le
    groupe de 5 candidats) -- une cible degeneree (0 partout) n'apprend
    rien sur 'qui gagne'. Verification stricte du protocole : jamais plus
    d'un gagnant par groupe (assertion, pas juste un filtre silencieux)."""
    d = pd.DataFrame({"course_id": np.asarray(course_ids), "y": np.asarray(y)}, index=X_rel.index)
    somme_par_course = d.groupby("course_id")["y"].transform("sum")
    assert (somme_par_course <= 1).all(), (
        "CIBLE CORROMPUE : plus d'un gagnant detecte dans un meme groupe de "
        "5 candidats -- violation du protocole (un seul gagnant possible par course).")
    masque = somme_par_course == 1
    return X_rel[masque], d.loc[masque, "y"].values, masque


def construire_par_course_candidat(df_vert_bloc, rang_candidat_series, suffixe):
    """Identique au mecanisme de piste7_etape2b et piste7_approche3 : la ou
    le modele candidat n'a pas de rang calcule (gagnant hors shortlist), le
    rang effectif retombe sur rang_geneal -- le candidat ne peut donc
    jamais degrader une course qu'il ne touche pas, et le Top-5 lui-meme
    n'est jamais modifie."""
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
    lib.log("PISTE 7 -- APPROCHE 2 -- CLASSIFICATION DIRECTE DU GAGNANT PARMI LES 5 CANDIDATS VERT -- 03/09/2026")
    lib.log("=== AUCUN TEST A / TEST B charge ni utilise. AUCUNE COTE. AUCUNE variable handicap_valeur. ===")
    lib.log("=== B+genealogie et le filtre VERT (seuil 0.5848) sont INCHANGES. Le Top-5 reste EXACTEMENT ===")
    lib.log("=== celui de la reference -- le candidat ne fait que reordonner les 5 chevaux deja retenus. ===")
    lib.log("=== Un seul candidat, un seul protocole, un seul verdict -- aucune variante lancee ici. ===")
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
    colonnes_interdites_presentes = [c for c in X_train_v3_geneal.columns if VARIABLE_INTERDITE in c]
    lib.log(f"  Verification variable interdite '{VARIABLE_INTERDITE}' : "
            f"{'PRESENTE -- ' + str(colonnes_interdites_presentes) if colonnes_interdites_presentes else 'absente des 240 variables (OK)'}.")
    assert not colonnes_interdites_presentes, "handicap_valeur detectee dans les variables -- interdite pour cette approche."

    lib.log("\n[1/11] Entrainement B+genealogie (memes hyperparametres que tous les runs precedents, FIGE)...")
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
    lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 7 approche 2 (classification 5 candidats)")
    df_val_reel = df_val[df_val["est_benchmark_reel"]].copy()

    lib.log("\n[2/11] Identification des courses VERT de VALIDATION (seuil fige, somme_top3_proba >= "
            f"{SEUIL_VERT_FIGE}, filtre INCHANGE)...")
    df_val_reel = calculer_somme_top3_proba(df_val_reel)
    par_course_val_ind = df_val_reel.drop_duplicates("course_id")[["course_id", "somme_top3_proba"]]
    n_val_total = len(par_course_val_ind)
    courses_vert_val = set(par_course_val_ind.loc[par_course_val_ind["somme_top3_proba"] >= SEUIL_VERT_FIGE, "course_id"])
    lib.log(f"  {len(courses_vert_val)}/{n_val_total} courses VALIDATION (benchmark reel) classees VERT "
            f"({round(100*len(courses_vert_val)/n_val_total,1)}%).")

    df_vert_val = df_val_reel[df_val_reel["course_id"].isin(courses_vert_val)].copy()
    df_vert_val["dans_shortlist"] = df_vert_val["rang_geneal"] <= K_SHORTLIST

    lib.log("\n[3/11] Identification des courses VERT de TRAIN (B+genealogie applique en PREDICTION SEULE, "
            "aucun reentrainement -- meme mecanisme deja approuve pour l'approche 3)...")
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
            f"({round(100*len(courses_vert_train)/n_train_total,1)}%) -- population d'entrainement du classifieur.")

    lib.log("\n[4/11] Construction du shortlist VERT-TRAIN (5 chevaux/course, sert UNIQUEMENT a "
            "l'entrainement du classifieur -- B+genealogie et le filtre VERT restent inchanges)...")
    shortlist_train_mask = train_scores["course_id"].isin(courses_vert_train) & (train_scores["rang_geneal"] <= K_SHORTLIST)
    shortlist_train = train_scores[shortlist_train_mask]
    lib.log(f"  Shortlist VERT-TRAIN : {len(shortlist_train)} lignes pour "
            f"{shortlist_train['course_id'].nunique()} courses VERT-TRAIN.")

    X_train_spec = X_train_v3_geneal.loc[shortlist_train.index]
    y_train_spec = np.asarray(y_train_gagnant)[shortlist_train.index.values]

    lib.log("\n[5/11] Sous-decoupage VERT de VALIDATION : VAL_FIT (selection des hyperparametres), "
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

    lib.log("\n[6/11] Construction des variables RELATIVES entre les 5 candidats (valeur du candidat moins "
            "moyenne des 4 AUTRES candidats de la meme course, sur les 240 variables existantes -- "
            "AUCUNE nouvelle donnee) pour TRAIN, VAL_FIT et VAL_CALIB...")
    X_train_spec_rel = construire_features_relatives(X_train_spec, shortlist_train["course_id"])
    X_valfit_spec_rel = construire_features_relatives(X_valfit_spec, df_vert_val_fit_sl["course_id"])
    X_calib_spec_rel = construire_features_relatives(X_calib_spec, df_vert_val_calib_sl["course_id"])
    lib.log(f"  Matrice finale (240 variables propres + 240 variables relatives = {X_train_spec_rel.shape[1]} "
            f"colonnes) : TRAIN={X_train_spec_rel.shape}, VAL_FIT={X_valfit_spec_rel.shape}, VAL_CALIB={X_calib_spec_rel.shape}.")

    lib.log("\n[7/11] Construction de la cible (classification binaire directe : ce candidat est-il le "
            "gagnant ?) -- verification stricte du protocole et filtrage des courses sans gagnant dans le "
            "Top-5 (cible degeneree, exclue de l'ENTRAINEMENT uniquement)...")
    X_train_spec_rel_f, y_train_spec_f, masque_train = filtrer_groupes_avec_gagnant(
        X_train_spec_rel, y_train_spec, shortlist_train["course_id"])
    n_courses_train_avant = shortlist_train["course_id"].nunique()
    n_courses_train_apres = shortlist_train.loc[masque_train, "course_id"].nunique()
    lib.log(f"  TRAIN : {len(shortlist_train)} lignes / {n_courses_train_avant} courses VERT-TRAIN avant filtre.")
    lib.log(f"  TRAIN : {len(X_train_spec_rel_f)} lignes / {n_courses_train_apres} courses conservees pour "
            f"l'entrainement (gagnant present dans le Top-5) -- {n_courses_train_avant - n_courses_train_apres} "
            f"courses VERT-TRAIN exclues de l'ENTRAINEMENT (gagnant hors Top-5, cible non exploitable).")
    lib.log(f"  Verification protocole : au plus un gagnant par groupe de 5 candidats -- assertion validee "
            f"(sinon le script se serait arrete en erreur).")
    lib.log(f"  Somme des cibles positives (TRAIN filtre) = {int(y_train_spec_f.sum())} pour "
            f"{n_courses_train_apres} courses -- doit etre strictement egal (verification finale) : "
            f"{'OK' if int(y_train_spec_f.sum()) == n_courses_train_apres else 'ANOMALIE'}.")

    lib.log("\n[8/11] Selection des hyperparametres (regularisation forte, meme grille GBM que tout le "
            "projet) -- entraine sur VERT-TRAIN FILTRE (gagnant present), evalue en AUC sur VAL_FIT VERT "
            "COMPLET (population non filtree, identique a la population reellement deployee)...")
    params, auc_val_fit = lib.entrainer_gbm_avec_grille(
        X_train_spec_rel_f, y_train_spec_f, X_valfit_spec_rel, y_valfit_spec, lib.GRILLE_GBM, "classif5-vert")
    lib.log(f"  >>> Hyperparametres retenus : {params} (AUC VAL_FIT={round(auc_val_fit,4)})")

    lib.log("\n[9/11] Entrainement final du classifieur sur VERT-TRAIN filtre (parametres figes ci-dessus)...")
    modele_classif = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    modele_classif.fit(X_train_spec_rel_f, y_train_spec_f)

    proba_calib = modele_classif.predict_proba(X_calib_spec_rel)[:, 1]
    auc_final = round(float(roc_auc_score(y_calib_spec, proba_calib)), 4)
    logloss_final = round(float(log_loss(y_calib_spec, np.clip(proba_calib, 1e-6, 1 - 1e-6))), 4)
    auc_score_geneal_seul = round(float(roc_auc_score(y_calib_spec, df_vert_val_calib_sl["score_geneal"].values)), 4)
    lib.log(f"\n  >>> AUC classifieur 5 candidats sur VAL_CALIB (population complete, {len(y_calib_spec)} lignes) : "
            f"{auc_final}   log-loss : {logloss_final}")
    lib.log(f"  >>> AUC de score_geneal seul (B+genealogie brut) sur la MEME population VAL_CALIB : "
            f"{auc_score_geneal_seul}  (verification d'attribution : le classifieur doit depasser cette "
            f"reference pour que son gain ne soit pas simplement redondant avec le score deja exploite).")

    lib.log("\n[10/11] Importance des variables (permutation, AUC, VAL_CALIB, 20 repetitions, top 20 sur "
            f"{X_calib_spec_rel.shape[1]} variables -- verification d'attribution : part des variables "
            "RELATIVES ('__rel') dans le top 20, pour juger si la comparaison directe entre candidats est "
            "reellement exploitee par le modele)...")
    imp = permutation_importance(
        modele_classif, X_calib_spec_rel, y_calib_spec, scoring="roc_auc", n_repeats=20, random_state=lib.RANDOM_SEED)
    ordre = np.argsort(imp.importances_mean)[::-1]
    colonnes = list(X_calib_spec_rel.columns)
    n_rel_top20 = 0
    for rang_i, i in enumerate(ordre[:20], start=1):
        nom = colonnes[i]
        est_relative = nom.endswith("__rel")
        n_rel_top20 += int(est_relative)
        lib.log(f"    {rang_i:2d}. {nom:45s} {'(RELATIVE)' if est_relative else '(propre)  '} "
                f"delta_AUC_moyen={round(float(imp.importances_mean[i]), 4):+.4f} "
                f"(+/-{round(float(imp.importances_std[i]), 4)})")
    lib.log(f"  >>> {n_rel_top20}/20 variables du top 20 sont des variables RELATIVES entre candidats.")

    proba_series = pd.Series(proba_calib, index=df_vert_val_calib_sl.index)
    rang_classif_calib = proba_series.groupby(df_vert_val_calib_sl["course_id"]).rank(method="min", ascending=False)
    par_course_calib = construire_par_course_candidat(df_vert_val_calib, rang_classif_calib, "classif5")

    lib.log("\n[11/11] " + "=" * 92)
    lib.log("=== COMPARAISON REFERENCE (B+genealogie brut) vs CLASSIFICATION 5 CANDIDATS -- VAL_CALIB ===")
    lib.log("=" * 100)
    resultats = {}
    for nom_bench, colonne_filtre in [("REEL", None), ("PROPRE", "est_benchmark_propre")]:
        sous = par_course_calib if colonne_filtre is None else par_course_calib[par_course_calib[colonne_filtre]]
        m_baseline = metriques(sous, "rang_final_baseline", "REFERENCE")
        m_candidat = metriques(sous, "rang_final_classif5", "CLASSIFICATION 5 CANDIDATS")
        resultats[nom_bench] = (m_baseline, m_candidat)
        lib.log(f"\n  -- benchmark {nom_bench} -- n courses VERT (VAL_CALIB) = {m_baseline['n_courses']} --")
        if m_baseline["top1_pct"] is not None:
            lib.log(f"     REFERENCE (B+genealogie brut)      : top1={m_baseline['top1_pct']}% top3={m_baseline['top3_pct']}% top5={m_baseline['top5_pct']}%")
            lib.log(f"     CLASSIFICATION 5 CANDIDATS          : top1={m_candidat['top1_pct']}% top3={m_candidat['top3_pct']}% top5={m_candidat['top5_pct']}%")
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

    lib.log("\n=== CONCLUSION -- approche 2, classification directe 5 candidats -- constat honnete, aucune "
            "optimisation a posteriori, un seul verdict ===")
    lib.log("=" * 100)
    lib.log(f"\n  Gain Top-1 REEL   : {delta_top1_reel:+}pt   Gain Top-1 PROPRE : {delta_top1_propre:+}pt")
    lib.log(f"  Gain Top-3 REEL   : {delta_top3_reel:+}pt   Gain Top-3 PROPRE : {delta_top3_propre:+}pt")
    lib.log(f"  Gain Top-5 REEL   : {delta_top5_reel:+}pt   Gain Top-5 PROPRE : {delta_top5_propre:+}pt")
    lib.log(f"  AUC classifieur (VAL_CALIB) : {auc_final}   AUC score_geneal seul (meme population) : {auc_score_geneal_seul}")
    lib.log(f"  Variables relatives dans le top 20 (importance permutation) : {n_rel_top20}/20")
    lib.log("\n  Lecture (a interpreter honnetement par Dorian, aucun seuil de succes impose ici -- le "
            "critere de reussite a ete fixe par Dorian AVANT ce run) :")
    lib.log("  - Gain Top-1 attendu : clair, stable entre REEL et PROPRE, sans degradation Top-3/Top-5.")
    lib.log("  - Si le gain est negatif, marginal, ou instable entre REEL/PROPRE, ou si Top-3/Top-5 se "
            "degrade significativement : rejeter le candidat, ne pas chercher de variante.")

    lib.log("\n===CSV_METRIQUES_APPROCHE2_START===")
    lignes_csv = ["benchmark,label,n_courses,top1_pct,top3_pct,top5_pct"]
    for nom_bench in ("REEL", "PROPRE"):
        m_baseline, m_candidat = resultats[nom_bench]
        lignes_csv.append(f"{nom_bench},REFERENCE,{m_baseline['n_courses']},{m_baseline['top1_pct']},{m_baseline['top3_pct']},{m_baseline['top5_pct']}")
        lignes_csv.append(f"{nom_bench},CLASSIFICATION_5,{m_candidat['n_courses']},{m_candidat['top1_pct']},{m_candidat['top3_pct']},{m_candidat['top5_pct']}")
    for ligne in lignes_csv:
        lib.log(ligne)
    lib.log("===CSV_METRIQUES_APPROCHE2_END===")

    lib.log(f"\n===AUC_CLASSIF5=== {auc_final}")
    lib.log(f"===AUC_SCORE_GENEAL_SEUL=== {auc_score_geneal_seul}")
    lib.log(f"===HYPERPARAMS_RETENUS=== {params}")
    lib.log(f"===N_COURSES_VERT_TRAIN_TOTAL=== {n_train_total}")
    lib.log(f"===N_COURSES_VERT_TRAIN_AVANT_FILTRE=== {n_courses_train_avant}")
    lib.log(f"===N_COURSES_VERT_TRAIN_APRES_FILTRE=== {n_courses_train_apres}")
    lib.log(f"===N_VARIABLES_RELATIVES_TOP20=== {n_rel_top20}")
    lib.log(f"===DELTA_TOP1_REEL=== {delta_top1_reel}")
    lib.log(f"===DELTA_TOP1_PROPRE=== {delta_top1_propre}")
    lib.log(f"===DELTA_TOP3_REEL=== {delta_top3_reel}")
    lib.log(f"===DELTA_TOP3_PROPRE=== {delta_top3_propre}")
    lib.log(f"===DELTA_TOP5_REEL=== {delta_top5_reel}")
    lib.log(f"===DELTA_TOP5_PROPRE=== {delta_top5_propre}")


if __name__ == "__main__":
    main()
