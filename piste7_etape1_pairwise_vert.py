# -*- coding: utf-8 -*-
"""
piste7_etape1_pairwise_vert.py -- ETAPE 1/3 de l'amelioration du choix du
n°1 (piste 7, demandee par Dorian le 01/09/2026), UNIQUEMENT sur les
courses VERT (filtre de confiance deja fige et valide -- ne pas y toucher).

Objectif : verifier si un modele de comparaison PAR PAIRE (plutot que
pointwise comme l'etage d'affinage rejete en piste 6, AUC ~0.61 sur TOUTES
les courses) degage un signal reel une fois restreint a la population VERT
(plus homogene) et reformule comme une tache de comparaison directe entre 2
chevaux de la short-list, plutot qu'une classification absolue.

Protocole strict valide par Dorian :
  - UNIQUEMENT les donnees et variables deja disponibles (les 240 colonnes
    v3+genealogie deja utilisees par B+genealogie) -- AUCUNE nouvelle
    variable a ce stade.
  - UNIQUEMENT les courses VERT, identifiees via les seuils DEJA FIGES de
    la piste 7 (somme_top3_proba >= 0.5848 sur score_geneal, calcule
    exactement comme dans piste7_phase2_confiance_directe.py -- le filtre
    lui-meme n'est PAS retouche).
  - Sous-decoupage chronologique de VALIDATION (courses VERT uniquement) :
    VAL_FIT (80%, entraine le modele par paire) / VAL_CALIB (20%, decide
    seul si le signal est reel -- jamais vu par le modele a l'entrainement).
  - AUCUN TEST A / TEST B dans cette phase (decision interdite dessus par
    Dorian) -- ni charge ni utilise ici.

Modele par paire : pour chaque course VERT, pour chaque paire ORDONNEE
(i, j) de chevaux de la short-list (rang_geneal <= 5, donc au plus 5
chevaux, generalement 5 puisque nb_partants_reel >= 3 et VERT implique en
pratique des champs suffisamment grands), on construit un exemple
d'entrainement : X = features(i) - features(j), y = 1 si i finit devant j
(position_arrivee_i < position_arrivee_j), 0 sinon. Les paires ex-aequo
(meme position_arrivee) sont exclues de l'ENTRAINEMENT et de l'EVALUATION
AUC (ambigues), mais PAS de l'inference (il faut bien classer tous les
chevaux de la short-list, y compris si deux d'entre eux ont fini a egalite
lors de courses passees -- non pertinent ici puisqu'on classe la course
COURANTE).

Reclassement : pour chaque cheval i de la short-list d'une course, score
agrege = somme sur tous les autres j de la short-list de P(i bat j) predit
par le modele. Les 5 chevaux sont ensuite classes par score agrege
decroissant -- c'est ce nouveau classement (rang_paire) qui remplace
rang_geneal pour designer le nouveau n°1, UNIQUEMENT au sein de la
short-list deja figee (donc le Top 5 est mathematiquement inchange -- meme
ensemble de 5 chevaux, seul l'ordre change).
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
K_SHORTLIST = 5

# Seuils FIGES de la piste 7 (phase2_confiance_directe, run du 01/09/2026) --
# NE PAS RETOUCHER. Indicateur retenu : somme_top3_proba, plus haut = plus
# confiant. VERT si somme_top3_proba >= SEUIL_VERT.
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
    lib.log(f"   [{label}] arbres retenus={modele.best_iteration_}")
    return modele


def calculer_somme_top3_proba(d):
    """Identique a piste7_phase2_confiance_directe.calculer_indicateurs_confiance,
    limite a la seule quantite necessaire ici (l'indicateur deja retenu et
    fige) -- softmax des score_geneal sur TOUT le champ de chaque course."""
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


def construire_paires_entrainement(df_shortlist, X_arr, exclure_ex_aequo=True):
    """Paires ORDONNEES (i, j) au sein de chaque course. X = X[i] - X[j],
    y = 1 si i finit devant j. Les index de df_shortlist doivent etre des
    positions valides directement dans X_arr (aucun reset_index en amont)."""
    X_pairs, y_pairs = [], []
    for _, groupe in df_shortlist.groupby("course_id", sort=False):
        indices = groupe.index.tolist()
        positions = groupe["position_arrivee"].values
        n = len(indices)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if exclure_ex_aequo and positions[a] == positions[b]:
                    continue
                label = 1 if positions[a] < positions[b] else 0
                X_pairs.append(X_arr[indices[a]] - X_arr[indices[b]])
                y_pairs.append(label)
    return np.array(X_pairs), np.array(y_pairs)


def calculer_rang_paire(df_shortlist, X_arr, modele):
    """Pour chaque course, score agrege par cheval = somme des P(bat j)
    predites face a tous les autres chevaux de sa short-list, puis rang
    (1 = meilleur score agrege). Retourne une Series indexee comme
    df_shortlist (memes index que les lignes d'origine)."""
    rangs = {}
    for _, groupe in df_shortlist.groupby("course_id", sort=False):
        indices = groupe.index.tolist()
        n = len(indices)
        if n < 2:
            for idx in indices:
                rangs[idx] = 1
            continue
        paires_X, cible_a = [], []
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                paires_X.append(X_arr[indices[a]] - X_arr[indices[b]])
                cible_a.append(a)
        proba = modele.predict_proba(np.array(paires_X))[:, 1]
        scores = np.zeros(n)
        for p, a in zip(proba, cible_a):
            scores[a] += p
        ordre = pd.Series(scores).rank(method="min", ascending=False).values
        for idx, r in zip(indices, ordre):
            rangs[idx] = int(r)
    return pd.Series(rangs, name="rang_paire")


def construire_par_course_reranked(df_vert_bloc, df_shortlist_avec_rang_paire):
    """Une ligne par course VERT : rang_final_baseline (rang_geneal du vrai
    gagnant, meme convention que partout ailleurs -- min en cas de
    dead-heat) et rang_final_reranked (rang_paire du gagnant s'il est dans
    la short-list, sinon identique au baseline -- le reclassement ne peut
    rien changer si le gagnant n'y est pas)."""
    d = df_vert_bloc.copy()
    d["rang_paire"] = df_shortlist_avec_rang_paire.reindex(d.index)
    d["rang_effectif_reranked"] = d["rang_paire"].fillna(d["rang_geneal"])

    gagnants = d[d["est_gagnant"] == 1]
    rang_baseline = gagnants.groupby("course_id")["rang_geneal"].min()
    rang_reranked = gagnants.groupby("course_id")["rang_effectif_reranked"].min()

    par_course = d.drop_duplicates("course_id").set_index("course_id").copy()
    par_course["rang_final_baseline"] = par_course.index.map(rang_baseline)
    par_course["rang_final_reranked"] = par_course.index.map(rang_reranked)
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
    lib.log("PISTE 7 -- ETAPE 1/3 -- RECLASSEMENT PAR PAIRE, COURSES VERT UNIQUEMENT -- 01/09/2026")
    lib.log("=== AUCUN TEST A / TEST B charge ni utilise dans cette phase (decision interdite dessus). ===")
    lib.log("=" * 100)

    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    X_train_v3_geneal = checkpoint["X_train_v3_geneal"]
    X_val_v3_geneal = checkpoint["X_val_v3_geneal"]
    y_train_place = checkpoint["y_train_place"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    df_val = checkpoint["df_val"].reset_index(drop=True)
    course_id_train = checkpoint["course_id_train"]
    X_arr = X_val_v3_geneal.to_numpy()  # aligne ligne-a-ligne avec df_val.index (0..N-1)

    lib.log(f"\n   Checkpoint charge : TRAIN={X_train_v3_geneal.shape}, VAL={X_val_v3_geneal.shape} "
            f"(240 variables deja existantes, aucune nouvelle).")

    # =========================================================================
    # B+genealogie, AUCUN changement -- identique a tous les runs precedents.
    # =========================================================================
    lib.log("\n[1/7] Entrainement B+genealogie (memes hyperparametres que tous les runs precedents)...")
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
    lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 7 etape 1")
    df_val_reel = df_val[df_val["est_benchmark_reel"]].copy()

    # =========================================================================
    # Filtre de confiance FIGE (piste 7) -- identification des courses VERT.
    # =========================================================================
    lib.log("\n[2/7] Identification des courses VERT (seuil fige, somme_top3_proba >= "
            f"{SEUIL_VERT_FIGE}, filtre INCHANGE)...")
    df_val_reel = calculer_somme_top3_proba(df_val_reel)
    par_course_ind = df_val_reel.drop_duplicates("course_id")[["course_id", "somme_top3_proba"]]
    n_val_total = len(par_course_ind)
    courses_vert = set(par_course_ind.loc[par_course_ind["somme_top3_proba"] >= SEUIL_VERT_FIGE, "course_id"])
    lib.log(f"   {len(courses_vert)}/{n_val_total} courses VALIDATION (benchmark reel) classees VERT "
            f"({round(100*len(courses_vert)/n_val_total,1)}%).")

    df_vert = df_val_reel[df_val_reel["course_id"].isin(courses_vert)].copy()
    df_vert["dans_shortlist"] = df_vert["rang_geneal"] <= K_SHORTLIST

    # =========================================================================
    # Sous-decoupage chronologique VAL_FIT (80%) / VAL_CALIB (20%), VERT uniquement.
    # =========================================================================
    courses_fit, courses_calib = sous_split_chronologique(df_vert.drop_duplicates("course_id"), 0.80)
    df_vert_fit = df_vert[df_vert["course_id"].isin(courses_fit)]
    df_vert_calib = df_vert[df_vert["course_id"].isin(courses_calib)]
    lib.log(f"\n[3/7] Sous-decoupage VERT : VAL_FIT={len(courses_fit)} courses (entraine le modele par paire), "
            f"VAL_CALIB={len(courses_calib)} courses (decide seul si le signal est reel).")

    df_vert_fit_sl = df_vert_fit[df_vert_fit["dans_shortlist"]]
    df_vert_calib_sl = df_vert_calib[df_vert_calib["dans_shortlist"]]

    # =========================================================================
    # Construction des paires + entrainement (grille evaluee sur VAL_CALIB,
    # meme convention que l'etage d'affinage de la piste 6).
    # =========================================================================
    lib.log("\n[4/7] Construction des paires (differences de features, ex-aequo exclus)...")
    X_pairs_fit, y_pairs_fit = construire_paires_entrainement(df_vert_fit_sl, X_arr)
    X_pairs_calib, y_pairs_calib = construire_paires_entrainement(df_vert_calib_sl, X_arr)
    lib.log(f"   Paires VAL_FIT (entrainement) : {len(X_pairs_fit)}")
    lib.log(f"   Paires VAL_CALIB (evaluation) : {len(X_pairs_calib)}")

    lib.log("\n[5/7] Recherche d'hyperparametres (grille existante, evaluee sur VAL_CALIB)...")
    params, auc_grille = lib.entrainer_gbm_avec_grille(
        X_pairs_fit, y_pairs_fit, X_pairs_calib, y_pairs_calib, lib.GRILLE_GBM, "etape1-paire-vert")
    modele_paire = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    modele_paire.fit(X_pairs_fit, y_pairs_fit)

    proba_calib = modele_paire.predict_proba(X_pairs_calib)[:, 1]
    auc_final = round(float(roc_auc_score(y_pairs_calib, proba_calib)), 4)
    logloss_final = round(float(log_loss(y_pairs_calib, np.clip(proba_calib, 1e-6, 1 - 1e-6))), 4)
    lib.log(f"\n   >>> AUC du modele par paire sur VAL_CALIB (courses VERT, jamais vu a l'entrainement) : {auc_final}")
    lib.log(f"   >>> log-loss du modele par paire sur VAL_CALIB : {logloss_final}")
    lib.log(f"   (reference piste 6, etage d'affinage pointwise, TOUTES les courses : AUC ~0.61)")

    # =========================================================================
    # Reclassement des courses VERT de VAL_CALIB, comparaison au brut.
    # =========================================================================
    lib.log("\n[6/7] Reclassement des 5 chevaux de chaque course VERT de VAL_CALIB...")
    rang_paire_calib = calculer_rang_paire(df_vert_calib_sl, X_arr, modele_paire)
    par_course_calib = construire_par_course_reranked(df_vert_calib, rang_paire_calib)

    lib.log("\n" + "=" * 100)
    lib.log("=== [7/7] COMPARAISON classement B+genealogie BRUT vs RECLASSEMENT PAR PAIRE -- VAL_CALIB, courses VERT ===")
    lib.log("=" * 100)
    for nom_bench, colonne_filtre in [("REEL", None), ("PROPRE", "est_benchmark_propre")]:
        sous = par_course_calib if colonne_filtre is None else par_course_calib[par_course_calib[colonne_filtre]]
        m_baseline = metriques(sous, "rang_final_baseline", f"BRUT B+genealogie ({nom_bench})")
        m_reranked = metriques(sous, "rang_final_reranked", f"RECLASSE par paire ({nom_bench})")
        lib.log(f"\n   -- benchmark {nom_bench} -- n courses VERT (VAL_CALIB) = {m_baseline['n_courses']} --")
        lib.log(f"      BRUT B+genealogie   : top1={m_baseline['top1_pct']}% top3={m_baseline['top3_pct']}% top5={m_baseline['top5_pct']}%")
        lib.log(f"      RECLASSE par paire  : top1={m_reranked['top1_pct']}% top3={m_reranked['top3_pct']}% top5={m_reranked['top5_pct']}%")
        if m_baseline['top1_pct'] is not None:
            delta_top1 = round(m_reranked['top1_pct'] - m_baseline['top1_pct'], 1)
            delta_top3 = round(m_reranked['top3_pct'] - m_baseline['top3_pct'], 1)
            lib.log(f"      DELTA               : top1={delta_top1:+}pt top3={delta_top3:+}pt "
                    f"(top5 doit rester strictement identique, verification de construction)")
        else:
            lib.log("      DELTA               : n/a (aucune course dans ce sous-ensemble)")

    lib.log("\n" + "=" * 100)
    lib.log("=== FIN ETAPE 1 -- rappel : reference deja validee sur VERT (VAL_CALIB de piste7_phase2) ~34-36% top1, ~73-74% top3, ~91-92% top5. ===")
    lib.log("=== Decision : si top1 ne progresse pas sans perte de top3, NE PAS construire l'etape 2. ===")
    lib.log("=" * 100)

    lib.log("\n===CSV_METRIQUES_RERANK_START===")
    lignes_csv = ["benchmark,label,n_courses,top1_pct,top3_pct,top5_pct"]
    for nom_bench, colonne_filtre in [("REEL", None), ("PROPRE", "est_benchmark_propre")]:
        sous = par_course_calib if colonne_filtre is None else par_course_calib[par_course_calib[colonne_filtre]]
        for colonne_rang, label in [("rang_final_baseline", "BRUT"), ("rang_final_reranked", "RECLASSE")]:
            m = metriques(sous, colonne_rang, label)
            lignes_csv.append(f"{nom_bench},{label},{m['n_courses']},{m['top1_pct']},{m['top3_pct']},{m['top5_pct']}")
    for ligne in lignes_csv:
        lib.log(ligne)
    lib.log("===CSV_METRIQUES_RERANK_END===")

    lib.log(f"\n===AUC_PAIRE_VAL_CALIB=== {auc_final}")
    lib.log(f"===LOGLOSS_PAIRE_VAL_CALIB=== {logloss_final}")


if __name__ == "__main__":
    main()
