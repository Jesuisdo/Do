# -*- coding: utf-8 -*-
"""
piste7_etape2_forme_adversaires_vert.py -- ETAPE 2/3 de l'amelioration du
choix du n°1 (piste 7, demandee par Dorian le 01/09/2026), UNIQUEMENT sur
les courses VERT (filtre de confiance deja fige et valide -- ne pas y
toucher). Fait suite a l'etape 1 (reclassement par paire, AUC=0.6842 mais
Top-1 en baisse) : cette etape teste si l'AJOUT DE NOUVELLES VARIABLES
(forme recente des adversaires) plutot qu'un changement de formulation du
modele permet de faire progresser le Top-1.

Point architectural, valide par Dorian le 01/09/2026 : le B+genealogie
utilise pour determiner les courses VERT/ORANGE/ROUGE N'EST PAS MODIFIE --
memes 240 variables, meme entrainement sur TRAIN, memes seuils figes
(0.5848 / 0.4662). Les nouvelles variables de forme des adversaires ne
sont utilisees QUE par un modele candidat SEPARE, entraine et evalue
UNIQUEMENT au sein des courses deja classees VERT par le filtre inchange,
exactement comme l'etape 1 (protocole VAL_FIT/VAL_CALIB, aucun TEST A/B).

Nouvelles variables construites ICI (aucune connexion Supabase -- tout est
deja present dans df_val du checkpoint, cf. variables_historiques.py) :
- adv_forme5_moyenne_adversaires / _n_adversaires_connus / _meilleur_adversaire
/ _pire_adversaire : agregats de forme_moy_position_5 (deja point-in-time)
sur les AUTRES partants de la meme course, exclusion par IDENTITE (position),
pas par valeur (contrairement a niveau_moyen_adversaires dans
variables_historiques.py, qui exclut par egalite de valeur -- plus fragile
en cas d'ex-aequo exact).
- adv_forme5_moyenne_top3_papier : forme_moy_position_5 moyenne des 3
adversaires (hors soi) les mieux classes selon rang_papier_taux_victoire
(deja existant, point-in-time).
- forme_moy_position_5 / forme_moy_position_10 / forme_tendance_5_vs_10 :
rang + z-score intra-course via le mecanisme GENERIQUE deja en place
(lib.ajouter_variables_relatives), jamais applique a ces variables jusqu'ici
(seules musique_dernier/musique_moy3 en beneficiaient).

Toutes ces variables sont calculees sur le CHAMP ENTIER de chaque course
VERT (tous les partants, pas seulement la short-list), APRES que toutes les
features individuelles point-in-time de la course ont deja ete construites
par variables_historiques.py -- aucune information posterieure a la course
cible n'est utilisee (meme garantie que niveau_moyen_adversaires).

Modele candidat : POINTWISE (contrairement a l'etape 1, pairwise) --
classification binaire est_gagnant, sur les 5 chevaux de la short-list de
chaque course VERT, features = [score_geneal, rang_geneal] + les variables
de forme des adversaires ci-dessus. Choix deliberement pointwise (comme
l'etage d'affinage rejete en piste 6, AUC~0.61 sur TOUTES les courses,
TOUTES variables generiques) pour tester precisement l'hypothese de Dorian :
l'etage pointwise echouait-il par manque d'INFORMATION (variables trop
generiques) ou par manque de FORMULATION adaptee ? Isole cette question de
celle deja tranchee a l'etape 1 (formulation pairwise seule, memes
variables : AUC 0.68 mais Top-1 en baisse).
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

# Seuil FIGE de la piste 7 (phase2_confiance_directe, run du 01/09/2026) --
# NE PAS RETOUCHER. Indicateur retenu : somme_top3_proba, plus haut = plus
# confiant. VERT si somme_top3_proba >= SEUIL_VERT_FIGE.
SEUIL_VERT_FIGE = 0.5848

FEATURES_CANDIDAT = [
    "score_geneal", "rang_geneal",
    "adv_forme5_moyenne_adversaires", "adv_forme5_n_adversaires_connus",
    "adv_forme5_meilleur_adversaire", "adv_forme5_pire_adversaire",
    "adv_forme5_moyenne_top3_papier",
    "forme_moy_position_5_rang_course", "forme_moy_position_5_z_course",
    "forme_moy_position_10_rang_course", "forme_moy_position_10_z_course",
    "forme_tendance_5_vs_10_rang_course", "forme_tendance_5_vs_10_z_course",
]
VARIABLES_NOUVELLES = [c for c in FEATURES_CANDIDAT if c not in ("score_geneal", "rang_geneal")]


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
    """Identique a piste7_phase2_confiance_directe / piste7_etape1 --
    l'indicateur deja fige (softmax des score_geneal sur TOUT le champ)."""
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


def construire_variables_forme_adversaires(df):
    """Ajoute, pour chaque partant d'une course VERT, des agregats de la
    forme recente (forme_moy_position_5, deja point-in-time) des AUTRES
    partants de la MEME course -- exclusion par IDENTITE (position dans le
    groupe), jamais par egalite de valeur (contrairement a
    niveau_moyen_adversaires dans variables_historiques.py, qui exclurait a
    tort deux chevaux ayant exactement le meme taux). Calcule sur le champ
    ENTIER de la course (tous les partants), pas seulement la short-list --
    aucune information posterieure a la course cible utilisee (seules des
    valeurs deja point-in-time des AUTRES chevaux sont agregees)."""
    d = df.reset_index(drop=True).copy()
    forme5 = d["forme_moy_position_5"].to_numpy(dtype=float)
    rang_papier = d["rang_papier_taux_victoire"].to_numpy(dtype=float)
    n_total = len(d)
    out_moy = np.full(n_total, np.nan)
    out_n = np.zeros(n_total, dtype=int)
    out_min = np.full(n_total, np.nan)
    out_max = np.full(n_total, np.nan)
    out_top3papier = np.full(n_total, np.nan)

    for _, idxs in d.groupby("course_id", sort=False).indices.items():
        idxs = list(idxs)
        sous_forme = forme5[idxs]
        sous_rang = rang_papier[idxs]
        m = len(idxs)
        for a in range(m):
            mask = np.ones(m, dtype=bool)
            mask[a] = False
            autres_forme = sous_forme[mask]
            autres_forme = autres_forme[~np.isnan(autres_forme)]
            gi = idxs[a]
            out_n[gi] = len(autres_forme)
            if len(autres_forme):
                out_moy[gi] = float(autres_forme.mean())
                out_min[gi] = float(autres_forme.min())
                out_max[gi] = float(autres_forme.max())
            autres_locaux = [b for b in range(m) if b != a and not np.isnan(sous_rang[b])]
            autres_locaux.sort(key=lambda b: sous_rang[b])
            top3_locaux = autres_locaux[:3]
            vals_top3 = [sous_forme[b] for b in top3_locaux if not np.isnan(sous_forme[b])]
            if vals_top3:
                out_top3papier[gi] = float(np.mean(vals_top3))

    d["adv_forme5_moyenne_adversaires"] = out_moy
    d["adv_forme5_n_adversaires_connus"] = out_n
    d["adv_forme5_meilleur_adversaire"] = out_min
    d["adv_forme5_pire_adversaire"] = out_max
    d["adv_forme5_moyenne_top3_papier"] = out_top3papier
    return d


def construire_par_course_candidat(df_vert_bloc, rang_candidat_series):
    """Une ligne par course VERT : rang_final_baseline (rang_geneal du vrai
    gagnant, min en cas de dead-heat) et rang_final_candidat (rang_candidat
    du gagnant s'il est dans la short-list, sinon identique au baseline --
    le reclassement ne peut rien changer si le gagnant n'y est pas, meme
    garantie de construction qu'a l'etape 1)."""
    d = df_vert_bloc.copy()
    d["rang_candidat"] = rang_candidat_series.reindex(d.index)
    d["rang_effectif_candidat"] = d["rang_candidat"].fillna(d["rang_geneal"])

    gagnants = d[d["est_gagnant"] == 1]
    rang_baseline = gagnants.groupby("course_id")["rang_geneal"].min()
    rang_candidat = gagnants.groupby("course_id")["rang_effectif_candidat"].min()

    par_course = d.drop_duplicates("course_id").set_index("course_id").copy()
    par_course["rang_final_baseline"] = par_course.index.map(rang_baseline)
    par_course["rang_final_candidat"] = par_course.index.map(rang_candidat)
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
    lib.log("PISTE 7 -- ETAPE 2/3 -- FORME RECENTE DES ADVERSAIRES, COURSES VERT UNIQUEMENT -- 01/09/2026")
    lib.log("=== AUCUN TEST A / TEST B charge ni utilise dans cette phase (decision interdite dessus). ===")
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

    # =========================================================================
    # B+genealogie, AUCUN changement -- identique a tous les runs precedents.
    # =========================================================================
    lib.log("\n[1/8] Entrainement B+genealogie (memes hyperparametres que tous les runs precedents)...")
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
    lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 7 etape 2")
    df_val_reel = df_val[df_val["est_benchmark_reel"]].copy()

    # =========================================================================
    # Filtre de confiance FIGE (piste 7) -- identification des courses VERT.
    # =========================================================================
    lib.log("\n[2/8] Identification des courses VERT (seuil fige, somme_top3_proba >= "
            f"{SEUIL_VERT_FIGE}, filtre INCHANGE)...")
    df_val_reel = calculer_somme_top3_proba(df_val_reel)
    par_course_ind = df_val_reel.drop_duplicates("course_id")[["course_id", "somme_top3_proba"]]
    n_val_total = len(par_course_ind)
    courses_vert = set(par_course_ind.loc[par_course_ind["somme_top3_proba"] >= SEUIL_VERT_FIGE, "course_id"])
    lib.log(f"  {len(courses_vert)}/{n_val_total} courses VALIDATION (benchmark reel) classees VERT "
            f"({round(100*len(courses_vert)/n_val_total,1)}%).")

    df_vert = df_val_reel[df_val_reel["course_id"].isin(courses_vert)].copy()
    df_vert["dans_shortlist"] = df_vert["rang_geneal"] <= K_SHORTLIST

    # =========================================================================
    # Nouvelles variables de forme des adversaires -- champ ENTIER des
    # courses VERT (avant filtrage a la short-list).
    # =========================================================================
    lib.log("\n[3/8] Construction des variables de forme recente des adversaires "
            "(champ entier des courses VERT, point-in-time)...")
    df_vert = construire_variables_forme_adversaires(df_vert)
    df_vert = lib.ajouter_variables_relatives(
        df_vert, ["forme_moy_position_5", "forme_moy_position_10", "forme_tendance_5_vs_10"])
    for col in VARIABLES_NOUVELLES:
        n_dispo = int(df_vert[col].notna().sum())
        lib.log(f"    {col:42s} couverture={round(100*n_dispo/len(df_vert),1)}% ({n_dispo}/{len(df_vert)})")

    if "niveau_moyen_adversaires" in df_vert.columns:
        corr = df_vert[["adv_forme5_moyenne_adversaires", "niveau_moyen_adversaires"]].corr().iloc[0, 1]
        lib.log(f"\n  Correlation adv_forme5_moyenne_adversaires vs niveau_moyen_adversaires (deja existant) : "
                f"{round(float(corr), 3) if pd.notna(corr) else 'NA'} (verification du risque de redondance)")

    # =========================================================================
    # Sous-decoupage chronologique VAL_FIT (80%) / VAL_CALIB (20%), VERT uniquement.
    # =========================================================================
    courses_fit, courses_calib = sous_split_chronologique(df_vert.drop_duplicates("course_id"), 0.80)
    df_vert_fit = df_vert[df_vert["course_id"].isin(courses_fit)]
    df_vert_calib = df_vert[df_vert["course_id"].isin(courses_calib)]
    lib.log(f"\n[4/8] Sous-decoupage VERT : VAL_FIT={len(courses_fit)} courses (entraine le modele candidat), "
            f"VAL_CALIB={len(courses_calib)} courses (decide seul si le signal est reel).")

    df_vert_fit_sl = df_vert_fit[df_vert_fit["dans_shortlist"]].copy()
    df_vert_calib_sl = df_vert_calib[df_vert_calib["dans_shortlist"]].copy()

    X_fit = df_vert_fit_sl[FEATURES_CANDIDAT].astype(float).to_numpy()
    y_fit = df_vert_fit_sl["est_gagnant"].astype(int).to_numpy()
    X_calib = df_vert_calib_sl[FEATURES_CANDIDAT].astype(float).to_numpy()
    y_calib = df_vert_calib_sl["est_gagnant"].astype(int).to_numpy()
    lib.log(f"\n[5/8] Matrice candidate : {len(FEATURES_CANDIDAT)} variables "
            f"(2 baseline + {len(VARIABLES_NOUVELLES)} nouvelles) -- "
            f"VAL_FIT={X_fit.shape}, VAL_CALIB={X_calib.shape}.")

    # =========================================================================
    # Modele candidat POINTWISE (bloc complet des variables prevues).
    # =========================================================================
    lib.log("\n[6/8] Recherche d'hyperparametres (grille existante, evaluee sur VAL_CALIB)...")
    params, auc_grille = lib.entrainer_gbm_avec_grille(
        X_fit, y_fit, X_calib, y_calib, lib.GRILLE_GBM, "etape2-forme-adversaires-vert")
    modele_candidat = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params)
    modele_candidat.fit(X_fit, y_fit)

    proba_calib = modele_candidat.predict_proba(X_calib)[:, 1]
    auc_final = round(float(roc_auc_score(y_calib, proba_calib)), 4)
    logloss_final = round(float(log_loss(y_calib, np.clip(proba_calib, 1e-6, 1 - 1e-6))), 4)
    lib.log(f"\n  >>> AUC du modele candidat (bloc complet) sur VAL_CALIB (courses VERT, jamais vu a l'entrainement) : {auc_final}")
    lib.log(f"  >>> log-loss du modele candidat sur VAL_CALIB : {logloss_final}")
    lib.log(f"  (reference piste 6 etage pointwise generique, TOUTES courses : AUC~0.61 -- "
            f"reference etape 1 pairwise, VERT seul, memes 240 variables : AUC=0.6842)")

    # =========================================================================
    # Importance des variables (permutation, sur VAL_CALIB) -- "detail des
    # variables reellement utiles" demande par Dorian.
    # =========================================================================
    lib.log("\n[7/8] Importance des variables (permutation importance, AUC, VAL_CALIB, 20 repetitions)...")
    imp = permutation_importance(
        modele_candidat, X_calib, y_calib, scoring="roc_auc",
        n_repeats=20, random_state=lib.RANDOM_SEED)
    ordre = np.argsort(imp.importances_mean)[::-1]
    for i in ordre:
        nom = FEATURES_CANDIDAT[i]
        marqueur = " <-- nouvelle variable" if nom in VARIABLES_NOUVELLES else " (baseline B+genealogie)"
        lib.log(f"    {nom:42s} delta_AUC_moyen={round(float(imp.importances_mean[i]), 4):+.4f} "
                f"(+/-{round(float(imp.importances_std[i]), 4)}){marqueur}")

    # =========================================================================
    # Reclassement des courses VERT de VAL_CALIB, comparaison au brut.
    # =========================================================================
    lib.log("\n[8/8] Reclassement des 5 chevaux de chaque course VERT de VAL_CALIB...")
    proba_series = pd.Series(proba_calib, index=df_vert_calib_sl.index)
    rang_candidat_calib = proba_series.groupby(df_vert_calib_sl["course_id"]).rank(method="min", ascending=False)
    par_course_calib = construire_par_course_candidat(df_vert_calib, rang_candidat_calib)

    lib.log("\n" + "=" * 100)
    lib.log("=== COMPARAISON classement B+genealogie BRUT vs MODELE CANDIDAT (forme adversaires) -- VAL_CALIB, courses VERT ===")
    lib.log("=" * 100)
    resultats_finaux = {}
    for nom_bench, colonne_filtre in [("REEL", None), ("PROPRE", "est_benchmark_propre")]:
        sous = par_course_calib if colonne_filtre is None else par_course_calib[par_course_calib[colonne_filtre]]
        m_baseline = metriques(sous, "rang_final_baseline", f"BRUT B+genealogie ({nom_bench})")
        m_candidat = metriques(sous, "rang_final_candidat", f"CANDIDAT forme-adversaires ({nom_bench})")
        resultats_finaux[nom_bench] = (m_baseline, m_candidat)
        lib.log(f"\n  -- benchmark {nom_bench} -- n courses VERT (VAL_CALIB) = {m_baseline['n_courses']} --")
        lib.log(f"     REFERENCE (BRUT B+genealogie)      : top1={m_baseline['top1_pct']}% top3={m_baseline['top3_pct']}% top5={m_baseline['top5_pct']}%")
        lib.log(f"     NOUVEAU (CANDIDAT forme-adversaires): top1={m_candidat['top1_pct']}% top3={m_candidat['top3_pct']}% top5={m_candidat['top5_pct']}%")
        if m_baseline['top1_pct'] is not None:
            delta_top1 = round(m_candidat['top1_pct'] - m_baseline['top1_pct'], 1)
            delta_top3 = round(m_candidat['top3_pct'] - m_baseline['top3_pct'], 1)
            delta_top5 = round(m_candidat['top5_pct'] - m_baseline['top5_pct'], 1)
            lib.log(f"     DELTA : top1={delta_top1:+}pt top3={delta_top3:+}pt top5={delta_top5:+}pt "
                    f"(top5 doit rester strictement identique par construction, verification ci-dessus)")
        else:
            lib.log("     DELTA : n/a (aucune course dans ce sous-ensemble)")

    lib.log("\n" + "=" * 100)
    lib.log("=== BLOC RESUME -- format demande par Dorian ===")
    lib.log("=" * 100)
    for nom_bench in ("REEL", "PROPRE"):
        m_baseline, m_candidat = resultats_finaux[nom_bench]
        lib.log(f"\n  [{nom_bench}] REFERENCE -> NOUVEAU (n={m_baseline['n_courses']})")
        if m_baseline['top1_pct'] is not None:
            lib.log(f"  Top-1 : {m_baseline['top1_pct']}% -> {m_candidat['top1_pct']}%")
            lib.log(f"  Top-3 : {m_baseline['top3_pct']}% -> {m_candidat['top3_pct']}%")
            lib.log(f"  Top-5 : {m_baseline['top5_pct']}% -> {m_candidat['top5_pct']}%")
            lib.log(f"  Delta : top1={round(m_candidat['top1_pct']-m_baseline['top1_pct'],1):+}pt "
                    f"top3={round(m_candidat['top3_pct']-m_baseline['top3_pct'],1):+}pt "
                    f"top5={round(m_candidat['top5_pct']-m_baseline['top5_pct'],1):+}pt")
        else:
            lib.log("  n/a (aucune course)")

    lib.log("\n" + "=" * 100)
    lib.log("=== FIN ETAPE 2 -- rappel : reference deja validee sur VERT (VAL_CALIB de piste7_phase2) ~34-36% top1, ~73-74% top3, ~91-92% top5. ===")
    lib.log("=== Decision (a appliquer honnetement, sans cherry-picking) : rejeter si le Top-1 ne progresse pas, ===")
    lib.log("=== ou si le gain est faible/instable, ou si Top-3/Top-5 se degradent significativement. ===")
    lib.log("=" * 100)

    lib.log("\n===CSV_METRIQUES_ETAPE2_START===")
    lignes_csv = ["benchmark,label,n_courses,top1_pct,top3_pct,top5_pct"]
    for nom_bench, colonne_filtre in [("REEL", None), ("PROPRE", "est_benchmark_propre")]:
        sous = par_course_calib if colonne_filtre is None else par_course_calib[par_course_calib[colonne_filtre]]
        for colonne_rang, label in [("rang_final_baseline", "REFERENCE"), ("rang_final_candidat", "NOUVEAU")]:
            m = metriques(sous, colonne_rang, label)
            lignes_csv.append(f"{nom_bench},{label},{m['n_courses']},{m['top1_pct']},{m['top3_pct']},{m['top5_pct']}")
    for ligne in lignes_csv:
        lib.log(ligne)
    lib.log("===CSV_METRIQUES_ETAPE2_END===")

    lib.log(f"\n===AUC_CANDIDAT_VAL_CALIB=== {auc_final}")
    lib.log(f"===LOGLOSS_CANDIDAT_VAL_CALIB=== {logloss_final}")


if __name__ == "__main__":
    main()
