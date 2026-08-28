# -*- coding: utf-8 -*-
"""
entrainer_v3_phase3_autopsie.py -- Autopsie du noyau dur (piste 3, demandee
par Dorian le 28/08/2026 apres le run genealogie) : B+genealogie est retenu
comme meilleur candidat actuel, mais les 71,3% de courses "toujours ratees"
(ni v3-gagnant ni B en pick #1) restent le vrai probleme. Cette phase 3 ne
teste AUCUNE nouvelle feature et ne lance AUCUN TEST A/B -- elle reutilise
le checkpoint DEJA CALCULE de la piste 3 (artefact cross-run, pas de
rechargement Supabase) pour :

  1. reproduire v3-gagnant, B et B+genealogie a l'identique (memes
     hyperparametres retenus, pas de nouvelle grille) ;
  2. comparer le pouvoir discriminant (d de Cohen gagnant vs non-gagnant)
     de TOUTES les familles de variables v3+genealogie, entre courses
     "faciles" (v3-gagnant OU B trouve le gagnant) et "noyau dur" (aucun
     des deux) -- pour reperer quelles familles perdent leur signal ;
  3. une analyse d'incertitude sur le noyau dur (ecarts de confiance,
     dispersion des scores, rang/probabilite du vrai gagnant) pour
     distinguer "courses reellement imprevisibles avec nos donnees" de
     "le modele a peut-etre le bon signal mais choisit mal".

Rien n'est ecrit en base. Rien n'est entraine en dehors de la reproduction
des 3 modeles deja valides.
"""
import gc
import itertools
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    pass

try:
    import lightgbm as lgb
except ImportError:
    pass

from variables_genealogie import COLONNES_GENEALOGIE

CHECKPOINT_PATH = "checkpoint_v3_phase1_genealogie.pkl"

# --- cartographie des familles de variables (demandee par Dorian). Couvre
# toutes les VARIABLES_NUMERIQUES (variables_config.py) + les 22 variables
# relatives au champ (v3_lib.VARIABLES_RELATIVES_CIBLES x rang/z) + les 20
# variables de genealogie (variables_genealogie.COLONNES_GENEALOGIE). ---
FAMILLES_BASE = {
    # forme recente (musique officielle + forme interne point-in-time)
    "musique_dernier": "forme_recente", "musique_moy3": "forme_recente", "musique_moy5": "forme_recente",
    "musique_tendance": "forme_recente", "musique_nb_incidents": "forme_recente", "musique_nb_courses_visibles": "forme_recente",
    "forme_n_courses_internes": "forme_recente", "forme_taux_victoire_carriere_interne": "forme_recente",
    "forme_taux_place_carriere_interne": "forme_recente", "forme_moy_position_5": "forme_recente",
    "forme_moy_position_10": "forme_recente", "forme_moy_position_20": "forme_recente",
    "forme_meilleure_position_5": "forme_recente", "forme_ecart_type_position_5": "forme_recente",
    "forme_tendance_5_vs_10": "forme_recente", "forme_nb_courses_disponibles": "forme_recente",
    "forme_a_au_moins_5": "forme_recente", "forme_a_au_moins_10": "forme_recente", "forme_a_au_moins_20": "forme_recente",
    # performances cheval (carriere globale + niveau du champ)
    "carriere_nb_courses": "perf_cheval", "carriere_taux_victoire": "perf_cheval", "carriere_taux_place": "perf_cheval",
    "gains_carriere": "perf_cheval", "gains_annee_encours": "perf_cheval", "gains_annee_precedente": "perf_cheval",
    "niveau_moyen_adversaires": "perf_cheval", "ecart_vs_niveau_moyen_champ": "perf_cheval", "rang_papier_taux_victoire": "perf_cheval",
    # jockey/driver
    "interne_jockey_nb": "jockey", "interne_jockey_taux_victoire": "jockey",
    "interne_jockey_taux_top2": "jockey", "interne_jockey_taux_top3": "jockey",
    # entraineur
    "interne_entraineur_nb": "entraineur", "interne_entraineur_taux_victoire": "entraineur",
    "interne_entraineur_taux_top2": "entraineur", "interne_entraineur_taux_top3": "entraineur",
    # couples
    "interne_jockey_cheval_nb": "couple_cheval_jockey", "interne_jockey_cheval_taux_victoire": "couple_cheval_jockey",
    "interne_jockey_cheval_taux_top2": "couple_cheval_jockey", "interne_jockey_cheval_taux_top3": "couple_cheval_jockey",
    "interne_entraineur_cheval_nb": "couple_cheval_entraineur", "interne_entraineur_cheval_taux_victoire": "couple_cheval_entraineur",
    "interne_entraineur_cheval_taux_top2": "couple_cheval_entraineur", "interne_entraineur_cheval_taux_top3": "couple_cheval_entraineur",
    "interne_jockey_entraineur_nb": "couple_jockey_entraineur", "interne_jockey_entraineur_taux_victoire": "couple_jockey_entraineur",
    "interne_jockey_entraineur_taux_top2": "couple_jockey_entraineur", "interne_jockey_entraineur_taux_top3": "couple_jockey_entraineur",
    # distance
    "distance_m": "distance", "distance_delta_vs_carriere": "distance",
    "interne_distance_nb": "distance", "interne_distance_taux_victoire": "distance",
    "interne_distance_taux_top2": "distance", "interne_distance_taux_top3": "distance",
    # terrain
    "terrain_valeur_penetrometre": "terrain", "terrain_score_ordinal": "terrain",
    "interne_terrain_nb": "terrain", "interne_terrain_taux_victoire": "terrain",
    "interne_terrain_taux_top2": "terrain", "interne_terrain_taux_top3": "terrain",
    # corde
    "place_corde": "corde",
    "interne_corde_relative_nb": "corde", "interne_corde_relative_taux_victoire": "corde",
    "interne_corde_relative_taux_top2": "corde", "interne_corde_relative_taux_top3": "corde",
    "biais_corde_hippo_distance_nb": "corde", "biais_corde_hippo_distance_taux_victoire": "corde",
    "biais_corde_hippo_distance_taux_top2": "corde", "biais_corde_hippo_distance_taux_top3": "corde",
    # poids
    "handicap_poids": "poids", "poids_condition_monte": "poids", "poids_delta_vs_carriere": "poids",
    # age
    "age": "age",
    # categorie / contexte course
    "montant_allocation": "categorie_course", "allocation_delta_vs_carriere": "categorie_course",
    # equipement
    "oeilleres_presence": "equipement", "deferre_present": "equipement",
    # repos
    "jours_repos": "repos",
    # historiques hippodrome / categorie / proprietaire / eleveur
    "interne_hippo_nb": "historique_hippo", "interne_hippo_taux_victoire": "historique_hippo",
    "interne_hippo_taux_top2": "historique_hippo", "interne_hippo_taux_top3": "historique_hippo",
    "interne_categorie_nb": "historique_categorie", "interne_categorie_taux_victoire": "historique_categorie",
    "interne_categorie_taux_top2": "historique_categorie", "interne_categorie_taux_top3": "historique_categorie",
    "interne_proprietaire_nb": "proprietaire_eleveur", "interne_proprietaire_taux_victoire": "proprietaire_eleveur",
    "interne_proprietaire_taux_top2": "proprietaire_eleveur", "interne_proprietaire_taux_top3": "proprietaire_eleveur",
    "interne_eleveur_nb": "proprietaire_eleveur", "interne_eleveur_taux_victoire": "proprietaire_eleveur",
    "interne_eleveur_taux_top2": "proprietaire_eleveur", "interne_eleveur_taux_top3": "proprietaire_eleveur",
    # divers
    "entraine_a_letranger": "contexte_pays",
    "meteo_temperature": "meteo", "meteo_force_vent": "meteo",
}
for _v in COLONNES_GENEALOGIE:
    FAMILLES_BASE[_v] = "genealogie"
for _v in lib.VARIABLES_RELATIVES_CIBLES:
    FAMILLES_BASE[f"{_v}_rang_course"] = "relatives_course"
    FAMILLES_BASE[f"{_v}_z_course"] = "relatives_course"


def cohen_d(df_sub, var):
    if var not in df_sub.columns:
        return np.nan
    winners = df_sub.loc[df_sub["position_arrivee"] == 1, var]
    non = df_sub.loc[df_sub["position_arrivee"] != 1, var]
    m_w, m_n = winners.mean(), non.mean()
    s_w, s_n = winners.std(), non.std()
    n_w, n_n = int(winners.notna().sum()), int(non.notna().sum())
    if n_w > 1 and n_n > 1 and pd.notna(s_w) and pd.notna(s_n):
        pooled = np.sqrt(((n_w - 1) * s_w ** 2 + (n_n - 1) * s_n ** 2) / max(n_w + n_n - 2, 1))
        return (m_w - m_n) / pooled if pooled and pooled > 0 else np.nan
    return np.nan


def groupes_consecutifs(course_id_iterable):
    return [len(list(g)) for _, g in itertools.groupby(list(course_id_iterable))]


def softmax_par_course(df, colonne_score):
    d = df[["course_id", colonne_score]].copy()
    d["_max"] = d.groupby("course_id")[colonne_score].transform("max")
    d["_exp"] = np.exp(d[colonne_score] - d["_max"])
    d["_somme"] = d.groupby("course_id")["_exp"].transform("sum")
    return d["_exp"] / d["_somme"]


def main():
    lib.log("=" * 100)
    lib.log("PISTE 3 -- PHASE 3 -- AUTOPSIE DU NOYAU DUR (71,3% de courses toujours ratees) -- 28/08/2026")
    lib.log("=" * 100)

    lib.log("\n[1/6] Chargement du checkpoint (reutilise, aucun rechargement Supabase)...")
    with open(CHECKPOINT_PATH, "rb") as f:
        ck = pickle.load(f)
    X_train_v3, X_val_v3 = ck["X_train_v3"], ck["X_val_v3"]
    X_train_v3_geneal, X_val_v3_geneal = ck["X_train_v3_geneal"], ck["X_val_v3_geneal"]
    y_train_gagnant = ck["y_train_gagnant"]
    df_val = ck["df_val"].reset_index(drop=True).copy()
    course_id_train = ck["course_id_train"]
    lib.log(f"   X_train_v3={X_train_v3.shape} X_train_v3_geneal={X_train_v3_geneal.shape} df_val={df_val.shape}")

    lib.log("\n[2/6] Reproduction des 3 modeles deja valides (memes hyperparametres/grille que les runs de reference)...")
    y_val_gagnant = ck["y_val_gagnant"]
    params_baseline, _ = lib.entrainer_gbm_avec_grille(
        X_train_v3, y_train_gagnant, X_val_v3, y_val_gagnant, lib.GRILLE_GBM, "v3-gagnant-baseline")
    m_v3 = HistGradientBoostingClassifier(random_state=lib.RANDOM_SEED, **params_baseline)
    m_v3.fit(X_train_v3, y_train_gagnant)
    proba_v3 = m_v3.predict_proba(X_val_v3)[:, 1]
    del m_v3
    gc.collect()

    groupes_train = groupes_consecutifs(course_id_train)
    groupes_val = groupes_consecutifs(df_val["course_id"])
    assert sum(groupes_train) == len(X_train_v3) and sum(groupes_val) == len(X_val_v3)
    y_train_graded = np.where(ck["y_train_gagnant"] == 1, 2, np.where(ck["y_train_place"] == 1, 1, 0))
    params_lgb = dict(
        objective="lambdarank", metric="ndcg", boosting_type="gbdt",
        num_leaves=31, max_depth=5, learning_rate=0.05, min_child_samples=40,
        reg_lambda=1.0, n_estimators=500, random_state=lib.RANDOM_SEED, verbosity=-1,
    )
    m_B = lgb.LGBMRanker(**params_lgb, eval_at=[1, 3, 5])
    m_B.fit(
        X_train_v3, y_train_graded, group=groupes_train,
        eval_set=[(X_val_v3, y_val_gagnant)], eval_group=[groupes_val], eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)],
    )
    score_B = m_B.predict(X_val_v3)
    del m_B
    gc.collect()

    m_geneal = lgb.LGBMRanker(**params_lgb, eval_at=[1, 3, 5])
    m_geneal.fit(
        X_train_v3_geneal, y_train_graded, group=groupes_train,
        eval_set=[(X_val_v3_geneal, y_val_gagnant)], eval_group=[groupes_val], eval_at=[1, 3, 5],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)],
    )
    score_geneal = m_geneal.predict(X_val_v3_geneal)
    del m_geneal
    gc.collect()

    df_val["proba_v3gagnant"] = proba_v3
    df_val["score_B"] = score_B
    df_val["score_geneal"] = score_geneal
    df_val["rang_v3gagnant"] = df_val.groupby("course_id")["proba_v3gagnant"].rank(method="min", ascending=False)
    df_val["rang_B"] = df_val.groupby("course_id")["score_B"].rank(method="min", ascending=False)
    df_val["rang_geneal"] = df_val.groupby("course_id")["score_geneal"].rank(method="min", ascending=False)

    lib.log("\n[3/6] Definition du noyau dur (v3-gagnant ET B manquent le gagnant, memes regles que le rapport piste 3)...")
    gagnants = df_val[df_val["position_arrivee"] == 1].copy()
    gagnants["noyau_dur"] = (gagnants["rang_v3gagnant"] != 1) & (gagnants["rang_B"] != 1)
    n_dur = int(gagnants["noyau_dur"].sum())
    n_total = len(gagnants)
    lib.log(f"   Noyau dur : {n_dur}/{n_total} courses = {round(100*n_dur/n_total,1)}%")
    courses_dur = set(gagnants.loc[gagnants["noyau_dur"], "course_id"])
    courses_faciles = set(gagnants.loc[~gagnants["noyau_dur"], "course_id"])
    df_val["noyau_dur"] = df_val["course_id"].isin(courses_dur)

    lib.log("\n[4/6] Analyse d'incertitude sur le noyau dur (scores de B+genealogie, softmax intra-course)...")
    df_val["proba_softmax_geneal"] = softmax_par_course(df_val, "score_geneal")

    def stats_par_course(d):
        d = d.sort_values("proba_softmax_geneal", ascending=False).reset_index(drop=True)
        n = len(d)
        p1 = d.loc[0, "proba_softmax_geneal"]
        p2 = d.loc[1, "proba_softmax_geneal"] if n > 1 else np.nan
        p5 = d.loc[4, "proba_softmax_geneal"] if n > 4 else np.nan
        s1 = d.loc[0, "score_geneal"]
        s2 = d.loc[1, "score_geneal"] if n > 1 else np.nan
        std_scores = d["score_geneal"].std()
        n_proches = int((d["proba_softmax_geneal"] >= 0.5 * p1).sum())
        gagnant = d[d["position_arrivee"] == 1]
        rang_vrai = float(gagnant["rang_geneal"].iloc[0]) if len(gagnant) else np.nan
        proba_vrai = float(gagnant["proba_softmax_geneal"].iloc[0]) if len(gagnant) else np.nan
        return pd.Series({
            "n_partants": n,
            "gap_score_top1_top2": (s1 - s2) if pd.notna(s2) else np.nan,
            "gap_proba_top1_top2": (p1 - p2) if pd.notna(p2) else np.nan,
            "gap_proba_top1_top5": (p1 - p5) if pd.notna(p5) else np.nan,
            "std_scores": std_scores,
            "n_chevaux_proches_du_pick": n_proches,
            "confiance_pick_modele": p1,
            "rang_vrai_gagnant": rang_vrai,
            "proba_vrai_gagnant": proba_vrai,
        })

    par_course = df_val.groupby("course_id", sort=False).apply(stats_par_course).reset_index()
    par_course["noyau_dur"] = par_course["course_id"].isin(courses_dur)

    lib.log("\n   -- Comparaison noyau dur vs courses faciles (moyennes/medianes) --")
    for grp, label in [(True, "NOYAU DUR"), (False, "FACILES")]:
        sous = par_course[par_course["noyau_dur"] == grp]
        lib.log(f"   {label} (n={len(sous)}) :")
        lib.log(f"      gap score #1-#2 (moy)     = {round(sous['gap_score_top1_top2'].mean(), 4)}")
        lib.log(f"      gap proba #1-#2 (moy)     = {round(sous['gap_proba_top1_top2'].mean(), 4)}")
        lib.log(f"      gap proba #1-#5 (moy)     = {round(sous['gap_proba_top1_top5'].mean(), 4)}")
        lib.log(f"      dispersion scores (moy)   = {round(sous['std_scores'].mean(), 4)}")
        lib.log(f"      chevaux proches du pick (moy) = {round(sous['n_chevaux_proches_du_pick'].mean(), 2)}")
        lib.log(f"      confiance du pick modele (moy/mediane) = {round(sous['confiance_pick_modele'].mean(), 4)} / {round(sous['confiance_pick_modele'].median(), 4)}")
        lib.log(f"      rang du vrai gagnant (moy/mediane)     = {round(sous['rang_vrai_gagnant'].mean(), 2)} / {round(sous['rang_vrai_gagnant'].median(), 1)}")
        lib.log(f"      proba du vrai gagnant (moy/mediane)    = {round(sous['proba_vrai_gagnant'].mean(), 4)} / {round(sous['proba_vrai_gagnant'].median(), 4)}")

    lib.log("\n   -- Sous-segmentation du noyau dur par confiance du pick du modele (bins) --")
    dur = par_course[par_course["noyau_dur"]].copy()
    dur["bin_confiance"] = pd.cut(
        dur["confiance_pick_modele"], bins=[0, 0.15, 0.25, 0.35, 1.01],
        labels=["<15% (tres dispute)", "15-25%", "25-35%", ">35% (pick confiant mais faux)"],
    )
    for b, sous in dur.groupby("bin_confiance", observed=True):
        if len(sous) == 0:
            continue
        lib.log(f"   {str(b):35s} n={len(sous):5d} rang_vrai_gagnant(moy)={round(sous['rang_vrai_gagnant'].mean(),2):>5} "
                f"proba_vrai_gagnant(moy)={round(sous['proba_vrai_gagnant'].mean(),4)}")

    lib.log("\n[5/6] Pouvoir discriminant des familles de variables (d de Cohen gagnant vs non-gagnant, faciles vs noyau dur)...")
    df_faciles = df_val[df_val["course_id"].isin(courses_faciles)]
    df_dur = df_val[df_val["course_id"].isin(courses_dur)]
    lignes = []
    for var, famille in FAMILLES_BASE.items():
        d_f = cohen_d(df_faciles, var)
        d_d = cohen_d(df_dur, var)
        if pd.isna(d_f) and pd.isna(d_d):
            continue
        perte = (abs(d_f) - abs(d_d)) if pd.notna(d_f) and pd.notna(d_d) else np.nan
        lignes.append({"variable": var, "famille": famille, "d_faciles": d_f, "d_noyau_dur": d_d, "perte_pouvoir": perte})
    tbl = pd.DataFrame(lignes)

    lib.log("\n   -- Par famille (moyenne de |d| faciles vs noyau dur, perte = |d_faciles| - |d_noyau_dur|) --")
    par_famille = tbl.groupby("famille").agg(
        n_variables=("variable", "count"),
        d_faciles_moy_abs=("d_faciles", lambda s: round(float(s.abs().mean()), 3)),
        d_noyau_dur_moy_abs=("d_noyau_dur", lambda s: round(float(s.abs().mean()), 3)),
        perte_moy=("perte_pouvoir", lambda s: round(float(s.mean()), 3)),
    ).reset_index().sort_values("perte_moy", ascending=False)
    for _, row in par_famille.iterrows():
        lib.log(f"   {row['famille']:25s} n_var={row['n_variables']:3d} |d|_faciles={row['d_faciles_moy_abs']:+.3f} "
                f"|d|_noyau_dur={row['d_noyau_dur_moy_abs']:+.3f} perte={row['perte_moy']:+.3f}")

    lib.log("\n   -- Top 20 variables individuelles par perte de pouvoir discriminant --")
    top_perte = tbl.reindex(tbl["perte_pouvoir"].abs().sort_values(ascending=False, na_position="last").index).head(20)
    for _, row in top_perte.iterrows():
        lib.log(f"   {row['variable']:45s} [{row['famille']:20s}] d_faciles={row['d_faciles']:+.3f} "
                f"d_noyau_dur={row['d_noyau_dur']:+.3f} perte={row['perte_pouvoir']:+.3f}")

    lib.log("\n   -- Variables qui GARDENT du pouvoir discriminant dans le noyau dur (|d_noyau_dur| >= 0.2, triees desc) --")
    garde = tbl[tbl["d_noyau_dur"].abs() >= 0.2].reindex(
        tbl["d_noyau_dur"].abs().sort_values(ascending=False, na_position="last").index
    ).dropna(subset=["d_noyau_dur"])
    garde = garde[garde["d_noyau_dur"].abs() >= 0.2]
    if len(garde):
        for _, row in garde.head(15).iterrows():
            lib.log(f"   {row['variable']:45s} [{row['famille']:20s}] d_noyau_dur={row['d_noyau_dur']:+.3f} (d_faciles={row['d_faciles']:+.3f})")
    else:
        lib.log("   Aucune variable ne conserve |d| >= 0.2 dans le noyau dur -- signal quasi plat sur TOUTES les familles actuelles.")

    lib.log("\n[6/6] Recapitulatif des taux top-1 (pour verification de coherence avec le rapport piste 3) :")
    for col, label in [("rang_v3gagnant", "v3-gagnant"), ("rang_B", "B"), ("rang_geneal", "B+genealogie")]:
        n_top1 = int((gagnants[col] == 1).sum())
        lib.log(f"   {label:15s} top1 = {n_top1}/{n_total} = {round(100*n_top1/n_total,1)}%")

    lib.log("\n" + "=" * 100)
    lib.log("=== RESUME -- autopsie descriptive uniquement, aucune feature testee, aucun TEST A ni TEST B lance. ===")
    lib.log("=" * 100)


if __name__ == "__main__":
    main()
