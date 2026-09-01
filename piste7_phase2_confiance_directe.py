# -*- coding: utf-8 -*-
"""
piste7_phase2_confiance_directe.py -- PHASE 2/2 de la piste 7, demandee par
Dorian le 01/09/2026 : abandon complet de l'etage d'affinage (piste 6, AUC
proche du hasard ~0.61) au profit d'un indicateur de confiance calcule
DIRECTEMENT sur les sorties du modele principal B+genealogie (score_geneal /
rang_geneal), sans aucun entrainement supplementaire.

Protocole valide par Dorian, dans cet ordre exact :
  1. comparer plusieurs indicateurs de confiance sur VALIDATION uniquement ;
  2. selectionner le meilleur indicateur selon VALIDATION uniquement ;
  3. figer les seuils VERT / ORANGE / ROUGE (valeurs concretes de
     l'indicateur retenu, pas des pourcentages) ;
  4. appliquer ces seuils TELS QUELS a TEST A puis TEST B (aucun
     ajustement, aucune nouvelle selection) ;
  5. le Top 5 reste le classement BRUT de B+genealogie (rang_geneal <= 5,
     inchange) ;
  6. l'amelioration du choix du n°1 (etape suivante, PAS traitee ici) est
     explicitement reportee a apres validation de cette etape.

ETAPE 1 (le seul modele entraine ici) : B+genealogie, EXACTEMENT les memes
hyperparametres/donnees que tous les runs precedents (piste 3/4/5/6) --
entraine sur TRAIN (70%), evalue sur VALIDATION pour l'early stopping
(convention deja utilisee dans tous les runs precedents).

INDICATEURS DE CONFIANCE (calcules sur l'ENSEMBLE du champ de chaque
course, pas seulement un sous-ensemble -- point cle par rapport a la piste
6 ou l'entropie etait calculee uniquement sur les 5 chevaux de la
short-list) :
  - entropie_champ : entropie de Shannon du softmax des scores
    B+genealogie sur TOUS les partants de la course (plus bas = plus
    confiant). Capture nativement l'effet taille de champ deja identifie
    (un petit champ produit mecaniquement une distribution moins uniforme).
  - somme_top3_proba : somme des 3 plus hautes pseudo-probabilites
    (indicateur deja valide en piste 5, reference a battre -- 76,1% de
    top3 a 20% de couverture sur VALIDATION lors de ce run precedent).
  - proba_pick1 : pseudo-probabilite (softmax) du pick #1.
  - ecart_1_2_normalise : (score#1 - score#2) / ecart-type des scores du
    champ -- marge brute du favori, normalisee pour comparer des champs de
    tailles differentes.

Ces 4 indicateurs sont TOUS des transformations deterministes des sorties
de B+genealogie -- aucun parametre n'est appris sur VALIDATION, seuls des
SEUILS (valeurs, pas des %) sont lus sur sa courbe de couverture.

DOUBLE BENCHMARK (reel/propre), calcule sur VALIDATION, TEST A et TEST B,
et rapporte SEPAREMENT pour chaque niveau de confiance sur TEST A et TEST B
(demande explicite de Dorian, 01/09/2026 -- pas seulement en reference
globale comme en piste 6).

TAUX DE PLACE : en plus de top1/top3/top5 (le rang donne par le modele au
VRAI GAGNANT), on rapporte le taux de reussite "place" du pick #1
(rang_geneal == 1) -- est-il arrive dans les places (position_arrivee <=
seuil PMU, deja disponible dans les variables point-in-time) -- calcule
avec lib.taux_reussite_top1, deja existante dans v3_lib.py.

Aucune cote/marche utilisee. Aucune course retiree du denominateur pour un
motif autre que les motifs d'exclusion du benchmark propre (voir
v3_lib.rapport_double_benchmark).
"""
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    from sklearn.metrics import roc_auc_score  # non utilise pour l'instant, garde pour coherence d'environnement
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

# Paliers utilises pour departager les indicateurs candidats (zone de
# couverture ou une strategie de selection par confiance a un interet
# pratique reel -- ni 100% [aucune selection], ni 10% [trop peu de
# courses jouables]). Regle fixee AVANT de regarder les resultats, pour
# eviter tout choix a posteriori qui favoriserait un indicateur au hasard.
PALIERS_DEPARTAGE = [20, 30, 40, 50, 60]


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


def calculer_indicateurs_confiance(d):
    """Reprend l'approche piste 5 (calculee sur l'ENSEMBLE du champ, pas
    une short-list) : pour chaque course, softmax des score_geneal sur
    TOUS les partants, puis derive plusieurs indicateurs. Ajoute en plus
    ecart_1_2_normalise (absent de piste 5), calcule a partir des memes
    quantites (aucun entrainement supplementaire)."""
    lignes = []
    for course_id, groupe in d.groupby("course_id", sort=False):
        scores = groupe["score_geneal"].values.astype(float)
        n = len(scores)
        ordre = np.argsort(-scores)
        scores_tries = scores[ordre]
        s1 = scores_tries[0]
        s2 = scores_tries[1] if n >= 2 else np.nan
        ecart_1_2 = s1 - s2 if n >= 2 else np.nan
        z = scores - scores.max()
        exp_z = np.exp(z)
        proba = exp_z / exp_z.sum()
        proba_tries = np.sort(proba)[::-1]
        proba_pick1 = proba_tries[0]
        somme_top3_proba = proba_tries[:3].sum()
        entropie_champ = float(-(proba * np.log(np.clip(proba, 1e-12, None))).sum())
        dispersion = float(np.std(scores))
        ecart_1_2_normalise = float(ecart_1_2 / dispersion) if dispersion > 0 and pd.notna(ecart_1_2) else np.nan
        for idx in groupe.index:
            lignes.append((idx, entropie_champ, somme_top3_proba, proba_pick1, ecart_1_2_normalise, dispersion))
    cols = ["_idx", "entropie_champ", "somme_top3_proba", "proba_pick1", "ecart_1_2_normalise", "dispersion"]
    df_ind = pd.DataFrame(lignes, columns=cols).set_index("_idx")
    return d.join(df_ind)


def construire_par_course(d):
    """Une ligne par course : rang_final = rang_geneal du VRAI gagnant
    (min en cas de dead-heat -- meme convention que piste 6), plus les
    indicateurs de confiance (identiques pour toutes les lignes d'une
    meme course, on prend la premiere occurrence)."""
    rang_gagnant = (
        d[d["est_gagnant"] == 1].groupby("course_id")["rang_geneal"].min()
    )
    par_course = d.drop_duplicates("course_id").set_index("course_id").copy()
    par_course["rang_final"] = par_course.index.map(rang_gagnant)
    return par_course.reset_index()


def courbe_couverture_avec_n(df_courses, indicateur, ascendant, label):
    """Courbe couverture -> top1/top3/top5 + n_courses EXPLICITE a chaque
    palier (jamais un % sans le n, regle permanente du projet)."""
    df_tri = df_courses.sort_values(indicateur, ascending=ascendant, na_position="last").reset_index(drop=True)
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


def appliquer_seuils_figes(df_courses, indicateur, ascendant, seuils, label):
    """seuils = {"vert": valeur, "orange": valeur}. Si ascendant=True,
    plus bas = plus confiant (ex. entropie) -> vert = indicateur <=
    seuils['vert']. Si ascendant=False, plus haut = plus confiant (ex.
    somme_top3_proba) -> vert = indicateur >= seuils['vert']."""
    d = df_courses.copy()
    if ascendant:
        d["niveau"] = np.where(
            d[indicateur] <= seuils["vert"], "VERT",
            np.where(d[indicateur] <= seuils["orange"], "ORANGE", "ROUGE"))
    else:
        d["niveau"] = np.where(
            d[indicateur] >= seuils["vert"], "VERT",
            np.where(d[indicateur] >= seuils["orange"], "ORANGE", "ROUGE"))
    return d


def rapport_par_niveau(df_courses_avec_niveau, df_lignes, label_bloc, label_benchmark):
    """Pour chaque niveau VERT/ORANGE/ROUGE : n, % du bloc (CE benchmark),
    top1/top3/top5 (rang du vrai gagnant), taux de place du pick #1
    (rang_geneal==1, via lib.taux_reussite_top1 -- deja existante)."""
    d = df_courses_avec_niveau
    n_total_bloc = len(d)
    lignes_avec_niveau = df_lignes.merge(d[["course_id", "niveau"]], on="course_id", how="inner")
    resultats = []
    for niveau in ["VERT", "ORANGE", "ROUGE"]:
        sous = d[d["niveau"] == niveau]
        n = len(sous)
        if n == 0:
            resultats.append({
                "bloc": label_bloc, "benchmark": label_benchmark, "niveau": niveau,
                "n_courses": 0, "pct_du_bloc": 0.0,
                "top1_pct": None, "top3_pct": None, "top5_pct": None, "taux_place_pick1_pct": None,
            })
            continue
        sous_lignes = lignes_avec_niveau[lignes_avec_niveau["niveau"] == niveau]
        n_place, n_reussis_place, pct_place = lib.taux_reussite_top1(sous_lignes, "rang_geneal", "cible_place")
        resultats.append({
            "bloc": label_bloc, "benchmark": label_benchmark, "niveau": niveau,
            "n_courses": n, "pct_du_bloc": round(100 * n / n_total_bloc, 1) if n_total_bloc else None,
            "top1_pct": round(100 * float((sous["rang_final"] == 1).mean()), 1),
            "top3_pct": round(100 * float((sous["rang_final"] <= 3).mean()), 1),
            "top5_pct": round(100 * float((sous["rang_final"] <= 5).mean()), 1),
            "taux_place_pick1_pct": pct_place,
        })
    return pd.DataFrame(resultats)


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


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn non installe.")
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError("lightgbm non installe.")

    lib.log("=" * 100)
    lib.log("PISTE 7 -- PHASE 2/2 -- CONFIANCE CALCULEE DIRECTEMENT SUR B+GENEALOGIE (SANS ETAGE D'AFFINAGE) -- 01/09/2026")
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
    # ETAPE 1 (le seul modele entraine) -- B+genealogie, AUCUN changement
    # =========================================================================
    lib.log("\n[ETAPE 1] Entrainement B+genealogie (memes hyperparametres que tous les runs precedents)...")
    groups_train = groupes_consecutifs(course_id_train)
    y_train_graded = np.where(y_train_gagnant == 1, 2, np.where(y_train_place == 1, 1, 0)).astype(int)
    groups_val = groupes_consecutifs(df_val["course_id"])
    modele = entrainer_lambdarank(
        X_train_v3_geneal, y_train_graded, groups_train, X_val_v3_geneal, checkpoint["y_val_gagnant"],
        groups_val, "B+genealogie")

    for nom_bloc, df_bloc, X_bloc in [("val", df_val, X_val_v3_geneal),
                                       ("testA", df_testA, X_testA_v3_geneal),
                                       ("testB", df_testB, X_testB_v3_geneal)]:
        df_bloc["score_geneal"] = modele.predict(X_bloc)
        df_bloc["rang_geneal"] = df_bloc.groupby("course_id")["score_geneal"].rank(method="min", ascending=False)

    exclusions = lib.charger_exclusions_benchmark()
    df_val = preparer_benchmark(df_val, exclusions, "VALIDATION -- piste 7")
    df_testA = preparer_benchmark(df_testA, exclusions, "TEST A -- piste 7")
    df_testB = preparer_benchmark(df_testB, exclusions, "TEST B -- piste 7")

    df_val_reel = df_val[df_val["est_benchmark_reel"]].copy()
    df_testA_reel = df_testA[df_testA["est_benchmark_reel"]].copy()
    df_testB_reel = df_testB[df_testB["est_benchmark_reel"]].copy()

    for nom, d in [("VALIDATION", df_val_reel), ("TEST A", df_testA_reel), ("TEST B", df_testB_reel)]:
        stats_rang, _ = lib.rang_distribution_gagnant(d, "rang_geneal")
        lib.log(f"   [B+genealogie seul, reference, {nom} (benchmark reel)] n={stats_rang['n_courses']} "
                f"top1={stats_rang['top1_pct']}% top3={stats_rang['cumul_top3_pct']}% "
                f"top5={stats_rang['cumul_top5_pct']}%")

    # =========================================================================
    # Top 5 = rang_geneal <= 5, BRUT, inchange (etape 4 du protocole). On
    # verifie juste, a titre diagnostique, la couverture du vrai gagnant.
    # =========================================================================
    for nom, d in [("VALIDATION", df_val_reel), ("TEST A", df_testA_reel), ("TEST B", df_testB_reel)]:
        d2 = d.copy()
        d2["dans_top5"] = d2["rang_geneal"] <= K_SHORTLIST
        gagnant_dans_top5 = d2[d2["est_gagnant"] == 1].groupby("course_id")["dans_top5"].any()
        n_courses = d2["course_id"].nunique()
        n_ok = int(gagnant_dans_top5.sum())
        lib.log(f"   [Top-5 brut B+genealogie, {nom}] {n_ok}/{n_courses} courses "
                f"({round(100*n_ok/n_courses,1)}%) ont le vrai gagnant dans le Top 5.")

    # =========================================================================
    # ETAPE 2 (protocole) -- indicateurs de confiance, calcules sur TOUT le
    # champ de chaque course (pas une short-list), pour TOUTES les courses.
    # =========================================================================
    lib.log("\n[ETAPE 2] Calcul des indicateurs de confiance (directement sur B+genealogie, tout le champ)...")
    df_val_reel = calculer_indicateurs_confiance(df_val_reel)
    df_testA_reel = calculer_indicateurs_confiance(df_testA_reel)
    df_testB_reel = calculer_indicateurs_confiance(df_testB_reel)
    # Le benchmark "propre" est un sous-ensemble du benchmark "reel" (memes
    # lignes, meme colonnes) -- les indicateurs sont deja calcules dessus.
    df_testA_propre = df_testA_reel[df_testA_reel["est_benchmark_propre"]].copy()
    df_testB_propre = df_testB_reel[df_testB_reel["est_benchmark_propre"]].copy()

    courses_val = construire_par_course(df_val_reel)
    courses_testA_reel = construire_par_course(df_testA_reel)
    courses_testB_reel = construire_par_course(df_testB_reel)
    courses_testA_propre = construire_par_course(df_testA_propre)
    courses_testB_propre = construire_par_course(df_testB_propre)

    # =========================================================================
    # ETAPE 1/2 DU PROTOCOLE -- comparer les indicateurs, UNIQUEMENT sur
    # VALIDATION, puis selectionner le meilleur UNIQUEMENT sur VALIDATION.
    # =========================================================================
    INDICATEURS = [
        ("entropie_champ", True, "entropie softmax (tout le champ)"),
        ("somme_top3_proba", False, "somme pseudo-proba top3 (reference piste 5)"),
        ("proba_pick1", False, "pseudo-proba du pick #1"),
        ("ecart_1_2_normalise", False, "ecart score #1-#2 normalise (/dispersion)"),
    ]

    lib.log("\n" + "=" * 100)
    lib.log("=== [1/6] COMPARAISON DES INDICATEURS DE CONFIANCE -- VALIDATION UNIQUEMENT ===")
    lib.log("=" * 100)
    courbes_val = {}
    for col, ascendant, label in INDICATEURS:
        courbe = courbe_couverture_avec_n(courses_val, col, ascendant, label)
        courbes_val[label] = (col, ascendant, courbe)
        lib.log(f"\n   -- {label} --")
        for _, row in courbe.iterrows():
            lib.log(f"      couverture_visee={row['couverture_visee_pct']:>3}% n={row['n_courses']:>5} "
                     f"seuil={row['seuil_indicateur']} -> top1={row['top1_pct']}% top3={row['top3_pct']}% "
                     f"top5={row['top5_pct']}%")

    lib.log("\n" + "=" * 100)
    lib.log(f"=== [2/6] SELECTION DU MEILLEUR INDICATEUR -- regle fixee a l'avance : "
            f"moyenne de top3_pct sur les paliers {PALIERS_DEPARTAGE} (VALIDATION uniquement) ===")
    lib.log("=" * 100)
    scores_selection = {}
    for label, (col, ascendant, courbe) in courbes_val.items():
        sous = courbe[courbe["couverture_visee_pct"].isin(PALIERS_DEPARTAGE)]
        moyenne_top3 = float(sous["top3_pct"].mean())
        scores_selection[label] = moyenne_top3
        lib.log(f"   {label:45s} moyenne top3 (paliers {PALIERS_DEPARTAGE}) = {round(moyenne_top3, 2)}%")
    meilleur_label = max(scores_selection, key=scores_selection.get)
    indicateur_retenu, ascendant_retenu, courbe_retenue = courbes_val[meilleur_label]
    lib.log(f"\n   INDICATEUR RETENU (VALIDATION uniquement) : {meilleur_label} "
            f"(colonne={indicateur_retenu}, {'plus bas = plus confiant' if ascendant_retenu else 'plus haut = plus confiant'})")

    # =========================================================================
    # ETAPE 3 DU PROTOCOLE -- figer les seuils VERT/ORANGE (valeurs de
    # l'indicateur retenu aux paliers 30%/60% de VALIDATION -- meme
    # convention que la piste 6, aucun nouveau palier choisi apres coup).
    # =========================================================================
    seuil_vert = float(courbe_retenue.loc[courbe_retenue["couverture_visee_pct"] == 30, "seuil_indicateur"].iloc[0])
    seuil_orange = float(courbe_retenue.loc[courbe_retenue["couverture_visee_pct"] == 60, "seuil_indicateur"].iloc[0])
    seuils_figes = {"vert": seuil_vert, "orange": seuil_orange}
    sens = "<=" if ascendant_retenu else ">="
    lib.log(f"\n[3/6] SEUILS FIGES (a partir de VALIDATION uniquement, jamais retouches ensuite) : "
            f"VERT si {indicateur_retenu} {sens} {seuil_vert}, ORANGE si {indicateur_retenu} {sens} {seuil_orange}, sinon ROUGE.")

    # =========================================================================
    # ETAPE 4 DU PROTOCOLE -- application SANS AUCUN AJUSTEMENT, TEST A puis
    # TEST B, benchmark reel ET benchmark propre rapportes separement.
    # =========================================================================
    lib.log("\n" + "=" * 100)
    lib.log("=== [4/6] APPLICATION DES SEUILS FIGES, SANS AUCUN AJUSTEMENT -- TEST A puis TEST B ===")
    lib.log("=" * 100)

    rapports = []
    for nom_test, courses_reel, lignes_reel, courses_propre, lignes_propre in [
        ("TEST A", courses_testA_reel, df_testA_reel, courses_testA_propre, df_testA_propre),
        ("TEST B", courses_testB_reel, df_testB_reel, courses_testB_propre, df_testB_propre),
    ]:
        courses_reel_niv = appliquer_seuils_figes(courses_reel, indicateur_retenu, ascendant_retenu, seuils_figes, nom_test)
        courses_propre_niv = appliquer_seuils_figes(courses_propre, indicateur_retenu, ascendant_retenu, seuils_figes, nom_test)

        rap_reel = rapport_par_niveau(courses_reel_niv, lignes_reel, nom_test, "REEL")
        rap_propre = rapport_par_niveau(courses_propre_niv, lignes_propre, nom_test, "PROPRE")
        rapports.append(rap_reel)
        rapports.append(rap_propre)

        for rap, nom_bench in [(rap_reel, "REEL"), (rap_propre, "PROPRE")]:
            lib.log(f"\n   -- [{nom_test} / benchmark {nom_bench}] --")
            for _, row in rap.iterrows():
                lib.log(f"      {row['niveau']:6s} n={row['n_courses']:>5} ({row['pct_du_bloc']}% du bloc) "
                         f"top1={row['top1_pct']}% top3={row['top3_pct']}% top5={row['top5_pct']}% "
                         f"taux_place_pick1={row['taux_place_pick1_pct']}%")

    df_rapports = pd.concat(rapports, ignore_index=True)

    # =========================================================================
    # Ventilation nb_partants / handicap, sur TEST A et TEST B (benchmark
    # reel, bout en bout) -- connaissance a preserver, demandee par Dorian.
    # =========================================================================
    lib.log("\n" + "=" * 100)
    lib.log("=== [5/6] VENTILATION nb_partants / handicap (TEST A + TEST B, benchmark reel) ===")
    lib.log("=" * 100)
    for nom, courses_bloc in [("TEST A", courses_testA_reel), ("TEST B", courses_testB_reel)]:
        cb = courses_bloc.copy()
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
    lib.log("=== [6/6] FIN -- resultats REELS de TEST A / TEST B, non ajustes. Seuils figes sur VALIDATION uniquement. ===")
    lib.log("=== Etape suivante (amelioration du choix du n°1) : PAS traitee dans ce run, sequencee apres validation. ===")
    lib.log("=" * 100)

    lib.log("\n===CSV_COMPARAISON_INDICATEURS_VALIDATION_START===")
    for label, (col, ascendant, courbe) in courbes_val.items():
        for ligne in courbe.to_csv(index=False).splitlines()[1:]:
            lib.log(ligne)
    lib.log("===CSV_COMPARAISON_INDICATEURS_VALIDATION_END===")

    lib.log("\n===CSV_RAPPORT_NIVEAUX_TESTAB_START===")
    for ligne in df_rapports.to_csv(index=False).splitlines():
        lib.log(ligne)
    lib.log("===CSV_RAPPORT_NIVEAUX_TESTAB_END===")


if __name__ == "__main__":
    main()
