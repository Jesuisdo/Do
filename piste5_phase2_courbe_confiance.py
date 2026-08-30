# -*- coding: utf-8 -*-
"""
piste5_phase2_courbe_confiance.py -- PHASE 2/2 de la piste 5 (selection par
confiance), demandee par Dorian le 30/08/2026 apres avoir constate que les
micro-optimisations de features stagnent autour de 55-65% de Top-3.

Nouvelle direction : au lieu de forcer une prediction sur 100% des courses,
on mesure si B+genealogie (notre meilleur modele a ce jour) peut atteindre
un Top-3 bien plus eleve (cible indicative : ~80%) en ne jouant que les
courses ou il est reellement confiant.

Reutilise a l'IDENTIQUE le checkpoint produit par
entrainer_v3_phase1_genealogie.py (AUCUNE modification des variables/split
-- memes 109+22 variables v3, memes 20 variables de genealogie point-in-time,
meme decoupage chronologique 70/15/15) et reentraine B+genealogie avec les
MEMES hyperparametres que piste 3/piste 4 (aucun reglage nouveau).

Pour chaque course de VALIDATION (benchmark reel), calcule plusieurs
indicateurs de confiance derives des scores B+genealogie UNIQUEMENT
(indicateurs "accord entre modeles" et "accord marche H-15" : hors perimetre
de ce run, traites separement piste 5 volet 4/5 -- necessitent d'autres
modeles / les cotes, non disponibles sur le grand historique) :
  - ecart_1_2 : score du pick #1 moins score du pick #2 (brut, meme
    convention que gap_confiance utilise piste 4) ;
  - ecart_1_3 : score du pick #1 moins score du pick #3 ;
  - proba_pick1 : softmax des scores intra-course (pseudo-probabilite,
    PAS une probabilite calibree -- LightGBM lambdarank ne produit pas de
    probabilites, le softmax est une transformation usuelle pour comparer
    des scores de ranking entre courses de tailles differentes) ;
  - somme_top3_proba : somme des 3 pseudo-probabilites les plus hautes ;
  - entropie : entropie de Shannon des pseudo-probabilites intra-course
    (plus bas = plus confiant) ;
  - dispersion : ecart-type des scores bruts intra-course ;
  - n_proches : nombre de chevaux dont le score est a moins de 0.5 ecart-type
    du score du pick #1 (plus bas = plus confiant, champ "clair").

Puis, pour CHAQUE indicateur, construit la courbe couverture/precision :
en selectionnant les X% de courses les plus confiantes (X = 100/90/75/60/
50/40/30/20), calcule Top-1/Top-2/Top-3/Top-5/precision sur ce sous-ensemble.

AJOUT (30/08/2026, demande de Dorian) : pour eviter de gonfler artificiellement
le Top-3 en jouant systematiquement 3 chevaux, on ajoute une regle de SELECTION
ADAPTATIVE DU NOMBRE DE CHEVAUX par course (k=1, 2 ou 3), fondee uniquement sur
la structure de confiance DE CETTE COURSE (pas de seuil global ajuste/optimise) :
  - k=1 si ecart_1_2 >= 0.75 * dispersion (le pick #1 est nettement detache) ;
  - sinon k=2 si ecart_1_3 >= 0.75 * dispersion (le pick #3 est nettement
    distance, meme si #1 et #2 sont proches) ;
  - sinon k=3.
Le coefficient 0.75 est un choix simple et unique (pas de grille, pas de
recherche de seuil) -- l'objectif de ce run est de MONTRER si une regle
adaptative raisonnable degage un vrai compromis nombre-de-chevaux/Top-3, pas
de maximiser la metrique. On rapporte, pour chaque palier de couverture de
courses : le nombre moyen de chevaux retenus, et le taux de reussite (le
gagnant est-il dans les k chevaux retenus) -- a comparer aux strategies FIXES
(toujours 1, toujours 2, toujours 3 chevaux).

AUCUN TEST A/B lance ici (validation uniquement). Point-in-time strict
herite du checkpoint phase 1 (aucune information post-depart).
"""
import itertools
import pickle

import numpy as np
import pandas as pd

import v3_lib as lib

try:
    import lightgbm as lgb
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    LIGHTGBM_DISPONIBLE = False

CHECKPOINT_PATH = "checkpoint_v3_phase1_genealogie.pkl"
PALIERS_COUVERTURE = [100, 90, 75, 60, 50, 40, 30, 20]


def groupes_consecutifs(course_id_iterable):
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


def calculer_indicateurs_confiance(d):
    """Ajoute les colonnes d'indicateurs de confiance, calculees PAR COURSE
    sur les scores B+genealogie (df deja restreint au benchmark reel)."""
    lignes = []
    for course_id, groupe in d.groupby("course_id", sort=False):
        scores = groupe["score_geneal"].values.astype(float)
        n = len(scores)
        ordre = np.argsort(-scores)
        scores_tries = scores[ordre]
        s1 = scores_tries[0]
        s2 = scores_tries[1] if n >= 2 else np.nan
        s3 = scores_tries[2] if n >= 3 else np.nan
        ecart_1_2 = s1 - s2 if n >= 2 else np.nan
        ecart_1_3 = s1 - s3 if n >= 3 else np.nan
        # softmax intra-course (stabilise)
        z = scores - scores.max()
        exp_z = np.exp(z)
        proba = exp_z / exp_z.sum()
        proba_tries = np.sort(proba)[::-1]
        proba_pick1 = proba_tries[0]
        somme_top3_proba = proba_tries[:3].sum()
        entropie = float(-(proba * np.log(np.clip(proba, 1e-12, None))).sum())
        dispersion = float(np.std(scores))
        seuil_proche = s1 - 0.5 * dispersion if dispersion > 0 else s1
        n_proches = int((scores >= seuil_proche).sum())
        for idx in groupe.index:
            lignes.append((idx, ecart_1_2, ecart_1_3, proba_pick1, somme_top3_proba,
                            entropie, dispersion, n_proches))
    cols = ["_idx", "ecart_1_2", "ecart_1_3", "proba_pick1", "somme_top3_proba",
            "entropie", "dispersion", "n_proches"]
    df_ind = pd.DataFrame(lignes, columns=cols).set_index("_idx")
    return d.join(df_ind)


def courbe_couverture(df_gagnants, indicateur, ascendant, label):
    """df_gagnants : une ligne par course (le gagnant), avec rang_geneal et
    l'indicateur de confiance. ascendant=True si une valeur PLUS BASSE de
    l'indicateur signifie PLUS confiant (ex: entropie, n_proches)."""
    df_tri = df_gagnants.sort_values(indicateur, ascending=ascendant, na_position="last").reset_index(drop=True)
    n_total = len(df_tri)
    lignes = []
    for pct in PALIERS_COUVERTURE:
        n_sel = max(1, int(round(n_total * pct / 100)))
        sous = df_tri.iloc[:n_sel]
        n = len(sous)
        top1 = round(100 * float((sous["rang_geneal"] == 1).mean()), 1)
        top2 = round(100 * float((sous["rang_geneal"] <= 2).mean()), 1)
        top3 = round(100 * float((sous["rang_geneal"] <= 3).mean()), 1)
        top5 = round(100 * float((sous["rang_geneal"] <= 5).mean()), 1)
        lignes.append({"indicateur": label, "couverture_pct": pct, "n_courses": n,
                        "top1_pct": top1, "top2_pct": top2, "top3_pct": top3, "top5_pct": top5})
    return pd.DataFrame(lignes)


def assigner_k_adaptatif(df):
    """k=1/2/3 par course, fonde uniquement sur ecart_1_2 / ecart_1_3 /
    dispersion DE CETTE COURSE (voir docstring module -- regle unique, non
    ajustee)."""
    disp = df["dispersion"].replace(0, np.nan)
    seuil = 0.75 * disp
    k = np.full(len(df), 3, dtype=int)
    k = np.where(df["ecart_1_3"] >= seuil, 2, k)
    k = np.where(df["ecart_1_2"] >= seuil, 1, k)
    # dispersion nulle ou n_partants<3 -> pas d'ecart_1_3 calculable, on
    # retombe sur 1 si ecart_1_2 dispo et grand, sinon 3 par prudence.
    k = np.where(disp.isna().values & df["ecart_1_2"].notna().values & (df["ecart_1_2"].values > 0), 1, k)
    return k.astype(int)


def courbe_couverture_adaptative(df_gagnants, indicateur, ascendant, label):
    """Meme tri par indicateur de confiance de COURSE, mais le nombre de
    chevaux joues (k) varie par course selon assigner_k_adaptatif. Compare
    aussi aux strategies fixes k=1/2/3 sur le MEME sous-ensemble de courses."""
    df_tri = df_gagnants.sort_values(indicateur, ascending=ascendant, na_position="last").reset_index(drop=True)
    n_total = len(df_tri)
    lignes = []
    for pct in PALIERS_COUVERTURE:
        n_sel = max(1, int(round(n_total * pct / 100)))
        sous = df_tri.iloc[:n_sel].copy()
        n = len(sous)
        k_adapt = assigner_k_adaptatif(sous)
        reussite_adapt = (sous["rang_geneal"].values <= k_adapt)
        lignes.append({
            "indicateur": label, "couverture_pct": pct, "n_courses": n,
            "k_moyen_adaptatif": round(float(k_adapt.mean()), 2),
            "pct_k1_adaptatif": round(100 * float((k_adapt == 1).mean()), 1),
            "pct_k2_adaptatif": round(100 * float((k_adapt == 2).mean()), 1),
            "pct_k3_adaptatif": round(100 * float((k_adapt == 3).mean()), 1),
            "top_adaptatif_pct": round(100 * float(reussite_adapt.mean()), 1),
            "top1_fixe_pct": round(100 * float((sous["rang_geneal"] == 1).mean()), 1),
            "top2_fixe_pct": round(100 * float((sous["rang_geneal"] <= 2).mean()), 1),
            "top3_fixe_pct": round(100 * float((sous["rang_geneal"] <= 3).mean()), 1),
        })
    return pd.DataFrame(lignes)


def main():
    if not lib.SKLEARN_DISPONIBLE:
        raise RuntimeError("scikit-learn non installe.")
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError("lightgbm non installe.")

    lib.log("=" * 100)
    lib.log("PISTE 5 -- PHASE 2/2 -- STRATEGIE DE SELECTION PAR CONFIANCE (B+genealogie) -- 30/08/2026")
    lib.log("=" * 100)

    with open(CHECKPOINT_PATH, "rb") as f:
        checkpoint = pickle.load(f)
    X_train_v3_geneal = checkpoint["X_train_v3_geneal"]
    X_val_v3_geneal = checkpoint["X_val_v3_geneal"]
    y_train_place = checkpoint["y_train_place"]
    y_train_gagnant = checkpoint["y_train_gagnant"]
    y_val_gagnant = checkpoint["y_val_gagnant"]
    df_val = checkpoint["df_val"].reset_index(drop=True)
    course_id_train = checkpoint["course_id_train"]

    lib.log(f"\n   Checkpoint charge : X_train_v3_geneal={X_train_v3_geneal.shape}, "
            f"X_val_v3_geneal={X_val_v3_geneal.shape}, df_val={df_val.shape}. "
            f"Memes features/split que piste 3 -- aucune reconstruction.")

    groups_train = groupes_consecutifs(course_id_train)
    groups_val = groupes_consecutifs(df_val["course_id"])
    assert sum(groups_train) == len(X_train_v3_geneal) and sum(groups_val) == len(X_val_v3_geneal)

    lib.log("\n[1/3] Entrainement B+genealogie (memes hyperparametres que piste 3/piste 4, AUCUN reglage)...")
    y_train_graded = np.where(y_train_gagnant == 1, 2, np.where(y_train_place == 1, 1, 0)).astype(int)
    modele_geneal = entrainer_lambdarank(
        X_train_v3_geneal, y_train_graded, groups_train, X_val_v3_geneal, y_val_gagnant, groups_val, "B+genealogie")
    df_val["score_geneal"] = modele_geneal.predict(X_val_v3_geneal)
    df_val["rang_geneal"] = df_val.groupby("course_id")["score_geneal"].rank(method="min", ascending=False)

    exclusions = lib.charger_exclusions_benchmark()
    df_val = lib.appliquer_benchmarks(df_val, exclusions)
    lib.rapport_double_benchmark(df_val, label="VALIDATION -- piste 5, selection par confiance")
    df_val_reel = df_val[df_val["est_benchmark_reel"]].copy()

    stats_rang, _ = lib.rang_distribution_gagnant(df_val_reel, "rang_geneal")
    lib.log(f"\n   Reference B+genealogie, TOUTES courses (benchmark reel) : "
            f"n={stats_rang['n_courses']} top1={stats_rang['top1_pct']}% "
            f"top3={stats_rang['cumul_top3_pct']}% top5={stats_rang['cumul_top5_pct']}%")

    lib.log("\n[2/3] Calcul des indicateurs de confiance par course (sur B+genealogie uniquement)...")
    df_val_reel = calculer_indicateurs_confiance(df_val_reel)
    df_val_reel["est_handicap"] = df_val_reel["categorie_particularite"].fillna("").str.contains("HANDICAP")

    gagnants = df_val_reel[df_val_reel["est_gagnant"] == 1].copy()
    lib.log(f"   {len(gagnants)} courses avec gagnant identifie (une ligne par course pour la courbe).")

    lib.log("\n[3/3] Courbes couverture -> Top1/Top3/Top5, par indicateur de confiance...")
    indicateurs = [
        ("ecart_1_2", False, "ecart score #1-#2"),
        ("ecart_1_3", False, "ecart score #1-#3"),
        ("proba_pick1", False, "pseudo-proba pick #1 (softmax)"),
        ("somme_top3_proba", False, "somme pseudo-proba top3"),
        ("entropie", True, "entropie (plus bas = plus confiant)"),
        ("dispersion", False, "dispersion (ecart-type) des scores"),
        ("n_proches", True, "n chevaux proches du pick #1 (plus bas = plus confiant)"),
    ]

    toutes_courbes = []
    for col, ascendant, label in indicateurs:
        courbe = courbe_couverture(gagnants, col, ascendant, label)
        toutes_courbes.append(courbe)
        lib.log(f"\n   -- Indicateur : {label} --")
        for _, row in courbe.iterrows():
            lib.log(f"      couverture={row['couverture_pct']:>3}% n={row['n_courses']:>5} "
                     f"top1={row['top1_pct']:>5}% top2={row['top2_pct']:>5}% top3={row['top3_pct']:>5}% "
                     f"top5={row['top5_pct']:>5}%")

    df_courbes = pd.concat(toutes_courbes, ignore_index=True)

    lib.log("\n" + "=" * 100)
    lib.log("=== SELECTION ADAPTATIVE DU NOMBRE DE CHEVAUX (k=1/2/3 selon la course, voir docstring) ===")
    lib.log("=" * 100)
    toutes_courbes_adapt = []
    for col, ascendant, label in indicateurs:
        courbe_adapt = courbe_couverture_adaptative(gagnants, col, ascendant, label)
        toutes_courbes_adapt.append(courbe_adapt)
        lib.log(f"\n   -- Indicateur (tri des courses) : {label} --")
        for _, row in courbe_adapt.iterrows():
            lib.log(f"      couverture={row['couverture_pct']:>3}% n={row['n_courses']:>5} "
                     f"k_moyen={row['k_moyen_adaptatif']:.2f} (k=1:{row['pct_k1_adaptatif']:>4}% "
                     f"k=2:{row['pct_k2_adaptatif']:>4}% k=3:{row['pct_k3_adaptatif']:>4}%) "
                     f"-> reussite_adaptatif={row['top_adaptatif_pct']:>5}%  "
                     f"[reference fixe : top1={row['top1_fixe_pct']}% top2={row['top2_fixe_pct']}% "
                     f"top3={row['top3_fixe_pct']}%]")
    df_courbes_adapt = pd.concat(toutes_courbes_adapt, ignore_index=True)

    lib.log("\n" + "=" * 100)
    lib.log("=== CIBLE 80% EN SELECTION ADAPTATIVE : couverture atteignable et k_moyen associe ===")
    lib.log("=" * 100)
    for col, ascendant, label in indicateurs:
        courbe_adapt = df_courbes_adapt[df_courbes_adapt["indicateur"] == label].sort_values(
            "couverture_pct", ascending=False)
        atteint = courbe_adapt[courbe_adapt["top_adaptatif_pct"] >= 80.0]
        if len(atteint):
            meilleure_couverture = atteint["couverture_pct"].max()
            ligne = atteint[atteint["couverture_pct"] == meilleure_couverture].iloc[0]
            lib.log(f"   {label:40s} 80%+ (adaptatif) des couverture={meilleure_couverture}% "
                    f"(n={ligne['n_courses']}, k_moyen={ligne['k_moyen_adaptatif']}, "
                    f"reussite={ligne['top_adaptatif_pct']}%)")
        else:
            meilleur = courbe_adapt["top_adaptatif_pct"].max()
            lib.log(f"   {label:40s} n'atteint jamais 80% en adaptatif sur les paliers testes (max={meilleur}%)")

    lib.log("\n" + "=" * 100)
    lib.log("=== SYNTHESE -- meilleur indicateur a chaque palier de couverture (classement par top3_pct) ===")
    lib.log("=" * 100)
    for pct in PALIERS_COUVERTURE:
        sous = df_courbes[df_courbes["couverture_pct"] == pct].sort_values("top3_pct", ascending=False)
        meilleur = sous.iloc[0]
        lib.log(f"   couverture={pct:>3}% (n~{meilleur['n_courses']}) -> meilleur indicateur = "
                f"{meilleur['indicateur']:40s} top1={meilleur['top1_pct']}% top3={meilleur['top3_pct']}% "
                f"top5={meilleur['top5_pct']}%")

    lib.log("\n" + "=" * 100)
    lib.log("=== CIBLE 80% TOP-3 : a quelle couverture (si atteignable) chaque indicateur l'atteint ===")
    lib.log("=" * 100)
    for col, ascendant, label in indicateurs:
        courbe = df_courbes[df_courbes["indicateur"] == label].sort_values("couverture_pct", ascending=False)
        atteint = courbe[courbe["top3_pct"] >= 80.0]
        if len(atteint):
            meilleure_couverture = atteint["couverture_pct"].max()
            ligne = atteint[atteint["couverture_pct"] == meilleure_couverture].iloc[0]
            lib.log(f"   {label:40s} atteint 80%+ Top-3 des couverture={meilleure_couverture}% "
                    f"(n={ligne['n_courses']}, top3 reel={ligne['top3_pct']}%)")
        else:
            meilleur_top3 = courbe["top3_pct"].max()
            lib.log(f"   {label:40s} n'atteint JAMAIS 80% Top-3 sur les paliers testes (max={meilleur_top3}%)")

    lib.log("\n" + "=" * 100)
    lib.log("=== DECOUPAGE HANDICAP vs NON-HANDICAP, sur le meilleur indicateur au palier 50% ===")
    lib.log("=" * 100)
    sous_50 = df_courbes[df_courbes["couverture_pct"] == 50].sort_values("top3_pct", ascending=False)
    meilleur_50 = sous_50.iloc[0]
    meilleur_col = [c for c, a, l in indicateurs if l == meilleur_50["indicateur"]][0]
    meilleur_ascendant = [a for c, a, l in indicateurs if l == meilleur_50["indicateur"]][0]
    for est_h, label_h in [(True, "HANDICAP"), (False, "NON HANDICAP")]:
        sous_gagnants = gagnants[gagnants["est_handicap"] == est_h]
        if len(sous_gagnants) < 5:
            continue
        courbe_h = courbe_couverture(sous_gagnants, meilleur_col, meilleur_ascendant, meilleur_50["indicateur"])
        ligne_50 = courbe_h[courbe_h["couverture_pct"] == 50]
        if len(ligne_50):
            r = ligne_50.iloc[0]
            lib.log(f"   {label_h:15s} n_total={len(sous_gagnants):>5} -> a 50% de couverture : "
                    f"n={r['n_courses']} top1={r['top1_pct']}% top3={r['top3_pct']}% top5={r['top5_pct']}%")

    lib.log("\n" + "=" * 100)
    lib.log("=== FIN -- resultat REEL de VALIDATION, non ajuste. AUCUN TEST A/B lance. ===")
    lib.log("=" * 100)

    lib.log("\n===CSV_COURBES_START===")
    for ligne_csv in df_courbes.to_csv(index=False).splitlines():
        lib.log(ligne_csv)
    lib.log("===CSV_COURBES_END===")

    lib.log("\n===CSV_COURBES_ADAPTATIF_START===")
    for ligne_csv in df_courbes_adapt.to_csv(index=False).splitlines():
        lib.log(ligne_csv)
    lib.log("===CSV_COURBES_ADAPTATIF_END===")


if __name__ == "__main__":
    main()
