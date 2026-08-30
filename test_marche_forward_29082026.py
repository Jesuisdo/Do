# -*- coding: utf-8 -*-
"""
test_marche_forward_29082026.py -- Piste 4 (cotes de marche), premier test
demande par Dorian le 29-30/08/2026 : B+genealogie (modele actuel de
reference) applique en OUT-OF-SAMPLE PUR sur les courses PLAT du 29/08/2026,
puis diagnostic (SANS reentrainement) pour savoir si le signal marche
(rang de cote PMU en direct, fenetres H-30/H-15/H-5) aurait corrige des
erreurs du modele.

CONTRAINTE METHODOLOGIQUE ASSUMEE (a lire avant d'interpreter les
resultats) : les cotes de marche n'existent que pour le 29/08/2026 (~1
jour de donnees). Une variable 100% manquante sur tout le TRAIN historique
ne peut pas etre apprise par un modele -- un "B+genealogie+marche"
reentraine sur l'historique complet n'aurait donc aucun sens statistique
aujourd'hui. Ce script fait donc DEUX choses distinctes et les garde bien
separees :
  1. Un forward-test HONNETE de B+genealogie (meme protocole, memes
     hyperparametres que le run de reference du 28/08/2026), entraine sur
     tout l'historique STRICTEMENT anterieur au 29/08, value sur les
     derniers 15% de courses de cet historique (early stopping), puis
     SCORE sur les courses PLAT du 29/08 -- un vrai test out-of-sample,
     jamais vu a l'entrainement.
  2. Un diagnostic DETERMINISTE (pas un modele) : sur les courses ou
     B+genealogie se trompe (le gagnant reel n'est pas son pick #1), on
     regarde si le favori du marche (cote la plus basse a H-15) aurait
     trouve le bon gagnant. Ce n'est PAS un "modele B+genealogie+marche" --
     c'est une mesure directe du pouvoir cor:recteur potentiel du signal
     marche, qui guidera la decision de construire un vrai modele combine
     une fois que plusieurs jours de donnees de marche seront disponibles.

DEUX BASES DE DONNEES DISTINCTES SONT UTILISEES ICI, DELIBEREMENT VIA DEUX
VARIABLES D'ENVIRONNEMENT SEPAREES (voir le workflow associe) :
  - DATABASE_URL       (= secrets.DATABASE_URL_PLAT) : historique
    resultats_courses/resultats_partants (meme REQUETE PLAT-only que tout
    le pipeline v3_lib.py existant).
  - DATABASE_URL_COTES (= secrets.DATABASE_URL, avec repli sur
    DATABASE_URL si absent) : table cotes_historique (cotes en direct,
    collectees ce soir par collect_live_odds_render.py).
Ce choix delibere elimine tout risque de supposer a tort que les deux
secrets pointent vers la meme base Supabase -- le script fonctionne
correctement que ce soit le cas ou non.

AUCUN TEST A/B lance ici (diagnostic/validation uniquement, comme demande
par Dorian). Rien n'est ecrit en base -- lecture seule des deux cotes.
"""
import itertools
import os
import time

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

import v3_lib as lib
from identite_chevaux import resoudre_identite_chevaux
from variables_historiques import construire_variables, trier_chronologiquement
from variables_config import VARIABLES_NUMERIQUES, VARIABLES_CATEGORIELLES
from variables_genealogie import construire_variables_genealogie, COLONNES_GENEALOGIE

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

try:
    import lightgbm as lgb
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    LIGHTGBM_DISPONIBLE = False

DATE_TEST = os.environ.get("DATE_TEST_PISTE4", "2026-08-29")
DATE_TEST_COMPACT = DATE_TEST.replace("-", "")
DATABASE_URL_COTES = os.environ.get("DATABASE_URL_COTES") or os.environ.get("DATABASE_URL")


# =============================================================================
# Fonctions de metriques -- copiees a l'identique de entrainer_v3_phase2_genealogie.py
# (deja verifiees et utilisees en production sur ce projet) pour rester
# comparables terme a terme avec les runs precedents.
# =============================================================================

def groupes_consecutifs(course_id_iterable):
    return [len(list(g)) for _, g in itertools.groupby(list(course_id_iterable))]


def ndcg_gagnant_at_k(df, rang_col, k, cible_col="est_gagnant"):
    d = df[df[cible_col] == 1]
    n = len(d)
    if n == 0:
        return float("nan")
    gains = d[rang_col].apply(lambda r: 1.0 / np.log2(r + 1) if r <= k else 0.0)
    return round(float(gains.mean()), 4)


def mrr_gagnant(df, rang_col, cible_col="est_gagnant"):
    d = df[df[cible_col] == 1]
    n = len(d)
    if n == 0:
        return float("nan")
    return round(float((1.0 / d[rang_col]).mean()), 4)


def calculer_toutes_metriques(df, rang_col, score_col, y_vrai, label, score_croissant_avec_victoire=True):
    """score_croissant_avec_victoire=False pour un signal ou une valeur
    BASSE indique une victoire probable (ex: la cote elle-meme -- favori =
    cote la plus basse) : on inverse le signe avant l'AUC pour rester dans
    la convention scikit-learn (score haut = classe positive probable)."""
    stats_rang, _ = lib.rang_distribution_gagnant(df, rang_col)
    ndcg3 = ndcg_gagnant_at_k(df, rang_col, 3)
    ndcg5 = ndcg_gagnant_at_k(df, rang_col, 5)
    mrr = mrr_gagnant(df, rang_col)
    auc = None
    if roc_auc_score is not None:
        try:
            score = df[score_col] if score_croissant_avec_victoire else -df[score_col]
            auc = round(roc_auc_score(y_vrai, score), 4)
        except ValueError as e:
            lib.log(f"   [{label}] AUC non calculable : {e}")
    lib.log(f"   {label:34s} n={stats_rang['n_courses']:>3}  top1={stats_rang['top1_pct']:>5}%  "
            f"top3={stats_rang['cumul_top3_pct']:>5}%  top5={stats_rang['cumul_top5_pct']:>5}%  "
            f"NDCG@3={ndcg3}  NDCG@5={ndcg5}  MRR={mrr}  AUC={auc}")
    return {
        "n_courses": stats_rang["n_courses"], "top1_pct": stats_rang["top1_pct"],
        "top3_pct": stats_rang["cumul_top3_pct"], "top5_pct": stats_rang["cumul_top5_pct"],
        "ndcg3": ndcg3, "ndcg5": ndcg5, "mrr": mrr, "auc": auc,
    }


def entrainer_lambdarank(X_train, y_train, groups_train, X_val, y_val_eval, groups_val, label):
    """Hyperparametres IDENTIQUES au run de reference du 28/08/2026
    (entrainer_v3_phase2_genealogie.py) -- aucune nouvelle recherche de
    grille, pour rester strictement comparable."""
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


# =============================================================================
# Cotes de marche -- lecture depuis la base LIVE (courses/cotes_historique),
# separee de la base historique (resultats_courses/resultats_partants).
# =============================================================================

def _connecter_avec_retry(dsn, n_tentatives=6, delai_s=10):
    """Le pooler Supabase renvoie parfois une erreur transitoire
    (EAUTHQUERY : 'connection to database not available') meme quand la
    base est saine -- deja observe sur ce projet (backfill-resultats.yml),
    et confirme non lie a une panne large (les collectes concurrentes sur
    le meme secret reussissent). On retente simplement avant d'abandonner."""
    derniere_erreur = None
    for tentative in range(1, n_tentatives + 1):
        try:
            return psycopg2.connect(dsn)
        except psycopg2.OperationalError as e:
            derniere_erreur = e
            lib.log(f"   [connexion cotes] tentative {tentative}/{n_tentatives} echouee : {e}".strip())
            if tentative < n_tentatives:
                time.sleep(delai_s)
    raise derniere_erreur

def charger_cotes_marche(course_ids):
    if not DATABASE_URL_COTES:
        raise RuntimeError("DATABASE_URL_COTES (ou a defaut DATABASE_URL) absente de l'environnement.")
    if not course_ids:
        return pd.DataFrame(columns=["course_id", "numero", "minutes_avant_depart", "cote"])
    conn = _connecter_avec_retry(DATABASE_URL_COTES)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT course_id, numero, minutes_avant_depart, cote
               FROM cotes_historique
               WHERE course_id = ANY(%s) AND cote IS NOT NULL""",
            (course_ids,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["cote"] = pd.to_numeric(df["cote"], errors="coerce")
        df["minutes_avant_depart"] = pd.to_numeric(df["minutes_avant_depart"], errors="coerce")
    return df


def _cote_la_plus_proche(cotes_df, cible_min, fenetre, suffixe):
    """Pour chaque (course_id, numero), la cote dont minutes_avant_depart
    est le plus proche de `cible_min`, restreinte a la fenetre [fenetre[0],
    fenetre[1]]. Retourne un DataFrame course_id, numero, cote_{suffixe}."""
    lo, hi = fenetre
    d = cotes_df[(cotes_df["minutes_avant_depart"] >= lo) & (cotes_df["minutes_avant_depart"] <= hi)].copy()
    if d.empty:
        return pd.DataFrame(columns=["course_id", "numero", f"cote_{suffixe}"])
    d["ecart"] = (d["minutes_avant_depart"] - cible_min).abs()
    d = d.sort_values("ecart").drop_duplicates(["course_id", "numero"], keep="first")
    return d[["course_id", "numero", "cote"]].rename(columns={"cote": f"cote_{suffixe}"})


def construire_features_marche(cotes_df):
    """Features de marche simples et directement lisibles, comme demande
    par Dorian (pas 50 variables) : cote aux 3 fenetres cles, rang de cote
    a H-15 (1 = favori), evolution H-30->H-15->H-5, ecart vs la moyenne du
    champ a H-15."""
    h30 = _cote_la_plus_proche(cotes_df, 30, (22, 38), "h30")
    h15 = _cote_la_plus_proche(cotes_df, 15, (7, 23), "h15")
    h5 = _cote_la_plus_proche(cotes_df, 5, (-3, 13), "h5")
    m = h30.merge(h15, on=["course_id", "numero"], how="outer").merge(h5, on=["course_id", "numero"], how="outer")
    m["rang_cote_h15"] = m.groupby("course_id")["cote_h15"].rank(method="min", ascending=True)
    m["delta_cote_h30_h15"] = m["cote_h15"] - m["cote_h30"]
    m["delta_cote_h15_h5"] = m["cote_h5"] - m["cote_h15"]
    moyenne_champ_h15 = m.groupby("course_id")["cote_h15"].transform("mean")
    m["ecart_vs_moyenne_champ_h15"] = m["cote_h15"] - moyenne_champ_h15
    return m


def top1_par_masque(df, rang_col, masque, label_segment):
    d = df[masque]
    n_gagnants = int((d["est_gagnant"] == 1).sum())
    if n_gagnants == 0:
        return None
    n_reussis = int(((d["est_gagnant"] == 1) & (d[rang_col] == 1)).sum())
    return {"segment": label_segment, "n_courses": n_gagnants, "n_reussis": n_reussis,
            "pct_top1": round(100 * n_reussis / n_gagnants, 1)}


def main():
    if not lib.DEPENDANCES_LOURDES_DISPONIBLES:
        raise RuntimeError("psycopg2 et/ou scikit-learn ne sont pas installes.")
    if not LIGHTGBM_DISPONIBLE:
        raise RuntimeError("lightgbm non installe.")

    lib.log("=" * 100)
    lib.log("PISTE 4 -- TEST MARCHE FORWARD -- B+genealogie (OOS 29/08/2026) + diagnostic marche -- 30/08/2026")
    lib.log("AUCUN TEST A/B -- diagnostic uniquement, decision laissee a Dorian.")
    lib.log("=" * 100)

    lib.log("\n[1/9] Chargement des donnees brutes PLAT (historique + 29/08 inclus, meme REQUETE que le pipeline existant)...")
    lignes = lib.charger_donnees_brutes()
    lib.log(f"   {len(lignes)} lignes brutes chargees.")
    n_lignes_29 = sum(1 for l in lignes if l["date_course"].isoformat() == DATE_TEST)
    lib.log(f"   Lignes du {DATE_TEST} deja presentes dans resultats_partants/resultats_courses : {n_lignes_29}")
    if n_lignes_29 == 0:
        lib.log("   AUCUNE ligne du 29/08 trouvee -- verifier que DATABASE_URL (historique) pointe bien vers la "
                "meme base que celle backfilled ce soir. Arret.")
        return

    lib.log("\n[2/9] Resolution d'identite des chevaux (identique v2/v3)...")
    horse_uids, rapport_identite = resoudre_identite_chevaux(lignes)
    for l, uid in zip(lignes, horse_uids):
        l["horse_uid"] = uid
    lib.log(f"   {rapport_identite['n_chevaux_distincts_resolus']} chevaux distincts resolus.")

    lib.log("\n[3/9] Construction des variables v3 + genealogie point-in-time (ensemble trie complet, y compris le "
            f"{DATE_TEST} -- chaque ligne de cette date ne voit que l'historique STRICTEMENT anterieur, aucune fuite)...")
    lignes_triees = trier_chronologiquement(lignes)
    features = construire_variables(lignes_triees)
    df = pd.DataFrame(features)
    features_geneal = construire_variables_genealogie(lignes_triees)
    df_geneal = pd.DataFrame(features_geneal)
    assert len(df_geneal) == len(df), "desalignement genealogie vs variables v3"
    for col in COLONNES_GENEALOGIE:
        df[col] = df_geneal[col].values
    del lignes, lignes_triees, features, features_geneal, df_geneal, horse_uids

    df = df[df["position_arrivee"].notna()].copy()
    df = df[df["nb_partants_reel"] >= 3].reset_index(drop=True)
    df["est_gagnant"] = (df["position_arrivee"] == 1).astype(int)
    df["cible_place"] = (df["position_arrivee"] <= df["seuil"]).astype(int)
    df = lib.ajouter_variables_relatives(df, lib.VARIABLES_RELATIVES_CIBLES)
    df["date_course_str"] = df["date_course"].astype(str)

    df_hist = df[df["date_course_str"] < DATE_TEST].copy()
    df_test_toutes = df[df["date_course_str"] == DATE_TEST].copy()
    lib.log(f"   Historique (< {DATE_TEST}) apres filtrage : {df_hist['course_id'].nunique()} courses, {len(df_hist)} lignes.")
    lib.log(f"   Courses PLAT du {DATE_TEST} (avec resultat, nb_partants>=3) : {df_test_toutes['course_id'].nunique()} courses.")
    del df

    lib.log(f"\n[4/9] Split chronologique STRICT : derniers 15% des courses de l'historique -> VALIDATION (early "
            f"stopping uniquement), le reste -> TRAIN. Le {DATE_TEST} est un TEST out-of-sample pur (jamais vu).")
    df_hist = df_hist.sort_values(["date_course", "course_id"]).reset_index(drop=True)
    courses_ordre = df_hist["course_id"].drop_duplicates().tolist()
    n = len(courses_ordre)
    n_train = int(n * 0.85)
    courses_train = set(courses_ordre[:n_train])
    courses_val = set(courses_ordre[n_train:])
    df_train = df_hist[df_hist["course_id"].isin(courses_train)].reset_index(drop=True)
    df_val = df_hist[df_hist["course_id"].isin(courses_val)].reset_index(drop=True)
    lib.log(f"   TRAIN={df_train['course_id'].nunique()} courses / {len(df_train)} lignes  "
            f"VALIDATION(early-stop)={df_val['course_id'].nunique()} courses  "
            f"TEST({DATE_TEST})={df_test_toutes['course_id'].nunique()} courses")
    del df_hist

    lib.log("\n[5/9] Construction des matrices v3+genealogie (memes colonnes reindexees sur train/val/test)...")
    variables_numeriques_v3 = lib.variables_numeriques_v3(VARIABLES_NUMERIQUES)
    X_train_v3 = lib.preparer_matrice(df_train, variables_numeriques_v3, VARIABLES_CATEGORIELLES)
    colonnes_v3 = X_train_v3.columns
    X_val_v3 = lib.preparer_matrice(df_val, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)
    X_test_v3 = lib.preparer_matrice(df_test_toutes, variables_numeriques_v3, VARIABLES_CATEGORIELLES, colonnes_dummies_reference=colonnes_v3)

    colonnes_numeriques_v3 = [c for c in colonnes_v3 if c in variables_numeriques_v3]
    degenerees_v3 = lib.colonnes_degenerees(X_train_v3, colonnes_numeriques_v3)
    if degenerees_v3:
        lib.log(f"   ATTENTION : {len(degenerees_v3)} colonne(s) v3 degeneree(s) sur TRAIN, exclue(s) : {degenerees_v3}")
    colonnes_v3_filtrees = [c for c in colonnes_v3 if c not in degenerees_v3]
    X_train_v3 = X_train_v3[colonnes_v3_filtrees]
    X_val_v3 = X_val_v3[colonnes_v3_filtrees]
    X_test_v3 = X_test_v3[colonnes_v3_filtrees]

    X_train_geneal = df_train[COLONNES_GENEALOGIE].reset_index(drop=True).astype("float32")
    degenerees_geneal = lib.colonnes_degenerees(X_train_geneal, COLONNES_GENEALOGIE)
    if degenerees_geneal:
        lib.log(f"   ATTENTION : {len(degenerees_geneal)} colonne(s) genealogie degeneree(s) sur TRAIN, exclue(s) : {degenerees_geneal}")
    colonnes_geneal_filtrees = [c for c in COLONNES_GENEALOGIE if c not in degenerees_geneal]
    X_train_geneal = X_train_geneal[colonnes_geneal_filtrees]
    X_val_geneal = df_val[colonnes_geneal_filtrees].reset_index(drop=True).astype("float32")
    X_test_geneal = df_test_toutes[colonnes_geneal_filtrees].reset_index(drop=True).astype("float32")

    X_train_full = pd.concat([X_train_v3.reset_index(drop=True), X_train_geneal], axis=1)
    X_val_full = pd.concat([X_val_v3.reset_index(drop=True), X_val_geneal], axis=1)
    X_test_full = pd.concat([X_test_v3.reset_index(drop=True), X_test_geneal], axis=1)
    lib.log(f"   Matrice finale B+genealogie : {X_train_full.shape[1]} colonnes "
            f"({len(colonnes_geneal_filtrees)} de genealogie).")

    lib.log("\n[6/9] Entrainement de B+genealogie (LGBMRanker lambdarank_graded, hyperparametres IDENTIQUES au run "
            "de reference du 28/08/2026)...")
    y_train_graded = np.where(df_train["est_gagnant"].values == 1, 2,
                               np.where(df_train["cible_place"].values == 1, 1, 0)).astype(int)
    groups_train = groupes_consecutifs(df_train["course_id"])
    groups_val = groupes_consecutifs(df_val["course_id"])
    modele = entrainer_lambdarank(X_train_full, y_train_graded, groups_train,
                                   X_val_full, df_val["est_gagnant"].values, groups_val, "B+genealogie")

    df_test_toutes["score_modele"] = modele.predict(X_test_full)
    df_test_toutes["rang_modele"] = df_test_toutes.groupby("course_id")["score_modele"].rank(method="min", ascending=False)

    lib.log("\n[7/9] Export des scores B+genealogie OOS (29/08) en CSV -- AUCUNE connexion a cotes_historique depuis")
    lib.log("      GitHub Actions dans cette version : le pooler Supabase a declenche un ECIRCUITBREAKER sur ce")
    lib.log("      secret ce soir (deux runs precedents). Le rapprochement avec le marche se fait desormais en")
    lib.log("      dehors de ce workflow, via le MCP Supabase direct (meme chemin fiable que le backfill de ce soir).")
    colonnes_export = ["course_id", "numero", "position_arrivee", "est_gagnant", "cible_place",
                        "rang_modele", "score_modele", "categorie_particularite"]
    df_export = df_test_toutes[colonnes_export].copy()
    nom_fichier_export = f"scores_modele_{DATE_TEST_COMPACT}.csv"
    df_export.to_csv(nom_fichier_export, index=False)
    n_lignes_export = len(df_export)
    n_courses_export = df_export["course_id"].nunique()
    lib.log(f"   {n_lignes_export} lignes exportees ({n_courses_export} courses) -> {nom_fichier_export}")
    lib.log("\n===CSV_SCORES_START===")
    for ligne_csv in df_export.to_csv(index=False).splitlines():
        lib.log(ligne_csv)
    lib.log("===CSV_SCORES_END===")
    lib.log("\n=== FIN (partie modele) -- comparaison avec le marche a faire hors GitHub Actions, via MCP Supabase. ===")
    return

    lib.log("\n[7/9] Jointure avec les cotes de marche (table cotes_historique, base LIVE separee, memes course_id)...")
    course_ids_test = df_test_toutes["course_id"].unique().tolist()
    cotes_df = charger_cotes_marche(course_ids_test)
    courses_avec_cotes = set(cotes_df["course_id"].unique()) if not cotes_df.empty else set()
    lib.log(f"   {len(courses_avec_cotes)}/{len(course_ids_test)} courses PLAT du {DATE_TEST} ont au moins une cote captee.")

    df_exploitable = df_test_toutes[df_test_toutes["course_id"].isin(courses_avec_cotes)].copy()
    n_exploitable = df_exploitable["course_id"].nunique()
    lib.log(f"   -> {n_exploitable} courses PLAT exploitables (resultat certain + cote captee).")

    if n_exploitable < 5:
        lib.log(f"\n   COUVERTURE INSUFFISANTE ({n_exploitable} courses exploitables, minimum retenu=5) -- "
                f"arret ici, pas de metriques marche calculees (bruit trop important sur un si petit echantillon).")
        lib.log("\n   Metriques B+genealogie seul (out-of-sample, TOUTES les courses PLAT du 29/08, avec ou sans cote) :")
        calculer_toutes_metriques(df_test_toutes, "rang_modele", "score_modele", df_test_toutes["est_gagnant"], "B+genealogie (OOS complet)")
        return

    feats_marche = construire_features_marche(cotes_df)
    df_exploitable = df_exploitable.merge(feats_marche, on=["course_id", "numero"], how="left")

    n_h30 = int(df_exploitable.groupby("course_id")["cote_h30"].apply(lambda s: s.notna().any()).sum())
    n_h15 = int(df_exploitable.groupby("course_id")["cote_h15"].apply(lambda s: s.notna().any()).sum())
    n_h5 = int(df_exploitable.groupby("course_id")["cote_h5"].apply(lambda s: s.notna().any()).sum())
    lib.log(f"   Couverture des fenetres (parmi les {n_exploitable} courses exploitables) : "
            f"H-30={n_h30}  H-15={n_h15}  H-5={n_h5}")

    lib.log("\n[8/9] METRIQUES -- priorite Top-3 -> Top-1 -> Top-5 -> NDCG@3/@5 -> MRR -> AUC")
    lib.log("=" * 100)
    lib.log("=== B+genealogie SEUL, out-of-sample pur, sur les courses PLAT EXPLOITABLES du 29/08 ===")
    res_modele = calculer_toutes_metriques(df_exploitable, "rang_modele", "score_modele",
                                            df_exploitable["est_gagnant"], "B+genealogie (OOS, exploitables)")

    df_exploitable["rang_cote_h15"] = df_exploitable.groupby("course_id")["cote_h15"].rank(method="min", ascending=True)
    df_marche_valide = df_exploitable.dropna(subset=["rang_cote_h15"])
    res_marche = None
    if df_marche_valide["course_id"].nunique() >= 3:
        lib.log("\n=== Marche SEUL (rang de cote a H-15, favori = cote la plus basse), meme sous-ensemble ===")
        res_marche = calculer_toutes_metriques(df_marche_valide, "rang_cote_h15", "cote_h15",
                                                df_marche_valide["est_gagnant"], "Marche seul (rang cote H-15)",
                                                score_croissant_avec_victoire=False)

    if res_marche:
        lib.log("\n=== DELTAS -- Marche seul vs B+genealogie (indicatif -- le marche n'est PAS un modele entraine) ===")
        lib.log(f"   top1={round(res_marche['top1_pct']-res_modele['top1_pct'],1):+}pt  "
                f"top3={round(res_marche['top3_pct']-res_modele['top3_pct'],1):+}pt  "
                f"top5={round(res_marche['top5_pct']-res_modele['top5_pct'],1):+}pt")

    lib.log("\n[9/9] DIAGNOSTIC -- combien des erreurs de B+genealogie le marche aurait-il corrigees ?")
    lib.log("=" * 100)
    gagnants = df_exploitable[df_exploitable["est_gagnant"] == 1].copy()
    n_gagnants = len(gagnants)
    erreurs_modele = gagnants[gagnants["rang_modele"] != 1]
    n_erreurs = len(erreurs_modele)
    lib.log(f"   B+genealogie place le vrai gagnant en pick #1 sur {n_gagnants - n_erreurs}/{n_gagnants} courses "
            f"exploitables ({round(100*(n_gagnants-n_erreurs)/n_gagnants,1) if n_gagnants else 0}%).")
    lib.log(f"   -> {n_erreurs} courses ou B+genealogie se trompe (le 'noyau dur' de ce sous-ensemble).")

    if n_erreurs and "rang_cote_h15" in erreurs_modele.columns:
        erreurs_modele_avec_cote = erreurs_modele.dropna(subset=["rang_cote_h15"])
        corrections = erreurs_modele_avec_cote[erreurs_modele_avec_cote["rang_cote_h15"] == 1]
        n_corr = len(corrections)
        n_base = len(erreurs_modele_avec_cote)
        lib.log(f"\n   *** CHIFFRE CLE *** : sur les {n_base} erreurs de B+genealogie ayant une cote H-15 exploitable, "
                f"le favori du marche aurait trouve le bon gagnant dans {n_corr} cas "
                f"({round(100*n_corr/n_base,1) if n_base else 0}%).")

        # Effet sur les handicaps
        d_full = df_exploitable.copy()
        d_full["est_handicap"] = d_full["categorie_particularite"].fillna("").str.contains("HANDICAP")
        gagnants_h = d_full[(d_full["est_gagnant"] == 1) & (d_full["est_handicap"])]
        erreurs_h = gagnants_h[gagnants_h["rang_modele"] != 1].dropna(subset=["rang_cote_h15"])
        if len(erreurs_h):
            corr_h = erreurs_h[erreurs_h["rang_cote_h15"] == 1]
            lib.log(f"\n   -- Effet sur les HANDICAPS : {len(corr_h)}/{len(erreurs_h)} erreurs corrigees par le marche "
                    f"({round(100*len(corr_h)/len(erreurs_h),1)}%).")
        else:
            lib.log("\n   -- Effet sur les HANDICAPS : aucune course handicap exploitable avec erreur de B+genealogie dans cet echantillon.")

        # Effet sur les courses a faible confiance du modele (ecart de score
        # faible entre le pick #1 et le pick #2 du modele -- proxy d'hesitation)
        d_full["rang_modele_int"] = d_full["rang_modele"]
        picks1 = d_full[d_full["rang_modele"] == 1][["course_id", "score_modele"]].rename(columns={"score_modele": "score_pick1"})
        picks2 = d_full[d_full["rang_modele"] == 2][["course_id", "score_modele"]].rename(columns={"score_modele": "score_pick2"})
        ecarts = picks1.merge(picks2, on="course_id", how="inner")
        ecarts["ecart_score_1_2"] = ecarts["score_pick1"] - ecarts["score_pick2"]
        if len(ecarts) >= 6:
            seuil_faible_confiance = ecarts["ecart_score_1_2"].quantile(1 / 3)
            courses_faible_confiance = set(ecarts.loc[ecarts["ecart_score_1_2"] <= seuil_faible_confiance, "course_id"])
            gagnants_fc = gagnants[gagnants["course_id"].isin(courses_faible_confiance)]
            erreurs_fc = gagnants_fc[gagnants_fc["rang_modele"] != 1].dropna(subset=["rang_cote_h15"])
            if len(erreurs_fc):
                corr_fc = erreurs_fc[erreurs_fc["rang_cote_h15"] == 1]
                lib.log(f"\n   -- Effet sur les courses a FAIBLE CONFIANCE du modele (tercile inferieur d'ecart de score "
                        f"pick1/pick2, n={len(courses_faible_confiance)} courses) : "
                        f"{len(corr_fc)}/{len(erreurs_fc)} erreurs corrigees par le marche "
                        f"({round(100*len(corr_fc)/len(erreurs_fc),1)}%).")
            else:
                lib.log("\n   -- Effet sur les courses a faible confiance : aucune erreur de B+genealogie dans ce sous-groupe.")
        else:
            lib.log("\n   -- Effet sur les courses a faible confiance : echantillon trop petit pour un tercile fiable (<6 courses).")
    else:
        lib.log("   Aucune erreur de B+genealogie sur ce sous-ensemble, ou aucune cote H-15 disponible -- rien a corriger.")

    lib.log("\n" + "=" * 100)
    lib.log("=== RESUME -- ce rapport est un DIAGNOSTIC de faisabilite (marche seul, deterministe), PAS un modele ===")
    lib.log(f"=== B+genealogie+marche reentraine sur l'historique complet n'est PAS encore possible : la feature ===")
    lib.log(f"=== marche est 100% absente du TRAIN historique (une seule journee de donnees, le {DATE_TEST}). ===")
    lib.log("=== AUCUN TEST A ni TEST B lance -- decision laissee a Dorian. ===")
    lib.log("=" * 100)


if __name__ == "__main__":
    main()
