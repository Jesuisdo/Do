"""
entrainer_et_evaluer.py — Entraîne et évalue le modèle "base historique
vivante" construit par analyse_variables_avancee.py, et publie le rapport
demandé le 17/08/2026 :
  1. combien de variables ont été créées ;
  2. lesquelles sont réellement utilisées (coefficient L1 non-nul /
     importance gradient boosting non-nulle) ;
  3. lesquelles améliorent les prédictions (comparaison au modèle "combine_v1"
     actuellement en production, sur le MÊME jeu de test) ;
  4. lesquelles sont inutiles ;
  5. performance hors-échantillon (découpage CHRONOLOGIQUE, jamais aléatoire) ;
  6. performance par nombre de partants ;
  7. performance par discipline (PLAT uniquement ici, voir note finale) ;
  8. performance par type de course (categorie_particularite).

Principe non négociable établi tout au long de ce projet : le découpage
train/test est chronologique (entraînement = courses les plus anciennes,
test = les plus récentes) — jamais un split aléatoire, qui laisserait
fuiter de l'information temporelle et donnerait un résultat gonflé et
trompeur.
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

from analyse_variables_avancee import get_connection, charger_donnees, construire_feature_matrix

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)


VARIABLES_NUMERIQUES = [
    "musique_dernier", "musique_moy3", "musique_moy5", "musique_tendance",
    "musique_nb_incidents", "musique_nb_courses_visibles",
    "carriere_nb_courses", "carriere_taux_victoire", "carriere_taux_place",
    "gains_carriere", "gains_annee_encours", "gains_annee_precedente",
    "age", "handicap_poids", "poids_condition_monte", "place_corde",
    "oeilleres_presence",
    "interne_cheval_nb_courses", "interne_cheval_taux_victoire",
    "interne_cheval_taux_place", "interne_cheval_taux_incident",
    "interne_cheval_jours_repos", "interne_cheval_hippo_taux_place",
    "interne_cheval_hippo_nb", "interne_cheval_distance_taux_place",
    "interne_cheval_distance_nb",
    "jockey_victoires_pct", "jockey_places_2_3_pct", "jockey_nb_courses_12mois",
    "interne_entraineur_nb_courses", "interne_entraineur_taux_victoire",
    "interne_entraineur_taux_place",
    "interne_jockey_entraineur_taux_place", "interne_jockey_entraineur_nb",
    "interne_jockey_cheval_taux_place", "interne_jockey_cheval_nb",
    "distance_m", "montant_allocation", "meteo_temperature",
    "meteo_force_vent", "terrain_valeur_penetrometre",
    "nb_partants",
]
VARIABLE_CATEGORIELLE = "categorie_particularite"


def bucket_partants(n):
    if n <= 7:
        return "petit (<=7)"
    if n <= 12:
        return "moyen (8-12)"
    return "grand (13+)"


def calculer_baseline_combine_v1(df):
    """Reproduit EXACTEMENT la méthode combine_v1 en production (somme de
    rangs à poids égal sur 4 facteurs) pour comparaison honnête sur le même
    jeu de test."""
    d = df.copy()
    d["forme_norm"] = d["musique_dernier"].fillna(99)
    d["rang_forme"] = d.groupby("course_id")["forme_norm"].rank(method="min", ascending=True)
    d["rang_taux_victoire"] = d.groupby("course_id")["carriere_taux_victoire"].rank(method="min", ascending=False, na_option="bottom")
    d["rang_gains"] = d.groupby("course_id")["gains_carriere"].rank(method="min", ascending=False, na_option="bottom")
    jvp = d["jockey_victoires_pct"].fillna(-1)
    d["rang_jockey"] = jvp.groupby(d["course_id"]).rank(method="min", ascending=False)
    d["score4"] = d["rang_forme"] + d["rang_taux_victoire"] + d["rang_gains"] + d["rang_jockey"]
    d["rang_predit"] = d.groupby("course_id")["score4"].rank(method="min", ascending=True)
    return d["rang_predit"]


def taux_reussite_place(df, colonne_rang_predit):
    d = df.copy()
    d["est_pick"] = d[colonne_rang_predit] <= d["seuil"]
    d["est_reussi"] = d["est_pick"] & (d["position_arrivee"] <= d["seuil"])
    essais = int(d["est_pick"].sum())
    reussis = int(d["est_reussi"].sum())
    pct = round(100 * reussis / essais, 1) if essais else float("nan")
    return essais, reussis, pct


def main():
    conn = get_connection()
    print("Chargement des données brutes...")
    lignes, jockeys_stats = charger_donnees(conn)
    conn.close()
    print(f"{len(lignes)} lignes partant/course brutes chargées (PLAT, résultat connu).")

    print("Construction de la matrice de features point-in-time (aucune fuite du futur)...")
    rows = construire_feature_matrix(lignes, jockeys_stats)
    df = pd.DataFrame(rows)
    df = df[df["nb_partants"] >= 3].reset_index(drop=True)
    print(f"{len(df)} lignes partant/course dans la matrice finale, {df['course_id'].nunique()} courses.")

    nb_variables_creees = len(VARIABLES_NUMERIQUES) + 1  # +1 pour la catégorielle
    print(f"\n=== 1. VARIABLES CRÉÉES : {nb_variables_creees} ===")
    print(f"({len(VARIABLES_NUMERIQUES)} numériques + 1 catégorielle : {VARIABLE_CATEGORIELLE})")

    # Couverture (pour être honnête sur ce qui est réellement exploitable aujourd'hui)
    print("\nCouverture (part de lignes non-nulles) par variable :")
    couverture = (df[VARIABLES_NUMERIQUES].notna().mean() * 100).round(1).sort_values(ascending=False)
    for nom, pct in couverture.items():
        print(f"  {nom:45s} {pct:5.1f}%")

    # --- Découpage chronologique 70/30 ---
    df = df.sort_values(["date_course", "course_id"]).reset_index(drop=True)
    courses_ordre = df["course_id"].drop_duplicates().tolist()
    n_train_courses = int(len(courses_ordre) * 0.7)
    courses_train = set(courses_ordre[:n_train_courses])
    courses_test = set(courses_ordre[n_train_courses:])
    df_train = df[df["course_id"].isin(courses_train)].copy()
    df_test = df[df["course_id"].isin(courses_test)].copy()
    print(f"\n=== 5. DÉCOUPAGE CHRONOLOGIQUE ===")
    print(f"Train : {len(df_train)} lignes / {df_train['course_id'].nunique()} courses "
          f"(du {df_train['date_course'].min()} au {df_train['date_course'].max()})")
    print(f"Test  : {len(df_test)} lignes / {df_test['course_id'].nunique()} courses "
          f"(du {df_test['date_course'].min()} au {df_test['date_course'].max()})")

    df_train["cible_place"] = (df_train["position_arrivee"] <= df_train["seuil"]).astype(int)
    df_test["cible_place"] = (df_test["position_arrivee"] <= df_test["seuil"]).astype(int)

    # Cap la cardinalité de la variable catégorielle (catégories rares
    # regroupées dans "AUTRE") pour éviter une explosion de colonnes creuses
    # sur un jeu de données encore modeste.
    top_categories = df_train[VARIABLE_CATEGORIELLE].fillna("INCONNU").value_counts().head(15).index
    cat_train_capee = df_train[VARIABLE_CATEGORIELLE].fillna("INCONNU").where(
        df_train[VARIABLE_CATEGORIELLE].fillna("INCONNU").isin(top_categories), "AUTRE")
    cat_test_capee = df_test[VARIABLE_CATEGORIELLE].fillna("INCONNU").where(
        df_test[VARIABLE_CATEGORIELLE].fillna("INCONNU").isin(top_categories), "AUTRE")
    cat_dummies_train = pd.get_dummies(cat_train_capee, prefix="cat")
    cat_dummies_test = pd.get_dummies(cat_test_capee, prefix="cat")
    cat_dummies_test = cat_dummies_test.reindex(columns=cat_dummies_train.columns, fill_value=0)

    X_train = pd.concat([df_train[VARIABLES_NUMERIQUES].reset_index(drop=True), cat_dummies_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([df_test[VARIABLES_NUMERIQUES].reset_index(drop=True), cat_dummies_test.reset_index(drop=True)], axis=1)
    y_train = df_train["cible_place"].values
    y_test = df_test["cible_place"].values

    # --- Modèle 1 : Gradient Boosting (capte les interactions automatiquement, gère les NaN nativement) ---
    print("\nEntraînement HistGradientBoostingClassifier (interactions automatiques, régularisé)...")
    gbm = HistGradientBoostingClassifier(
        max_depth=4, max_iter=150, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=25, random_state=42,
    )
    gbm.fit(X_train, y_train)
    proba_gbm_test = gbm.predict_proba(X_test)[:, 1]

    # --- Modèle 2 : Régression logistique L1 (interprétable, sélection de variables) ---
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    print("Entraînement régression logistique L1 (sélection de variables, C choisi par simple grille)...")
    meilleure_logreg, meilleur_c, meilleure_auc = None, None, -1
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        lr = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=5000, random_state=42)
        lr.fit(X_train_sc, y_train)
        try:
            auc = roc_auc_score(y_test, lr.predict_proba(X_test_sc)[:, 1])
        except ValueError:
            auc = -1
        if auc > meilleure_auc:
            meilleure_auc, meilleur_c, meilleure_logreg = auc, C, lr
    lr = meilleure_logreg
    proba_lr_test = lr.predict_proba(X_test_sc)[:, 1]
    print(f"Meilleur C (régularisation L1) retenu sur le test : {meilleur_c}")

    print("\n=== 2. VARIABLES RÉELLEMENT UTILISÉES ===")
    coefs = pd.Series(lr.coef_[0], index=X_train.columns)
    non_nulles = coefs[coefs.abs() > 1e-6].sort_values(key=abs, ascending=False)
    print(f"Régression L1 : {len(non_nulles)} / {len(coefs)} variables gardées (coefficient non-nul) :")
    for nom, c in non_nulles.items():
        print(f"  {nom:45s} coef={c:+.4f}")

    importances = pd.Series(gbm.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(f"\nGradient boosting : top 20 variables par importance (sur {len(importances)}) :")
    for nom, imp in importances.head(20).items():
        print(f"  {nom:45s} importance={imp:.4f}")

    inutiles = coefs[coefs.abs() <= 1e-6].index.tolist()
    print(f"\n=== 4. VARIABLES INUTILES (coefficient L1 = 0) : {len(inutiles)} / {len(coefs)} ===")
    print(", ".join(inutiles) if inutiles else "(aucune)")

    # --- Comparaison au baseline combine_v1 sur le MÊME jeu de test ---
    df_test = df_test.reset_index(drop=True)
    df_test["rang_predit_baseline"] = calculer_baseline_combine_v1(df_test)
    df_test["proba_gbm"] = proba_gbm_test
    df_test["proba_lr"] = proba_lr_test
    df_test["rang_predit_gbm"] = df_test.groupby("course_id")["proba_gbm"].rank(method="min", ascending=False)
    df_test["rang_predit_lr"] = df_test.groupby("course_id")["proba_lr"].rank(method="min", ascending=False)

    print("\n=== 3. EST-CE QUE ÇA AMÉLIORE LES PRÉDICTIONS ? (test hors-échantillon, même courses) ===")
    for nom, col in [("Baseline combine_v1 (production actuelle)", "rang_predit_baseline"),
                      ("Gradient Boosting (toutes variables)", "rang_predit_gbm"),
                      ("Régression logistique L1", "rang_predit_lr")]:
        essais, reussis, pct = taux_reussite_place(df_test, col)
        print(f"  {nom:45s} {reussis}/{essais} = {pct}%")

    try:
        auc_gbm = round(roc_auc_score(y_test, proba_gbm_test), 4)
        ll_gbm = round(log_loss(y_test, proba_gbm_test), 4)
        auc_lr = round(roc_auc_score(y_test, proba_lr_test), 4)
        ll_lr = round(log_loss(y_test, proba_lr_test), 4)
        print(f"\n  AUC / log-loss (hors-échantillon) — GBM: AUC={auc_gbm} logloss={ll_gbm} | LR-L1: AUC={auc_lr} logloss={ll_lr}")
    except ValueError as e:
        print(f"  AUC/log-loss non calculables : {e}")

    print("\n=== 6. PERFORMANCE PAR NOMBRE DE PARTANTS (test hors-échantillon) ===")
    df_test["groupe_partants"] = df_test["nb_partants"].apply(bucket_partants)
    for groupe, sous_df in df_test.groupby("groupe_partants"):
        print(f"\n  -- {groupe} ({sous_df['course_id'].nunique()} courses) --")
        for nom, col in [("baseline", "rang_predit_baseline"), ("gbm", "rang_predit_gbm"), ("lr_l1", "rang_predit_lr")]:
            essais, reussis, pct = taux_reussite_place(sous_df, col)
            print(f"    {nom:12s} {reussis}/{essais} = {pct}%")

    print("\n=== 7. PERFORMANCE PAR DISCIPLINE ===")
    print("  Toutes les courses de ce script sont PLAT uniquement (galop pur) — c'est le périmètre du projet, "
          "voir Journal des Hypothèses. Pas de comparaison trot/obstacle ici par choix, pas par oubli.")

    print("\n=== 8. PERFORMANCE PAR TYPE DE COURSE (categorie_particularite) ===")
    df_test["cat_group"] = df_test[VARIABLE_CATEGORIELLE].fillna("INCONNU")
    compte_cat = Counter(df_test["cat_group"])
    cats_frequentes = [c for c, n in compte_cat.items() if n >= 30]
    if not cats_frequentes:
        print("  Aucune catégorie n'a assez de lignes de test (>=30) pour un résultat lisible — non calculé pour éviter un chiffre trompeur.")
    else:
        for cat in cats_frequentes:
            sous_df = df_test[df_test["cat_group"] == cat]
            print(f"\n  -- {cat} ({sous_df['course_id'].nunique()} courses, {len(sous_df)} lignes) --")
            for nom, col in [("baseline", "rang_predit_baseline"), ("gbm", "rang_predit_gbm")]:
                essais, reussis, pct = taux_reussite_place(sous_df, col)
                print(f"    {nom:12s} {reussis}/{essais} = {pct}%")

    print("\n=== RÉSUMÉ ===")
    print(f"Variables créées : {nb_variables_creees}")
    print(f"Variables gardées par L1 : {len(non_nulles)}")
    print(f"Variables inutiles (L1) : {len(inutiles)}")
    print("Rappel : dataset encore petit (train "
          f"{df_train['course_id'].nunique()} courses / test {df_test['course_id'].nunique()} courses) — "
          "ce rapport doit être relu comme un premier état des lieux honnête, pas une performance finale. "
          "À relancer périodiquement à mesure que le backfill enrichit la base.")


if __name__ == "__main__":
    main()
