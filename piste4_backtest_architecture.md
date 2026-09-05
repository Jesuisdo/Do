# Architecture du backtest ROI — Piste 4 (B+généalogie → VERT → Top-5 → marché H-15)

Statut (mis à jour le 05/09/2026) : **rien n'est exécuté sur données réelles**. Le pipeline de SÉLECTION (2.1 → 2.5, quel cheval jouer) est désormais implémenté et testé sur données synthétiques dans `piste4_backtest_engine.py`, y compris la comparaison "modèle seul" vs "modèle + marché H-15" et le rendement cumulé dans le temps. Aucun paramètre de MISE (2.6/2.7 : cote minimale acceptable, taille de mise) n'est choisi, `run_backtest()` reste verrouillé et refuse de s'exécuter quelle que soit la demande. Objectif : être prêt à lancer un vrai test hors échantillon dès que (a) le volume de cotes H-15 sera jugé suffisant par Dorian ET (b) une règle de mise sera explicitement choisie par Dorian.

## 0. Cadre

Référence modèle figée (ne plus retester) :

- B+généalogie → filtre VERT → Top-5
- Top-1 VERT ~38-40%, Top-3 ~72%, Top-5 ~91-92%

Ce système ne remplace pas B+généalogie. Il ajoute une couche de **décision de pari** au-dessus du Top-5 VERT, informée par la cote marché à H-15. Aucun modèle combiné B+généalogie+marché n'est construit ici.

## 1. Pipeline complet

```
1. Prédiction B+généalogie (OOS)         → scores_modele_{date}.csv
2. Filtre VERT                            → sous-ensemble fiable du peloton
3. Classement Top-5 (parmi VERT)          → 5 chevaux candidats max
4. Cote H-15 point-in-time                → jointure sur cotes_historique
5. Choix du cheval (stratégie)            → 1 cheval ou aucun pari
6. Cote minimale acceptable (règle)       → valide ou annule le pari
7. Décision de pari                       → pari_place = 0/1, mise
8. Résultat réel                          → resultats_partants
9. Gain / perte                           → par pari
10. Agrégation                            → ROI, drawdown, taux de réussite, n paris
11. Comparaison à des stratégies de référence
```

Chaque étage est une fonction pure, testable indépendamment, pour pouvoir rejouer le pipeline sur n'importe quelle stratégie sans dupliquer le code de collecte/jointure.

## 2. Détail par étage

### 2.1 Prédiction B+généalogie
Source : `test_marche_forward_29082026.py` (paramétré par `DATE_TEST_PISTE4`), export `scores_modele_{date}.csv` :
`course_id, numero, position_arrivee, est_gagnant, cible_place, rang_modele, score_modele, categorie_particularite`.
Entraînement strictement OOS — inchangé, hyperparamètres non touchés.

### 2.2 Filtre VERT — RÉSOLU (04/09/2026)
Source de vérité identifiée par inspection directe du dépôt : `piste7_phase2_confiance_directe.py` calibre l'indicateur de confiance sur VALIDATION (comparaison de plusieurs indicateurs : entropie_champ, somme_top3_proba, proba_pick1, ecart_1_2_normalise) et fige un seuil. Ce seuil est ensuite repris tel quel, en dur, dans les scripts suivants de la piste 7 (`piste7_diagnostic_marge_top1_vert.py`, `piste7_approche3_specialise_vert.py`) :

```
SEUIL_VERT_FIGE = 0.5848   # sur somme_top3_proba
```

`somme_top3_proba` = somme des 3 plus hautes pseudo-probabilités obtenues en appliquant un softmax à `score_geneal` (= `score_modele` dans l'export de `test_marche_forward_29082026.py`, même modèle B+généalogie) sur **tout le champ** de la course (pas seulement le Top-5). Une course est VERT si `somme_top3_proba >= 0.5848`. Le Top-5 reste le classement brut (`rang_geneal`/`rang_modele <= 5`), inchangé par ce filtre.

Comme `score_modele` est déjà présent dans l'export `scores_modele_{date}.csv`, ce calcul se fait entièrement en aval, sans modifier `test_marche_forward_29082026.py` ni `v3_lib.py`. Le seuil est dupliqué (avec attribution en commentaire) dans `piste4_backtest_engine.py` plutôt que factorisé dans `v3_lib.py`, pour ne pas toucher à un module partagé sans validation explicite — un refactor de factorisation resterait une décision séparée à te soumettre.

**Limite découverte au passage** : les CSV `scores_modele_{date}.csv` (détail par cheval) ne sont pas persistés dans le dépôt au-delà de l'exécution qui les produit ; seul l'agrégat par course (gagnant uniquement) entre dans `piste4_marche_journalier.csv`. Pour un backtest rétroactif sur plusieurs jours déjà traités, il manque donc le détail par cheval (VERT, Top-5, score) de ces jours-là. Voir section 6.

### 2.3 Top-5
Parmi les chevaux VERT d'une course, les 5 meilleurs `rang_modele` (ou `score_modele` décroissant). S'il y a moins de 5 chevaux VERT, Top-5 = tous les VERT disponibles (pas de complément par des non-VERT).

### 2.4 Cote H-15 point-in-time
Reprend exactement la méthodologie déjà validée le 30/08 (aucune redéfinition) :
- `minutes_avant_depart >= 15`, on prend le minimum de `minutes_avant_depart` parmi ces lignes (mise à jour la plus récente disponible au moment ou avant H-15).
- Jamais de logique "closest to target".
- Dédupliquer par cheval avant de calculer `rank()` (jamais sur les lignes brutes non agrégées).
- Couverture minimale 70% des partants ayant un résultat ; sinon la course est exclue de cet étage pour cette course (le pari n'est pas évalué, ce n'est pas une valeur estimée).

### 2.5 Choix du cheval (stratégie) — IMPLÉMENTÉ le 05/09/2026 (2 stratégies, demande explicite de Dorian)
Le moteur accepte plusieurs stratégies interchangeables via le registre `SELECTION_STRATEGIES`. Deux sont maintenant implémentées et testées (données synthétiques uniquement) :
- `top1_modele` : toujours le rang_modele=1 du Top-5 VERT (ignore le marché) — référence "B+généalogie seul".
- `top1_marche_h15_dans_top5_vert` : le cheval du Top-5 VERT avec la meilleure cote H-15 (favori du marché, RESTREINT aux 5 chevaux déjà retenus par le modèle) — pipeline complet demandé par Dorian.

Ces deux stratégies servent uniquement à la COMPARAISON modèle-seul vs modèle+marché (`compare_modele_seul_vs_marche_h15`) ; aucune n'est "choisie" comme stratégie de pari finale — ça reste une décision de mise (2.6/2.7), pas de sélection. D'autres stratégies (ex. `arbitrage_faible_confiance`, `top1_modele_si_cote_h15_confirme`) restent des idées non implémentées, à ajouter seulement sur demande explicite.

### 2.6 Cote minimale acceptable
Paramètre configurable (`cote_min_acceptable`), pas de valeur figée. Si la cote H-15 du cheval sélectionné est en dessous du seuil, le pari n'est pas placé (`pari_place = 0`) mais la course reste dans les données pour ne pas biaiser le comptage.

### 2.7 Décision de pari
Combine 2.5 + 2.6 → `cheval_selectionne`, `pari_place`, `mise`. Mise par défaut = mise fixe (paramétrable plus tard, pas de gestion de bankroll dynamique tant que non demandée).

### 2.8 Résultat réel
Jointure sur `resultats_partants` (position d'arrivée, gagnant, placé).

### 2.9 Gain / perte
`gain_brut = mise × (cote_h15 − 1)` si gagnant, sinon `−mise`. Cote utilisée = celle captée à H-15 au moment de la décision (jamais une cote a posteriori — même contrainte point-in-time que pour le ranking).

### 2.10 Agrégation — IMPLÉMENTÉ le 05/09/2026 (`compute_metrics`, `compute_metrics_par_segment`, `compute_rendement_cumule`)
Par stratégie × règle de mise, cumulé et par sous-groupe (global, `faible_confiance`, `handicap`) :
- ROI = gain net total / mise totale
- taux de réussite (winrate) = paris gagnants / paris placés
- nombre de paris (placés vs courses évaluées, pour voir le taux de "skip")
- drawdown max = plus grande baisse depuis un pic sur la courbe de bankroll cumulée (dans l'ordre chronologique)
- rendement cumulé dans le temps (`compute_rendement_cumule`) : mise/gain/ROI/winrate cumulés pari après pari, triés par date — nécessite que chaque `RaceContext` porte sa date (champ ajouté le 05/09/2026).

### 2.11 Stratégies de référence pour comparaison
Aucune stratégie de pari n'a de sens seule — toujours comparer à des baselines simples, mêmes squelettes que 2.5 :
- `flat_top1_modele` (B+généalogie seul, sans marché, mise fixe systématique)
- `flat_favori_marche` (favori du marché seul, sans modèle)
- `flat_aleatoire_top5_vert` (tirage aléatoire dans le Top-5 VERT, pour borne basse)
- toute stratégie testée (2.5) doit battre `flat_top1_modele` en ROI pour être considérée comme un gain réel.

## 3. Schéma de données

### 3.1 `piste4_backtest_paris.csv` (une ligne par course évaluée)
```
date, course_id, strategie_id, regle_mise_id, n_partants, handicap, gap_confiance,
faible_confiance, somme_top3_proba, niveau_vert, seuil_vert_utilise,
top5_vert_numeros, cheval_selectionne, rang_modele_selectionne,
cote_h15_selectionne, cote_min_acceptable, pari_place, mise,
position_arrivee_selectionne, gagnant, place,
gain_brut, resultat_net, bankroll_cumulee
```
(`somme_top3_proba`, `niveau_vert`, `seuil_vert_utilise` : traçabilité du filtre VERT reproduit depuis piste7, cf. section 2.2.)

### 3.2 `piste4_backtest_summary.csv` (une ligne par stratégie × règle de mise × date de mise à jour)
```
date_maj, strategie_id, regle_mise_id, n_courses_evaluees, n_paris_places,
n_paris_gagnants, taux_reussite, mise_totale, gain_total, roi_pct,
bankroll_finale, drawdown_max_pct,
n_paris_faible_confiance, roi_faible_confiance_pct,
n_paris_handicap, roi_handicap_pct
```

Ces deux fichiers sont distincts de `piste4_marche_journalier.csv` (qui reste dédié à l'étude de corrélation modèle/marché, inchangé).

## 4. Modularité du moteur

`piste4_backtest_engine.py` (committé le 05/09/2026) expose :
- un registre `SELECTION_STRATEGIES` (nom → fonction) — **2 entrées actives depuis le 05/09/2026** : `top1_modele`, `top1_marche_h15_dans_top5_vert` (voir 2.5).
- un registre `BETTING_RULES` (nom → fonction de seuil/mise) — **toujours vide**, aucune règle de mise choisie.
- `simuler_strategie(contexts, strategy, rule, mise, resultats)` : rejoue une stratégie sur un ensemble de courses (paramètres fournis par l'appelant, rien de codé en dur).
- `compare_modele_seul_vs_marche_h15(contexts, rule, mise, resultats)` : lance les 2 stratégies enregistrées sur le même ensemble de courses et retourne le tableau comparatif.
- `compute_rendement_cumule(bets)` : courbe de rendement cumulé dans le temps.
- `run_backtest(date_range, strategie_id, regle_mise_id)` : **reste verrouillé sans condition** (lève toujours une RuntimeError) — la présence de stratégies dans `SELECTION_STRATEGIES` ne suffit pas à le débloquer, `BETTING_RULES` vide bloque toujours l'exécution complète du pipeline bout-en-bout sur données réelles.

Ajouter une stratégie de sélection = ajouter une fonction au registre `SELECTION_STRATEGIES` (fait pour les 2 ci-dessus). Ajouter une règle de mise à `BETTING_RULES` reste une décision explicite de Dorian, non prise ici. Aucun cœur du pipeline (collecte, jointure point-in-time, calcul ROI) n'a été modifié pour ajouter ces 2 stratégies.

## 5. Garde-fous

- Pas d'exécution tant que le volume de courses avec cote H-15 exploitable n'est pas jugé suffisant par toi.
- Pas de choix de stratégie ni de seuil de cote minimale sans ta décision.
- Pas de modèle combiné B+généalogie+marché.
- Pas de modification de `collect_live_odds_render.py`, `.github/workflows/collecte-cotes.yml`, ni des hyperparamètres LightGBM.
- Pas de commit sur le dépôt sans confirmation explicite.
- Pas de nouveau test exploratoire lancé sans te demander avant.

## 6. Prérequis avant le premier vrai backtest

Volume nécessaire (indicatif, décision finale à Dorian) : un volume de courses avec couverture H-15 ≥ 70% suffisant pour qu'un ROI cumulé ne soit pas dominé par le bruit — de l'ordre de plusieurs dizaines à ~150-200 courses selon la stratégie testée. État au 05/09/2026 : **35 courses** cumulées avec cov_h15_pct ≥ 70% (20 le 29/08, 15 le 30/08, dans `piste4_marche_journalier.csv`) — encore loin du bas de la fourchette indicative, la collecte quotidienne continue.

**Persistance — RÉSOLUE le 04-05/09/2026** (option (a) ci-dessous retenue et mise en œuvre) : le détail par cheval (VERT, Top-5, score) est maintenant persisté chaque jour dans `piste4_scores_detail_journalier.csv` (nouveau fichier, schéma en section 3, ne remplace pas `piste4_marche_journalier.csv`), alimenté automatiquement par la tâche planifiée `piste4-accumulation-marche` (étape 2.g). Initialisé et vérifié avec le 30/08/2026 (279 lignes, 27 courses). Il manque encore, pour joindre ce détail à la cote H-15 exacte par cheval dans un futur run de backtest bout-en-bout, un export équivalent des cotes point-in-time par cheval (actuellement calculées à la volée en session via Supabase, jamais persistées telles quelles) — à trancher avec Dorian si un troisième fichier cumulatif s'avère nécessaire une fois le volume suffisant.

**Restant avant un premier vrai backtest exécuté :**
1. Volume H-15 suffisant (voir ci-dessus, 35 courses à date).
2. Une règle de mise explicitement choisie par Dorian (`BETTING_RULES` reste vide).
3. Le go explicite de Dorian pour lever le verrou de `run_backtest()`.
Rien d'autre ne bloque : le pipeline de sélection (2.1→2.5), la comparaison modèle-seul vs marché, et les métriques (2.10) sont implémentés et testés (données synthétiques) depuis le 05/09/2026.
