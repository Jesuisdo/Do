# Préparation du passage de cap — module marché, erreurs historiques, architectures de décision

Document de pré-enregistrement (05/09/2026), rédigé **avant** tout accès aux résultats d'un backtest sur les cotes H-15. Objectif explicite de Dorian : que le passage à un gros backtest, une fois le volume suffisant (~150-200 courses à `cov_h15_pct >= 70%`), ne nécessite aucune préparation supplémentaire ni aucun choix improvisé après coup. Aucun élément de ce document n'a été calibré ou ajusté sur les 35 courses H-15 actuellement accumulées : chantier 1 et 3 sont des spécifications pures (aucune donnée testée), chantier 2 s'appuie exclusivement sur des diagnostics déjà exécutés en VALIDATION (VAL_CALIB, 798-826 courses), totalement disjoints de l'échantillon H-15.

Statut : aucun entraînement lancé, aucun seuil optimisé, aucune modification de `piste4_backtest_engine.py`, de `piste4_marche_journalier.csv`, ni du pipeline de production. `run_backtest()` reste verrouillé.

## Chantier 1 — Module marché H-15 : variables exploitables sans fuite

Toutes les variables ci-dessous respectent la méthodologie point-in-time déjà figée (section 2.4 de `piste4_backtest_architecture.md`) : `minutes_avant_depart >= w`, on prend le minimum de `minutes_avant_depart` parmi les lignes qualifiées (la ligne la plus récente disponible au moment ou avant H-w), jamais de logique "closest to target", dédoublonnage par cheval avant tout calcul de rang, couverture minimale 70% des partants sinon la course est exclue de cet étage. La table source est `cotes_historique` (Supabase, colonnes : `course_id`, `numero`, `horodatage`, `cote`, `cote_reference`, `tendance`, `favori`, `minutes_avant_depart`).

**1. `cote_h15`** — cote captée à la fenêtre H-15 telle que définie ci-dessus. Déjà implémentée dans `piste4_backtest_engine.py` (`compute_point_in_time_odds`). Aucune fuite : jamais une cote captée après H-15.

**2. `rang_cote_h15`** — rang croissant de `cote_h15` (1 = plus petite cote = favori) au sein du champ de partants ayant une `cote_h15` valide pour la course. Calculé uniquement sur les chevaux couverts par la fenêtre H-15 (pas sur le champ entier si couverture partielle) pour ne pas biaiser le rang par des non-couverts.

**3. `favori_h15`** — booléen, `rang_cote_h15 == 1`. Alternative : réutiliser directement la colonne `favori` déjà calculée par la collecte à l'horodatage retenu pour H-15 (à vérifier qu'elle est bien calculée sur le même snapshot que celui retenu par la règle point-in-time — sinon recalculer localement comme ci-dessus pour rester cohérent avec la convention H-15 du moteur).

**4. `ecart_cote_top5`** — parmi les chevaux du Top-5 VERT (déjà sélectionnés par le modèle, indépendamment du marché), écart entre `cote_h15` du moins coté et `cote_h15` du deuxième moins coté du Top-5. Défini en absolu (`cote_h15_2e - cote_h15_1er`) et en relatif (`(cote_h15_2e - cote_h15_1er) / cote_h15_1er`). Mesure la marge de confiance du marché entre son favori et son second choix, restreinte au sous-ensemble déjà validé par le modèle (cohérent avec la contrainte "marché restreint au Top-5 VERT" déjà actée dans `top1_marche_h15_dans_top5_vert_strategy`).

**5. `accord_desaccord_modele_marche`** — variable catégorielle à 3 niveaux, calculée uniquement sur le Top-5 VERT :
   - `accord_total` : le cheval `rang_modele == 1` est aussi celui avec `rang_cote_h15` minimal parmi le Top-5 (favori du marché parmi les 5 retenus par le modèle).
   - `desaccord_partiel` : le favori marché du Top-5 est classé `rang_modele` 2 à 5.
   - `desaccord_total` : réservé si on veut distinguer un cas où l'écart de rang est maximal (`rang_modele == 5`) — optionnel, à activer seulement si le volume le justifie pour ne pas fragmenter inutilement les catégories.

**6. `evolution_cote_h15` (évolution disponible avant H-15)** — deux variantes, toutes deux strictement antérieures à H-15 :
   - `delta_cote_h60_h15` : `cote_h15 - cote_h60` (cote captée à la fenêtre H-60, même méthodologie point-in-time avec `w=60`). Négatif = le cheval s'est resserré (plus parié) entre H-60 et H-15.
   - `tendance_h15` : réutilisation directe de la colonne `tendance` déjà calculée par la collecte au snapshot retenu pour H-15 (si son mode de calcul est déjà point-in-time — à vérifier dans `collect_live_odds_render.py` avant de la considérer fiable ; sinon ne pas l'utiliser telle quelle).
   Vérifié empiriquement sur `cotes_historique` : la profondeur de collecte va typiquement jusqu'à H-700/H-720 avec des dizaines de snapshots par cheval, donc `cote_h60` est disponible dans l'immense majorité des cas où `cote_h15` l'est.

**7. `relation_score_geneal_cote`** — deux mesures, purement descriptives (aucun seuil, aucun modèle) :
   - `correlation_rang_modele_rang_cote` : corrélation de Spearman entre `rang_modele` et `rang_cote_h15` sur le champ Top-5 VERT d'un jour donné (mesure agrégée, pas par course individuelle vu le faible nombre de points par course).
   - `proba_implicite_marche` : `1 / cote_h15` normalisée (softmax ou simple division par la somme des `1/cote_h15` du champ couvert) à comparer à la pseudo-probabilité déjà existante `softmax(score_geneal)` utilisée pour `somme_top3_proba`. Permet de comparer directement confiance modèle vs confiance marché sur la même échelle, sans construire de score combiné (ça, c'est le chantier 3).

**8. Autres informations de marché disponibles avant H-15 (recensées, non encore priorisées)** :
   - `couverture_h15_pct` : déjà existante (`cov_h15_pct`), rappelée ici car c'est la donnée de fiabilité de toutes les variables ci-dessus.
   - `dispersion_cotes_h15` : écart-type ou coefficient de variation des `cote_h15` sur le champ entier couvert (mesure si le marché est tranché ou hésitant sur la course entière, pas seulement le Top-5).
   - `rang_cote_h15_du_pick_modele` : le rang de cote (sur le champ entier, pas restreint au Top-5) du cheval `rang_modele == 1`. Différent de `accord_desaccord_modele_marche` qui est restreint au Top-5 — celui-ci donne une mesure continue même quand il n'y a pas accord.
   - `n_partants_avec_cote_h15` : taille du dénominateur derrière `couverture_h15_pct`, utile pour distinguer une course à 70% de couverture sur 20 partants d'une course à 70% sur 6 partants.

Aucune de ces variables n'a été calculée sur les 35 courses actuelles ni sur un échantillon quelconque — ce sont des définitions, pas des résultats.

## Chantier 2 — Erreurs historiques du Top-1 B+généalogie : synthèse de l'existant

Aucune requête nouvelle sur le modèle, aucun réentraînement. Ce qui suit provient de deux runs déjà exécutés et déjà conclus (GitHub Actions, artefacts encore disponibles), sur des populations de VALIDATION disjointes des 35 courses H-15 :

**Diagnostic marge #1/#2 (piste7, run du 01/09/2026, 826 courses VERT de VAL_CALIB)** :
- Taux d'erreur Top-1 global : 64.4% (532/826) — cohérent avec le Top-1 VERT ~35-36% déjà documenté.
- Répartition par zone de marge softmax #1/#2 (quartiles figés sur VAL_FIT, jamais retouchés) : Top-1 monte de 28.4% (marge très faible) à 47.0% (marge forte) — un gradient existe, mais reste modeste.
- Les erreurs Top-1 sont concentrées à 54.3% dans les deux zones de marge les plus faibles, qui ne représentent que 49.2% des courses — sur-représentation faible (+5pt), pas un signal fort.
- Corrélation de Spearman marge/succès Top-1 : rho = 0.174 (positif mais faible), et la séquence par décile n'est pas monotone (4 inversions sur 9 transitions).
- Conclusion déjà tirée par ce rapport lui-même : signal présent mais trop faible et non monotone pour justifier une correction ciblée sur la zone d'ambiguïté (hypothèse "option A" non confirmée). Recommandation déjà écrite : ne pas poursuivre cette piste spécifique en l'état.

**Piste "forme récente des adversaires" (piste7 étape 2, run du 01/09/2026, 798 courses VERT VAL_CALIB)** : gain Top-1 de +0.4pt (37.7% → 38.1%), Top-3 en légère baisse (-0.3pt). Conclusion déjà actée : rejeté (gain non significatif, cf. protocole "rejeter si le Top-1 ne progresse pas ou si le gain est faible/instable").

**Piste "handicap_valeur" (piste7, run du 01/09/2026, 798 courses VERT VAL_CALIB)** : la variable brute n'apporte rien (Top-1 inchangé, Top-3 -0.5pt). La version rang/z-score dans le champ (candidat B) montre un Top-3 +1.4pt (72.4% → 73.8%) mais un Top-1 strictement inchangé (37.7% → 37.7%) — c'est-à-dire que la variable réordonne un peu le Top-3/Top-5 sans jamais changer qui gagne le plus souvent en position 1. Signal réel mais qui n'aide pas la question posée par Dorian (le Top-1 qui perd).

**Ce qui n'existe pas encore et resterait à faire (identifié, pas exécuté)** : aucun rapport déjà produit ne ventile les erreurs Top-1 par profil concret (distance, corde, discipline, hippodrome, taille du champ, catégorie de course/handicap au sens `categorie_particularite`, ou cheval favori-papier vs outsider). Une vraie réponse à "le modèle surévalue/sous-évalue certains profils" demanderait une analyse purement descriptive (comptage de taux de réussite Top-1 par sous-groupe, sur le même échantillon VAL_CALIB déjà utilisé ci-dessus, sans aucun entraînement) — faisable rapidement le jour où on rouvre le checkpoint de validation, mais non faite à ce stade pour ne pas multiplier les runs GitHub Actions (chaque run recharge ~980k lignes et réentraîne B+généalogie, ~7-8 min et ~260 Mo de checkpoint) sans validation préalable que c'est bien ce que tu veux prioriser.

## Chantier 3 — Architectures de décision conceptuelles (à ne pas tester maintenant)

Les 4 stratégies ci-dessous sont des spécifications. Aucune n'a été exécutée, aucun paramètre n'a été choisi par observation de résultats.

**A. Marché comme confirmation**
- Données nécessaires : `rang_modele`, Top-5 VERT, `cote_h15`, `rang_cote_h15` (chantier 1).
- Règle : parier le cheval `rang_modele == 1` uniquement si `accord_desaccord_modele_marche == accord_total` (le marché confirme le choix du modèle). Sinon, ne pas parier (`pari_place = 0`).
- Métriques : Top-1/Top-3/Top-5 (sur le sous-ensemble joué), winrate, ROI, rendement cumulé, drawdown max, nombre de paris, cote moyenne, taux de courses jouées (celui-ci sera par construction plus bas que 100%, à mesurer précisément).
- Risque d'overfitting : faible sur la règle elle-même (binaire, aucun seuil réglable), mais le taux de courses jouées peut être très bas avec seulement 150-200 courses — risque d'un échantillon de paris trop restreint pour conclure (à surveiller via le nombre de paris, pas à corriger en resserrant/élargissant la règle après coup).

**B. Marché comme arbitre en cas de désaccord**
- Données nécessaires : idem A, plus une cote minimale acceptable (à définir par Dorian, cf. `cote_min_acceptable` déjà prévu en section 2.6 de l'architecture).
- Règle : si `accord_total`, parier le cheval du modèle. Si `desaccord_partiel`, basculer sur le favori marché du Top-5 (celui avec `rang_cote_h15` minimal) à condition que sa cote reste au-dessus de `cote_min_acceptable` ; sinon ne pas parier.
- Métriques : identiques à A, plus une comparaison explicite du sous-ensemble "arbitré" (desaccord_partiel) vs le sous-ensemble "confirmé" (accord_total), pour voir si l'arbitrage ajoute de la valeur ou en détruit.
- Risque d'overfitting : plus élevé que A car il introduit un paramètre libre (`cote_min_acceptable`) — ce paramètre doit être fixé par Dorian AVANT le backtest et non ajusté en fonction du ROI obtenu, sans quoi la comparaison à la stratégie A perd toute validité.

**C. Score modèle/marché combiné**
- Données nécessaires : `score_geneal` (softmax déjà existant), `proba_implicite_marche` (chantier 1, point 7).
- Règle : score combiné `= poids * proba_modele + (1 - poids) * proba_implicite_marche`, le cheval retenu est celui du Top-5 VERT avec le score combiné maximal. `poids` est un paramètre à fixer AVANT le backtest (par exemple 0.5, ou une valeur choisie par Dorian) — jamais recherché par grid-search sur l'échantillon de test, ce qui serait un réentraînement déguisé et violerait directement la consigne "aucune optimisation après coup".
- Métriques : identiques à A/B, plus la comparaison à un score modèle seul et à un score marché seul (`proba_implicite_marche` pure) pour vérifier que la combinaison apporte réellement quelque chose au-delà de chaque composant pris isolément.
- Risque d'overfitting : le plus élevé des quatre — c'est la seule stratégie avec un paramètre continu. À isoler explicitement de toute recherche de valeur optimale ; si Dorian veut un jour l'optimiser, cela devra se faire sur un échantillon dédié, distinct de celui du gros backtest, jamais sur le même.

**D. Filtrage des situations trop incertaines**
- Données nécessaires : `ecart_cote_top5`, `dispersion_cotes_h15`, `gap_confiance` (déjà existant).
- Règle : ne pas parier (indépendamment du choix du cheval) si le marché est jugé trop indécis — par exemple `ecart_cote_top5` relatif en dessous d'un seuil, ou `dispersion_cotes_h15` en dessous d'un seuil, éventuellement combiné à `faible_confiance == 1` (déjà calculé). Le cheval joué quand la course n'est pas filtrée reste `rang_modele == 1` (cette stratégie ne change pas le choix du cheval, seulement la décision de jouer ou non).
- Métriques : identiques à A, avec une attention particulière au taux de courses jouées (cette stratégie le réduit par construction) et à la comparaison winrate/ROI du sous-ensemble filtré vs non filtré.
- Risque d'overfitting : élevé si les seuils de filtrage sont choisis en observant quelles courses filtrées auraient été perdantes — les seuils doivent être fixés par une logique indépendante du résultat (par exemple une valeur ronde ou un quantile calculé sur une période disjointe), jamais en cherchant a posteriori le seuil qui améliore le ROI sur l'échantillon du backtest.

**Point commun aux 4 stratégies** : toutes se limitent au Top-5 VERT déjà produit par B+généalogie (aucune ne remet en cause la sélection amont), toutes utilisent uniquement des variables de marché disponibles strictement avant H-15 (chantier 1), et pour aucune un paramètre libre n'a reçu de valeur choisie dans ce document — les valeurs (poids, seuils, cote minimale) restent des décisions explicites de Dorian, à prendre avant le lancement du gros backtest et non après avoir vu les résultats.

## Ce qui reste à décider par Dorian avant le gros backtest

1. Confirmer ou amender les 4 architectures ci-dessus (en ajouter, en retirer, en préciser les paramètres).
2. Fixer les valeurs des paramètres libres identifiés (poids de la stratégie C, seuils de la stratégie D, `cote_min_acceptable` de la stratégie B) — décision à prendre sans avoir vu le ROI de chacune.
3. Décider si l'analyse descriptive par profil (fin du chantier 2) est une priorité avant le gros backtest ou peut attendre après.
4. Le volume H-15 (35 courses actuellement, objectif 150-200) et le feu vert pour lever `run_backtest()`.

Rien d'autre ne bloque : le jour où le volume est atteint et les paramètres ci-dessus fixés, le gros backtest peut être lancé sans travail de préparation supplémentaire.
