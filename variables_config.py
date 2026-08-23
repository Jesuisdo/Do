# -*- coding: utf-8 -*-
"""
variables_config.py — Liste explicite des variables produites par
`variables_historiques.construire_variables`, séparée du code de modélisation
pour pouvoir être testée sans dépendre de scikit-learn/psycopg2 (non
disponibles hors GitHub Actions pour ce projet).

La cohérence entre cette liste et les clés réellement produites par
construire_variables() est vérifiée par test_variables_config.py.
"""

# Colonnes techniques / cible / non utilisées comme predicteur direct
COLONNES_META = [
    "course_id", "date_course", "numero", "horse_uid", "position_arrivee",
    "nb_partants_reel", "partants_declares", "seuil", "hippodrome",
]

VARIABLES_NUMERIQUES = [
    # musique (forme encodee independante de notre historique interne)
    "musique_dernier", "musique_moy3", "musique_moy5", "musique_tendance",
    "musique_nb_incidents", "musique_nb_courses_visibles",
    # carriere globale (fournie par PMU, dispo des le 1er jour)
    "carriere_nb_courses", "carriere_taux_victoire", "carriere_taux_place",
    "gains_carriere", "gains_annee_encours", "gains_annee_precedente",
    # signalement / poids / equipement
    "age", "handicap_poids", "poids_condition_monte", "poids_delta_vs_carriere",
    "oeilleres_presence", "deferre_present", "place_corde",
    # course (contexte connu avant le depart)
    "distance_m", "distance_delta_vs_carriere", "montant_allocation",
    "allocation_delta_vs_carriere", "meteo_temperature", "meteo_force_vent",
    "terrain_valeur_penetrometre", "terrain_score_ordinal",
    # repos
    "jours_repos",
    # forme recente interne, fenetres 5/10/20 (point-in-time)
    "forme_n_courses_internes", "forme_taux_victoire_carriere_interne",
    "forme_taux_place_carriere_interne", "forme_moy_position_5",
    "forme_moy_position_10", "forme_moy_position_20",
    "forme_meilleure_position_5", "forme_ecart_type_position_5",
    "forme_tendance_5_vs_10", "forme_nb_courses_disponibles",
    "forme_a_au_moins_5", "forme_a_au_moins_10", "forme_a_au_moins_20",
    # historiques internes par contexte (nb, taux_victoire, taux_top2, taux_top3)
    "interne_distance_nb", "interne_distance_taux_victoire", "interne_distance_taux_top2", "interne_distance_taux_top3",
    "interne_terrain_nb", "interne_terrain_taux_victoire", "interne_terrain_taux_top2", "interne_terrain_taux_top3",
    "interne_hippo_nb", "interne_hippo_taux_victoire", "interne_hippo_taux_top2", "interne_hippo_taux_top3",
    "interne_categorie_nb", "interne_categorie_taux_victoire", "interne_categorie_taux_top2", "interne_categorie_taux_top3",
    "interne_corde_relative_nb", "interne_corde_relative_taux_victoire", "interne_corde_relative_taux_top2", "interne_corde_relative_taux_top3",
    "interne_jockey_nb", "interne_jockey_taux_victoire", "interne_jockey_taux_top2", "interne_jockey_taux_top3",
    "interne_entraineur_nb", "interne_entraineur_taux_victoire", "interne_entraineur_taux_top2", "interne_entraineur_taux_top3",
    "interne_jockey_entraineur_nb", "interne_jockey_entraineur_taux_victoire", "interne_jockey_entraineur_taux_top2", "interne_jockey_entraineur_taux_top3",
    "interne_jockey_cheval_nb", "interne_jockey_cheval_taux_victoire", "interne_jockey_cheval_taux_top2", "interne_jockey_cheval_taux_top3",
    "interne_entraineur_cheval_nb", "interne_entraineur_cheval_taux_victoire", "interne_entraineur_cheval_taux_top2", "interne_entraineur_cheval_taux_top3",
    "biais_corde_hippo_distance_nb", "biais_corde_hippo_distance_taux_victoire", "biais_corde_hippo_distance_taux_top2", "biais_corde_hippo_distance_taux_top3",
    "interne_proprietaire_nb", "interne_proprietaire_taux_victoire", "interne_proprietaire_taux_top2", "interne_proprietaire_taux_top3",
    "interne_eleveur_nb", "interne_eleveur_taux_victoire", "interne_eleveur_taux_top2", "interne_eleveur_taux_top3",
    # pays d'entrainement
    "entraine_a_letranger",
    # niveau des adversaires (calcule a partir de LEURS stats carriere pre-course, pas du resultat du jour)
    "niveau_moyen_adversaires", "ecart_vs_niveau_moyen_champ", "rang_papier_taux_victoire",
]

VARIABLES_CATEGORIELLES = [
    "categorie_particularite", "condition_age", "condition_sexe", "type_piste",
    "sexe", "distance_bucket", "terrain_bucket", "corde_bucket_relatif",
    "repos_bucket", "race_cheval", "pays_cheval", "robe",
]

TOUTES_LES_CLES_ATTENDUES = set(COLONNES_META) | set(VARIABLES_NUMERIQUES) | set(VARIABLES_CATEGORIELLES)
