import glob
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2


P_MIN = 0.1393
EDGE_MIN = -0.03
SCORES_DIR = os.environ.get("SCORES_DIR", ".")
VERT_THRESHOLD = 0.5848
DB = os.environ["DATABASE_URL"]

def generer_scores():
    date_test = os.environ.get("DATE_ROI")

    if not date_test:
        raise SystemExit(
            "[ROI] DATE_ROI manquante. "
            "Format attendu : YYYY-MM-DD"
        )

    date_compacte = date_test.replace("-", "")
    fichier = f"scores_modele_{date_compacte}.csv"

    if os.path.exists(fichier):
        print(f"[ROI] Scores déjà présents : {fichier}")
        return

    print(f"[ROI] Génération B+généalogie pour {date_test}...")

    env = os.environ.copy()
    env["DATE_TEST_PISTE4"] = date_test

    subprocess.run(
        ["python3", "test_marche_forward_29082026.py"],
        env=env,
        check=True
    )

    if not os.path.exists(fichier):
        raise SystemExit(
            f"[ROI] Le scoring n'a pas créé {fichier}"
        )

    print(f"[ROI] Scores générés : {fichier}")

def charger_scores():
    files = sorted(
        glob.glob(os.path.join(SCORES_DIR, "scores_modele_*.csv"))
    )

    if not files:
        print("[ROI] Aucun scores_modele_*.csv disponible.")
        return None

    df = pd.concat(
        [pd.read_csv(f) for f in files],
        ignore_index=True
    )

    df["numero"] = pd.to_numeric(
        df["numero"], errors="coerce"
    ).astype("Int64")

    df["score_modele"] = pd.to_numeric(
        df["score_modele"], errors="coerce"
    )

    df = df.dropna(
        subset=["course_id", "numero", "score_modele"]
    ).copy()

    df["numero"] = df["numero"].astype(int)

    max_score = (
        df.groupby("course_id")["score_modele"]
        .transform("max")
    )

    df["_exp"] = np.exp(
        df["score_modele"] - max_score
    )

    df["_sum_exp"] = (
        df.groupby("course_id")["_exp"]
        .transform("sum")
    )

    df["p"] = df["_exp"] / df["_sum_exp"]

    return df.drop(
        columns=["_exp", "_sum_exp"]
    )


def charger_cotes(conn, course_ids):
    query = """
        SELECT
            course_id,
            numero,
            horodatage,
            minutes_avant_depart,
            cote,
            source
        FROM cotes_historique
        WHERE source = 'pmu'
          AND minutes_avant_depart BETWEEN 13 AND 17
          AND cote IS NOT NULL
          AND cote > 1
          AND course_id = ANY(%s)
    """

    return pd.read_sql_query(
        query,
        conn,
        params=(course_ids,)
    )


def choisir_h15(df, odds):
    model_sets = (
        df.groupby("course_id")["numero"]
        .apply(lambda s: set(map(int, s)))
        .to_dict()
    )

    odds["numero"] = pd.to_numeric(
        odds["numero"], errors="coerce"
    ).astype("Int64")

    odds = odds.dropna(subset=["numero"]).copy()
    odds["numero"] = odds["numero"].astype(int)

    odds["cote"] = pd.to_numeric(
        odds["cote"], errors="coerce"
    )

    odds["minutes_avant_depart"] = pd.to_numeric(
        odds["minutes_avant_depart"],
        errors="coerce"
    )

    chosen_rows = []

    for cid, runners in model_sets.items():
        oc = odds[odds["course_id"] == cid].copy()

        candidates = []

        for (h, m), g in oc.groupby(
            ["horodatage", "minutes_avant_depart"],
            dropna=False
        ):
            covered = set(g["numero"].astype(int))

            if runners.issubset(covered):
                candidates.append(
                    (
                        abs(float(m) - 15.0),
                        str(h),
                        h,
                        float(m),
                        g
                    )
                )

        if not candidates:
            continue

        candidates.sort(
            key=lambda x: (x[0], x[1])
        )

        best_distance = candidates[0][0]

        tied = [
            x for x in candidates
            if x[0] == best_distance
        ]

        best = sorted(
            tied,
            key=lambda x: x[1],
            reverse=True
        )[0]

        _, _, h, m, g = best

        g = (
            g.sort_values("horodatage")
            .drop_duplicates("numero", keep="last")
        )

        g = g[
            g["numero"].isin(runners)
        ][
            ["course_id", "numero", "cote"]
        ].copy()

        g["minutes_avant_depart"] = m
        g["horodatage"] = h

        chosen_rows.append(g)

    if not chosen_rows:
        return pd.DataFrame()

    return pd.concat(
        chosen_rows,
        ignore_index=True
    )


def creer_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS roi_rule_b_daily (
                date_course date PRIMARY KEY,
                model_courses integer NOT NULL DEFAULT 0,
                strict_h15_courses integer NOT NULL DEFAULT 0,
                selections integer NOT NULL DEFAULT 0,
                wins integer NOT NULL DEFAULT 0,
                stake double precision NOT NULL DEFAULT 0,
                gross double precision NOT NULL DEFAULT 0,
                net double precision NOT NULL DEFAULT 0,
                roi double precision,
                calcule_at timestamptz NOT NULL DEFAULT now()
            )
        """)

    conn.commit()


def enregistrer(conn, date_course, valeurs):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO roi_rule_b_daily (
                date_course,
                model_courses,
                strict_h15_courses,
                selections,
                wins,
                stake,
                gross,
                net,
                roi,
                calcule_at
            )
            VALUES (
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s
            )
            ON CONFLICT (date_course)
            DO UPDATE SET
                model_courses = EXCLUDED.model_courses,
                strict_h15_courses =
                    EXCLUDED.strict_h15_courses,
                selections = EXCLUDED.selections,
                wins = EXCLUDED.wins,
                stake = EXCLUDED.stake,
                gross = EXCLUDED.gross,
                net = EXCLUDED.net,
                roi = EXCLUDED.roi,
                calcule_at = EXCLUDED.calcule_at
        """, (
            date_course,
            valeurs["model_courses"],
            valeurs["strict_h15_courses"],
            valeurs["selections"],
            valeurs["wins"],
            valeurs["stake"],
            valeurs["gross"],
            valeurs["net"],
            valeurs["roi"],
            datetime.now(timezone.utc)
        ))

    conn.commit()


def main():
    print("=== ROI RULE B H15 AUTOMATIQUE ===")
    generer_scores()  
    df = charger_scores()

    if df is None or df.empty:
        return

    course_ids = (
        df["course_id"]
        .drop_duplicates()
        .tolist()
    )

    with psycopg2.connect(DB) as conn:
        creer_table(conn)

        odds = charger_cotes(
            conn,
            course_ids
        )

        if odds.empty:
            print("[ROI] Aucune cote PMU H13-H17.")
            return

        chosen = choisir_h15(
            df,
            odds
        )

        if chosen.empty:
            print("[ROI] Aucune course strict H15.")
            return

        merged = df.merge(
            chosen,
            on=["course_id", "numero"],
            how="inner"
        )
# Confiance de la course = somme des probabilités du Top 3
course_confidence = (
    merged.sort_values(
        ["course_id", "p"],
        ascending=[True, False]
    )
    .groupby("course_id")
    .head(3)
    .groupby("course_id")["p"]
    .sum()
)

green_course_ids = set(
    course_confidence[
        course_confidence >= VERT_THRESHOLD
    ].index
)

merged["is_green"] = merged["course_id"].isin(green_course_ids)

print(
    f"[VERT] {len(green_course_ids)} courses vertes "
    f"(seuil Top3 >= {VERT_THRESHOLD})"
) 

merged["edge"] = (
            merged["p"]
            - 1.0 / merged["cote"]
        )

        bets = merged[
            (merged["p"] >= P_MIN)
            & (merged["edge"] >= EDGE_MIN)
        ].copy()
green_bets = bets[
    bets["is_green"]
].copy()

bets["date_course"] = (
            bets["course_id"]
            .str.slice(0, 10)
        )

        bets["win"] = (
            pd.to_numeric(
                bets["position_arrivee"],
                errors="coerce"
            ) == 1
        ).astype(int)
green_bets["date_course"] = (
    green_bets["course_id"]
    .str.slice(0, 10)
)

green_bets["win"] = (
    pd.to_numeric(
        green_bets["position_arrivee"],
        errors="coerce"
    ) == 1
).astype(int)
        df["date_course"] = (
            df["course_id"]
            .str.slice(0, 10)
        )

        strict_ids = set(
            chosen["course_id"].unique()
        )

        dates = sorted(
            bets["date_course"].unique()
        )

        for date_course in dates:
            day_bets = bets[
                bets["date_course"] == date_course
            ].copy()
            day_green_bets = green_bets[
                green_bets["date_course"] == date_course
            ].copy()
            day_model = df[
                df["date_course"] == date_course
            ]

            strict_day = len([
                cid for cid in strict_ids
                if str(cid).startswith(date_course)
            ])

            selections = len(day_bets)
            wins = int(day_bets["win"].sum())

            stake = float(selections)

            gross = float(
                day_bets.loc[
                    day_bets["win"] == 1,
                    "cote"
                ].sum()
            )

            net = gross - stake

            roi = (
                100.0 * net / stake
                if stake
                else None
            )
        green_selections = len(day_green_bets)
        green_wins = int(day_green_bets["win"].sum())
        green_stake = float(green_selections)

        green_gross = float(
            day_green_bets.loc[
                day_green_bets["win"] == 1,
                "cote"
            ].sum()
        )

        green_net = green_gross - green_stake

        green_roi = (
            100.0 * green_net / green_stake
            if green_stake
            else None
        )

        print(
            f"[VERT ROI] {date_course} | "
            f"bets={green_selections} | "
            f"wins={green_wins} | "
            f"stake={green_stake:.2f} | "
            f"gross={green_gross:.2f} | "
            f"net={green_net:+.2f} | "
            f"ROI={green_roi:.2f}%"
            if green_roi is not None
            else f"[VERT ROI] {date_course} | aucun pari"
        )
         valeurs = {
                "model_courses":
                    int(day_model["course_id"].nunique()),
                "strict_h15_courses":
                    int(strict_day),
                "selections":
                    int(selections),
                "wins":
                    int(wins),
                "stake":
                    stake,
                "gross":
                    gross,
                "net":
                    net,
                "roi":
                    roi
            }

            enregistrer(
                conn,
                date_course,
                valeurs
            )

            print(
                f"[ROI] {date_course} | "
                f"H15={strict_day} | "
                f"bets={selections} | "
                f"wins={wins} | "
                f"stake={stake:.2f} | "
                f"gross={gross:.2f} | "
                f"net={net:+.2f} | "
                f"ROI={roi:.2f}%"
            )

        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COALESCE(SUM(selections),0),
                    COALESCE(SUM(wins),0),
                    COALESCE(SUM(stake),0),
                    COALESCE(SUM(gross),0)
                FROM roi_rule_b_daily
            """)

            selections, wins, stake, gross = cur.fetchone()

        stake = float(stake)
        gross = float(gross)
        net = gross - stake

        roi_global = (
            100.0 * net / stake
            if stake
            else None
        )

        print("")
        print("=== ROI GLOBAL STOCKE ===")
        print(f"Selections : {selections}")
        print(f"Wins       : {wins}")
        print(f"Stake      : {stake:.2f}")
        print(f"Gross      : {gross:.2f}")
        print(f"Net        : {net:+.2f}")

        if roi_global is not None:
            print(f"ROI        : {roi_global:.2f}%")


if __name__ == "__main__":
    main()
