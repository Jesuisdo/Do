import glob, os
import numpy as np
import pandas as pd
import psycopg2

P_MIN = 0.1393
EDGE_MIN = -0.03
SCORES_DIR = os.environ.get('SCORES_DIR', 'scoring')
DB = os.environ['DATABASE_URL']

files = sorted(glob.glob(os.path.join(SCORES_DIR, 'scores_modele_*.csv')))
if not files:
    raise SystemExit('Aucun scores_modele_*.csv trouve')

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df['numero'] = pd.to_numeric(df['numero'], errors='coerce').astype('Int64')
df['score_modele'] = pd.to_numeric(df['score_modele'], errors='coerce')
df = df.dropna(subset=['course_id','numero','score_modele']).copy()
df['numero'] = df['numero'].astype(int)

def add_softmax(g):
    x = g['score_modele'].to_numpy(float)
    e = np.exp(x - np.max(x))
    g = g.copy()
    g['p'] = e / e.sum()
    return g

df = df.groupby('course_id', group_keys=False).apply(add_softmax, include_groups=False).reset_index(drop=True)
course_ids = df['course_id'].drop_duplicates().tolist()
model_sets = df.groupby('course_id')['numero'].apply(lambda s: set(map(int,s))).to_dict()

with psycopg2.connect(DB) as conn:
    q = '''
    SELECT course_id, numero, horodatage, minutes_avant_depart, cote, source
    FROM cotes_historique
    WHERE source='pmu'
      AND minutes_avant_depart BETWEEN 13 AND 17
      AND cote IS NOT NULL AND cote > 1
      AND course_id = ANY(%s)
    '''
    odds = pd.read_sql_query(q, conn, params=(course_ids,))

if odds.empty:
    raise SystemExit('Aucune cote PMU H13-H17 trouvee')
odds['numero'] = pd.to_numeric(odds['numero'], errors='coerce').astype('Int64')
odds = odds.dropna(subset=['numero']).copy()
odds['numero'] = odds['numero'].astype(int)
odds['cote'] = pd.to_numeric(odds['cote'], errors='coerce')
odds['minutes_avant_depart'] = pd.to_numeric(odds['minutes_avant_depart'], errors='coerce')

chosen_rows = []
status_rows = []
for cid, runners in model_sets.items():
    oc = odds[odds.course_id == cid].copy()
    candidates = []
    for (h, m), g in oc.groupby(['horodatage','minutes_avant_depart'], dropna=False):
        covered = set(g['numero'].astype(int))
        if runners.issubset(covered):
            candidates.append((abs(float(m)-15.0), -float(m), str(h), h, float(m), g))
    if not candidates:
        status_rows.append({'course_id':cid,'strict_h15':0,'minutes_avant_depart':np.nan,'horodatage':''})
        continue
    # closest to H15; on exact tie, keep latest horodatage
    candidates.sort(key=lambda z:(z[0], z[2]), reverse=False)
    bestdist = candidates[0][0]
    tied = [z for z in candidates if z[0] == bestdist]
    best = sorted(tied, key=lambda z:z[2], reverse=True)[0]
    _,_,_,h,m,g = best
    g = g.sort_values('horodatage').drop_duplicates('numero', keep='last')
    g = g[g['numero'].isin(runners)][['course_id','numero','cote']].copy()
    g['minutes_avant_depart'] = m
    g['horodatage'] = h
    chosen_rows.append(g)
    status_rows.append({'course_id':cid,'strict_h15':1,'minutes_avant_depart':m,'horodatage':h})

chosen = pd.concat(chosen_rows, ignore_index=True) if chosen_rows else pd.DataFrame(columns=['course_id','numero','cote','minutes_avant_depart','horodatage'])
status = pd.DataFrame(status_rows)
merged = df.merge(chosen, on=['course_id','numero'], how='inner')
merged['edge'] = merged['p'] - 1.0/merged['cote']
bets = merged[(merged['p'] >= P_MIN) & (merged['edge'] >= EDGE_MIN)].copy()
bets['date'] = bets['course_id'].str.slice(0,10)
bets['win'] = (pd.to_numeric(bets['position_arrivee'], errors='coerce') == 1).astype(int)

model_courses = df.course_id.nunique()
strict_courses = status.strict_h15.sum()
sel = len(bets)
wins = int(bets.win.sum()) if sel else 0
stake_h = float(sel)
gross_h = float((bets.loc[bets.win==1,'cote']).sum()) if sel else 0.0
net_h = gross_h - stake_h
roi_h = 100*net_h/stake_h if stake_h else np.nan
avg_odds = float(bets.cote.mean()) if sel else np.nan

if sel:
    bets['n_course'] = bets.groupby('course_id')['numero'].transform('count')
    bets['stake_course_split'] = 1.0 / bets['n_course']
    bets['gross_course_split'] = np.where(bets.win==1, bets.cote / bets.n_course, 0.0)
    cb = bets.groupby('course_id').agg(n_bets=('numero','count'),wins=('win','sum'),gross=('gross_course_split','sum')).reset_index()
    betting_courses = len(cb)
    course_hits = int((cb.wins>0).sum())
    stake_c = float(betting_courses)
    gross_c = float(cb.gross.sum())
    net_c = gross_c - stake_c
    roi_c = 100*net_c/stake_c if stake_c else np.nan
    course_hit = 100*course_hits/betting_courses if betting_courses else np.nan
else:
    betting_courses=course_hits=0; stake_c=gross_c=net_c=0.0; roi_c=course_hit=np.nan

lines = [
    f'Model courses: {model_courses}',
    f'Strict H15 courses: {int(strict_courses)}',
    f'Rule B selections: {sel}',
    f'Rule B wins: {wins}',
    f'Win rate: {(100*wins/sel if sel else float("nan")):.2f}%',
    f'Average odds: {avg_odds:.3f}',
    f'1u/horse stake: {stake_h:.2f}',
    f'1u/horse gross: {gross_h:.2f}',
    f'1u/horse net: {net_h:.2f}',
    f'1u/horse ROI: {roi_h:.2f}%',
    f'Betting courses: {betting_courses}',
    f'Course hits: {course_hits}',
    f'Course hit rate: {course_hit:.2f}%',
    f'1u/course split stake: {stake_c:.2f}',
    f'1u/course split gross: {gross_c:.2f}',
    f'1u/course split net: {net_c:.2f}',
    f'1u/course split ROI: {roi_c:.2f}%',
]

if sel:
    daily=[]
    for d,g in bets.groupby('date'):
        st=len(g); wi=int(g.win.sum()); gr=float(g.loc[g.win==1,'cote'].sum()); ro=100*(gr-st)/st if st else np.nan
        daily.append(f'{d}: selections={st}, wins={wi}, ROI_horse={ro:.2f}%')
    lines += ['','Daily:'] + daily

summary='\n'.join(lines)
print(summary)
open('eval_h15_rule_b_summary.txt','w',encoding='utf-8').write(summary+'\n')
bets.sort_values(['course_id','p'], ascending=[True,False]).to_csv('eval_h15_rule_b_selections.csv', index=False)
status.sort_values('course_id').to_csv('eval_h15_course_status.csv', index=False)
