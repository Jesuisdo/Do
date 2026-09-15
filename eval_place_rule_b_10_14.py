import re, json, urllib.request
import pandas as pd

sel = pd.read_csv('eval_h15_rule_b_selections.csv')

def rapports(date, course_id):
    m = re.search(r'_R(\d+)C(\d+)$', course_id)
    if not m: return []
    r,c=m.groups(); d=pd.to_datetime(date).strftime('%d%m%Y')
    url=f'https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{d}/R{r}/C{c}/rapports-definitifs?combinaisonEnTableau=true&specialisation=INTERNET'
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req,timeout=20) as x: data=json.loads(x.read().decode())
        return data if isinstance(data,list) else data.get('rapportsDefinitifs', data.get('rapports', []))
    except Exception: return []

def norm_type(x): return str(x or '').upper().replace(' ','_')
def comb_nums(r):
    c=r.get('combinaison',[])
    if isinstance(c,(int,str)): c=[c]
    out=[]
    for x in c:
        try: out.append(int(x))
        except: pass
    return out

def payout1(r, mise_base):
    for k in ('dividendePourUnEuro','rapportPourUnEuro'):
        if r.get(k) is not None:
            try: return float(r[k]) / 100.0
            except: pass
    if r.get('rapport_pour_1_euro') is not None:
        try: return float(r['rapport_pour_1_euro'])
        except: pass
    try:
        div=float(r.get('dividende')); mb=float(mise_base or 100)
        return div/mb if mb > 0 else None
    except: return None

rows=[]; cache={}
for _,s in sel.iterrows():
    date=str(s.get('date', str(s.course_id)[:10])); key=(date,s.course_id)
    if key not in cache: cache[key]=rapports(*key)
    reps=cache[key]; place=None
    for bloc in reps:
        typ=norm_type(bloc.get('typePari') or bloc.get('pari'))
        if 'SIMPLE_PLACE' not in typ and typ not in ('SIMPLE_PLACÉ','SIMPLE_PLACE','E_SIMPLE_PLACE'): continue
        mise=bloc.get('miseBase',100)
        rs=bloc.get('rapports',[]) if isinstance(bloc.get('rapports'),list) else [bloc]
        for r in rs:
            if int(s.numero) in comb_nums(r): place=payout1(r,mise); break
        if place is not None: break
    rows.append({**s.to_dict(),'rapport_place_1e':place})

out=pd.DataFrame(rows)
covered_courses=set(out.loc[out.rapport_place_1e.notna(),'course_id'])
settled=out[out.course_id.isin(covered_courses)].copy()
settled['place_hit']=settled.rapport_place_1e.notna()
settled['retour_place']=settled.rapport_place_1e.fillna(0.0)
cote_col=next((c for c in ['cote','cote_h15','odds'] if c in settled.columns),None)
if not cote_col: raise RuntimeError('Colonne cote H15 absente')
settled['retour_gagnant']=settled.apply(lambda r:float(r[cote_col]) if float(r.position_arrivee)==1 else 0.0,axis=1)
settled.to_csv('eval_place_rule_b_selections.csv',index=False)

n=len(settled)
# Exact same selections, three strategies.
g_stake=n; g_gross=settled.retour_gagnant.sum(); g_net=g_gross-g_stake; g_roi=g_net/g_stake*100 if n else float('nan')
p_stake=n; p_gross=settled.retour_place.sum(); p_net=p_gross-p_stake; p_roi=p_net/p_stake*100 if n else float('nan')
gp_stake=2*n; gp_gross=(settled.retour_gagnant+settled.retour_place).sum(); gp_net=gp_gross-gp_stake; gp_roi=gp_net/gp_stake*100 if n else float('nan')
summary=f'''COMPARAISON STRICTE MEMES SELECTIONS\nCOURSES COUVERTES: {len(covered_courses)}\nSELECTIONS COMMUNES: {n}\nGAGNANTS: {int((settled.position_arrivee.astype(float)==1).sum())}\nPLACES PAYANTES: {int(settled.place_hit.sum())}\nSIMPLE GAGNANT stake={g_stake:.2f} gross={g_gross:.2f} net={g_net:.2f} ROI={g_roi:.2f}%\nSIMPLE PLACE stake={p_stake:.2f} gross={p_gross:.2f} net={p_net:.2f} ROI={p_roi:.2f}%\nGAGNANT+PLACE (1u+1u) stake={gp_stake:.2f} gross={gp_gross:.2f} net={gp_net:.2f} ROI={gp_roi:.2f}%\n'''
open('eval_place_rule_b_summary.txt','w').write(summary); print(summary)
