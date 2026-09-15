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
    # PMU JSON monetary dividends are integer euro-cents. Explicit per-euro fields,
    # when present, are also normalized from cents. Never divide dividende by miseBase:
    # miseBase itself is in cents and dividende is the payout for that base stake.
    for k in ('dividendePourUnEuro','rapportPourUnEuro'):
        if r.get(k) is not None:
            try: return float(r[k]) / 100.0
            except: pass
    if r.get('rapport_pour_1_euro') is not None:
        try: return float(r['rapport_pour_1_euro'])
        except: pass
    try:
        div=float(r.get('dividende'))
        mb=float(mise_base or 100)
        if mb <= 0: return None
        # div cents paid for mb cents stake => payout for EUR1 = div/mb euros.
        return div/mb
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
            if int(s.numero) in comb_nums(r):
                place=payout1(r,mise); break
        if place is not None: break
    rows.append({**s.to_dict(),'rapport_place_1e':place})

out=pd.DataFrame(rows); out.to_csv('eval_place_rule_b_selections.csv',index=False)
# Only selections for which PMU actually published a Simple Place dividend can be settled.
valid=out[out.rapport_place_1e.notna()].copy()
# A returned placed dividend identifies a payable horse. Non-payable selected horses on a
# covered course must still be losses, so determine course coverage and settle all selections
# from courses where at least one Simple Place block was successfully parsed.
covered_courses=set(valid.course_id)
settled=out[out.course_id.isin(covered_courses)].copy()
settled['place_hit']=settled.rapport_place_1e.notna()
settled['retour_place']=settled.rapport_place_1e.fillna(0.0)
stake=len(settled); gross=settled.retour_place.sum(); roi=(gross-stake)/stake*100 if stake else float('nan')
cote_col=next((c for c in ['cote','cote_h15','odds'] if c in settled.columns),None)
if cote_col:
    settled['retour_gagnant']=settled.apply(lambda r:float(r[cote_col]) if float(r.position_arrivee)==1 else 0,axis=1)
    gp_stake=2*len(settled); gp_gross=(settled.retour_place+settled.retour_gagnant).sum(); gp_roi=(gp_gross-gp_stake)/gp_stake*100
else: gp_stake=gp_gross=gp_roi=float('nan')
summary=f'''SELECTIONS B TOTAL: {len(out)}\nCOURSES AVEC SIMPLE PLACE PMU: {len(covered_courses)}\nSELECTIONS REGLEES: {len(settled)}\nSELECTIONS PLACEES PAYANTES: {int(settled.place_hit.sum())}\nSIMPLE PLACE stake={stake:.2f} gross={gross:.2f} net={gross-stake:.2f} ROI={roi:.2f}%\nGAGNANT+PLACE (1u+1u) stake={gp_stake:.2f} gross={gp_gross:.2f} net={gp_gross-gp_stake:.2f} ROI={gp_roi:.2f}%\n'''
open('eval_place_rule_b_summary.txt','w').write(summary); print(summary)
