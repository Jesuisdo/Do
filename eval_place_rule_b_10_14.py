import os, glob, re, json, urllib.request
import pandas as pd

# Selections exactes produites par l'evaluation H15 precedente.
sel = pd.read_csv('eval_h15_rule_b_selections.csv')

# PMU online = masse internet, celle pertinente pour un pari en ligne.
def rapports(date, course_id):
    m = re.search(r'_R(\d+)C(\d+)$', course_id)
    if not m: return []
    r,c=m.groups(); d=pd.to_datetime(date).strftime('%d%m%Y')
    url=f'https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{d}/R{r}/C{c}/rapports-definitifs?combinaisonEnTableau=true&specialisation=INTERNET'
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req,timeout=20) as x: data=json.loads(x.read().decode())
        return data if isinstance(data,list) else data.get('rapportsDefinitifs', data.get('rapports', []))
    except Exception as e:
        return []

def norm_type(x): return str(x or '').upper().replace(' ','_')
def comb_nums(r):
    c=r.get('combinaison',[])
    if isinstance(c,(int,str)): c=[c]
    out=[]
    for x in c:
        try: out.append(int(x))
        except: pass
    return out

def payout1(r):
    for k in ('dividendePourUnEuro','rapportPourUnEuro','rapport_pour_1_euro'):
        if r.get(k) is not None:
            try: return float(r[k])
            except: pass
    # PMU dividende est souvent en centimes pour mise de base; utiliser seulement si miseBase connue.
    try:
        div=float(r.get('dividende')); mb=float(r.get('_miseBase',100)); return div/mb
    except: return None

rows=[]; cache={}
for _,s in sel.iterrows():
    key=(str(s.get('date', str(s.course_id)[:10])),s.course_id)
    if key not in cache: cache[key]=rapports(*key)
    reps=cache[key]
    place=None
    for bloc in reps:
        typ=norm_type(bloc.get('typePari') or bloc.get('pari'))
        mise=bloc.get('miseBase',100)
        rs=bloc.get('rapports',[]) if isinstance(bloc.get('rapports'),list) else [bloc]
        if 'SIMPLE_PLACE' in typ or typ in ('SIMPLE_PLACÉ','SIMPLE_PLACE'):
            for r in rs:
                rr=dict(r); rr['_miseBase']=mise
                if int(s.numero) in comb_nums(rr):
                    place=payout1(rr); break
        if place is not None: break
    rows.append({**s.to_dict(),'rapport_place_1e':place})

out=pd.DataFrame(rows)
out.to_csv('eval_place_rule_b_selections.csv',index=False)
valid=out[out.rapport_place_1e.notna()].copy()
valid['place_hit']=valid['position_arrivee'].fillna(99).astype(float)<=3
valid['retour_place']=valid.apply(lambda r:r.rapport_place_1e if r.place_hit else 0,axis=1)
# 1 euro simple place par selection
stake=len(valid); gross=valid.retour_place.sum(); roi=(gross-stake)/stake*100 if stake else float('nan')
# Gagnant-place: 1 euro gagnant + 1 euro place. cote = H15 gagnante deja stockee dans selection.
# detecte colonne cote selon script precedent
cote_col=next((c for c in ['cote','cote_h15','odds'] if c in valid.columns),None)
if cote_col:
    valid['retour_gagnant']=valid.apply(lambda r:float(r[cote_col]) if float(r.position_arrivee)==1 else 0,axis=1)
    gp_stake=2*len(valid); gp_gross=(valid.retour_place+valid.retour_gagnant).sum(); gp_roi=(gp_gross-gp_stake)/gp_stake*100
else:
    gp_stake=gp_gross=gp_roi=float('nan')
summary=f'''SELECTIONS B TOTAL: {len(out)}\nSELECTIONS AVEC RAPPORT PLACE: {len(valid)}\nSIMPLE PLACE stake={stake:.2f} gross={gross:.2f} net={gross-stake:.2f} ROI={roi:.2f}%\nGAGNANT+PLACE (1u+1u) stake={gp_stake:.2f} gross={gp_gross:.2f} net={gp_gross-gp_stake:.2f} ROI={gp_roi:.2f}%\n'''
open('eval_place_rule_b_summary.txt','w').write(summary)
print(summary)
