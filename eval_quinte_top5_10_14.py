import re,json,urllib.request
import pandas as pd

sel=pd.read_csv('eval_h15_rule_b_selections.csv')
# Reconstruct model ranking from the full score files downloaded by workflow.
frames=[]
for f in ['scores_modele_20260910.csv','scores_modele_20260911.csv','scores_modele_20260912.csv','scores_modele_20260913.csv','scores_modele_20260914.csv']:
    try: frames.append(pd.read_csv(f))
    except: pass
scores=pd.concat(frames,ignore_index=True)
scorecol=next(c for c in ['score_model','score','proba','probabilite'] if c in scores.columns)
numcol=next(c for c in ['numero','num_cheval','num'] if c in scores.columns)
scores=scores.sort_values(['course_id',scorecol],ascending=[True,False])
top5=scores.groupby('course_id').head(5).groupby('course_id')[numcol].apply(lambda x:[int(v) for v in x]).to_dict()

def reps(date,cid):
 m=re.search(r'_R(\d+)C(\d+)$',cid)
 if not m:return []
 r,c=m.groups(); d=pd.to_datetime(date).strftime('%d%m%Y')
 u=f'https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{d}/R{r}/C{c}/rapports-definitifs?combinaisonEnTableau=true&specialisation=INTERNET'
 try:
  q=urllib.request.Request(u,headers={'User-Agent':'Mozilla/5.0'})
  with urllib.request.urlopen(q,timeout=20) as x:z=json.loads(x.read().decode())
  return z if isinstance(z,list) else z.get('rapportsDefinitifs',z.get('rapports',[]))
 except:return []

def nums(x):
 c=x.get('combinaison',[]); c=c if isinstance(c,list) else [c]; o=[]
 for v in c:
  try:o.append(int(v))
  except:pass
 return o

def pay(r,mb):
 for k in ('dividendePourUnEuro','rapportPourUnEuro'):
  if r.get(k)!=None:return float(r[k])/100
 try:return float(r.get('dividende'))/float(mb or 100)
 except:return None

# unique model courses; API itself tells us whether Quinté reports exist
courses=scores[['course_id']].drop_duplicates().copy(); rows=[]
for cid in courses.course_id:
 sub=scores[scores.course_id==cid]; date=str(sub.iloc[0].get('date',cid[:10])); rr=reps(date,cid)
 q=[]
 for b in rr:
  typ=str(b.get('typePari') or b.get('pari') or '').upper()
  if 'QUINTE' in typ or 'QUINTÉ' in typ:
   rs=b.get('rapports',[]) if isinstance(b.get('rapports'),list) else [b]
   for r in rs:q.append((typ,str(r.get('sousTypeRapport') or r.get('libelle') or r.get('typeRapport') or '').upper(),nums(r),pay(r,b.get('miseBase',200))))
 if not q or cid not in top5:continue
 pred=top5[cid]; ret=0.; hit='PERDU'; official=None
 # exact hierarchy: order, disorder, 4/5, bonus3
 for level in ['ORDRE','DESORDRE','4SUR5','BONUS 3']:
  cand=[]
  for typ,lab,comb,p in q:
   tag=(typ+' '+lab).replace('É','E')
   if level=='ORDRE' and 'DESORDRE' not in tag and 'ORDRE' in tag and pred==comb[:5]:cand.append(p)
   elif level=='DESORDRE' and 'DESORDRE' in tag and set(pred)==set(comb[:5]):cand.append(p)
   elif level=='4SUR5' and ('4SUR5' in tag or '4 SUR 5' in tag) and len(set(pred)&set(comb))>=4:cand.append(p)
   elif level=='BONUS 3' and ('BONUS_3' in tag or 'BONUS 3' in tag) and len(set(pred)&set(comb))>=3:cand.append(p)
  if cand:
   ret=max(x for x in cand if x is not None);hit=level;break
 rows.append({'course_id':cid,'top5':'-'.join(map(str,pred)),'rang_paye':hit,'retour_pour_1e':ret})
out=pd.DataFrame(rows);out.to_csv('eval_quinte_top5.csv',index=False)
N=len(out);gross=out.retour_pour_1e.sum() if N else 0
summary=f'''QUINTE TOP5 MODELE 10-14 SEP\nCOURSES QUINTE AVEC RAPPORT: {N}\nORDRE: {(out.rang_paye=='ORDRE').sum() if N else 0}\nDESORDRE: {(out.rang_paye=='DESORDRE').sum() if N else 0}\nBONUS 4SUR5: {(out.rang_paye=='4SUR5').sum() if N else 0}\nBONUS 3: {(out.rang_paye=='BONUS 3').sum() if N else 0}\nPERDUS: {(out.rang_paye=='PERDU').sum() if N else 0}\nSTAKE NORMALISE 1u/course: {N:.2f}\nGROSS: {gross:.2f}\nNET: {gross-N:.2f}\nROI: {((gross-N)/N*100 if N else float('nan')):.2f}%\n'''
print(summary);open('eval_quinte_top5_summary.txt','w').write(summary)
