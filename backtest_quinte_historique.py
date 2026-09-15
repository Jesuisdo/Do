import os,re,json,requests,subprocess
from datetime import date,timedelta
import pandas as pd

START=pd.Timestamp(os.getenv('DATE_DEBUT','2026-01-01'))
END=pd.Timestamp(os.getenv('DATE_FIN','2026-09-14'))
rows=[]

def pmu_reports(day,r,c):
 d=day.strftime('%d%m%Y')
 u=f'https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{d}/R{r}/C{c}/rapports-definitifs?combinaisonEnTableau=true&specialisation=INTERNET'
 try:
  z=requests.get(u,timeout=15,headers={'User-Agent':'Mozilla/5.0'}).json()
  return z if isinstance(z,list) else z.get('rapportsDefinitifs',z.get('rapports',[]))
 except:return []

def program(day):
 d=day.strftime('%d%m%Y')
 u=f'https://online.turfinfo.api.pmu.fr/rest/client/1/programme/{d}?meteo=true&specialisation=INTERNET'
 try:return requests.get(u,timeout=20,headers={'User-Agent':'Mozilla/5.0'}).json()
 except:return {}

def get_quinte_courses(z):
 out=[]
 for reunion in z.get('programme',{}).get('reunions',z.get('reunions',[])):
  r=reunion.get('numOfficiel') or reunion.get('numReunion')
  for c in reunion.get('courses',[]):
   n=c.get('numOrdre') or c.get('numCourse')
   paris=c.get('paris') or c.get('parisDisponibles') or []
   s=json.dumps(paris,ensure_ascii=False).upper()
   discipline=str(c.get('discipline','')).upper()
   if ('QUINTE' in s or 'QUINTÉ' in s) and ('PLAT' in discipline or discipline==''):
    out.append((int(r),int(n)))
 return out

def nums(x):
 c=x.get('combinaison',[]); c=c if isinstance(c,list) else [c]
 o=[]
 for v in c:
  try:o.append(int(v))
  except:pass
 return o

def payout(r,b):
 for k in ('dividendePourUnEuro','rapportPourUnEuro'):
  if r.get(k) is not None:return float(r[k])/100
 try:return float(r.get('dividende'))/float(b.get('miseBase') or 100)
 except:return None

def score_day(day):
 ds=day.strftime('%Y-%m-%d'); fn=f'scores_modele_{day.strftime("%Y%m%d")}.csv'
 if os.path.exists(fn):return pd.read_csv(fn)
 env=os.environ.copy();env['DATE_TEST_PISTE4']=ds
 try:
  p=subprocess.run(['python','test_marche_forward_29082026.py'],env=env,capture_output=True,text=True,timeout=900)
  if os.path.exists(fn):return pd.read_csv(fn)
 except:pass
 return None

cur=START
while cur<=END:
 qs=get_quinte_courses(program(cur))
 if qs:
  sc=score_day(cur)
  if sc is not None and len(sc):
   scol='score_modele' if 'score_modele' in sc.columns else None
   if scol:
    for r,c in qs:
     cid_match=sc[sc.course_id.astype(str).str.endswith(f'_R{r}C{c}')].sort_values(scol,ascending=False)
     if len(cid_match)<5:continue
     pred=[int(x) for x in cid_match.head(5).numero]
     rr=pmu_reports(cur,r,c); q=[]
     for b in rr:
      typ=str(b.get('typePari') or b.get('pari') or '').upper().replace('É','E')
      if 'QUINTE' not in typ:continue
      rs=b.get('rapports',[]) if isinstance(b.get('rapports'),list) else [b]
      for x in rs:
       lab=str(x.get('sousTypeRapport') or x.get('libelle') or x.get('typeRapport') or '').upper().replace('É','E')
       q.append((typ+' '+lab,nums(x),payout(x,b)))
     if not q:continue
     hit='PERDU';ret=0.
     for lev in ['ORDRE','DESORDRE','4SUR5','BONUS 3']:
      cand=[]
      for tag,comb,pay in q:
       ok=(lev=='ORDRE' and 'DESORDRE' not in tag and 'ORDRE' in tag and pred==comb[:5]) or (lev=='DESORDRE' and 'DESORDRE' in tag and set(pred)==set(comb[:5])) or (lev=='4SUR5' and ('4SUR5' in tag or '4 SUR 5' in tag) and len(set(pred)&set(comb))>=4) or (lev=='BONUS 3' and ('BONUS_3' in tag or 'BONUS 3' in tag) and len(set(pred)&set(comb))>=3)
       if ok and pay is not None:cand.append(pay)
      if cand:hit=lev;ret=max(cand);break
     rows.append({'date':cur.strftime('%Y-%m-%d'),'course_id':cid_match.iloc[0].course_id,'top5':'-'.join(map(str,pred)),'rang_paye':hit,'retour_pour_1e':ret})
 cur+=pd.Timedelta(days=1)

out=pd.DataFrame(rows);out.to_csv('backtest_quinte_historique.csv',index=False)
N=len(out);gross=out.retour_pour_1e.sum() if N else 0
cnt=lambda x:int((out.rang_paye==x).sum()) if N else 0
summary=f'''BACKTEST QUINTE HISTORIQUE B+GENEALOGIE\nPERIODE: {START.date()} -> {END.date()}\nCOURSES EXPLOITABLES: {N}\nORDRE: {cnt('ORDRE')}\nDESORDRE: {cnt('DESORDRE')}\nBONUS 4SUR5: {cnt('4SUR5')}\nBONUS 3: {cnt('BONUS 3')}\nPERDUS: {cnt('PERDU')}\nSTAKE 1u/QUINTE: {N:.2f}\nGROSS: {gross:.2f}\nNET: {gross-N:.2f}\nROI: {((gross-N)/N*100 if N else float('nan')):.2f}%\n'''
print(summary);open('backtest_quinte_historique_summary.txt','w').write(summary)
