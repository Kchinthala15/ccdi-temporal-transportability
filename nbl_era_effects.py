import os; os.makedirs('work',exist_ok=True)
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
d=pd.read_csv('work/nbl_cohort.csv')
d['year']=pd.to_numeric(d['diagnoses.year_of_diagnosis'],errors='coerce')
d['age']=pd.to_numeric(d['diagnoses.age_at_diagnosis'],errors='coerce')
d['t']=pd.to_numeric(d.efs_days,errors='coerce')
core=d.t.notna()&d.first_event.notna()&d.MYCN.isin(['Amplified','Not Amplified'])&d.ploidy.isin(['Hyperdiploid','Diploid'])&d['diagnoses.inss_stage'].isin(['Stage 1','Stage 2A','Stage 2B','Stage 3','Stage 4','Stage 4S'])&d.age.notna()&d.year.notna()
x=d[core].copy(); x['e']=(x.first_event!='Censored').astype(int); x['t']=x.t.clip(lower=1)
x['age18']=(x.age>=547).astype(int); x['st4']=(x['diagnoses.inss_stage']=='Stage 4').astype(int)
x['mycn']=(x.MYCN=='Amplified').astype(int); x['dip']=(x.ploidy=='Diploid').astype(int); x['late']=(x.year>2008).astype(int)
F=['age18','st4','mycn','dip']
rows=[]
for era,g in [('<=2008',x[x.late==0]),('2009-2012',x[x.late==1])]:
    r=PHReg(g.t,g[F],status=g.e).fit()
    for i,f in enumerate(F):
        rows.append((f,era,np.exp(r.params[i]),np.exp(r.params[i]-1.96*r.bse[i]),np.exp(r.params[i]+1.96*r.bse[i])))
res=pd.DataFrame(rows,columns=['factor','era','HR','lo','hi'])
# interaction model
X=x[F].copy()
for f in F: X[f+'_x_late']=x[f]*x.late
X['late']=x.late
r=PHReg(x.t,X,status=x.e).fit()
names=list(X.columns); pint={f:r.pvalues[names.index(f+'_x_late')] for f in F}
ratio={f:np.exp(r.params[names.index(f+'_x_late')]) for f in F}
print(res.round(2).to_string(index=False)); print('interaction p:',{k:round(v,3) for k,v in pint.items()}); print('ratio of HRs late/early:',{k:round(v,2) for k,v in ratio.items()})
# within-subgroup events
for f in F: print(f, 'events by era&level', x.groupby(['late',f]).e.sum().to_dict())
res.to_csv('nbl_era_hr.csv',index=False)
pd.Series(pint).to_csv('nbl_era_pint.csv')
