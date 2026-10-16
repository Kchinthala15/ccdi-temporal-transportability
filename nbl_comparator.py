import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
rng=np.random.default_rng(11)
d=pd.read_csv('work/nbl_cohort.csv')
d['year']=pd.to_numeric(d['diagnoses.year_of_diagnosis'],errors='coerce'); d['age']=pd.to_numeric(d['diagnoses.age_at_diagnosis'],errors='coerce'); d['t']=pd.to_numeric(d.efs_days,errors='coerce')
core=d.t.notna()&d.first_event.notna()&d.MYCN.isin(['Amplified','Not Amplified'])&d.ploidy.isin(['Hyperdiploid','Diploid'])&d['diagnoses.inss_stage'].isin(['Stage 1','Stage 2A','Stage 2B','Stage 3','Stage 4','Stage 4S'])&d.age.notna()&d.year.notna()
x=d[core].copy(); x['e']=x.first_event!='Censored'; x['t']=x.t.clip(lower=1)
x['a318']=(x.age>318).astype(int); x['st4']=(x['diagnoses.inss_stage']=='Stage 4').astype(int); x['dip']=(x.ploidy=='Diploid').astype(int)
L=['a318','st4','dip']; dev=x[x.year<=2008]; val=x[x.year>2008]; H=3*365.25
S=lambda g: Surv.from_arrays(g.e.values,g.t.values)
m=CoxPHSurvivalAnalysis(alpha=1e-4).fit(dev[L],S(dev))
C=lambda g,lp: concordance_index_censored(g.e.values,g.t.values,lp)[0]
def boot(g,lp):
    v=[C(g.iloc[s],lp[s]) for s in (rng.choice(len(g),len(g)) for _ in range(500))]; return np.percentile(v,[2.5,97.5])
for nm,g in [('dev',dev),('val',val)]:
    lp=m.predict(g[L]); lo,hi=boot(g,lp); print(f'Li-spec C {nm} {C(g,lp):.3f} ({lo:.3f}-{hi:.3f})')
cs=CoxPHSurvivalAnalysis(alpha=1e-6).fit(m.predict(val[L]).reshape(-1,1),S(val)); print('cal slope %.2f'%cs.coef_[0])
p=np.array([f(H) for f in m.predict_survival_function(val[L])]); tt,s=kaplan_meier_estimator(val.e.values,val.t.values); print('3y pred %.3f obs %.3f'%(p.mean(),s[tt<=H][-1]))
