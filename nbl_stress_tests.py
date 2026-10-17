# Stress tests for NCI AYA Milestone 1: optimism-corrected development C-index,
# and robustness of the age x era interaction to era cut-point and continuous calendar time.
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from statsmodels.duration.hazard_regression import PHReg
rng=np.random.default_rng(2026)
d=pd.read_csv('work/nbl_cohort.csv')
d['year']=pd.to_numeric(d['diagnoses.year_of_diagnosis'],errors='coerce')
d['age']=pd.to_numeric(d['diagnoses.age_at_diagnosis'],errors='coerce')
d['t']=pd.to_numeric(d.efs_days,errors='coerce')
core=d.t.notna()&d.first_event.notna()&d.MYCN.isin(['Amplified','Not Amplified'])&d.ploidy.isin(['Hyperdiploid','Diploid'])&d['diagnoses.inss_stage'].isin(['Stage 1','Stage 2A','Stage 2B','Stage 3','Stage 4','Stage 4S'])&d.age.notna()&d.year.notna()
x=d[core].copy(); x['e']=(x.first_event!='Censored'); x['t']=x.t.clip(lower=1)
x['age18']=(x.age>=547).astype(int)
for s,c in [('Stage 4','st4'),('Stage 4S','st4s'),('Stage 3','st3')]: x[c]=(x['diagnoses.inss_stage']==s).astype(int)
x['mycn']=(x.MYCN=='Amplified').astype(int); x['dip']=(x.ploidy=='Diploid').astype(int)
X=['age18','st3','st4','st4s','mycn','dip']
print('N',len(x),'events',int(x.e.sum()))
dev=x[x.year<=2008].reset_index(drop=True); val=x[x.year>2008].reset_index(drop=True)
S=lambda g: Surv.from_arrays(g.e.values,g.t.values)
C=lambda g,lp: concordance_index_censored(g.e.values,g.t.values,lp)[0]
m=CoxPHSurvivalAnalysis(alpha=1e-4).fit(dev[X],S(dev))
app=C(dev,m.predict(dev[X])); ext=C(val,m.predict(val[X]))
opt=[]
for b in range(300):
    i=rng.integers(0,len(dev),len(dev)); bs=dev.iloc[i]
    try:
        mb=CoxPHSurvivalAnalysis(alpha=1e-4).fit(bs[X],S(bs))
        opt.append(C(bs,mb.predict(bs[X]))-C(dev,mb.predict(dev[X])))
    except Exception: pass
o=np.mean(opt)
print(f'Apparent dev C {app:.3f}; bootstrap optimism {o:.3f} (B={len(opt)}); optimism-corrected dev C {app-o:.3f}; temporal validation C {ext:.3f}')
F=['age18','st4','mycn','dip']
def inter(late):
    Z=x[F].copy()
    for f in F: Z[f+'_x']=x[f]*late
    Z['late']=late
    r=PHReg(x.t,Z,status=x.e.astype(int)).fit(); n=list(Z.columns)
    return {f:(np.exp(r.params[n.index(f+'_x')]),r.pvalues[n.index(f+'_x')]) for f in F}
rows=[]
for cut in [2006,2007,2008,2009,2010]:
    late=(x.year>cut).astype(int); res=inter(late)
    rows.append(dict(spec=f'era split after {cut}',n_early=int((late==0).sum()),n_late=int(late.sum()),
        age_ratio=round(res['age18'][0],2),age_p=round(res['age18'][1],4),mycn_ratio=round(res['mycn'][0],2),mycn_p=round(res['mycn'][1],3)))
cy=(x.year-2008).astype(float); res=inter(cy)
rows.append(dict(spec='continuous calendar year (per year)',n_early=None,n_late=None,age_ratio=round(res['age18'][0],3),age_p=round(res['age18'][1],4),mycn_ratio=round(res['mycn'][0],3),mycn_p=round(res['mycn'][1],3)))
out=pd.DataFrame(rows); print(out.to_string(index=False))
out.to_csv('nbl_stress_interaction.csv',index=False)
pd.DataFrame([dict(apparent_dev_C=app,bootstrap_optimism=o,optimism_corrected_dev_C=app-o,temporal_validation_C=ext,B=len(opt))]).to_csv('nbl_stress_optimism.csv',index=False)
