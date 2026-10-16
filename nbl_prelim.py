import pandas as pd, numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
rng=np.random.default_rng(7)
d=pd.read_csv('work/nbl_cohort.csv')
d['year']=pd.to_numeric(d['diagnoses.year_of_diagnosis'],errors='coerce')
d['age']=pd.to_numeric(d['diagnoses.age_at_diagnosis'],errors='coerce')
d['t']=pd.to_numeric(d.efs_days,errors='coerce')
core=d.t.notna()&d.first_event.notna()&d.MYCN.isin(['Amplified','Not Amplified'])&d.ploidy.isin(['Hyperdiploid','Diploid'])&d['diagnoses.inss_stage'].isin(['Stage 1','Stage 2A','Stage 2B','Stage 3','Stage 4','Stage 4S'])&d.age.notna()&d.year.notna()
x=d[core].copy()
x['e']=x.first_event!='Censored'
x['t']=x.t.clip(lower=1)
x['age18']=(x.age>=547).astype(int)
x['st4']=(x['diagnoses.inss_stage']=='Stage 4').astype(int)
x['st4s']=(x['diagnoses.inss_stage']=='Stage 4S').astype(int)
x['st3']=(x['diagnoses.inss_stage']=='Stage 3').astype(int)
x['mycn']=(x.MYCN=='Amplified').astype(int)
x['dip']=(x.ploidy=='Diploid').astype(int)
x['risk']=x['diagnoses.cog_neuroblastoma_risk_group']
X=['age18','st3','st4','st4s','mycn','dip']
dev=x[x.year<=2008]; val=x[x.year>2008]
print('N core',len(x),'events',x.e.sum()); print('dev',len(dev),dev.e.sum(),'val',len(val),val.e.sum())
print('Stage4 share dev %.2f val %.2f; MYCN amp dev %.2f val %.2f; age>=18mo dev %.2f val %.2f'%(dev.st4.mean(),val.st4.mean(),dev.mycn.mean(),val.mycn.mean(),dev.age18.mean(),val.age18.mean()))
print('median FU (reverse KM approx: censored times) dev %d val %d'%(dev.t[~dev.e].median(),val.t[~val.e].median()))
H=3*365.25
def surv(df): return Surv.from_arrays(df.e.values,df.t.values)
m=CoxPHSurvivalAnalysis(alpha=1e-4).fit(dev[X],surv(dev))
print('coefs',dict(zip(X,np.round(m.coef_,2))))
def cidx(df,lp): return concordance_index_censored(df.e.values,df.t.values,lp)[0]
def boot_c(df,lp,B=300):
    v=[]; idx=np.arange(len(df))
    for _ in range(B):
        s=rng.choice(idx,len(idx)); 
        try: v.append(cidx(df.iloc[s],lp[s]))
        except: pass
    return np.percentile(v,[2.5,97.5])
lp_dev=m.predict(dev[X]); lp_val=m.predict(val[X])
for nm,df,lp in [('dev',dev,lp_dev),('val',val,lp_val)]:
    lo,hi=boot_c(df,lp); print(f'Cox model C {nm}: {cidx(df,lp):.3f} ({lo:.3f}-{hi:.3f})')
# calibration slope: refit cox on lp in val
cs=CoxPHSurvivalAnalysis(alpha=1e-6).fit(lp_val.reshape(-1,1),surv(val)); print('calibration slope val %.2f'%cs.coef_[0])
# 3-yr predicted vs observed
def pred3(model,df):
    fns=model.predict_survival_function(df[X]); return np.array([f(min(H,f.x[-1])) for f in fns])
def km3(df):
    tt,s=kaplan_meier_estimator(df.e.values,df.t.values); return s[tt<=H][-1] if (tt<=H).any() else 1.0
p=pred3(m,val); print('3-yr EFS val: predicted %.3f observed KM %.3f'%(p.mean(),km3(val)))
pdv=pred3(m,dev); print('3-yr EFS dev: predicted %.3f observed KM %.3f'%(pdv.mean(),km3(dev)))
# COG risk group
rg=val.dropna(subset=['risk']); rgd=dev.dropna(subset=['risk'])
code={'Low Risk':0,'Intermediate Risk':1,'High Risk':2}
print('COG risk C val: %.3f (n=%d)'%(cidx(rg,rg.risk.map(code).values.astype(float)),len(rg)),' dev: %.3f (n=%d)'%(cidx(rgd,rgd.risk.map(code).values.astype(float)),len(rgd)))
for r in ['Low Risk','Intermediate Risk','High Risk']:
    print(' ',r,'3yr EFS dev %.3f (n=%d) val %.3f (n=%d)'%(km3(rgd[rgd.risk==r]),(rgd.risk==r).sum(),km3(rg[rg.risk==r]),(rg.risk==r).sum()))
# subgroup performance in val
val=val.assign(p3=p,lp=lp_val)
for nm,mask in [('MYCN amplified',val.mycn==1),('MYCN not amplified',val.mycn==0),('age<18mo',val.age18==0),('age>=18mo',val.age18==1),('Stage 4',val.st4==1),('non-Stage 4',val.st4==0),('Diploid',val.dip==1),('Hyperdiploid',val.dip==0)]:
    g=val[mask]
    try: c=cidx(g,g.lp.values)
    except: c=np.nan
    print(f'  {nm}: n={len(g)} ev={g.e.sum()} C={c:.3f} pred3={g.p3.mean():.3f} obs3={km3(g):.3f} O-E={km3(g)-g.p3.mean():+.3f}')
# sensitivity: exclude generic Event
y=x[x.first_event!='Event']; dv=y[y.year<=2008]; vl=y[y.year>2008]
m2=CoxPHSurvivalAnalysis(alpha=1e-4).fit(dv[X],surv(dv)); print('Sensitivity (no generic Event): val C %.3f, n=%d ev=%d'%(cidx(vl,m2.predict(vl[X])),len(vl),vl.e.sum()))
