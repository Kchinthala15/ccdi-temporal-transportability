import os; os.makedirs('work',exist_ok=True)
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from statsmodels.duration.hazard_regression import PHReg
rng=np.random.default_rng(7)
d=pd.read_csv('work/aml_cohort.csv')
d=d[d['treatments.protocol_identifier']!='AAML0631']  # APL trial excluded
d['favmut']=np.where(d[['npm1','cebpa']].isna().all(axis=1),np.nan,(d[['npm1','cebpa']]==1).any(axis=1).astype(float))
d['advcyto']=np.where(d[['mono7','del5q']].isna().all(axis=1),np.nan,(d[['mono7','del5q']]==1).any(axis=1).astype(float))
need=['flt3itd','favmut','cbf','advcyto','kmt2a','wbc','age_y','time','event','year']
x=d.dropna(subset=need).copy()
x['t']=x.time.clip(lower=1); x['e']=x.event.astype(bool)
x['age10']=(x.age_y>=10).astype(int); x['wbc100']=(x.wbc>=100).astype(int)
for c in ['flt3itd','favmut','cbf','advcyto','kmt2a']: x[c]=x[c].astype(int)
X=['age10','wbc100','cbf','favmut','flt3itd','advcyto','kmt2a']
x['late']=(x.year>=2011).astype(int)
dev=x[x.late==0]; val=x[x.late==1]
print('excluded APL; N with outcome', len(d),'complete',len(x),'events',x.e.sum())
print('dev',len(dev),dev.e.sum(),'val',len(val),val.e.sum())
print('median censored FU dev %d val %d'%(dev.t[~dev.e].median(),val.t[~val.e].median()))
print('prevalence dev/val:',{c:(round(dev[c].mean(),2),round(val[c].mean(),2)) for c in X})
H=3*365.25
S=lambda df: Surv.from_arrays(df.e.values,df.t.values)
m=CoxPHSurvivalAnalysis(alpha=1e-4).fit(dev[X],S(dev))
print('coefs',dict(zip(X,np.round(m.coef_,2))))
cidx=lambda df,lp: concordance_index_censored(df.e.values,df.t.values,lp)[0]
def boot_c(df,lp,B=300):
    v=[];idx=np.arange(len(df))
    for _ in range(B):
        s=rng.choice(idx,len(idx)); v.append(cidx(df.iloc[s],lp[s]))
    return np.percentile(v,[2.5,97.5])
lpd=m.predict(dev[X]); lpv=m.predict(val[X])
for nm,df,lp in [('dev (apparent)',dev,lpd),('val',val,lpv)]:
    lo,hi=boot_c(df,lp); print(f'C {nm}: {cidx(df,lp):.3f} ({lo:.3f}-{hi:.3f})')
cs=CoxPHSurvivalAnalysis(alpha=1e-6).fit(lpv.reshape(-1,1),S(val)); 
# bootstrap slope CI
sl=[]
for _ in range(300):
    s=rng.choice(np.arange(len(val)),len(val)); sl.append(CoxPHSurvivalAnalysis(alpha=1e-6).fit(lpv[s].reshape(-1,1),S(val.iloc[s])).coef_[0])
print('calibration slope val %.2f (%.2f-%.2f)'%(cs.coef_[0],*np.percentile(sl,[2.5,97.5])))
def pred3(df): return np.array([f(min(H,f.x[-1])) for f in m.predict_survival_function(df[X])])
def km3(df):
    tt,s=kaplan_meier_estimator(df.e.values,df.t.values); return s[tt<=H][-1] if (tt<=H).any() else 1.0
p=pred3(val); print('3-yr EFS val pred %.3f obs %.3f | dev pred %.3f obs %.3f'%(p.mean(),km3(val),pred3(dev).mean(),km3(dev)))
val=val.assign(p3=p)
def boot_oe(g,B=300):
    v=[];idx=np.arange(len(g))
    for _ in range(B):
        s=g.iloc[rng.choice(idx,len(idx))]; v.append(km3(s)-s.p3.mean())
    return np.percentile(v,[2.5,97.5])
rows=[]
for nm,mask in [('FLT3-ITD',val.flt3itd==1),('No FLT3-ITD',val.flt3itd==0),('CBF',val.cbf==1),('NPM1/CEBPA',val.favmut==1),('KMT2A-r',val.kmt2a==1),('Age >=10 y',val.age10==1),('Age <10 y',val.age10==0),('WBC >=100',val.wbc100==1)]:
    g=val[mask]; lo,hi=boot_oe(g); oe=km3(g)-g.p3.mean()
    rows.append((nm,len(g),int(g.e.sum()),oe,lo,hi)); print(f'  {nm}: n={len(g)} ev={g.e.sum()} pred3={g.p3.mean():.3f} obs3={km3(g):.3f} O-P={oe:+.3f} ({lo:+.3f},{hi:+.3f})')
pd.DataFrame(rows,columns=['group','n','events','oe','lo','hi']).to_csv('aml_subgroup_oe.csv',index=False)
# era-specific HRs and interactions (mutually adjusted)
F=X
res=[]
for era,g in [('<=2010',x[x.late==0]),('2011+',x[x.late==1])]:
    r=PHReg(g.t,g[F],status=g.e.astype(int)).fit()
    for i,f in enumerate(F): res.append((f,era,np.exp(r.params[i]),np.exp(r.params[i]-1.96*r.bse[i]),np.exp(r.params[i]+1.96*r.bse[i])))
Xi=x[F].copy()
for f in F: Xi[f+'_x']=x[f]*x.late
Xi['late']=x.late
r=PHReg(x.t,Xi,status=x.e.astype(int)).fit(); nmz=list(Xi.columns)
res=pd.DataFrame(res,columns=['factor','era','HR','lo','hi'])
print(res.round(2).to_string(index=False))
print('interaction p',{f:round(r.pvalues[nmz.index(f+'_x')],3) for f in F})
for f in ['flt3itd','cbf','favmut']: print(f,'events by era/level',x.groupby(['late',f]).e.sum().to_dict())
res.to_csv('aml_era_hr.csv',index=False)
# sensitivity: alternative era definition (AAML0531 vs AAML1031 only)
y=x[x['treatments.protocol_identifier'].isin(['AAML0531','AAML1031'])]
dv=y[y.late==0]; vl=y[y.late==1]
m2=CoxPHSurvivalAnalysis(alpha=1e-4).fit(dv[X],S(dv)); print('Sensitivity AAML0531->AAML1031: n %d/%d C val %.3f'%(len(dv),len(vl),cidx(vl,m2.predict(vl[X]))))
