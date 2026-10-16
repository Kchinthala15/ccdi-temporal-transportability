import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
c=pd.read_csv('data/TARGET-AML/clinical.tsv',sep='\t',dtype=str,low_memory=False).replace("'--",pd.NA)
lf=c[['cases.submitter_id','diagnoses.days_to_last_follow_up']].dropna().drop_duplicates('cases.submitter_id')
d=pd.read_csv('work/aml_cohort.csv').merge(lf,on='cases.submitter_id',how='left')
d['lfu']=d['diagnoses.days_to_last_follow_up'].astype(float)
mis=(d.event==0)&d.lfu.notna()&(d.lfu!=d.time)
print('censored mismatches in cohort',mis.sum()); print(d.loc[mis,['cases.submitter_id','time','lfu','year']])
d=d[d['treatments.protocol_identifier']!='AAML0631']
d['favmut']=np.where(d[['npm1','cebpa']].isna().all(axis=1),np.nan,(d[['npm1','cebpa']]==1).any(axis=1).astype(float))
d['advcyto']=np.where(d[['mono7','del5q']].isna().all(axis=1),np.nan,(d[['mono7','del5q']]==1).any(axis=1).astype(float))
x=d.dropna(subset=['flt3itd','favmut','cbf','advcyto','kmt2a','wbc','age_y','time','event','year']).copy()
x=x[~(((x.event==0)&x.lfu.notna()&(x.lfu!=x.time))|(x.time<=0))]
x['t']=x.time.clip(lower=1); x['e']=x.event.astype(bool); x['age10']=(x.age_y>=10).astype(int); x['wbc100']=(x.wbc>=100).astype(int)
X=['age10','wbc100','cbf','favmut','flt3itd','advcyto','kmt2a']
dev=x[x.year<=2010]; val=x[x.year>=2011]
S=lambda df: Surv.from_arrays(df.e.values,df.t.values)
m=CoxPHSurvivalAnalysis(alpha=1e-4).fit(dev[X],S(dev))
C=lambda df: concordance_index_censored(df.e.values,df.t.values,m.predict(df[X]))[0]
lp=m.predict(val[X]); sl=CoxPHSurvivalAnalysis(alpha=1e-6).fit(lp.reshape(-1,1),S(val)).coef_[0]
H=3*365.25
p3=np.array([f(min(H,f.x[-1])) for f in m.predict_survival_function(val[X])]); val=val.assign(p3=p3)
def km3(g):
    tt,s=kaplan_meier_estimator(g.e.values,g.t.values); return s[tt<=H][-1]
print('n',len(x),'dev',len(dev),'val',len(val),'C dev %.3f val %.3f slope %.2f'%(C(dev),C(val),sl))
for nm,g in [('FLT3-ITD',val[val.flt3itd==1]),('KMT2A',val[val.kmt2a==1])]: print(nm,'O-P %+.3f'%(km3(g)-g.p3.mean()))
