import pandas as pd, numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from statsmodels.duration.hazard_regression import PHReg
d=pd.read_csv('work/aml_cohort_independent.csv')
X=['age10','wbc100','cbf','favmut','flt3itd','advcyto','kmt2a']
dev=d[d.late==0]; val=d[d.late==1]
print('N',len(d),'events',d.event.sum(),'| dev',len(dev),dev.event.sum(),'| val',len(val),val.event.sum())
print('year range dev',dev.year.min(),dev.year.max(),'val',val.year.min(),val.year.max())
print('time<=1 rows', (d.time<=1).sum(), d[d.time<=1][['time','event']].values.tolist())
m=CoxPHSurvivalAnalysis(alpha=1e-4).fit(dev[X].astype(float).values,Surv.from_arrays(dev.event.astype(bool),dev.time))
print('coef/HR',dict(zip(X,np.round(np.exp(m.coef_),3))))
def C(df):
    lp=m.predict(df[X].astype(float).values)
    return concordance_index_censored(df.event.astype(bool),df.time,lp)[0],lp
cd,_=C(dev); cv,lpv=C(val)
print('C dev %.4f val %.4f'%(cd,cv))
r=PHReg(val.time.values,lpv.reshape(-1,1),status=val.event.values,ties='breslow').fit()
print('cal slope %.3f (SE %.3f, 95%% CI %.3f-%.3f)'%(r.params[0],r.bse[0],*r.conf_int()[0]))
def S3(df):
    sf=m.predict_survival_function(df[X].astype(float).values)
    return np.array([f(1095.75) if 1095.75<=f.x[-1] else f(f.x[-1]) for f in sf])
def km3(df):
    t,s=kaplan_meier_estimator(df.event.astype(bool),df.time)
    return s[t<=1095.75][-1]
# check max dev time
print('dev max time',dev.time.max())
p=S3(val);print('val 3y EFS: pred %.4f KM %.4f O-P %.4f'%(p.mean(),km3(val),km3(val)-p.mean()))
pdv=S3(dev);print('dev 3y EFS: pred %.4f KM %.4f'%(pdv.mean(),km3(dev)))
for name,mask in [('FLT3-ITD+',val.flt3itd==1),('KMT2A-r',val.kmt2a==1),('age<10',val.age10==0),('CBF',val.cbf==1)]:
    s=val[mask.values];ps=p[mask.values]
    print('%-10s n=%d ev=%d KM=%.4f pred=%.4f O-P=%+.4f'%(name,len(s),s.event.sum(),km3(s),ps.mean(),km3(s)-ps.mean()))
for era,df in [('dev',dev),('val',val)]:
    r=PHReg(df.time.values,df[X].astype(float).values,status=df.event.values,ties='breslow').fit()
    for v in ['flt3itd','age10']:
        i=X.index(v);ci=np.exp(r.conf_int()[i])
        print(era,v,'HR %.3f (%.3f-%.3f) p=%.4g'%(np.exp(r.params[i]),ci[0],ci[1],r.pvalues[i]))
Z=d[X].astype(float).copy()
for v in X: Z[v+'_x_late']=Z[v]*d.late
Z['late']=d.late
r=PHReg(d.time.values,Z.values,status=d.event.values,ties='breslow').fit()
for i,c in enumerate(Z.columns):
    print('pooled %-16s HR %.3f p=%.4g'%(c,np.exp(r.params[i]),r.pvalues[i]))
# efron sensitivity for interactions
r2=PHReg(d.time.values,Z.values,status=d.event.values,ties='efron').fit()
print('efron interaction p:',{c:round(p,4) for c,p in zip(Z.columns,r2.pvalues) if 'x_late' in c})
