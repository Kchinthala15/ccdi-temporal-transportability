import os; os.makedirs('work',exist_ok=True)
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv
try:
    for f in ['Regular','Bold']: fm.fontManager.addfont(f'/usr/share/fonts/truetype/crosextra/Carlito-{f}.ttf')
    plt.rcParams.update({'font.family':'Carlito','font.size':12.5})
except OSError:
    plt.rcParams.update({'font.size':12.5})  # Carlito font not installed; default font used
NAVY='#1B2A41'; TEAL='#138A8A'; GREY='#8A94A3'
rng=np.random.default_rng(3)
d=pd.read_csv('work/nbl_cohort.csv')
d['year']=pd.to_numeric(d['diagnoses.year_of_diagnosis'],errors='coerce'); d['age']=pd.to_numeric(d['diagnoses.age_at_diagnosis'],errors='coerce'); d['t']=pd.to_numeric(d.efs_days,errors='coerce')
core=d.t.notna()&d.first_event.notna()&d.MYCN.isin(['Amplified','Not Amplified'])&d.ploidy.isin(['Hyperdiploid','Diploid'])&d['diagnoses.inss_stage'].isin(['Stage 1','Stage 2A','Stage 2B','Stage 3','Stage 4','Stage 4S'])&d.age.notna()&d.year.notna()
x=d[core].copy(); x['e']=x.first_event!='Censored'; x['t']=x.t.clip(lower=1)
x['age18']=(x.age>=547).astype(int); x['st3']=(x['diagnoses.inss_stage']=='Stage 3').astype(int); x['st4']=(x['diagnoses.inss_stage']=='Stage 4').astype(int); x['st4s']=(x['diagnoses.inss_stage']=='Stage 4S').astype(int)
x['mycn']=(x.MYCN=='Amplified').astype(int); x['dip']=(x.ploidy=='Diploid').astype(int)
X=['age18','st3','st4','st4s','mycn','dip']; dev=x[x.year<=2008]; val=x[x.year>2008].copy(); H=3*365.25
m=CoxPHSurvivalAnalysis(alpha=1e-4).fit(dev[X],Surv.from_arrays(dev.e.values,dev.t.values))
val['p3']=[f(H) for f in m.predict_survival_function(val[X])]
def km3(g):
    tt,s=kaplan_meier_estimator(g.e.values,g.t.values); return s[tt<=H][-1]
subs=[('Age < 18 months',val.age18==0),('Age ≥ 18 months',val.age18==1),('MYCN not amplified',val.mycn==0),('MYCN amplified',val.mycn==1),('Non-Stage 4',val.st4==0),('Stage 4',val.st4==1),('Hyperdiploid',val.dip==0),('Diploid',val.dip==1)]
oe=[]
for nm,mk in subs:
    g=val[mk].reset_index(drop=True); est=km3(g)-g.p3.mean()
    bs=[]
    for _ in range(400):
        s=g.iloc[rng.choice(len(g),len(g))]; bs.append(km3(s)-s.p3.mean())
    lo,hi=np.percentile(bs,[2.5,97.5]); oe.append((nm,est*100,lo*100,hi*100,len(g)))
oe=pd.DataFrame(oe,columns=['sub','oe','lo','hi','n']); oe.to_csv('nbl_subgroup_oe.csv',index=False); print(oe.round(1))
hr=pd.read_csv('nbl_era_hr.csv'); pint=pd.read_csv('nbl_era_pint.csv',index_col=0).iloc[:,0]
lab={'age18':'Age ≥ 18 months','st4':'INSS Stage 4','mycn':'MYCN amplified','dip':'Diploid tumor'}
fig,ax=plt.subplots(1,2,figsize=(6.5,4.6),gridspec_kw={'width_ratios':[1,1.05]})
a=ax[0]; fs=['age18','mycn','st4','dip']
for i,f in enumerate(fs):
    for j,(era,c,off) in enumerate([('<=2008',GREY,0.15),('2009-2012',TEAL,-0.15)]):
        r=hr[(hr.factor==f)&(hr.era==era)].iloc[0]; y=len(fs)-1-i+off
        a.plot([r.lo,r.hi],[y,y],color=c,lw=1.6); a.plot(r.HR,y,'o',color=c,ms=5)
    a.text(5.6,len(fs)-1-i,f'{pint[f]:.3f}',va='center',fontsize=12,color=NAVY)
a.axvline(1,color='#BBBBBB',lw=0.8,ls='--'); a.set_xscale('log'); a.set_xticks([0.5,1,2,4]); a.set_xticklabels(['0.5','1','2','4']); a.set_xlim(0.45,9.0); a.set_ylim(-0.5,len(fs)-0.3); a.text(5.6,len(fs)-0.55,'p, era',fontsize=12,color=NAVY,va='bottom',fontweight='bold')
a.set_yticks(range(len(fs))); a.set_yticklabels([lab[f].replace(' ','\n',1) for f in fs][::-1]); a.set_xlabel('Hazard ratio\n(95% CI)')
a.set_title('A. Effect by era',loc='left',fontsize=13,fontweight='bold',color=NAVY)
a.plot([],[],'o-',color=GREY,label='Diagnosed ≤2008'); a.plot([],[],'o-',color=TEAL,label='Diagnosed 2009–2012'); fig.legend(fontsize=12.5,frameon=False,loc='lower left',ncol=2,bbox_to_anchor=(0.02,0.0))
b=ax[1]; yy=np.arange(len(oe))[::-1]
for y,r in zip(yy,oe.itertuples()):
    c=TEAL if r.oe>0 else NAVY; b.plot([r.lo,r.hi],[y,y],color=c,lw=1.6); b.plot(r.oe,y,'s',color=c,ms=5)
b.axvline(0,color='#BBBBBB',lw=0.8,ls='--'); b.set_yticks(yy); b.set_yticklabels([f'{s}\n(n={n})' for s,n in zip(oe['sub'],oe.n)],fontsize=12)
b.set_xlabel('Observed − predicted 3-yr\nrecorded EFS (points)',fontsize=12.5)
b.set_title('B. Calibration, 2009–2012',loc='left',fontsize=13,fontweight='bold',color=NAVY)
for s in ['top','right']: a.spines[s].set_visible(False); b.spines[s].set_visible(False)
plt.tight_layout(rect=(0,0.08,1,1)); plt.savefig('nbl_fig1.png',dpi=300)
