import os; os.makedirs('work',exist_ok=True)
import pandas as pd, numpy as np, sys
R='data/TARGET-AML/'
na=["'--"]
cl=pd.read_csv(R+'clinical.tsv',sep='\t',dtype=str,na_values=na,keep_default_na=False)
fu=pd.read_csv(R+'follow_up.tsv',sep='\t',dtype=str,na_values=na,keep_default_na=False)
ID='cases.submitter_id'
dq=[]
# outcome
o=fu[fu['follow_ups.first_event'].notna()][[ID,'follow_ups.first_event','follow_ups.days_to_first_event','follow_ups.follow_up_id']].drop_duplicates()
g=o.groupby(ID).agg(nfe=('follow_ups.first_event','nunique'),nd=('follow_ups.days_to_first_event','nunique'),nrow=('follow_ups.follow_up_id','nunique'))
dq.append(f"outcome: cases={len(g)}, cases w/ >1 follow_up row={int((g.nrow>1).sum())}, conflicting first_event={int((g.nfe>1).sum())}, conflicting days={int((g.nd>1).sum())}")
dq.append("first_event values: "+str(o.drop_duplicates(ID)['follow_ups.first_event'].value_counts().to_dict()))
o=o.drop_duplicates(ID)
o['event']=(o['follow_ups.first_event']!='Censored').astype(int)
t=pd.to_numeric(o['follow_ups.days_to_first_event'])
dq.append(f"time: missing={int(t.isna().sum())}, <=0: {int((t<=0).sum())}, <1: {int((t<1).sum())}, max={t.max()}")
o['time']=t.clip(lower=1)
o=o[[ID,'event','time']]
# baseline
b=cl[[ID,'diagnoses.age_at_diagnosis','diagnoses.year_of_diagnosis','treatments.protocol_identifier']].drop_duplicates()
def agg(s):
    v=s.dropna().unique(); return v[0] if len(v)==1 else (np.nan if len(v)==0 else '|'.join(sorted(v)))
bb=b.groupby(ID).agg(agg).reset_index()
for c in bb.columns[1:]:
    dq.append(f"baseline {c}: conflicting cases={int(bb[c].astype(str).str.contains('|',regex=False).sum())}")
dq.append("protocols: "+str(bb['treatments.protocol_identifier'].value_counts(dropna=False).to_dict()))
bb=bb[bb['treatments.protocol_identifier'].fillna('')!='AAML0631']  # conflicts containing AAML0631 handled below
bb=bb[~bb['treatments.protocol_identifier'].fillna('').str.contains('AAML0631')]
bb['age']=pd.to_numeric(bb['diagnoses.age_at_diagnosis'],errors='coerce')
bb['year']=pd.to_numeric(bb['diagnoses.year_of_diagnosis'],errors='coerce')
# molecular
m=fu[fu['molecular_tests.molecular_test_id'].notna()].drop_duplicates('molecular_tests.molecular_test_id')
dq.append(f"molecular rows raw={fu['molecular_tests.molecular_test_id'].notna().sum()} unique test ids={len(m)}")
gs=m['molecular_tests.gene_symbol'];vt=m['molecular_tests.variant_type'];ch=m['molecular_tests.chromosome']
def factor(mask,name):
    s=m[mask]; r=s.groupby(ID)['molecular_tests.test_result'].agg(lambda x:1.0 if (x=='Positive').any() else (0.0 if (x=='Negative').any() else np.nan))
    # conflicts
    c=s.groupby(ID)['molecular_tests.test_result'].agg(lambda x:(x=='Positive').any() and (x=='Negative').any())
    dq.append(f"{name}: tests={len(s)}, cases={r.notna().sum()}, pos={int((r==1).sum())}, pos/neg conflict cases={int(c.sum())}, results={s['molecular_tests.test_result'].value_counts().to_dict()}")
    return r.rename(name)
F=[factor((gs=='FLT3')&(vt=='Internal Tandem Duplication'),'flt3itd'),
   factor((gs=='NPM1')&(vt=='Mutation, NOS'),'npm1'),
   factor((gs=='CEBPA')&(vt=='Mutation, NOS'),'cebpa'),
   factor(m['molecular_tests.chromosomal_translocation']=='t(8;21)','t821'),
   factor((vt=='Inversion')&(ch=='chr16'),'inv16'),
   factor((m['molecular_tests.aneuploidy']=='Monosomy')&(ch=='chr7'),'mono7'),
   factor((vt=='Deletion')&(ch=='chr5'),'del5q'),
   factor((gs=='KMT2A')&(vt=='Translocation'),'kmt2a')]
M=pd.concat(F,axis=1)
def orc(a,b):
    r=pd.Series(np.nan,index=M.index); both=M[[a,b]]
    r[both.notna().any(axis=1)]=0.0; r[(both==1).any(axis=1)]=1.0; return r
M['favmut']=orc('npm1','cebpa');M['cbf']=orc('t821','inv16');M['advcyto']=orc('mono7','del5q')
w=m[m['molecular_tests.laboratory_test']=='Leukocytes']
dq.append(f"WBC tests={len(w)}, units={w['molecular_tests.test_units'].value_counts(dropna=False).to_dict()}")
wv=pd.to_numeric(w['molecular_tests.test_value'],errors='coerce')
w=w.assign(v=wv); wg=w.groupby(ID)['v']
dq.append(f"WBC cases={wg.size().shape[0]}, cases with >1 distinct value={int((wg.nunique()>1).sum())}, nonnumeric={int(wv.isna().sum())}, min={wv.min()}, max={wv.max()}")
M['wbc']=wg.first()  # see dq note on multiples
M['wbcmax']=wg.max()
d=o.merge(bb[[ID,'age','year']],on=ID).merge(M.reset_index().rename(columns={'index':ID}),on=ID,how='left')
d['age10']=np.where(d.age.notna(),(d.age/365.25>=10).astype(float),np.nan)
d['wbc100']=np.where(d.wbc.notna(),(d.wbc>=100).astype(float),np.nan)
X=['age10','wbc100','cbf','favmut','flt3itd','advcyto','kmt2a']
dq.append(f"after outcome+baseline merge (excl AAML0631): {len(d)}; missing per var: "+str(d[X+['year']].isna().sum().to_dict()))
cc=d.dropna(subset=X+['year','time','event']).copy()
cc['late']=(cc.year>=2011).astype(int)
cc.to_csv('work/aml_cohort_independent.csv',index=False)
print('\n'.join(dq))
