import os; os.makedirs('work',exist_ok=True)
import pandas as pd, numpy as np, sys
raw=sys.argv[1] if len(sys.argv)>1 else 'data/TARGET-AML'
c=pd.read_csv(raw+'/clinical.tsv',sep='\t',dtype=str,low_memory=False).replace("'--",pd.NA)
f=pd.read_csv(raw+'/follow_up.tsv',sep='\t',dtype=str,low_memory=False).replace("'--",pd.NA)
id='cases.submitter_id'
# outcome
ev=f[f['follow_ups.first_event'].notna()][[id,'follow_ups.first_event','follow_ups.days_to_first_event']].drop_duplicates()
print('outcome rows',len(ev),'unique',ev[id].nunique())
ev=ev.drop_duplicates(id)
# baseline
b=c[[id,'diagnoses.age_at_diagnosis','diagnoses.year_of_diagnosis']].dropna().drop_duplicates()
print('baseline dup check',b[id].duplicated().sum())
b=b.drop_duplicates(id)
# protocol
p=c[[id,'treatments.protocol_identifier']].dropna().drop_duplicates()
print('protocol per case >1:',(p.groupby(id).size()>1).sum())
p=p.drop_duplicates(id)
# molecular
m=f[f['molecular_tests.molecular_test_id'].notna()].drop_duplicates('molecular_tests.molecular_test_id')
def flag(mask):
    s=m[mask].groupby(id)['molecular_tests.test_result'].agg(lambda r: 1 if (r=='Positive').any() else (0 if (r=='Negative').any() else np.nan))
    return s
g=m['molecular_tests.gene_symbol']; v=m['molecular_tests.variant_type']; t=m['molecular_tests.chromosomal_translocation']; ch=m['molecular_tests.chromosome']; an=m['molecular_tests.aneuploidy']
F=pd.DataFrame({
 'flt3itd':flag((g=='FLT3')&(v=='Internal Tandem Duplication')),
 'npm1':flag((g=='NPM1')&(v=='Mutation, NOS')),
 'cebpa':flag((g=='CEBPA')&(v=='Mutation, NOS')),
 't821':flag(t=='t(8;21)'),
 'inv16':flag((v=='Inversion')&(ch=='chr16')),
 'mono7':flag((an=='Monosomy')&(ch=='chr7')),
 'del5q':flag((v=='Deletion')&(ch=='chr5')),
 'kmt2a':flag((g=='KMT2A')&(v=='Translocation')),
})
wbc=m[m['molecular_tests.laboratory_test']=='Leukocytes'].groupby(id)['molecular_tests.test_value'].first().astype(float)
F['wbc']=wbc
d=ev.merge(b,on=id).merge(p,on=id,how='left').merge(F,left_on=id,right_index=True,how='left')
d['time']=d['follow_ups.days_to_first_event'].astype(float)
d['event']=(d['follow_ups.first_event']!='Censored').astype(int)
d['age_y']=d['diagnoses.age_at_diagnosis'].astype(float)/365.25
d['year']=d['diagnoses.year_of_diagnosis'].astype(int)
d['cbf']=np.where(d[['t821','inv16']].isna().all(1),np.nan,(d[['t821','inv16']]==1).any(axis=1).astype(float))
d.to_csv('work/aml_cohort.csv',index=False)
print(len(d),'cases with outcome+baseline'); print(d['follow_ups.first_event'].value_counts())
print(pd.crosstab(d['year'],d['treatments.protocol_identifier'].fillna('NA')))
print(d[['flt3itd','npm1','cebpa','cbf','mono7','del5q','kmt2a','wbc']].notna().sum())
print('time<=0',(d.time<=0).sum(), 'time na',d.time.isna().sum())
