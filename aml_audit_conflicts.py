"""Audit TARGET-AML export for (1) conflicting outcome records per patient and (2) conflicting molecular results.
Usage: python3 audit_conflicts.py <folder with follow_up.tsv>"""
import pandas as pd, sys
f=pd.read_csv(sys.argv[1]+'/follow_up.tsv',sep='\t',dtype=str,low_memory=False).replace("'--",pd.NA)
ID='cases.submitter_id'
o=f[f['follow_ups.first_event'].notna()][[ID,'follow_ups.first_event','follow_ups.days_to_first_event']]
print('outcome rows:',len(o),'| patients:',o[ID].nunique(),'| exact duplicate rows:',o.duplicated().sum())
k=o.drop_duplicates().groupby(ID).size(); print('patients with >1 distinct outcome record (label or time):',(k>1).sum())
m=f[f['molecular_tests.molecular_test_id'].notna()].drop_duplicates('molecular_tests.molecular_test_id')
g=m['molecular_tests.gene_symbol'];v=m['molecular_tests.variant_type'];t=m['molecular_tests.chromosomal_translocation'];ch=m['molecular_tests.chromosome'];an=m['molecular_tests.aneuploidy']
defs={'FLT3-ITD':(g=='FLT3')&(v=='Internal Tandem Duplication'),'NPM1':(g=='NPM1')&(v=='Mutation, NOS'),'CEBPA':(g=='CEBPA')&(v=='Mutation, NOS'),
 't(8;21)':t=='t(8;21)','inv(16)':(v=='Inversion')&(ch=='chr16'),'monosomy 7':(an=='Monosomy')&(ch=='chr7'),'del(5q)':(v=='Deletion')&(ch=='chr5'),'KMT2A-r':(g=='KMT2A')&(v=='Translocation')}
for name,mask in defs.items():
    r=m[mask].groupby(ID)['molecular_tests.test_result'].agg(set)
    both=r.apply(lambda s:{'Positive','Negative'}<=s).sum(); multi=(m[mask].groupby(ID).size()>1).sum()
    print(f'{name}: patients tested {len(r)}, with >1 record {multi}, with both Positive and Negative {both}')
w=m[m['molecular_tests.laboratory_test']=='Leukocytes'].groupby(ID)['molecular_tests.test_value'].nunique()
print('WBC: patients',len(w),'| with >1 distinct value',(w>1).sum())
d=m[m['molecular_tests.gene_symbol'].isin(['FLT3','NPM1','CEBPA'])]
print('days_to_test populated for gene-mutation records:',d['molecular_tests.days_to_test'].notna().sum(),'of',len(d))
print('days_to_test values for WBC/blast records:',m.loc[m['molecular_tests.laboratory_test'].isin(['Leukocytes','Blast Count']),'molecular_tests.days_to_test'].dropna().unique())
