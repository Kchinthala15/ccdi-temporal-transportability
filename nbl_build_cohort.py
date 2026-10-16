import os; os.makedirs('work',exist_ok=True)
"""Rebuild the TARGET-NBL patient-level cohort (cohort.csv) from the GDC clinical export.
Usage: python3 build_cohort.py <folder with clinical.tsv and follow_up.tsv> <output cohort.csv>"""
import pandas as pd, sys
raw, out = sys.argv[1], sys.argv[2]
ID='cases.submitter_id'
c=pd.read_csv(f'{raw}/clinical.tsv',sep='\t',dtype=str,low_memory=False).replace("'--",pd.NA)
f=pd.read_csv(f'{raw}/follow_up.tsv',sep='\t',dtype=str,low_memory=False).replace("'--",pd.NA)
cols=['diagnoses.age_at_diagnosis','diagnoses.inss_stage','diagnoses.mitosis_karyorrhexis_index','diagnoses.inpc_grade',
      'diagnoses.cog_neuroblastoma_risk_group','diagnoses.year_of_diagnosis','demographic.sex_at_birth','demographic.vital_status',
      'demographic.days_to_death','diagnoses.days_to_last_follow_up','diagnoses.primary_diagnosis','treatments.protocol_identifier']
base=pd.DataFrame({ID:sorted(c[ID].unique())})
for col in cols:   # first non-missing value per case
    base=base.merge(c[[ID,col]].dropna().drop_duplicates(ID),on=ID,how='left')
m=f[f['molecular_tests.molecular_test_id'].notna()].drop_duplicates('molecular_tests.molecular_test_id')
mycn=m[(m['molecular_tests.gene_symbol']=='MYCN')&(m['molecular_tests.variant_type']=='Amplification')][[ID,'molecular_tests.test_result']].drop_duplicates(ID).rename(columns={'molecular_tests.test_result':'MYCN'})
ploidy=m[m['molecular_tests.ploidy'].notna()][[ID,'molecular_tests.ploidy']].drop_duplicates(ID).rename(columns={'molecular_tests.ploidy':'ploidy'})
ev=f[f['follow_ups.first_event'].notna()][[ID,'follow_ups.first_event','follow_ups.days_to_first_event']].drop_duplicates(ID).rename(columns={'follow_ups.first_event':'first_event','follow_ups.days_to_first_event':'efs_days'})
d=base.merge(mycn,on=ID,how='left').merge(ploidy,on=ID,how='left').merge(ev,on=ID,how='left')
d.to_csv(out,index=False); print(d.shape)
