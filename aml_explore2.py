import pandas as pd,sys
f=pd.read_csv(sys.argv[1],sep='\t',dtype=str,low_memory=False).replace("'--",pd.NA)
m=f[f['molecular_tests.molecular_test_id'].notna()].drop_duplicates('molecular_tests.molecular_test_id')
m['key']=(m['molecular_tests.gene_symbol'].fillna('')+'|'+m['molecular_tests.second_gene_symbol'].fillna('')+'|'+m['molecular_tests.variant_type'].fillna('')+'|'+m['molecular_tests.chromosomal_translocation'].fillna('')+'|'+m['molecular_tests.aneuploidy'].fillna('')+'|'+m['molecular_tests.chromosome'].fillna('')+m['molecular_tests.chromosome_arm'].fillna('')+'|'+m['molecular_tests.laboratory_test'].fillna('')+'|'+m['molecular_tests.timepoint_category'].fillna(''))
t=m.groupby(['key','molecular_tests.test_result'])['cases.submitter_id'].nunique().unstack(fill_value=0)
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
print(t.to_string())
