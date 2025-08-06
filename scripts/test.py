import pandas as pd
from format_matrix import combine_df

wolkes_data = pd.read_csv('../data/wolkes.tsv', sep='\t')
bayley_data = pd.read_csv('../data/bayley.tsv', sep='\t')

wolkes_data = combine_df(wolkes_data)
bayley_data = combine_df(bayley_data)
comb = wolkes_data.merge(bayley_data, on='subjectID', how='outer')
comb = comb.set_index('subjectID')
targets = comb[['cognitive_score_52', 'cognitive_score_0', 'vocalisation_0', 'vocalisation_52']].copy()