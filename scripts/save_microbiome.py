import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
rand = 10

diversity = pd.read_csv('../data/alpha_diversity.tsv', sep='\t')
microbiome_data = pd.read_csv('../data/genus.tsv', sep='\t')

def combine_df(df):
    # separate into two dataframes for timepoints 0 and 52
    df_0 = df[df['sampleID'].str.endswith('_0')].copy()
    df_52 = df[df['sampleID'].str.endswith('_52')].copy()
    # remove the timepoint from sampleID
    df_0['sampleID'] = df_0['sampleID'].str.replace('_0', '', regex=False)
    df_52['sampleID'] = df_52['sampleID'].str.replace('_52', '', regex=False)
    # rename feature columns to include timepoint
    df_0.columns = [col + '_0' if col != 'sampleID' else col for col in df_0.columns]
    df_52.columns = [col + '_52' if col != 'sampleID' else col for col in df_52.columns]
    # when merging, if there are any subjects not in both timepoints, have NaN for that timepoint
    merged_df = pd.merge(df_0, df_52, on='sampleID', how='outer')
    return merged_df

microbiome_data = combine_df(microbiome_data)
diversity = diversity.drop(columns=['observed'])
diversity = combine_df(diversity)


imp = SimpleImputer(strategy='median')
microbiome_data.iloc[:, 1:] = imp.fit_transform(microbiome_data.iloc[:, 1:])

microbiome_data = microbiome_data.set_index('sampleID')
diversity = diversity.set_index('sampleID')


ra_clr = microbiome_data.apply(
    lambda row: np.log1p(row) - np.log1p(row).mean(),
    axis=1,
    result_type='broadcast'
)

scaler = StandardScaler()
ra_scaled = scaler.fit_transform(ra_clr)

pca = PCA(n_components=10, random_state=rand)
pcs = pca.fit_transform(ra_scaled)
pcs_df = pd.DataFrame(pcs, columns=[f'pc{i+1}' for i in range(10)])

print("Explained var:", pca.explained_variance_ratio_.sum())

#add sampleID to pcs_df
pcs_df['subjectID'] = microbiome_data.index
pcs_df = pcs_df.set_index('subjectID')

# combine the columns of pcs_df with diversity based on sampleID
combined_df = pd.merge(pcs_df, diversity, left_index=True, right_index=True, how='outer')
combined_df.to_csv('../data/microbiome_pcs.tsv', sep='\t', index=True)


#get makeup of pcs
feature_names = microbiome_data.columns
loadings = pd.DataFrame(
    pca.components_,
    columns=feature_names,
    index=[f'pc{i+1}' for i in range(10)]
)
loadings.to_csv('../data/pc_contributions.tsv', sep='\t')
