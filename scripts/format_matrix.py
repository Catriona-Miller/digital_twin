import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Needed to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import joblib

PRS_data = pd.read_csv('../data/genetics.tsv', sep='\t')
aa_data = pd.read_csv('../data/aa.tsv', sep='\t')
microbiome_data = pd.read_csv('../data/microbiome_pcs.tsv', sep='\t')
wolkes_data = pd.read_csv('../data/wolkes.tsv', sep='\t')
bayley_data = pd.read_csv('../data/bayley.tsv', sep='\t')
economic_data = pd.read_csv('../data/economics.tsv', sep='\t')
head_data = pd.read_csv('../data/head.tsv', sep='\t')
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
education_data = pd.read_csv('../data/education.tsv', sep='\t')
family_data = pd.read_csv('../data/family.tsv', sep='\t')
surveillance_data = pd.read_csv('../data/surveillance.tsv', sep='\t')

# for aa_data, only keep _0 rows
aa_data = aa_data[aa_data['sampleID'].str.endswith('_0')].copy()
# remove the timepoint from sampleID
aa_data['subjectID'] = aa_data['sampleID'].str.replace('_0', '', regex=False)
aa_data.drop(columns=['sampleID'], inplace=True)

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
    merged_df.rename(columns={'sampleID': 'subjectID'}, inplace=True)
    return merged_df

head_data = combine_df(head_data)

economic_features = economic_data[['subjectID', 'Household_heads_income', 'Family_expenditure_food_clothes_utilities']]
education_features = education_data[['subjectID','Fathers_Educational_attainment','Mothers_Educational_attainment']]
family_features = family_data[['subjectID', 'Number_of_living_children']]
surveillance_features = surveillance_data[['subjectID', 'antibiotic', 'fail_no.failure']]
meta_features = meta_data[['subjectID','Sex','Delivery_Mode','PoB','BF']]

# one-hot encode sex, delivery mode, and place of birth
meta_features = pd.get_dummies(meta_features, columns=['Sex', 'Delivery_Mode', 'PoB'])

# combine all dataframes on subjectID
all_data = [
    PRS_data, aa_data, head_data, microbiome_data, economic_features, 
    education_features, family_features, surveillance_features, meta_features
]

all_subjects = meta_data['subjectID'].unique().tolist()
# create a new dataframe with all subjectIDs
combined_df = pd.DataFrame({'subjectID': list(all_subjects)})
# merge each binned dataframe into the combined dataframe
for df in all_data:
    combined_df = pd.merge(combined_df, df, on='subjectID', how='outer')

combined_df.to_csv('../data/combined_matrix.tsv', sep='\t', index=False)



def impute_and_scale_df(df):
    """
    df : DataFrame with subject_id as index
    returns: imputed & z-scaled DataFrame (values stay scaled)
    """
    imp = IterativeImputer(max_iter=20, random_state=10, skip_complete=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_imp_scaled = imp.fit_transform(X_scaled)
    return pd.DataFrame(
        X_imp_scaled,           # <- keep the scaled values
        index=df.index,
        columns=df.columns
    ), scaler                   # also return the scaler

combined_imputed_scaled, fitted_scaler = impute_and_scale_df(
    combined_df.set_index('subjectID')
)

# save both
combined_imputed_scaled.to_csv('../data/combined_imputed_scaled.tsv', sep='\t', index=True)
joblib.dump(fitted_scaler, '../data/scaler.save')