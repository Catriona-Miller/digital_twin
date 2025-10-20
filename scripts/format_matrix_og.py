# This script formats the input data for the VAE model by imputing missing values and scaling the features. Different datasets combined.

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Needed to enable IterativeImputer (even though not called)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np


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
anthro_data = pd.read_csv('../data/anthro.tsv', sep='\t')
fcis_data = pd.read_csv('../data/fcis.tsv', sep='\t')
glitter_data = pd.read_csv('../data/glitter.tsv', sep='\t')
household_data = pd.read_csv('../data/household.tsv', sep='\t')
lipids_data = pd.read_csv('../data/lipids.tsv', sep='\t')
psd_data = pd.read_csv('../data/psd.tsv', sep='\t')
pss_data = pd.read_csv('../data/pss.tsv', sep='\t')
sanitation_data = pd.read_csv('../data/sanitation.tsv', sep='\t')
sleep_data = pd.read_csv('../data/sleep.tsv', sep='\t')
vitamin_data = pd.read_csv('../data/vitamin.tsv', sep='\t')

# for aa_data, only keep _0 rows
aa_data = aa_data[aa_data['sampleID'].str.endswith('_0')].copy()
# remove the timepoint from sampleID
aa_data['subjectID'] = aa_data['sampleID'].str.replace('_0', '', regex=False)
aa_data.drop(columns=['sampleID'], inplace=True)

# same with head data
head_data = head_data[head_data['sampleID'].str.endswith('_0')].copy()
head_data['subjectID'] = head_data['sampleID'].str.replace('_0', '', regex=False)
head_data.drop(columns=['sampleID'], inplace=True)

# same with anthro data
anthro_data = anthro_data[anthro_data['sampleID'].str.endswith('_0')].copy()
anthro_data['subjectID'] = anthro_data['sampleID'].str.replace('_0', '', regex=False)
anthro_data.drop(columns=['sampleID'], inplace=True)

# same with lipids data
lipids_data = lipids_data[lipids_data['sampleID'].str.endswith('_0')].copy()
lipids_data['subjectID'] = lipids_data['sampleID'].str.replace('_0', '', regex=False)
lipids_data.drop(columns=['sampleID'], inplace=True)

# same with psd data
psd_data = psd_data[psd_data['sampleID'].str.endswith('_0')].copy()
psd_data['subjectID'] = psd_data['sampleID'].str.replace('_0', '', regex=False)
psd_data.drop(columns=['sampleID'], inplace=True)

# same with pss data
pss_data = pss_data[pss_data['sampleID'].str.endswith('_0')].copy()
pss_data['subjectID'] = pss_data['sampleID'].str.replace('_0', '', regex=False)
pss_data.drop(columns=['sampleID'], inplace=True)

# same with sleep data
sleep_data = sleep_data[sleep_data['sampleID'].str.endswith('_0')].copy()
sleep_data['subjectID'] = sleep_data['sampleID'].str.replace('_0', '', regex=False)
sleep_data.drop(columns=['sampleID'], inplace=True)

# same with vitamin data
vitamin_data = vitamin_data[vitamin_data['sampleID'].str.endswith('_0')].copy()
vitamin_data['subjectID'] = vitamin_data['sampleID'].str.replace('_0', '', regex=False)
vitamin_data.drop(columns=['sampleID'], inplace=True)

# same with wolkes data
wolkes_data = wolkes_data[wolkes_data['sampleID'].str.endswith('_0')].copy()
wolkes_data['subjectID'] = wolkes_data['sampleID'].str.replace('_0', '', regex=False)
wolkes_data.drop(columns=['sampleID'], inplace=True)

# same with bayley data
bayley_data = bayley_data[bayley_data['sampleID'].str.endswith('_0')].copy()
bayley_data['subjectID'] = bayley_data['sampleID'].str.replace('_0', '', regex=False)
bayley_data.drop(columns=['sampleID'], inplace=True)

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

fcis_data = combine_df(fcis_data)
glitter_data = combine_df(glitter_data)

economic_features = economic_data[['subjectID', 'Household_heads_income', 'Family_expenditure_food_clothes_utilities']]
# make Mothers_income binary and include
economic_features['Mothers_income'] = economic_data['Mothers_income'].apply(lambda x: 'null' if x == 0 else 'earning')
economic_features = pd.get_dummies(economic_features, columns=['Mothers_income'])
education_features = education_data[['subjectID','Fathers_Educational_attainment','Mothers_Educational_attainment', 'Family_owns_the_home_they_live_in']]
education_features = pd.get_dummies(education_features, columns=['Family_owns_the_home_they_live_in'])
family_features = family_data[['subjectID', 'Number_of_living_children']]
surveillance_features = surveillance_data[['subjectID', 'antibiotic', 'fail_no.failure']]
meta_features = meta_data[['subjectID','Sex','Delivery_Mode','PoB','BF', 'Feed', 'Supplementation', 'Ethnicity']]
fcis_data = fcis_data[['subjectID', 'Total_FCIS_0']]
glitter_data = glitter_data[['subjectID', 'glitter_seconds_0']]
household_data = household_data[['subjectID', 'Members_in_household', 'Principal_roofing_material', 'Cooking_gas', 'Tabletop', 
                                 'Chair', 'Working_TV', 'Number_of_rooms_in_current_household', 'Household_food_availability']]
household_data = pd.get_dummies(household_data, columns=['Principal_roofing_material', 'Cooking_gas', 'Tabletop', 'Chair', 'Working_TV', 'Household_food_availability'])
# one-hot encode sex, delivery mode, and place of birth
meta_features = pd.get_dummies(meta_features, columns=['Sex', 'Delivery_Mode', 'PoB', 'Feed', 'Supplementation', 'Ethnicity'])
sanitation_data = sanitation_data[['subjectID', 'agent_used_before_feeding_child', 'method_used_before_feeding_child', 'agent_used_before_eating', 'method_used_before_eating', 
                                   'agent_used_after_defecating', 'Open_drain_beside_house', 'Place_for_cooking_for_household', 
                                   'Toilet_facility_shared_with_other_households', 'Principal_type_of_toilet_facility_used_by_household_members']]
sanitation_data = pd.get_dummies(sanitation_data, columns=['Open_drain_beside_house', 'Toilet_facility_shared_with_other_households', 'Principal_type_of_toilet_facility_used_by_household_members'])
sanitation_data['soap_before_feeding_child'] = sanitation_data['agent_used_before_feeding_child'] == 'Soap'
sanitation_data['water_before_feeding_child'] = sanitation_data['agent_used_before_feeding_child'] == 'Water'
sanitation_data['both_before_feeding_child'] = sanitation_data['method_used_before_feeding_child'] == 'Both hands'
sanitation_data['right_hand_before_feeding_child'] = sanitation_data['method_used_before_feeding_child'] == 'Right hand'
sanitation_data['soap_before_eating'] = sanitation_data['agent_used_before_eating'] == 'Soap'
sanitation_data['water_before_eating'] = sanitation_data['agent_used_before_eating'] == 'Water'
sanitation_data['both_before_eating'] = sanitation_data['method_used_before_eating'] == 'Both hands'
sanitation_data['right_hand_before_eating'] = sanitation_data['method_used_before_eating'] == 'Right hand'
sanitation_data['soap_after_defecating'] = sanitation_data['agent_used_after_defecating'] == 'Soap'
sanitation_data['water_after_defecating'] = sanitation_data['agent_used_after_defecating'] == 'Water'
sanitation_data['cooking_indoors'] = sanitation_data['Place_for_cooking_for_household'] == 'Inside house'
sanitation_data['cooking_outside'] = sanitation_data['Place_for_cooking_for_household'] == 'Outdoors'
sanitation_data.drop(columns=['agent_used_before_feeding_child', 'method_used_before_feeding_child', 'agent_used_before_eating', 
                              'method_used_before_eating', 'agent_used_after_defecating', 'Place_for_cooking_for_household'], inplace=True)

sleep_data = sleep_data[['subjectID', 'Bedtime_difficulty', 'Night_sleep']]

# combine all dataframes on subjectID. Add or remove from this below e.g. lipids and psd
all_data = [
    PRS_data, aa_data, microbiome_data, economic_features, head_data, meta_features, education_features,
    family_features, surveillance_features, anthro_data, fcis_data, glitter_data, household_data, 
    pss_data, sanitation_data, sleep_data, vitamin_data, wolkes_data, bayley_data]

all_subjects = meta_data['subjectID'].unique().tolist()
# create a new dataframe with all subjectIDs
combined_df = pd.DataFrame({'subjectID': list(all_subjects)})
# merge each binned dataframe into the combined dataframe
for df in all_data:
    combined_df = pd.merge(combined_df, df, on='subjectID', how='outer')

combined_df.to_csv('../data/combined_matrix_large.tsv', sep='\t', index=False)

dummy_cols = [
    col for col in combined_df.columns
    if combined_df[col].dropna().nunique() == 2 and set(combined_df[col].dropna().unique()) <= {0, 1}
]

categorical_cols = ['Number_of_rooms_in_current_household'] + dummy_cols

def impute_and_scale_df(df, categorical_cols=None):
    """
    Imputes missing values for categorical and continuous variables separately and then scales them separately too
    df : DataFrame with subject_id as index
    returns: imputed & z-scaled DataFrame (values stay scaled), scaler
    """

    continuous_cols = [col for col in df.columns if col not in categorical_cols]

    imp = IterativeImputer(max_iter=20, random_state=10, skip_complete=True)
    scaler = StandardScaler()

    cont_imputed = imp.fit_transform(df[continuous_cols])
    cont_scaled = scaler.fit_transform(cont_imputed)

    cat_imputed = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])

    X_combined = np.hstack([cont_scaled, cat_imputed])
    all_cols = continuous_cols + categorical_cols

    df_out = pd.DataFrame(
        X_combined,
        index=df.index,
        columns=all_cols
    )
    return df_out, scaler 

combined_imputed_scaled, fitted_scaler = impute_and_scale_df(combined_df.set_index('subjectID'), categorical_cols=categorical_cols)

# save both
#combined_imputed_scaled.to_csv('../data/combined_imputed_scaled_large_nolip_psd.tsv', sep='\t', index=True)
#joblib.dump(fitted_scaler, '../data/scaler_large_nolip_psd.save')

# also save a file with just the editable ones for what if scenarios
# editable_sources = [household_data, sanitation_data, microbiome_data, aa_data, vitamin_data]
# editable_cols = [col for df in editable_sources for col in df.columns if col != 'subjectID']

# combined_imputed_scaled[editable_cols].to_csv('../data/combined_imputed_scaled_large_nolip_sleep_editable.tsv', sep='\t', index=True)