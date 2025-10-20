# This is the final script. It calculates cluster specific changes based on the average changes in clusters from what_if_counterfactuals.py

import pandas as pd
import numpy as np
import torch
import joblib
import os

from what_if_counterfactuals import predict_df

save_recipe = False
save_aa_ranges = False

selected_features = [
    'leucine', 'pc6', 'pc2', 'tyrosine', 'isoleucine',
    'aspartic_acid', 'methionine', 'glutamine', 'taurine', 'tmao'
]
restrict_to_selected_features = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intervention_data = pd.read_csv('../Outcomes/population_wlz_deltas_neg1.tsv', sep='\t')
clusters = pd.read_csv('../Outcomes/subject_cluster_assignments.tsv', sep='\t')

output_dir = '../Outcomes/cluster_interventions_original_units'
os.makedirs(output_dir, exist_ok=True)

df_scaled = pd.read_csv('../data/combined_imputed_scaled_large_nolip_psd.tsv', sep='\t', index_col=0)
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
mam_ids = meta_data[meta_data['Condition'] == 'MAM']['subjectID']
df_scaled = df_scaled.loc[df_scaled.index.intersection(mam_ids)]
ckpt = torch.load('../models/vae_heads_nolip_psd_mse.pt', map_location=DEVICE)
target_scaler = ckpt.get('target_scaler', None)
regression_cols = ckpt.get('regression_cols', ['cognitive_score_52', 'vocalisation_52', 'WLZ_WHZ_52'])

wlz_idx = regression_cols.index('WLZ_WHZ_52')
wlz_mean = target_scaler.mean_[wlz_idx]
wlz_std = target_scaler.scale_[wlz_idx]

intervention_profiles = intervention_data.pivot_table(
    index='subjectID', 
    columns='feature', 
    values='delta'
).fillna(0)

## Set up for scaling the amount of change req
# Load the scaler and original data to get scaling parameters for continuous features
SCALER_PATH = '../data/scaler_large_nolip_psd.save'
COMBINED_MATRIX_PATH = '../data/combined_matrix_large.tsv'

scaler = joblib.load(SCALER_PATH)
combined_df = pd.read_csv(COMBINED_MATRIX_PATH, sep='\t')

# Identify categorical/dummy columns (which should not be unscaled)
dummy_cols = [
    col for col in combined_df.columns
    if col != 'subjectID'
    and combined_df[col].dropna().nunique() == 2
    and set(combined_df[col].dropna().unique()) <= {0, 1}
]
categorical_cols = ['Number_of_rooms_in_current_household'] + dummy_cols

# Identify continuous columns (these are the ones we need to unscale)
continuous_cols = [c for c in combined_df.columns if c not in categorical_cols and c != 'subjectID']

# Create a dictionary mapping each continuous feature to its standard deviation (scale)
scales_dict = dict(zip(continuous_cols, scaler.scale_))


# add cluster ids to intervention profiles
intervention_profiles = intervention_profiles.join(clusters.set_index('subjectID'), on='subjectID')
# get from meta.tsv if these are people who would have not recovered
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
# unrecovered subjectIDs
unrecovered_ids = set(meta_data[meta_data['Recovery'] == 'No recovery']['subjectID'])

CONSENSUS_THRESHOLD = 0.4

# Prepare an empty dataframe to store our new, smarter archetypes
archetype_interventions_list = []

# Group by cluster to analyze each one separately
grouped_profiles = intervention_profiles.groupby('cluster_id')

for cluster_num, cluster_df in grouped_profiles:
    
    # This will store the intervention recipe for this specific cluster
    smart_archetype = {}
    
    # Don't include the 'cluster_id' column in the feature calculations
    features_to_check = cluster_df.columns.drop('cluster_id', errors='ignore')

    for feature in features_to_check:
        
        # Isolate the deltas for this feature, excluding zeros (no change needed)
        deltas = cluster_df[feature][cluster_df[feature] != 0]
        
        if deltas.empty:
            smart_archetype[feature] = 0
            continue

        # Calculate the proportion of positive changes
        num_positive = (deltas > 0).sum()
        proportion_positive = num_positive / len(deltas)

        # check if mostly positive or negative
        if proportion_positive >= CONSENSUS_THRESHOLD:
            mean_increase = deltas[deltas > 0].mean()
            std_increase = deltas[deltas > 0].std()
            smart_archetype[feature] = mean_increase + 1.5*std_increase
        elif (1 - proportion_positive) >= CONSENSUS_THRESHOLD:
            smart_archetype[feature] = deltas[deltas < 0].mean()
        else:
            # don't change if doesn't go one way or the other
            smart_archetype[feature] = 0
            
    # Add the completed recipe to list
    archetype_interventions_list.append(pd.Series(smart_archetype, name=cluster_num))

# Combine the list of series into a final df
archetype_interventions = pd.DataFrame(archetype_interventions_list)

print("four cluster types:")
print(archetype_interventions)


# id those below target wlz
_, preds_base = predict_df(df_scaled)
base_below = df_scaled.loc[preds_base['WLZ_WHZ_52'] < -1.0]

for cluster_num in archetype_interventions.index:

    # Get the intervention recipe for the current cluster
    cluster_delta = archetype_interventions.loc[cluster_num]

    if restrict_to_selected_features:
        cluster_delta = cluster_delta.copy()
        for feat in cluster_delta.index:
            if feat not in selected_features:
                cluster_delta[feat] = 0

    # Create a copy to store the unscaled values
    unscaled_archetype = cluster_delta.copy()
    
    # Unscale the delta for each continuous feature
    for feature, scaled_delta in unscaled_archetype.items():
        if feature in scales_dict:
            unscaled_archetype[feature] = scaled_delta * scales_dict[feature]
    
    # Filter for only the features that have a non-zero change
    unscaled_archetype_to_save = unscaled_archetype[unscaled_archetype != 0].reset_index()
    unscaled_archetype_to_save.columns = ['feature', 'required_change_original_units']
    
    # Save the unscaled intervention to its own tsv file
    if save_recipe:
        file_path = os.path.join(output_dir, f'cluster_{cluster_num}_intervention_plan.tsv')
        unscaled_archetype_to_save.to_csv(file_path, sep='\t', index=False)


    # Apply this cluster to the whole base_below group
    df_below_int = base_below.copy()
    for feat, change_value in cluster_delta.items():
        if feat in df_below_int.columns:
            df_below_int[feat] = df_below_int[feat] + change_value

    # Predict and count the recovered
    _, preds_below_int = predict_df(df_below_int)
    recovered_all = preds_below_int[preds_below_int['WLZ_WHZ_52'] >= -1.0]
    total_newly_recovered = len(recovered_all)

    # How many from cluster_num recovered
    num_in_cluster = (clusters['cluster_id'] == cluster_num).sum()
    
    # Get the subjectIDs that originally belong to this cluster
    subjects_in_cluster = intervention_profiles[intervention_profiles['cluster_id'] == cluster_num].index
    at_risk_in_cluster = base_below.index.intersection(subjects_in_cluster)
    total_at_risk_in_cluster = len(at_risk_in_cluster)
    recovered_in_cluster = recovered_all.index.intersection(at_risk_in_cluster)
    num_recovered_in_cluster = len(recovered_in_cluster)

    unrecovered_in_cluster = set(subjects_in_cluster).intersection(unrecovered_ids)
    recovered_unrecovered_in_cluster = recovered_all.index.intersection(unrecovered_in_cluster)
    num_recovered_unrecovered_in_cluster = len(recovered_unrecovered_in_cluster)
    total_unrecovered_in_cluster = len(unrecovered_in_cluster)

    print(f"Cluster {cluster_num}: {num_recovered_unrecovered_in_cluster} of {total_unrecovered_in_cluster} ({(num_recovered_unrecovered_in_cluster/total_unrecovered_in_cluster*100 if total_unrecovered_in_cluster else 0):.2f}%) unrecovered individuals recovered with the intervention change.")


# overall recovery
thresh = -1.0


at_risk_subject_ids = intervention_data['subjectID'].unique()
base_below = df_scaled.loc[df_scaled.index.intersection(at_risk_subject_ids)]
total_at_risk = len(base_below)

all_profiles = intervention_data.pivot_table(
    index='subjectID', 
    columns='feature', 
    values='delta'
).fillna(0)

# Filter these profiles to only include the unrecovered individuals
unrecovered_profiles = all_profiles.loc[base_below.index]
# code is same as cluster specific just for everyone in one
CONSENSUS_THRESHOLD = 0.4

overall_smart_archetype = {}

for feature in unrecovered_profiles.columns:
    
    deltas = unrecovered_profiles[feature][unrecovered_profiles[feature] != 0]

    if deltas.empty:
        overall_smart_archetype[feature] = 0
        continue

    # Calculate the proportion of people needing an inc
    proportion_positive = (deltas > 0).sum() / len(deltas)


    if proportion_positive >= CONSENSUS_THRESHOLD:
        overall_smart_archetype[feature] = deltas[deltas > 0].mean()
    elif (1 - proportion_positive) >= CONSENSUS_THRESHOLD:
        overall_smart_archetype[feature] = deltas[deltas < 0].mean()
    else:
        overall_smart_archetype[feature] = 0

overall_archetype_series = pd.Series(overall_smart_archetype)

df_below_int = base_below.copy()

# Apply the calculated changes
for feature, change_value in overall_archetype_series.items():
    if feature in df_below_int.columns:
        df_below_int[feature] = df_below_int[feature] + change_value

# Predict new WLZ scores on the modified data
_, preds_below_int = predict_df(df_below_int)

# Count how many are now at or above the -1.0 threshold
num_recovered = (preds_below_int['WLZ_WHZ_52'] >= thresh).sum()

recovered_in_preds = preds_below_int.loc[preds_below_int.index.intersection(unrecovered_ids)]
num_recovered_would_have = (recovered_in_preds['WLZ_WHZ_52'] >= thresh).sum()
print(f"{num_recovered_would_have} of {len(unrecovered_ids)} ({num_recovered_would_have/len(unrecovered_ids)*100:.2f}%) unrecovered individuals recovered with the intervention change.")

# for the clusters, save a tsv file with the amino acid ranges for each cluster
aa_data = pd.read_csv('../data/aa.tsv', sep='\t')
# for aa_data, only keep _0 rows
aa_data = aa_data[aa_data['sampleID'].str.endswith('_0')].copy()
# remove the timepoint from sampleID
aa_data['subjectID'] = aa_data['sampleID'].str.replace('_0', '', regex=False)
aa_data.drop(columns=['sampleID'], inplace=True)
# add the cluster assignments to aa_data
aa_data = aa_data.join(clusters.set_index('subjectID'), on='subjectID', how='inner')
# for each cluster, get the min and max of each amino acid
for cluster_num, cluster_df in aa_data.groupby('cluster_id'):
    aa_ranges = {}
    for col in aa_data.columns:
        if col not in ['subjectID', 'cluster_id']:
            aa_ranges[col] = (cluster_df[col].min(), cluster_df[col].max())
    aa_ranges_df = pd.DataFrame(aa_ranges, index=['min', 'max']).T
    file_path = os.path.join(output_dir, f'cluster_{cluster_num}_amino_acid_ranges.tsv')
    if save_aa_ranges:
        aa_ranges_df.to_csv(file_path, sep='\t', index=True)
