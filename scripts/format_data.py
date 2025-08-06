# Format the data into bins of same width labelled by letters
import numpy as np
import pandas as pd
import qbiome
from qbiome.quantizer import Quantizer
from qbiome.qnet_orchestrator import QnetOrchestrator
from qbiome.hypothesis import Hypothesis


save = False
train = True

PRS_data = pd.read_csv('../data/genetics.tsv', sep='\t')
aa_data = pd.read_csv('../data/aa.tsv', sep='\t')
microbiome_data = pd.read_csv('../data/genus.tsv', sep='\t')
wolkes_data = pd.read_csv('../data/wolkes.tsv', sep='\t')
bayley_data = pd.read_csv('../data/bayley.tsv', sep='\t')
economic_data = pd.read_csv('../data/economics.tsv', sep='\t')
head_data = pd.read_csv('../data/head.tsv', sep='\t')
meta_data = pd.read_csv('../data/meta.tsv', sep='\t')
education_data = pd.read_csv('../data/education.tsv', sep='\t')
family_data = pd.read_csv('../data/family.tsv', sep='\t')
surveillance_data = pd.read_csv('../data/surveillance.tsv', sep='\t')

# for each column bar subectID, bin the values into 26 bins and label them A-Z
def bin_data(df, bin_no=26):
    binned_df = df.copy()
    for col in df.columns[1:]:  # Skip the first column (subjectID)
        # skip columns that are all zeroes and remove from binned_df
        if df[col].sum() == 0:
            binned_df.drop(columns=[col], inplace=True)
            continue
        # Create bins of equal width
        bins = np.linspace(df[col].min(), df[col].max(), bin_no+1)
        labels = [chr(i) for i in range(65, 65+bin_no)]  # A-Z for 26 bins
        binned_df[col] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return binned_df

# same as bin_data but here sampleIDs are *_timepoint (0 or 52). I want instead the features to be timepoints
def bin_data_timepoints(df, bin_no=26):
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
    binned_df = bin_data(merged_df, bin_no)
    return binned_df

binned_PRS_data = bin_data(PRS_data)
binned_aa_data = bin_data_timepoints(aa_data)
binned_head_data = bin_data_timepoints(head_data)
binned_microbiome_data = bin_data_timepoints(microbiome_data)
binned_wolkes_data = bin_data_timepoints(wolkes_data, bin_no=9)
binned_bayley_data = bin_data_timepoints(bayley_data)

economic_features = economic_data[['subjectID', 'Household_heads_income', 'Family_expenditure_food_clothes_utilities']]
binned_economic_features = bin_data(economic_features)
# make Mothers_income binary and include
binned_economic_features['Mothers_income'] = economic_data['Mothers_income'].apply(lambda x: 'A' if x == 0 else 'B')

education_features = education_data[['subjectID','Fathers_Educational_attainment','Mothers_Educational_attainment']]
binned_education_features = bin_data(education_features, bin_no=15)

family_features = family_data[['subjectID', 'Number_of_living_children']]
binned_family_features = bin_data(family_features, bin_no=6)

surveillance_features = surveillance_data[['subjectID', 'antibiotic', 'fail_no.failure']]
binned_surveillance_features = bin_data(surveillance_features)

meta_features = meta_data[['subjectID','Sex','Delivery_Mode','PoB','BF']]
meta_features['Sex'] = meta_features['Sex'].apply(lambda x: 'A' if x == 'Male' else 'B')
meta_features['Delivery_Mode'] = meta_features['Delivery_Mode'].apply(lambda x: 'A' if x == 'Vaginal' else 'B')
meta_features['PoB'] = meta_features['PoB'].apply(lambda x: 'A' if x == 'Clinic' else 'B')
# For breastfeeding, bin numbers 0-7 into A-H
meta_features['BF'] = pd.cut(meta_features['BF'], bins=8, labels=[chr(i) for i in range(65, 73)], include_lowest=True)

all_data = [binned_PRS_data, binned_aa_data, binned_head_data, binned_microbiome_data, 
            binned_wolkes_data, binned_bayley_data, binned_economic_features, 
            binned_education_features, binned_family_features, binned_surveillance_features, meta_features]
# combine all binned data into one dataframe. If a subjectID is not in all dataframes, fill with NaN
# first, get all unique subjectIDs from all dataframes
all_subjects = set(binned_PRS_data['subjectID'])
for df in all_data:
    all_subjects.update(df[df.columns[0]].unique())
# create a new dataframe with all subjectIDs
combined_df = pd.DataFrame({'subjectID': list(all_subjects)})
# merge each binned dataframe into the combined dataframe
for df in all_data:
    combined_df = pd.merge(combined_df, df, on='subjectID', how='outer')

# for any features in combined_df that don't finish with _0 or _52, add _0 to the end
for col in combined_df.columns:
    if col != 'subjectID' and not col.endswith('_0') and not col.endswith('_52'):
        combined_df.rename(columns={col: col + '_0'}, inplace=True)

# separate the combined dataframe into meta_data['Condition'] == MAM and Well-nourished based on subjectID
mam_subjects = meta_data[meta_data['Condition'] == 'MAM']['subjectID']
control_subjects = meta_data[meta_data['Condition'] == 'Well-nourished']['subjectID']
mam_df = combined_df[combined_df['subjectID'].isin(mam_subjects)]
control_df = combined_df[combined_df['subjectID'].isin(control_subjects)]

# separate the mam dataframe into localRUSF and ERUSF
localRUSF = meta_data[meta_data['Feed'] == 'Local RUSF (A)']['subjectID']
eRUSF = meta_data[meta_data['Feed'] == 'ERUSF (B)']['subjectID']
localRUSF_df = mam_df[mam_df['subjectID'].isin(localRUSF)]
eRUSF_df = mam_df[mam_df['subjectID'].isin(eRUSF)]

if save:
    meta_features.to_csv('../data/binned/binned_meta.tsv', sep='\t', index=False)
    binned_surveillance_features.to_csv('../data/binned/binned_surveillance.tsv', sep='\t', index=False)
    binned_family_features.to_csv('../data/binned/binned_family.tsv', sep='\t', index=False)
    binned_education_features.to_csv('../data/binned/binned_education.tsv', sep='\t', index=False)
    binned_bayley_data.to_csv('../data/binned/binned_bayley.tsv', sep='\t', index=False)
    binned_PRS_data.to_csv('../data/binned/binned_genetics.tsv', sep='\t', index=False)
    binned_aa_data.to_csv('../data/binned/binned_aa.tsv', sep='\t', index=False)
    binned_head_data.to_csv('../data/binned/binned_head.tsv', sep='\t')
    binned_microbiome_data.to_csv('../data/binned/binned_genus.tsv', sep='\t', index=False)
    binned_wolkes_data.to_csv('../data/binned/binned_wolkes.tsv', sep='\t', index=False)
    binned_economic_features.to_csv('../data/binned/binned_economic.tsv', sep='\t', index=False)
    combined_df.to_csv('../data/binned/binned_all.tsv', sep='\t', index=False)
    mam_df.to_csv('../data/binned/binned_mam.tsv', sep='\t', index=False)
    control_df.to_csv('../data/binned/binned_control.tsv', sep='\t', index=False)
    localRUSF_df.to_csv('../data/binned/binned_localRUSF.tsv', sep='\t', index=False)
    eRUSF_df.to_csv('../data/binned/binned_ERUSF.tsv', sep='\t', index=False)

quantizer = Quantizer()
mam_df['subject_id'] = mam_df['subjectID']
mam_df.drop(columns=['subjectID'], inplace=True)
mam_features, mam_labels = quantizer.get_qnet_inputs(mam_df)

qnet_orchestrator = QnetOrchestrator(quantizer)

if train:
    qnet_orchestrator.train_qnet(
        mam_features, mam_labels, alpha=0.3, min_samples_split=2, out_fname=None
    )
    qnet_orchestrator.save_qnet("../models/mam_qnet.pkl",GZIP=True)

# qnet_orchestrator.load_qnet("../models/mam_qnet.pkl.gz",GZIP=True)
# hypothesis = Hypothesis(
#     quantizer=quantizer,qnet_orchestrator=qnet_orchestrator, detailed_labels=True)
# hypothesis.causal_constraint = 0
# hypothesis.no_self_loops = False

# hypothesis.get()
# print("Number of hypotheses found:", len(hypothesis.hypotheses))
# print(hypothesis.hypotheses.head())
#hypothesis.to_dot("hypothesis_1_10.dot")
#hypothesis.hypotheses.sort_values("src")

#from qbiome.network import Network
#Network('hypothesis_1_10.dot',outfile='net.png').get()