# figure functions to make plots based on outputs from what_if_counterfactuals.py. Ones in paper are create_population_fig and create_full_clustermap_clusters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from what_if_counterfactuals import run_single_subject_cf
from scipy.cluster.hierarchy import fcluster
from matplotlib.patches import Patch 
import os
from statannotations.Annotator import Annotator

def create_population_fig(intervention_data):
    '''
    Creates a combined bar and strip plot to visualise both
    population-level average interventions required and individual-level variations.
    Displays the top 10 features with the highest mean absolute delta. Everything in z-scaled units
    '''
    mean_abs_delta = intervention_data.groupby('feature')['delta'].apply(lambda x: x.abs().mean())
    mean_delta = intervention_data.groupby('feature')['delta'].mean()

    summary_df = pd.DataFrame({
        'mean_abs_delta': mean_abs_delta,
        'mean_delta': mean_delta
    }).sort_values('mean_abs_delta', ascending=False).head(10)

    # Filter the raw data to include only the top N features
    plot_data = intervention_data[intervention_data['feature'].isin(summary_df.index)]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create a direction column for the hue param (needed now if give palette?)
    summary_df['Direction'] = np.where(summary_df['mean_delta'] > 0, 'Net Increase', 'Net Decrease')

    sns.barplot(
        x='mean_delta',
        y=summary_df.index,
        data=summary_df,
        hue='Direction',
        palette={'Net Increase': '#2ca02c', 'Net Decrease': '#d62728'},
        dodge=False,
        alpha=0.6,  
        ax=ax
    )

    # Put dots on top
    sns.stripplot(
        x='delta',
        y='feature',
        data=plot_data,
        ax=ax,
        order=summary_df.index,
        color='black',          
        jitter=0.3,
        s=3,
        alpha=0.5
    )

    ax.set_xlabel('Recommended Change in Scaled Units', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()

    plt.tight_layout()
    fig.savefig('../Outcomes/figure_combined_neg1_small.pdf', format='pdf', dpi=450, bbox_inches='tight')

def create_radar_plot(intervention_data: pd.DataFrame, top_n_features: int = 8):
    """
    Generates a radar plot comparing the intervention strategies for some individuals.
    """
    mean_abs_delta = intervention_data.groupby('feature')['delta'].apply(lambda x: x.abs().mean())
    top_features = mean_abs_delta.nlargest(top_n_features).index.tolist()

    # Choose three random individuals
    subject_ids = intervention_data['subjectID'].unique()
    np.random.seed(10)
    selected_subjects = np.random.choice(subject_ids, size=3, replace=False)
    
    plot_data = intervention_data[
        (intervention_data['subjectID'].isin(selected_subjects)) &
        (intervention_data['feature'].isin(top_features))
    ]

    radar_df = plot_data.pivot(
        index='subjectID',
        columns='feature',
        values='delta'
    ).fillna(0)
    
    radar_df = radar_df[top_features]
    
    labels = radar_df.columns.tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    #plot radar for one individual
    def add_to_radar(subject_id, colour):
        values = radar_df.loc[subject_id].tolist()
        values += values[:1] 
        ax.plot(angles, values, color=colour, linewidth=2, linestyle='solid', label=subject_id)
        ax.fill(angles, values, color=colour, alpha=0.25)

    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, subject_id in enumerate(selected_subjects):
        add_to_radar(subject_id, colour=colours[i])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    ax.set_rlabel_position(180 / num_vars)
    circle_angles = np.linspace(0, 2 * np.pi, 100)
    # Draw the circle at radius 0
    ax.plot(circle_angles, [0] * 100, color='black', linewidth=1)   
    plt.title('Personalised Intervention Radar for Three Individuals', size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig('../Outcomes/figure_radar_plot_three_1.png', dpi=300, bbox_inches='tight')

def create_full_clustermap(intervention_data: pd.DataFrame, top_n_features: int = 10):
    """
    Generates a clustered heatmap for all individuals in the dataset. Uses top N features by mean absolute delta.
    All data is z-scaled
    """
    mean_abs_delta = intervention_data.groupby('feature')['delta'].apply(lambda x: x.abs().mean())
    top_features = mean_abs_delta.nlargest(top_n_features).index.tolist()

    heatmap_data = intervention_data[intervention_data['feature'].isin(top_features)]

    heatmap_matrix = heatmap_data.pivot(
        index='subjectID',
        columns='feature',
        values='delta'
    ).fillna(0)
    
    heatmap_matrix = heatmap_matrix[top_features]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    g = sns.clustermap(
        heatmap_matrix,
        method='ward',
        cmap='vlag',
        center=0,
        figsize=(7, 9),
        yticklabels=False,
        cbar_kws={'label': 'Recommended Change'}
    )
    g.ax_heatmap.set_xlabel('Top 10 Features', fontsize=10, fontweight='bold')
    g.ax_heatmap.set_ylabel('') 
    
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.savefig('../Outcomes/figure_full_clustermap_neg1_small.svg', format='svg', dpi=450, bbox_inches='tight')


def create_full_clustermap_clusters(intervention_data: pd.DataFrame, top_n_features: int = 10, n_clusters: int = 2):
    """
    Same as above, just outputs the cluster assignments (four clusters) to tsv and adds colour on side
    Should just combine both functions and make tsv output optional but haven't yet...
    """

    mean_abs_delta = intervention_data.groupby('feature')['delta'].apply(lambda x: x.abs().mean())
    top_features = mean_abs_delta.nlargest(top_n_features).index.tolist()
    heatmap_data = intervention_data[intervention_data['feature'].isin(top_features)]
    heatmap_matrix = heatmap_data.pivot(index='subjectID', columns='feature', values='delta').fillna(0)
    heatmap_matrix = heatmap_matrix[top_features]

    # Perform clustering and extract cluster labels
    clustergrid = sns.clustermap(heatmap_matrix, method='ward', cmap='vlag', center=0)
    plt.close() 

    cluster_labels = fcluster(clustergrid.dendrogram_row.linkage, n_clusters, criterion='maxclust')
    clusters = pd.Series(cluster_labels, index=heatmap_matrix.index, name='cluster_id')
    
    # Create coloured labels for visualisation
    unique_clusters = sorted(clusters.unique())
    palette = sns.color_palette("hls", n_clusters)
    lut = dict(zip(unique_clusters, palette))
    row_colours = clusters.map(lut)

    # Redraw the clustermap with the colored side-bar
    g = sns.clustermap(
        heatmap_matrix,
        method='ward',
        cmap='vlag',
        center=0,
        figsize=(16, 12),
        yticklabels=False,
        row_colors=row_colours,
        cbar_kws={'label': 'Recommended Change (Delta)'}
    )

    # Create and add the legend for the clusters
    legend_patches = [Patch(facecolor=colour, edgecolor='black', label=f'Cluster {label}') 
                      for label, colour in lut.items()]

    g.ax_heatmap.set_xlabel('Top 10 Features', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('')
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)

    g.savefig('../Outcomes/figure_full_clustermap_clusters_small.pdf', format='pdf', dpi=450, bbox_inches='tight')
    clusters.to_csv('../Outcomes/subject_cluster_assignments_small.tsv', sep='\t', header=True)
    
    return clusters

def create_individual_dumbbell_plot(subject_id: str,
                                    top_n: int = 10):
    """
    Creates a dumbbell plot to visualise the before-and-after changes
    for a single individual's counterfactual analysis. Only includes top n features by abs change
    """

    # get the counterfactual result for subject (from what_if_counterfactuals)
    cf_results = run_single_subject_cf(subject_id, target_wlz_gain=1.0)
    
    changes_df = cf_results['changes_df']
    wlz_base = cf_results['wlz_base']
    wlz_cf = cf_results['wlz_cf']

    plot_data = changes_df.head(top_n).sort_values('delta')
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # plot delta as horizontal lines with point for x0 and x_cf at each end
    ax.hlines(y=plot_data.index, xmin=plot_data['x0'], xmax=plot_data['x_cf'],
              color='gray', alpha=0.7, linewidth=2, zorder=1)
    ax.scatter(plot_data['x0'], plot_data.index, color='#1f77b4', s=80, label='Original (x0)', zorder=2)
    ax.scatter(plot_data['x_cf'], plot_data.index, color='#ff7f0e', s=80, label='Counterfactual (x_cf)', zorder=2)

    ax.set_xlabel('Scaled Value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    wlz_delta = wlz_cf - wlz_base
    title = (
        f'Counterfactual Changes for {subject_id}\n'
        f'WLZ Score: {wlz_base:.2f} → {wlz_cf:.2f} (Delta {wlz_delta:+.2f})'
    )
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    
    fig.savefig(f'../Outcomes/dumbbell_plot_{subject_id}_1.png', dpi=300, bbox_inches='tight')

def plot_aa_violins_ratio(aa_file_path: str = '../data/aa.tsv', outcomes_dir: str = '../Outcomes'):
    """
    Plots violin ratios of essential to non-essential amino acids
    for individuals in each cluster (from pop cluster fig), separately for timepoints 0 and 52.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # get cluster assignments
    assign_path = '../Outcomes/subject_cluster_assignments_small.tsv'
    cl_df = pd.read_csv(assign_path, sep='\t')
    cl_df = cl_df.set_index(['subjectID'])

    aa = pd.read_csv(aa_file_path, sep='\t')
    aa = aa.set_index(aa.columns[0])

    # get essential and non-essential aas
    essential_aas = ['arginine', 'histidine', 'isoleucine', 'leucine', 'lysine', 'methionine', 'phenylalanine', 'taurine', 'threonine', 'trytophan', 'valine']
    non_essential_aas = ['alanine', 'asparagine', 'aspartic_acid', 'citrulline', 'glutamic_acid', 'glutamine', 'glycine', 'hydroxyproline', 'proline', 'serine', 'tyrosine','ornithine']

    aa_0 = aa[aa.index.str.endswith('_0')].copy()
    aa_52 = aa[aa.index.str.endswith('_52')].copy()

    # add cluster number as col to aa_0
    aa_0['subjectID'] = aa_0.index.str.replace('_0', '', regex=False)
    aa_0 = aa_0.join(cl_df, on='subjectID', how='inner')
    aa_0 = aa_0.drop(columns=['subjectID'])

    # add cluster number as col to aa_52
    aa_52['subjectID'] = aa_52.index.str.replace('_52', '', regex=False)
    aa_52 = aa_52.join(cl_df, on='subjectID', how='inner')
    aa_52 = aa_52.drop(columns=['subjectID'])

    # get the ratio of essential to non-essential aas for both timepoints
    aa_0['ess_noness_ratio'] = aa_0[essential_aas].sum(axis=1) / aa_0[non_essential_aas].sum(axis=1)
    aa_52['ess_noness_ratio'] = aa_52[essential_aas].sum(axis=1) / aa_52[non_essential_aas].sum(axis=1)

    # Plot the ratios at a given timepoint
    def plot_violins(data: pd.DataFrame, timepoint: int, pairs: list = [(1, 2)]):
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='cluster_id', y='ess_noness_ratio', data=data, palette='Set2')
        plt.title(f'Essential to Non-Essential AA Ratio by Cluster (Timepoint {timepoint})', fontsize=14)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Essential/Non-Essential Ratio', fontsize=12)
        plt.tight_layout()

        # add statistical annotations between two clusters
        annotator = Annotator(plt.gca(), pairs, data=data, x='cluster_id', y='ess_noness_ratio')
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=0)
        annotator.apply_and_annotate()

        #plt.savefig(os.path.join(outcomes_dir, f'aa_ess_noness_ratio_timepoint_{timepoint}.pdf'), format='pdf', dpi=300, bbox_inches='tight')

    # Plot for timepoint 0
    plot_violins(aa_0, timepoint=0)
    # Plot for timepoint 52
    plot_violins(aa_52, timepoint=52)

    # Also, the difference between timepoints
    aa_diff = aa_52[essential_aas + non_essential_aas].reset_index(drop=True) - aa_0[essential_aas + non_essential_aas].reset_index(drop=True)
    aa_diff['cluster_id'] = aa_52['cluster_id'].values
    aa_diff['ess_noness_ratio'] = aa_diff[essential_aas].sum(axis=1) / aa_diff[non_essential_aas].sum(axis=1)
    plot_violins(aa_diff, timepoint='52-0 (difference)')

    # Add in healthys to timepoint 0
    meta = pd.read_csv('../data/meta.tsv', sep='\t')
    healthy_ids = meta[meta['Condition'] == 'Well-nourished']['subjectID'].tolist()
    aa_healthy = aa.loc[[f'{sid}_0' for sid in healthy_ids if f'{sid}_0' in aa.index]].copy()
    aa_healthy['cluster_id'] = 3
    aa_healthy['ess_noness_ratio'] = aa_healthy[essential_aas].sum(axis=1) / aa_healthy[non_essential_aas].sum(axis=1)
    aa_0_with_healthy = pd.concat([aa_0, aa_healthy], axis=0)
    plot_violins(aa_0_with_healthy, timepoint='0 with Healthys', pairs=[(1,2), (1,3), (2,3)])

    # print mean ratios for each cluster inc healthys at timepoint 0
    mean_ratios = aa_0_with_healthy.groupby('cluster_id')['ess_noness_ratio'].mean()
    print("Mean Essential to Non-Essential AA Ratios at Timepoint 0 (including Healthys):")
    print(mean_ratios)

    # Additionally, plot ratio for just some non-essential AAs
    non_essential_aas = ['alanine', 'serine', 'ornithine']

    # get the ratio of essential to non-essential aas for both timepoints
    aa_0['ess_noness_ratio'] = aa_0[essential_aas].sum(axis=1) / aa_0[non_essential_aas].sum(axis=1)
    aa_52['ess_noness_ratio'] = aa_52[essential_aas].sum(axis=1) / aa_52[non_essential_aas].sum(axis=1)

    plot_violins(aa_0, timepoint='0 (non-essential subset)')
    plot_violins(aa_52, timepoint='52 (non-essential subset)')

def plot_aa_violins_by_cluster(aa_file_path: str = '../data/aa.tsv', outcomes_dir: str = '../Outcomes'):
    """
    Plots violin plots of some AA levels (Valine, leucine and isoleucine)
    for individuals in each cluster (from pop cluster fig), separately for timepoints 0 and 52.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    # get cluster assignments
    assign_path = '../Outcomes/subject_cluster_assignments_small.tsv'
    cl_df = pd.read_csv(assign_path, sep='\t')
    cl_df = cl_df.set_index(['subjectID'])

    aa = pd.read_csv(aa_file_path, sep='\t')
    aa = aa.set_index(aa.columns[0])

    # These are ones we are interested in as diff between clusters
    desired_aas = ['valine', 'leucine', 'isoleucine', 'ornithine']
    aa = aa[desired_aas]

    aa_0 = aa[aa.index.str.endswith('_0')].copy()
    aa_52 = aa[aa.index.str.endswith('_52')].copy()

    # add cluster number as col to aa_0
    aa_0['subjectID'] = aa_0.index.str.replace('_0', '', regex=False)
    aa_0 = aa_0.join(cl_df, on='subjectID', how='inner')
    aa_0 = aa_0.drop(columns=['subjectID'])

    # add cluster number as col to aa_52
    aa_52['subjectID'] = aa_52.index.str.replace('_52', '', regex=False)
    aa_52 = aa_52.join(cl_df, on='subjectID', how='inner')
    aa_52 = aa_52.drop(columns=['subjectID'])

    # Plotting function. Also calcs MWU between all pairs
    def plot_violins(data: pd.DataFrame, timepoint: int, pairs: list = [(1, 2)]):
        fig, axes = plt.subplots(1, 4, figsize=(30, 15))
        axes = axes.flatten()
        clusters = sorted(cl_df['cluster_id'].unique())

        for i, aa_name in enumerate(aa.columns):
            ax = axes[i]
            sns.boxplot(
                x='cluster_id',
                y=aa_name,
                data=data,
                palette='Set2',
                ax=ax,
            )

            annot = Annotator(ax, pairs, data=data, x='cluster_id', y=aa_name)
            annot.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=0, comparisons_correction='Benjamini-Hochberg')
            annot.apply_and_annotate()

            ax.set_title(f'{aa_name.replace("_", " ").title()} Levels by Cluster (Timepoint {timepoint})', fontsize=14)
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel(f'{aa_name.replace("_", " ").title()} Level', fontsize=12)

        plt.tight_layout()
        out_path = os.path.join(outcomes_dir, f'aa_box_timepoint_{timepoint}.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
    
    # Plot for timepoint 0
    plot_violins(aa_0, timepoint=0)
    # add in healthys to timepoint 0
    meta = pd.read_csv('../data/meta.tsv', sep='\t')
    healthy_ids = meta[meta['Condition'] == 'Well-nourished']['subjectID'].tolist()
    aa_healthy = aa.loc[[f'{sid}_0' for sid in healthy_ids if f'{sid}_0' in aa.index]].copy()
    aa_healthy['cluster_id'] = 3
    aa_0_with_healthy = pd.concat([aa_0, aa_healthy], axis=0)
    plot_violins(aa_0_with_healthy, timepoint='0 with Healthys', pairs=[(1,2), (1,3), (2,3)])
    # Plot for timepoint 52
    plot_violins(aa_52, timepoint=52)
    # also plot the difference between the two (52 - 0). Keep cluster_id as is
    aa_0_aligned = aa_0.copy()
    aa_52_aligned = aa_52.copy()
    aa_0_aligned.index = aa_0_aligned.index.str.replace('_0', '', regex=False)
    aa_52_aligned.index = aa_52_aligned.index.str.replace('_52', '', regex=False)
    # get log2 difference of all columns except cluster_id
    aa_diff = np.log2( aa_52_aligned.drop(columns=['cluster_id']) / aa_0_aligned.drop(columns=['cluster_id']) )
    aa_diff['cluster_id'] = aa_0_aligned['cluster_id']
    plot_violins(aa_diff, timepoint='Difference (52-0)')

if __name__ == "__main__":
    intervention_data = pd.read_csv('../Outcomes/population_wlz_deltas_neg1_small.tsv', sep='\t')
    ##remove x0 and x_cf columns
    #intervention_data = intervention_data[['subjectID', 'feature', 'delta']]
    #create_population_fig(intervention_data)
    # create_radar_plot(intervention_data, top_n_features=10)
    #create_full_clustermap(intervention_data, top_n_features=10)
    # create_individual_dumbbell_plot(subject_id='LCC1010')
    #create_full_clustermap_clusters(intervention_data, top_n_features=10, n_clusters=2)

    ## Below fn has to be run after cluster_intervention.py, other ones above can be either, either, or :)
    plot_aa_violins_ratio(
        aa_file_path='../data/aa.tsv',
        outcomes_dir='../Outcomes'
    )