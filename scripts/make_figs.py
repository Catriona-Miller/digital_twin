import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from what_if_counterfactuals import run_single_subject_cf

def create_population_fig(intervention_data):
    '''
    FINISH
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

    # Create a 'direction' column for the hue param (needed now if give palette?)
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

    ax.set_title('Population Average and Individual Personalisation', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Recommended Change (Δ) in Scaled Units', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.axvline(0, color='black', linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Dominant Direction')

    plt.tight_layout()
    fig.savefig('Outcomes/figure_combined.png', dpi=300, bbox_inches='tight')

def create_individual_heatmap(intervention_data: pd.DataFrame, top_n_features: int = 20, n_individuals: int = 15):
    """
    Generates a clustered heatmap showing personalized intervention "fingerprints"
    for a selection of individuals.
    """

    mean_abs_delta = intervention_data.groupby('feature')['delta'].apply(lambda x: x.abs().mean())
    top_features = mean_abs_delta.nlargest(top_n_features).index.tolist()

    # 2. Select a representative sample of individuals
    subject_ids = intervention_data['subjectID'].unique()
    np.random.seed(10)
    selected_subjects = np.random.choice(subject_ids, size=n_individuals, replace=False)

    heatmap_data = intervention_data[
        (intervention_data['subjectID'].isin(selected_subjects)) &
        (intervention_data['feature'].isin(top_features))
    ]

    heatmap_matrix = heatmap_data.pivot(
        index='subjectID',
        columns='feature',
        values='delta'
    )
    heatmap_matrix = heatmap_matrix[top_features]
    heatmap_matrix = heatmap_matrix.fillna(0) 

    plt.style.use('seaborn-v0_8-whitegrid')
    
    g = sns.clustermap(
        heatmap_matrix,
        method='ward',   
        cmap='vlag',   
        center=0,          
        figsize=(15, 10),
        linewidths=.75,
        cbar_kws={'label': 'Recommended Change (Δ)'}
    )

    g.ax_heatmap.set_xlabel('Top 10 Features', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('Individuals', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_title('Personalised Intervention Requirements', fontsize=16, fontweight='bold', pad=20)
    
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.savefig('Outcomes/individuals_heatmap.png', dpi=300, bbox_inches='tight')

def create_radar_plot(intervention_data: pd.DataFrame, top_n_features: int = 8):
    """
    Generates a radar plot comparing the intervention strategies for two individuals.
    """
    mean_abs_delta = intervention_data.groupby('feature')['delta'].apply(lambda x: x.abs().mean())
    top_features = mean_abs_delta.nlargest(top_n_features).index.tolist()

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

    def add_to_radar(subject_id, color):
        values = radar_df.loc[subject_id].tolist()
        values += values[:1] 
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=subject_id)
        ax.fill(angles, values, color=color, alpha=0.25)

    colours = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, subject_id in enumerate(selected_subjects):
        add_to_radar(subject_id, color=colours[i])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    ax.set_rlabel_position(180 / num_vars)
    circle_angles = np.linspace(0, 2 * np.pi, 100)
    # Draw the circle at radius 0
    ax.plot(circle_angles, [0] * 100, color='black', linewidth=1)   
    plt.title('Personalized Intervention Radar for Three Individuals', size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.savefig('Outcomes/figure_radar_plot_three.png', dpi=300, bbox_inches='tight')

def create_full_clustermap(intervention_data: pd.DataFrame, top_n_features: int = 10):
    """
    Generates a clustered heatmap for ALL individuals in the dataset,
    revealing population-level patterns and clusters.
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
        figsize=(15, 12),
        yticklabels=False,
        cbar_kws={'label': 'Recommended Change (Δ)'}
    )

    g.ax_heatmap.set_title('Population-wide Intervention Patterns', fontsize=16, fontweight='bold', pad=20)
    g.ax_heatmap.set_xlabel('Top 10 Features', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('') 
    
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    g.savefig('Outcomes/figure_full_clustermap.png', dpi=300, bbox_inches='tight')

def create_individual_dumbbell_plot(subject_id: str,
                                    top_n: int = 10):
    """
    Creates a dumbbell plot to visualize the before-and-after changes
    for a single individual's counterfactual analysis.
    """
    cf_results = run_single_subject_cf(subject_id, target_wlz_gain=1.0)
    
    # Extract the data for plotting
    changes_df = cf_results['changes_df']
    wlz_base = cf_results['wlz_base']
    wlz_cf = cf_results['wlz_cf']

    # 3. Prepare the data for plotting
    plot_data = changes_df.head(top_n).sort_values('delta')
    
    # 4. Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.hlines(y=plot_data.index, xmin=plot_data['x0'], xmax=plot_data['x_cf'],
              color='gray', alpha=0.7, linewidth=2, zorder=1)
    ax.scatter(plot_data['x0'], plot_data.index, color='#1f77b4', s=80, label='Original (x0)', zorder=2)
    ax.scatter(plot_data['x_cf'], plot_data.index, color='#ff7f0e', s=80, label='Counterfactual (x_cf)', zorder=2)

    # 5. Aesthetics and Labels
    ax.set_xlabel('Scaled Value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    wlz_delta = wlz_cf - wlz_base
    title = (
        f'Counterfactual Changes for {subject_id}\n'
        f'WLZ Score: {wlz_base:.2f} → {wlz_cf:.2f} (Δ {wlz_delta:+.2f})'
    )
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    
    # 6. Save the figure
    fig.savefig(f'Outcomes/dumbbell_plot_{subject_id}.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    intervention_data = pd.read_csv('../Outcomes/population_wlz_deltas.tsv', sep='\t')
    create_population_fig(intervention_data)
    create_individual_heatmap(intervention_data, top_n_features=10)
    create_radar_plot(intervention_data, top_n_features=10)
    create_full_clustermap(intervention_data, top_n_features=10)
    create_individual_dumbbell_plot(subject_id='LCC1010')