import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# --- GLOBAL FONT AND PLOT SETTINGS (IEEE Compliance) ---
# Force matplotlib to use Type 42 (TrueType) fonts for IEEE compliance
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'legend.title_fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# --- LABEL MAPPINGS ---
LABEL_MAPPING = {
    'standard_kl': 'Forward KL (Numerical Approximator)',
    'reverse_kl': 'Reverse KL (Numerical Approximator)',
    'arithmetic_mean': 'Arithmetic Mean (Exact Analytical)',
    'geometric_mean': 'Geometric Mean (Exact Analytical)',
    'weighted_visits_kl': 'Visit-Weighted KL (Proposed)'
}

COLORS = {
    'Forward KL (Numerical Approximator)': '#3498db', # Blue
    'Reverse KL (Numerical Approximator)': '#e74c3c', # Red
    'Arithmetic Mean (Exact Analytical)': '#95a5a6',  # Grey
    'Geometric Mean (Exact Analytical)': '#2ecc71',   # Green
    'Visit-Weighted KL (Proposed)': '#9b59b6'         # Purple
}

CACHE_FILE = "results/data_cache.pkl"

def load_or_cache_data(checkpoints_dir="checkpoints"):
    """Loads parsed simulation data from cache if available, otherwise parses all pickle files."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}... (Fast!)")
        print("Note: If you ran new simulations, delete this file to force a refresh.")
        return pd.read_pickle(CACHE_FILE)

    print(f"No cache found. Scanning {checkpoints_dir} for trial results... (This might take a minute)")
    
    results = []
    checkpoint_path = Path(checkpoints_dir)
    
    if not checkpoint_path.exists():
        print(f"Error: Directory {checkpoints_dir} not found.")
        return pd.DataFrame()

    files = list(checkpoint_path.glob("*.pkl"))
    total_files = len(files)
    
    if total_files == 0:
        print("No .pkl files found in checkpoints directory.")
        return pd.DataFrame()

    print(f"Found {total_files} files. Processing...")

    for i, file_path in enumerate(files):
        if "consolidated" in file_path.name:
            continue
            
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
                # Initialize variables
                grid_size_str = "unknown"
                grid_area = 0
                n_agents = 0
                method = "unknown"
                pattern = "unknown"
                interval = "unknown"

                # Prefer task_metadata dictionary if it exists
                meta = data.get('task_metadata', {})
                if meta:
                    gs = meta.get('grid_size', (0,0))
                    grid_size_str = f"{gs[0]}x{gs[1]}"
                    grid_area = gs[0] * gs[1]
                    n_agents = meta.get('n_agents', 0)
                    pattern = meta.get('pattern', 'unknown')
                    interval = meta.get('merge_interval', 'unknown')
                    method = meta.get('merge_method', 'unknown')
                else:
                    # Fallback to parsing filename parts
                    parts = file_path.stem.split('_')
                    for part in parts:
                        if part.startswith("grid"):
                            grid_size_str = part.replace("grid", "")
                            try:
                                r, c = map(int, grid_size_str.split('x'))
                                grid_area = r * c
                            except:
                                grid_area = 0
                        elif part.startswith("agents"):
                            try:
                                n_agents = int(part.replace("agents", ""))
                            except:
                                n_agents = 0
                        elif part in ['standard', 'reverse', 'geometric', 'arithmetic', 'weighted']:
                            if part == 'standard': method = 'standard_kl'
                            elif part == 'reverse': method = 'reverse_kl'
                            elif part == 'geometric': method = 'geometric_mean'
                            elif part == 'arithmetic': method = 'arithmetic_mean'
                            elif part == 'weighted': method = 'weighted_visits_kl'
                        elif part.startswith("interval"):
                            interval = part.replace("interval", "")
                        elif part in ['stationary', 'random', 'evasive', 'patrol']:
                            pattern = part

                target_found = data.get('target_found', False)
                first_discovery_step = data.get('first_discovery_step', np.nan)
                
                results.append({
                    'grid_str': grid_size_str,
                    'grid_area': grid_area,
                    'n_agents': n_agents,
                    'pattern': pattern,
                    'interval': str(interval),
                    'method': method,
                    'found': 1 if target_found else 0,
                    'steps_to_find': first_discovery_step if target_found else 2500 # Penalty cap
                })
                
        except Exception as e:
            continue

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{total_files} files...")

    if not results:
        print("No valid trial data extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    
    # Whitelist Filter and Clean Labels
    known_methods = list(LABEL_MAPPING.keys())
    df = df[df['method'].isin(known_methods)]
    df['method'] = df['method'].map(LABEL_MAPPING)
    df = df.sort_values('grid_area')
    
    # Save to cache
    print("Parsing complete! Saving dataframe to cache for next time...")
    df.to_pickle(CACHE_FILE)
    
    return df

def generate_configuration_table(df):
    """Generates a breakdown table grouping by all distinct configurations."""
    print("Generating comprehensive configuration tables...")
    
    summary_all = df.groupby(['grid_str', 'n_agents', 'pattern', 'interval', 'method']).agg(
        total_trials=('found', 'count'),
        success_rate=('found', 'mean'),
        avg_steps_all=('steps_to_find', 'mean')
    ).reset_index()
    
    success_df = df[df['found'] == 1]
    summary_success = success_df.groupby(['grid_str', 'n_agents', 'pattern', 'interval', 'method']).agg(
        avg_steps_successful=('steps_to_find', 'mean')
    ).reset_index()
    
    final_table = pd.merge(summary_all, summary_success, on=['grid_str', 'n_agents', 'pattern', 'interval', 'method'], how='left')
    
    final_table['success_rate'] = (final_table['success_rate'] * 100).round(2).astype(str) + '%'
    final_table['avg_steps_all'] = final_table['avg_steps_all'].round(1)
    final_table['avg_steps_successful'] = final_table['avg_steps_successful'].round(1)
    final_table['avg_steps_successful'] = final_table['avg_steps_successful'].fillna("N/A (0% Success)")
    
    final_table = final_table.sort_values(['grid_str', 'n_agents', 'pattern', 'interval', 'method'])
    final_table.to_csv('results/comprehensive_configuration_table.csv', index=False)
    
    try:
        pivot_table = final_table.pivot_table(
            index=['grid_str', 'n_agents', 'pattern', 'interval'],
            columns='method',
            values=['success_rate', 'avg_steps_successful', 'avg_steps_all'],
            aggfunc='first'
        )
        pivot_table.to_csv('results/comprehensive_configuration_table_pivoted.csv')
    except Exception as e:
        print(f"Could not generate pivoted table: {e}")

def plot_grid_size_2d(df, colors):
    df_plot = df.copy() 
    
    # Extract dimension purely to guarantee we sort them in the correct ascending order
    df_plot['grid_dimension'] = df_plot['grid_str'].apply(lambda x: int(x.split('x')[0]))

    # Create the mapping for PERFECTLY EQUAL spacing (0, 1, 2, 3...)
    tick_mapping = df_plot[['grid_dimension', 'grid_str']].drop_duplicates().sort_values('grid_dimension')
    tick_mapping['x_index'] = range(len(tick_mapping)) # This is the magic for equal spacing
    
    # Merge this equidistant index back into the main dataframe
    df_plot = df_plot.merge(tick_mapping[['grid_str', 'x_index']], on='grid_str')

    x_ticks = tick_mapping['x_index'].values
    x_labels = tick_mapping['grid_str'].values

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # Aggregate data using the new equally-spaced index
    agg_data = df_plot.groupby(['x_index', 'method']).agg({
        'found': 'mean',
        'steps_to_find': 'mean'
    }).reset_index()
    
    # --- Subplot 1: Success Rate ---
    sns.lineplot(data=agg_data, x='x_index', y='found', hue='method', 
                 palette=colors, marker='o', linewidth=2.5, ax=axes[0])
    axes[0].set_title('Success Rate vs Grid Size')
    axes[0].set_xlabel('Grid Size')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_ylim(-0.05, 1.05)
    
    # Apply the string labels to the equidistant ticks
    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
    axes[0].get_legend().remove() 
    
    # --- Subplot 2: Steps to Discovery ---
    sns.lineplot(data=agg_data, x='x_index', y='steps_to_find', hue='method', 
                 palette=colors, marker='o', linewidth=2.5, ax=axes[1])
    axes[1].set_title('Avg Steps to Discovery vs Grid Size (Includes Failed)')
    axes[1].set_xlabel('Grid Size')
    axes[1].set_ylabel('Steps')
    
    # Apply the string labels here as well
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
    
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Merge Method')
    
    plt.tight_layout()
    plt.savefig('results/grid_size_effect_2d.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def plot_3d_scatter(df, colors):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    jitter_strength = 0.05
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        jittered_z = method_data['found'] + np.random.uniform(-jitter_strength, jitter_strength, len(method_data))
        
        ax.scatter(
            method_data['grid_area'],
            method_data['steps_to_find'],
            jittered_z,
            c=colors.get(method, 'gray'),
            label=method,
            alpha=0.6,
            edgecolors='w',
            s=50
        )

    ax.set_xlabel('Grid Size (Area)', labelpad=15)
    ax.set_ylabel('Steps to Discovery', labelpad=15)
    ax.set_zlabel('Success (0=Fail, 1=Success)', labelpad=15)
    ax.set_title('3D Analysis: Grid Size vs Steps vs Success')
    
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left', title='Merge Method')
    
    plt.tight_layout()
    plt.savefig('results/grid_size_effect_3d_static.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def analyze_target_finding(checkpoints_dir="checkpoints"):
    # Now uses the caching system
    df = load_or_cache_data(checkpoints_dir)
    
    if df.empty:
        return

    print(f"Analyzed {len(df)} valid trials.")
    
    # --- Analysis & Visualization ---
    sns.set_style("whitegrid")
    
    method_order = [
        'Forward KL (Numerical Approximator)',
        'Arithmetic Mean (Exact Analytical)',
        'Reverse KL (Numerical Approximator)',
        'Geometric Mean (Exact Analytical)',
        'Visit-Weighted KL (Proposed)'
    ]
    
    # 1. Success Rate by Method
    print("Generating Target Success Rate Bar Plot...")
    plt.figure(figsize=(12, 7))
    success_rates = df.groupby('method')['found'].mean().reset_index()
    sns.barplot(data=success_rates, x='method', y='found', hue='method', order=method_order, palette=COLORS, legend=False)
    plt.title('Target Discovery Success Rate by Method')
    plt.ylabel('Success Rate')
    plt.xlabel('Belief Merging Method')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig('results/target_success_rate.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    # 2. Steps Distribution (Successful Trials Only)
    print("Generating Steps Distribution Box Plot...")
    plt.figure(figsize=(12, 7))
    successful_trials = df[df['found'] == 1]
    if not successful_trials.empty:
        sns.boxplot(data=successful_trials, x='method', y='steps_to_find', hue='method', order=method_order, palette=COLORS, legend=False, showfliers=False)
        plt.title('Distribution of Steps to Find Target (Successful Trials)')
        plt.ylabel('Steps')
        plt.xlabel('Belief Merging Method')
        plt.xticks(rotation=25, ha='right')
        plt.tight_layout()
        plt.savefig('results/steps_to_find_distribution.pdf', format='pdf', bbox_inches='tight')
        plt.close()

    # --- PLOT 1: 2D Grid Size Analysis ---
    print("Generating 2D Grid Size Analysis...")
    plot_grid_size_2d(df, COLORS)

    # --- PLOT 2: Interactive 3D Scatter Analysis ---
    print("Generating 3D Grid Analysis...")
    plot_3d_scatter(df, COLORS)

    # --- TABLE: Comprehensive Configuration Breakdown ---
    generate_configuration_table(df)

    # Global Summary Stats
    print("Generating Global Summary CSV...")
    summary = df.groupby('method').agg({
        'found': ['count', 'sum', 'mean'],
        'steps_to_find': ['mean', 'median', 'std']
    })
    summary.to_csv('results/target_finding_summary.csv')
    print("\nAnalysis complete. Visualizations and tables saved to 'results/' directory.")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    analyze_target_finding()