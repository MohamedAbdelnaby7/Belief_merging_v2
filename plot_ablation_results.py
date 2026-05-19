import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

def parse_pkls_to_csv(ckpt_dir, output_csv):
    """Reads raw .pkl files, filters methods, aggregates data, and saves to CSV."""
    print(f"  -> No CSV found for {ckpt_dir.name}. Parsing .pkl files...")
    results = []
    
    # We only care about these two methods for this analysis
    target_methods = ['arithmetic_mean', 'weighted_visits_kl']
    
    for file_path in ckpt_dir.glob("*.pkl"):
        if "consolidated" in file_path.name or "ERROR" in file_path.name:
            continue
            
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            meta = data.get('task_metadata', {})
            if not meta:
                continue # Skip if metadata is missing
                
            method = meta.get('merge_method', 'unknown')
            if method not in target_methods:
                continue # Drop methods we aren't analyzing
                
            gs = meta.get('grid_size', (0,0))
            
            target_found = data.get('target_found', False)
            first_discovery_step = data.get('first_discovery_step', 2500)
            if not target_found: 
                first_discovery_step = 2500 # Apply penalty cap for failed trials
                
            results.append({
                'grid_str': f"{gs[0]}x{gs[1]}",
                'n_agents': meta.get('n_agents', 0),
                'pattern': meta.get('pattern', 'unknown'),
                'interval': str(meta.get('merge_interval', 'unknown')),
                'method': method,
                'found': 1 if target_found else 0,
                'steps_to_find': first_discovery_step
            })
        except Exception as e:
            continue # Silently skip corrupted files
            
    if not results:
        print(f"  -> WARNING: No valid trials found for target methods in {ckpt_dir}")
        return False

    # Aggregate into the "Comprehensive Table" format
    df_raw = pd.DataFrame(results)
    
    summary_all = df_raw.groupby(['grid_str', 'n_agents', 'pattern', 'interval', 'method']).agg(
        total_trials=('found', 'count'),
        success_rate=('found', 'mean'),
        avg_steps_all=('steps_to_find', 'mean')
    ).reset_index()
    
    # Save the CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_all.to_csv(output_csv, index=False)
    print(f"  -> Aggregation complete. Saved to {output_csv}")
    return True

def load_ablation_data():
    """Scans the directories, builds CSVs if missing, and loads the data."""
    ckpt_base = Path("checkpoints_ablation")
    res_base = Path("results_ablation")
    
    if not ckpt_base.exists():
        print(f"Error: Directory {ckpt_base} not found.")
        return {}

    # Sort alphabetically to keep profiles in logical order (e.g., Profile 1, 2, 3...)
    profiles = sorted([d for d in ckpt_base.iterdir() if d.is_dir()])
    data_dict = {}

    print(f"Found {len(profiles)} noise profiles. Loading data...")

    for p_dir in profiles:
        profile_name = p_dir.name
        res_dir = res_base / profile_name
        csv_path = res_dir / "comprehensive_configuration_table.csv"
        
        # Build the CSV if it doesn't exist yet
        if not csv_path.exists():
            success = parse_pkls_to_csv(p_dir, csv_path)
            if not success:
                continue

        df = pd.read_csv(csv_path)

        # 1. Clean extreme intervals
        drop_vals = ['0', '0.0', '000 (Full Comm)', 'inf', 'inf (No Comm)']
        df['interval_str'] = df['interval'].astype(str).str.replace('.0', '', regex=False)
        df = df[~df['interval_str'].isin(drop_vals)].copy()

        # 2. Clean success rate (convert to 0.0 - 1.0 float)
        if df['success_rate'].dtype == object:
            df['success_rate'] = df['success_rate'].str.rstrip('%').astype(float) / 100.0
        elif df['success_rate'].max() > 1.0:
            df['success_rate'] = df['success_rate'] / 100.0

        # 3. Calculate mapping variables
        def get_area(g_str):
            try:
                r, c = map(int, g_str.split('x'))
                return r * c
            except:
                return 0
        df['grid_area'] = df['grid_str'].apply(get_area)
        df['interval_num'] = df['interval_str'].astype(int)

        # 4. Calculate Winners
        scenario_cols = ['grid_str', 'n_agents', 'pattern', 'interval']
        max_succ = df.groupby(scenario_cols)['success_rate'].transform('max')
        df['is_success_winner'] = np.isclose(df['success_rate'], max_succ, atol=1e-4).astype(int)

        min_steps = df.groupby(scenario_cols)['avg_steps_all'].transform('min')
        df['is_steps_winner'] = np.isclose(df['avg_steps_all'], min_steps, atol=1e-1).astype(int)

        data_dict[profile_name] = df
        print(f"  - Loaded {profile_name} ({len(df)} specific scenario evaluations)")

    return data_dict

def generate_ablation_plots(data_dict):
    if not data_dict:
        return

    profiles = list(data_dict.keys())
    num_profiles = len(profiles)
    
    # Target methods and their colors
    methods_order = ['arithmetic_mean', 'weighted_visits_kl']
    colors = {
        'arithmetic_mean': '#95a5a6',     # Gray baseline
        'weighted_visits_kl': '#9b59b6'   # Purple proposed method
    }

    # Attributes to plot: (DataFrame Column, Display Name, Plot Type for metrics)
    attributes = [
        ('n_agents', 'Number of Agents', 'line'),
        ('interval_num', 'Communication Interval', 'line'),
        ('pattern', 'Target Pattern', 'bar'),
        ('grid_area', 'Grid Size (Area)', 'line')
    ]

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    # ==========================================
    # PLOT 1: WINS / TIES COUNTS (HISTOGRAM/BARS)
    # ==========================================
    print("\nGenerating Plot 1: Win/Tie Counts across Profiles...")
    fig1, axes1 = plt.subplots(4, num_profiles * 2, figsize=(6 * num_profiles, 20))
    
    for row_idx, (attr, attr_name, _) in enumerate(attributes):
        for prof_idx, prof_name in enumerate(profiles):
            df = data_dict[prof_name]
            plot_cols = [m for m in methods_order if m in df['method'].unique()]
            
            col_succ = prof_idx * 2
            col_step = prof_idx * 2 + 1
            
            ax_succ = axes1[row_idx, col_succ]
            ax_step = axes1[row_idx, col_step]

            # Aggregate wins
            agg_succ = df.groupby([attr, 'method'])['is_success_winner'].sum().unstack(fill_value=0)
            agg_step = df.groupby([attr, 'method'])['is_steps_winner'].sum().unstack(fill_value=0)
            
            # Filter and order columns safely
            agg_succ = agg_succ[[m for m in plot_cols if m in agg_succ.columns]]
            agg_step = agg_step[[m for m in plot_cols if m in agg_step.columns]]

            # Plot Success Wins
            agg_succ.plot(kind='bar', ax=ax_succ, color=[colors.get(m, '#000') for m in agg_succ.columns], edgecolor='black', width=0.8)
            # Plot Step Wins
            agg_step.plot(kind='bar', ax=ax_step, color=[colors.get(m, '#000') for m in agg_step.columns], edgecolor='black', width=0.8)

            # Formatting
            for ax in [ax_succ, ax_step]:
                ax.tick_params(axis='x', rotation=0 if attr != 'pattern' else 15)
                ax.set_xlabel(attr_name if row_idx == 3 else "")
                if ax.get_legend(): ax.get_legend().remove()

            ax_succ.set_ylabel("Success Wins" if prof_idx == 0 else "")
            ax_step.set_ylabel("Efficiency Wins" if prof_idx == 0 else "")

            # Set top titles
            if row_idx == 0:
                ax_succ.set_title(f"[{prof_name}]\nSuccess Rate Wins", fontsize=14, pad=10)
                ax_step.set_title(f"[{prof_name}]\nEfficiency (Steps) Wins", fontsize=14, pad=10)

    # Master Legend
    handles, labels = axes1[0, 0].get_legend_handles_labels()
    fig1.legend(handles, labels, title='Merge Method', loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=14)
    fig1.suptitle("Ablation Study: Number of Scenarios Won Across Sensor Noise Profiles", fontsize=22, y=1.05)
    
    plt.tight_layout()
    out1 = 'results_ablation/ablation_wins_grid.png'
    fig1.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {out1}")


    # ==========================================
    # PLOT 2: ACTUAL METRICS (SUCCESS % & STEPS)
    # ==========================================
    print("Generating Plot 2: Actual Metrics across Profiles...")
    fig2, axes2 = plt.subplots(4, num_profiles * 2, figsize=(6 * num_profiles, 20))

    for row_idx, (attr, attr_name, plot_type) in enumerate(attributes):
        for prof_idx, prof_name in enumerate(profiles):
            df = data_dict[prof_name]
            plot_cols = [m for m in methods_order if m in df['method'].unique()]
            
            col_succ = prof_idx * 2
            col_step = prof_idx * 2 + 1
            
            ax_succ = axes2[row_idx, col_succ]
            ax_step = axes2[row_idx, col_step]

            # Aggregate actual metrics
            agg = df.groupby([attr, 'method']).agg({'success_rate': 'mean', 'avg_steps_all': 'mean'}).reset_index()

            # Plot Success Metrics
            if plot_type == 'line':
                sns.lineplot(data=agg, x=attr, y='success_rate', hue='method', hue_order=plot_cols, palette=colors, marker='o', linewidth=3, ax=ax_succ)
                sns.lineplot(data=agg, x=attr, y='avg_steps_all', hue='method', hue_order=plot_cols, palette=colors, marker='o', linewidth=3, ax=ax_step)
                if attr == 'interval_num':
                    ax_succ.set_xticks(sorted(df['interval_num'].unique()))
                    ax_step.set_xticks(sorted(df['interval_num'].unique()))
            else:
                sns.barplot(data=agg, x=attr, y='success_rate', hue='method', hue_order=plot_cols, palette=colors, edgecolor='black', ax=ax_succ)
                sns.barplot(data=agg, x=attr, y='avg_steps_all', hue='method', hue_order=plot_cols, palette=colors, edgecolor='black', ax=ax_step)

            # Formatting
            ax_succ.set_ylim(-0.05, 1.05)
            
            for ax in [ax_succ, ax_step]:
                ax.tick_params(axis='x', rotation=0 if attr != 'pattern' else 15)
                ax.set_xlabel(attr_name if row_idx == 3 else "")
                if ax.get_legend(): ax.get_legend().remove()

            ax_succ.set_ylabel("Avg Success Rate" if prof_idx == 0 else "")
            ax_step.set_ylabel("Avg Steps" if prof_idx == 0 else "")

            if row_idx == 0:
                ax_succ.set_title(f"[{prof_name}]\nActual Success Rate", fontsize=14, pad=10)
                ax_step.set_title(f"[{prof_name}]\nActual Discovery Steps", fontsize=14, pad=10)

    # Master Legend
    handles, labels = axes2[0, 0].get_legend_handles_labels()
    fig2.legend(handles, labels, title='Merge Method', loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=14)
    fig2.suptitle("Ablation Study: True Performance Metrics Across Sensor Noise Profiles", fontsize=22, y=1.05)
    
    plt.tight_layout()
    out2 = 'results_ablation/ablation_metrics_grid.png'
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {out2}")

if __name__ == "__main__":
    data = load_ablation_data()
    generate_ablation_plots(data)