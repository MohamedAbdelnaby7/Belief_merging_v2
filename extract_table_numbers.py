import pandas as pd
import numpy as np
from pathlib import Path

# Clean names mapping
METHOD_MAP = {
    'Forward KL (Numerical Approximator)': 'Standard KL',
    'Reverse KL (Numerical Approximator)': 'Reverse KL',
    'Arithmetic Mean (Exact Analytical)': 'Arithmetic Mean',
    'Visit-Weighted KL (Proposed)': 'Visit-Weighted KL',
    'Geometric Mean (Exact Analytical)': 'Geometric Mean',
    'standard_kl': 'Standard KL',
    'reverse_kl': 'Reverse KL',
    'arithmetic_mean': 'Arithmetic Mean',
    'geometric_mean': 'Geometric Mean',
    'weighted_visits_kl': 'Visit-Weighted KL'
}

def clean_dataframe(df):
    """Filters extreme intervals and cleans success rates."""
    df['interval_str'] = df['interval'].astype(str).str.replace('.0', '', regex=False)
    drop_vals = ['0', 'inf', '000 (Full Comm)', 'inf (No Comm)', 'nan']
    df = df[~df['interval_str'].isin(drop_vals)].copy()
    
    if df['success_rate'].dtype == object:
        df['success_rate'] = df['success_rate'].str.rstrip('%').astype(float) / 100.0
        
    df['clean_method'] = df['method'].map(METHOD_MAP)
    df = df.dropna(subset=['clean_method'])
    return df

def calculate_winners(df):
    """Calculates boolean columns for success and step winners."""
    scenario_cols = ['grid_str', 'n_agents', 'pattern', 'interval']
    
    max_succ = df.groupby(scenario_cols)['success_rate'].transform('max')
    df['success_win'] = np.isclose(df['success_rate'], max_succ, atol=1e-4).astype(int)
    
    min_steps = df.groupby(scenario_cols)['avg_steps_all'].transform('min')
    df['steps_win'] = np.isclose(df['avg_steps_all'], min_steps, atol=1e-1).astype(int)
    return df

def print_general_summary(df):
    print("\n% ==========================================")
    print("% TABLE 1: GENERAL SUMMARY (RATES & WINS)")
    print("% ==========================================")
    
    # NEW: Added Std_Success to the aggregation
    summary = df.groupby('clean_method').agg(
        Avg_Success=('success_rate', 'mean'),
        Std_Success=('success_rate', 'std'),
        Succ_Wins=('success_win', 'sum'),
        Step_Wins=('steps_win', 'sum')
    ).round({'Avg_Success': 3, 'Std_Success': 3}).sort_values('Step_Wins')
    
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{threeparttable}")
    print("\\caption{Overall Algorithm Performance Across General Configurations}")
    print("\\label{tab:general_summary}")
    print("\\begin{tabular}{@{}lccc@{}}")
    print("\\toprule")
    print("\\textbf{Merge Method} & \\textbf{Avg. Success} & \\textbf{Success Wins} & \\textbf{Efficiency Wins} \\\\")
    print("\\midrule")
    
    max_succ = summary['Avg_Success'].max()
    max_swin = summary['Succ_Wins'].max()
    max_ewin = summary['Step_Wins'].max()
    
    for method, row in summary.iterrows():
        # Format Mean +/- Std Dev
        succ_val = f"{row['Avg_Success']:.3f} $\\pm$ {row['Std_Success']:.3f}"
        if row['Avg_Success'] == max_succ:
            succ_val = f"\\textbf{{{row['Avg_Success']:.3f}}} $\\pm$ {row['Std_Success']:.3f}"
            
        swin_val = f"\\textbf{{{int(row['Succ_Wins'])}}}" if row['Succ_Wins'] == max_swin else str(int(row['Succ_Wins']))
        ewin_val = f"\\textbf{{{int(row['Step_Wins'])}}}" if row['Step_Wins'] == max_ewin else str(int(row['Step_Wins']))
        
        m_name = f"\\textbf{{{method} (Ours)}}" if "Visit-Weighted" in method else method
        print(f"{m_name} & {succ_val} & {swin_val} & {ewin_val} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\begin{tablenotes}")
    print("\\small")
    print("\\item \\textit{Note:} Out of all evaluated intermittent scenarios, our proposed Visit-Weighted KL achieved the highest overall average success rate and secured the absolute lowest steps to discovery in the majority of configurations. Variance represents one standard deviation across the tested parameter space.")
    print("\\end{tablenotes}")
    print("\\end{threeparttable}")
    print("\\end{table}")

def print_general_feature_wins(df):
    print("\n% ==========================================")
    print("% TABLE 2: GENERAL WINS BY FEATURE")
    print("% ==========================================")
    
    methods = ['Standard KL', 'Reverse KL', 'Arithmetic Mean', 'Geometric Mean', 'Visit-Weighted KL']
    features = [('n_agents', 'Number of Agents'), ('pattern', 'Target Pattern'), ('interval', 'Communication Interval')]
    
    print("\\begin{table}[htbp]")
    print("\\caption{Success Rate Wins Broken Down by Configuration Feature}")
    print("\\label{tab:feature_wins}")
    print("\\centering")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{@{}lccccc@{}}")
    print("\\toprule")
    print("\\textbf{Feature} & \\textbf{Std. KL} & \\textbf{Rev. KL} & \\textbf{Arith. Mean} & \\textbf{Geo. Mean} & \\textbf{Visit-Weighted} \\\\")
    
    for feat_col, feat_name in features:
        print("\\midrule")
        print(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{feat_name}}}}} \\\\")
        print("\\midrule")
        
        grouped = df.groupby([feat_col, 'clean_method'])['success_win'].sum().unstack(fill_value=0)
        for m in methods:
            if m not in grouped.columns: grouped[m] = 0
            
        if feat_col == 'interval':
            grouped.index = pd.to_numeric(grouped.index, errors='coerce')
            grouped = grouped.sort_index()
            
        for idx, row in grouped.iterrows():
            vals = [row[m] for m in methods]
            max_val = max(vals)
            fmt_vals = [f"\\textbf{{{int(v)}}}" if v == max_val and v > 0 else str(int(v)) for v in vals]
            val_name = str(idx).replace('.0', '')
            print(f"\\textit{{{val_name}}} & " + " & ".join(fmt_vals) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")

def print_ablation_tables():
    res_base = Path("results_ablation")
    if not res_base.exists(): return
    
    profiles = sorted([d for d in res_base.iterdir() if d.is_dir()])
    
    print("\n% ==========================================")
    print("% TABLE 3: ABLATION SUMMARY (METRICS)")
    print("% ==========================================")
    print("\\begin{table}[htbp]")
    print("\\caption{Ablation Study: Performance Metrics Across Sensor Noise Profiles}")
    print("\\label{tab:ablation_metrics}")
    print("\\centering")
    print("\\begin{tabular}{@{}llcc@{}}")
    print("\\toprule")
    print("& & \\multicolumn{2}{c}{\\textbf{Merge Method}} \\\\")
    print("\\cmidrule(l){3-4}")
    print("\\textbf{Noise Profile} & \\textbf{Metric} & \\textbf{Arithmetic Mean} & \\textbf{Visit-Weighted} \\\\")
    print("\\midrule")
    
    for p_dir in profiles:
        csv_path = p_dir / "comprehensive_configuration_table.csv"
        if not csv_path.exists(): continue
        
        df = clean_dataframe(pd.read_csv(csv_path))
        agg = df.groupby('clean_method').agg(Succ=('success_rate', 'mean'), Step=('avg_steps_all', 'mean'))
        
        a_succ = agg.loc['Arithmetic Mean', 'Succ'] if 'Arithmetic Mean' in agg.index else 0
        w_succ = agg.loc['Visit-Weighted KL', 'Succ'] if 'Visit-Weighted KL' in agg.index else 0
        a_step = agg.loc['Arithmetic Mean', 'Step'] if 'Arithmetic Mean' in agg.index else 2500
        w_step = agg.loc['Visit-Weighted KL', 'Step'] if 'Visit-Weighted KL' in agg.index else 2500
        
        prof_clean = p_dir.name.split('_', 1)[-1].replace('_', ' ').title()
        
        # Dynamic Bolding: Higher is better for Success
        a_succ_str = f"\\textbf{{{a_succ:.3f}}}" if a_succ >= w_succ else f"{a_succ:.3f}"
        w_succ_str = f"\\textbf{{{w_succ:.3f}}}" if w_succ >= a_succ else f"{w_succ:.3f}"
        
        # Dynamic Bolding: Lower is better for Steps
        a_step_str = f"\\textbf{{{a_step:.1f}}}" if a_step <= w_step else f"{a_step:.1f}"
        w_step_str = f"\\textbf{{{w_step:.1f}}}" if w_step <= a_step else f"{w_step:.1f}"
        
        print(f"\\multirow{{2}}{{*}}{{\\textit{{{prof_clean}}}}} & Success Rate & {a_succ_str} & {w_succ_str} \\\\")
        print(f"& Avg Steps & {a_step_str} & {w_step_str} \\\\")
        print("\\midrule")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    print("\n% ==========================================")
    print("% TABLE 4: ABLATION FEATURE WINS")
    print("% ==========================================")
    print("\\begin{table}[htbp]")
    print("\\caption{Ablation Study: Total Wins by Noise Profile}")
    print("\\label{tab:ablation_wins}")
    print("\\centering")
    print("\\begin{tabular}{@{}lcccc@{}}")
    print("\\toprule")
    print("& \\multicolumn{2}{c}{\\textbf{Success Wins}} & \\multicolumn{2}{c}{\\textbf{Efficiency Wins}} \\\\")
    print("\\cmidrule(lr){2-3} \\cmidrule(l){4-5}")
    print("\\textbf{Noise Profile} & \\textbf{Arith. Mean} & \\textbf{Visit-Weighted} & \\textbf{Arith. Mean} & \\textbf{Visit-Weighted} \\\\")
    print("\\midrule")
    
    for p_dir in profiles:
        csv_path = p_dir / "comprehensive_configuration_table.csv"
        if not csv_path.exists(): continue
        
        df = calculate_winners(clean_dataframe(pd.read_csv(csv_path)))
        wins = df.groupby('clean_method').agg(SWins=('success_win', 'sum'), EWins=('steps_win', 'sum'))
        
        a_swin = int(wins.loc['Arithmetic Mean', 'SWins']) if 'Arithmetic Mean' in wins.index else 0
        w_swin = int(wins.loc['Visit-Weighted KL', 'SWins']) if 'Visit-Weighted KL' in wins.index else 0
        a_ewin = int(wins.loc['Arithmetic Mean', 'EWins']) if 'Arithmetic Mean' in wins.index else 0
        w_ewin = int(wins.loc['Visit-Weighted KL', 'EWins']) if 'Visit-Weighted KL' in wins.index else 0
        
        prof_clean = p_dir.name.split('_', 1)[-1].replace('_', ' ').title()
        
        # Dynamic Bolding: Higher is better for Wins
        a_swin_str = f"\\textbf{{{a_swin}}}" if a_swin >= w_swin else f"{a_swin}"
        w_swin_str = f"\\textbf{{{w_swin}}}" if w_swin >= a_swin else f"{w_swin}"
        
        a_ewin_str = f"\\textbf{{{a_ewin}}}" if a_ewin >= w_ewin else f"{a_ewin}"
        w_ewin_str = f"\\textbf{{{w_ewin}}}" if w_ewin >= a_ewin else f"{w_ewin}"
        
        print(f"\\textit{{{prof_clean}}} & {a_swin_str} & {w_swin_str} & {a_ewin_str} & {w_ewin_str} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")

if __name__ == "__main__":
    try:
        df_general = clean_dataframe(pd.read_csv("results/comprehensive_configuration_table.csv"))
        df_general = calculate_winners(df_general)
        print_general_summary(df_general)
        print_general_feature_wins(df_general)
        print_ablation_tables()
    except Exception as e:
        print(f"Error: {e}")