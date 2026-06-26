"""
Analysis and visualization for the trust-decay belief merging paper.

Primary metric: belief map quality Q_t = mean P(belief at true target location).
Secondary metrics: binary success rate, steps to discovery (in supplementary tables).

All plots are saved as IEEE-compliant PDFs (pdf.fonttype=42).
"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── IEEE compliance ───────────────────────────────────────────────────────────
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

plt.rcParams.update({
    'font.size':             11,
    'font.family':           'sans-serif',
    'axes.labelsize':        13,
    'axes.titlesize':        14,
    'legend.fontsize':       11,
    'legend.title_fontsize': 12,
    'xtick.labelsize':       11,
    'ytick.labelsize':       11,
})

# ── Method labels and colours ─────────────────────────────────────────────────
LABEL_MAPPING = {
    'standard_kl':              'Forward KL (Baseline)',
    'reverse_kl':               'Reverse KL (Baseline)',
    'arithmetic_mean':          'Arithmetic Mean (Exact)',
    'geometric_mean':           'Geometric Mean (Exact)',
    'weighted_visits_kl':       'Visit-Weighted KL (Proposed)',
    'weighted_visits_kl_reset': 'Visit-Weighted KL Reset (Proposed)',
    'trust_decay_kl':           'Trust-Decay KL (Proposed)',
    'trust_decay_kl_adaptive':  'Trust-Decay KL Adaptive (Proposed)',
}

COLORS = {
    'Forward KL (Baseline)':                '#3498db',   # blue
    'Reverse KL (Baseline)':                '#e74c3c',   # red
    'Arithmetic Mean (Exact)':              '#95a5a6',   # grey
    'Geometric Mean (Exact)':               '#2ecc71',   # green
    'Visit-Weighted KL (Proposed)':         '#9b59b6',   # purple
    'Visit-Weighted KL Reset (Proposed)':   '#8e44ad',   # dark purple
    'Trust-Decay KL (Proposed)':            '#e67e22',   # orange
    'Trust-Decay KL Adaptive (Proposed)':   '#d35400',   # dark orange
}

# Methods listed in display order for all plots
METHOD_ORDER = [
    'Arithmetic Mean (Exact)',
    'Geometric Mean (Exact)',
    'Forward KL (Baseline)',
    'Reverse KL (Baseline)',
    'Visit-Weighted KL (Proposed)',
    'Visit-Weighted KL Reset (Proposed)',
    'Trust-Decay KL (Proposed)',
    'Trust-Decay KL Adaptive (Proposed)',
]

PROPOSED = {
    'Visit-Weighted KL (Proposed)',
    'Visit-Weighted KL Reset (Proposed)',
    'Trust-Decay KL (Proposed)',
    'Trust-Decay KL Adaptive (Proposed)',
}

CACHE_FILE = "results/data_cache.pkl"


# =============================================================================
# Data loading
# =============================================================================

def load_or_cache_data(checkpoints_dir="checkpoints"):
    """
    Parse all checkpoint .pkl files and return a tidy DataFrame.
    Reads from cache on repeat runs; delete results/data_cache.pkl to refresh.
    """
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}.")
        print("Delete that file to force a fresh parse after new experiments.")
        return pd.read_pickle(CACHE_FILE)

    print(f"Scanning {checkpoints_dir} for trial results...")
    results = []
    checkpoint_path = Path(checkpoints_dir)

    if not checkpoint_path.exists():
        print(f"Error: {checkpoints_dir} not found.")
        return pd.DataFrame()

    files = list(checkpoint_path.glob("*.pkl"))
    if not files:
        print("No .pkl files found.")
        return pd.DataFrame()

    print(f"Found {len(files)} files. Parsing...")

    for i, file_path in enumerate(files):
        if "consolidated" in file_path.name:
            continue
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            meta = data.get('task_metadata', {})
            if meta:
                gs            = meta.get('grid_size', (0, 0))
                grid_size_str = f"{gs[0]}x{gs[1]}"
                grid_area     = gs[0] * gs[1]
                n_agents      = meta.get('n_agents', 0)
                pattern       = meta.get('pattern', 'unknown')
                interval      = meta.get('merge_interval', 'unknown')
                method        = meta.get('merge_method', 'unknown')
                comm_model    = meta.get('comm_model', 'fixed')
            else:
                # Fallback: parse filename
                grid_size_str = 'unknown'
                grid_area = n_agents = 0
                pattern = interval = comm_model = 'unknown'
                method = 'unknown'
                for part in file_path.stem.split('_'):
                    if part.startswith('grid'):
                        grid_size_str = part.replace('grid', '')
                        try:
                            r, c = map(int, grid_size_str.split('x'))
                            grid_area = r * c
                        except Exception:
                            pass
                    elif part.startswith('agents'):
                        try:
                            n_agents = int(part.replace('agents', ''))
                        except Exception:
                            pass
                    elif part in ('stationary', 'random', 'evasive', 'patrol'):
                        pattern = part
                    elif part.startswith('interval'):
                        interval = part.replace('interval', '')
                    elif part in ('fixed', 'poisson'):
                        comm_model = part

            target_found = data.get('target_found', False)
            steps        = data.get('first_discovery_step', 2500)
            if not target_found:
                steps = 2500  # penalty cap

            # Primary metric: belief map quality
            avg_bq   = data.get('avg_belief_quality',   np.nan)
            final_bq = data.get('final_belief_quality', np.nan)

            results.append({
                'grid_str':           grid_size_str,
                'grid_area':          grid_area,
                'n_agents':           n_agents,
                'pattern':            pattern,
                'interval':           str(interval),
                'method':             method,
                'comm_model':         comm_model,
                'found':              int(target_found),
                'steps_to_find':      steps,
                'avg_belief_quality': avg_bq,
                'final_belief_quality': final_bq,
            })

        except Exception:
            continue

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} processed...")

    if not results:
        print("No valid data extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Keep only known methods and remap labels
    known = list(LABEL_MAPPING.keys())
    df = df[df['method'].isin(known)].copy()
    df['method'] = df['method'].map(LABEL_MAPPING)
    df = df.sort_values('grid_area').reset_index(drop=True)

    os.makedirs("results", exist_ok=True)
    df.to_pickle(CACHE_FILE)
    print(f"Saved cache ({len(df)} trials).")
    return df


# =============================================================================
# Plot 1 — Belief quality vs expected communication interval  [PRIMARY]
# =============================================================================

def plot_belief_quality_vs_interval(df, colors):
    """
    The core ablation plot.  Shows how mean Q_t (belief map quality) varies
    with expected communication interval for each merge method.
    Each panel is one grid size.  Proposed methods are drawn with thicker lines.
    """
    if 'avg_belief_quality' not in df.columns or df['avg_belief_quality'].isna().all():
        print("No belief quality data available; skipping belief_quality_vs_interval plot.")
        return

    grid_sizes = sorted(df['grid_str'].unique(),
                        key=lambda s: int(s.split('x')[0]))
    n_grids = len(grid_sizes)
    if n_grids == 0:
        return

    fig, axes = plt.subplots(1, n_grids, figsize=(5 * n_grids, 5), sharey=True)
    if n_grids == 1:
        axes = [axes]

    for ax, gs in zip(axes, grid_sizes):
        sub = df[df['grid_str'] == gs].copy()
        try:
            sub['interval_num'] = pd.to_numeric(sub['interval'], errors='coerce')
        except Exception:
            continue
        sub = sub.dropna(subset=['interval_num', 'avg_belief_quality'])
        agg = (sub.groupby(['interval_num', 'method'])['avg_belief_quality']
                  .mean().reset_index())

        for method in METHOD_ORDER:
            m_data = agg[agg['method'] == method]
            if m_data.empty:
                continue
            m_data = m_data.sort_values('interval_num')
            lw     = 2.5 if method in PROPOSED else 1.2
            ls     = '-'  if method in PROPOSED else '--'
            ax.plot(m_data['interval_num'], m_data['avg_belief_quality'],
                    label=method, color=colors.get(method, '#aaaaaa'),
                    linewidth=lw, linestyle=ls, marker='o', markersize=5)

        ax.set_title(f"Grid {gs}")
        ax.set_xlabel("Expected Comm. Interval (steps)")
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"Mean Belief Quality $Q_t$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Merge Method",
               loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.15), fontsize=9)
    fig.suptitle("Belief Map Quality vs Communication Interval", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('results/belief_quality_vs_interval.pdf',
                format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved: belief_quality_vs_interval.pdf")


# =============================================================================
# Plot 2 — Belief quality vs grid size  [PRIMARY]
# =============================================================================

def plot_belief_quality_vs_grid_size(df, colors):
    """
    How belief quality scales with grid size for each method.
    Equal spacing on x-axis regardless of actual grid area.
    """
    if 'avg_belief_quality' not in df.columns or df['avg_belief_quality'].isna().all():
        print("No belief quality data; skipping belief_quality_vs_grid_size.")
        return

    df_plot = df.copy()
    df_plot['grid_dim'] = df_plot['grid_str'].apply(lambda x: int(x.split('x')[0]))
    tick_map = (df_plot[['grid_dim', 'grid_str']]
                .drop_duplicates()
                .sort_values('grid_dim')
                .reset_index(drop=True))
    tick_map['x_idx'] = tick_map.index
    df_plot = df_plot.merge(tick_map[['grid_str', 'x_idx']], on='grid_str')

    agg = (df_plot.groupby(['x_idx', 'method'])['avg_belief_quality']
               .mean().reset_index())

    fig, ax = plt.subplots(figsize=(12, 5))
    for method in METHOD_ORDER:
        m_data = agg[agg['method'] == method].sort_values('x_idx')
        if m_data.empty:
            continue
        lw = 2.5 if method in PROPOSED else 1.2
        ls = '-'  if method in PROPOSED else '--'
        ax.plot(m_data['x_idx'], m_data['avg_belief_quality'],
                label=method, color=colors.get(method, '#aaaaaa'),
                linewidth=lw, linestyle=ls, marker='o', markersize=5)

    ax.set_xticks(tick_map['x_idx'].values)
    ax.set_xticklabels(tick_map['grid_str'].values, rotation=45, ha='right')
    ax.set_xlabel("Grid Size")
    ax.set_ylabel(r"Mean Belief Quality $Q_t$")
    ax.set_title("Belief Map Quality vs Grid Size")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Method")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/belief_quality_vs_grid_size.pdf',
                format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved: belief_quality_vs_grid_size.pdf")


# =============================================================================
# Plot 3 — Method summary bar chart (belief quality)  [PRIMARY]
# =============================================================================

def plot_belief_quality_summary(df, colors):
    """
    Overall mean belief quality per method with 95% CI.
    This is the single-panel summary figure for the paper.
    """
    if 'avg_belief_quality' not in df.columns or df['avg_belief_quality'].isna().all():
        print("No belief quality data; skipping summary plot.")
        return

    order = [m for m in METHOD_ORDER if m in df['method'].unique()]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=df, x='method', y='avg_belief_quality',
                order=order, palette=colors, errorbar=('ci', 95),
                capsize=0.05, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(r"Mean Belief Quality $Q_t$")
    ax.set_title("Overall Belief Map Quality by Merge Method (95% CI)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    # Highlight proposed methods
    for patch, label in zip(ax.patches, order):
        if label in PROPOSED:
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/belief_quality_summary.pdf',
                format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved: belief_quality_summary.pdf")


# =============================================================================
# Plot 4 — Belief quality across sensor profiles  [ABLATION]
# =============================================================================

def plot_sensor_profile_ablation(results_ablation_dir="results_ablation", colors=None):
    """
    Loads per-profile CSVs from the noise ablation study and produces a
    grouped bar chart of mean belief quality across sensor profiles.
    Requires the ablation to have been run.
    """
    if colors is None:
        colors = COLORS

    ablation_path = Path(results_ablation_dir)
    if not ablation_path.exists():
        print(f"Ablation directory {results_ablation_dir} not found; skipping.")
        return

    profile_data = []
    for p_dir in sorted(ablation_path.iterdir()):
        csv_path = p_dir / "comprehensive_configuration_table.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if 'avg_belief_quality' not in df.columns:
            continue
        df['profile'] = p_dir.name.split('_', 1)[-1].replace('_', ' ').title()
        profile_data.append(df)

    if not profile_data:
        print("No ablation belief quality data found.")
        return

    combined = pd.concat(profile_data, ignore_index=True)

    # Remap method labels
    known = list(LABEL_MAPPING.keys())
    combined = combined[combined['method'].isin(known)].copy()
    combined['method'] = combined['method'].map(LABEL_MAPPING)

    order = [m for m in METHOD_ORDER if m in combined['method'].unique()]
    profiles = sorted(combined['profile'].unique())

    fig, ax = plt.subplots(figsize=(13, 5))
    x      = np.arange(len(profiles))
    width  = 0.8 / max(len(order), 1)

    for k, method in enumerate(order):
        vals = []
        errs = []
        for profile in profiles:
            sub = combined[(combined['method'] == method) &
                           (combined['profile'] == profile)]['avg_belief_quality']
            vals.append(sub.mean() if len(sub) else np.nan)
            errs.append(sub.sem() * 1.96 if len(sub) > 1 else 0)
        offset = (k - len(order) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=method, color=colors.get(method, '#aaaaaa'),
                      yerr=errs, capsize=3,
                      edgecolor='black' if method in PROPOSED else 'none',
                      linewidth=1.0 if method in PROPOSED else 0)

    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=20, ha='right')
    ax.set_ylabel(r"Mean Belief Quality $Q_t$")
    ax.set_title("Belief Map Quality Across Sensor Noise Profiles")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Method", fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/sensor_profile_ablation.pdf',
                format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved: sensor_profile_ablation.pdf")


# =============================================================================
# Plot 5 — Fixed vs Poisson communication comparison  [NEW CONTRIBUTION]
# =============================================================================

def plot_fixed_vs_poisson(df, colors):
    """
    Side-by-side comparison of belief quality under fixed vs Poisson
    communication schedules.  Only produced when both comm_model values
    are present in the data.
    """
    if 'comm_model' not in df.columns:
        return
    models = df['comm_model'].unique()
    if len(models) < 2:
        print("Only one comm_model in data; skipping fixed vs Poisson plot.")
        return

    order  = [m for m in METHOD_ORDER if m in df['method'].unique()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, model in zip(axes, ['fixed', 'poisson']):
        sub = df[df['comm_model'] == model]
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.barplot(data=sub, x='method', y='avg_belief_quality',
                    order=order, palette=colors,
                    errorbar=('ci', 95), capsize=0.05, ax=ax)
        ax.set_title(f"Communication: {model.capitalize()}")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
        ax.grid(True, axis='y', alpha=0.3)

    axes[0].set_ylabel(r"Mean Belief Quality $Q_t$")
    axes[1].set_ylabel("")
    fig.suptitle("Belief Quality: Fixed vs Poisson Communication", fontsize=14)
    plt.tight_layout()
    plt.savefig('results/fixed_vs_poisson.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved: fixed_vs_poisson.pdf")


# =============================================================================
# Plot 6 — N-agent scaling  [SUPPORTS THEORY SECTION]
# =============================================================================

def plot_n_agent_scaling(df, colors):
    """
    Belief quality as a function of team size N for each method.
    Supports the log2(N) scaling claim in the theory section.
    """
    if 'avg_belief_quality' not in df.columns or df['avg_belief_quality'].isna().all():
        return

    order = [m for m in METHOD_ORDER if m in df['method'].unique()]
    agg   = (df.groupby(['n_agents', 'method'])['avg_belief_quality']
               .mean().reset_index())

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in order:
        m_data = agg[agg['method'] == method].sort_values('n_agents')
        if m_data.empty:
            continue
        lw = 2.5 if method in PROPOSED else 1.2
        ls = '-'  if method in PROPOSED else '--'
        ax.plot(m_data['n_agents'], m_data['avg_belief_quality'],
                label=method, color=colors.get(method, '#aaaaaa'),
                linewidth=lw, linestyle=ls, marker='o', markersize=6)

    ax.set_xlabel("Number of Agents N")
    ax.set_ylabel(r"Mean Belief Quality $Q_t$")
    ax.set_title("Belief Map Quality vs Team Size")
    ax.set_xticks(sorted(df['n_agents'].unique()))
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Method")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/n_agent_scaling.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved: n_agent_scaling.pdf")


# =============================================================================
# Plot 7 — Target pattern breakdown  [SUPPLEMENTARY]
# =============================================================================

def plot_pattern_breakdown(df, colors):
    """
    Belief quality separated by target movement pattern.
    Stationary vs evasive shows whether trust decay helps more when
    the target is harder to localise.
    """
    if 'avg_belief_quality' not in df.columns or df['avg_belief_quality'].isna().all():
        return

    patterns = sorted(df['pattern'].unique())
    order    = [m for m in METHOD_ORDER if m in df['method'].unique()]
    n_pat    = len(patterns)
    if n_pat == 0:
        return

    fig, axes = plt.subplots(1, n_pat, figsize=(5 * n_pat, 5), sharey=True)
    if n_pat == 1:
        axes = [axes]

    for ax, pat in zip(axes, patterns):
        sub = df[df['pattern'] == pat]
        sns.barplot(data=sub, x='method', y='avg_belief_quality',
                    order=order, palette=colors,
                    errorbar=('ci', 95), capsize=0.04, ax=ax)
        ax.set_title(f"Pattern: {pat.capitalize()}")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    axes[0].set_ylabel(r"Mean Belief Quality $Q_t$")
    fig.suptitle("Belief Quality by Target Movement Pattern", fontsize=14)
    plt.tight_layout()
    plt.savefig('results/pattern_breakdown.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved: pattern_breakdown.pdf")


# =============================================================================
# Supplementary — binary success rate and steps (kept for completeness)
# =============================================================================

def plot_supplementary_binary(df, colors):
    """
    Binary success rate and steps-to-discovery.
    Not the primary metric but included for comparison with prior work.
    """
    order = [m for m in METHOD_ORDER if m in df['method'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Success rate
    sns.barplot(data=df, x='method', y='found',
                order=order, palette=colors,
                errorbar=('ci', 95), capsize=0.05, ax=axes[0])
    axes[0].set_title('Target Discovery Success Rate')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_xlabel('')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=25, ha='right')
    axes[0].grid(True, axis='y', alpha=0.3)

    # Steps distribution (successful trials only)
    succ = df[df['found'] == 1]
    if not succ.empty:
        sns.boxplot(data=succ, x='method', y='steps_to_find',
                    order=order, palette=colors,
                    showfliers=False, ax=axes[1])
        axes[1].set_title('Steps to Discovery (Successful Trials)')
        axes[1].set_ylabel('Steps')
        axes[1].set_xlabel('')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=25, ha='right')
        axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/supplementary_binary_metrics.pdf',
                format='pdf', bbox_inches='tight')
    plt.close()
    print("Saved: supplementary_binary_metrics.pdf")


# =============================================================================
# Configuration table for LaTeX  (unchanged structure, adds belief quality)
# =============================================================================

def generate_configuration_table(df):
    """
    Full breakdown table with belief quality as the primary metric.
    Saves two CSVs: flat and pivoted. Use extract_table_numbers.py for LaTeX.
    """
    print("Generating configuration tables...")
    group_cols = ['grid_str', 'n_agents', 'pattern', 'interval', 'method']

    agg_cols = {'found': ['count', 'mean']}
    if 'avg_belief_quality' in df.columns:
        agg_cols['avg_belief_quality'] = ['mean', 'std']
        agg_cols['steps_to_find']      = 'mean'

    summary = df.groupby(group_cols).agg(agg_cols).reset_index()
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns]
    summary.rename(columns={
        'found_count': 'total_trials',
        'found_mean':  'success_rate',
    }, inplace=True)

    summary['success_rate'] = (summary['success_rate'] * 100).round(2).astype(str) + '%'

    os.makedirs("results", exist_ok=True)
    summary.to_csv('results/comprehensive_configuration_table.csv', index=False)

    try:
        pivot = summary.pivot_table(
            index=['grid_str', 'n_agents', 'pattern', 'interval'],
            columns='method',
            values=[c for c in summary.columns if c not in
                    ['grid_str', 'n_agents', 'pattern', 'interval', 'method']],
            aggfunc='first'
        )
        pivot.to_csv('results/comprehensive_configuration_table_pivoted.csv')
    except Exception as e:
        print(f"Could not generate pivoted table: {e}")

    print("Saved comprehensive_configuration_table.csv")


# =============================================================================
# Main entry point
# =============================================================================

def analyze_target_finding(checkpoints_dir="checkpoints", comm_model="poisson"):
    """
    Args:
        checkpoints_dir: path to checkpoint .pkl files.
        comm_model:      which communication model to analyse ('fixed' or
                         'poisson').  All metric plots use only this subset.
                         The fixed-vs-Poisson comparison plot uses both.
    """
    df_all = load_or_cache_data(checkpoints_dir)
    if df_all.empty:
        return

    # ── Filter to the requested communication model ───────────────────────────
    # Every plot except plot_fixed_vs_poisson must use a single comm_model so
    # fixed-interval and Poisson-interval results are never averaged together.
    if 'comm_model' in df_all.columns and comm_model in df_all['comm_model'].unique():
        df = df_all[df_all['comm_model'] == comm_model].copy()
    else:
        print(f"Warning: comm_model='{comm_model}' not found in data. "
              f"Available: {sorted(df_all.get('comm_model', pd.Series()).unique())}. "
              f"Using all data.")
        df = df_all.copy()

    print(f"\n{len(df)} trials for comm_model='{comm_model}' (of {len(df_all)} total).")
    print(f"Methods present:    {sorted(df['method'].unique())}")
    print(f"Grid sizes present: {sorted(df['grid_str'].unique())}")
    print(f"Belief quality NaN: {df['avg_belief_quality'].isna().sum()} / {len(df)}\n")

    os.makedirs("results", exist_ok=True)
    sns.set_style("whitegrid")

    # PRIMARY METRIC PLOTS — filtered to comm_model only
    print("--- Primary metric plots ---")
    plot_belief_quality_summary(df, COLORS)
    plot_belief_quality_vs_interval(df, COLORS)
    plot_belief_quality_vs_grid_size(df, COLORS)

    # ABLATION AND THEORY SUPPORT PLOTS — filtered to comm_model only
    print("--- Ablation and theory support plots ---")
    plot_n_agent_scaling(df, COLORS)
    plot_pattern_breakdown(df, COLORS)
    plot_sensor_profile_ablation(colors=COLORS)

    # COMPARISON PLOT — uses full df_all so both models appear side by side
    print("--- Fixed vs Poisson comparison ---")
    plot_fixed_vs_poisson(df_all, COLORS)

    # SUPPLEMENTARY — filtered to comm_model only
    print("--- Supplementary binary metrics ---")
    plot_supplementary_binary(df, COLORS)

    # CONFIGURATION TABLE — filtered to comm_model only
    generate_configuration_table(df)

    # GLOBAL SUMMARY CSV
    summary_cols = ['found', 'steps_to_find']
    if 'avg_belief_quality' in df.columns:
        summary_cols.append('avg_belief_quality')
    summary = df.groupby('method')[summary_cols].agg(['mean', 'std', 'count'])
    summary.to_csv('results/method_summary.csv')

    print("\nAll outputs saved to results/")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # Set comm_model='fixed' here if you want to analyse the fixed-interval runs.
    analyze_target_finding(comm_model="poisson")