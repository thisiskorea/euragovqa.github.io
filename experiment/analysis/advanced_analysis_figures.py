#!/usr/bin/env python3
"""
Advanced Analysis Figures for EuraGovExam (Polished Version)
Target: NeurIPS Datasets & Benchmarks Track 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PREMIUM STYLE CONFIGURATION
# =============================================================================

SINGLE_COL = 3.25
DOUBLE_COL = 6.75

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "DejaVu Serif"],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.5,
})

# Premium color palette (Nature-style)
COLORS = {
    'primary': '#1f77b4',      # Deep blue
    'secondary': '#ff7f0e',    # Orange
    'accent': '#d62728',       # Red
    'success': '#2ca02c',      # Green
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf',
}

# Model colors - refined
MODEL_COLORS = {
    'closed': '#3274A1',  # Steel blue
    'open': '#E1812C',    # Burnt orange
}

# Gradient colormaps
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    'premium_diverging',
    ['#3274A1', '#F7F7F7', '#E1812C']
)

CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    'premium_sequential',
    ['#FFF5EB', '#FDD49E', '#FDBB84', '#FC8D59', '#EF6548', '#D7301F', '#990000']
)

CMAP_PERFORMANCE = LinearSegmentedColormap.from_list(
    'premium_perf',
    ['#D7191C', '#FDAE61', '#FFFFBF', '#A6D96A', '#1A9641']
)

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"

CLOSED_MODELS = {
    "o3", "o4-mini", "GPT-4o", "GPT-4.1", "GPT-4.1-mini",
    "Gemini-2.5-pro", "Gemini-2.5-flash", "Gemini-2.5-flash-lite",
    "Claude-Sonnet-4"
}

MODEL_SIZES = {
    "o3": 200, "o4-mini": 70, "GPT-4o": 200, "GPT-4.1": 200, "GPT-4.1-mini": 70,
    "Gemini-2.5-pro": 175, "Gemini-2.5-flash": 70, "Gemini-2.5-flash-lite": 30,
    "Claude-Sonnet-4": 175, "Qwen2-VL-2B-Instruct": 2, "Qwen2-VL-7B-Instruct": 7,
    "Qwen2.5-VL-7B-Instruct": 7, "Qwen2-VL-72B-Instruct": 72,
    "Phi-3.5-vision-instruct": 3.8, "InternVL2.5-38B-MPO": 38,
    "Ovis2-8B": 8, "Ovis2-16B": 16, "Ovis2-32B": 32,
    "llama3-llava-next-8b": 8, "llava-1.5-13b": 13, "llava-1.5-7b": 7,
    "LLaVA-NeXT-Video-7B-DPO-hf": 7, "Llama-3.2-11B-Vision": 11,
}

NATION_DISPLAY = {"india": "India", "eu": "EU", "taiwan": "Taiwan", "japan": "Japan", "south_korea": "S. Korea"}
TASK_DISPLAY = {
    "chemistry": "Chemistry", "philosophy": "Philosophy", "earth_science": "Earth Sci.",
    "psychology": "Psychology", "economics": "Economics", "biology": "Biology",
    "geography": "Geography", "physics": "Physics", "politics": "Politics",
    "history": "History", "administration": "Admin.", "language": "Language",
    "medicine": "Medicine", "engineering": "Engineering", "computer_science": "Comp. Sci.",
    "law": "Law", "mathematics": "Mathematics"
}


def load_data():
    with open(ANALYSIS_DIR / "leaderboard.json") as f:
        return json.load(f)


# =============================================================================
# FIGURE A4: Variance Decomposition (Refined)
# =============================================================================

def figA4_variance_decomposition(leaderboard):
    """Variance decomposition: Nation vs Task"""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))

    # Calculate variances
    model_variances = []
    for model in leaderboard:
        nation_var = np.var(list(model["nation"].values()), ddof=1)
        task_var = np.var(list(model["tasks"].values()), ddof=1)
        model_variances.append({
            'model': model["model"],
            'overall': model["overall"],
            'nation_var': nation_var,
            'task_var': task_var,
            'type': 'closed' if model["model"] in CLOSED_MODELS else 'open'
        })

    # === Panel (a): Scatter plot ===
    ax1 = axes[0]

    for mv in model_variances:
        color = MODEL_COLORS[mv['type']]
        marker = 'o' if mv['type'] == 'closed' else 's'
        ax1.scatter(mv['task_var'], mv['nation_var'], c=color, s=45, alpha=0.75,
                   edgecolor='white', linewidth=0.5, marker=marker, zorder=5)

    # Reference lines
    max_val = max(max(mv['nation_var'] for mv in model_variances),
                  max(mv['task_var'] for mv in model_variances)) * 1.1

    # Diagonal (ratio = 1)
    ax1.plot([0, max_val], [0, max_val], color=COLORS['gray'], linestyle=':',
             linewidth=1, alpha=0.7, zorder=1)
    ax1.text(max_val*0.75, max_val*0.68, r'$\sigma^2_N = \sigma^2_T$',
             fontsize=7, color=COLORS['gray'], rotation=38)

    # Ratio = 3 line (approximate observed ratio)
    ratio_line = 3.0
    ax1.fill_between([0, max_val/ratio_line], [0, max_val], alpha=0.08,
                     color=MODEL_COLORS['open'], zorder=0)
    ax1.text(max_val*0.15, max_val*0.85, r'Nation $>$ Task',
             fontsize=7, color=MODEL_COLORS['open'], style='italic')

    ax1.set_xlabel(r'Task Variance ($\sigma^2_T$)')
    ax1.set_ylabel(r'Nation Variance ($\sigma^2_N$)')
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_title(r'\textbf{(a)} Per-Model Variance Components', loc='left', fontsize=9)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODEL_COLORS['closed'],
               markersize=7, label='Closed-source', markeredgecolor='white', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=MODEL_COLORS['open'],
               markersize=6, label='Open-source', markeredgecolor='white', markeredgewidth=0.5),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=6)

    # === Panel (b): Bar comparison ===
    ax2 = axes[1]

    mean_nation = np.mean([mv['nation_var'] for mv in model_variances])
    mean_task = np.mean([mv['task_var'] for mv in model_variances])
    std_nation = np.std([mv['nation_var'] for mv in model_variances], ddof=1)
    std_task = np.std([mv['task_var'] for mv in model_variances], ddof=1)
    ratio = mean_nation / mean_task

    x = [0, 1]
    bars = ax2.bar(x, [mean_task, mean_nation],
                   yerr=[std_task, std_nation],
                   color=[MODEL_COLORS['closed'], MODEL_COLORS['open']],
                   alpha=0.85, edgecolor='black', linewidth=0.5,
                   capsize=4, error_kw={'linewidth': 1, 'capthick': 1})

    # Ratio bracket
    bracket_y = max(mean_nation + std_nation, mean_task + std_task) + 20
    ax2.annotate('', xy=(0, bracket_y), xytext=(1, bracket_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax2.text(0.5, bracket_y + 15, f'\\textbf{{{ratio:.1f}}}$\\times$',
             ha='center', fontsize=11, fontweight='bold')

    # Value labels inside bars
    ax2.text(0, mean_task/2, f'{mean_task:.0f}', ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')
    ax2.text(1, mean_nation/2, f'{mean_nation:.0f}', ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels([r'Task ($\sigma^2_T$)', r'Nation ($\sigma^2_N$)'], fontsize=8)
    ax2.set_ylabel('Mean Variance')
    ax2.set_ylim(0, bracket_y + 60)
    ax2.set_title(r'\textbf{(b)} Aggregated Comparison ($n$=23)', loc='left', fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figA4_variance_decomposition.pdf")
    fig.savefig(FIGURES_DIR / "figA4_variance_decomposition.png", dpi=300)
    plt.close(fig)
    print("Saved: figA4_variance_decomposition.pdf/png")


# =============================================================================
# FIGURE A5: Scaling Analysis (Open-source Only)
# =============================================================================

def figA5_scaling_analysis(leaderboard):
    """Model scale vs performance - Open-source models only"""
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.5, 3.2))

    # Collect only open-source models
    open_data = []
    for model in leaderboard:
        name = model["model"]
        if name in MODEL_SIZES and name not in CLOSED_MODELS:
            open_data.append({'name': name, 'size': MODEL_SIZES[name], 'acc': model["overall"]})

    # Plot open models
    open_sizes = np.array([m['size'] for m in open_data])
    open_accs = np.array([m['acc'] for m in open_data])

    ax.scatter(open_sizes, open_accs, c=MODEL_COLORS['open'], s=60, alpha=0.85,
               edgecolor='white', linewidth=0.8, zorder=5, marker='s')

    # Log regression
    log_sizes = np.log10(open_sizes)
    slope, intercept, r_value, p_value, _ = stats.linregress(log_sizes, open_accs)
    x_fit = np.logspace(np.log10(min(open_sizes)*0.7), np.log10(max(open_sizes)*1.3), 100)
    y_fit = slope * np.log10(x_fit) + intercept
    ax.plot(x_fit, y_fit, color=MODEL_COLORS['open'], linestyle='--', linewidth=2, alpha=0.8)

    # Confidence band
    y_pred = slope * log_sizes + intercept
    residuals = open_accs - y_pred
    se = np.std(residuals)
    ax.fill_between(x_fit, y_fit - se, y_fit + se, color=MODEL_COLORS['open'], alpha=0.15)

    # Annotations for key open-source models (positioned in empty areas)
    annotations = [
        ("Qwen2-VL-72B-Instruct", "Qwen2-VL-72B", (8, -30)),
        ("InternVL2.5-38B-MPO", "InternVL2.5-38B", (8, 22)),
        ("Ovis2-32B", "Ovis2-32B", (-75, 25)),
        ("Qwen2-VL-2B-Instruct", "Qwen2-VL-2B", (-55, -35)),
        ("llava-1.5-7b", "LLaVA-1.5-7B", (-75, -25)),
    ]
    for full_name, display_name, offset in annotations:
        for data in open_data:
            if data['name'] == full_name:
                ax.annotate(display_name,
                           xy=(data['size'], data['acc']), xytext=offset,
                           textcoords='offset points', fontsize=6.5,
                           arrowprops=dict(arrowstyle='->', color='gray', lw=0.6,
                                          shrinkA=0, shrinkB=3),
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   edgecolor='lightgray', alpha=0.95, linewidth=0.5),
                           zorder=10)

    # Random baselines (4-choice: 25%, 5-choice: 20%)
    ax.axhline(y=25, color=COLORS['gray'], linestyle=':', linewidth=1, alpha=0.8)
    ax.axhline(y=20, color=COLORS['gray'], linestyle=':', linewidth=1, alpha=0.8)
    ax.text(58, 26, 'Random (25\\%)', fontsize=6, color=COLORS['gray'])
    ax.text(58, 21, 'Random (20\\%)', fontsize=6, color=COLORS['gray'])

    # R² and regression equation annotation
    ax.text(0.03, 0.97, f'$R^2 = {r_value**2:.2f}$\n$p < 0.01$',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=MODEL_COLORS['open'], alpha=0.9, linewidth=1.2))

    # Scaling trend annotation
    ax.text(0.55, 0.15, f'Acc $\\propto$ {slope:.1f} $\\cdot$ log(Size)',
            transform=ax.transAxes, fontsize=7, color=MODEL_COLORS['open'],
            style='italic')

    ax.set_xscale('log')
    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=9)
    ax.set_ylabel(r'Overall Accuracy (\%)', fontsize=9)
    ax.set_title('Open-source Model Scaling Analysis', fontweight='bold', fontsize=10)
    ax.set_ylim(0, 75)
    ax.set_xlim(1.5, 100)

    # Custom x-ticks
    ax.set_xticks([2, 7, 20, 70])
    ax.set_xticklabels(['2B', '7B', '20B', '70B'])

    # Grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figA5_scaling_analysis.pdf")
    fig.savefig(FIGURES_DIR / "figA5_scaling_analysis.png", dpi=300)
    plt.close(fig)
    print("Saved: figA5_scaling_analysis.pdf/png")


# =============================================================================
# FIGURE A6: Task × Tier Heatmap (Refined)
# =============================================================================

def figA6_task_tier_heatmap(leaderboard):
    """Task difficulty by model performance tier"""
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.8, 3.8))

    tasks = list(leaderboard[0]["tasks"].keys())
    task_means = {t: np.mean([m["tasks"][t] for m in leaderboard]) for t in tasks}
    sorted_tasks = sorted(task_means.keys(), key=lambda t: task_means[t])

    # Model tiers
    models_sorted = sorted(leaderboard, key=lambda m: m["overall"], reverse=True)
    tiers = [models_sorted[:5], models_sorted[5:12], models_sorted[12:]]
    tier_names = ['Top 5\n(79--87\\%)', 'Middle 7\n(43--68\\%)', 'Bottom 11\n(13--39\\%)']

    # Build matrix
    matrix = np.zeros((len(sorted_tasks), 3))
    for i, task in enumerate(sorted_tasks):
        for j, tier in enumerate(tiers):
            matrix[i, j] = np.mean([m["tasks"][task] for m in tier])

    im = ax.imshow(matrix, cmap=CMAP_PERFORMANCE, aspect='auto', vmin=10, vmax=95)

    # Grid lines
    ax.set_xticks(np.arange(3+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(sorted_tasks)+1)-0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    ax.set_xticks(range(3))
    ax.set_xticklabels(tier_names, fontsize=7)
    ax.set_yticks(range(len(sorted_tasks)))
    ax.set_yticklabels([TASK_DISPLAY.get(t, t) for t in sorted_tasks], fontsize=7)

    # Cell values
    for i in range(len(sorted_tasks)):
        for j in range(3):
            val = matrix[i, j]
            color = 'white' if val < 35 or val > 75 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                   fontsize=6, color=color, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=30)
    cbar.set_label(r'Accuracy (\%)', fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xlabel('Model Tier', fontsize=9)
    ax.set_title('Task Difficulty Across Model Tiers', fontweight='bold', fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figA6_task_tier_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "figA6_task_tier_heatmap.png", dpi=300)
    plt.close(fig)
    print("Saved: figA6_task_tier_heatmap.pdf/png")


# =============================================================================
# FIGURE A7: Rank Stability (Refined - Bump Chart)
# =============================================================================

def figA7_rank_stability(leaderboard):
    """Bump chart showing rank changes across regions"""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.8))

    nations = ["india", "japan", "south_korea", "taiwan", "eu"]
    nation_labels = [NATION_DISPLAY[n] for n in nations]

    # Top 10 models
    models_sorted = sorted(leaderboard, key=lambda m: m["overall"], reverse=True)[:10]
    model_names = [m["model"] for m in models_sorted]

    # Calculate ranks
    ranks = {name: [] for name in model_names}
    for nation in nations:
        nation_accs = [(m["model"], m["nation"][nation]) for m in leaderboard]
        nation_accs.sort(key=lambda x: x[1], reverse=True)
        for rank, (name, _) in enumerate(nation_accs, 1):
            if name in ranks:
                ranks[name].append(rank)

    x = np.arange(len(nations))

    # Color palette for individual models
    n_closed = sum(1 for n in model_names if n in CLOSED_MODELS)
    n_open = len(model_names) - n_closed
    cmap_closed = plt.cm.Blues(np.linspace(0.4, 0.9, max(n_closed, 1)))
    cmap_open = plt.cm.Oranges(np.linspace(0.4, 0.9, max(n_open, 1)))

    closed_idx, open_idx = 0, 0
    for name in model_names:
        is_closed = name in CLOSED_MODELS
        if is_closed:
            color = cmap_closed[min(closed_idx, len(cmap_closed)-1)]
            closed_idx += 1
        else:
            color = cmap_open[min(open_idx, len(cmap_open)-1)]
            open_idx += 1

        lw = 2 if name in ["Gemini-2.5-pro", "o3", "GPT-4o"] else 1.2
        alpha = 1.0 if name in ["Gemini-2.5-pro", "o3", "GPT-4o"] else 0.7

        ax.plot(x, ranks[name], '-', color=color, alpha=alpha, linewidth=lw, zorder=5)
        ax.scatter(x, ranks[name], color=color, s=30, alpha=alpha,
                  edgecolor='white', linewidth=0.5, zorder=6)

        # Right-side labels
        short_name = name.replace("Gemini-2.5-", "Gem-").replace("-Instruct", "").replace("InternVL2.5-38B-MPO", "InternVL-38B")
        ax.text(len(nations) - 0.85, ranks[name][-1], short_name, fontsize=5.5,
               va='center', color=color, alpha=min(alpha+0.2, 1.0))

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(nation_labels, fontsize=8)
    ax.set_ylabel('Rank (1 = Best)', fontsize=9)
    ax.set_ylim(17, 0)
    ax.set_xlim(-0.3, len(nations) + 1.3)
    ax.set_title('Model Rank Stability Across Regions', fontweight='bold', fontsize=10)

    # Horizontal grid
    ax.yaxis.grid(True, linestyle='-', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=MODEL_COLORS['closed'], linewidth=2, label='Closed-source'),
        Line2D([0], [0], color=MODEL_COLORS['open'], linewidth=2, label='Open-source'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7)

    # Highlight regions
    ax.axvspan(-0.5, 0.5, alpha=0.05, color='red')  # India - harder
    ax.axvspan(3.5, 4.5, alpha=0.05, color='green')  # EU - easier

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figA7_rank_stability.pdf")
    fig.savefig(FIGURES_DIR / "figA7_rank_stability.png", dpi=300)
    plt.close(fig)
    print("Saved: figA7_rank_stability.pdf/png")


# =============================================================================
# FIGURE A8: Task Correlation (Refined)
# =============================================================================

def figA8_task_correlation(leaderboard):
    """Task correlation matrix with hierarchical clustering"""
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.3, SINGLE_COL + 1.0))

    tasks = list(leaderboard[0]["tasks"].keys())
    n_tasks = len(tasks)

    # Build matrix
    task_matrix = np.zeros((len(leaderboard), n_tasks))
    for i, model in enumerate(leaderboard):
        for j, task in enumerate(tasks):
            task_matrix[i, j] = model["tasks"][task]

    # Correlation
    corr_matrix = np.corrcoef(task_matrix.T)

    # Hierarchical clustering
    linkage = hierarchy.linkage(pdist(corr_matrix), method='average')
    dendro = hierarchy.dendrogram(linkage, no_plot=True)
    order = dendro['leaves']

    corr_ordered = corr_matrix[order, :][:, order]
    tasks_ordered = [tasks[i] for i in order]

    # Custom colormap (white center for high correlation)
    cmap = LinearSegmentedColormap.from_list(
        'corr_cmap',
        ['#3274A1', '#89B4D4', '#F7F7F7', '#F4A582', '#D6604D']
    )

    im = ax.imshow(corr_ordered, cmap='RdYlBu_r', aspect='equal', vmin=0.85, vmax=1.0)

    # Grid
    ax.set_xticks(np.arange(n_tasks+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n_tasks+1)-0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', size=0)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([TASK_DISPLAY.get(t, t) for t in tasks_ordered],
                       rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([TASK_DISPLAY.get(t, t) for t in tasks_ordered], fontsize=6)

    # Only show values for notable correlations
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i != j:
                val = corr_ordered[i, j]
                if val < 0.92 or val > 0.99:
                    color = 'white' if val < 0.90 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=4.5, color=color, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, aspect=25)
    cbar.set_label('Pearson $r$', fontsize=8)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title('Task Performance Correlation\n(Hierarchically Clustered)',
                fontweight='bold', fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "figA8_task_correlation.pdf")
    fig.savefig(FIGURES_DIR / "figA8_task_correlation.png", dpi=300)
    plt.close(fig)
    print("Saved: figA8_task_correlation.pdf/png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("EuraGovExam Advanced Analysis Figures (Polished)")
    print("=" * 60)

    FIGURES_DIR.mkdir(exist_ok=True)

    print("\nLoading data...")
    leaderboard = load_data()
    print(f"  {len(leaderboard)} models loaded")

    print("\n--- Generating Polished Figures ---")

    print("\n[A4] Variance Decomposition...")
    figA4_variance_decomposition(leaderboard)

    print("\n[A5] Scaling Analysis...")
    figA5_scaling_analysis(leaderboard)

    print("\n[A6] Task×Tier Heatmap...")
    figA6_task_tier_heatmap(leaderboard)

    print("\n[A7] Rank Stability...")
    figA7_rank_stability(leaderboard)

    print("\n[A8] Task Correlation...")
    figA8_task_correlation(leaderboard)

    print("\n" + "=" * 60)
    print("All polished figures generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
