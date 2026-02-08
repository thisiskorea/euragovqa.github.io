#!/usr/bin/env python3
"""
Publication-Quality Visualizations for EuraGovExam Benchmark
Target: NeurIPS Datasets & Benchmarks Track 2025

Design principles:
- Single-column (3.25") and double-column (6.75") NeurIPS widths
- Colorblind-friendly palettes (IBM Design / Tableau)
- LaTeX-style typography
- Minimal chartjunk, high data-ink ratio
- Consistent visual language across all figures
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path
import seaborn as sns
from scipy import stats

# =============================================================================
# PUBLICATION STYLE CONFIGURATION
# =============================================================================

# NeurIPS column widths (inches)
SINGLE_COL = 3.25
DOUBLE_COL = 6.75
FULL_PAGE_HEIGHT = 9.0

# Try to use LaTeX rendering
try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    })
    USE_LATEX = True
except:
    USE_LATEX = False

# Publication-quality defaults
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
})

# Colorblind-friendly palette (IBM Design Language)
COLORS = {
    'blue': '#648FFF',
    'purple': '#785EF0',
    'magenta': '#DC267F',
    'orange': '#FE6100',
    'yellow': '#FFB000',
    'gray': '#6C757D',
    'dark': '#212529',
}

# Track colors (consistent across paper)
TRACK_COLORS = {
    'A': '#648FFF',  # Image-only: blue
    'B': '#FFB000',  # Text-only: yellow/gold
    'C': '#DC267F',  # Multimodal: magenta
}

# Model type colors
MODEL_COLORS = {
    'closed': '#648FFF',
    'open': '#FE6100',
}

# Diverging colormap for heatmaps (red-white-blue)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'custom_diverging',
    ['#DC267F', '#FFFFFF', '#648FFF']
)

# Sequential colormap for accuracy
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    'custom_sequential',
    ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#99000D']
)

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

CLOSED_MODELS = {
    "o3", "o4-mini", "GPT-4o", "GPT-4.1", "GPT-4.1-mini",
    "Gemini-2.5-pro", "Gemini-2.5-flash", "Gemini-2.5-flash-lite",
    "Claude-Sonnet-4"
}

OPEN_MODELS = {
    "Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct",
    "Qwen2-VL-72B-Instruct", "Phi-3.5-vision-instruct",
    "InternVL2.5-38B-MPO", "Ovis2-8B", "Ovis2-16B", "Ovis2-32B",
    "llama3-llava-next-8b", "llava-1.5-13b", "llava-1.5-7b",
    "LLaVA-NeXT-Video-7B-DPO-hf", "Llama-3.2-11B-Vision"
}

MODEL_FAMILIES = {
    "OpenAI": ["o3", "o4-mini", "GPT-4o", "GPT-4.1", "GPT-4.1-mini"],
    "Google": ["Gemini-2.5-pro", "Gemini-2.5-flash", "Gemini-2.5-flash-lite"],
    "Anthropic": ["Claude-Sonnet-4"],
    "Qwen": ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2-VL-72B-Instruct"],
    "LLaVA": ["llama3-llava-next-8b", "llava-1.5-13b", "llava-1.5-7b", "LLaVA-NeXT-Video-7B-DPO-hf"],
    "Ovis": ["Ovis2-8B", "Ovis2-16B", "Ovis2-32B"],
    "Other": ["Phi-3.5-vision-instruct", "InternVL2.5-38B-MPO", "Llama-3.2-11B-Vision"]
}

NATION_ORDER = ["india", "japan", "south_korea", "taiwan", "eu"]
NATION_DISPLAY = {"india": "India", "eu": "EU", "taiwan": "Taiwan", "japan": "Japan", "south_korea": "S. Korea"}

TASK_ORDER = ["earth_science", "geography", "history", "physics", "economics",
              "politics", "engineering", "law", "psychology", "administration",
              "biology", "chemistry", "language", "computer_science", "medicine",
              "philosophy", "mathematics"]
TASK_DISPLAY = {
    "chemistry": "Chem.", "philosophy": "Phil.", "earth_science": "Earth Sci.",
    "psychology": "Psych.", "economics": "Econ.", "biology": "Bio.",
    "geography": "Geo.", "physics": "Phys.", "politics": "Poli.",
    "history": "Hist.", "administration": "Admin.", "language": "Lang.",
    "medicine": "Med.", "engineering": "Eng.", "computer_science": "CS",
    "law": "Law", "mathematics": "Math."
}


def load_data():
    """Load all required JSON data files."""
    with open(ANALYSIS_DIR / "leaderboard.json") as f:
        leaderboard = json.load(f)
    with open(RESULTS_DIR / "vce_analysis.json") as f:
        vce_data = json.load(f)
    with open(ANALYSIS_DIR / "interaction_analysis_results.json") as f:
        interaction_data = json.load(f)
    with open(ANALYSIS_DIR / "pairwise_test_results.json") as f:
        pairwise_data = json.load(f)
    return leaderboard, vce_data, interaction_data, pairwise_data


def add_significance_brackets(ax, x1, x2, y, h, text, fontsize=6):
    """Add significance bracket annotation."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5, c='black')
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=fontsize)


# =============================================================================
# FIGURE 1: Model Family Comparison (Main Result)
# =============================================================================

def fig1_model_family_comparison(leaderboard):
    """
    Two-panel figure comparing closed vs open models.
    (a) Box plot with individual points
    (b) Family-level breakdown with error bars
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))

    # Classify models
    closed_scores = [(m["model"], m["overall"]) for m in leaderboard if m["model"] in CLOSED_MODELS]
    open_scores = [(m["model"], m["overall"]) for m in leaderboard if m["model"] in OPEN_MODELS]

    closed_vals = [x[1] for x in closed_scores]
    open_vals = [x[1] for x in open_scores]

    # Statistics
    t_stat, p_value = stats.ttest_ind(closed_vals, open_vals)
    pooled_std = np.sqrt(((len(closed_vals)-1)*np.var(closed_vals, ddof=1) +
                          (len(open_vals)-1)*np.var(open_vals, ddof=1)) /
                         (len(closed_vals) + len(open_vals) - 2))
    cohens_d = (np.mean(closed_vals) - np.mean(open_vals)) / pooled_std

    # Panel (a): Box plot
    ax1 = axes[0]
    positions = [0, 1]
    bp = ax1.boxplot([closed_vals, open_vals], positions=positions, widths=0.5,
                     patch_artist=True, showfliers=False)

    bp['boxes'][0].set_facecolor(MODEL_COLORS['closed'])
    bp['boxes'][1].set_facecolor(MODEL_COLORS['open'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
        box.set_edgecolor('black')
        box.set_linewidth(0.5)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(0.5)
    for cap in bp['caps']:
        cap.set_linewidth(0.5)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1)

    # Scatter individual points with jitter
    np.random.seed(42)
    jitter = 0.08
    ax1.scatter(np.random.normal(0, jitter, len(closed_vals)), closed_vals,
               c=MODEL_COLORS['closed'], s=20, alpha=0.8, edgecolor='white', linewidth=0.3, zorder=5)
    ax1.scatter(np.random.normal(1, jitter, len(open_vals)), open_vals,
               c=MODEL_COLORS['open'], s=20, alpha=0.8, edgecolor='white', linewidth=0.3, zorder=5)

    # Annotations
    ax1.axhline(y=25, color=COLORS['gray'], linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.text(1.6, 25, 'Random\nbaseline', fontsize=6, color=COLORS['gray'], va='center')

    # Significance bracket
    max_y = max(max(closed_vals), max(open_vals))
    add_significance_brackets(ax1, 0, 1, max_y + 3, 2, f'$p$ < 0.001, $d$ = {cohens_d:.1f}')

    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Closed-source\n($n$=9)', 'Open-source\n($n$=14)'])
    ax1.set_ylabel(r'Overall Accuracy (\%)')
    ax1.set_ylim(0, 105)
    ax1.set_title('(a) Closed vs. Open Source', fontweight='bold', loc='left')

    # Panel (b): Family breakdown
    ax2 = axes[1]

    family_data = []
    for family, models in MODEL_FAMILIES.items():
        scores = [m["overall"] for m in leaderboard if m["model"] in models]
        if scores:
            family_data.append({
                'family': family,
                'mean': np.mean(scores),
                'std': np.std(scores, ddof=1) if len(scores) > 1 else 0,
                'n': len(scores),
                'type': 'closed' if family in ['OpenAI', 'Google', 'Anthropic'] else 'open'
            })

    # Sort by mean accuracy
    family_data.sort(key=lambda x: x['mean'], reverse=True)

    x = np.arange(len(family_data))
    colors = [MODEL_COLORS[f['type']] for f in family_data]

    bars = ax2.bar(x, [f['mean'] for f in family_data],
                   yerr=[f['std'] for f in family_data],
                   capsize=2, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.3,
                   error_kw={'linewidth': 0.5, 'capthick': 0.5})

    ax2.axhline(y=25, color=COLORS['gray'], linestyle='--', linewidth=0.5, alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f['family'] for f in family_data], rotation=35, ha='right')
    ax2.set_ylabel(r'Overall Accuracy (\%)')
    ax2.set_ylim(0, 105)
    ax2.set_title('(b) Performance by Model Family', fontweight='bold', loc='left')

    # Add n labels
    for i, f in enumerate(family_data):
        ax2.text(i, f['mean'] + f['std'] + 2, f"$n$={f['n']}",
                ha='center', fontsize=5, color=COLORS['gray'])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=MODEL_COLORS['closed'], alpha=0.8, label='Closed-source', edgecolor='black', linewidth=0.3),
        mpatches.Patch(facecolor=MODEL_COLORS['open'], alpha=0.8, label='Open-source', edgecolor='black', linewidth=0.3),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', frameon=False)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_model_family_comparison.pdf")
    fig.savefig(FIGURES_DIR / "fig_model_family_comparison.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_model_family_comparison.pdf/png")

    return cohens_d, p_value


# =============================================================================
# FIGURE 2: Regional and Task Difficulty Analysis
# =============================================================================

def fig2_difficulty_analysis(leaderboard):
    """
    Two-panel figure showing difficulty by region and task.
    Uses gradient bars sorted by difficulty.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))

    # Compute averages
    task_means = {}
    nation_means = {}

    for model in leaderboard:
        for task in model["tasks"]:
            if task not in task_means:
                task_means[task] = []
            task_means[task].append(model["tasks"][task])
        for nation in model["nation"]:
            if nation not in nation_means:
                nation_means[nation] = []
            nation_means[nation].append(model["nation"][nation])

    task_avg = {t: (np.mean(v), np.std(v)) for t, v in task_means.items()}
    nation_avg = {n: (np.mean(v), np.std(v)) for n, v in nation_means.items()}

    # Panel (a): Region difficulty
    ax1 = axes[0]
    sorted_nations = sorted(nation_avg.items(), key=lambda x: x[1][0])

    y_pos = np.arange(len(sorted_nations))
    means = [x[1][0] for x in sorted_nations]
    stds = [x[1][1] for x in sorted_nations]
    labels = [NATION_DISPLAY.get(x[0], x[0]) for x in sorted_nations]

    # Color by difficulty
    norm = plt.Normalize(min(means), max(means))
    colors = plt.cm.RdYlGn(norm(means))

    bars = ax1.barh(y_pos, means, xerr=stds, height=0.6,
                    color=colors, edgecolor='black', linewidth=0.3,
                    capsize=2, error_kw={'linewidth': 0.5, 'capthick': 0.5})

    ax1.axvline(x=25, color=COLORS['gray'], linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel(r'Average Accuracy (\%)')
    ax1.set_xlim(0, 75)
    ax1.set_title('(a) Difficulty by Region', fontweight='bold', loc='left')

    # Add value labels
    for bar, mean in zip(bars, means):
        ax1.text(mean + stds[bars.index(bar)] + 1, bar.get_y() + bar.get_height()/2,
                f'{mean:.1f}', va='center', fontsize=6)

    # Panel (b): Task difficulty (top 10 hardest)
    ax2 = axes[1]
    sorted_tasks = sorted(task_avg.items(), key=lambda x: x[1][0])[:12]  # Show 12 hardest

    y_pos = np.arange(len(sorted_tasks))
    means = [x[1][0] for x in sorted_tasks]
    stds = [x[1][1] for x in sorted_tasks]
    labels = [TASK_DISPLAY.get(x[0], x[0]) for x in sorted_tasks]

    norm = plt.Normalize(min(means), max(means))
    colors = plt.cm.RdYlGn(norm(means))

    bars = ax2.barh(y_pos, means, xerr=stds, height=0.6,
                    color=colors, edgecolor='black', linewidth=0.3,
                    capsize=2, error_kw={'linewidth': 0.5, 'capthick': 0.5})

    ax2.axvline(x=25, color=COLORS['gray'], linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel(r'Average Accuracy (\%)')
    ax2.set_xlim(0, 75)
    ax2.set_title('(b) Difficulty by Task (12 Hardest)', fontweight='bold', loc='left')

    for bar, mean, std in zip(bars, means, stds):
        ax2.text(mean + std + 1, bar.get_y() + bar.get_height()/2,
                f'{mean:.1f}', va='center', fontsize=6)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_difficulty_analysis.pdf")
    fig.savefig(FIGURES_DIR / "fig_difficulty_analysis.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_difficulty_analysis.pdf/png")


# =============================================================================
# FIGURE 3: VCE (Visual Causal Effect) Analysis
# =============================================================================

def fig3_vce_analysis(vce_data):
    """
    Three-track accuracy comparison showing visual perception effects.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.4))

    nations = ["Japan", "India", "South Korea", "Taiwan", "EU"]
    nation_stats = vce_data["nation_stats"]

    # Panel (a): 3-Track accuracy
    ax1 = axes[0]
    x = np.arange(len(nations))
    width = 0.22

    acc_a = [nation_stats[n]["acc_a"] * 100 for n in nations]
    acc_b = [nation_stats[n]["acc_b"] * 100 for n in nations]
    acc_c = [nation_stats[n]["acc_c"] * 100 for n in nations]

    bars1 = ax1.bar(x - width, acc_a, width, label='Track A (Image)',
                    color=TRACK_COLORS['A'], edgecolor='black', linewidth=0.3)
    bars2 = ax1.bar(x, acc_b, width, label='Track B (Text)',
                    color=TRACK_COLORS['B'], edgecolor='black', linewidth=0.3)
    bars3 = ax1.bar(x + width, acc_c, width, label='Track C (Both)',
                    color=TRACK_COLORS['C'], edgecolor='black', linewidth=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace("South Korea", "S. Korea") for n in nations], fontsize=7)
    ax1.set_ylabel(r'Accuracy (\%)')
    ax1.set_ylim(0, 95)
    ax1.legend(loc='upper left', ncol=1, fontsize=6)
    ax1.set_title('(a) Three-Track Accuracy by Region', fontweight='bold', loc='left')

    # Add gridlines
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax1.set_axisbelow(True)

    # Panel (b): Visual Noise vs Benefit
    ax2 = axes[1]

    noise = [nation_stats[n]["noise"] for n in nations]
    benefit = [nation_stats[n]["benefit"] for n in nations]

    x = np.arange(len(nations))
    width = 0.35

    bars1 = ax2.bar(x - width/2, noise, width, label='Visual Noise',
                    color=COLORS['magenta'], edgecolor='black', linewidth=0.3)
    bars2 = ax2.bar(x + width/2, benefit, width, label='Visual Benefit',
                    color=COLORS['blue'], edgecolor='black', linewidth=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace("South Korea", "S. Korea") for n in nations], fontsize=7)
    ax2.set_ylabel('Count ($n$=40 per region)')
    ax2.legend(loc='upper right', fontsize=6)
    ax2.set_title('(b) Visual Noise vs. Benefit Cases', fontweight='bold', loc='left')

    # Annotate Japan
    japan_idx = 0
    ax2.annotate('', xy=(japan_idx - width/2, noise[japan_idx] + 0.3),
                xytext=(japan_idx - width/2, noise[japan_idx] + 1.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['magenta'], lw=1))
    ax2.text(japan_idx - width/2, noise[japan_idx] + 1.7, 'Highest',
            fontsize=6, ha='center', color=COLORS['magenta'])

    ax2.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_vce_analysis.pdf")
    fig.savefig(FIGURES_DIR / "fig_vce_analysis.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_vce_analysis.pdf/png")


# =============================================================================
# FIGURE 4: Model×Region Interaction Heatmap
# =============================================================================

def fig4_interaction_heatmap(leaderboard):
    """
    Heatmap showing relative model performance across regions.
    Delta = region_accuracy - overall_accuracy
    """
    # Sort models by overall accuracy
    models_sorted = sorted(leaderboard, key=lambda x: x["overall"], reverse=True)
    model_names = [m["model"] for m in models_sorted]

    nations = ["india", "japan", "south_korea", "taiwan", "eu"]

    # Build delta matrix
    delta_matrix = np.zeros((len(model_names), len(nations)))

    for i, model in enumerate(models_sorted):
        overall = model["overall"]
        for j, nation in enumerate(nations):
            nation_acc = model["nation"].get(nation, 0)
            delta_matrix[i, j] = nation_acc - overall

    # Create figure
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 0.8, 4.5))

    # Heatmap
    vmax = 25
    im = ax.imshow(delta_matrix, cmap=DIVERGING_CMAP, aspect='auto',
                   vmin=-vmax, vmax=vmax)

    # Gridlines
    ax.set_xticks(np.arange(len(nations)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(model_names)+1)-0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    # Labels
    ax.set_xticks(range(len(nations)))
    ax.set_xticklabels([NATION_DISPLAY.get(n, n) for n in nations], fontsize=7)
    ax.set_yticks(range(len(model_names)))

    # Color model names by type
    ytick_labels = []
    for name in model_names:
        if name in CLOSED_MODELS:
            ytick_labels.append(name)
        else:
            ytick_labels.append(name)
    ax.set_yticklabels(ytick_labels, fontsize=6)

    # Color y-tick labels
    for i, label in enumerate(ax.get_yticklabels()):
        if model_names[i] in CLOSED_MODELS:
            label.set_color(MODEL_COLORS['closed'])
        else:
            label.set_color(MODEL_COLORS['open'])

    # Add text for extreme values
    for i in range(len(model_names)):
        for j in range(len(nations)):
            val = delta_matrix[i, j]
            if abs(val) >= 12:
                color = 'white' if abs(val) > 18 else 'black'
                ax.text(j, i, f'{val:+.0f}', ha='center', va='center',
                       fontsize=5, color=color, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(r'$\Delta$ (Region $-$ Overall) \%p', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xlabel('Region', fontsize=8)
    ax.set_title('Model $\\times$ Region Interaction', fontweight='bold', fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_interaction_heatmap.pdf")
    fig.savefig(FIGURES_DIR / "fig_interaction_heatmap.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_interaction_heatmap.pdf/png")


# =============================================================================
# FIGURE 5: Per-Model Error Fingerprint (Representative Models)
# =============================================================================

def fig5_error_fingerprint(leaderboard):
    """
    Task-level performance fingerprints for representative models.
    Uses radar/spider chart style but as grouped bar for clarity.
    """
    # Select representative models
    representative = ["Gemini-2.5-pro", "o3", "Claude-Sonnet-4",
                     "Qwen2-VL-72B-Instruct", "GPT-4o", "InternVL2.5-38B-MPO"]

    tasks = sorted(leaderboard[0]["tasks"].keys())

    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 3.8))
    axes = axes.flatten()

    for idx, model_name in enumerate(representative):
        model_data = next((m for m in leaderboard if m["model"] == model_name), None)
        if model_data is None:
            continue

        ax = axes[idx]

        # Get task accuracies sorted by value
        task_accs = [(TASK_DISPLAY.get(t, t), model_data["tasks"][t]) for t in tasks]
        task_accs.sort(key=lambda x: x[1])

        y_pos = np.arange(len(task_accs))
        values = [x[1] for x in task_accs]
        labels = [x[0] for x in task_accs]

        # Color gradient
        norm = plt.Normalize(0, 100)
        colors = plt.cm.RdYlGn(norm(values))

        bars = ax.barh(y_pos, values, height=0.7, color=colors,
                      edgecolor='black', linewidth=0.2)

        ax.axvline(x=25, color=COLORS['gray'], linestyle='--', linewidth=0.4, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_xlabel(r'Accuracy (\%)', fontsize=6)
        ax.tick_params(axis='x', labelsize=5)

        # Title with overall score
        overall = model_data["overall"]
        model_type = "C" if model_name in CLOSED_MODELS else "O"
        ax.set_title(f'{model_name}\n(Overall: {overall:.1f}\\%)', fontsize=7, fontweight='bold')

        # Mark model type
        color = MODEL_COLORS['closed'] if model_name in CLOSED_MODELS else MODEL_COLORS['open']
        ax.spines['left'].set_color(color)
        ax.spines['left'].set_linewidth(2)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_error_fingerprint.pdf")
    fig.savefig(FIGURES_DIR / "fig_error_fingerprint.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_error_fingerprint.pdf/png")


# =============================================================================
# FIGURE 6: Pairwise Effect Size Matrix
# =============================================================================

def fig6_pairwise_effects(pairwise_data):
    """
    Cohen's h effect size matrix for top models.
    Lower triangle heatmap with significance markers.
    """
    models = pairwise_data["models_compared"]
    tests = pairwise_data["pairwise_tests"]

    n = len(models)
    effect_matrix = np.full((n, n), np.nan)
    sig_matrix = np.zeros((n, n), dtype=bool)

    for test in tests:
        m1, m2 = test["model_1"], test["model_2"]
        if m1 in models and m2 in models:
            i, j = models.index(m1), models.index(m2)
            effect_matrix[i, j] = test["cohens_h"]
            effect_matrix[j, i] = test["cohens_h"]
            sig_matrix[i, j] = test["significant_bonferroni"]
            sig_matrix[j, i] = test["significant_bonferroni"]

    # Create figure
    fig, ax = plt.subplots(figsize=(SINGLE_COL + 1.2, SINGLE_COL + 0.8))

    # Mask upper triangle
    mask = np.triu(np.ones_like(effect_matrix, dtype=bool))
    masked_data = np.ma.array(effect_matrix, mask=mask)

    im = ax.imshow(masked_data, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1.1)

    # Add annotations
    for i in range(n):
        for j in range(i):
            h = effect_matrix[i, j]
            sig = sig_matrix[i, j]
            if not np.isnan(h):
                color = 'white' if h > 0.6 else 'black'
                text = f'{h:.2f}'
                if not sig:
                    text = f'({h:.2f})'
                ax.text(j, i, text, ha='center', va='center', fontsize=5, color=color)

    # Labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(models, fontsize=6)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Cohen's $h$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Effect size reference lines
    for val, label in [(0.2, 'S'), (0.5, 'M'), (0.8, 'L')]:
        cbar.ax.axhline(y=val, color='black', linestyle='-', linewidth=0.3)
        cbar.ax.text(1.3, val, label, fontsize=5, transform=cbar.ax.transAxes, va='center')

    ax.set_title("Pairwise Effect Sizes (Top 10 Models)\n\\textit{Parentheses: not significant after Bonferroni}",
                fontsize=8, fontweight='bold')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_pairwise_effects.pdf")
    fig.savefig(FIGURES_DIR / "fig_pairwise_effects.png", dpi=300)
    plt.close(fig)
    print("Saved: fig_pairwise_effects.pdf/png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("EuraGovExam Publication Figures")
    print("Target: NeurIPS Datasets & Benchmarks Track 2025")
    print("=" * 60)

    FIGURES_DIR.mkdir(exist_ok=True)

    print(f"\nLaTeX rendering: {'enabled' if USE_LATEX else 'disabled'}")

    print("\nLoading data...")
    leaderboard, vce_data, interaction_data, pairwise_data = load_data()
    print(f"  {len(leaderboard)} models loaded")

    print("\n" + "-" * 40)
    print("Generating publication figures...")
    print("-" * 40)

    print("\n[1/6] Model Family Comparison...")
    cohens_d, p_val = fig1_model_family_comparison(leaderboard)
    print(f"       Cohen's d = {cohens_d:.2f}, p = {p_val:.2e}")

    print("\n[2/6] Difficulty Analysis...")
    fig2_difficulty_analysis(leaderboard)

    print("\n[3/6] VCE Analysis...")
    fig3_vce_analysis(vce_data)

    print("\n[4/6] Interaction Heatmap...")
    fig4_interaction_heatmap(leaderboard)

    print("\n[5/6] Error Fingerprints...")
    fig5_error_fingerprint(leaderboard)

    print("\n[6/6] Pairwise Effect Sizes...")
    fig6_pairwise_effects(pairwise_data)

    print("\n" + "=" * 60)
    print("All figures generated!")
    print(f"Output: {FIGURES_DIR}/fig_*.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
