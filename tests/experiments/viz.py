import polars as pl
import matplotlib.pyplot as plt
import numpy as np

LABELS = [
    'Baseline\n(LLM only)',
    'RAG\nonly',
    'Search\nonly',
    'Full Agent\n(RAG+Search)',
]
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

def add_value_labels(ax, values, fmt, offset=0.015):
    max_val = max(values) if values else 1
    for i, v in enumerate(values):
        ax.text(i, v + (max_val * offset), f'{v:{fmt}}',
                ha='center', fontsize=11, fontweight='bold')

def plot_benchmarks(
    scored_path="benchmark_results/benchmark_scored.jsonl",
    comparison_path="benchmark_results/benchmark_comparison.jsonl",
    out_path="benchmark_results/benchmarks_overview.png",
):
    s_df = pl.read_ndjson(scored_path).to_pandas()
    c_df = pl.read_ndjson(comparison_path).to_pandas()

    latencies = [
        c_df['base_latency'].mean(),
        c_df['rag_latency'].mean(),
        c_df['search_latency'].mean(),
        c_df['agent_latency'].mean(),
    ]

    score_cols = ['score_baseline', 'score_rag', 'score_search', 'score_agent']
    avg_scores = [s_df[c].mean() for c in score_cols]
    sum_scores = [s_df[c].sum() for c in score_cols]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1 — Latency
    bars = axes[0, 0].bar(LABELS, latencies, color=COLORS, edgecolor='black', linewidth=0.5)
    add_value_labels(axes[0, 0], latencies, '.1f')
    axes[0, 0].set_ylabel('Average Latency (sec)')
    axes[0, 0].set_title('Latency', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].tick_params(axis='x', labelsize=9)

    # 2 — Average Score
    axes[0, 1].bar(LABELS, avg_scores, color=COLORS, edgecolor='black', linewidth=0.5)
    add_value_labels(axes[0, 1], avg_scores, '.2f')
    axes[0, 1].set_ylabel('Average Rating (1–5)')
    axes[0, 1].set_title('Average Score', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim(0, 5.5)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', labelsize=9)

    # 3 — Sum Score
    axes[0, 2].bar(LABELS, sum_scores, color=COLORS, edgecolor='black', linewidth=0.5)
    add_value_labels(axes[0, 2], sum_scores, '.0f')
    axes[0, 2].set_ylabel('Sum of Scores')
    axes[0, 2].set_title('Score Sum', fontsize=14, fontweight='bold')
    axes[0, 2].grid(axis='y', alpha=0.3)
    axes[0, 2].tick_params(axis='x', labelsize=9)

    # 4 — Score Distribution (box plot)
    score_data = [s_df[c].dropna() for c in score_cols]
    bp = axes[1, 0].boxplot(score_data, labels=LABELS, patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_ylabel('Score Distribution')
    axes[1, 0].set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0.5, 5.5)
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', labelsize=9)

    # 5 — Score by Category (if available)
    if 'category' in s_df.columns and s_df['category'].nunique() > 1:
        categories = s_df['category'].unique()
        x = np.arange(len(categories))
        width = 0.2

        for i, (col, color) in enumerate(zip(score_cols, COLORS)):
            cat_means = [s_df[s_df['category'] == cat][col].mean() for cat in categories]
            axes[1, 1].bar(x + i*width, cat_means, width, label=LABELS[i], color=color, alpha=0.8)

        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Average Score')
        axes[1, 1].set_title('Score by Category', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x + 1.5*width)
        axes[1, 1].set_xticklabels(categories, rotation=45, ha='right')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].set_ylim(0, 5.5)
    else:
        axes[1, 1].text(0.5, 0.5, 'No category data\nor single category',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Score by Category', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

    # 6 — Score by Difficulty (if available)
    if 'difficulty' in s_df.columns and s_df['difficulty'].nunique() > 1:
        difficulties = sorted(s_df['difficulty'].unique())
        x = np.arange(len(difficulties))
        width = 0.2

        for i, (col, color) in enumerate(zip(score_cols, COLORS)):
            diff_means = [s_df[s_df['difficulty'] == diff][col].mean() for diff in difficulties]
            axes[1, 2].bar(x + i*width, diff_means, width, label=LABELS[i], color=color, alpha=0.8)

        axes[1, 2].set_xlabel('Difficulty')
        axes[1, 2].set_ylabel('Average Score')
        axes[1, 2].set_title('Score by Difficulty', fontsize=14, fontweight='bold')
        axes[1, 2].set_xticks(x + 1.5*width)
        axes[1, 2].set_xticklabels(difficulties)
        axes[1, 2].legend(fontsize=8)
        axes[1, 2].grid(axis='y', alpha=0.3)
        axes[1, 2].set_ylim(0, 5.5)
    else:
        axes[1, 2].text(0.5, 0.5, 'No difficulty data\nor single difficulty',
                        ha='center', va='center', fontsize=12)
        axes[1, 2].set_title('Score by Difficulty', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()

    # Print detailed analysis
    print("\n" + "="*60)
    print("DETAILED SCORE ANALYSIS")
    print("="*60)
    print(f"\nAverage Scores:")
    for label, score in zip(LABELS, avg_scores):
        print(f"  {label.replace(chr(10), ' '):30} {score:.2f}")

    # Check score differences
    print(f"\nScore Differences (vs Baseline):")
    for i, (label, score) in enumerate(zip(LABELS[1:], avg_scores[1:])):
        diff = score - avg_scores[0]
        print(f"  {label.replace(chr(10), ' '):30} {diff:+.2f}")

    # Check standard deviations
    print(f"\nScore Standard Deviations:")
    for label, col in zip(LABELS, score_cols):
        std = s_df[col].std()
        print(f"  {label.replace(chr(10), ' '):30} {std:.2f}")

    # Check if scores are identical
    identical_count = 0
    for idx, row in s_df.iterrows():
        if row[score_cols].nunique() == 1:
            identical_count += 1
    print(f"\nQuestions with identical scores across all methods: {identical_count}/{len(s_df)}")
    print(f"Percentage: {100*identical_count/len(s_df):.1f}%")

if __name__ == "__main__":
    plot_benchmarks()
