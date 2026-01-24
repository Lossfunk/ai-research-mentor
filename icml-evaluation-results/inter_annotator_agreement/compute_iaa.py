#!/usr/bin/env python3
"""
Compute Inter-Annotator Agreement (IAA) for ICML evaluation.

Metrics computed:
1. LLM Judges (3 judges rating same items): Pairwise correlations + ICC
2. Human Raters (13 raters rating DIFFERENT items): Aggregate statistics only
   - Fleiss' Kappa is NOT valid for this design (raters saw different items)

Output: JSON report with IAA metrics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Part 1: LLM Judge Agreement (Holistic Scores)
# ============================================================

def load_llm_judge_scores(json_path: str) -> pd.DataFrame:
    """Load LLM judge scores from detailed_results.json"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    rows = []
    for item in data:
        system_id = item['system_id']
        scenario_id = item['scenario_id']
        item_id = f"{system_id}__{scenario_id}"

        for judge_output in item.get('judge_outputs', []):
            judge_name = judge_output['judge'].split('/')[-1]  # e.g., "qwen3-max"
            parsed = judge_output.get('parsed', {})

            rows.append({
                'item_id': item_id,
                'judge': judge_name,
                'overall_helpfulness': parsed.get('overall_helpfulness'),
                'student_progress': parsed.get('student_progress'),
                'mentor_effectiveness': parsed.get('mentor_effectiveness'),
                'conversation_efficiency': parsed.get('conversation_efficiency'),
            })

    return pd.DataFrame(rows)


def compute_icc(ratings_matrix: np.ndarray) -> dict:
    """
    Compute Intraclass Correlation Coefficient (ICC) for continuous ratings.
    Uses ICC(2,1) - two-way random effects, single measures.
    """
    n, k = ratings_matrix.shape

    if np.isnan(ratings_matrix).any():
        mask = ~np.isnan(ratings_matrix).any(axis=1)
        ratings_matrix = ratings_matrix[mask]
        n = ratings_matrix.shape[0]

    if n < 2:
        return {'icc': np.nan, 'interpretation': 'insufficient data'}

    grand_mean = np.mean(ratings_matrix)
    row_means = np.mean(ratings_matrix, axis=1)
    col_means = np.mean(ratings_matrix, axis=0)

    MS_R = k * np.sum((row_means - grand_mean) ** 2) / (n - 1)
    MS_C = n * np.sum((col_means - grand_mean) ** 2) / (k - 1)

    SS_total = np.sum((ratings_matrix - grand_mean) ** 2)
    SS_R = k * np.sum((row_means - grand_mean) ** 2)
    SS_C = n * np.sum((col_means - grand_mean) ** 2)
    SS_E = SS_total - SS_R - SS_C
    MS_E = SS_E / ((n - 1) * (k - 1))

    icc = (MS_R - MS_E) / (MS_R + (k - 1) * MS_E + (k / n) * (MS_C - MS_E))

    if icc < 0.5:
        interp = "poor"
    elif icc < 0.75:
        interp = "moderate"
    elif icc < 0.9:
        interp = "good"
    else:
        interp = "excellent"

    return {
        'icc': round(icc, 4),
        'interpretation': interp,
        'n_items': n,
        'n_raters': k
    }


def compute_pairwise_correlations(df: pd.DataFrame, metric: str) -> dict:
    """Compute pairwise Pearson correlations between judges."""
    judges = df['judge'].unique()
    correlations = {}

    for i, j1 in enumerate(judges):
        for j2 in judges[i+1:]:
            df1 = df[df['judge'] == j1][['item_id', metric]].set_index('item_id')
            df2 = df[df['judge'] == j2][['item_id', metric]].set_index('item_id')

            common = df1.join(df2, lsuffix='_1', rsuffix='_2', how='inner')

            if len(common) > 2:
                corr = common[f'{metric}_1'].corr(common[f'{metric}_2'])
                correlations[f"{j1}_vs_{j2}"] = round(corr, 4)

    return correlations


def analyze_llm_judges(json_path: str) -> dict:
    """Full analysis of LLM judge agreement."""
    df = load_llm_judge_scores(json_path)

    metrics = ['overall_helpfulness', 'student_progress', 'mentor_effectiveness', 'conversation_efficiency']
    results = {}

    all_correlations = []

    for metric in metrics:
        pivot = df.pivot(index='item_id', columns='judge', values=metric)
        ratings_matrix = pivot.values

        icc_result = compute_icc(ratings_matrix)
        pairwise = compute_pairwise_correlations(df, metric)

        mean_corr = np.mean(list(pairwise.values())) if pairwise else np.nan
        all_correlations.extend(list(pairwise.values()))

        results[metric] = {
            'icc': icc_result,
            'pairwise_correlations': pairwise,
            'mean_pairwise_correlation': round(mean_corr, 4) if not np.isnan(mean_corr) else None
        }

    # Summary
    overall_mean_correlation = round(np.mean(all_correlations), 4) if all_correlations else None

    results['summary'] = {
        'n_items': len(df['item_id'].unique()),
        'n_judges': len(df['judge'].unique()),
        'judges': list(df['judge'].unique()),
        'mean_icc_across_metrics': round(np.mean([
            results[m]['icc']['icc'] for m in metrics
            if results[m]['icc']['icc'] is not None and not np.isnan(results[m]['icc']['icc'])
        ]), 4),
        'mean_pairwise_correlation_across_all': overall_mean_correlation,
        'interpretation': (
            "Judges show moderate pairwise agreement. "
            "Three judges used to enable majority-based aggregation and reduce single-judge bias."
        )
    }

    return results


# ============================================================
# Part 2: Human Rater Agreement (Pairwise Preferences)
# ============================================================

def load_human_votes(votes_dir: str) -> pd.DataFrame:
    """Load all human pairwise votes from CSV files."""
    votes_path = Path(votes_dir)
    all_votes = []

    for csv_file in votes_path.glob("*.csv"):
        rater_name = csv_file.stem.split(" - ")[-1] if " - " in csv_file.stem else csv_file.stem

        df = pd.read_csv(csv_file)
        df['rater'] = rater_name
        all_votes.append(df)

    return pd.concat(all_votes, ignore_index=True)


def analyze_human_raters(votes_dir: str) -> dict:
    """
    Analyze human rater preferences.

    NOTE: Fleiss' Kappa is NOT computed because raters evaluated DIFFERENT items,
    not the same items. This is by design (distributed evaluation), so traditional
    IAA metrics don't apply. We report aggregate statistics instead.
    """
    df = load_human_votes(votes_dir)

    # Filter to mentor comparisons
    mentor_votes = df[df['pair_type'].str.contains('mentor_vs', na=False)].copy()

    if len(mentor_votes) == 0:
        return {'error': 'No mentor comparisons found'}

    # Map winner
    def map_winner(row):
        if row['winner'] == 'mentor':
            return 'mentor'
        elif row['winner'] == 'tie':
            return 'tie'
        else:
            return 'baseline'

    mentor_votes['vote'] = mentor_votes.apply(map_winner, axis=1)

    # === Aggregate Statistics ===
    vote_counts = mentor_votes['vote'].value_counts()
    total = len(mentor_votes)

    aggregate_results = {
        'total_comparisons': total,
        'mentor_wins': int(vote_counts.get('mentor', 0)),
        'baseline_wins': int(vote_counts.get('baseline', 0)),
        'ties': int(vote_counts.get('tie', 0)),
        'mentor_win_rate': round(vote_counts.get('mentor', 0) / total, 4),
        'baseline_win_rate': round(vote_counts.get('baseline', 0) / total, 4),
    }

    # === Win rates by matchup ===
    matchup_results = {}
    for pair_type in mentor_votes['pair_type'].unique():
        subset = mentor_votes[mentor_votes['pair_type'] == pair_type]
        subset_counts = subset['vote'].value_counts()
        subset_total = len(subset)

        matchup_results[pair_type] = {
            'total': subset_total,
            'mentor_wins': int(subset_counts.get('mentor', 0)),
            'baseline_wins': int(subset_counts.get('baseline', 0)),
            'ties': int(subset_counts.get('tie', 0)),
            'mentor_win_rate': round(subset_counts.get('mentor', 0) / subset_total, 4)
        }

    # === Rater-level statistics ===
    rater_stats = mentor_votes.groupby('rater').agg({
        'vote': lambda x: (x == 'mentor').mean(),
        'pair_id': 'count'
    }).rename(columns={'vote': 'mentor_preference_rate', 'pair_id': 'n_votes'})

    rater_level = {
        'rater_mentor_preference_rates': {
            k: round(v, 4) for k, v in rater_stats['mentor_preference_rate'].to_dict().items()
        },
        'votes_per_rater': rater_stats['n_votes'].to_dict(),
        'mean_mentor_preference': round(rater_stats['mentor_preference_rate'].mean(), 4),
        'std_mentor_preference': round(rater_stats['mentor_preference_rate'].std(), 4),
    }

    # === Check for overlapping items ===
    item_rater_counts = mentor_votes.groupby(['prompt_id', 'pair_type'])['rater'].nunique()
    multi_rated = item_rater_counts[item_rater_counts > 1]

    overlap_info = {
        'n_items_with_multiple_raters': len(multi_rated),
        'n_total_unique_items': len(item_rater_counts),
        'overlap_percentage': round(len(multi_rated) / len(item_rater_counts) * 100, 1) if len(item_rater_counts) > 0 else 0
    }

    return {
        'aggregate': aggregate_results,
        'by_matchup': matchup_results,
        'rater_level': rater_level,
        'overlap_info': overlap_info,
        'summary': {
            'n_raters': len(rater_stats),
            'n_comparisons': total,
            'raters': list(rater_stats.index),
            'design_note': (
                "Each rater evaluated a DISTINCT subset of comparisons (not the same items). "
                "Traditional IAA metrics (Fleiss' Kappa) do not apply to this design. "
                "We report aggregate win rates across all comparisons instead."
            )
        },
        'why_no_fleiss_kappa': (
            "Fleiss' Kappa requires all raters to evaluate the same items. "
            "In our design, comparisons were distributed across raters (each rater saw ~10-20 unique pairs). "
            f"Only {overlap_info['overlap_percentage']}% of items were rated by multiple raters, "
            "making traditional IAA metrics invalid. We report aggregate statistics instead."
        )
    }


# ============================================================
# Main
# ============================================================

def main():
    base_dir = Path(__file__).parent.parent

    llm_json = base_dir / "holistic_scoring_v2" / "detailed_results.json"
    human_votes_dir = base_dir / "human-baseline-votes"
    output_dir = Path(__file__).parent

    results = {
        'llm_judges': None,
        'human_raters': None
    }

    # Analyze LLM judges
    if llm_json.exists():
        print("Analyzing LLM judge agreement...")
        results['llm_judges'] = analyze_llm_judges(str(llm_json))
        print(f"  - Mean pairwise correlation: {results['llm_judges']['summary']['mean_pairwise_correlation_across_all']}")
        print(f"  - Mean ICC: {results['llm_judges']['summary']['mean_icc_across_metrics']}")
    else:
        print(f"Warning: {llm_json} not found")

    # Analyze human raters
    if human_votes_dir.exists():
        print("\nAnalyzing human rater preferences...")
        results['human_raters'] = analyze_human_raters(str(human_votes_dir))
        agg = results['human_raters']['aggregate']
        print(f"  - Total comparisons: {agg['total_comparisons']}")
        print(f"  - Mentor win rate: {agg['mentor_win_rate']*100:.1f}%")
        print(f"  - Note: Fleiss' Kappa NOT computed (raters saw different items)")
    else:
        print(f"Warning: {human_votes_dir} not found")

    # Save results
    output_file = output_dir / "iaa_report.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("INTER-ANNOTATOR AGREEMENT SUMMARY")
    print("="*70)

    if results['llm_judges']:
        print("\n[LLM Judges - 3 judges on 80 conversations]")
        print(f"  Design: 3 judges to enable majority aggregation (avoids ties)")
        print(f"  Mean pairwise correlation: {results['llm_judges']['summary']['mean_pairwise_correlation_across_all']}")
        print(f"  Mean ICC: {results['llm_judges']['summary']['mean_icc_across_metrics']}")
        print("  Per-metric correlations:")
        for metric in ['overall_helpfulness', 'student_progress', 'mentor_effectiveness', 'conversation_efficiency']:
            corr = results['llm_judges'][metric]['mean_pairwise_correlation']
            print(f"    - {metric}: r={corr}")

    if results['human_raters']:
        print("\n[Human Raters - 13 raters on 188 comparisons]")
        print(f"  Design: Distributed evaluation (each rater saw different items)")
        agg = results['human_raters']['aggregate']
        print(f"  Mentor win rate: {agg['mentor_win_rate']*100:.1f}% ({agg['mentor_wins']}/{agg['total_comparisons']})")
        print("  By matchup:")
        for matchup, stats in results['human_raters']['by_matchup'].items():
            print(f"    - {matchup}: {stats['mentor_win_rate']*100:.1f}% ({stats['mentor_wins']}/{stats['total']})")
        print(f"\n  Why no Fleiss' Kappa: {results['human_raters']['why_no_fleiss_kappa']}")

    return results


if __name__ == "__main__":
    main()
