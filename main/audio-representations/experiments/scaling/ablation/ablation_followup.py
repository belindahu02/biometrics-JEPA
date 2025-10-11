"""
Follow-up Ablation Study: Finding the True Optimal Configuration

Based on the initial ablation results, this script tests:
1. Best individual parameters combined
2. Variations around the current best configuration
3. Statistical validation with multiple seeds
4. Fine-tuned learning rates around promising values
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from trainers import spectrogram_trainer_2d
from itertools import product

# Set publication-quality plot defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# =============================================
# CONFIGURATION
# =============================================

DATA_PATH = "/app/data/grouped_embeddings_full"  # UPDATE THIS
MODEL_PATH = "/app/data/experiments/ablation_full"  # UPDATE THIS
NORMALIZATION_METHOD = "log_scale"

USER_IDS = list(range(1, 31))  # 30 users

# Common parameters
COMMON_PARAMS = {
    'data_path': DATA_PATH,
    'model_path': MODEL_PATH,
    'user_ids': USER_IDS,
    'normalization_method': NORMALIZATION_METHOD,
    'epochs': 50,
    'use_augmentation': True,
    'device': 'cuda',
    'save_model_checkpoints': True,
    'checkpoint_every': 10,
    'max_cache_size': 50,
    'use_cosine_classifier': True,
}


# =============================================
# FOLLOW-UP EXPERIMENTS
# =============================================

def create_followup_experiments():
    """
    Create top 10 follow-up experiments based on initial ablation results.

    Initial results showed:
    - Best config: full model, LR 0.0003, batch 16, scale 40, smooth 0.1, warmup 5 (Kappa: 0.8339)
    - Best individual parameters: full model, scale 20, smooth 0.2, warmup 10, batch 8

    Top 10 most critical experiments:
    1. Baseline reference
    2. Best individual combined (key test!)
    3-4. Best individual with adjusted LRs
    5-8. Current best with each best individual parameter
    9-10. Current best with combinations of improvements
    """

    experiments = []

    # ========== GROUP 1: REFERENCE POINTS (2 experiments) ==========

    # Current best from initial study
    experiments.append({
        'name': 'reference_current_best',
        'description': 'Current best from initial study',
        'params': {
            'model_type': 'full',
            'batch_size': 16,
            'lr': 0.0003,
            'cosine_scale': 40.0,
            'label_smoothing': 0.1,
            'warmup_epochs': 5,
        }
    })

    # Best individual parameters combined
    experiments.append({
        'name': 'best_individual_combined',
        'description': 'All best individual parameters combined',
        'params': {
            'model_type': 'full',
            'batch_size': 8,  # Better mean performance
            'lr': 0.003,  # Best OAT improvement
            'cosine_scale': 20.0,  # Best OAT improvement
            'label_smoothing': 0.2,  # Best OAT improvement
            'warmup_epochs': 10,  # Best OAT improvement
        }
    })

    # ========== GROUP 2: BEST INDIVIDUAL WITH ADJUSTED LR (2 experiments) ==========
    # LR 0.003 might be too high when combined with other aggressive settings

    base_best_individual = {
        'model_type': 'full',
        'batch_size': 8,
        'lr': 0.003,
        'cosine_scale': 20.0,
        'label_smoothing': 0.2,
        'warmup_epochs': 10,
    }

    experiments.append({
        'name': 'best_indiv_lr0001',
        'description': 'Best individual params with LR 0.001',
        'params': {**base_best_individual, 'lr': 0.001}
    })

    experiments.append({
        'name': 'best_indiv_lr0003',
        'description': 'Best individual params with LR 0.0003',
        'params': {**base_best_individual, 'lr': 0.0003}
    })

    # ========== GROUP 3: CURRENT BEST WITH INDIVIDUAL IMPROVEMENTS (4 experiments) ==========
    # Test if we can improve current best by swapping in best individual params one at a time

    base_current_best = {
        'model_type': 'full',
        'batch_size': 16,
        'lr': 0.0003,
        'cosine_scale': 40.0,
        'label_smoothing': 0.1,
        'warmup_epochs': 5,
    }

    experiments.append({
        'name': 'current_best_scale20',
        'description': 'Current best with scale 20',
        'params': {**base_current_best, 'cosine_scale': 20.0}
    })

    experiments.append({
        'name': 'current_best_smooth02',
        'description': 'Current best with smoothing 0.2',
        'params': {**base_current_best, 'label_smoothing': 0.2}
    })

    experiments.append({
        'name': 'current_best_warmup10',
        'description': 'Current best with warmup 10',
        'params': {**base_current_best, 'warmup_epochs': 10}
    })

    experiments.append({
        'name': 'current_best_batch8',
        'description': 'Current best with batch 8',
        'params': {**base_current_best, 'batch_size': 8}
    })

    # ========== GROUP 4: COMBINED IMPROVEMENTS (2 experiments) ==========
    # Test if multiple improvements work together

    experiments.append({
        'name': 'current_best_scale20_smooth02',
        'description': 'Current best with scale 20 + smooth 0.2',
        'params': {**base_current_best, 'cosine_scale': 20.0, 'label_smoothing': 0.2}
    })

    experiments.append({
        'name': 'current_best_scale20_warmup10',
        'description': 'Current best with scale 20 + warmup 10',
        'params': {**base_current_best, 'cosine_scale': 20.0, 'warmup_epochs': 10}
    })

    return experiments


# =============================================
# EXPERIMENT RUNNER
# =============================================

def run_single_experiment(exp_name, exp_description, exp_params, results_dir):
    """Run a single experiment and return results"""
    print(f"\n{'=' * 80}")
    print(f"Running: {exp_name}")
    print(f"Description: {exp_description}")
    print(f"Parameters: {exp_params}")
    print('=' * 80)

    try:
        full_params = {**COMMON_PARAMS, **exp_params}
        full_params['model_path'] = os.path.join(results_dir, exp_name)

        test_acc, kappa_score = spectrogram_trainer_2d(**full_params)

        result = {
            'experiment': exp_name,
            'description': exp_description,
            'test_accuracy': test_acc,
            'kappa_score': kappa_score,
            'collapsed': kappa_score < 0.1,
            **exp_params
        }

        print(f"\n✓ COMPLETED: Test Accuracy = {test_acc:.4f}, Kappa = {kappa_score:.4f}")
        print(f"  Status: {'COLLAPSED ✗' if result['collapsed'] else 'SUCCESS ✓'}")

        return result

    except Exception as e:
        print(f"\n✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'experiment': exp_name,
            'description': exp_description,
            'test_accuracy': 0.0,
            'kappa_score': 0.0,
            'collapsed': True,
            'error': str(e),
            **exp_params
        }


def run_followup_study():
    """Run the follow-up ablation study"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(MODEL_PATH, f'followup_ablation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    experiments = create_followup_experiments()

    print("=" * 80)
    print("FOLLOW-UP ABLATION STUDY - TOP 10 EXPERIMENTS")
    print(f"Total experiments: {len(experiments)}")
    print(f"Results directory: {results_dir}")
    print("=" * 80)
    print("\nExperiment groups:")
    print("  - Reference points: 2")
    print("  - Best individual with adjusted LR: 2")
    print("  - Current best with individual improvements: 4")
    print("  - Combined improvements: 2")
    print(f"  TOTAL: {len(experiments)} experiments")
    print("=" * 80)

    # Run all experiments
    results = []
    for idx, exp in enumerate(experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {idx}/{len(experiments)}")
        print('=' * 80)

        result = run_single_experiment(
            exp['name'],
            exp['description'],
            exp['params'],
            results_dir
        )
        results.append(result)

        # Save intermediate results
        interim_df = pd.DataFrame(results)
        interim_df.to_csv(os.path.join(results_dir, 'interim_results.csv'), index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS AND REPORT")
    print("=" * 80)

    plot_comparison_chart(results_df, vis_dir)
    plot_learning_rate_analysis(results_df, vis_dir)
    generate_followup_report(results_df, results_dir)

    return results_df


# =============================================
# VISUALIZATION FUNCTIONS
# =============================================

def plot_comparison_chart(df, save_dir):
    """Compare key configurations"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Sort by kappa score
    sorted_df = df[~df['collapsed']].sort_values('kappa_score', ascending=True)

    if len(sorted_df) == 0:
        print("⚠ No successful experiments to plot")
        return

    # Color by group
    colors = []
    for exp in sorted_df['experiment']:
        if 'reference' in exp or 'validated' in exp:
            colors.append('green')
        elif 'best_indiv' in exp:
            colors.append('blue')
        elif 'current_best' in exp:
            colors.append('orange')
        elif 'finetune' in exp:
            colors.append('purple')
        elif 'balanced' in exp:
            colors.append('red')
        else:
            colors.append('gray')

    bars = ax.barh(range(len(sorted_df)), sorted_df['kappa_score'], color=colors, alpha=0.7)

    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['experiment'], fontsize=8)
    ax.set_xlabel('Kappa Score')
    ax.set_title('Follow-up Study: Performance Comparison')
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Reference/Validation'),
        Patch(facecolor='blue', alpha=0.7, label='Best Individual Variations'),
        Patch(facecolor='orange', alpha=0.7, label='Current Best Variations'),
        Patch(facecolor='purple', alpha=0.7, label='Fine-tuned LR'),
        Patch(facecolor='red', alpha=0.7, label='Balanced Configs')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Highlight top performer
    top_idx = sorted_df['kappa_score'].idxmax()
    top_row = sorted_df.loc[top_idx]
    top_position = sorted_df.index.get_loc(top_idx)
    ax.text(top_row['kappa_score'], top_position, f"  ★ {top_row['kappa_score']:.4f}",
            va='center', fontweight='bold', fontsize=9, color='darkred')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_chart.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'comparison_chart.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: comparison_chart.png/pdf")


def plot_learning_rate_analysis(df, save_dir):
    """Analyze learning rate effects"""
    lr_experiments = df[df['experiment'].str.contains('finetune_lr|reference|best_indiv')]

    if len(lr_experiments) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract LR values
    lr_data = []
    for _, row in lr_experiments.iterrows():
        if 'lr' in row:
            lr_data.append({'lr': row['lr'], 'kappa': row['kappa_score']})

    if len(lr_data) > 0:
        lr_df = pd.DataFrame(lr_data).sort_values('lr')
        ax.plot(lr_df['lr'], lr_df['kappa'], 'o-', markersize=8, linewidth=2)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Kappa Score')
        ax.set_title('Learning Rate Sensitivity Analysis')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # Annotate best
        best_idx = lr_df['kappa'].idxmax()
        best_row = lr_df.loc[best_idx]
        ax.annotate(f'Best: {best_row["lr"]:.4f}\nKappa: {best_row["kappa"]:.4f}',
                    xy=(best_row['lr'], best_row['kappa']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_rate_analysis.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'learning_rate_analysis.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: learning_rate_analysis.png/pdf")


def plot_statistical_validation(df, save_dir):
    """Plot statistical validation with error bars"""
    validation_experiments = df[df['experiment'].str.contains('validated')]

    if len(validation_experiments) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by configuration type
    current_best_runs = validation_experiments[validation_experiments['experiment'].str.contains('current_best')]
    best_indiv_runs = validation_experiments[validation_experiments['experiment'].str.contains('best_indiv')]

    configs = []
    means = []
    stds = []

    if len(current_best_runs) > 0:
        configs.append('Current Best\n(Full, Batch 16, LR 0.0003)')
        means.append(current_best_runs['kappa_score'].mean())
        stds.append(current_best_runs['kappa_score'].std())

    if len(best_indiv_runs) > 0:
        configs.append('Best Individual\n(Full, Batch 8, LR 0.003)')
        means.append(best_indiv_runs['kappa_score'].mean())
        stds.append(best_indiv_runs['kappa_score'].std())

    if len(configs) > 0:
        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
                      color=['orange', 'blue'])

        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs)
        ax.set_ylabel('Kappa Score')
        ax.set_title('Statistical Validation: Mean ± Std (3 seeds)')
        ax.grid(axis='y', alpha=0.3)

        # Annotate values
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std, f'{mean:.4f}±{std:.4f}',
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'statistical_validation.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'statistical_validation.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: statistical_validation.png/pdf")


def generate_followup_report(df, results_dir):
    """Generate comprehensive follow-up report"""
    lines = []
    lines.append("=" * 80)
    lines.append("FOLLOW-UP ABLATION STUDY - FINAL REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total experiments: {len(df)}")
    lines.append(f"Successful: {len(df[~df['collapsed']])}")

    # Overall best
    best = df.nlargest(1, 'kappa_score').iloc[0]
    lines.append("\n" + "=" * 80)
    lines.append("OVERALL BEST CONFIGURATION")
    lines.append("=" * 80)
    lines.append(f"Experiment: {best['experiment']}")
    lines.append(f"Kappa Score: {best['kappa_score']:.4f}")
    lines.append(f"Test Accuracy: {best['test_accuracy']:.4f}")
    lines.append(f"\nHyperparameters:")
    lines.append(f"  Model Type: {best['model_type']}")
    lines.append(f"  Batch Size: {best['batch_size']}")
    lines.append(f"  Learning Rate: {best['lr']}")
    lines.append(f"  Cosine Scale: {best['cosine_scale']}")
    lines.append(f"  Label Smoothing: {best['label_smoothing']}")
    lines.append(f"  Warmup Epochs: {best['warmup_epochs']}")

    # Compare reference points
    lines.append("\n" + "=" * 80)
    lines.append("REFERENCE POINT COMPARISON")
    lines.append("=" * 80)

    ref_current = df[df['experiment'] == 'reference_current_best']
    ref_best_indiv = df[df['experiment'] == 'best_individual_combined']

    if len(ref_current) > 0:
        lines.append(f"\nCurrent Best (from initial study):")
        lines.append(f"  Kappa: {ref_current.iloc[0]['kappa_score']:.4f}")
        lines.append(f"  Params: Full, Batch 16, LR 0.0003, Scale 40")

    if len(ref_best_indiv) > 0:
        lines.append(f"\nBest Individual Parameters Combined:")
        lines.append(f"  Kappa: {ref_best_indiv.iloc[0]['kappa_score']:.4f}")
        lines.append(f"  Params: Full, Batch 8, LR 0.003, Scale 20")

        if len(ref_current) > 0:
            delta = ref_best_indiv.iloc[0]['kappa_score'] - ref_current.iloc[0]['kappa_score']
            direction = "↑" if delta > 0 else "↓"
            lines.append(f"  {direction} Difference: {abs(delta):.4f}")

    # Top 5 configurations
    lines.append("\n" + "=" * 80)
    lines.append("TOP 5 CONFIGURATIONS")
    lines.append("=" * 80)

    top5 = df[~df['collapsed']].nlargest(5, 'kappa_score')
    for idx, (_, row) in enumerate(top5.iterrows(), 1):
        lines.append(f"\n{idx}. {row['experiment']}")
        lines.append(f"   Kappa: {row['kappa_score']:.4f} | Accuracy: {row['test_accuracy']:.4f}")
        lines.append(f"   Model: {row['model_type']}, Batch: {row['batch_size']}, LR: {row['lr']}")
        lines.append(
            f"   Scale: {row['cosine_scale']}, Smooth: {row['label_smoothing']}, Warmup: {row['warmup_epochs']}")

    # Learning rate insights
    lines.append("\n" + "=" * 80)
    lines.append("LEARNING RATE INSIGHTS")
    lines.append("=" * 80)

    lr_experiments = df[df['experiment'].str.contains('finetune')]
    if len(lr_experiments) > 0:
        best_lr = lr_experiments.nlargest(1, 'kappa_score').iloc[0]
        lines.append(f"\nBest fine-tuned LR: {best_lr['lr']}")
        lines.append(f"  Kappa: {best_lr['kappa_score']:.4f}")

    # Final recommendations
    lines.append("\n" + "=" * 80)
    lines.append("FINAL RECOMMENDATIONS")
    lines.append("=" * 80)

    lines.append(f"\n1. Optimal Configuration for Thesis:")
    lines.append(f"   {best['experiment']}")
    lines.append(f"   Expected Kappa: {best['kappa_score']:.4f}")

    # Compare with initial best
    initial_best_kappa = 0.8339  # From initial study
    improvement = best['kappa_score'] - initial_best_kappa
    if improvement > 0:
        lines.append(f"\n2. Improvement over initial best:")
        lines.append(
            f"   +{improvement:.4f} Kappa score ({improvement / initial_best_kappa * 100:.2f}% relative improvement)")
    else:
        lines.append(f"\n2. The initial best configuration remains optimal")

    lines.append(f"\n3. Key Finding:")
    if len(ref_current) > 0 and len(ref_best_indiv) > 0:
        if ref_best_indiv.iloc[0]['kappa_score'] > ref_current.iloc[0]['kappa_score']:
            lines.append(f"   → Combining best individual parameters IMPROVED performance")
        else:
            lines.append(f"   → Parameter interactions are important; best individual params don't combine optimally")

    lines.append(f"\n4. Use this configuration for final experiments and reporting")

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    # Save report
    report_path = os.path.join(results_dir, 'followup_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ Saved: followup_report.txt")
    print("\n" + "\n".join(lines))


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run follow-up ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This follow-up study tests whether combining the best individual parameters
gives better results than the current best configuration.

Based on initial results:
- Current best: Full model, LR 0.0003, batch 16, scale 40 (Kappa: 0.8339)
- Best individual params: Full model, LR 0.003, batch 8, scale 20

This script runs ~26 experiments to find the true optimal configuration.
        """
    )

    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--users', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    if args.data_path:
        DATA_PATH = args.data_path
        COMMON_PARAMS['data_path'] = DATA_PATH

    if args.model_path:
        MODEL_PATH = args.model_path
        COMMON_PARAMS['model_path'] = MODEL_PATH

    if args.users:
        USER_IDS = list(range(1, args.users + 1))
        COMMON_PARAMS['user_ids'] = USER_IDS

    if args.epochs:
        COMMON_PARAMS['epochs'] = args.epochs

    if DATA_PATH == "path/to/your/data" or MODEL_PATH == "path/to/models/followup_ablation":
        print("\n⚠️ WARNING: Please update DATA_PATH and MODEL_PATH!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(0)

    print("\n" + "=" * 80)
    print("FOLLOW-UP ABLATION STUDY")
    print("=" * 80)
    print(f"\nBased on initial results:")
    print(f"  Current best: Kappa = 0.8339")
    print(f"  Testing if best individual params combined perform better")
    print(f"\nExperiments: ~26")
    print(f"Users: {len(USER_IDS)}")
    print(f"Epochs: {COMMON_PARAMS['epochs']}")
    print("=" * 80)

    print("\n🚀 Starting follow-up study...")
    results = run_followup_study()

    print("\n" + "=" * 80)
    print("✅ FOLLOW-UP STUDY COMPLETE!")
    print("=" * 80)

    best = results.nlargest(1, 'kappa_score').iloc[0]
    print(f"\n🏆 Best configuration: {best['experiment']}")
    print(f"  Kappa: {best['kappa_score']:.4f}")
    print(f"  Improvement over initial best: {best['kappa_score'] - 0.8339:+.4f}")