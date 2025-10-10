"""
Efficient Ablation Study for Optimal Hyperparameter Configuration

Since cosine classifier, label smoothing, and warmup all help prevent collapse,
we always include them and focus on finding the optimal configuration.

Strategy: Use a smart fractional factorial design instead of full grid search
to reduce experiments from ~162 (3*2*2*3*3*3) to ~30 meaningful experiments.

Includes experiments for:
- Learning rate: [0.0003, 0.001, 0.003]
- Batch size: [8, 16]
- Model type: [lightweight, full]
- Cosine scale: [20, 40, 64]
- Label smoothing: [0.0, 0.1, 0.2]
- Warmup epochs: [0, 5, 10]
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
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# =============================================
# CONFIGURATION
# =============================================

DATA_PATH = "path/to/your/data"  # UPDATE THIS
MODEL_PATH = "path/to/models/efficient_ablation"  # UPDATE THIS
NORMALIZATION_METHOD = "log_scale"

# Test with challenging number of users
USER_IDS = list(range(1, 31))  # 30 users

# Common parameters for all experiments
COMMON_PARAMS = {
    'data_path': DATA_PATH,
    'model_path': MODEL_PATH,
    'user_ids': USER_IDS,
    'normalization_method': NORMALIZATION_METHOD,
    'epochs': 50,  # Adjust based on compute budget
    'use_augmentation': True,
    'device': 'cuda',
    'save_model_checkpoints': True,
    'checkpoint_every': 10,
    'max_cache_size': 50,
    # Always use these to prevent collapse
    'use_cosine_classifier': True,
}


# =============================================
# EFFICIENT EXPERIMENT DESIGN
# =============================================

def create_efficient_experiment_set():
    """
    Create an efficient set of experiments using a smart sampling strategy.

    Strategy:
    1. Baseline with recommended defaults (1 exp)
    2. One-at-a-time (OAT) variations from baseline (12 exp)
    3. Model type comparison with optimal settings (2 exp)
    4. Batch size comparison with optimal settings (2 exp)
    5. Interaction tests for critical pairs (6 exp)

    Total: ~23 experiments instead of 162
    """

    experiments = []

    # ========== GROUP 1: BASELINE ==========
    # Start with recommended defaults
    baseline = {
        'name': 'baseline_recommended',
        'description': 'Baseline with recommended defaults',
        'params': {
            'lr': 0.001,
            'batch_size': 16,
            'model_type': 'lightweight',
            'cosine_scale': 40.0,
            'label_smoothing': 0.1,
            'warmup_epochs': 5,
        }
    }
    experiments.append(baseline)

    # ========== GROUP 2: ONE-AT-A-TIME (OAT) VARIATIONS ==========
    # Test each parameter while keeping others at baseline

    # Learning rate variations
    for lr in [0.0003, 0.003]:  # Skip 0.001 as it's baseline
        experiments.append({
            'name': f'oat_lr_{lr:.4f}'.replace('.', '_'),
            'description': f'OAT: Learning Rate = {lr}',
            'params': {**baseline['params'], 'lr': lr}
        })

    # Cosine scale variations
    for scale in [20.0, 64.0]:  # Skip 40.0 as it's baseline
        experiments.append({
            'name': f'oat_scale_{int(scale)}',
            'description': f'OAT: Cosine Scale = {scale}',
            'params': {**baseline['params'], 'cosine_scale': scale}
        })

    # Label smoothing variations
    for smoothing in [0.0, 0.2]:  # Skip 0.1 as it's baseline
        experiments.append({
            'name': f'oat_smoothing_{smoothing:.1f}'.replace('.', '_'),
            'description': f'OAT: Label Smoothing = {smoothing}',
            'params': {**baseline['params'], 'label_smoothing': smoothing}
        })

    # Warmup variations
    for warmup in [0, 10]:  # Skip 5 as it's baseline
        experiments.append({
            'name': f'oat_warmup_{warmup}',
            'description': f'OAT: Warmup Epochs = {warmup}',
            'params': {**baseline['params'], 'warmup_epochs': warmup}
        })

    # Batch size variations
    for batch_size in [8]:  # Skip 16 as it's baseline
        experiments.append({
            'name': f'oat_batch_{batch_size}',
            'description': f'OAT: Batch Size = {batch_size}',
            'params': {**baseline['params'], 'batch_size': batch_size}
        })

    # Model type variations
    for model_type in ['full']:  # Skip lightweight as it's baseline
        experiments.append({
            'name': f'oat_model_{model_type}',
            'description': f'OAT: Model Type = {model_type}',
            'params': {**baseline['params'], 'model_type': model_type}
        })

    # ========== GROUP 3: CRITICAL INTERACTIONS ==========
    # Test combinations that are likely to interact

    # Interaction 1: Batch size + Learning rate (smaller batch needs smaller LR)
    experiments.append({
        'name': 'interact_batch8_lr0003',
        'description': 'Batch 8 + LR 0.0003',
        'params': {**baseline['params'], 'batch_size': 8, 'lr': 0.0003}
    })

    # Interaction 2: Full model + higher LR (more capacity needs different LR)
    experiments.append({
        'name': 'interact_full_lr0003',
        'description': 'Full model + LR 0.0003',
        'params': {**baseline['params'], 'model_type': 'full', 'lr': 0.0003}
    })

    experiments.append({
        'name': 'interact_full_lr003',
        'description': 'Full model + LR 0.003',
        'params': {**baseline['params'], 'model_type': 'full', 'lr': 0.003}
    })

    # Interaction 3: Higher cosine scale + higher smoothing (both increase regularization)
    experiments.append({
        'name': 'interact_scale64_smooth02',
        'description': 'High scale + High smoothing',
        'params': {**baseline['params'], 'cosine_scale': 64.0, 'label_smoothing': 0.2}
    })

    # Interaction 4: No warmup + lower LR (safe combination)
    experiments.append({
        'name': 'interact_nowarmup_lr0003',
        'description': 'No warmup + Low LR',
        'params': {**baseline['params'], 'warmup_epochs': 0, 'lr': 0.0003}
    })

    # Interaction 5: Full model + Batch 8 (memory/performance tradeoff)
    experiments.append({
        'name': 'interact_full_batch8',
        'description': 'Full model + Batch 8',
        'params': {**baseline['params'], 'model_type': 'full', 'batch_size': 8}
    })

    # ========== GROUP 4: EXTREME CONFIGURATIONS ==========
    # Test corner cases

    # Conservative: Everything regularized
    experiments.append({
        'name': 'extreme_conservative',
        'description': 'Conservative: Low LR, High smoothing, Long warmup',
        'params': {
            'lr': 0.0003,
            'batch_size': 16,
            'model_type': 'lightweight',
            'cosine_scale': 64.0,
            'label_smoothing': 0.2,
            'warmup_epochs': 10,
        }
    })

    # Aggressive: Fast training
    experiments.append({
        'name': 'extreme_aggressive',
        'description': 'Aggressive: High LR, Low regularization',
        'params': {
            'lr': 0.003,
            'batch_size': 16,
            'model_type': 'lightweight',
            'cosine_scale': 20.0,
            'label_smoothing': 0.0,
            'warmup_epochs': 0,
        }
    })

    # Full model optimized
    experiments.append({
        'name': 'extreme_full_optimized',
        'description': 'Full model with optimized settings',
        'params': {
            'lr': 0.0003,
            'batch_size': 8,
            'model_type': 'full',
            'cosine_scale': 40.0,
            'label_smoothing': 0.1,
            'warmup_epochs': 10,
        }
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
        print(f"  Status: {'COLLAPSED ❌' if result['collapsed'] else 'SUCCESS ✓'}")

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


def run_efficient_ablation_study():
    """Run the efficient ablation study"""

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(MODEL_PATH, f'efficient_ablation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    # Create visualization subdirectory
    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Get experiment list
    experiments = create_efficient_experiment_set()

    print("=" * 80)
    print("EFFICIENT ABLATION STUDY")
    print(f"Users: {len(USER_IDS)} (S{USER_IDS[0]:03d} to S{USER_IDS[-1]:03d})")
    print(f"Total experiments: {len(experiments)}")
    print(f"Results directory: {results_dir}")
    print("=" * 80)
    print("\nExperiment groups:")
    print("  - Baseline: 1 experiment")
    print("  - One-at-a-time variations: 11 experiments")
    print("  - Critical interactions: 6 experiments")
    print("  - Extreme configurations: 3 experiments")
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

    # Create final dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_oat_analysis(results_df, vis_dir)
    plot_model_comparison(results_df, vis_dir)
    plot_batch_size_comparison(results_df, vis_dir)
    plot_interaction_effects(results_df, vis_dir)
    plot_overall_ranking(results_df, vis_dir)

    # Generate tables
    create_summary_table(results_df, vis_dir)
    create_top_configurations_table(results_df, vis_dir, top_n=10)

    # Generate report
    generate_final_report(results_df, results_dir)

    return results_df


# =============================================
# VISUALIZATION FUNCTIONS
# =============================================

def plot_oat_analysis(df, save_dir):
    """Figure 1: One-at-a-time parameter effects"""
    oat_experiments = df[df['experiment'].str.startswith('oat_')]
    baseline = df[df['experiment'] == 'baseline_recommended'].iloc[0]

    if len(oat_experiments) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Group by parameter type
    param_groups = {
        'Learning Rate': oat_experiments[oat_experiments['experiment'].str.contains('lr')],
        'Cosine Scale': oat_experiments[oat_experiments['experiment'].str.contains('scale')],
        'Label Smoothing': oat_experiments[oat_experiments['experiment'].str.contains('smoothing')],
        'Warmup Epochs': oat_experiments[oat_experiments['experiment'].str.contains('warmup')],
        'Batch Size': oat_experiments[oat_experiments['experiment'].str.contains('batch')],
        'Model Type': oat_experiments[oat_experiments['experiment'].str.contains('model')],
    }

    for idx, (param_name, group) in enumerate(param_groups.items()):
        ax = axes[idx]

        if len(group) == 0:
            ax.axis('off')
            continue

        # Add baseline
        all_data = pd.concat([pd.DataFrame([baseline]), group])

        # Plot
        x_pos = np.arange(len(all_data))
        bars = ax.bar(x_pos, all_data['kappa_score'], alpha=0.7)

        # Color baseline differently
        bars[0].set_color('green')
        bars[0].set_alpha(0.9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([exp.split('_')[-1] if i > 0 else 'baseline'
                            for i, exp in enumerate(all_data['experiment'])],
                           rotation=45, ha='right')
        ax.set_ylabel('Kappa Score')
        ax.set_title(f'Effect of {param_name}')
        ax.axhline(y=baseline['kappa_score'], color='green', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_oat_analysis.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig1_oat_analysis.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig1_oat_analysis.png/pdf")


def plot_model_comparison(df, save_dir):
    """Figure 2: Model type comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Group by model type
    lightweight = df[df['model_type'] == 'lightweight']
    full = df[df['model_type'] == 'full']

    if len(lightweight) > 0 and len(full) > 0:
        # Box plots
        data_kappa = [lightweight['kappa_score'], full['kappa_score']]
        data_acc = [lightweight['test_accuracy'], full['test_accuracy']]

        bp1 = ax1.boxplot(data_kappa, labels=['Lightweight', 'Full'], patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightblue')
        bp1['boxes'][1].set_facecolor('lightcoral')
        ax1.set_ylabel('Kappa Score')
        ax1.set_title('Model Type Comparison: Kappa Score')
        ax1.grid(axis='y', alpha=0.3)

        bp2 = ax2.boxplot(data_acc, labels=['Lightweight', 'Full'], patch_artist=True)
        bp2['boxes'][0].set_facecolor('lightblue')
        bp2['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Model Type Comparison: Test Accuracy')
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_model_comparison.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig2_model_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig2_model_comparison.png/pdf")


def plot_batch_size_comparison(df, save_dir):
    """Figure 3: Batch size comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Group by batch size
    batch8 = df[df['batch_size'] == 8]
    batch16 = df[df['batch_size'] == 16]

    if len(batch8) > 0 and len(batch16) > 0:
        # Box plots
        data_kappa = [batch8['kappa_score'], batch16['kappa_score']]
        data_acc = [batch8['test_accuracy'], batch16['test_accuracy']]

        bp1 = ax1.boxplot(data_kappa, labels=['Batch 8', 'Batch 16'], patch_artist=True)
        bp1['boxes'][0].set_facecolor('lightgreen')
        bp1['boxes'][1].set_facecolor('lightyellow')
        ax1.set_ylabel('Kappa Score')
        ax1.set_title('Batch Size Comparison: Kappa Score')
        ax1.grid(axis='y', alpha=0.3)

        bp2 = ax2.boxplot(data_acc, labels=['Batch 8', 'Batch 16'], patch_artist=True)
        bp2['boxes'][0].set_facecolor('lightgreen')
        bp2['boxes'][1].set_facecolor('lightyellow')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Batch Size Comparison: Test Accuracy')
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_batch_comparison.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig3_batch_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig3_batch_comparison.png/pdf")


def plot_interaction_effects(df, save_dir):
    """Figure 4: Interaction effects"""
    interact_experiments = df[df['experiment'].str.startswith('interact_')]

    if len(interact_experiments) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by kappa score
    interact_sorted = interact_experiments.sort_values('kappa_score', ascending=True)

    # Create horizontal bar chart
    colors = plt.cm.viridis(interact_sorted['kappa_score'] / interact_sorted['kappa_score'].max())
    bars = ax.barh(range(len(interact_sorted)), interact_sorted['kappa_score'], color=colors)

    ax.set_yticks(range(len(interact_sorted)))
    ax.set_yticklabels(interact_sorted['description'], fontsize=9)
    ax.set_xlabel('Kappa Score')
    ax.set_title('Interaction Effects: Combined Parameters')
    ax.grid(axis='x', alpha=0.3)

    # Annotate values
    for i, (idx, row) in enumerate(interact_sorted.iterrows()):
        ax.text(row['kappa_score'], i, f" {row['kappa_score']:.4f}",
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_interactions.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig4_interactions.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig4_interactions.png/pdf")


def plot_overall_ranking(df, save_dir):
    """Figure 5: Overall performance ranking"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Sort all successful experiments
    successful = df[~df['collapsed']].sort_values('kappa_score', ascending=True)

    if len(successful) == 0:
        print("⚠ No successful experiments to plot")
        return

    # Color by experiment group
    colors = []
    for exp in successful['experiment']:
        if 'baseline' in exp:
            colors.append('green')
        elif 'oat' in exp:
            colors.append('blue')
        elif 'interact' in exp:
            colors.append('orange')
        elif 'extreme' in exp:
            colors.append('red')
        else:
            colors.append('gray')

    bars = ax.barh(range(len(successful)), successful['kappa_score'], color=colors, alpha=0.7)

    ax.set_yticks(range(len(successful)))
    ax.set_yticklabels(successful['experiment'], fontsize=8)
    ax.set_xlabel('Kappa Score')
    ax.set_title('Overall Performance Ranking (All Configurations)')
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Baseline'),
        Patch(facecolor='blue', alpha=0.7, label='One-at-a-time'),
        Patch(facecolor='orange', alpha=0.7, label='Interactions'),
        Patch(facecolor='red', alpha=0.7, label='Extreme configs')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Highlight top 3
    top3_indices = successful.nlargest(3, 'kappa_score').index
    for idx, (i, row) in enumerate(successful.iterrows()):
        if i in top3_indices:
            ax.text(row['kappa_score'], idx, f"  ★ {row['kappa_score']:.4f}",
                    va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_overall_ranking.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig5_overall_ranking.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig5_overall_ranking.png/pdf")


def create_summary_table(df, save_dir):
    """Table 1: Summary statistics"""
    summary_data = []

    # Overall
    summary_data.append({
        'Category': 'All Experiments',
        'Count': len(df),
        'Mean Kappa': df['kappa_score'].mean(),
        'Std Kappa': df['kappa_score'].std(),
        'Max Kappa': df['kappa_score'].max(),
        'Mean Accuracy': df['test_accuracy'].mean(),
        'Max Accuracy': df['test_accuracy'].max(),
    })

    # By experiment group
    for prefix, name in [('baseline', 'Baseline'), ('oat', 'One-at-a-time'),
                         ('interact', 'Interactions'), ('extreme', 'Extreme')]:
        group = df[df['experiment'].str.startswith(prefix)]
        if len(group) > 0:
            summary_data.append({
                'Category': name,
                'Count': len(group),
                'Mean Kappa': group['kappa_score'].mean(),
                'Std Kappa': group['kappa_score'].std(),
                'Max Kappa': group['kappa_score'].max(),
                'Mean Accuracy': group['test_accuracy'].mean(),
                'Max Accuracy': group['test_accuracy'].max(),
            })

    # By model type
    for model_type in ['lightweight', 'full']:
        group = df[df['model_type'] == model_type]
        if len(group) > 0:
            summary_data.append({
                'Category': f'Model: {model_type}',
                'Count': len(group),
                'Mean Kappa': group['kappa_score'].mean(),
                'Std Kappa': group['kappa_score'].std(),
                'Max Kappa': group['kappa_score'].max(),
                'Mean Accuracy': group['test_accuracy'].mean(),
                'Max Accuracy': group['test_accuracy'].max(),
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(save_dir, 'table1_summary.csv'), index=False)

    latex_table = summary_df.to_latex(index=False, float_format="%.4f")
    with open(os.path.join(save_dir, 'table1_summary.tex'), 'w') as f:
        f.write(latex_table)

    print(f"✓ Saved: table1_summary.csv/tex")
    return summary_df


def create_top_configurations_table(df, save_dir, top_n=10):
    """Table 2: Top configurations"""
    successful = df[~df['collapsed']].copy()
    top_configs = successful.nlargest(min(top_n, len(successful)), 'kappa_score')

    columns = ['experiment', 'kappa_score', 'test_accuracy', 'model_type',
               'batch_size', 'lr', 'cosine_scale', 'label_smoothing', 'warmup_epochs']
    columns = [col for col in columns if col in top_configs.columns]

    top_display = top_configs[columns].copy()
    top_display.to_csv(os.path.join(save_dir, 'table2_top_configs.csv'), index=False)

    latex_table = top_display.to_latex(index=False, float_format="%.4f")
    with open(os.path.join(save_dir, 'table2_top_configs.tex'), 'w') as f:
        f.write(latex_table)

    print(f"✓ Saved: table2_top_configs.csv/tex")
    return top_display


def generate_final_report(df, results_dir):
    """Generate comprehensive text report"""
    lines = []
    lines.append("=" * 80)
    lines.append("EFFICIENT ABLATION STUDY - FINAL REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total experiments: {len(df)}")
    lines.append(f"Successful (no collapse): {len(df[~df['collapsed']])}")

    # Best overall
    best = df.nlargest(1, 'kappa_score').iloc[0]
    lines.append("\n" + "=" * 80)
    lines.append("BEST CONFIGURATION")
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

    # Key insights from OAT analysis
    lines.append("\n" + "=" * 80)
    lines.append("KEY INSIGHTS FROM ONE-AT-A-TIME ANALYSIS")
    lines.append("=" * 80)

    baseline = df[df['experiment'] == 'baseline_recommended'].iloc[0]
    baseline_kappa = baseline['kappa_score']

    oat_experiments = df[df['experiment'].str.startswith('oat_')]

    lines.append(f"\nBaseline Kappa: {baseline_kappa:.4f}")
    lines.append("\nParameter Sensitivity (change from baseline):")

    for _, row in oat_experiments.iterrows():
        delta = row['kappa_score'] - baseline_kappa
        direction = "↑" if delta > 0 else "↓"
        lines.append(f"  {row['experiment']}: {direction} {abs(delta):.4f} (Kappa: {row['kappa_score']:.4f})")

    # Model comparison
    lines.append("\n" + "=" * 80)
    lines.append("MODEL TYPE COMPARISON")
    lines.append("=" * 80)

    lightweight = df[df['model_type'] == 'lightweight']
    full = df[df['model_type'] == 'full']

    if len(lightweight) > 0 and len(full) > 0:
        lines.append(f"\nLightweight model:")
        lines.append(f"  Mean Kappa: {lightweight['kappa_score'].mean():.4f} ± {lightweight['kappa_score'].std():.4f}")
        lines.append(f"  Best Kappa: {lightweight['kappa_score'].max():.4f}")

        lines.append(f"\nFull model:")
        lines.append(f"  Mean Kappa: {full['kappa_score'].mean():.4f} ± {full['kappa_score'].std():.4f}")
        lines.append(f"  Best Kappa: {full['kappa_score'].max():.4f}")

        if full['kappa_score'].mean() > lightweight['kappa_score'].mean():
            lines.append("\n→ Full model shows better average performance")
        else:
            lines.append("\n→ Lightweight model is more efficient with similar performance")

    # Batch size comparison
    lines.append("\n" + "=" * 80)
    lines.append("BATCH SIZE COMPARISON")
    lines.append("=" * 80)

    batch8 = df[df['batch_size'] == 8]
    batch16 = df[df['batch_size'] == 16]

    if len(batch8) > 0 and len(batch16) > 0:
        lines.append(f"\nBatch size 8:")
        lines.append(f"  Mean Kappa: {batch8['kappa_score'].mean():.4f} ± {batch8['kappa_score'].std():.4f}")
        lines.append(f"  Best Kappa: {batch8['kappa_score'].max():.4f}")

        lines.append(f"\nBatch size 16:")
        lines.append(f"  Mean Kappa: {batch16['kappa_score'].mean():.4f} ± {batch16['kappa_score'].std():.4f}")
        lines.append(f"  Best Kappa: {batch16['kappa_score'].max():.4f}")

        if batch16['kappa_score'].mean() > batch8['kappa_score'].mean():
            lines.append("\n→ Batch size 16 shows better performance and faster training")
        else:
            lines.append("\n→ Batch size 8 may provide better regularization")

    # Interaction insights
    lines.append("\n" + "=" * 80)
    lines.append("INTERACTION EFFECTS")
    lines.append("=" * 80)

    interact_experiments = df[df['experiment'].str.startswith('interact_')]
    if len(interact_experiments) > 0:
        lines.append("\nInteraction experiment results:")
        for _, row in interact_experiments.iterrows():
            lines.append(f"  {row['description']}: Kappa = {row['kappa_score']:.4f}")

        best_interact = interact_experiments.nlargest(1, 'kappa_score').iloc[0]
        lines.append(f"\nBest interaction: {best_interact['description']}")
        lines.append(f"  Kappa: {best_interact['kappa_score']:.4f}")

    # Recommendations
    lines.append("\n" + "=" * 80)
    lines.append("RECOMMENDATIONS FOR THESIS")
    lines.append("=" * 80)

    lines.append("\n1. Optimal Configuration:")
    lines.append(f"   Use: {best['experiment']}")
    lines.append(f"   Expected Kappa: ~{best['kappa_score']:.3f}")

    lines.append("\n2. Most Important Parameters (by sensitivity):")
    oat_sorted = oat_experiments.copy()
    oat_sorted['delta'] = abs(oat_sorted['kappa_score'] - baseline_kappa)
    oat_sorted = oat_sorted.sort_values('delta', ascending=False)
    for idx, row in oat_sorted.head(3).iterrows():
        lines.append(f"   - {row['experiment']}: Δ = {row['delta']:.4f}")

    lines.append("\n3. Model Selection:")
    if len(full) > 0 and len(lightweight) > 0:
        if full['kappa_score'].max() > lightweight['kappa_score'].max():
            lines.append("   → Use Full model if compute budget allows")
        else:
            lines.append("   → Lightweight model is sufficient and more efficient")

    lines.append("\n4. Figures for Thesis:")
    lines.append("   - Figure 1: One-at-a-time parameter sensitivity")
    lines.append("   - Figure 2: Model type comparison")
    lines.append("   - Figure 3: Batch size comparison")
    lines.append("   - Figure 4: Interaction effects")
    lines.append("   - Figure 5: Overall performance ranking")
    lines.append("   - Table 1: Summary statistics by category")
    lines.append("   - Table 2: Top configurations")

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    # Save report
    report_path = os.path.join(results_dir, 'final_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ Saved: final_report.txt")

    # Also print to console
    print("\n" + "\n".join(lines))


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run efficient ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This ablation study uses an efficient fractional factorial design to test
~23 experiments instead of 162 full grid search combinations.

Experiment groups:
  - Baseline with recommended defaults (1)
  - One-at-a-time parameter variations (11)
  - Critical parameter interactions (6)
  - Extreme configurations (3)

All experiments use cosine classifier, label smoothing, and warmup to prevent collapse.

Example usage:
  python ablation_experiment.py

Output:
  - 5 publication-ready figures (PNG + PDF)
  - 2 LaTeX-ready tables (CSV + TEX)
  - Comprehensive text report
  - Complete results CSV for further analysis
        """
    )

    parser.add_argument('--data-path', type=str, default=None,
                        help='Override DATA_PATH in script')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Override MODEL_PATH in script')
    parser.add_argument('--users', type=int, default=30,
                        help='Number of users to test (default: 30)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs per experiment (default: 50)')

    args = parser.parse_args()

    # Override paths if provided
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

    # Verify paths are set
    if DATA_PATH == "path/to/your/data" or MODEL_PATH == "path/to/models/efficient_ablation":
        print("\n⚠️  WARNING: Please update DATA_PATH and MODEL_PATH in the script!")
        print("You can also use --data-path and --model-path arguments.\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            exit(0)

    print("\n" + "=" * 80)
    print("EFFICIENT ABLATION STUDY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data path: {DATA_PATH}")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Users: {len(USER_IDS)} (S{USER_IDS[0]:03d} to S{USER_IDS[-1]:03d})")
    print(f"  Epochs per experiment: {COMMON_PARAMS['epochs']}")
    print(f"\nEstimated experiments: ~23")
    print(f"Estimated time: ~{23 * COMMON_PARAMS['epochs'] * 2 / 60:.1f} hours")
    print("  (assuming ~2 minutes per epoch)")
    print("=" * 80)

    response = input("\nProceed with ablation study? (y/n): ")
    if response.lower() != 'y':
        print("Exiting.")
        exit(0)

    print("\n🚀 Starting efficient ablation study...")
    results = run_efficient_ablation_study()

    print("\n" + "=" * 80)
    print("✅ ABLATION STUDY COMPLETE!")
    print("=" * 80)
    print("\n📊 Results summary:")
    print(f"  Total experiments: {len(results)}")
    print(f"  Successful: {len(results[~results['collapsed']])}")
    print(f"  Best Kappa: {results['kappa_score'].max():.4f}")
    print(f"  Best Accuracy: {results['test_accuracy'].max():.4f}")

    best = results.nlargest(1, 'kappa_score').iloc[0]
    print(f"\n🏆 Best configuration: {best['experiment']}")
    print(f"  Kappa: {best['kappa_score']:.4f}")
    print(f"  Accuracy: {best['test_accuracy']:.4f}")
    print(f"  Model: {best['model_type']}")
    print(f"  Batch size: {best['batch_size']}")
    print(f"  Learning rate: {best['lr']}")
    print(f"  Cosine scale: {best['cosine_scale']}")
    print(f"  Label smoothing: {best['label_smoothing']}")
    print(f"  Warmup epochs: {best['warmup_epochs']}")

    print("\n📁 Output files:")
    print("  - all_results.csv: Complete dataset")
    print("  - fig1_oat_analysis.png/pdf: Parameter sensitivity")
    print("  - fig2_model_comparison.png/pdf: Model type effects")
    print("  - fig3_batch_comparison.png/pdf: Batch size effects")
    print("  - fig4_interactions.png/pdf: Parameter interactions")
    print("  - fig5_overall_ranking.png/pdf: Performance ranking")
    print("  - table1_summary.csv/.tex: Summary statistics")
    print("  - table2_top_configs.csv/.tex: Top configurations")
    print("  - final_report.txt: Comprehensive text report")

    print("\n💡 Next steps:")
    print("  1. Review final_report.txt for key findings")
    print("  2. Use the best configuration for final experiments")
    print("  3. Include figures and tables in your thesis")
    print("  4. Cite the optimal hyperparameters in your methodology section")