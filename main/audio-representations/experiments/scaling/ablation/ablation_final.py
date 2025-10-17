"""
Final Reproducible Ablation Study for Thesis

Based on reproducibility testing, this study uses:
- Constant LR after warmup (deterministic, reproducible)
- Random seeds for all experiments
- Reduced emphasis on label_smoothing=0.2 (high variance)
- Validation phase: Top 5 configs run with 3 seeds for mean±std

Total experiments: ~23 main + 15 validation = ~38 experiments
Estimated time: ~32 hours (38 experiments × 50 epochs × 2 min/epoch / 60)
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from trainers_cosine import spectrogram_trainer_2d
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

DATA_PATH = "/app/data/grouped_embeddings_full"
MODEL_PATH = "/app/data/experiments/ablation_final"
NORMALIZATION_METHOD = "log_scale"

USER_IDS = list(range(1, 31))  # 30 users

# Common parameters - NOW WITH REPRODUCIBILITY
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
    # NEW: For reproducibility
    'lr_scheduler_type': 'constant',  # Deterministic, same performance as plateau
    'random_seed': 42,  # Will be varied for validation runs
}

# Seeds for validation phase
VALIDATION_SEEDS = [42, 123, 456]


# =============================================
# EXPERIMENT DESIGN
# =============================================

def create_main_experiment_set():
    """
    Main experiments with single seed to explore hyperparameter space.

    Changes from original:
    - All use constant LR scheduler (reproducible)
    - All use random_seed=42
    - Reduced label_smoothing=0.2 experiments (unstable)
    """

    experiments = []

    # ========== GROUP 1: BASELINE ==========
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

    # Learning rate variations
    for lr in [0.0003, 0.003]:
        experiments.append({
            'name': f'oat_lr_{lr:.4f}'.replace('.', '_'),
            'description': f'OAT: Learning Rate = {lr}',
            'params': {**baseline['params'], 'lr': lr}
        })

    # Cosine scale variations
    for scale in [20.0, 64.0]:
        experiments.append({
            'name': f'oat_scale_{int(scale)}',
            'description': f'OAT: Cosine Scale = {scale}',
            'params': {**baseline['params'], 'cosine_scale': scale}
        })

    # Label smoothing variations - ONLY 0.1 now (0.2 showed high variance)
    experiments.append({
        'name': 'oat_smoothing_0_0',
        'description': 'OAT: Label Smoothing = 0.0',
        'params': {**baseline['params'], 'label_smoothing': 0.0}
    })

    # Warmup variations
    for warmup in [0, 10]:
        experiments.append({
            'name': f'oat_warmup_{warmup}',
            'description': f'OAT: Warmup Epochs = {warmup}',
            'params': {**baseline['params'], 'warmup_epochs': warmup}
        })

    # Batch size variations
    experiments.append({
        'name': 'oat_batch_8',
        'description': 'OAT: Batch Size = 8',
        'params': {**baseline['params'], 'batch_size': 8}
    })

    # Model type variations
    experiments.append({
        'name': 'oat_model_full',
        'description': 'OAT: Model Type = full',
        'params': {**baseline['params'], 'model_type': 'full'}
    })

    # ========== GROUP 3: CRITICAL INTERACTIONS ==========

    experiments.append({
        'name': 'interact_batch8_lr0003',
        'description': 'Batch 8 + LR 0.0003',
        'params': {**baseline['params'], 'batch_size': 8, 'lr': 0.0003}
    })

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

    # REMOVED: scale64_smooth02 (smoothing 0.2 unstable)

    experiments.append({
        'name': 'interact_nowarmup_lr0003',
        'description': 'No warmup + Low LR',
        'params': {**baseline['params'], 'warmup_epochs': 0, 'lr': 0.0003}
    })

    experiments.append({
        'name': 'interact_full_batch8',
        'description': 'Full model + Batch 8',
        'params': {**baseline['params'], 'model_type': 'full', 'batch_size': 8}
    })

    experiments.append({
        'name': 'interact_full_batch8_lr0003',
        'description': 'Full model + Batch 8 + LR 0.0003',
        'params': {**baseline['params'], 'model_type': 'full', 'batch_size': 8, 'lr': 0.0003}
    })

    # ========== GROUP 4: EXTREME CONFIGURATIONS ==========

    experiments.append({
        'name': 'extreme_conservative',
        'description': 'Conservative: Low LR, High smoothing, Long warmup',
        'params': {
            'lr': 0.0003,
            'batch_size': 16,
            'model_type': 'lightweight',
            'cosine_scale': 64.0,
            'label_smoothing': 0.1,  # Changed from 0.2 to 0.1 (more stable)
            'warmup_epochs': 10,
        }
    })

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

    # ========== GROUP 5: SCALE INTERACTIONS (NEW) ==========
    # Test scale with full model

    experiments.append({
        'name': 'interact_full_scale20',
        'description': 'Full model + Scale 20',
        'params': {**baseline['params'], 'model_type': 'full', 'cosine_scale': 20.0}
    })

    experiments.append({
        'name': 'interact_full_scale64',
        'description': 'Full model + Scale 64',
        'params': {**baseline['params'], 'model_type': 'full', 'cosine_scale': 64.0}
    })

    return experiments


def create_validation_experiments(main_results_df, top_n=5):
    """
    Create validation experiments for top N configurations.
    Each config is run with 3 different seeds to get mean±std.
    """

    # Get top N configs
    top_configs = main_results_df.nlargest(top_n, 'kappa_score')

    validation_experiments = []

    for idx, row in top_configs.iterrows():
        base_name = row['experiment']

        # Create 3 versions with different seeds
        for seed_idx, seed in enumerate(VALIDATION_SEEDS, 1):
            validation_experiments.append({
                'name': f'{base_name}_seed{seed}',
                'description': f'Validation: {row["description"]} (seed {seed})',
                'base_config': base_name,
                'params': {
                    'lr': row['lr'],
                    'batch_size': row['batch_size'],
                    'model_type': row['model_type'],
                    'cosine_scale': row['cosine_scale'],
                    'label_smoothing': row['label_smoothing'],
                    'warmup_epochs': row['warmup_epochs'],
                },
                'random_seed': seed,
                'is_validation': True
            })

    return validation_experiments


# =============================================
# EXPERIMENT RUNNER
# =============================================

def run_single_experiment(exp_name, exp_description, exp_params, results_dir, random_seed=None):
    """Run a single experiment and return results"""
    print(f"\n{'=' * 80}")
    print(f"Running: {exp_name}")
    print(f"Description: {exp_description}")
    print(f"Parameters: {exp_params}")
    if random_seed:
        print(f"Seed: {random_seed}")
    print('=' * 80)

    try:
        full_params = {**COMMON_PARAMS, **exp_params}
        full_params['model_path'] = os.path.join(results_dir, exp_name)

        # Override seed if specified
        if random_seed is not None:
            full_params['random_seed'] = random_seed

        test_acc, kappa_score = spectrogram_trainer_2d(**full_params)

        result = {
            'experiment': exp_name,
            'description': exp_description,
            'test_accuracy': test_acc,
            'kappa_score': kappa_score,
            'collapsed': kappa_score < 0.1,
            'random_seed': full_params['random_seed'],
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
            'random_seed': random_seed if random_seed else COMMON_PARAMS['random_seed'],
            **exp_params
        }


def run_final_ablation_study():
    """Run the final reproducible ablation study with validation"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(MODEL_PATH, f'final_ablation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # ========== PHASE 1: MAIN EXPERIMENTS ==========
    print("=" * 80)
    print("PHASE 1: MAIN EXPERIMENTS")
    print("=" * 80)

    main_experiments = create_main_experiment_set()

    print(f"Users: {len(USER_IDS)} (S{USER_IDS[0]:03d} to S{USER_IDS[-1]:03d})")
    print(f"Main experiments: {len(main_experiments)}")
    print(f"Using constant LR scheduler (reproducible)")
    print(f"Random seed: {COMMON_PARAMS['random_seed']}")
    print("=" * 80)

    main_results = []
    for idx, exp in enumerate(main_experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"MAIN EXPERIMENT {idx}/{len(main_experiments)}")
        print('=' * 80)

        result = run_single_experiment(
            exp['name'],
            exp['description'],
            exp['params'],
            results_dir
        )
        result['phase'] = 'main'
        main_results.append(result)

        # Save intermediate
        interim_df = pd.DataFrame(main_results)
        interim_df.to_csv(os.path.join(results_dir, 'main_results.csv'), index=False)

    main_df = pd.DataFrame(main_results)

    # ========== PHASE 2: VALIDATION EXPERIMENTS ==========
    print("\n" + "=" * 80)
    print("PHASE 2: VALIDATION OF TOP 5 CONFIGURATIONS")
    print("=" * 80)
    print(f"Running top 5 configs with {len(VALIDATION_SEEDS)} seeds each")
    print(f"Seeds: {VALIDATION_SEEDS}")
    print("=" * 80)

    validation_experiments = create_validation_experiments(main_df, top_n=5)

    validation_results = []
    for idx, exp in enumerate(validation_experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"VALIDATION {idx}/{len(validation_experiments)}")
        print('=' * 80)

        result = run_single_experiment(
            exp['name'],
            exp['description'],
            exp['params'],
            results_dir,
            random_seed=exp['random_seed']
        )
        result['phase'] = 'validation'
        result['base_config'] = exp['base_config']
        validation_results.append(result)

        # Save intermediate
        interim_df = pd.DataFrame(validation_results)
        interim_df.to_csv(os.path.join(results_dir, 'validation_results.csv'), index=False)

    validation_df = pd.DataFrame(validation_results)

    # ========== COMBINE AND ANALYZE ==========
    all_results = pd.concat([main_df, validation_df], ignore_index=True)
    all_results.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)

    # Generate analysis
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS")
    print("=" * 80)

    plot_oat_analysis(main_df, vis_dir)
    plot_model_comparison(main_df, vis_dir)
    plot_batch_size_comparison(main_df, vis_dir)
    plot_interaction_effects(main_df, vis_dir)
    plot_overall_ranking(main_df, vis_dir)
    plot_validation_results(validation_df, vis_dir)

    create_summary_table(main_df, vis_dir)
    create_top_configurations_table(main_df, vis_dir, top_n=10)
    create_validated_configs_table(validation_df, vis_dir)

    generate_final_report(main_df, validation_df, results_dir)

    return main_df, validation_df


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

        all_data = pd.concat([pd.DataFrame([baseline]), group])
        x_pos = np.arange(len(all_data))
        bars = ax.bar(x_pos, all_data['kappa_score'], alpha=0.7)

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

    lightweight = df[df['model_type'] == 'lightweight']
    full = df[df['model_type'] == 'full']

    if len(lightweight) > 0 and len(full) > 0:
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

    batch8 = df[df['batch_size'] == 8]
    batch16 = df[df['batch_size'] == 16]

    if len(batch8) > 0 and len(batch16) > 0:
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

    interact_sorted = interact_experiments.sort_values('kappa_score', ascending=True)
    colors = plt.cm.viridis(interact_sorted['kappa_score'] / interact_sorted['kappa_score'].max())
    bars = ax.barh(range(len(interact_sorted)), interact_sorted['kappa_score'], color=colors)

    ax.set_yticks(range(len(interact_sorted)))
    ax.set_yticklabels(interact_sorted['description'], fontsize=9)
    ax.set_xlabel('Kappa Score')
    ax.set_title('Interaction Effects: Combined Parameters')
    ax.grid(axis='x', alpha=0.3)

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

    successful = df[~df['collapsed']].sort_values('kappa_score', ascending=True)

    if len(successful) == 0:
        print("⚠ No successful experiments to plot")
        return

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
    ax.set_title('Overall Performance Ranking (Main Experiments)')
    ax.grid(axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Baseline'),
        Patch(facecolor='blue', alpha=0.7, label='One-at-a-time'),
        Patch(facecolor='orange', alpha=0.7, label='Interactions'),
        Patch(facecolor='red', alpha=0.7, label='Extreme configs')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

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


def plot_validation_results(validation_df, save_dir):
    """Figure 6: Validation results with error bars"""
    if len(validation_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by base config
    base_configs = validation_df['base_config'].unique()

    means = []
    stds = []
    labels = []

    for config in base_configs:
        config_data = validation_df[validation_df['base_config'] == config]
        means.append(config_data['kappa_score'].mean())
        stds.append(config_data['kappa_score'].std())
        labels.append(config.replace('_', '\n'))

    # Sort by mean
    sorted_indices = np.argsort(means)
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, means, xerr=stds, capsize=5, alpha=0.7, color='steelblue')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Kappa Score (Mean ± Std across 3 seeds)')
    ax.set_title('Validated Top 5 Configurations')
    ax.grid(axis='x', alpha=0.3)

    # Annotate values
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean, i, f"  {mean:.4f}±{std:.4f}", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_validation.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'fig6_validation.pdf'), bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: fig6_validation.png/pdf")


# =============================================
# TABLE FUNCTIONS
# =============================================

def create_summary_table(df, save_dir):
    """Table 1: Summary statistics"""
    summary_data = []

    summary_data.append({
        'Category': 'All Experiments',
        'Count': len(df),
        'Mean Kappa': df['kappa_score'].mean(),
        'Std Kappa': df['kappa_score'].std(),
        'Max Kappa': df['kappa_score'].max(),
        'Mean Accuracy': df['test_accuracy'].mean(),
        'Max Accuracy': df['test_accuracy'].max(),
    })

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
    """Table 2: Top configurations from main experiments"""
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


def create_validated_configs_table(validation_df, save_dir):
    """Table 3: Validated configurations with mean±std"""
    if len(validation_df) == 0:
        return

    validated_data = []

    for config in validation_df['base_config'].unique():
        config_data = validation_df[validation_df['base_config'] == config]

        validated_data.append({
            'Configuration': config,
            'Mean Kappa': config_data['kappa_score'].mean(),
            'Std Kappa': config_data['kappa_score'].std(),
            'Mean Accuracy': config_data['test_accuracy'].mean(),
            'Std Accuracy': config_data['test_accuracy'].std(),
            'Model': config_data.iloc[0]['model_type'],
            'Batch Size': config_data.iloc[0]['batch_size'],
            'LR': config_data.iloc[0]['lr'],
            'Scale': config_data.iloc[0]['cosine_scale'],
            'Smoothing': config_data.iloc[0]['label_smoothing'],
            'Warmup': config_data.iloc[0]['warmup_epochs'],
        })

    validated_df = pd.DataFrame(validated_data)
    validated_df = validated_df.sort_values('Mean Kappa', ascending=False)
    validated_df.to_csv(os.path.join(save_dir, 'table3_validated.csv'), index=False)

    latex_table = validated_df.to_latex(index=False, float_format="%.4f")
    with open(os.path.join(save_dir, 'table3_validated.tex'), 'w') as f:
        f.write(latex_table)

    print(f"✓ Saved: table3_validated.csv/tex")
    return validated_df


# =============================================
# REPORT GENERATION
# =============================================

def generate_final_report(main_df, validation_df, results_dir):
    """Generate comprehensive final report"""
    lines = []
    lines.append("=" * 80)
    lines.append("FINAL REPRODUCIBLE ABLATION STUDY - REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Main experiments: {len(main_df)}")
    lines.append(f"Validation runs: {len(validation_df)}")
    lines.append(f"Total experiments: {len(main_df) + len(validation_df)}")
    lines.append(f"\nReproducibility measures:")
    lines.append(f"  - Random seed: {COMMON_PARAMS['random_seed']} (main phase)")
    lines.append(f"  - Validation seeds: {VALIDATION_SEEDS}")
    lines.append(f"  - LR scheduler: constant (deterministic)")

    # Best from main experiments
    best_main = main_df.nlargest(1, 'kappa_score').iloc[0]
    lines.append("\n" + "=" * 80)
    lines.append("BEST CONFIGURATION (Main Experiments)")
    lines.append("=" * 80)
    lines.append(f"Experiment: {best_main['experiment']}")
    lines.append(f"Kappa Score: {best_main['kappa_score']:.4f}")
    lines.append(f"Test Accuracy: {best_main['test_accuracy']:.4f}")
    lines.append(f"\nHyperparameters:")
    lines.append(f"  Model Type: {best_main['model_type']}")
    lines.append(f"  Batch Size: {best_main['batch_size']}")
    lines.append(f"  Learning Rate: {best_main['lr']}")
    lines.append(f"  Cosine Scale: {best_main['cosine_scale']}")
    lines.append(f"  Label Smoothing: {best_main['label_smoothing']}")
    lines.append(f"  Warmup Epochs: {best_main['warmup_epochs']}")

    # Validated results
    lines.append("\n" + "=" * 80)
    lines.append("VALIDATED TOP 5 CONFIGURATIONS (Mean ± Std)")
    lines.append("=" * 80)

    for config in validation_df['base_config'].unique():
        config_data = validation_df[validation_df['base_config'] == config]
        mean_kappa = config_data['kappa_score'].mean()
        std_kappa = config_data['kappa_score'].std()

        lines.append(f"\n{config}:")
        lines.append(f"  Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")
        lines.append(
            f"  Accuracy: {config_data['test_accuracy'].mean():.4f} ± {config_data['test_accuracy'].std():.4f}")
        lines.append(f"  Reproducibility: {'GOOD' if std_kappa < 0.02 else 'MODERATE' if std_kappa < 0.05 else 'POOR'}")

    # Key insights
    lines.append("\n" + "=" * 80)
    lines.append("KEY INSIGHTS")
    lines.append("=" * 80)

    baseline = main_df[main_df['experiment'] == 'baseline_recommended'].iloc[0]
    oat_experiments = main_df[main_df['experiment'].str.startswith('oat_')]

    lines.append(f"\nBaseline Kappa: {baseline['kappa_score']:.4f}")
    lines.append("\nMost impactful parameters (OAT analysis):")

    oat_sorted = oat_experiments.copy()
    oat_sorted['delta'] = abs(oat_sorted['kappa_score'] - baseline['kappa_score'])
    oat_sorted = oat_sorted.sort_values('delta', ascending=False)

    for idx, row in oat_sorted.head(5).iterrows():
        delta = row['kappa_score'] - baseline['kappa_score']
        direction = "↑" if delta > 0 else "↓"
        lines.append(f"  {row['experiment']}: {direction} {abs(delta):.4f}")

    # Model and batch comparisons
    lines.append("\n" + "=" * 80)
    lines.append("MODEL AND BATCH SIZE ANALYSIS")
    lines.append("=" * 80)

    lightweight = main_df[main_df['model_type'] == 'lightweight']
    full = main_df[main_df['model_type'] == 'full']

    if len(lightweight) > 0 and len(full) > 0:
        lines.append(f"\nModel comparison:")
        lines.append(f"  Lightweight: {lightweight['kappa_score'].mean():.4f} (mean)")
        lines.append(f"  Full: {full['kappa_score'].mean():.4f} (mean)")
        lines.append(
            f"  Conclusion: {'Full model is better' if full['kappa_score'].mean() > lightweight['kappa_score'].mean() else 'Lightweight is sufficient'}")

    batch8 = main_df[main_df['batch_size'] == 8]
    batch16 = main_df[main_df['batch_size'] == 16]

    if len(batch8) > 0 and len(batch16) > 0:
        lines.append(f"\nBatch size comparison:")
        lines.append(f"  Batch 8: {batch8['kappa_score'].mean():.4f} (mean)")
        lines.append(f"  Batch 16: {batch16['kappa_score'].mean():.4f} (mean)")

    # Thesis recommendations
    lines.append("\n" + "=" * 80)
    lines.append("RECOMMENDATIONS FOR THESIS")
    lines.append("=" * 80)

    # Find best validated config
    best_validated = None
    best_validated_mean = 0
    best_validated_std = 1

    for config in validation_df['base_config'].unique():
        config_data = validation_df[validation_df['base_config'] == config]
        mean = config_data['kappa_score'].mean()
        std = config_data['kappa_score'].std()

        if mean > best_validated_mean:
            best_validated = config
            best_validated_mean = mean
            best_validated_std = std

    lines.append("\n1. RECOMMENDED CONFIGURATION FOR THESIS:")
    lines.append(f"   Configuration: {best_validated}")
    lines.append(f"   Performance: {best_validated_mean:.4f} ± {best_validated_std:.4f} Kappa")
    lines.append(f"   Reproducibility: {best_validated_std:.4f} std (across 3 seeds)")

    # Get params
    best_config_row = validation_df[validation_df['base_config'] == best_validated].iloc[0]
    lines.append(f"\n   Hyperparameters:")
    lines.append(f"     - Model: {best_config_row['model_type']}")
    lines.append(f"     - Batch size: {best_config_row['batch_size']}")
    lines.append(
        f"     - Learning rate: {best_config_row['lr']} (constant after {best_config_row['warmup_epochs']}-epoch warmup)")
    lines.append(f"     - Cosine scale: {best_config_row['cosine_scale']}")
    lines.append(f"     - Label smoothing: {best_config_row['label_smoothing']}")
    lines.append(f"     - Warmup: {best_config_row['warmup_epochs']} epochs")

    lines.append("\n2. HOW TO REPORT IN THESIS:")
    lines.append(f'   "We achieved {best_validated_mean:.4f} ± {best_validated_std:.4f} Cohen\'s Kappa')
    lines.append(f'    (mean ± std across 3 random seeds) using the following configuration..."')

    lines.append("\n3. FIGURES TO INCLUDE:")
    lines.append("   - Figure 1: OAT parameter sensitivity")
    lines.append("   - Figure 2: Model type comparison")
    lines.append("   - Figure 5: Overall ranking")
    lines.append("   - Figure 6: Validated results with error bars (MOST IMPORTANT)")
    lines.append("   - Table 3: Validated configurations")

    lines.append("\n4. KEY METHODOLOGICAL NOTES:")
    lines.append("   - All experiments use random seeds for reproducibility")
    lines.append("   - Constant LR after warmup (deterministic)")
    lines.append("   - Label smoothing 0.2 excluded (high variance in testing)")
    lines.append("   - Results validated with 3 independent seeds")

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    # Save
    report_path = os.path.join(results_dir, 'final_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ Saved: final_report.txt")
    print("\n" + "\n".join(lines))


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Run final reproducible ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Final reproducible ablation study with validation phase.

Changes from original:
  - Uses constant LR scheduler (reproducible)
  - Sets random seeds for all experiments
  - Validates top 5 configs with 3 seeds each
  - Reports mean ± std for validated configs

Total experiments: ~23 main + 15 validation = ~38
Estimated time: ~32 hours
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

    print("\n" + "=" * 80)
    print("FINAL REPRODUCIBLE ABLATION STUDY")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data path: {DATA_PATH}")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Users: {len(USER_IDS)}")
    print(f"  Epochs: {COMMON_PARAMS['epochs']}")
    print(f"  Scheduler: {COMMON_PARAMS['lr_scheduler_type']}")
    print(f"  Main seed: {COMMON_PARAMS['random_seed']}")
    print(f"  Validation seeds: {VALIDATION_SEEDS}")
    print(f"\nPhase 1: ~23 main experiments")
    print(f"Phase 2: ~15 validation experiments (top 5 × 3 seeds)")
    print(f"Total: ~38 experiments")
    print(f"Estimated time: ~{38 * COMMON_PARAMS['epochs'] * 2 / 60:.1f} hours")
    print("=" * 80)

    print("\n🚀 Starting final ablation study...")
    main_results, validation_results = run_final_ablation_study()

    print("\n" + "=" * 80)
    print("✅ FINAL ABLATION STUDY COMPLETE!")
    print("=" * 80)

    # Final summary
    best_validated = None
    best_mean = 0

    for config in validation_results['base_config'].unique():
        config_data = validation_results[validation_results['base_config'] == config]
        mean = config_data['kappa_score'].mean()
        if mean > best_mean:
            best_validated = config
            best_mean = mean

    if best_validated:
        config_data = validation_results[validation_results['base_config'] == best_validated]
        print(f"\n🏆 BEST VALIDATED CONFIGURATION:")
        print(f"   {best_validated}")
        print(f"   Kappa: {config_data['kappa_score'].mean():.4f} ± {config_data['kappa_score'].std():.4f}")
        print(f"   (Mean ± Std across {len(VALIDATION_SEEDS)} seeds)")

    print("\n📊 Output files:")
    print("  - final_report.txt: Complete analysis")
    print("  - fig6_validation.png: Validated results (USE THIS IN THESIS!)")
    print("  - table3_validated.csv: Mean±std for top configs")
    print("  - all_results.csv: Complete dataset")