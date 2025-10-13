"""
Reproducibility Testing Script - Scheduler Comparison

Tests the most critical configurations from both ablation studies
with TWO different learning rate schedulers to identify the cause
of non-reproducible results.

Scheduler types tested:
1. ReduceLROnPlateau (current setup) - metric-based, non-deterministic
2. Constant LR after warmup - deterministic, reproducible

Critical configs:
1. Initial best (Kappa 0.8339)
2. Follow-up best (Kappa 0.8135)

Each config × scheduler type is run 3 times with different seeds.
Total: 4 configs (2 base × 2 schedulers) × 3 seeds = 12 experiments
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from trainers_cosine import spectrogram_trainer_2d

# Set publication-quality plot defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# =============================================
# CONFIGURATION
# =============================================

DATA_PATH = "/app/data/grouped_embeddings_full"  # UPDATE THIS
MODEL_PATH = "/app/data/experiments/ablation_reproducibility"  # UPDATE THIS
NORMALIZATION_METHOD = "log_scale"

USER_IDS = list(range(1, 31))  # 30 users

# Common parameters for all experiments
COMMON_PARAMS = {
    'data_path': DATA_PATH,
    'model_path': MODEL_PATH,
    'user_ids': USER_IDS,
    'normalization_method': NORMALIZATION_METHOD,
    'epochs': 50,
    'use_augmentation': True,
    'device': 'cuda',
    'save_model_checkpoints': True,  # MUST be True for logger to work
    'max_cache_size': 50,
    'use_cosine_classifier': True,
}

# Seeds for reproducibility testing
TEST_SEEDS = [42, 123, 456]


# =============================================
# CRITICAL CONFIGURATIONS TO TEST
# =============================================

def get_critical_configs():
    """
    Define the most critical configurations to test for reproducibility.

    Tests each config with TWO scheduler types:
    1. ReduceLROnPlateau (current, non-deterministic)
    2. Constant LR after warmup (deterministic)
    """

    base_configs = [
        {
            'name': 'initial_best',
            'description': 'Initial ablation best (interact_full_lr0003)',
            'expected_kappa': 0.8339,
            'params': {
                'model_type': 'full',
                'batch_size': 16,
                'lr': 0.0003,
                'cosine_scale': 40.0,
                'label_smoothing': 0.1,
                'warmup_epochs': 5,
            }
        },
        {
            'name': 'followup_best',
            'description': 'Follow-up best (current_best_smooth02)',
            'expected_kappa': 0.8135,
            'params': {
                'model_type': 'full',
                'batch_size': 16,
                'lr': 0.0003,
                'cosine_scale': 40.0,
                'label_smoothing': 0.2,
                'warmup_epochs': 5,
            }
        }
    ]

    # Create configs for both scheduler types
    configs = []

    for base_config in base_configs:
        # Version 1: With ReduceLROnPlateau (current setup)
        configs.append({
            'name': f"{base_config['name']}_plateau",
            'description': f"{base_config['description']} + ReduceLROnPlateau",
            'expected_kappa': base_config['expected_kappa'],
            'params': {
                **base_config['params'],
                'lr_scheduler_type': 'plateau'
            }
        })

        # Version 2: Constant LR after warmup (deterministic)
        configs.append({
            'name': f"{base_config['name']}_constant",
            'description': f"{base_config['description']} + Constant LR",
            'expected_kappa': None,  # We don't know what to expect yet
            'params': {
                **base_config['params'],
                'lr_scheduler_type': 'constant'
            }
        })

    return configs


# =============================================
# REPRODUCIBILITY TESTING
# =============================================

def run_single_test(config_name, config_desc, config_params, seed, run_number, results_dir):
    """Run a single test with a specific seed"""
    print(f"\n{'=' * 80}")
    print(f"Testing: {config_name} (Run {run_number}/3, Seed {seed})")
    print(f"Description: {config_desc}")
    print('=' * 80)

    try:
        # Combine params with seed
        full_params = {**COMMON_PARAMS, **config_params, 'random_seed': seed}
        full_params['model_path'] = os.path.join(results_dir, f'{config_name}_seed{seed}')

        # Run training
        test_acc, kappa_score = spectrogram_trainer_2d(**full_params)

        result = {
            'config_name': config_name,
            'config_description': config_desc,
            'seed': seed,
            'run_number': run_number,
            'test_accuracy': test_acc,
            'kappa_score': kappa_score,
            'collapsed': kappa_score < 0.1,
            **config_params
        }

        print(f"\n✓ COMPLETED: Accuracy = {test_acc:.4f}, Kappa = {kappa_score:.4f}")

        return result

    except Exception as e:
        print(f"\n✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'config_name': config_name,
            'config_description': config_desc,
            'seed': seed,
            'run_number': run_number,
            'test_accuracy': 0.0,
            'kappa_score': 0.0,
            'collapsed': True,
            'error': str(e),
            **config_params
        }


def run_reproducibility_tests():
    """Run reproducibility tests on critical configurations"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(MODEL_PATH, f'reproducibility_test_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    vis_dir = os.path.join(results_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    configs = get_critical_configs()

    print("=" * 80)
    print("REPRODUCIBILITY TESTING - SCHEDULER COMPARISON")
    print("=" * 80)
    print(f"\nTesting {len(configs)} configurations (2 base configs × 2 scheduler types)")
    print(f"Each config will be run {len(TEST_SEEDS)} times with different seeds")
    print(f"Total experiments: {len(configs) * len(TEST_SEEDS)}")
    print(f"\nScheduler types:")
    print(f"  1. ReduceLROnPlateau (current setup, non-deterministic)")
    print(f"  2. Constant LR after warmup (deterministic)")
    print(f"\nResults directory: {results_dir}")
    print("=" * 80)

    # Run all tests
    all_results = []

    for config_idx, config in enumerate(configs, 1):
        print(f"\n{'#' * 80}")
        print(f"# CONFIGURATION {config_idx}/{len(configs)}: {config['name']}")
        if config['expected_kappa'] is not None:
            print(f"# Expected Kappa: {config['expected_kappa']:.4f}")
        else:
            print(f"# Expected Kappa: Unknown (new scheduler type)")
        print(f"{'#' * 80}")

        config_results = []

        for run_num, seed in enumerate(TEST_SEEDS, 1):
            result = run_single_test(
                config['name'],
                config['description'],
                config['params'],
                seed,
                run_num,
                results_dir
            )
            result['expected_kappa'] = config['expected_kappa']
            config_results.append(result)
            all_results.append(result)

            # Save intermediate results
            interim_df = pd.DataFrame(all_results)
            interim_df.to_csv(os.path.join(results_dir, 'interim_results.csv'), index=False)

        # Report on this config
        kappas = [r['kappa_score'] for r in config_results if not r['collapsed']]
        if len(kappas) > 0:
            print(f"\n{'=' * 80}")
            print(f"SUMMARY FOR {config['name']}:")
            if config['expected_kappa'] is not None:
                print(f"  Expected Kappa: {config['expected_kappa']:.4f}")
            else:
                print(f"  Expected Kappa: Unknown (new scheduler type)")
            print(f"  Mean Kappa: {np.mean(kappas):.4f}")
            print(f"  Std Kappa: {np.std(kappas):.4f}")
            print(f"  Range: [{np.min(kappas):.4f}, {np.max(kappas):.4f}]")
            if config['expected_kappa'] is not None:
                print(f"  Deviation from expected: {np.mean(kappas) - config['expected_kappa']:+.4f}")
            print('=' * 80)

    # Create final dataframe
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)

    # Generate analysis
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS")
    print("=" * 80)

    analyze_reproducibility(results_df, vis_dir, results_dir)

    return results_df


# =============================================
# ANALYSIS FUNCTIONS
# =============================================

def analyze_reproducibility(df, vis_dir, results_dir):
    """Analyze reproducibility and generate report"""

    # Calculate statistics for each config
    summary_data = []

    for config_name in df['config_name'].unique():
        config_df = df[df['config_name'] == config_name]
        successful = config_df[~config_df['collapsed']]

        if len(successful) > 0:
            expected = config_df['expected_kappa'].iloc[0]
            mean_kappa = successful['kappa_score'].mean()
            std_kappa = successful['kappa_score'].std()
            min_kappa = successful['kappa_score'].min()
            max_kappa = successful['kappa_score'].max()
            range_kappa = max_kappa - min_kappa

            # Deviation only if expected is known
            if expected is not None and not np.isnan(expected):
                deviation = mean_kappa - expected
            else:
                deviation = 0.0

            # Reproducibility score: lower is better
            # For configs without expected value, only consider std
            reproducibility_score = std_kappa + abs(deviation) if deviation != 0 else std_kappa

            # Determine scheduler type from config name
            scheduler_type = 'plateau' if '_plateau' in config_name else 'constant'

            summary_data.append({
                'Config': config_name,
                'Scheduler': scheduler_type,
                'Expected': expected if expected is not None else np.nan,
                'Mean': mean_kappa,
                'Std': std_kappa,
                'Min': min_kappa,
                'Max': max_kappa,
                'Range': range_kappa,
                'Deviation': deviation,
                'Reproducibility Score': reproducibility_score,
                'Status': 'GOOD' if std_kappa < 0.02 else 'NEEDS ATTENTION'
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Mean', ascending=False)

    # Save summary
    summary_df.to_csv(os.path.join(results_dir, 'reproducibility_summary.csv'), index=False)

    # Generate visualizations
    plot_reproducibility_bars(df, vis_dir)
    plot_scheduler_comparison(summary_df, vis_dir)
    plot_variance_analysis(summary_df, vis_dir)
    plot_deviation_analysis(summary_df, vis_dir)

    # Generate report
    generate_reproducibility_report(summary_df, df, results_dir)


def plot_reproducibility_bars(df, save_dir):
    """Plot kappa scores with error bars for each config"""
    fig, ax = plt.subplots(figsize=(12, 6))

    configs = df['config_name'].unique()
    x_pos = np.arange(len(configs))

    means = []
    stds = []
    expected_vals = []
    has_expected = []

    for config in configs:
        config_df = df[df['config_name'] == config]
        successful = config_df[~config_df['collapsed']]
        means.append(successful['kappa_score'].mean())
        stds.append(successful['kappa_score'].std())

        expected = config_df['expected_kappa'].iloc[0]
        if expected is not None and not pd.isna(expected):
            expected_vals.append(expected)
            has_expected.append(True)
        else:
            expected_vals.append(0)  # Placeholder, won't be plotted
            has_expected.append(False)

    # Plot bars with error bars
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7,
                  color='steelblue', label='Measured (Mean ± Std)')

    # Plot expected values only where they exist
    expected_x = [x for x, has_exp in zip(x_pos, has_expected) if has_exp]
    expected_y = [y for y, has_exp in zip(expected_vals, has_expected) if has_exp]

    if len(expected_x) > 0:
        ax.scatter(expected_x, expected_y, color='red', s=100, marker='*',
                   zorder=10, label='Expected from Original Study')
        ax.plot(expected_x, expected_y, 'r--', alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Kappa Score')
    ax.set_title('Reproducibility Test: Measured vs Expected Performance')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Annotate deviations only where expected exists
    for i, (mean, expected, has_exp) in enumerate(zip(means, expected_vals, has_expected)):
        if has_exp:
            deviation = mean - expected
            color = 'green' if abs(deviation) < 0.05 else 'red'
            ax.text(i, max(mean, expected) + 0.02, f'{deviation:+.3f}',
                    ha='center', va='bottom', color=color, fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reproducibility_bars.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'reproducibility_bars.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved: reproducibility_bars.png/pdf")


def plot_scheduler_comparison(summary_df, save_dir):
    """Compare plateau vs constant scheduler directly"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract base config names (without scheduler suffix)
    summary_df['base_config'] = summary_df['Config'].str.replace('_plateau', '').str.replace('_constant', '')

    # Group by base config
    base_configs = summary_df['base_config'].unique()

    # Plot 1: Mean Kappa comparison
    x_pos = np.arange(len(base_configs))
    width = 0.35

    plateau_means = []
    constant_means = []
    plateau_stds = []
    constant_stds = []

    for base_config in base_configs:
        plateau_row = summary_df[(summary_df['base_config'] == base_config) &
                                 (summary_df['Scheduler'] == 'plateau')]
        constant_row = summary_df[(summary_df['base_config'] == base_config) &
                                  (summary_df['Scheduler'] == 'constant')]

        if len(plateau_row) > 0:
            plateau_means.append(plateau_row.iloc[0]['Mean'])
            plateau_stds.append(plateau_row.iloc[0]['Std'])
        else:
            plateau_means.append(0)
            plateau_stds.append(0)

        if len(constant_row) > 0:
            constant_means.append(constant_row.iloc[0]['Mean'])
            constant_stds.append(constant_row.iloc[0]['Std'])
        else:
            constant_means.append(0)
            constant_stds.append(0)

    ax1.bar(x_pos - width / 2, plateau_means, width, yerr=plateau_stds,
            label='ReduceLROnPlateau', alpha=0.8, capsize=5, color='coral')
    ax1.bar(x_pos + width / 2, constant_means, width, yerr=constant_stds,
            label='Constant LR', alpha=0.8, capsize=5, color='steelblue')

    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Mean Kappa Score')
    ax1.set_title('Scheduler Comparison: Mean Performance')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([c.replace('_', '\n') for c in base_configs], fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Variance comparison
    ax2.bar(x_pos - width / 2, plateau_stds, width,
            label='ReduceLROnPlateau', alpha=0.8, color='coral')
    ax2.bar(x_pos + width / 2, constant_stds, width,
            label='Constant LR', alpha=0.8, color='steelblue')

    ax2.axhline(y=0.02, color='green', linestyle='--', label='Good threshold (0.02)', alpha=0.5)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Scheduler Comparison: Reproducibility (Lower is Better)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([c.replace('_', '\n') for c in base_configs], fontsize=9)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scheduler_comparison.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'scheduler_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved: scheduler_comparison.png/pdf")


def plot_variance_analysis(summary_df, save_dir):
    """Plot variance analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Standard deviation
    colors = ['green' if s < 0.02 else 'orange' if s < 0.05 else 'red'
              for s in summary_df['Std']]
    ax1.barh(summary_df['Config'], summary_df['Std'], color=colors, alpha=0.7)
    ax1.axvline(x=0.02, color='green', linestyle='--', label='Good (< 0.02)')
    ax1.axvline(x=0.05, color='orange', linestyle='--', label='Acceptable (< 0.05)')
    ax1.set_xlabel('Standard Deviation (Kappa)')
    ax1.set_title('Variance Across Seeds')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Range
    ax2.barh(summary_df['Config'], summary_df['Range'], color='steelblue', alpha=0.7)
    ax2.set_xlabel('Range (Max - Min)')
    ax2.set_title('Performance Range')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'variance_analysis.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'variance_analysis.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved: variance_analysis.png/pdf")


def plot_deviation_analysis(summary_df, save_dir):
    """Plot deviation from expected values"""
    # Only plot configs that have expected values
    plot_df = summary_df[summary_df['Expected'].notna()].copy()

    if len(plot_df) == 0:
        print("⚠ No configs with expected values to plot deviation")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if abs(d) < 0.05 else 'orange' if abs(d) < 0.10 else 'red'
              for d in plot_df['Deviation']]

    bars = ax.barh(plot_df['Config'], plot_df['Deviation'], color=colors, alpha=0.7)

    ax.axvline(x=0, color='black', linewidth=2)
    ax.axvline(x=-0.05, color='green', linestyle='--', alpha=0.5, label='±5% threshold')
    ax.axvline(x=0.05, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.10, color='orange', linestyle='--', alpha=0.5, label='±10% threshold')
    ax.axvline(x=0.10, color='orange', linestyle='--', alpha=0.5)

    ax.set_xlabel('Deviation from Expected (Measured - Expected)')
    ax.set_title('Accuracy: Measured vs Original Study (ReduceLROnPlateau only)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Annotate values
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        ax.text(row['Deviation'], i, f" {row['Deviation']:+.3f}",
                va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'deviation_analysis.png'), bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'deviation_analysis.pdf'), bbox_inches='tight')
    plt.close()
    print("✓ Saved: deviation_analysis.png/pdf")


def generate_reproducibility_report(summary_df, full_df, results_dir):
    """Generate comprehensive reproducibility report"""
    lines = []
    lines.append("=" * 80)
    lines.append("REPRODUCIBILITY TESTING REPORT - SCHEDULER COMPARISON")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Configurations tested: {len(summary_df)}")
    lines.append(f"Seeds used: {TEST_SEEDS}")
    lines.append(f"\nScheduler types tested:")
    lines.append(f"  - ReduceLROnPlateau (metric-based, non-deterministic)")
    lines.append(f"  - Constant LR after warmup (deterministic)")

    # Scheduler comparison
    lines.append("\n" + "=" * 80)
    lines.append("SCHEDULER COMPARISON")
    lines.append("=" * 80)

    plateau_configs = summary_df[summary_df['Scheduler'] == 'plateau']
    constant_configs = summary_df[summary_df['Scheduler'] == 'constant']

    if len(plateau_configs) > 0 and len(constant_configs) > 0:
        lines.append(f"\nReduceLROnPlateau scheduler:")
        lines.append(f"  Mean Kappa: {plateau_configs['Mean'].mean():.4f}")
        lines.append(f"  Mean Std: {plateau_configs['Std'].mean():.4f}")
        lines.append(
            f"  Reproducible configs: {len(plateau_configs[plateau_configs['Status'] == 'GOOD'])}/{len(plateau_configs)}")

        lines.append(f"\nConstant LR scheduler:")
        lines.append(f"  Mean Kappa: {constant_configs['Mean'].mean():.4f}")
        lines.append(f"  Mean Std: {constant_configs['Std'].mean():.4f}")
        lines.append(
            f"  Reproducible configs: {len(constant_configs[constant_configs['Status'] == 'GOOD'])}/{len(constant_configs)}")

        # Determine which is better
        if constant_configs['Std'].mean() < plateau_configs['Std'].mean():
            improvement = (plateau_configs['Std'].mean() - constant_configs['Std'].mean()) / plateau_configs[
                'Std'].mean() * 100
            lines.append(f"\n✅ CONSTANT LR is {improvement:.1f}% more reproducible")
            lines.append(f"   (Lower variance across seeds)")
        else:
            lines.append(f"\n⚠️  ReduceLROnPlateau shows better reproducibility (unexpected)")

        # Performance comparison
        perf_diff = constant_configs['Mean'].mean() - plateau_configs['Mean'].mean()
        if abs(perf_diff) < 0.01:
            lines.append(f"\n   Performance is similar (diff: {perf_diff:+.4f})")
        elif perf_diff > 0:
            lines.append(f"\n   Constant LR also performs better (+{perf_diff:.4f})")
        else:
            lines.append(f"\n   Trade-off: Plateau performs better (+{abs(perf_diff):.4f}), but less reproducible")

    # Overall assessment
    lines.append("\n" + "=" * 80)
    lines.append("OVERALL ASSESSMENT")
    lines.append("=" * 80)

    good_configs = summary_df[summary_df['Status'] == 'GOOD']
    bad_configs = summary_df[summary_df['Status'] == 'NEEDS ATTENTION']

    lines.append(f"\nConfigurations with GOOD reproducibility: {len(good_configs)}/{len(summary_df)}")
    lines.append(f"Configurations NEEDING ATTENTION: {len(bad_configs)}/{len(summary_df)}")

    if len(good_configs) == len(summary_df):
        lines.append("\n✅ ALL CONFIGURATIONS ARE REPRODUCIBLE!")
        lines.append("   Your training pipeline is stable and reliable.")
    elif len(bad_configs) > 0:
        lines.append("\n⚠️  REPRODUCIBILITY ISSUES DETECTED")
        lines.append("   Some configurations show high variance or deviation from expected.")

    # Detailed results
    lines.append("\n" + "=" * 80)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 80)

    for _, row in summary_df.iterrows():
        lines.append(f"\n{row['Config'].upper()}")
        lines.append(f"  Status: {row['Status']}")
        lines.append(f"  Scheduler: {row['Scheduler']}")

        if not pd.isna(row['Expected']):
            lines.append(f"  Expected Kappa: {row['Expected']:.4f}")
            lines.append(f"  Measured Kappa: {row['Mean']:.4f} ± {row['Std']:.4f}")
            lines.append(f"  Deviation: {row['Deviation']:+.4f} ({row['Deviation'] / row['Expected'] * 100:+.1f}%)")
        else:
            lines.append(f"  Expected Kappa: Unknown (new scheduler type)")
            lines.append(f"  Measured Kappa: {row['Mean']:.4f} ± {row['Std']:.4f}")

        lines.append(f"  Range: [{row['Min']:.4f}, {row['Max']:.4f}]")

        if row['Status'] == 'GOOD':
            lines.append(f"  ✅ Reproducible - Low variance, close to expected")
        else:
            if row['Std'] >= 0.02:
                lines.append(f"  ⚠️  High variance (std = {row['Std']:.4f})")
            if not pd.isna(row['Expected']) and abs(row['Deviation']) >= 0.05:
                lines.append(f"  ⚠️  Large deviation from expected ({row['Deviation']:+.4f})")

    # Identify best configuration
    lines.append("\n" + "=" * 80)
    lines.append("BEST CONFIGURATION")
    lines.append("=" * 80)

    best_config = summary_df.nlargest(1, 'Mean').iloc[0]
    lines.append(f"\nHighest performing configuration: {best_config['Config']}")
    lines.append(f"  Mean Kappa: {best_config['Mean']:.4f} ± {best_config['Std']:.4f}")
    lines.append(f"  Reproducibility: {best_config['Status']}")

    if best_config['Status'] == 'GOOD':
        lines.append(f"\n✅ RECOMMENDED FOR THESIS:")
        lines.append(f"   Use {best_config['Config']} configuration")
        lines.append(f"   Expected performance: {best_config['Mean']:.4f} ± {best_config['Std']:.4f}")
    else:
        lines.append(f"\n⚠️  Best config has reproducibility issues!")

        # Find best reproducible config
        good_sorted = summary_df[summary_df['Status'] == 'GOOD'].sort_values('Mean', ascending=False)
        if len(good_sorted) > 0:
            best_repro = good_sorted.iloc[0]
            lines.append(f"\n   Consider using: {best_repro['Config']}")
            lines.append(f"   Mean Kappa: {best_repro['Mean']:.4f} ± {best_repro['Std']:.4f}")
            lines.append(f"   Trade-off: -{(best_config['Mean'] - best_repro['Mean']):.4f} Kappa for reproducibility")

    # Recommendations
    lines.append("\n" + "=" * 80)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 80)

    # Scheduler recommendation
    lines.append("\n1. SCHEDULER CHOICE:")
    if len(constant_configs) > 0 and len(plateau_configs) > 0:
        if constant_configs['Std'].mean() < plateau_configs['Std'].mean():
            lines.append("   ✅ USE CONSTANT LR AFTER WARMUP")
            lines.append("      Reasons:")
            lines.append("      - More reproducible (lower variance)")
            lines.append("      - Deterministic behavior")
            lines.append("      - Easier to compare experiments")
            if constant_configs['Mean'].mean() >= plateau_configs['Mean'].mean():
                lines.append("      - Equal or better performance")
        else:
            lines.append("   ⚠️  ReduceLROnPlateau showed better reproducibility (unexpected)")
            lines.append("      This needs further investigation")

    if len(bad_configs) > 0:
        lines.append("\n2. REPRODUCIBILITY ISSUES FOUND:")
        for _, row in bad_configs.iterrows():
            lines.append(f"   - {row['Config']}: High variance (std={row['Std']:.4f})")

        lines.append("\n3. POSSIBLE CAUSES:")
        lines.append("   - ReduceLROnPlateau triggering at different times")
        lines.append("   - Random initialization variations")
        lines.append("   - Data loading/shuffling differences")

        lines.append("\n4. NEXT STEPS FOR THESIS:")
        lines.append("   ✅ Switch to constant LR after warmup for all experiments")
        lines.append("   ✅ Re-run your ablation studies with new scheduler")
        lines.append("   ✅ Report results with confidence intervals")
    else:
        lines.append("\n2. FOR THESIS:")
        lines.append("   ✅ Your setup is reproducible!")
        lines.append("   ✅ Use the best configuration confidently")
        lines.append("   ✅ Report mean ± std across seeds")

    # Best overall config
    best_overall = summary_df.nlargest(1, 'Mean').iloc[0]
    lines.append("\n3. BEST CONFIGURATION:")
    lines.append(f"   Config: {best_overall['Config']}")
    lines.append(f"   Mean Kappa: {best_overall['Mean']:.4f} ± {best_overall['Std']:.4f}")
    lines.append(f"   Scheduler: {best_overall['Scheduler']}")
    lines.append(f"   Reproducibility: {best_overall['Status']}")

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    # Save report
    report_path = os.path.join(results_dir, 'reproducibility_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ Saved: reproducibility_report.txt")
    print("\n" + "\n".join(lines))


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Test reproducibility of critical configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script tests reproducibility by comparing two scheduler types:

1. ReduceLROnPlateau (current setup)
   - Reduces LR when validation stops improving
   - Non-deterministic: triggers at different times between runs

2. Constant LR after warmup (proposed fix)
   - Keeps LR constant after warmup period
   - Deterministic: same LR schedule every run

Configurations tested:
1. Initial best (Kappa 0.8339)
2. Follow-up best (Kappa 0.8135)

Each config tested with both schedulers, 3 seeds each.
Total: 12 experiments (2 configs × 2 schedulers × 3 seeds)

NOTE: You need to update trainers_cosine.py to add the 
'lr_scheduler_type' parameter for this to work!
        """
    )

    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--users', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: only 2 seeds instead of 3')

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

    if args.quick:
        TEST_SEEDS = [42, 123]
        print("\n⚡ Quick mode: Using 2 seeds instead of 3")

    if DATA_PATH == "path/to/your/data" or MODEL_PATH == "path/to/models/reproducibility_test":
        print("\n⚠️ WARNING: Please update DATA_PATH and MODEL_PATH!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(0)

    print("\n" + "=" * 80)
    print("REPRODUCIBILITY TESTING - SCHEDULER COMPARISON")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data path: {DATA_PATH}")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Users: {len(USER_IDS)}")
    print(f"  Epochs: {COMMON_PARAMS['epochs']}")
    print(f"  Seeds: {TEST_SEEDS}")
    print(f"\nTesting 2 configurations with 2 scheduler types")
    print(f"Total experiments: {4 * len(TEST_SEEDS)}")
    print(f"Estimated time: ~{4 * len(TEST_SEEDS) * COMMON_PARAMS['epochs'] * 2 / 60:.1f} hours")
    print("=" * 80)

    print("\n🔬 Starting reproducibility tests...")
    results = run_reproducibility_tests()

    print("\n" + "=" * 80)
    print("✅ REPRODUCIBILITY TESTING COMPLETE!")
    print("=" * 80)
    print("\n📊 Check the following files:")
    print("  - reproducibility_report.txt: Detailed analysis")
    print("  - reproducibility_summary.csv: Statistical summary")
    print("  - scheduler_comparison.png: Plateau vs Constant comparison")
    print("  - reproducibility_bars.png: Visual comparison")
    print("  - variance_analysis.png: Variance across seeds")
    print("  - deviation_analysis.png: Deviation from expected")

    print("\n🔍 KEY QUESTION ANSWERED:")
    print("  Is ReduceLROnPlateau causing non-reproducible results?")
    print("  → Check scheduler_comparison.png and the report!")