"""
Ablation Study to Identify Which Change Fixed the Collapse

This script systematically tests each improvement individually and in combination
to identify which change(s) are responsible for preventing the collapse.
"""

import os
import json
import pandas as pd
from datetime import datetime
from trainers_cosine import spectrogram_trainer_2d

# =============================================
# CONFIGURATION
# =============================================

DATA_PATH = "/app/data/grouped_embeddings_full"  # UPDATE THIS
MODEL_PATH = "/app/data/experiments/ablation"  # UPDATE THIS
NORMALIZATION_METHOD = "log_scale"
MODEL_TYPE = "lightweight"

# Test with a challenging number of users (e.g., 30 or 40)
# Use the same user_ids for all experiments for fair comparison
USER_IDS = list(range(1, 31))  # 30 users - adjust based on where collapse occurs

# Common parameters for all experiments
COMMON_PARAMS = {
    'data_path': DATA_PATH,
    'model_path': MODEL_PATH,
    'user_ids': USER_IDS,
    'normalization_method': NORMALIZATION_METHOD,
    'model_type': MODEL_TYPE,
    'epochs': 50,  # Reduced for faster experiments, increase if needed
    'batch_size': 16,
    'use_augmentation': True,
    'device': 'cuda',
    'save_model_checkpoints': True,
    'checkpoint_every': 10,
    'max_cache_size': 50,
}

# =============================================
# EXPERIMENT CONFIGURATIONS
# =============================================

EXPERIMENTS = [
    {
        'name': '1_baseline',
        'description': 'Baseline: No improvements (should collapse)',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': False,
            'cosine_scale': 30.0,  # Not used
            'label_smoothing': 0.0,  # No smoothing
            'warmup_epochs': 0,  # No warmup
        }
    },
    {
        'name': '2_cosine_only',
        'description': 'Only Cosine Classifier',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': True,
            'cosine_scale': 30.0,
            'label_smoothing': 0.0,
            'warmup_epochs': 0,
        }
    },
    {
        'name': '3_label_smoothing_only',
        'description': 'Only Label Smoothing',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': False,
            'cosine_scale': 30.0,
            'label_smoothing': 0.1,
            'warmup_epochs': 0,
        }
    },
    {
        'name': '4_warmup_only',
        'description': 'Only LR Warmup',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': False,
            'cosine_scale': 30.0,
            'label_smoothing': 0.0,
            'warmup_epochs': 5,
        }
    },
    {
        'name': '5_cosine_plus_smoothing',
        'description': 'Cosine Classifier + Label Smoothing',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': True,
            'cosine_scale': 30.0,
            'label_smoothing': 0.1,
            'warmup_epochs': 0,
        }
    },
    {
        'name': '6_cosine_plus_warmup',
        'description': 'Cosine Classifier + LR Warmup',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': True,
            'cosine_scale': 30.0,
            'label_smoothing': 0.0,
            'warmup_epochs': 5,
        }
    },
    {
        'name': '7_smoothing_plus_warmup',
        'description': 'Label Smoothing + LR Warmup',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': False,
            'cosine_scale': 30.0,
            'label_smoothing': 0.1,
            'warmup_epochs': 5,
        }
    },
    {
        'name': '8_all_combined',
        'description': 'All improvements combined (should work best)',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': True,
            'cosine_scale': 30.0,
            'label_smoothing': 0.1,
            'warmup_epochs': 5,
        }
    },
]

# Optional: Test different cosine scales
COSINE_SCALE_EXPERIMENTS = [
    {
        'name': '9_cosine_scale_20',
        'description': 'Cosine with scale=20',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': True,
            'cosine_scale': 20.0,
            'label_smoothing': 0.0,
            'warmup_epochs': 0,
        }
    },
    {
        'name': '10_cosine_scale_50',
        'description': 'Cosine with scale=50',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': True,
            'cosine_scale': 50.0,
            'label_smoothing': 0.0,
            'warmup_epochs': 0,
        }
    },
    {
        'name': '11_cosine_scale_64',
        'description': 'Cosine with scale=64',
        'params': {
            'lr': 0.001,
            'use_cosine_classifier': True,
            'cosine_scale': 64.0,
            'label_smoothing': 0.0,
            'warmup_epochs': 0,
        }
    },
]


# =============================================
# EXPERIMENT RUNNER
# =============================================

def run_ablation_study(include_scale_experiments=False):
    """
    Run all ablation experiments and save results.

    Args:
        include_scale_experiments: Whether to include cosine scale experiments
    """

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(MODEL_PATH, f'ablation_study_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    # Determine which experiments to run
    experiments_to_run = EXPERIMENTS.copy()
    if include_scale_experiments:
        experiments_to_run.extend(COSINE_SCALE_EXPERIMENTS)

    results = []

    print("=" * 80)
    print(f"ABLATION STUDY: Testing {len(experiments_to_run)} configurations")
    print(f"Users: {len(USER_IDS)} (S{USER_IDS[0]:03d} to S{USER_IDS[-1]:03d})")
    print(f"Results will be saved to: {results_dir}")
    print("=" * 80)

    for idx, experiment in enumerate(experiments_to_run, 1):
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {idx}/{len(experiments_to_run)}: {experiment['name']}")
        print(f"Description: {experiment['description']}")
        print(f"Parameters: {experiment['params']}")
        print('=' * 80)

        try:
            # Merge common params with experiment-specific params
            full_params = {**COMMON_PARAMS, **experiment['params']}

            # Update model path to include experiment name
            full_params['model_path'] = os.path.join(results_dir, experiment['name'])

            # Run training
            test_acc, kappa_score = spectrogram_trainer_2d(**full_params)

            # Record results
            result = {
                'experiment': experiment['name'],
                'description': experiment['description'],
                'test_accuracy': test_acc,
                'kappa_score': kappa_score,
                'collapsed': kappa_score < 0.1,  # Heuristic: Kappa < 0.1 indicates collapse
                **experiment['params']
            }
            results.append(result)

            print(f"\n✓ COMPLETED: Test Accuracy = {test_acc:.4f}, Kappa = {kappa_score:.4f}")
            print(f"  Status: {'COLLAPSED ❌' if result['collapsed'] else 'SUCCESS ✓'}")

        except Exception as e:
            print(f"\n✗ FAILED: {str(e)}")
            result = {
                'experiment': experiment['name'],
                'description': experiment['description'],
                'test_accuracy': 0.0,
                'kappa_score': 0.0,
                'collapsed': True,
                'error': str(e),
                **experiment['params']
            }
            results.append(result)

    # Save results
    results_df = pd.DataFrame(results)

    # Save CSV
    csv_path = os.path.join(results_dir, 'ablation_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save JSON
    json_path = os.path.join(results_dir, 'ablation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(results_df[['experiment', 'description', 'test_accuracy', 'kappa_score', 'collapsed']].to_string(index=False))

    # Analyze results
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    successful = results_df[~results_df['collapsed']]
    collapsed = results_df[results_df['collapsed']]

    print(f"\nSuccessful experiments (no collapse): {len(successful)}/{len(results_df)}")
    if len(successful) > 0:
        print("\nSuccessful configurations:")
        for _, row in successful.iterrows():
            print(f"  - {row['experiment']}: Kappa={row['kappa_score']:.4f}, Acc={row['test_accuracy']:.4f}")
            if row['use_cosine_classifier']:
                print(f"    ✓ Cosine Classifier (scale={row['cosine_scale']})")
            if row['label_smoothing'] > 0:
                print(f"    ✓ Label Smoothing ({row['label_smoothing']})")
            if row['warmup_epochs'] > 0:
                print(f"    ✓ LR Warmup ({row['warmup_epochs']} epochs)")

    print(f"\nCollapsed experiments: {len(collapsed)}/{len(results_df)}")
    if len(collapsed) > 0:
        print("\nCollapsed configurations:")
        for _, row in collapsed.iterrows():
            print(f"  - {row['experiment']}: Kappa={row['kappa_score']:.4f}")

    # Determine most important factor
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Check individual components
    cosine_only = results_df[results_df['experiment'] == '2_cosine_only']
    smoothing_only = results_df[results_df['experiment'] == '3_label_smoothing_only']
    warmup_only = results_df[results_df['experiment'] == '4_warmup_only']

    if not cosine_only.empty and not cosine_only['collapsed'].iloc[0]:
        print("✓ Cosine Classifier ALONE prevents collapse")
    elif not cosine_only.empty:
        print("✗ Cosine Classifier alone does NOT prevent collapse")

    if not smoothing_only.empty and not smoothing_only['collapsed'].iloc[0]:
        print("✓ Label Smoothing ALONE prevents collapse")
    elif not smoothing_only.empty:
        print("✗ Label Smoothing alone does NOT prevent collapse")

    if not warmup_only.empty and not warmup_only['collapsed'].iloc[0]:
        print("✓ LR Warmup ALONE prevents collapse")
    elif not warmup_only.empty:
        print("✗ LR Warmup alone does NOT prevent collapse")

    # Check if combination is needed
    baseline = results_df[results_df['experiment'] == '1_baseline']
    all_combined = results_df[results_df['experiment'] == '8_all_combined']

    if not baseline.empty and baseline['collapsed'].iloc[0]:
        print("\n✓ Baseline collapsed as expected")

    if not all_combined.empty and not all_combined['collapsed'].iloc[0]:
        print("✓ All improvements combined prevent collapse")

        # Find minimum combination needed
        print("\nMinimum combination needed:")
        for _, row in successful.iterrows():
            components = []
            if row['use_cosine_classifier']:
                components.append('Cosine')
            if row['label_smoothing'] > 0:
                components.append('Smoothing')
            if row['warmup_epochs'] > 0:
                components.append('Warmup')

            if len(components) == 1:
                print(f"  → ONLY {components[0]} is sufficient!")
                break
            elif len(components) == 2:
                print(f"  → {' + '.join(components)} combination works")

    print("\n" + "=" * 80)
    print(f"Full results saved to: {results_dir}")
    print("=" * 80)

    return results_df


# =============================================
# QUICK TEST (Optional)
# =============================================

def run_quick_test():
    """
    Quick test with just 3 key experiments to save time.
    Run this first to get a quick answer.
    """
    quick_experiments = [
        EXPERIMENTS[2],  # label smoothing
        EXPERIMENTS[3]  # warmup
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(MODEL_PATH, f'quick_ablation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("QUICK ABLATION TEST (3 experiments)")
    print("=" * 80)

    results = []
    for idx, experiment in enumerate(quick_experiments, 1):
        print(f"\nExperiment {idx}/3: {experiment['name']}")

        full_params = {**COMMON_PARAMS, **experiment['params']}
        full_params['model_path'] = os.path.join(results_dir, experiment['name'])

        test_acc, kappa_score = spectrogram_trainer_2d(**full_params)

        result = {
            'experiment': experiment['name'],
            'test_accuracy': test_acc,
            'kappa_score': kappa_score,
            'collapsed': kappa_score < 0.1,
        }
        results.append(result)

        print(f"Result: Acc={test_acc:.4f}, Kappa={kappa_score:.4f}, "
              f"Status={'COLLAPSED' if result['collapsed'] else 'SUCCESS'}")

    return pd.DataFrame(results)


# =============================================
# MAIN
# =============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with only 3 experiments')
    parser.add_argument('--include-scales', action='store_true',
                        help='Include cosine scale experiments')
    args = parser.parse_args()

    if args.quick:
        print("Running QUICK test...")
        results = run_quick_test()
    else:
        print("Running FULL ablation study...")
        results = run_ablation_study(include_scale_experiments=args.include_scales)

    print("\n✓ Ablation study complete!")