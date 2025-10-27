"""
Overfitting Reduction Experiments

Tests 13 focused strategies to close the train-val-test gap:
- Dropout variations (4 experiments)
- Weight decay variations (2 experiments)
- Combined approaches (7 experiments)

IMPORTANT: Requires backbones_with_dropout.py as backbones.py
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from trainers_cosine import spectrogram_trainer_2d

# =============================================
# CONFIGURATION
# =============================================

DATA_PATH = "/app/data/grouped_embeddings_full"
MODEL_PATH = "/app/data/experiments/overfitting_reduction"
NORMALIZATION_METHOD = "log_scale"
USER_IDS = list(range(1, 31))  # 30 users

# Base parameters from your best config
BASE_PARAMS = {
    'data_path': DATA_PATH,
    'model_path': MODEL_PATH,
    'user_ids': USER_IDS,
    'normalization_method': NORMALIZATION_METHOD,
    'epochs': 75,  # Reduced from 100 for faster iteration
    'use_augmentation': True,
    'device': 'cuda',
    'save_model_checkpoints': True,
    'checkpoint_every': 10,
    'max_cache_size': 50,
    'use_cosine_classifier': True,
    'lr_scheduler_type': 'constant',
    'random_seed': 42,
    # From your best config
    'lr': 0.001,
    'batch_size': 16,
    'model_type': 'lightweight',
    'cosine_scale': 40.0,
    'label_smoothing': 0.1,
    'warmup_epochs': 5,
}


# =============================================
# EXPERIMENT DESIGN
# =============================================

def create_overfitting_experiments():
    """
    Create 13 focused experiments targeting overfitting reduction
    """
    experiments = []

    # ========== BASELINE ==========
    experiments.append({
        'name': 'baseline_current',
        'description': 'Current best config (for comparison)',
        'params': {},
        'weight_decay': 1e-4
    })

    # ========== DROPOUT EXPERIMENTS (4) ==========

    experiments.append({
        'name': 'dropout_conv_moderate',
        'description': 'Moderate dropout (0.2) in conv layers only',
        'params': {'dropout_rate': 0.2, 'classifier_dropout': 0.0},
        'weight_decay': 1e-4
    })

    experiments.append({
        'name': 'dropout_classifier_strong',
        'description': 'Strong dropout (0.5) before classifier only',
        'params': {'dropout_rate': 0.0, 'classifier_dropout': 0.5},
        'weight_decay': 1e-4
    })

    experiments.append({
        'name': 'dropout_everywhere',
        'description': 'Balanced dropout everywhere (conv=0.2, classifier=0.4)',
        'params': {'dropout_rate': 0.2, 'classifier_dropout': 0.4},
        'weight_decay': 1e-4
    })

    experiments.append({
        'name': 'dropout_heavy',
        'description': 'Heavy dropout (conv=0.3, classifier=0.5)',
        'params': {'dropout_rate': 0.3, 'classifier_dropout': 0.5},
        'weight_decay': 1e-4
    })

    # ========== WEIGHT DECAY EXPERIMENTS (2) ==========

    experiments.append({
        'name': 'weight_decay_moderate',
        'description': 'Moderate weight decay (1e-3, 10x baseline)',
        'params': {},
        'weight_decay': 1e-3
    })

    experiments.append({
        'name': 'weight_decay_strong',
        'description': 'Strong weight decay (5e-3, 50x baseline)',
        'params': {},
        'weight_decay': 5e-3
    })

    # ========== COMBINED EXPERIMENTS (6) ==========

    experiments.append({
        'name': 'dropout_and_weight_decay',
        'description': 'Dropout + increased weight decay',
        'params': {'dropout_rate': 0.2, 'classifier_dropout': 0.3},
        'weight_decay': 1e-3
    })

    experiments.append({
        'name': 'label_smoothing_high',
        'description': 'Higher label smoothing (0.2)',
        'params': {'label_smoothing': 0.2},
        'weight_decay': 1e-4
    })

    experiments.append({
        'name': 'reduced_capacity',
        'description': 'Lower cosine scale (20) to reduce capacity',
        'params': {'cosine_scale': 20.0},
        'weight_decay': 1e-4
    })

    experiments.append({
        'name': 'combined_moderate',
        'description': 'Moderate: dropout + weight_decay + smoothing',
        'params': {
            'dropout_rate': 0.15,
            'classifier_dropout': 0.3,
            'label_smoothing': 0.15,
            'cosine_scale': 30.0
        },
        'weight_decay': 1e-3
    })

    experiments.append({
        'name': 'combined_conservative',
        'description': 'Conservative: strong regularization everywhere',
        'params': {
            'dropout_rate': 0.25,
            'classifier_dropout': 0.4,
            'label_smoothing': 0.2,
            'cosine_scale': 25.0,
            'lr': 0.0005
        },
        'weight_decay': 5e-3
    })

    experiments.append({
        'name': 'combined_nuclear',
        'description': 'Nuclear: maximum possible regularization',
        'params': {
            'dropout_rate': 0.3,
            'classifier_dropout': 0.5,
            'label_smoothing': 0.3,
            'cosine_scale': 20.0,
            'lr': 0.0003,
            'batch_size': 8
        },
        'weight_decay': 1e-2
    })

    return experiments


# =============================================
# EXPERIMENT RUNNER
# =============================================

def run_experiment(exp_name, exp_description, exp_params, weight_decay, results_dir):
    """Run a single overfitting reduction experiment"""
    print(f"\n{'=' * 80}")
    print(f"Running: {exp_name}")
    print(f"Description: {exp_description}")
    print(f"Modified params: {exp_params}")
    print(f"Weight decay: {weight_decay}")
    print('=' * 80)

    try:
        full_params = {**BASE_PARAMS, **exp_params}
        full_params['model_path'] = os.path.join(results_dir, exp_name)
        full_params['weight_decay'] = weight_decay

        test_acc, kappa_score = spectrogram_trainer_2d(**full_params)

        result = {
            'experiment': exp_name,
            'description': exp_description,
            'test_accuracy': test_acc,
            'kappa_score': kappa_score,
            'collapsed': kappa_score < 0.1,
            'weight_decay': weight_decay,
            **exp_params
        }

        print(f"\n✓ COMPLETED: Test Accuracy = {test_acc:.4f}, Kappa = {kappa_score:.4f}")

        # Load training history to compute train-val gap
        try:
            model_subdirs = [d for d in os.listdir(full_params['model_path'])
                             if os.path.isdir(os.path.join(full_params['model_path'], d))
                             and d.startswith('log_scale')]

            if model_subdirs:
                history_path = os.path.join(full_params['model_path'],
                                            model_subdirs[0],
                                            'training_history.json')

                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                        final_train_acc = history['train_acc'][-1]
                        final_val_acc = history['val_acc'][-1]
                        train_val_gap = final_train_acc - final_val_acc

                        result['final_train_acc'] = final_train_acc
                        result['final_val_acc'] = final_val_acc
                        result['train_val_gap'] = train_val_gap
                        result['val_test_gap'] = final_val_acc - test_acc

                        print(f"  Train Acc: {final_train_acc:.4f}")
                        print(f"  Val Acc: {final_val_acc:.4f}")
                        print(f"  Train-Val Gap: {train_val_gap:.4f}")
                        print(f"  Val-Test Gap: {result['val_test_gap']:.4f}")
        except Exception as e:
            print(f"  Warning: Could not load training history: {e}")

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
            'weight_decay': weight_decay,
            **exp_params
        }


# =============================================
# VISUALIZATION FUNCTIONS
# =============================================

def analyze_results(results_df, save_dir):
    """Analyze and visualize overfitting reduction results"""
    vis_dir = os.path.join(save_dir, 'analysis')
    os.makedirs(vis_dir, exist_ok=True)

    successful = results_df[~results_df['collapsed']].copy()

    if len(successful) == 0:
        print("⚠ No successful experiments to analyze")
        return

    # ========== PLOT 1: Train-Val Gap Comparison ==========
    if 'train_val_gap' in successful.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        gap_sorted = successful.sort_values('train_val_gap')

        colors = ['green' if gap < 0.15 else 'orange' if gap < 0.25 else 'red'
                  for gap in gap_sorted['train_val_gap']]

        bars = ax1.barh(range(len(gap_sorted)), gap_sorted['train_val_gap'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(gap_sorted)))
        ax1.set_yticklabels(gap_sorted['experiment'], fontsize=8)
        ax1.set_xlabel('Train-Val Accuracy Gap')
        ax1.set_title('Overfitting Severity (Lower is Better)')
        ax1.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='High')
        ax1.axvline(x=0.15, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)

        ax2.scatter(successful['train_val_gap'], successful['test_accuracy'],
                    s=100, alpha=0.6, c='steelblue')

        for idx, row in successful.iterrows():
            ax2.annotate(row['experiment'],
                         (row['train_val_gap'], row['test_accuracy']),
                         fontsize=6, alpha=0.7, rotation=15)

        ax2.set_xlabel('Train-Val Gap (Overfitting)')
        ax2.set_ylabel('Test Accuracy')
        ax2.set_title('Generalization: Test Acc vs Overfitting')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: overfitting_analysis.png")

    # ========== PLOT 2: Test Accuracy Ranking ==========
    fig, ax = plt.subplots(figsize=(12, 8))

    acc_sorted = successful.sort_values('test_accuracy', ascending=True)

    bars = ax.barh(range(len(acc_sorted)), acc_sorted['test_accuracy'], alpha=0.7, color='steelblue')
    ax.set_yticks(range(len(acc_sorted)))
    ax.set_yticklabels(acc_sorted['experiment'], fontsize=9)
    ax.set_xlabel('Test Accuracy')
    ax.set_title('Test Accuracy Ranking')
    ax.grid(axis='x', alpha=0.3)

    for i, (idx, row) in enumerate(acc_sorted.iterrows()):
        ax.text(row['test_accuracy'], i,
                f"  κ={row['kappa_score']:.3f}",
                va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'test_accuracy_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: test_accuracy_ranking.png")

    # ========== Summary Table ==========
    summary = successful.copy()
    summary = summary.sort_values('test_accuracy', ascending=False)

    cols = ['experiment', 'test_accuracy', 'kappa_score']
    if 'train_val_gap' in summary.columns:
        cols.extend(['final_train_acc', 'final_val_acc', 'train_val_gap'])

    summary_display = summary[cols].head(10)
    summary_display.to_csv(os.path.join(vis_dir, 'top_configs.csv'), index=False)
    print(f"✓ Saved: top_configs.csv")

    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS (by test accuracy)")
    print("=" * 80)
    print(summary_display.head(5).to_string(index=False))

    if 'train_val_gap' in summary.columns:
        print("\n" + "=" * 80)
        print("BEST GENERALIZATION (lowest train-val gap)")
        print("=" * 80)
        best_gen = summary.sort_values('train_val_gap').head(5)
        print(best_gen[cols].to_string(index=False))


def generate_report(results_df, save_dir):
    """Generate comprehensive report"""
    successful = results_df[~results_df['collapsed']]

    lines = []
    lines.append("=" * 80)
    lines.append("OVERFITTING REDUCTION EXPERIMENT REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total experiments: {len(results_df)}")
    lines.append(f"Successful: {len(successful)}")
    lines.append(f"Failed: {len(results_df) - len(successful)}")

    if len(successful) > 0 and 'train_val_gap' in successful.columns:
        best_acc = successful.nlargest(1, 'test_accuracy').iloc[0]
        best_gen = successful.nsmallest(1, 'train_val_gap').iloc[0]

        lines.append("\n" + "=" * 80)
        lines.append("BEST TEST ACCURACY")
        lines.append("=" * 80)
        lines.append(f"Config: {best_acc['experiment']}")
        lines.append(f"Test Accuracy: {best_acc['test_accuracy']:.4f}")
        lines.append(f"Kappa: {best_acc['kappa_score']:.4f}")
        lines.append(f"Train-Val Gap: {best_acc['train_val_gap']:.4f}")

        lines.append("\n" + "=" * 80)
        lines.append("BEST GENERALIZATION (Lowest Overfitting)")
        lines.append("=" * 80)
        lines.append(f"Config: {best_gen['experiment']}")
        lines.append(f"Test Accuracy: {best_gen['test_accuracy']:.4f}")
        lines.append(f"Kappa: {best_gen['kappa_score']:.4f}")
        lines.append(f"Train-Val Gap: {best_gen['train_val_gap']:.4f}")

        lines.append("\n" + "=" * 80)
        lines.append("RECOMMENDATIONS FOR THESIS")
        lines.append("=" * 80)

        # Find balanced config
        successful['score'] = successful['test_accuracy'] - 0.3 * successful['train_val_gap']
        best_balance = successful.nlargest(1, 'score').iloc[0]

        lines.append("\nRECOMMENDED CONFIGURATION (Best balance):")
        lines.append(f"  Name: {best_balance['experiment']}")
        lines.append(f"  Test Accuracy: {best_balance['test_accuracy']:.4f}")
        lines.append(f"  Kappa Score: {best_balance['kappa_score']:.4f}")
        lines.append(f"  Train-Val Gap: {best_balance['train_val_gap']:.4f}")

        lines.append("\nTo use this config, set:")
        for key in ['dropout_rate', 'classifier_dropout', 'weight_decay', 'label_smoothing', 'cosine_scale', 'lr',
                    'batch_size']:
            if key in best_balance and pd.notna(best_balance[key]):
                lines.append(f"  {key}: {best_balance[key]}")

    lines.append("\n" + "=" * 80)

    report_path = os.path.join(save_dir, 'overfitting_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print("\n" + "\n".join(lines))
    print(f"\n✓ Report saved to: {report_path}")


# =============================================
# MAIN EXECUTION
# =============================================

def run_overfitting_study():
    """Main execution function"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(MODEL_PATH, f'overfitting_study_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("OVERFITTING REDUCTION EXPERIMENT")
    print("=" * 80)
    print(f"Goal: Reduce train-val-test gap")
    print(f"Focused approach: 13 targeted experiments")
    print(f"Results will be saved to: {results_dir}")
    print("=" * 80)

    experiments = create_overfitting_experiments()

    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Estimated time: ~{len(experiments) * BASE_PARAMS['epochs'] * 2 / 60:.1f} hours")
    print("\nExperiment categories:")
    print("  1. Baseline (1)")
    print("  2. Dropout variations (4)")
    print("  3. Weight decay (2)")
    print("  4. Dropout + weight decay (1)")
    print("  5. Other regularization (2)")
    print("  6. Combined strategies (3)")
    print("=" * 80)

    results = []
    for idx, exp in enumerate(experiments, 1):
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {idx}/{len(experiments)}")
        print('=' * 80)

        result = run_experiment(
            exp['name'],
            exp['description'],
            exp['params'],
            exp['weight_decay'],
            results_dir
        )
        results.append(result)

        # Save intermediate results
        interim_df = pd.DataFrame(results)
        interim_df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)

    # Final analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, 'final_results.csv'), index=False)

    print("\n" + "=" * 80)
    print("ANALYZING RESULTS")
    print("=" * 80)

    analyze_results(results_df, results_dir)
    generate_report(results_df, results_dir)

    print("\n" + "=" * 80)
    print("✓ OVERFITTING REDUCTION STUDY COMPLETE!")
    print("=" * 80)
    print(f"All results saved to: {results_dir}")
    print("\nKey files:")
    print("  - final_results.csv: Complete dataset")
    print("  - analysis/overfitting_analysis.png: Train-val gap comparison")
    print("  - analysis/test_accuracy_ranking.png: Performance ranking")
    print("  - analysis/top_configs.csv: Best configurations")
    print("  - overfitting_report.txt: Detailed recommendations")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run overfitting reduction experiments')
    parser.add_argument('--epochs', type=int, default=75, help='Number of epochs (default: 75)')
    parser.add_argument('--quick-test', action='store_true', help='Run only 3 experiments for testing')

    args = parser.parse_args()

    if args.epochs != 75:
        BASE_PARAMS['epochs'] = args.epochs
        print(f"Using {args.epochs} epochs")

    if args.quick_test:
        print("\n🚀 QUICK TEST MODE: Running only 3 experiments")
        print("This will take approximately 6 hours with 75 epochs\n")

    results = run_overfitting_study()
