# eval_classifier.py - Simplified version for downstream evaluation during JEPA training

import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime
from trainers_2d import spectrogram_trainer_2d
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Run downstream classification evaluation')
    parser.add_argument('--eval_only', action='store_true', help='Run single evaluation only')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--samples_per_user', type=int, default=50, help='Samples per user for evaluation')
    parser.add_argument('--data_path', type=str, default=None, help='Path to classification data')
    parser.add_argument('--model_path', type=str, default=None, help='Path for model checkpoints')
    parser.add_argument('--user_ids', type=str, default='1', help='Comma-separated user IDs')
    parser.add_argument('--normalization_method', type=str, default='none', help='Normalization method')
    parser.add_argument('--model_type', type=str, default='lightweight', help='Model type')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of evaluation runs')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per run')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')

    return parser.parse_args()


def run_evaluation(args):
    """Run downstream classification evaluation"""

    # Setup paths
    if args.data_path is None:
        args.data_path = os.path.join(args.output_dir, "grouped_embeddings")

    if args.model_path is None:
        args.model_path = os.path.join(args.output_dir, "model_checkpoints")

    # Parse user IDs
    user_ids = [int(x.strip()) for x in args.user_ids.split(',')]

    print(f"Running downstream evaluation:")
    print(f"  - Samples per user: {args.samples_per_user}")
    print(f"  - Users: {user_ids}")
    print(f"  - Runs: {args.n_runs}")
    print(f"  - Epochs per run: {args.epochs}")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Output dir: {args.output_dir}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    # Run multiple evaluations for robust estimation
    accuracies = []
    kappa_scores = []

    for run_idx in range(args.n_runs):
        print(f"\n--- Evaluation run {run_idx + 1}/{args.n_runs} ---")

        try:
            test_acc, kappa_score = spectrogram_trainer_2d(
                samples_per_user=args.samples_per_user,
                data_path=args.data_path,
                model_path=args.model_path,
                user_ids=user_ids,
                normalization_method=args.normalization_method,
                model_type=args.model_type,
                batch_size=16 if args.model_type == 'lightweight' else 8,
                epochs=args.epochs,
                lr=0.001,
                device=args.device,
                use_augmentation=False,  # Disable for consistent evaluation
                save_model_checkpoints=False,  # Don't save intermediate checkpoints
                max_cache_size=50,
            )

            accuracies.append(test_acc)
            kappa_scores.append(kappa_score)

            print(f"Run {run_idx + 1} - Accuracy: {test_acc:.4f}, Kappa: {kappa_score:.4f}")

        except Exception as e:
            print(f"❌ Run {run_idx + 1} failed: {e}")
            continue

    if not accuracies:
        print("❌ All evaluation runs failed!")
        return None, None

    # Calculate statistics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_kappa = np.mean(kappa_scores)
    std_kappa = np.std(kappa_scores)

    print(f"\n📊 Evaluation Results:")
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Kappa:    {mean_kappa:.4f} ± {std_kappa:.4f}")
    print(f"Successful runs: {len(accuracies)}/{args.n_runs}")

    # Save detailed results
    results = {
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'mean_kappa': mean_kappa,
        'std_kappa': std_kappa,
        'all_accuracies': accuracies,
        'all_kappa_scores': kappa_scores,
        'successful_runs': len(accuracies),
        'total_runs': args.n_runs,
        'evaluation_config': {
            'samples_per_user': args.samples_per_user,
            'user_ids': user_ids,
            'normalization_method': args.normalization_method,
            'model_type': args.model_type,
            'epochs': args.epochs,
            'device': args.device
        },
        'timestamp': datetime.now().isoformat()
    }

    # Save as both JSON and pickle
    json_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    pickle_file = os.path.join(args.output_dir, 'evaluation_results.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ Results saved to {json_file} and {pickle_file}")

    return mean_accuracy, mean_kappa


def main():
    args = parse_args()

    if args.eval_only:
        # Run single evaluation
        accuracy, kappa = run_evaluation(args)
        if accuracy is not None:
            print(f"\n🎯 Final downstream accuracy: {accuracy:.4f}")
            return accuracy
        else:
            print("❌ Evaluation failed")
            return None
    else:
        print("❌ This script is designed for eval_only mode")
        return None


if __name__ == "__main__":
    result = main()
    if result is not None:
        # Exit with success and print result for easy parsing
        print(f"FINAL_ACCURACY:{result:.6f}")
        sys.exit(0)
    else:
        sys.exit(1)