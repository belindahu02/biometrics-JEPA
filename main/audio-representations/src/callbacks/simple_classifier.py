#!/usr/bin/env python3
"""
Simplified classification script for downstream evaluation.
Uses all available samples and runs 5 experiments to get average accuracy.
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import time

# Import the 2D trainer (adjust import path as needed)
try:
    from trainers_2d import spectrogram_trainer_2d
except ImportError:
    print("Warning: Could not import spectrogram_trainer_2d. Make sure trainers_2d.py is in the path.")
    sys.exit(1)


def count_samples_in_directory(data_path, user_ids=None):
    """
    Count the number of available samples per user in the data directory.

    Args:
        data_path: Path to the grouped embeddings directory
        user_ids: List of user IDs to check (if None, auto-detect)

    Returns:
        dict: {user_id: sample_count}
    """
    sample_counts = {}
    data_path = Path(data_path)

    if not data_path.exists():
        print(f"Data path does not exist: {data_path}")
        return sample_counts

    # Look for user directories or files
    for item in data_path.iterdir():
        if item.is_dir():
            # Count .npy files in user directory
            npy_files = list(item.glob("*.npy"))
            if npy_files:
                try:
                    user_id = int(item.name)
                    if user_ids is None or user_id in user_ids:
                        sample_counts[user_id] = len(npy_files)
                except ValueError:
                    # Directory name is not a number, skip
                    continue
        elif item.suffix == '.npy':
            # Files directly in the directory - try to extract user ID from filename
            try:
                # Assuming filename format contains user info
                parts = item.stem.split('_')
                for part in parts:
                    if part.startswith('user') or part.startswith('User'):
                        user_id = int(part.replace('user', '').replace('User', ''))
                        if user_ids is None or user_id in user_ids:
                            sample_counts[user_id] = sample_counts.get(user_id, 0) + 1
                        break
            except (ValueError, IndexError):
                continue

    return sample_counts


def run_simple_classification(
        data_path: str,
        output_dir: str,
        model_path: str = "./temp_models",
        user_ids: list = None,
        num_runs: int = 5,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 0.001,
        normalization_method: str = 'none',
        model_type: str = 'lightweight',
        device: str = 'cuda',
        verbose: bool = True
):
    """
    Run simplified classification with all available samples.

    Args:
        data_path: Path to grouped embeddings
        output_dir: Directory to save results
        model_path: Directory to save model checkpoints
        user_ids: List of user IDs (if None, auto-detect)
        num_runs: Number of experimental runs
        epochs: Training epochs per run
        batch_size: Batch size for training
        lr: Learning rate
        normalization_method: Normalization method
        model_type: Model architecture type
        device: Device to use ('cuda' or 'cpu')
        verbose: Whether to print detailed logs

    Returns:
        dict: Results including accuracy and kappa scores
    """

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Auto-detect user IDs if not provided
    if user_ids is None:
        sample_counts = count_samples_in_directory(data_path)
        user_ids = list(sample_counts.keys())
        if verbose:
            print(f"Auto-detected user IDs: {user_ids}")
            for user_id, count in sample_counts.items():
                print(f"  User {user_id}: {count} samples")
    else:
        sample_counts = count_samples_in_directory(data_path, user_ids)
        if verbose:
            print(f"Using specified user IDs: {user_ids}")
            for user_id in user_ids:
                count = sample_counts.get(user_id, 0)
                print(f"  User {user_id}: {count} samples")

    if not user_ids:
        raise ValueError("No valid user IDs found in the data directory")

    # Determine samples per user (use all available)
    min_samples = min(sample_counts.values()) if sample_counts else 0
    max_samples = max(sample_counts.values()) if sample_counts else 0

    if min_samples == 0:
        raise ValueError("No samples found for the specified users")

    # Use all samples (or minimum if users have different counts)
    samples_per_user = min_samples

    if verbose:
        print(f"\nExperiment Configuration:")
        print(f"  Data path: {data_path}")
        print(f"  Users: {len(user_ids)} ({min(user_ids)}-{max(user_ids)})")
        print(f"  Samples per user: {samples_per_user} (min: {min_samples}, max: {max_samples})")
        print(f"  Number of runs: {num_runs}")
        print(f"  Epochs per run: {epochs}")
        print(f"  Model type: {model_type}")
        print(f"  Normalization: {normalization_method}")
        print(f"  Device: {device}")

    # Storage for results
    accuracies = []
    kappa_scores = []
    run_details = []

    start_time = time.time()

    # Run experiments
    for run_idx in range(num_runs):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Run {run_idx + 1}/{num_runs}")
            print(f"{'=' * 50}")

        run_start_time = time.time()

        try:
            # Run classification
            test_acc, kappa_score = spectrogram_trainer_2d(
                samples_per_user=samples_per_user,
                data_path=data_path,
                model_path=model_path,
                user_ids=user_ids,
                normalization_method=normalization_method,
                model_type=model_type,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                device=device,
                use_augmentation=True,
                save_model_checkpoints=False,  # Don't save checkpoints for evaluation
                checkpoint_every=epochs + 1,  # Effectively disable checkpointing
                max_cache_size=100,
                verbose=verbose
            )

            accuracies.append(test_acc)
            kappa_scores.append(kappa_score)

            run_time = time.time() - run_start_time

            run_details.append({
                'run': run_idx + 1,
                'accuracy': float(test_acc),
                'kappa': float(kappa_score),
                'duration_seconds': run_time,
                'samples_per_user': samples_per_user,
                'success': True
            })

            if verbose:
                print(f"Run {run_idx + 1} completed in {run_time:.1f}s")
                print(f"  Accuracy: {test_acc:.4f}")
                print(f"  Kappa: {kappa_score:.4f}")

        except Exception as e:
            print(f"Error in run {run_idx + 1}: {e}")
            run_details.append({
                'run': run_idx + 1,
                'accuracy': None,
                'kappa': None,
                'duration_seconds': time.time() - run_start_time,
                'samples_per_user': samples_per_user,
                'success': False,
                'error': str(e)
            })
            continue

    total_time = time.time() - start_time

    # Calculate statistics
    valid_accuracies = [acc for acc in accuracies if acc is not None]
    valid_kappas = [kappa for kappa in kappa_scores if kappa is not None]

    if not valid_accuracies:
        raise RuntimeError("All runs failed - no valid results obtained")

    results = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(data_path),
            'output_dir': str(output_dir),
            'samples_per_user': samples_per_user,
            'user_ids': user_ids,
            'num_users': len(user_ids),
            'total_runs_attempted': num_runs,
            'successful_runs': len(valid_accuracies),
            'epochs_per_run': epochs,
            'model_type': model_type,
            'normalization_method': normalization_method,
            'batch_size': batch_size,
            'learning_rate': lr,
            'device': device,
            'total_duration_seconds': total_time
        },
        'results': {
            'accuracies': valid_accuracies,
            'kappa_scores': valid_kappas,
            'mean_accuracy': float(np.mean(valid_accuracies)),
            'std_accuracy': float(np.std(valid_accuracies)),
            'mean_kappa': float(np.mean(valid_kappas)),
            'std_kappa': float(np.std(valid_kappas)),
            'max_accuracy': float(np.max(valid_accuracies)),
            'min_accuracy': float(np.min(valid_accuracies)),
            'max_kappa': float(np.max(valid_kappas)),
            'min_kappa': float(np.min(valid_kappas))
        },
        'run_details': run_details
    }

    # Save results
    results_file = Path(output_dir) / "classification_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Successful runs: {len(valid_accuracies)}/{num_runs}")
        print(f"Mean accuracy: {np.mean(valid_accuracies):.4f} ± {np.std(valid_accuracies):.4f}")
        print(f"Mean kappa: {np.mean(valid_kappas):.4f} ± {np.std(valid_kappas):.4f}")
        print(f"Results saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Simple downstream classification evaluation")
    parser.add_argument("--data_path", required=True, help="Path to grouped embeddings directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--model_path", default="./temp_models", help="Directory for model checkpoints")
    parser.add_argument("--user_ids", nargs='+', type=int, help="User IDs to include (default: auto-detect)")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of experimental runs")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per run")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--normalization_method", default="none", help="Normalization method")
    parser.add_argument("--model_type", default="lightweight", help="Model architecture type")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        results = run_simple_classification(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_path=args.model_path,
            user_ids=args.user_ids,
            num_runs=args.num_runs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            normalization_method=args.normalization_method,
            model_type=args.model_type,
            device=args.device,
            verbose=args.verbose
        )

        # Return the mean accuracy as the primary metric
        print(f"FINAL_ACCURACY: {results['results']['mean_accuracy']:.4f}")

    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()