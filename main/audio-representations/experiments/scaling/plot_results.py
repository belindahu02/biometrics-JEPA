# =============================================
# plot_results.py with User-based Scaling
# =============================================

from trainers_cosine import spectrogram_trainer_2d  # Updated import for 2D trainer
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime

# Server config
DATA_PATH = "/app/data/grouped_embeddings_full"
OUTPUT_PATH = "/app/experiments/scaling/graph_data"
GRAPH_PATH = "/app/experiments/scaling/graph"
CHECKPOINT_PATH = "/app/experiments/scaling/graph_checkpoints"
MODEL_PATH = "/app/experiments/scaling/model_checkpoints"

# local config
# DATA_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/grouped_embeddings"
# OUTPUT_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/scaling/graph_data"
# GRAPH_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/scaling/graph"
# CHECKPOINT_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/scaling/graph_checkpoints"
# MODEL_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/scaling/model_checkpoints"

RUNS_PER_USER_COUNT = 10

# Create directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Updated experiment parameters
TOTAL_USERS_AVAILABLE = 109 # Users S001 to S109
NORMALIZATION_METHOD = 'log_scale'
MODEL_TYPE = 'lightweight'  # Options: 'lightweight', 'full'

variable_name = "number of users"
model_name = f"spectrogram_2d_{NORMALIZATION_METHOD}_{MODEL_TYPE}_users"
# Changed to test different numbers of users instead of samples per user
user_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109]

# Checkpoint file names
checkpoint_file = os.path.join(CHECKPOINT_PATH, f"{model_name}_checkpoint.json")
results_backup_file = os.path.join(CHECKPOINT_PATH, f"{model_name}_results_backup.pkl")


def save_checkpoint(current_idx, current_itr, acc, kappa, experiment_start_time):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        'current_user_count_idx': current_idx,
        'current_iteration': current_itr,
        'completed_user_counts': current_idx,
        'total_user_counts': len(user_counts),
        'experiment_start_time': experiment_start_time,
        'last_checkpoint_time': datetime.now().isoformat(),
        'normalization_method': NORMALIZATION_METHOD,
        'model_type': MODEL_TYPE,
        'user_counts_completed': user_counts[:current_idx],
        'results_shape': {
            'acc_lengths': [len(sublist) for sublist in acc],
            'kappa_lengths': [len(sublist) for sublist in kappa]
        }
    }

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    # Save results backup using pickle to handle variable-length sublists
    if acc and kappa:
        import pickle
        backup_data = {
            'test_acc_list': acc,
            'kappa_score_list': kappa,
            'completed_user_counts': user_counts[:current_idx]
        }

        with open(results_backup_file.replace('.npz', '.pkl'), 'wb') as f:
            pickle.dump(backup_data, f)

    print(f"Checkpoint saved: {current_idx}/{len(user_counts)} user counts completed")


def load_checkpoint():
    """Load previous progress from checkpoint file"""
    if not os.path.exists(checkpoint_file):
        return None, None, [], []

    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)

        start_idx = checkpoint_data.get('current_user_count_idx', 0)
        start_itr = checkpoint_data.get('current_iteration', 0)

        # Load results backup if it exists
        acc, kappa = [], []
        if os.path.exists(results_backup_file):
            try:
                import pickle
                with open(results_backup_file, 'rb') as f:
                    backup_data = pickle.load(f)
                acc = backup_data.get('test_acc_list', [])
                kappa = backup_data.get('kappa_score_list', [])
            except Exception as e:
                print(f"Warning: Could not load backup results: {e}")
                acc, kappa = [], []

        print(f"Resuming from checkpoint: {start_idx}/{len(user_counts)} user counts completed")
        print(f"Last checkpoint: {checkpoint_data.get('last_checkpoint_time', 'unknown')}")

        return start_idx, start_itr, acc, kappa

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, [], []


def save_intermediate_results(acc, kappa, current_idx):
    """Save intermediate results that can be plotted, handling failed runs"""
    # Check if we have any data at all
    if len(acc) == 0 and len(kappa) == 0:
        print("Warning: No data to save yet (acc and kappa are both empty)")
        return

    # Ensure we have at least some valid data
    valid_acc_count = sum(1 for sublist in acc if len(sublist) > 0)
    valid_kappa_count = sum(1 for sublist in kappa if len(sublist) > 0)

    if valid_acc_count == 0 and valid_kappa_count == 0:
        print("Warning: No valid runs completed yet (all sublists are empty)")
        return

    print(f"Saving intermediate results: {valid_acc_count} valid accuracy runs, {valid_kappa_count} valid kappa runs")

    # Save using pickle to handle variable lengths and empty sublists
    intermediate_file = os.path.join(OUTPUT_PATH, f"{model_name}_intermediate.pkl")
    import pickle

    intermediate_data = {
        'test_acc_list': acc,
        'kappa_score_list': kappa,
        'user_counts_completed': user_counts[:current_idx],
        'valid_acc_runs': valid_acc_count,
        'valid_kappa_runs': valid_kappa_count,
        'total_attempted_runs': len(acc) * RUNS_PER_USER_COUNT,
        'save_timestamp': datetime.now().isoformat()
    }

    try:
        with open(intermediate_file, 'wb') as f:
            pickle.dump(intermediate_data, f)
        print(f"Intermediate data saved to {intermediate_file}")
    except Exception as e:
        print(f"Failed to save intermediate data: {e}")
        return

    # Create intermediate plots only if we have valid data
    try:
        create_plots(acc, kappa, current_idx, suffix="_intermediate")
        print(f"Intermediate plots created")
    except Exception as e:
        print(f"Could not create intermediate plots: {e}")


def create_plots(acc, kappa, num_completed, suffix=""):
    """Create plots with current results, handling empty sublists from failed runs"""
    if len(acc) == 0 and len(kappa) == 0:
        print("Warning: No data available for plotting")
        return

    current_user_counts = user_counts[:num_completed]

    # Handle variable-length sublists and empty sublists from failed runs
    kappa_max = []
    acc_max = []
    valid_user_counts = []

    for i in range(min(len(acc), len(kappa), len(current_user_counts))):
        has_valid_acc = i < len(acc) and len(acc[i]) > 0
        has_valid_kappa = i < len(kappa) and len(kappa[i]) > 0

        # Only include data points where we have at least one valid result
        if has_valid_acc or has_valid_kappa:
            valid_user_counts.append(current_user_counts[i])

            if has_valid_acc:
                acc_max.append(np.max(acc[i]))
            else:
                acc_max.append(np.nan)  # Use NaN for missing data

            if has_valid_kappa:
                kappa_max.append(np.max(kappa[i]))
            else:
                kappa_max.append(np.nan)  # Use NaN for missing data

    if len(valid_user_counts) == 0:
        print("Warning: No valid data points available for plotting")
        return

    print(f"Plotting {len(valid_user_counts)} data points (some may have NaN values)")

    # Kappa plot
    plt.figure(figsize=(12, 8))
    # Plot with NaN handling - matplotlib will skip NaN points
    plt.plot(valid_user_counts, kappa_max, 'b-o', label=f'{model_name}', linewidth=2, markersize=6)
    plt.title(f"Kappa Score vs {variable_name} (2D {NORMALIZATION_METHOD} normalization, {MODEL_TYPE} model){suffix}")
    plt.xlabel(variable_name)
    plt.ylabel("Kappa Score")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add text showing number of valid points
    valid_kappa_count = np.sum(~np.isnan(kappa_max))
    plt.text(0.02, 0.98, f'Valid points: {valid_kappa_count}/{len(valid_user_counts)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_graph = os.path.join(GRAPH_PATH, f"kappa_2d_{NORMALIZATION_METHOD}_{MODEL_TYPE}_users{suffix}.jpg")
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(12, 8))
    plt.plot(valid_user_counts, acc_max, 'r-s', label=f'{model_name}', linewidth=2, markersize=6)
    plt.title(f"Test Accuracy vs {variable_name} (2D {NORMALIZATION_METHOD} normalization, {MODEL_TYPE} model){suffix}")
    plt.xlabel(variable_name)
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add text showing number of valid points
    valid_acc_count = np.sum(~np.isnan(acc_max))
    plt.text(0.02, 0.98, f'Valid points: {valid_acc_count}/{len(valid_user_counts)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    output_graph = os.path.join(GRAPH_PATH, f"acc_2d_{NORMALIZATION_METHOD}_{MODEL_TYPE}_users{suffix}.jpg")
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plots saved with {valid_acc_count} valid accuracy points and {valid_kappa_count} valid kappa points")


# Main execution
print(f"Running 2D spectrogram experiments with variable number of users")
print(f"Configuration:")
print(f"  - Normalization method: {NORMALIZATION_METHOD}")
print(f"  - Model type: {MODEL_TYPE}")
print(f"  - Testing {len(user_counts)} different user counts: {user_counts}")
print(f"  - Using 100% of available samples per user")
print(f"  - Session-based data splitting to avoid leakage")
print(f"  - Training will run for 100 epochs per experiment")

# Try to load previous progress
start_idx, start_itr, acc, kappa = load_checkpoint()
if start_idx is None:
    start_idx, start_itr = 0, 0
    acc, kappa = [], []
    experiment_start_time = datetime.now().isoformat()
else:
    # Load experiment start time from checkpoint
    with open(checkpoint_file, 'r') as f:
        checkpoint_data = json.load(f)
    experiment_start_time = checkpoint_data.get('experiment_start_time', datetime.now().isoformat())

print(f"Starting from user count index {start_idx}, iteration {start_itr}")

# Track experiment timing
experiment_start = time.time()

try:
    for i in range(start_idx, len(user_counts)):
        num_users = user_counts[i]
        user_ids = list(range(1, num_users + 1))  # Users 1 through num_users
        print(f"\n{'=' * 60}")
        print(f"Testing with {num_users} users (S001-S{num_users:03d}) ({i + 1}/{len(user_counts)})")
        print(f"{'=' * 60}")

        # If resuming mid-user-count, use existing results, otherwise start fresh
        if i == start_idx and start_itr > 0:
            acc_temp = acc[i] if i < len(acc) else []
            kappa_temp = kappa[i] if i < len(kappa) else []
        else:
            acc_temp = []
            kappa_temp = []

        for itr in range(start_itr if i == start_idx else 0, RUNS_PER_USER_COUNT):
            run_start_time = time.time()
            print(f"\n--- Run {itr + 1}/{RUNS_PER_USER_COUNT} for {num_users} users ---")

            try:
                # Call 2D trainer with updated parameters
                test_acc, kappa_score = spectrogram_trainer_2d(
                    data_path=DATA_PATH,
                    model_path=MODEL_PATH,
                    user_ids=user_ids,  # Pass the specific subset of users
                    normalization_method=NORMALIZATION_METHOD,
                    model_type=MODEL_TYPE,
                    epochs=100,
                    batch_size=16,
                    lr=0.0003,  # Consider reducing to 0.0003 or 0.0005
                    use_augmentation=True,
                    device='cuda',
                    save_model_checkpoints=True,
                    checkpoint_every=RUNS_PER_USER_COUNT,
                    max_cache_size=50,
                    # New parameters for collapse prevention
                    use_cosine_classifier=True,  # Enable cosine classifier
                    cosine_scale=30.0,  # Temperature scaling (try 50.0 or 64.0 if still collapsing)
                    label_smoothing=0.1,  # Label smoothing factor
                    warmup_epochs=5  # Warmup period
                )

                acc_temp.append(test_acc)
                kappa_temp.append(kappa_score)

                run_time = time.time() - run_start_time
                print(f"Run {itr + 1} completed in {run_time:.1f}s")
                print(f"   Test Accuracy: {test_acc:.4f}")
                print(f"   Kappa Score: {kappa_score:.4f}")

                # Save checkpoint after each iteration
                if i < len(acc):
                    acc[i] = acc_temp
                    kappa[i] = kappa_temp
                else:
                    acc.append(acc_temp)
                    kappa.append(kappa_temp)

                save_checkpoint(i, itr + 1, acc, kappa, experiment_start_time)

            except Exception as e:
                print(f"Error in run {itr + 1}: {e}")
                print(f"   Continuing with next run...")
                # Save checkpoint even on error
                save_checkpoint(i, itr, acc, kappa, experiment_start_time)
                continue

        # Reset start_itr for next user count
        start_itr = 0

        if acc_temp:
            if i < len(acc):
                acc[i] = acc_temp
                kappa[i] = kappa_temp
            else:
                acc.append(acc_temp)
                kappa.append(kappa_temp)

            avg_acc = np.mean(acc_temp)
            std_acc = np.std(acc_temp)
            avg_kappa = np.mean(kappa_temp)
            std_kappa = np.std(kappa_temp)

            print(f"\nSummary for {num_users} users:")
            print(f"   Accuracy: {avg_acc:.4f} ± {std_acc:.4f} (n={len(acc_temp)})")
            print(f"   Kappa:    {avg_kappa:.4f} ± {std_kappa:.4f} (n={len(kappa_temp)})")

            save_intermediate_results(acc, kappa, i + 1)
            print(f"Intermediate results saved and plotted")

        else:
            print(f"Warning: No successful runs for {num_users} users")

        # Final checkpoint for this user count
        save_checkpoint(i + 1, 0, acc, kappa, experiment_start_time)

        # Estimate remaining time
        if i > start_idx:
            elapsed_time = time.time() - experiment_start
            avg_time_per_user_count = elapsed_time / (i - start_idx + 1)
            remaining_user_counts = len(user_counts) - i - 1
            est_remaining_time = remaining_user_counts * avg_time_per_user_count
            print(f"Estimated remaining time: {est_remaining_time / 3600:.1f} hours")

except KeyboardInterrupt:
    print("\n\nExperiment interrupted by user. Saving current progress...")
    save_intermediate_results(acc, kappa, len(acc))

except Exception as e:
    print(f"\n\nUnexpected error: {e}")
    print("Saving current progress...")
    save_intermediate_results(acc, kappa, len(acc))
    raise

# Final results processing
if len(acc) > 0:
    print(f"\n{'=' * 60}")
    print(f"Processing final results...")
    print(f"{'=' * 60}")
    print(f"Results collected: {len(acc)} user counts")

    # Debug: Print the structure of acc and kappa
    for i, (a_list, k_list) in enumerate(zip(acc, kappa)):
        print(f"User count {user_counts[i]:2d}: {len(a_list)} accuracy values, {len(k_list)} kappa values")

    # Save final results using pickle (most reliable)
    final_results = {
        'test_acc_list': acc,
        'kappa_score_list': kappa,
        'user_counts': user_counts[:len(acc)],
        'normalization_method': NORMALIZATION_METHOD,
        'model_type': MODEL_TYPE,
        'experiment_info': {
            'total_user_counts_tested': len(acc),
            'variable_name': variable_name,
            'model_name': model_name,
            'epochs_per_run': 100,
            'runs_per_user_count': RUNS_PER_USER_COUNT,
            'max_users_tested': max(user_counts[:len(acc)]) if acc else 0,
            'data_format': '2D_spectrograms',
            'split_type': 'session_based'
        }
    }

    # Save as pickle
    import pickle

    pickle_file = os.path.join(OUTPUT_PATH, f"{model_name}.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(final_results, f)

    # Also try to save as npz with padded arrays (optional)
    try:
        max_length = max(len(sublist) for sublist in acc) if acc else 0

        if max_length > 0:
            # Create padded arrays
            acc_padded = np.full((len(acc), max_length), np.nan)
            kappa_padded = np.full((len(kappa), max_length), np.nan)

            for i, sublist in enumerate(acc):
                if len(sublist) > 0:
                    acc_padded[i, :len(sublist)] = sublist
            for i, sublist in enumerate(kappa):
                if len(sublist) > 0:
                    kappa_padded[i, :len(sublist)] = sublist

            # Save final results as npz too
            output_file = os.path.join(OUTPUT_PATH, f"{model_name}.npz")
            np.savez(output_file,
                     test_acc=acc_padded,
                     kappa_score=kappa_padded,
                     user_counts=np.array(user_counts[:len(acc)]),
                     max_runs_per_user_count=max_length)
            print(f"Results saved to both {pickle_file} and {output_file}")
            print(f"Padded array shapes: acc={acc_padded.shape}, kappa={kappa_padded.shape}")
        else:
            print(f"Results saved to {pickle_file}")
    except Exception as e:
        print(f"Could not save .npz format: {e}")
        print(f"Results saved to {pickle_file}")

    # Create final plots
    try:
        create_plots(acc, kappa, len(acc))
        print(f"Final plots saved")
    except Exception as e:
        print(f"Could not create final plots: {e}")

    # Print final summary statistics
    print(f"\nFINAL EXPERIMENT SUMMARY - 2D SPECTROGRAMS (USER SCALING)")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_TYPE} with {NORMALIZATION_METHOD} normalization")
    print(f"Split method: Session-based (10 train, 2 val, 2 test sessions per user)")
    print(f"Epochs per run: 100")
    print(f"{'=' * 60}")

    for i, (a_list, k_list) in enumerate(zip(acc, kappa)):
        if len(a_list) > 0 and len(k_list) > 0:
            print(f"Users {user_counts[i]:3d}: Acc={np.mean(a_list):.4f}±{np.std(a_list):.4f}, "
                  f"Kappa={np.mean(k_list):.4f}±{np.std(k_list):.4f}")

    # Calculate overall performance
    if len(acc) > 0:
        all_acc = [val for sublist in acc for val in sublist]
        all_kappa = [val for sublist in kappa for val in sublist]
        if len(all_acc) > 0:
            print(f"\nOverall Performance:")
            print(f"Mean Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
            print(f"Mean Kappa:    {np.mean(all_kappa):.4f} ± {np.std(all_kappa):.4f}")
            print(f"Total experiments: {len(all_acc)}")
