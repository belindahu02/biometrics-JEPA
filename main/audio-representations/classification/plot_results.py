# =============================================
# ENHANCED plot_results.py with Checkpointing
# =============================================

from trainers import spectrogram_trainer
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime

# Configuration
# DATA_PATH = "/app/data/grouped_embeddings"
# OUTPUT_PATH = "/app/data/graph_data"
# GRAPH_PATH = "/app/data/graphs"
# CHECKPOINT_PATH = "/app/data/graph_checkpoints"

DATA_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/classification_input"
OUTPUT_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graph_data"
GRAPH_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graphs"
CHECKPOINT_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graph_checkpoints"

# Create directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

USER_IDS = list(range(1, 2))
CONVERSION_METHOD = 'pca'

variable_name = "samples per user"
model_name = f"spectrogram_1d_converted_{CONVERSION_METHOD}"
variable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50, 55, 60]

# Checkpoint file names
checkpoint_file = os.path.join(CHECKPOINT_PATH, f"{model_name}_checkpoint.json")
results_backup_file = os.path.join(CHECKPOINT_PATH, f"{model_name}_results_backup.pkl")


def save_checkpoint(current_idx, current_itr, acc, kappa, experiment_start_time):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        'current_sample_idx': current_idx,
        'current_iteration': current_itr,
        'completed_samples': current_idx,
        'total_samples': len(variable),
        'experiment_start_time': experiment_start_time,
        'last_checkpoint_time': datetime.now().isoformat(),
        'conversion_method': CONVERSION_METHOD,
        'variable_values_completed': variable[:current_idx],
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
            'completed_variables': variable[:current_idx]
        }

        with open(results_backup_file.replace('.npz', '.pkl'), 'wb') as f:
            pickle.dump(backup_data, f)

    print(f"Checkpoint saved: {current_idx}/{len(variable)} samples completed")


def load_checkpoint():
    """Load previous progress from checkpoint file"""
    if not os.path.exists(checkpoint_file):
        return None, None, [], []

    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)

        start_idx = checkpoint_data.get('current_sample_idx', 0)
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
                # Try loading old npz format as fallback
                old_npz_file = results_backup_file.replace('.pkl', '.npz')
                if os.path.exists(old_npz_file):
                    try:
                        backup_data = np.load(old_npz_file, allow_pickle=True)
                        if 'test_acc_list' in backup_data:
                            acc = backup_data['test_acc_list'].tolist()
                        elif 'test_acc' in backup_data:
                            acc = backup_data['test_acc'].tolist()

                        if 'kappa_score_list' in backup_data:
                            kappa = backup_data['kappa_score_list'].tolist()
                        elif 'kappa_score' in backup_data:
                            kappa = backup_data['kappa_score'].tolist()
                    except Exception as e2:
                        print(f"Warning: Could not load old npz backup either: {e2}")
                        acc, kappa = [], []

        print(f"Resuming from checkpoint: {start_idx}/{len(variable)} samples completed")
        print(f"Last checkpoint: {checkpoint_data.get('last_checkpoint_time', 'unknown')}")

        return start_idx, start_itr, acc, kappa

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, [], []


def save_intermediate_results(acc, kappa, current_idx):
    """Save intermediate results that can be plotted"""
    if len(acc) == 0:
        return

    # Save using pickle to handle variable lengths
    intermediate_file = os.path.join(OUTPUT_PATH, f"{model_name}_intermediate.pkl")
    import pickle

    intermediate_data = {
        'test_acc_list': acc,
        'kappa_score_list': kappa,
        'variables_completed': variable[:current_idx]
    }

    with open(intermediate_file, 'wb') as f:
        pickle.dump(intermediate_data, f)

    # Create intermediate plots
    create_plots(acc, kappa, current_idx, suffix="_intermediate")


def create_plots(acc, kappa, num_completed, suffix=""):
    """Create plots with current results"""
    if len(acc) == 0:
        return

    current_variables = variable[:num_completed]

    # Handle variable-length sublists by taking maximum values when available
    kappa_max = []
    acc_max = []

    for i in range(min(len(acc), len(kappa))):
        if len(acc[i]) > 0:
            acc_max.append(np.max(acc[i]))
        if len(kappa[i]) > 0:
            kappa_max.append(np.max(kappa[i]))

    # Ensure we have matching lengths
    plot_variables = current_variables[:min(len(kappa_max), len(acc_max))]
    kappa_max = kappa_max[:len(plot_variables)]
    acc_max = acc_max[:len(plot_variables)]

    if len(kappa_max) == 0 or len(acc_max) == 0:
        print("No valid results to plot yet")
        return

    # Kappa plot
    plt.figure(figsize=(12, 8))
    plt.plot(plot_variables, kappa_max, 'm', label=model_name, linewidth=2, marker='o')
    plt.title(f"Kappa Score vs {variable_name} ({CONVERSION_METHOD} conversion){suffix}")
    plt.xlabel(variable_name)
    plt.ylabel("Kappa Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_graph = os.path.join(GRAPH_PATH, f"kappa_{CONVERSION_METHOD}{suffix}.jpg")
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(12, 8))
    plt.plot(plot_variables, acc_max, 'm', label=model_name, linewidth=2, marker='o')
    plt.title(f"Test Accuracy vs {variable_name} ({CONVERSION_METHOD} conversion){suffix}")
    plt.xlabel(variable_name)
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_graph = os.path.join(GRAPH_PATH, f"acc_{CONVERSION_METHOD}{suffix}.jpg")
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()


# Main execution
print(f"Running experiments with conversion method: {CONVERSION_METHOD}")
print(f"Testing {len(variable)} different sample sizes...")

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

print(f"Starting from sample index {start_idx}, iteration {start_itr}")

try:
    for i in range(start_idx, len(variable)):
        samples_per_user = variable[i]
        print(f"\n--- Testing {samples_per_user} samples per user ({i + 1}/{len(variable)}) ---")

        # If resuming mid-sample, use existing results, otherwise start fresh
        if i == start_idx and start_itr > 0:
            acc_temp = acc[i] if i < len(acc) else []
            kappa_temp = kappa[i] if i < len(kappa) else []
        else:
            acc_temp = []
            kappa_temp = []

        for itr in range(start_itr if i == start_idx else 0, 10):
            print(f"  Run {itr + 1}/10...")
            try:
                test_acc, kappa_score = spectrogram_trainer(
                    samples_per_user=samples_per_user,
                    data_path=DATA_PATH,
                    user_ids=USER_IDS,
                    conversion_method=CONVERSION_METHOD
                )
                acc_temp.append(test_acc)
                kappa_temp.append(kappa_score)
                print(f"    Acc: {test_acc:.4f}, Kappa: {kappa_score:.4f}")

                # Save checkpoint after each iteration
                if i < len(acc):
                    acc[i] = acc_temp
                    kappa[i] = kappa_temp
                else:
                    acc.append(acc_temp)
                    kappa.append(kappa_temp)

                save_checkpoint(i, itr + 1, acc, kappa, experiment_start_time)

            except Exception as e:
                print(f"    Error in run {itr + 1}: {e}")
                # Save checkpoint even on error
                save_checkpoint(i, itr, acc, kappa, experiment_start_time)
                continue

        # Reset start_itr for next sample size
        start_itr = 0

        if acc_temp:
            if i >= len(acc):
                acc.append(acc_temp)
                kappa.append(kappa_temp)
            print(f"  Completed {samples_per_user} samples/user - Avg Acc: {np.mean(acc_temp):.4f}")

            # Save intermediate results and plots every few iterations
            if (i + 1) % 3 == 0:  # Every 3 sample sizes
                save_intermediate_results(acc, kappa, i + 1)

        else:
            print(f"  No successful runs for {samples_per_user} samples/user")

        # Final checkpoint for this sample size
        save_checkpoint(i + 1, 0, acc, kappa, experiment_start_time)

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
    # Convert to padded arrays for final saving (pad shorter lists with NaN)
    max_length = max(len(sublist) for sublist in acc) if acc else 0

    if max_length > 0:
        # Create padded arrays
        acc_padded = np.full((len(acc), max_length), np.nan)
        kappa_padded = np.full((len(kappa), max_length), np.nan)

        for i, sublist in enumerate(acc):
            acc_padded[i, :len(sublist)] = sublist
        for i, sublist in enumerate(kappa):
            kappa_padded[i, :len(sublist)] = sublist

        # Save final results
        output_file = os.path.join(OUTPUT_PATH, f"{model_name}.npz")
        np.savez(output_file,
                 test_acc=acc_padded,
                 kappa_score=kappa_padded,
                 test_acc_list=acc,  # Also save as lists
                 kappa_score_list=kappa,
                 variables=variable[:len(acc)],
                 allow_pickle=True)
        print(f"\nFinal results saved to {output_file}")
        print(f"Accuracy shape: {acc_padded.shape}")
        print(f"Kappa shape: {kappa_padded.shape}")

        # Create final plots
        create_plots(acc, kappa, len(acc))
        print(f"Final plots saved as graphs/kappa_{CONVERSION_METHOD}.jpg and graphs/acc_{CONVERSION_METHOD}.jpg")

    # Clean up checkpoint files
    try:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        if os.path.exists(results_backup_file):
            os.remove(results_backup_file)
        # Also clean up old npz files if they exist
        old_npz_file = results_backup_file.replace('.pkl', '.npz')
        if os.path.exists(old_npz_file):
            os.remove(old_npz_file)
        print("Checkpoint files cleaned up")
    except Exception as e:
        print(f"Warning: Could not clean up some checkpoint files: {e}")

else:
    print("No results to save or plot")