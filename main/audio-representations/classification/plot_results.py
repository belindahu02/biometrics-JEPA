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

# # Configuration
DATA_PATH = "/app/data/grouped_embeddings"
OUTPUT_PATH = "/app/data/graph_data"
GRAPH_PATH = "/app/data/graphs"
CHECKPOINT_PATH = "/app/data/graph_checkpoints"

# DATA_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/classification_input"
# OUTPUT_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graph_data"
# GRAPH_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graphs"
# CHECKPOINT_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graph_checkpoints"

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
print(f"Training will run for 150 epochs per experiment (increased from 100)")

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
        print(f"\n{'=' * 60}")
        print(f"Testing {samples_per_user} samples per user ({i + 1}/{len(variable)})")
        print(f"{'=' * 60}")

        # If resuming mid-sample, use existing results, otherwise start fresh
        if i == start_idx and start_itr > 0:
            acc_temp = acc[i] if i < len(acc) else []
            kappa_temp = kappa[i] if i < len(kappa) else []
        else:
            acc_temp = []
            kappa_temp = []

        for itr in range(start_itr if i == start_idx else 0, 10):
            run_start_time = time.time()
            print(f"\n--- Run {itr + 1}/10 for {samples_per_user} samples/user ---")

            try:
                # Call trainer with increased epochs
                test_acc, kappa_score = spectrogram_trainer(
                    samples_per_user=samples_per_user,
                    data_path=DATA_PATH,
                    user_ids=USER_IDS,
                    conversion_method=CONVERSION_METHOD,
                    epochs=150,  # Increased from default
                    batch_size=8,
                    lr=0.001
                )

                acc_temp.append(test_acc)
                kappa_temp.append(kappa_score)

                run_time = time.time() - run_start_time
                print(f"✅ Run {itr + 1} completed in {run_time:.1f}s")
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
                print(f"❌ Error in run {itr + 1}: {e}")
                print(f"   Continuing with next run...")
                # Save checkpoint even on error
                save_checkpoint(i, itr, acc, kappa, experiment_start_time)
                continue

        # Reset start_itr for next sample size
        start_itr = 0

        if acc_temp:
            if i >= len(acc):
                acc.append(acc_temp)
                kappa.append(kappa_temp)

            avg_acc = np.mean(acc_temp)
            std_acc = np.std(acc_temp)
            avg_kappa = np.mean(kappa_temp)
            std_kappa = np.std(kappa_temp)

            print(f"\n📊 Summary for {samples_per_user} samples/user:")
            print(f"   Accuracy: {avg_acc:.4f} ± {std_acc:.4f} (n={len(acc_temp)})")
            print(f"   Kappa:    {avg_kappa:.4f} ± {std_kappa:.4f} (n={len(kappa_temp)})")

            # Save intermediate results and plots every few iterations
            if (i + 1) % 3 == 0:  # Every 3 sample sizes
                save_intermediate_results(acc, kappa, i + 1)
                print(f"💾 Intermediate results saved and plotted")

        else:
            print(f"⚠️  No successful runs for {samples_per_user} samples/user")

        # Final checkpoint for this sample size
        save_checkpoint(i + 1, 0, acc, kappa, experiment_start_time)

        # Estimate remaining time
        if i > start_idx:
            avg_time_per_sample = (time.time() - time.time()) / (i - start_idx + 1)  # This needs to be tracked properly
            remaining_samples = len(variable) - i - 1
            est_remaining_time = remaining_samples * avg_time_per_sample
            print(f"⏱️  Estimated remaining time: {est_remaining_time / 3600:.1f} hours")

except KeyboardInterrupt:
    print("\n\n🛑 Experiment interrupted by user. Saving current progress...")
    save_intermediate_results(acc, kappa, len(acc))

except Exception as e:
    print(f"\n\n💥 Unexpected error: {e}")
    print("Saving current progress...")
    save_intermediate_results(acc, kappa, len(acc))
    raise

# Final results processing
if len(acc) > 0:
    print(f"\n{'=' * 60}")
    print(f"Processing final results...")
    print(f"{'=' * 60}")
    print(f"Results collected: {len(acc)} sample sizes")

    # Debug: Print the structure of acc and kappa
    for i, (a_list, k_list) in enumerate(zip(acc, kappa)):
        print(f"Sample size {variable[i]:2d}: {len(a_list)} accuracy values, {len(k_list)} kappa values")

    # Save final results using pickle (most reliable)
    final_results = {
        'test_acc_list': acc,
        'kappa_score_list': kappa,
        'variables': variable[:len(acc)],
        'conversion_method': CONVERSION_METHOD,
        'experiment_info': {
            'total_sample_sizes_tested': len(acc),
            'variable_name': variable_name,
            'model_name': model_name,
            'epochs_per_run': 150,
            'runs_per_sample_size': 10
        }
    }

    # Save as pickle
    import pickle

    pickle_file = os.path.join(OUTPUT_PATH, f"{model_name}.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(final_results, f)

    # Also try to save as npz with padded arrays (optional, for analysis tools that expect numpy)
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
                     variables=np.array(variable[:len(acc)]),
                     max_runs_per_sample=max_length)
            print(f"✅ Results saved to both {pickle_file} and {output_file}")
            print(f"Padded array shapes: acc={acc_padded.shape}, kappa={kappa_padded.shape}")
        else:
            print(f"✅ Results saved to {pickle_file}")
    except Exception as e:
        print(f"⚠️  Could not save .npz format: {e}")
        print(f"✅ Results saved to {pickle_file}")

    # Create final plots
    try:
        create_plots(acc, kappa, len(acc))
        print(f"✅ Final plots saved as graphs/kappa_{CONVERSION_METHOD}.jpg and graphs/acc_{CONVERSION_METHOD}.jpg")
    except Exception as e:
        print(f"⚠️  Could not create final plots: {e}")

    # Print final summary statistics
    print(f"\n📈 FINAL EXPERIMENT SUMMARY")
    print(f"{'=' * 60}")
    for i, (a_list, k_list) in enumerate(zip(acc, kappa)):
        if len(a_list) > 0 and len(k_list) > 0:
            print(f"Samples/user {variable[i]:2d}: Acc={np.mean(a_list):.4f}±{np.std(a_list):.4f}, "
                  f"Kappa={np.mean(k_list):.4f}±{np.std(k_list):.4f}")

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
        print("🧹 Checkpoint files cleaned up")
    except Exception as e:
        print(f"⚠️  Could not clean up some checkpoint files: {e}")

else:
    print("❌ No results to save or plot")