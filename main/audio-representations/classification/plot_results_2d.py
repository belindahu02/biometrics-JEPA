# =============================================
# ENHANCED plot_results.py for 2D Spectrograms with Checkpointing
# =============================================

from trainers_2d import spectrogram_trainer_2d  # Updated import for 2D trainer
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime

# Configuration
# DATA_PATH = "/app/data/grouped_embeddings"
# OUTPUT_PATH = "/app/data/graph_data_2d"
# GRAPH_PATH = "/app/data/graph_2d"
# CHECKPOINT_PATH = "/app/data/graph_checkpoints_2d"
#
DATA_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/classification_input"
OUTPUT_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graph_data_2d"
GRAPH_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graphs_2d"
CHECKPOINT_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graph_checkpoints_2d"

# Create directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

USER_IDS = list(range(1, 2))  # Increased to more users for better evaluation
NORMALIZATION_METHOD = 'none'  # Changed from CONVERSION_METHOD to NORMALIZATION_METHOD
MODEL_TYPE = 'full'  # Options: 'lightweight', 'full'

variable_name = "samples per user"
model_name = f"spectrogram_2d_{NORMALIZATION_METHOD}_{MODEL_TYPE}"
variable_percentages = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
TOTAL_SAMPLES_PER_USER = 133 #142
variable = [max(1, round(p / 100 * TOTAL_SAMPLES_PER_USER)) for p in variable_percentages]

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
        'normalization_method': NORMALIZATION_METHOD,  # Updated field name
        'model_type': MODEL_TYPE,
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
                acc, kappa = [], []

        print(f"Resuming from checkpoint: {start_idx}/{len(variable)} samples completed")
        print(f"Last checkpoint: {checkpoint_data.get('last_checkpoint_time', 'unknown')}")

        return start_idx, start_itr, acc, kappa

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, [], []


def save_intermediate_results(acc, kappa, current_idx):
    """Save intermediate results that can be plotted, handling failed runs"""
    # Check if we have any data at all
    if len(acc) == 0 and len(kappa) == 0:
        print("⚠️ No data to save yet (acc and kappa are both empty)")
        return

    # Ensure we have at least some valid data
    valid_acc_count = sum(1 for sublist in acc if len(sublist) > 0)
    valid_kappa_count = sum(1 for sublist in kappa if len(sublist) > 0)

    if valid_acc_count == 0 and valid_kappa_count == 0:
        print("⚠️ No valid runs completed yet (all sublists are empty)")
        return

    print(
        f"💾 Saving intermediate results: {valid_acc_count} valid accuracy runs, {valid_kappa_count} valid kappa runs")

    # Save using pickle to handle variable lengths and empty sublists
    intermediate_file = os.path.join(OUTPUT_PATH, f"{model_name}_intermediate.pkl")
    import pickle

    intermediate_data = {
        'test_acc_list': acc,
        'kappa_score_list': kappa,
        'variables_completed': variable[:current_idx],
        'valid_acc_runs': valid_acc_count,
        'valid_kappa_runs': valid_kappa_count,
        'total_attempted_runs': len(acc) * 10,  # Assuming 10 runs per sample size
        'save_timestamp': datetime.now().isoformat()
    }

    try:
        with open(intermediate_file, 'wb') as f:
            pickle.dump(intermediate_data, f)
        print(f"✅ Intermediate data saved to {intermediate_file}")
    except Exception as e:
        print(f"❌ Failed to save intermediate data: {e}")
        return

    # Create intermediate plots only if we have valid data
    try:
        create_plots(acc, kappa, current_idx, suffix="_intermediate")
        print(f"✅ Intermediate plots created")
    except Exception as e:
        print(f"⚠️ Could not create intermediate plots: {e}")


def create_plots(acc, kappa, num_completed, suffix=""):
    """Create plots with current results, handling empty sublists from failed runs"""
    if len(acc) == 0 and len(kappa) == 0:
        print("⚠️ No data available for plotting")
        return

    current_variables = variable[:num_completed]

    # Handle variable-length sublists and empty sublists from failed runs
    kappa_max = []
    acc_max = []
    valid_variables = []

    for i in range(min(len(acc), len(kappa), len(current_variables))):
        has_valid_acc = i < len(acc) and len(acc[i]) > 0
        has_valid_kappa = i < len(kappa) and len(kappa[i]) > 0

        # Only include data points where we have at least one valid result
        if has_valid_acc or has_valid_kappa:
            valid_variables.append(current_variables[i])

            if has_valid_acc:
                acc_max.append(np.max(acc[i]))
            else:
                acc_max.append(np.nan)  # Use NaN for missing data

            if has_valid_kappa:
                kappa_max.append(np.max(kappa[i]))
            else:
                kappa_max.append(np.nan)  # Use NaN for missing data

    if len(valid_variables) == 0:
        print("⚠️ No valid data points available for plotting")
        return

    print(f"📊 Plotting {len(valid_variables)} data points (some may have NaN values)")

    # Kappa plot
    plt.figure(figsize=(12, 8))
    # Plot with NaN handling - matplotlib will skip NaN points
    plt.plot(valid_variables, kappa_max, 'b-o', label=f'{model_name}', linewidth=2, markersize=6)
    plt.title(f"Kappa Score vs {variable_name} (2D {NORMALIZATION_METHOD} normalization, {MODEL_TYPE} model){suffix}")
    plt.xlabel(variable_name)
    plt.ylabel("Kappa Score")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add text showing number of valid points
    valid_kappa_count = np.sum(~np.isnan(kappa_max))
    plt.text(0.02, 0.98, f'Valid points: {valid_kappa_count}/{len(valid_variables)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_graph = os.path.join(GRAPH_PATH, f"kappa_2d_{NORMALIZATION_METHOD}_{MODEL_TYPE}{suffix}.jpg")
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(12, 8))
    plt.plot(valid_variables, acc_max, 'r-s', label=f'{model_name}', linewidth=2, markersize=6)
    plt.title(f"Test Accuracy vs {variable_name} (2D {NORMALIZATION_METHOD} normalization, {MODEL_TYPE} model){suffix}")
    plt.xlabel(variable_name)
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add text showing number of valid points
    valid_acc_count = np.sum(~np.isnan(acc_max))
    plt.text(0.02, 0.98, f'Valid points: {valid_acc_count}/{len(valid_variables)}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    output_graph = os.path.join(GRAPH_PATH, f"acc_2d_{NORMALIZATION_METHOD}_{MODEL_TYPE}{suffix}.jpg")
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Plots saved with {valid_acc_count} valid accuracy points and {valid_kappa_count} valid kappa points")


def compare_with_1d_results(acc, kappa):
    """Create comparison plots with 1D PCA results if available"""
    try:
        # Try to load 1D PCA results for comparison
        pca_file = os.path.join(OUTPUT_PATH, "spectrogram_1d_converted_pca.pkl")
        if os.path.exists(pca_file):
            import pickle
            with open(pca_file, 'rb') as f:
                pca_results = pickle.load(f)

            pca_acc = pca_results.get('test_acc_list', [])
            pca_kappa = pca_results.get('kappa_score_list', [])
            pca_variables = pca_results.get('variables', [])

            # Get max values for comparison
            pca_acc_max = [np.max(sublist) for sublist in pca_acc if len(sublist) > 0]
            pca_kappa_max = [np.max(sublist) for sublist in pca_kappa if len(sublist) > 0]

            # Get current 2D results
            acc_max = [np.max(sublist) for sublist in acc if len(sublist) > 0]
            kappa_max = [np.max(sublist) for sublist in kappa if len(sublist) > 0]

            # Ensure matching lengths for comparison
            min_len = min(len(acc_max), len(pca_acc_max))
            if min_len > 0:
                # Comparison plots
                plt.figure(figsize=(15, 6))

                # Accuracy comparison
                plt.subplot(1, 2, 1)
                plt.plot(variable[:min_len], acc_max[:min_len], 'b-o',
                         label=f'2D {NORMALIZATION_METHOD} ({MODEL_TYPE})', linewidth=2)
                plt.plot(pca_variables[:min_len], pca_acc_max[:min_len], 'r-s', label='1D PCA', linewidth=2)
                plt.title(f"Accuracy Comparison: 2D vs 1D PCA")
                plt.xlabel(variable_name)
                plt.ylabel("Test Accuracy")
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Kappa comparison
                plt.subplot(1, 2, 2)
                plt.plot(variable[:min_len], kappa_max[:min_len], 'b-o',
                         label=f'2D {NORMALIZATION_METHOD} ({MODEL_TYPE})', linewidth=2)
                plt.plot(pca_variables[:min_len], pca_kappa_max[:min_len], 'r-s', label='1D PCA', linewidth=2)
                plt.title(f"Kappa Comparison: 2D vs 1D PCA")
                plt.xlabel(variable_name)
                plt.ylabel("Kappa Score")
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.tight_layout()
                comparison_file = os.path.join(GRAPH_PATH, f"comparison_2d_vs_1d_{MODEL_TYPE}.jpg")
                plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"✅ Comparison plot saved to {comparison_file}")

                # Print numerical comparison
                print(f"\n📊 Performance Comparison (last {min_len} data points):")
                avg_acc_2d = np.mean(acc_max[:min_len])
                avg_acc_1d = np.mean(pca_acc_max[:min_len])
                avg_kappa_2d = np.mean(kappa_max[:min_len])
                avg_kappa_1d = np.mean(pca_kappa_max[:min_len])

                print(f"Average Accuracy - 2D: {avg_acc_2d:.4f}, 1D PCA: {avg_acc_1d:.4f}")
                print(f"Average Kappa    - 2D: {avg_kappa_2d:.4f}, 1D PCA: {avg_kappa_1d:.4f}")
                print(
                    f"Improvement      - Acc: {(avg_acc_2d - avg_acc_1d) * 100:.2f}%, Kappa: {(avg_kappa_2d - avg_kappa_1d) * 100:.2f}%")

    except Exception as e:
        print(f"⚠️ Could not create comparison plots: {e}")


# Main execution
print(f"Running 2D spectrogram experiments")
print(f"Configuration:")
print(f"  - Normalization method: {NORMALIZATION_METHOD}")
print(f"  - Model type: {MODEL_TYPE}")
print(f"  - Users: {len(USER_IDS)} ({min(USER_IDS)}-{max(USER_IDS)})")
print(f"  - Testing {len(variable)} different sample sizes...")
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

print(f"Starting from sample index {start_idx}, iteration {start_itr}")

# Track experiment timing
experiment_start = time.time()

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
                # Call 2D trainer with updated parameters

                test_acc, kappa_score = spectrogram_trainer_2d(
                    samples_per_user=samples_per_user,  # Start smaller than before
                    data_path=DATA_PATH,
                    user_ids=USER_IDS,
                    normalization_method=NORMALIZATION_METHOD,
                    model_type=MODEL_TYPE,  # Use lightweight model
                    batch_size=8 if MODEL_TYPE == 'full' else 16,  # Adjust batch size based on model
                    epochs=100,
                    lr=0.001,
                    device='cuda',  # or 'cpu'
                    use_augmentation=True,  # Start without augmentation
                    save_model_checkpoints=True,
                    checkpoint_every=10,
                    max_cache_size=50  # NEW PARAMETER - keep only 50 spectrograms in cache
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
            save_intermediate_results(acc, kappa, i + 1)
            print(f"💾 Intermediate results saved and plotted")

        else:
            print(f"⚠️ No successful runs for {samples_per_user} samples/user")

        # Final checkpoint for this sample size
        save_checkpoint(i + 1, 0, acc, kappa, experiment_start_time)

        # Estimate remaining time
        if i > start_idx:
            elapsed_time = time.time() - experiment_start
            avg_time_per_sample = elapsed_time / (i - start_idx + 1)
            remaining_samples = len(variable) - i - 1
            est_remaining_time = remaining_samples * avg_time_per_sample
            print(f"⏱️ Estimated remaining time: {est_remaining_time / 3600:.1f} hours")

except KeyboardInterrupt:
    print("\n\n🛑 Experiment interrupted by user. Saving current progress...")
    save_intermediate_results(acc, kappa, len(acc))

except Exception as e:
    print(f"\n\n💥 Unexpected error: {e}")
    print("Saving current progress...")
    save_intermediate_results(acc, kappa, len(acc))
    raise

# Final results processing
if len(acc) >= len(variable):
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
        'normalization_method': NORMALIZATION_METHOD,  # Updated field name
        'model_type': MODEL_TYPE,
        'experiment_info': {
            'total_sample_sizes_tested': len(acc),
            'variable_name': variable_name,
            'model_name': model_name,
            'epochs_per_run': 100,
            'runs_per_sample_size': 10,
            'user_ids': USER_IDS,
            'data_format': '2D_spectrograms'
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
        print(f"⚠️ Could not save .npz format: {e}")
        print(f"✅ Results saved to {pickle_file}")

    # Create final plots
    try:
        create_plots(acc, kappa, len(acc))
        print(
            f"✅ Final plots saved as graphs/acc_2d_{NORMALIZATION_METHOD}_{MODEL_TYPE}.jpg and graphs/kappa_2d_{NORMALIZATION_METHOD}_{MODEL_TYPE}.jpg")
    except Exception as e:
        print(f"⚠️ Could not create final plots: {e}")

    # Create comparison plots with 1D results if available
    try:
        compare_with_1d_results(acc, kappa)
    except Exception as e:
        print(f"⚠️ Could not create comparison plots: {e}")

    # Print final summary statistics
    print(f"\n📈 FINAL EXPERIMENT SUMMARY - 2D SPECTROGRAMS")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_TYPE} with {NORMALIZATION_METHOD} normalization")
    print(f"Users: {len(USER_IDS)}, Epochs per run: 100")
    print(f"{'=' * 60}")

    for i, (a_list, k_list) in enumerate(zip(acc, kappa)):
        if len(a_list) > 0 and len(k_list) > 0:
            print(f"Samples/user {variable[i]:2d}: Acc={np.mean(a_list):.4f}±{np.std(a_list):.4f}, "
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

else:
    print("❌ No results to save or plot")