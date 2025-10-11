from trainers import *
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Base output directory on host
base_dir = "/app/data/experiments/scaling/baselines"
# base_dir = "test/"

# Make sure these exist
graph_data_dir = os.path.join(base_dir, "da/graph_data")
graphs_dir = os.path.join(base_dir, "da/graphs")
checkpoint_dir = os.path.join(base_dir, "da/checkpoints")

os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

variable_name = "number of users"
model_name = "eeg_mmi_user_scaling_da"
iterations = 3

# Variable is now number of users instead of samples per user
variable = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109]

# Checkpoint file paths
checkpoint_file = os.path.join(checkpoint_dir, f"{model_name}_checkpoint.json")
results_file = os.path.join(checkpoint_dir, f"{model_name}_results.npz")


def load_checkpoint():
    """Load checkpoint if it exists"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"\n{'=' * 70}")
        print("RESUMING FROM CHECKPOINT")
        print(f"Last completed: {checkpoint['last_num_users']} users, iteration {checkpoint['last_iteration']}")
        print(f"{'=' * 70}\n")
        return checkpoint
    return None


def save_checkpoint(num_users, iteration, acc, kappa):
    """Save checkpoint after each iteration"""
    checkpoint = {
        'last_num_users': num_users,
        'last_iteration': iteration,
        'completed_users': variable[:variable.index(num_users) + 1] if iteration == iterations - 1 else variable[
                                                                                                        :variable.index(
                                                                                                            num_users)]
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)

    # Save results
    np.savez(
        results_file,
        test_acc=np.array(acc),
        kappa_score=np.array(kappa),
        num_users=variable[:len(acc)]
    )


def load_results():
    """Load partial results if they exist"""
    if os.path.exists(results_file):
        data = np.load(results_file)
        acc = data['test_acc'].tolist()
        kappa = data['kappa_score'].tolist()
        return acc, kappa
    return [], []


def should_skip(num_users, iteration, checkpoint):
    """Determine if this iteration should be skipped based on checkpoint"""
    if checkpoint is None:
        return False

    last_users = checkpoint['last_num_users']
    last_iter = checkpoint['last_iteration']

    # Skip if we've completed this user count
    if num_users < last_users:
        return True

    # Skip if we're on the same user count but this iteration is already done
    if num_users == last_users and iteration <= last_iter:
        return True

    return False


def plot_results(acc, kappa, completed_users):
    """Generate all plots with current results"""
    acc_array = np.array(acc)
    kappa_array = np.array(kappa)

    # Plot maximum kappa score
    kappa_max = np.max(kappa_array, axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(completed_users, kappa_max, 'm-o', label=model_name, linewidth=2, markersize=8)
    plt.title(f"Kappa Score vs {variable_name}", fontsize=14, fontweight='bold')
    plt.xlabel(variable_name, fontsize=12)
    plt.ylabel("Kappa Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'kappa.jpg'), dpi=300)
    plt.close()

    # Plot maximum test accuracy
    acc_max = np.max(acc_array, axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(completed_users, acc_max, 'm-o', label=model_name, linewidth=2, markersize=8)
    plt.title(f"Test Accuracy vs {variable_name}", fontsize=14, fontweight='bold')
    plt.xlabel(variable_name, fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'acc.jpg'), dpi=300)
    plt.close()

    # Plot mean with error bars
    kappa_mean = np.mean(kappa_array, axis=1)
    kappa_std = np.std(kappa_array, axis=1)
    acc_mean = np.mean(acc_array, axis=1)
    acc_std = np.std(acc_array, axis=1)

    # Kappa with error bars
    plt.figure(figsize=(12, 8))
    plt.errorbar(completed_users, kappa_mean, yerr=kappa_std, fmt='m-o',
                 label=model_name, linewidth=2, markersize=8, capsize=5)
    plt.title(f"Kappa Score vs {variable_name} (Mean ± Std)", fontsize=14, fontweight='bold')
    plt.xlabel(variable_name, fontsize=12)
    plt.ylabel("Kappa Score", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'kappa_mean_std.jpg'), dpi=300)
    plt.close()

    # Accuracy with error bars
    plt.figure(figsize=(12, 8))
    plt.errorbar(completed_users, acc_mean, yerr=acc_std, fmt='m-o',
                 label=model_name, linewidth=2, markersize=8, capsize=5)
    plt.title(f"Test Accuracy vs {variable_name} (Mean ± Std)", fontsize=14, fontweight='bold')
    plt.xlabel(variable_name, fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'acc_mean_std.jpg'), dpi=300)
    plt.close()

    print(f"Plots updated with {len(completed_users)} data points")


# Load checkpoint and partial results
checkpoint = load_checkpoint()
acc, kappa = load_results()

print(f"Starting experiments with {len(variable)} user counts")
print(f"User counts to test: {variable}")

for num_users in variable:
    # If we already completed this user count, load the results and continue
    if checkpoint and num_users in checkpoint.get('completed_users', []):
        print(f"\n{'=' * 70}")
        print(f"Skipping {num_users} users (already completed)")
        print(f"{'=' * 70}\n")
        continue

    acc_temp = []
    kappa_temp = []

    print(f"\n{'=' * 70}")
    print(f"Running experiments with {num_users} users ({iterations} iterations)")
    print(f"{'=' * 70}\n")

    for itr in range(iterations):
        # Skip if already completed
        if should_skip(num_users, itr, checkpoint):
            print(f"Skipping iteration {itr + 1}/{iterations} for {num_users} users (already completed)")
            continue

        print(f"\nIteration {itr + 1}/{iterations} for {num_users} users")

        try:
            test_acc, kappa_score = trainer(num_users)
            acc_temp.append(test_acc)
            kappa_temp.append(kappa_score)
            print(f"Iteration {itr + 1} - Acc: {test_acc:.4f}, Kappa: {kappa_score:.4f}")

            # Save checkpoint after each iteration
            if len(acc) == variable.index(num_users):
                # First iteration for this user count
                acc.append(acc_temp.copy())
                kappa.append(kappa_temp.copy())
            else:
                # Additional iterations
                acc[variable.index(num_users)] = acc_temp.copy()
                kappa[variable.index(num_users)] = kappa_temp.copy()

            save_checkpoint(num_users, itr, acc, kappa)
            print(f"Checkpoint saved after iteration {itr + 1}")

        except Exception as e:
            print(f"\n{'!' * 70}")
            print(f"ERROR during iteration {itr + 1} for {num_users} users:")
            print(f"{str(e)}")
            print(f"Progress has been saved. You can resume from this point.")
            print(f"{'!' * 70}\n")
            raise

    # After completing all iterations for this user count
    if len(acc_temp) == iterations:
        print(f"\nCompleted {num_users} users - Avg Acc: {np.mean(acc_temp):.4f}, Avg Kappa: {np.mean(kappa_temp):.4f}")

        # Generate updated plots
        completed_users = variable[:len(acc)]
        plot_results(acc, kappa, completed_users)

# Final save
acc_array = np.array(acc)
kappa_array = np.array(kappa)

np.savez(
    os.path.join(graph_data_dir, model_name + ".npz"),
    test_acc=acc_array,
    kappa_score=kappa_array,
    num_users=variable[:len(acc_array)]
)

print(f"\nFinal results saved:")
print(f"Accuracy shape: {acc_array.shape}")
print(f"Kappa shape: {kappa_array.shape}")

# Generate final plots
plot_results(acc, kappa, variable[:len(acc)])

print("\n" + "=" * 70)
print("All experiments completed!")
print("=" * 70)

# Clean up checkpoint files on successful completion
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print("Checkpoint file removed (experiments completed successfully)")