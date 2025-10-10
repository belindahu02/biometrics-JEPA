from trainers import *
import numpy as np
import matplotlib.pyplot as plt

# Base output directory on host
# base_dir = "test/"
base_dir = "/app/data/experiments/scaling/baselines"

# Make sure these exist
graph_data_dir = os.path.join(base_dir, "da/graph_data")
graphs_dir = os.path.join(base_dir, "da/graphs")

os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

variable_name = "number of users"
model_name = "eeg_mmi_user_scaling_da"
iterations = 3

# Variable is now number of users instead of samples per user
variable = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109]

acc = []
kappa = []

for num_users in variable:
    acc_temp = []
    kappa_temp = []

    print(f"\n{'=' * 70}")
    print(f"Running experiments with {num_users} users ({iterations} iterations)")
    print(f"{'=' * 70}\n")

    for itr in range(iterations):
        print(f"\nIteration {itr + 1}/{iterations} for {num_users} users")
        test_acc, kappa_score = trainer(num_users)
        acc_temp.append(test_acc)
        kappa_temp.append(kappa_score)
        print(f"Iteration {itr + 1} - Acc: {test_acc:.4f}, Kappa: {kappa_score:.4f}")

    acc.append(acc_temp)
    kappa.append(kappa_temp)

    # Save intermediate results
    acc_array = np.array(acc)
    kappa_array = np.array(kappa)
    np.savez(
      os.path.join(graph_data_dir, model_name + "_intermediate.npz"),
      test_acc=acc_array,
      kappa_score=kappa_array,
      num_users=variable[:len(acc_array)]
    )
    print(f"\nCompleted {num_users} users - Avg Acc: {np.mean(acc_temp):.4f}, Avg Kappa: {np.mean(kappa_temp):.4f}")

# Convert to arrays
acc = np.array(acc)
kappa = np.array(kappa)

np.savez(os.path.join(graph_data_dir, model_name + ".npz"),
         test_acc=acc, kappa_score=kappa)
print(f"\nFinal results saved:")
print(f"Accuracy shape: {acc.shape}")
print(f"Kappa shape: {kappa.shape}")

# Plot maximum kappa score
kappa_max = np.max(kappa, axis=1)
plt.figure(figsize=(12, 8))
plt.plot(variable, kappa_max, 'm-o', label=model_name, linewidth=2, markersize=8)
plt.title(f"Kappa Score vs {variable_name}", fontsize=14, fontweight='bold')
plt.xlabel(variable_name, fontsize=12)
plt.ylabel("Kappa Score", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'kappa.jpg'), dpi=300)
plt.close()
print("Kappa plot saved to graphs/kappa.jpg")

# Plot maximum test accuracy
acc_max = np.max(acc, axis=1)
plt.figure(figsize=(12, 8))
plt.plot(variable, acc_max, 'm-o', label=model_name, linewidth=2, markersize=8)
plt.title(f"Test Accuracy vs {variable_name}", fontsize=14, fontweight='bold')
plt.xlabel(variable_name, fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'acc.jpg'), dpi=300)
plt.close()
print("Accuracy plot saved to graphs/acc.jpg")

# Plot mean with error bars
kappa_mean = np.mean(kappa, axis=1)
kappa_std = np.std(kappa, axis=1)
acc_mean = np.mean(acc, axis=1)
acc_std = np.std(acc, axis=1)

# Kappa with error bars
plt.figure(figsize=(12, 8))
plt.errorbar(variable, kappa_mean, yerr=kappa_std, fmt='m-o',
             label=model_name, linewidth=2, markersize=8, capsize=5)
plt.title(f"Kappa Score vs {variable_name} (Mean ± Std)", fontsize=14, fontweight='bold')
plt.xlabel(variable_name, fontsize=12)
plt.ylabel("Kappa Score", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'kappa_mean_std.jpg'), dpi=300)
plt.close()
print("Kappa mean/std plot saved to graphs/kappa_mean_std.jpg")

# Accuracy with error bars
plt.figure(figsize=(12, 8))
plt.errorbar(variable, acc_mean, yerr=acc_std, fmt='m-o',
             label=model_name, linewidth=2, markersize=8, capsize=5)
plt.title(f"Test Accuracy vs {variable_name} (Mean ± Std)", fontsize=14, fontweight='bold')
plt.xlabel(variable_name, fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'acc_mean_std.jpg'), dpi=300)
plt.close()
print("Accuracy mean/std plot saved to graphs/acc_mean_std.jpg")

print("\n" + "=" * 70)
print("All experiments completed!")
print("=" * 70)