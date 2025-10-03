from trainers import *
import numpy as np
import matplotlib.pyplot as plt

# Base output directory on host
base_dir = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data"

# Make sure these exist
graph_data_dir = os.path.join(base_dir, "supervised/graph_data")
graphs_dir = os.path.join(base_dir, "supervised/graphs")

os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

variable_name = "number of users"
model_name = "eeg_mmi_user_scaling"

# Test with increasing numbers of users: 10, 20, 30, ..., 100, 109
variable = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109]

acc = []
kappa = []

for num_users in variable:
    acc_temp = []
    kappa_temp = []

    print(f"\n{'=' * 60}")
    print(f"Running experiments with {num_users} users")
    print(f"{'=' * 60}\n")

    # Run 10 iterations for each user count
    for itr in range(10):
        print(f"\nIteration {itr + 1}/10 for {num_users} users")
        test_acc, kappa_score = trainer(num_users)
        acc_temp.append(test_acc)
        kappa_temp.append(kappa_score)

    acc.append(acc_temp)
    kappa.append(kappa_temp)

acc = np.array(acc)
kappa = np.array(kappa)

# Save results
np.savez(os.path.join(graph_data_dir, model_name+".npz"),
         test_acc=acc, kappa_score=kappa)
print(f"\nFinal results shape:")
print(f"Accuracy: {acc.shape}")
print(f"Kappa: {kappa.shape}")

# Plot Kappa score
kappa_mean = np.mean(kappa, axis=1)
kappa_std = np.std(kappa, axis=1)
kappa_max = np.max(kappa, axis=1)

plt.figure(figsize=(12, 8))
plt.plot(variable, kappa_mean, 'm-o', label=model_name + ' (mean)', linewidth=2)
plt.fill_between(variable, kappa_mean - kappa_std, kappa_mean + kappa_std,
                 alpha=0.2, color='m')
plt.plot(variable, kappa_max, 'm--', label=model_name + ' (max)', alpha=0.6)
plt.title("Kappa Score vs " + variable_name, fontsize=14)
plt.xlabel(variable_name, fontsize=12)
plt.ylabel("Kappa Score", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'kappa.jpg'))
plt.show()
plt.close()

# Plot Test Accuracy
acc_mean = np.mean(acc, axis=1)
acc_std = np.std(acc, axis=1)
acc_max = np.max(acc, axis=1)

plt.figure(figsize=(12, 8))
plt.plot(variable, acc_mean, 'm-o', label=model_name + ' (mean)', linewidth=2)
plt.fill_between(variable, acc_mean - acc_std, acc_mean + acc_std,
                 alpha=0.2, color='m')
plt.plot(variable, acc_max, 'm--', label=model_name + ' (max)', alpha=0.6)
plt.title("Test Accuracy vs " + variable_name, fontsize=14)
plt.xlabel(variable_name, fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, 'acc.jpg'))
plt.show()
plt.close()

# Print summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
for i, num_users in enumerate(variable):
    print(f"\n{num_users} users:")
    print(f"  Accuracy - Mean: {acc_mean[i]:.4f}, Std: {acc_std[i]:.4f}, Max: {acc_max[i]:.4f}")
    print(f"  Kappa    - Mean: {kappa_mean[i]:.4f}, Std: {kappa_std[i]:.4f}, Max: {kappa_max[i]:.4f}")