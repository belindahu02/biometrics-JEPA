# =============================================
# MODIFIED plot_results.py
# =============================================

from trainers import spectrogram_trainer  # Use new trainer
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
# DATA_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/classification_input"  # UPDATE THIS PATH
# OUTPUT_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/output_path"
# GRAPH_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graphs"
DATA_PATH = "/app/data/grouped_embeddings"
OUTPUT_PATH = "/app/data/graph_data"
GRAPH_PATH = "/app/data/graphs"
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(GRAPH_PATH, exist_ok=True)
USER_IDS = list(range(1, 2))  # Users 1 to 109
CONVERSION_METHOD = 'pca'  # Options: 'pca', 'downsample', 'average_bands', 'mel_bands'

variable_name = "samples per user"
model_name = f"spectrogram_1d_converzted_{CONVERSION_METHOD}"
variable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50, 55, 60]

acc = []
kappa = []

print(f"Running experiments with conversion method: {CONVERSION_METHOD}")
print(f"Testing {len(variable)} different sample sizes...")

for i, samples_per_user in enumerate(variable):
    print(f"\n--- Testing {samples_per_user} samples per user ({i + 1}/{len(variable)}) ---")

    acc_temp = []
    kappa_temp = []

    for itr in range(10):  # 10 runs per configuration
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
        except Exception as e:
            print(f"    Error in run {itr + 1}: {e}")
            continue

    if acc_temp:  # Only append if we have results
        acc.append(acc_temp)
        kappa.append(kappa_temp)
        print(f"  Completed {samples_per_user} samples/user - Avg Acc: {np.mean(acc_temp):.4f}")
    else:
        print(f"  No successful runs for {samples_per_user} samples/user")
acc = np.array(acc)
kappa = np.array(kappa)

# Save results
output_file = os.path.join(OUTPUT_PATH, f"{model_name}.npz")
np.savez(output_file, test_acc=acc, kappa_score=kappa)
print(f"\nResults saved to graph_data/{model_name}.npz")
print(f"Accuracy shape: {acc.shape}")
print(f"Kappa shape: {kappa.shape}")

# Plot results
# Convert to numpy arrays

if len(acc) > 0:
    # Kappa plot
    kappa_max = np.max(kappa, axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(variable[:len(kappa_max)], kappa_max, 'm', label=model_name, linewidth=2, marker='o')
    plt.title(f"Kappa Score vs {variable_name} ({CONVERSION_METHOD} conversion)")
    plt.xlabel(variable_name)
    plt.ylabel("Kappa Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_graph = os.path.join(GRAPH_PATH, f"kappa_{CONVERSION_METHOD}.jpg")
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Accuracy plot
    acc_max = np.max(acc, axis=1)
    plt.figure(figsize=(12, 8))
    plt.plot(variable[:len(acc_max)], acc_max, 'm', label=model_name, linewidth=2, marker='o')
    plt.title(f"Test Accuracy vs {variable_name} ({CONVERSION_METHOD} conversion)")
    plt.xlabel(variable_name)
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_graph = os.path.join(GRAPH_PATH, f"acc_{CONVERSION_METHOD}.jpg")
    plt.savefig(output_graph, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Plots saved as graphs/kappa_{CONVERSION_METHOD}.jpg and graphs/acc_{CONVERSION_METHOD}.jpg")
else:
    print("No results to plot")

