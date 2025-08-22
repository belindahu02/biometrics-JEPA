# =============================================
# COMPARISON SCRIPT (NEW FILE: compare_methods.py)
# =============================================

"""
Script to compare different conversion methods
Run this to find the best conversion method for your data
"""

from trainers import spectrogram_trainer
import numpy as np
import matplotlib.pyplot as plt


def compare_conversion_methods(data_path, user_ids, samples_per_user=10, n_runs=5):
    """Compare different spectrogram-to-1D conversion methods"""

    methods = ['pca', 'downsample', 'average_bands', 'mel_bands']
    results = {}

    print(f"Comparing conversion methods with {samples_per_user} samples per user...")

    for method in methods:
        print(f"\n--- Testing method: {method} ---")

        acc_list = []
        kappa_list = []

        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...")
            try:
                test_acc, kappa_score = spectrogram_trainer(
                    samples_per_user=samples_per_user,
                    data_path=data_path,
                    user_ids=user_ids,
                    conversion_method=method
                )
                acc_list.append(test_acc)
                kappa_list.append(kappa_score)
                print(f"    Acc: {test_acc:.4f}, Kappa: {kappa_score:.4f}")
            except Exception as e:
                print(f"    Error: {e}")
                continue

        if acc_list:
            results[method] = {
                'accuracy_mean': np.mean(acc_list),
                'accuracy_std': np.std(acc_list),
                'kappa_mean': np.mean(kappa_list),
                'kappa_std': np.std(kappa_list),
                'accuracy_raw': acc_list,
                'kappa_raw': kappa_list
            }
            print(f"  {method} - Avg Acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f}")

    return results


if __name__ == "__main__":
    # UPDATE THESE PATHS
    DATA_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/classification_input"
    USER_IDS = list(range(1, 2))  # Users 1 to 109

    # Compare methods
    comparison_results = compare_conversion_methods(DATA_PATH, USER_IDS)

    # Print summary
    print("\n" + "=" * 50)
    print("CONVERSION METHOD COMPARISON RESULTS")
    print("=" * 50)

    for method, results in comparison_results.items():
        print(f"{method.upper()}:")
        print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        print(f"  Kappa:    {results['kappa_mean']:.4f} ± {results['kappa_std']:.4f}")
        print()

    # Find best method
    if comparison_results:
        best_method = max(comparison_results.keys(),
                          key=lambda x: comparison_results[x]['accuracy_mean'])
        print(f"BEST METHOD: {best_method.upper()}")
        print(f"Best Accuracy: {comparison_results[best_method]['accuracy_mean']:.4f}")