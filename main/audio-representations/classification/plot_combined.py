import os
import numpy as np
import matplotlib.pyplot as plt

#
GRAPH_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/combined_graphs"
# GRAPH_PATH = "/app/data/combined_graphs"

# def plotter(paths, names, variables, variable_name, graph_name):
#     kappa = []
#     for path in paths:
#         data = np.load(path)
#         score = data["kappa_score"]
#         cal_score = np.mean(score, axis=1)
#         kappa.append(cal_score)
#
#     plt.figure(figsize=(12, 8))
#     for i in range(len(kappa)):
#         plt.plot(variables, kappa[i], label=names[i])
#     plt.title(f"kappa score vs {variable_name}")
#     plt.xlabel(variable_name)
#     plt.ylabel("kappa score")
#     plt.legend()
#
#     # Create graphs folder if it doesn't exist
#     os.makedirs(GRAPH_PATH, exist_ok=True)
#
#     output = os.path.join(GRAPH_PATH, f"{graph_name}.jpg")
#     plt.savefig(output)
#     plt.show()
#     plt.close()
#     return True

def plotter(paths, names, variable_name, graph_name):
    plt.figure(figsize=(12, 8))

    for i, path in enumerate(paths):
        data = np.load(path)
        score = data["kappa_score"]
        cal_score = np.mean(score, axis=1)

        # Use variables if available, otherwise default to 1..N
        if "variables" in data:
            variables = data["variables"]
        else:
            variables = np.arange(1, len(cal_score) + 1)

        plt.plot(variables, cal_score, label=names[i], marker='o')

    plt.title(f"kappa score vs {variable_name}")
    plt.xlabel(variable_name)
    plt.ylabel("kappa score")
    plt.legend()

    # Create graphs folder if it doesn't exist
    os.makedirs(GRAPH_PATH, exist_ok=True)

    output = os.path.join(GRAPH_PATH, f"{graph_name}.jpg")
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return True


paths = [
    "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/baselines/musicid_scen1_supervised.npz",
    "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/baselines/musicid_scen1_DA.npz",
    "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/baselines/musicid_scen1_multi task.npz",
    "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/baselines/musicid_scen1_simsiam.npz",
    "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/graph_data/spectrogram_1d_converted_pca.npz"  # <-- new file
]

# paths = [
#     "/app/data/baselines/musicid_scen1_supervised.npz",
#     "/app/data/baselines/musicid_scen1_DA.npz",
#     "/app/data/baselines/musicid_scen1_multi task.npz",
#     "/app/data/baselines/musicid_scen1_simsiam.npz",
#     "/app/data/graph_data/spectrogram_1d_converted_pca.npz"  # <-- new file
# ]

names = [
    "supervised",
    "data augmentations",
    "multi task learning",
    "simsiam",
    "JEPA"
]

variable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50, 55, 60]
variable_name = "samples per user"
# graph_name = "Kappa MMI"

graph_name = "Kappa MMI v2"


plotter(paths, names, variable_name, graph_name)
