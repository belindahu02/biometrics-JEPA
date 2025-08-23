import os
import numpy as np
import matplotlib.pyplot as plt

#
# GRAPH_PATH = "/Users/belindahu/Desktop/thesis/biometrics-JEPA/main/audio-representations/data/combined_graphs"
GRAPH_PATH = "/app/data/combined_graphs"

def plotter(paths, names, variables, variable_name, graph_name):
    kappa = []
    for path in paths:
        data = np.load(path)
        score = data["kappa_score"]
        cal_score = np.mean(score, axis=1)
        kappa.append(cal_score)

    plt.figure(figsize=(12, 8))
    for i in range(len(kappa)):
        plt.plot(variables, kappa[i], label=names[i])
    plt.title(f"kappa score vs {variable_name}")
    plt.xlabel(variable_name)
    plt.ylabel("kappa score")
    plt.legend()

    # Create graphs folder if it doesn't exist
    os.makedirs(GRAPH_PATH, exist_ok=True)

    output = os.path.join(GRAPH_PATH, f"{graph_name}.npz")
    plt.savefig(output)
    plt.show()
    plt.close()
    return True

paths = [
    "/app/data/baselines/musicid_scen1_supervised.npz",
    "/app/data/baselines/musicid_scen1_DA.npz",
    "/app/data/baselines/multi_task/musicid_scen1_multi task.npz",
    "/app/data/baselines/musicid_scen1_simsiam.npz",
    "/app/data/graph_data/jepa.npz"  # <-- new file
]

names = [
    "supervised",
    "data augmentations",
    "multi task learning",
    "simsiam",
    "JEPA"
]

variable = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 45, 50, 55, 60]
variable_name = "samples per user"
graph_name = "Kappa MMI"

plotter(paths, names, variable, variable_name, graph_name)
