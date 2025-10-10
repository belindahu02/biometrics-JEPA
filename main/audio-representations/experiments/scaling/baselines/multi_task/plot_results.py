from trainers import *
from pre_trainers import *
import numpy as np
import matplotlib.pyplot as plt

# Base output directory on host
# base_dir = "test/"
base_dir = "/app/data/experiments/scaling/baselines"

# Make sure these exist
graph_data_dir = os.path.join(base_dir, "multi_task/graph_data")
graphs_dir = os.path.join(base_dir, "multi_task/graphs")

os.makedirs(graph_data_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

def plotspu(ft):
    """
    Plot results varying the number of users in classification task.

    Args:
        ft: Fine-tuning configuration (0-5)
    """
    scen = 1
    fet_extrct = pre_trainer(scen=scen)

    if ft == 0:
        model_name = "multi_task"
    else:
        model_name = "ft"+str(ft)+"_multi_task"

    variable_name = "number of users"
    # Number of users to test: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109
    variable = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 109]

    acc = []
    kappa = []

    for num_users in variable:
        acc_temp = []
        kappa_temp = []
        # Run 10 iterations for each configuration
        for itr in range(10):
            print(f"\nIteration {itr + 1}/10 for {num_users} users")
            test_acc, kappa_score = trainer(num_users, fet_extrct, scen, ft=ft)
            acc_temp.append(test_acc)
            kappa_temp.append(kappa_score)
        acc.append(acc_temp)
        kappa.append(kappa_temp)

    acc = np.array(acc)
    kappa = np.array(kappa)

    np.savez(os.path.join(graph_data_dir, model_name + ".npz"),
             test_acc=acc, kappa_score=kappa)
    print(f"\nFinal results shape:")
    print(acc.shape)
    print(kappa.shape)

    # Plot kappa score
    kappa_mean = np.mean(kappa, axis=1)
    kappa_std = np.std(kappa, axis=1)
    kappa_max = np.max(kappa, axis=1)

    plt.figure(figsize=(12, 8))
    plt.plot(variable, kappa_mean, 'm-', linewidth=2, label=model_name + ' (mean)')
    plt.fill_between(variable, kappa_mean - kappa_std, kappa_mean + kappa_std,
                     alpha=0.3, color='m')
    plt.plot(variable, kappa_max, 'm--', linewidth=1, label=model_name + ' (max)')
    plt.title("Kappa Score vs " + variable_name)
    plt.xlabel(variable_name)
    plt.ylabel("Kappa Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if ft == 0:
        plt.savefig(os.path.join(graphs_dir, 'kappa.jpg'))
    else:
        plt.savefig(os.path.join(graphs_dir, 'kappa_ft' + str(ft)+'.jpg'))
    plt.close()

    # Plot accuracy
    acc_mean = np.mean(acc, axis=1)
    acc_std = np.std(acc, axis=1)
    acc_max = np.max(acc, axis=1)

    plt.figure(figsize=(12, 8))
    plt.plot(variable, acc_mean, 'm-', linewidth=2, label=model_name + ' (mean)')
    plt.fill_between(variable, acc_mean - acc_std, acc_mean + acc_std,
                     alpha=0.3, color='m')
    plt.plot(variable, acc_max, 'm--', linewidth=1, label=model_name + ' (max)')
    plt.title("Test Accuracy vs " + variable_name)
    plt.xlabel(variable_name)
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if ft == 0:
        plt.savefig(os.path.join(graphs_dir, 'acc.jpg'))
    else:
        plt.savefig(os.path.join(graphs_dir, 'acc_ft' + str(ft)+'.jpg'))
    plt.close()

    return True