import torch
import pandas as pd
from tqdm import tqdm

path = f"experiments/online_mlp_headcmm_repeated_split_mnist"

results = pd.read_csv(f"{path}/results_file.txt")

results["average_accuracy"] = ""
results["average_accuracy_final"] = 0.0
results["average_forgetting"] = ""
results["average_forgetting_final"] = 1.0
results["bwt"] = 0.0
for index, row in tqdm(results.iterrows()):
    experiment_name = row["experiment_name"]
    experiment_path = f"{path}/{experiment_name}"
    results_path = f"{experiment_path}/results/holdout"

    training_task_history = torch.load(f"{results_path}/training_task_history_tensor.pth")
    n_tasks = training_task_history.shape[0]
    average_accuracy = torch.sum(torch.tril(training_task_history), dim=1) / torch.arange(1, n_tasks + 1)
    f = torch.zeros((n_tasks, n_tasks))
    for k in range(n_tasks):
        for j in range(k):
            f[k, j] = torch.max(training_task_history[:k, j]) - training_task_history[k, j]
    average_forgetting = torch.sum(f[1:, :], dim=1)
    average_forgetting = average_forgetting / torch.arange(1, n_tasks)

    bwt = max(torch.sum(torch.tril(training_task_history - torch.diag(training_task_history), diagonal=-1)[1:, :]).item() * 2 / (n_tasks * (n_tasks - 1)), 0)

    results["average_accuracy"][index] = "'" + str(average_accuracy.tolist()) + "'"
    results["average_forgetting"][index] = "'" + str(average_forgetting.tolist()) + "'"
    results["average_accuracy_final"][index] = average_accuracy[-1].item()
    results["average_forgetting_final"][index] = average_forgetting[-1].item()
    results["bwt"][index] = bwt

results.to_csv(f"{path}/results_file_extended.csv")