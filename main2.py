from model import Simple_MLP
from experiment import build_experiment
from itertools import product
import matplotlib.pyplot as plt
import os


"""
datasets = ["cifar10", "split_cifar100", "split_mnist"]
optimizer_values = ["adam", "sgd"]
"""

"""
split_mnist
sgd

10:     12411.pts-51.maxi  finito
30:     12119.pts-51.maxi  finito
100:    29008.pts-9.kronos finito
300:    29377.pts-9.kronos finito


cifar10
sgd

10:     12411.pts-51.maxi  
30:     12119.pts-51.maxi  
100:    29008.pts-9.kronos 
300:    29377.pts-9.kronos 
"""

"""
Start: 14:40
Target: 240
"""

datasets = ["split_mnist"]
optimizer_values = ["sgd"]

path = f"experiments/online_mlp_cmm_{datasets[0]}_{optimizer_values[0]}/"

hidden_unit_values = [10, 30, 100, 300, 1000]
learning_rate_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]
batch_size_values = [1, 10, 100]
memory_units_values = [3, 10, 30, 100]
delta_values = [0.01, 0.03, 0.1, 0.3]

hidden_unit_values = [1000]

path = f"experiments/online_mlp_cmm_{datasets[0]}_{optimizer_values[0]}_hu{hidden_unit_values[0]}/"

to_do = list(product(
    hidden_unit_values,
    learning_rate_values,
    optimizer_values,
    batch_size_values,
    memory_units_values,
    delta_values
))

results_file = [
    "experiment_name,dataset_name,hidden_units,learning_rate,optimizer,batch_size,memory_units,delta,accuracy\n"
]

for dataset_name in datasets:
    if dataset_name == "cifar10":
        input_dim = 32*32*3
        output_dim = 10
    elif dataset_name == "cifar100":
        input_dim = 32*32*3
        output_dim = 100
    elif dataset_name == "mnist":
        input_dim = 28*28
        output_dim = 10
    elif dataset_name == "split_cifar100":
        input_dim = 32*32*3
        output_dim = 100
    elif dataset_name == "split_mnist":
        input_dim = 28*28
        output_dim = 10

    for exp_id, exp_param in enumerate(to_do):
        hidden_units = exp_param[0]
        learning_rate = exp_param[1]
        optimizer = exp_param[2]
        batch_size = exp_param[3]
        memory_units = exp_param[4]
        delta = int(exp_param[5] * (memory_units - 1)) + 1

        cmm_args={'base_m': memory_units, 'delta': delta}

        if os.path.isdir(f"{path}exp{str(exp_id).rjust(4, '0')}") and os.path.isfile(f"{path}exp{str(exp_id).rjust(4, '0')}/results/holdout/accuracy.txt"):
            with open(f"{path}exp{str(exp_id).rjust(4, '0')}/results/holdout/accuracy.txt", "r") as f:
                accuracy = f.read()
        else:
            model = Simple_MLP(input_dim, hidden_units, output_dim, dropout=0.0, output="logits", cmm=True, cmm_args=cmm_args)

            config = dict(
                path = path,
                name = f"exp{str(exp_id).rjust(4, '0')}",
                model = model,
                type = "online",
                dataset = dataset_name,
                sorted = True,
                epoch_number = 1,
                loss_function_name = "cross_entropy_loss",
                optimizer_name = optimizer,
                optimizer_args = dict(
                    lr = learning_rate,
                    weight_decay = 0.0001
                ),
                batch_size = batch_size,
                metrics = "accuracy",
                device = "cuda:1",
                dataloader_args = dict(
                    num_workers = 4
                ),
                validation = 0.2,
                evaluation_batch_size = 100,
                evaluation_step = "task"
            )

            experiment = build_experiment(config)
            experiment.run()

            accuracy = experiment.results["holdout"]["accuracy"]

        results_file.append(f"obj1_{dataset_name}_exp{str(exp_id).rjust(3, '0')},{dataset_name},{hidden_units},{learning_rate},{optimizer},{batch_size},{memory_units},{delta},{accuracy}\n")

        plt.close("all")

with open(f"{path}results_file.txt", "a+") as f:
    f.writelines(results_file)
