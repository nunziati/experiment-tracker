from model import Simple_MLP
from experiment import build_experiment
from itertools import product


"""datasets = ["cifar10", "cifar100", "mnist"]"""
datasets = ["mnist"]
path = "experiments/joint_mlp_NOcmm_mnist/"

hidden_unit_values = [30, 100, 300, 1000, 3000]
learning_rate_values = [0.1, 0.01, 0.001, 0.0001]
optimizer_values = ["adam", "sgd"]
batch_size_values = [100, 1000]

to_do = list(product(
    hidden_unit_values,
    learning_rate_values,
    optimizer_values,
    batch_size_values
))

results_file = [
    "experiment_name,dataset_name,hidden_units,learning_rate,optimizer,batch_size,accuracy,best_epoch\n"
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

        model = Simple_MLP(input_dim, hidden_units, output_dim, dropout=0.0, output="logits")

        config = dict(
            path = path,
            name = f"obj1_{dataset_name}_exp{str(exp_id).rjust(3, '0')}", # da codificare
            model = model,
            type = "joint",
            dataset = dataset_name,
            sorted = False,
            epoch_number = 100,
            loss_function_name = "cross_entropy_loss",
            optimizer_name = optimizer,
            optimizer_args = dict(
                lr = learning_rate,
                weight_decay = 0.0001
            ),
            batch_size = batch_size,
            metrics = "accuracy",
            device = "cuda:0",
            dataloader_args = dict(
                num_workers = 4
            ),
            validation = 0.2,
            evaluation_batch_size = 500,
            evaluation_step = 500
        )

        experiment = build_experiment(config)
        experiment.run()

        accuracy = experiment.results["holdout"]["accuracy"]
        best_epoch = experiment.results["holdout"]["best_epoch"]

        results_file.append(f"obj1_{dataset_name}_exp{str(exp_id).rjust(3, '0')},{dataset_name},{hidden_units},{learning_rate},{optimizer},{batch_size},{accuracy},{best_epoch}\n")

with open(f"{path}results_file.txt", "a+") as f:
    f.writelines(results_file)
