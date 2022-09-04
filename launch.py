from model import Simple_Cifar10_MLP, Simple_Cifar10_MLP_CMM
from experiment import build_experiment
from itertools import product
from multiprocessing.dummy import Pool as ThreadPool

def build_and_run_experiment(arg):
    experiment_id, params = arg
    batch_size = params[0]
    hidden_units = params[1]
    base_m = params[2]
    delta_proportion = params[3]
    learning_rate = params[4]
    delta = max(int(delta_proportion * base_m), 1)

    model = Simple_Cifar10_MLP_CMM(base_m, delta, hidden_units)

    config = dict(
        name = "simple_mlp_cmm_gridsearch_" + str(experiment_id).rjust(5, "0"),
        model = model,
        type = "online",
        dataset = "cifar10",
        sorted = True,
        epoch_number=3,
        loss_function_name="cross_entropy_loss",
        optimizer_name="sgd",
        optimizer_args = dict(
            lr=learning_rate,
            weight_decay=0.0001
        ),
        batch_size=batch_size,
        metrics="accuracy",
        device="cuda:1",
        dataloader_args=dict(
            num_workers=4
        ),
        validation=0.2,
        evaluation_batch_size=500,
        evaluation_step=500
    )

    experiment = build_experiment(config)
    experiment.run()

batch_size_values = [1, 5, 20]
hidden_units_values = [30, 100, 300, 1000]
base_m_values = [10, 20, 50, 100, 200, 500, 1000]
delta_proportion_values = [0.01, 0.02, 0.05, 0.1, 0.2]
learning_rate_values = [0.1, 0.01, 0.001, 0.0001]

to_do = list(product(
    batch_size_values,
    hidden_units_values,
    base_m_values,
    delta_proportion_values,
    learning_rate_values
))

with open("experiments.txt", "a+") as f:
    for idx, params in enumerate(to_do):
        batch_size = params[0]
        hidden_units = params[1]
        base_m = params[2]
        delta_proportion = params[3]
        learning_rate = params[4]
        delta = max(int(delta_proportion * base_m), 1)
        f.write(f"Experiment id = {str(idx).rjust(5, '0')}\t\tBatch size = {str(batch_size).ljust(4)}\t\tHidden units = {str(hidden_units).ljust(4)}\t\tbase_m = {str(base_m).ljust(5)}\t\tdelta = {str(delta).ljust(4)}\t\tdelta_proportion = {str(delta_proportion).ljust(8)}\t\tBatch size = {str(learning_rate).ljust(8)}")


pool = ThreadPool(8)
results = pool.map(build_and_run_experiment, enumerate(to_do))