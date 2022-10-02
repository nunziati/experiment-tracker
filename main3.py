from model import Simple_MLP
from experiment import build_experiment

path = "experiments/"
dataset_name = "split_mnist"

name = "prova_adam"

hidden_units = 30
cmm_args = dict(
    base_m = 10,
    delta = 3
)
optimizer = "adam"
learning_rate = 0.00001
batch_size = 10

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

model = Simple_MLP(input_dim, hidden_units, output_dim, dropout=0.0, output="logits", cmm=True, cmm_args=cmm_args)

config = dict(
    path = path,
    name = name,
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