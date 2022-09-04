from model import Simple_Cifar10_MLP, Simple_Cifar10_MLP_CMM
from experiment import build_experiment


model = Simple_Cifar10_MLP_CMM(base_m=100, delta=10, hidden_units=100)

config = dict(
    name = "simple_mlp_cmm_1g",
    model = model,
    type = "online",
    dataset = "cifar10",
    sorted = True,
    epoch_number=3,
    loss_function_name="cross_entropy_loss",
    optimizer_name="sgd",
    optimizer_args = dict(
        lr=0.001,
        weight_decay=0.0001
    ),
    batch_size=1,
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