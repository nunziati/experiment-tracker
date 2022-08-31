import model as md
from dataset import get_cifar10_dataset
import algorithm as ta
import experiment as ex


my_model = md.Simple_Cifar10_CNN()

my_training_set, my_test_set = get_cifar10_dataset()

training_function_args = dict(
    loss_function_name="cross_entropy_loss",
    optimizer_name="sgd",
    optimizer_args = dict(
        lr=0.001,
        weight_decay=0.0001
    ),
    batch_size=50,
    device="cuda:0",
    dataloader_args=dict(
        num_workers=4
    ),
    evaluation_step=10
)

my_training_algorithm = ta.ETAlgorithm(ta.single_pass_online_train, training_function_args)

test_function_args = dict(
    metrics="accuracy",
    batch_size=None,
    shuffle=True,
    device="cuda:0",
    dataloader_args={"num_workers": 4}
)

# my_test_algorithm = ta.ETAlgorithm(ta.evaluate, test_function_args)

# my_pipeline = ta.ETPipeline("holdout", [my_training_algorithm, my_test_algorithm])

experiment = ex.ETExperiment("prova_gpu_online_task", my_model, my_training_set, my_test_set, my_training_algorithm)

experiment.run()


