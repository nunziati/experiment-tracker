import torch
import torchvision
from torchvision import transforms

import model as md
import dataset as ds
import algorithm as ta
import experiment as ex

transform_cifar10 = transforms.Compose([transforms.ToTensor()])

train_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
test_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)

test_set_loader = torch.utils.data.DataLoader(test_set_cifar10, batch_size=8, shuffle=False, num_workers=4)

my_model = md.Cifar10_CNN()

my_training_set = ds.ETDataset(train_set_cifar10)

my_test_set = ds.ETDataset(test_set_cifar10)

training_function_args = dict(
    epoch_number=2,
    loss_function_name="cross_entropy_loss",
    optimizer_name="adam",
    lr=0.001,
    weight_decay=0,
    batch_size=64,
    shuffle=True,
    device="cuda:0",
    dataloader_args={"num_workers": 4}
)

my_training_algorithm = ta.ETAlgorithm(ta.train, training_function_args)

test_function_args = dict(
    metrics="accuracy",
    batch_size=None,
    shuffle=True,
    device="cuda:0",
    dataloader_args={"num_workers": 4}
)

my_test_algorithm = ta.ETAlgorithm(ta.evaluate, test_function_args)

my_pipeline = ta.ETPipeline("holdout", [my_training_algorithm, my_test_algorithm])

experiment = ex.ETExperiment("prova", my_model, my_training_set, my_test_set, my_pipeline)

experiment.run()

"""
Algoritmi:
    - training mini-batch (che può diventare benissimo batch, mini-batch, online)
    - test accuratezza
    - sort dataset
    - 
"""


# cifar10_cnn = cifar10_cnn.to(device)

# train(cifar10_cnn, train_set_loader, 100, device, optimizer="adam", lr=0.01, weight_decay=0)

# acc = evaluate(cifar10_cnn, test_set_loader, device)
