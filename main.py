import torch
import torchvision
from torchvision import transforms

import model as md
import dataset as ds
import training_algorithm as ta
import experiment as ex

transform_cifar10 = transforms.Compose([transforms.ToTensor()])

train_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
test_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)

test_set_loader = torch.utils.data.DataLoader(test_set_cifar10, batch_size=8, shuffle=False, num_workers=4)

def evaluate(model, data, device):
    # save the train/eval mode of the network and change it to training mode
    training = model.training
    model.eval()

    acc = 0
    for img_mini_batch, label_mini_batch in data:
         # send the mini-batch to the device memory
        img_mini_batch = img_mini_batch.to(device)
        label_mini_batch = label_mini_batch.to(device)
        
        logits = model(img_mini_batch)
        predicted_labels = torch.argmax(logits, dim=1)
        acc += torch.sum(torch.eq(predicted_labels, label_mini_batch))
    
    print(acc)
    print(len(data))
    acc /= len(data)

    # recover the initial train/eval mode
    if training: model.train()

    return acc

my_model = md.Cifar10_CNN()

my_training_set = ds.ETDataset(train_set_cifar10)

my_test_set = ds.ETDataset(test_set_cifar10)

training_function_args = dict(
    epoch_number=10,
    optimizer="adam",
    lr=0.001,
    weight_decay=0,
    batch_size=64,
    shuffle=True,
    device="cuda:0",
    dataloader_args={"num_workers": 4}
)

my_training_algorithm = ta.ETTraining_algorithm(ta.train, training_function_args)

experiment = ex.Experiment("prova", my_model, my_training_set, my_test_set, my_training_algorithm)
experiment.run()

device = torch.device("cuda:0")


# cifar10_cnn = cifar10_cnn.to(device)

# train(cifar10_cnn, train_set_loader, 100, device, optimizer="adam", lr=0.01, weight_decay=0)

# acc = evaluate(cifar10_cnn, test_set_loader, device)
