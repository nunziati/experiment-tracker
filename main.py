import torch
import torchvision
from torchvision import transforms
from models import cifar10_cnn
from training_algorithm import train

transform_cifar10 = transforms.Compose([transforms.ToTensor()])

train_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
test_set_cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)

train_set_loader = torch.utils.data.DataLoader(train_set_cifar10, batch_size=64, shuffle=True, num_workers=4)
test_set_loader = torch.utils.data.DataLoader(test_set_cifar10, batch_size=8, shuffle=False, num_workers=4)

def initialize(model):
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)

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

device = torch.device("cuda:0")

cifar10_cnn = cifar10_cnn.to(device)

initialize(cifar10_cnn)

train(cifar10_cnn, train_set_loader, 100, device, optimizer="adam", lr=0.01, weight_decay=0)

acc = evaluate(cifar10_cnn, test_set_loader, device)
