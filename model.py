from re import S
import torch
from linear_cmm import LinearCMM


class MyConv2d(torch.nn.Conv2d):
    """Custom version of torch.nn.Conv2d, built because paddint="same" is not a valid option in some versions of pytorch.
        There is no need of a forward method, because torch.nn.Conv2d already defines it."""

    def __init__(self, in_maps, out_maps, *args, padding="same", **kwargs):
        """Constructor of the class is the same as the constructor of the parent class.
        
        Args:
            the same as for torch.nn.Conv2d
            padding: if "same", not sure if it works for even values of kernel_size
        """

        if isinstance(padding, str):
            if padding == "same":
                # computing the padding (or padding tuple) in a way that the input and output diensions are the same
                if isinstance(kwargs["kernel_size"], int):
                    padding = (kwargs["kernel_size"] - 1) // 2
                elif isinstance(kwargs["kernel_size"], tuple):
                    padding = ((x - 1) // 2 for x in kwargs["kernel_size"])
                else:
                    raise TypeError("kernel_size must be int or tuple of int")
            else:
                raise ValueError("padding str must be 'same'")

        # use the computed padding and the other parameters to initialize the instance of torch.nn.Conv2d
        super(MyConv2d, self).__init__(in_maps, out_maps, *args, padding=padding, **kwargs)   


class Simple_Cifar10_MLP(torch.nn.Module):
    def __init__(self):
        super(Simple_Cifar10_MLP, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32*32*3, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.model(x)

    def description(self):
        return """torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(32*32*3, 32*32*6),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(32*32*6, 32*32*6),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(32*32*6, 32*32*6)
)"""

class Simple_Cifar10_MLP_CMM(torch.nn.Module):
    def __init__(self, base_m, delta, hidden_units):
        super(Simple_Cifar10_MLP_CMM, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            LinearCMM(32*32*3, hidden_units, base_m=base_m, delta=delta),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            LinearCMM(hidden_units, hidden_units, base_m=base_m, delta=delta),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            LinearCMM(hidden_units, 10, base_m=base_m, delta=delta)
        )

    def forward(self, x):
        return self.model(x)

    def description(self):
        return """torch.nn.Sequential(
    torch.nn.Flatten(),
    LinearCMM(32*32*3, 1000),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    LinearCMM(1000, 1000),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    LinearCMM(1000, 10)
)"""


class Cifar10_CNN(torch.nn.Module):
    def __init__(self):
        super(Cifar10_CNN, self).__init__() 
        
        self.model = torch.nn.Sequential(
            MyConv2d(3, 32, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            MyConv2d(32, 32, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.3),

            MyConv2d(32, 64, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            MyConv2d(64, 64, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.5),

            MyConv2d(64, 128, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            MyConv2d(128, 128, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.5),

            torch.nn.Flatten(),
            torch.nn.Linear(4*4*128, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

    def description(self):
        return """torch.nn.Sequential(
    MyConv2d(3, 32, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(32),
    MyConv2d(32, 32, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(32),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(0.3),

    MyConv2d(32, 64, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(64),
    MyConv2d(64, 64, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(64),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(0.5),

    MyConv2d(64, 128, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(128),
    MyConv2d(128, 128, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(128),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(0.5),

    torch.nn.Flatten(),
    torch.nn.Linear(4*4*128, 128),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(128),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(128, 10)
)"""


class Cifar10_CNN_CMM(torch.nn.Module):
    def __init__(self):
        super(Cifar10_CNN_CMM, self).__init__() 
        
        self.model = torch.nn.Sequential(
            MyConv2d(3, 32, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            MyConv2d(32, 32, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.3),

            MyConv2d(32, 64, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            MyConv2d(64, 64, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.5),

            MyConv2d(64, 128, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            MyConv2d(128, 128, padding="same", kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(0.5),

            torch.nn.Flatten(),
            LinearCMM(4*4*128, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.5),
            LinearCMM(128, 10)
        )

    def forward(self, x):
        return self.model(x)

    def description(self):
        return """torch.nn.Sequential(
    MyConv2d(3, 32, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(32),
    MyConv2d(32, 32, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(32),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(0.3),

    MyConv2d(32, 64, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(64),
    MyConv2d(64, 64, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(64),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(0.5),

    MyConv2d(64, 128, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(128),
    MyConv2d(128, 128, padding="same", kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.BatchNorm2d(128),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Dropout(0.5),

    torch.nn.Flatten(),
    torch.nn.Linear(4*4*128, 128),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(128),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(128, 10)
)"""