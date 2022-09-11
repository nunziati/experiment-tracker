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


class Simple_MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes, dropout=0.0, flatten=True, cmm=False, cmm_args={}):
        super(Simple_MLP, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.dropout = dropout
        self.flatten = flatten
        self.cmm = cmm
        layer_type = LinearCMM if cmm else torch.nn.Linear
        cmm_args = cmm_args if cmm else {}

        flatten = (torch.nn.Flatten(),) if flatten else ()

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            layer_type(input_size, hidden_units, **cmm_args),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            layer_type(hidden_units, num_classes, **cmm_args)
        )

    def forward(self, x):
        return self.model(x)

    def description(self):
        layer_type = "LinearCMM" if self.cmm else "torch.nn.Linear"

        repr = "torch.nn.Flatten()\n" if self.flatten else ""
        repr += f"{layer_type}({self.input_size}, {self.hidden_units}, {str(self.cmm_args)}),\n"
        repr += "torch.nn.ReLU(),\n"
        repr += f"torch.nn.Dropout(p={self.dropout}),\n" if self.dropout != 1 else ""
        repr += f"{layer_type}({self.hidden_units}, {self.num_classes}, {str(self.cmm_args)})"

        return f"torch.nn.Sequential(\n{repr}\n)"
        

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