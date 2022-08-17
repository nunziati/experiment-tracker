import torch

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