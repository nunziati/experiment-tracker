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
    def __init__(self, input_size, hidden_units, num_classes, dropout=0.0, flatten=True, cmm=False, cmm_args={}, output="logits"):
        super(Simple_MLP, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.dropout = dropout
        self.flatten = flatten
        self.cmm = cmm
        self.output = output
        layer_type = LinearCMM if cmm else torch.nn.Linear
        self.cmm_args = cmm_args if cmm else {}

        flatten = (torch.nn.Flatten(),) if flatten else ()

        self.flatten = torch.nn.Flatten()
        self.linear1 = layer_type(input_size, hidden_units, **cmm_args)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear2 = layer_type(hidden_units, num_classes, **cmm_args)
        if output == "sigmoid":
            self.output_layer = torch.nn.Sigmoid()
        elif output == "softmax":
            self.output_layer = torch.nn.Softmax()

    def get_continual_model_parameters(self):
        return dict(
            cmm_params_layer1 = self.linear1.get_continual_model_parameters(),
            cmm_params_layer2 = self.linear2.get_continual_model_parameters()
        )
    
    def compute_extended_output(self, x):
        O = self.flatten(x)
        
        if self.cmm:
            output1 = self.linear1.compute_extended_output(O)
            O = self.relu1(output1["output"])
        else:
            O = self.linear1(O)
            O = self.relu1(O)
        
        O = self.dropout(O)

        if self.cmm:
            output2 = self.linear2.compute_extended_output(O)
            A = output2["output"]
        else:
            A = self.linear2(O)

        O = A if self.output == "logits" else self.output_layer(A)

        output_dict = dict(
            logits = A,
            output = O,
        )

        if self.cmm:
            output_dict.update(
                dict(
                    memory_model_parameters = dict(
                        cmm_params_layer1 = output1["memory_model_parameters"],
                        cmm_params_layer2 = output2["memory_model_parameters"]
                    )
                )
            )

        return output_dict

    def forward(self, x):
        return self.compute_extended_output(x)["output"]

    def description(self):
        layer_type = "LinearCMM" if self.cmm else "torch.nn.Linear"

        repr = "torch.nn.Flatten()\n" if self.flatten else ""
        repr += f"{layer_type}({self.input_size}, {self.hidden_units}, {'' if not self.cmm else str(self.cmm_args)}),\n"
        repr += "torch.nn.ReLU(),\n"
        repr += f"torch.nn.Dropout(p={self.dropout}),\n" if self.dropout != 1 else ""
        repr += f"{layer_type}({self.hidden_units}, {self.num_classes}, {'' if not self.cmm else str(self.cmm_args)}),\n"
        repr += f"torch.nn.Sigmoid(),\n" if self.output == "sigmoid" else f"torch.nn.Softmax(),\n" if self.output == "softmax" else ""

        return f"torch.nn.Sequential({repr}\n)"
        
class Half_CMM(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes, dropout=0.0, flatten=True, cmm=False, cmm_args={}, output="logits"):
        super(Half_CMM, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.dropout = dropout
        self.flatten = flatten
        self.cmm = cmm
        self.output = output
        layer_type = LinearCMM if cmm else torch.nn.Linear
        self.cmm_args = cmm_args if cmm else {}

        flatten = (torch.nn.Flatten(),) if flatten else ()

        self.flatten = torch.nn.Flatten()
        self.linear1 = layer_type(input_size, hidden_units, **cmm_args)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear2 = torch.nn.Linear(hidden_units, num_classes)
        if output == "sigmoid":
            self.output_layer = torch.nn.Sigmoid()
        elif output == "softmax":
            self.output_layer = torch.nn.Softmax()

    def get_continual_model_parameters(self):
        return dict(
            cmm_params_layer1 = self.linear1.get_continual_model_parameters()
        )
    
    def compute_extended_output(self, x):
        O = self.flatten(x)
        
        if self.cmm:
            output1 = self.linear1.compute_extended_output(O)
            O = self.relu1(output1["output"])
        else:
            O = self.linear1(O)
            O = self.relu1(O)
        
        O = self.dropout(O)

        A = self.linear2(O)

        O = A if self.output == "logits" else self.output_layer(A)

        output_dict = dict(
            logits = A,
            output = O,
        )

        if self.cmm:
            output_dict.update(
                dict(
                    memory_model_parameters = dict(
                        cmm_params_layer1 = output1["memory_model_parameters"]
                    )
                )
            )

        return output_dict

    def forward(self, x):
        return self.compute_extended_output(x)["output"]

    def description(self):
        layer_type = "LinearCMM" if self.cmm else "torch.nn.Linear"

        repr = "torch.nn.Flatten()\n" if self.flatten else ""
        repr += f"{layer_type}({self.input_size}, {self.hidden_units}, {'' if not self.cmm else str(self.cmm_args)}),\n"
        repr += "torch.nn.ReLU(),\n"
        repr += f"torch.nn.Dropout(p={self.dropout}),\n" if self.dropout != 1 else ""
        repr += f"torch.nn.Linear({self.hidden_units}, {self.num_classes}),\n"
        repr += f"torch.nn.Sigmoid(),\n" if self.output == "sigmoid" else f"torch.nn.Softmax(),\n" if self.output == "softmax" else ""

        return f"torch.nn.Sequential({repr}\n)"
        

class Head_CMM(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes, dropout=0.0, flatten=True, cmm=False, cmm_args={}, output="logits"):
        super(Head_CMM, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.dropout = dropout
        self.flatten = flatten
        self.cmm = cmm
        self.output = output
        layer_type = LinearCMM if cmm else torch.nn.Linear
        self.cmm_args = cmm_args if cmm else {}

        flatten = (torch.nn.Flatten(),) if flatten else ()

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(hidden_units, num_classes)
        self.relu1 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear2 = layer_type(input_size, hidden_units, **cmm_args)
        if output == "sigmoid":
            self.output_layer = torch.nn.Sigmoid()
        elif output == "softmax":
            self.output_layer = torch.nn.Softmax()

    def get_continual_model_parameters(self):
        return dict(
            cmm_params_layer2 = self.linear2.get_continual_model_parameters()
        )
    
    def compute_extended_output(self, x):
        O = self.flatten(x)
        
        O = self.linear1(O)
        O = self.relu1(O)
        
        O = self.dropout(O)

        if self.cmm:
            output2 = self.linear2.compute_extended_output(O)
            A = output2["output"]
        else:
            A = self.linear2(O)

        O = A if self.output == "logits" else self.output_layer(A)

        output_dict = dict(
            logits = A,
            output = O,
        )

        if self.cmm:
            output_dict.update(
                dict(
                    memory_model_parameters = dict(
                        cmm_params_layer2 = output2["memory_model_parameters"]
                    )
                )
            )

        return output_dict

    def forward(self, x):
        return self.compute_extended_output(x)["output"]

    def description(self):
        layer_type = "LinearCMM" if self.cmm else "torch.nn.Linear"

        repr = "torch.nn.Flatten()\n" if self.flatten else ""
        repr += f"torch.nn.Linear({self.hidden_units}, {self.num_classes}),\n"
        repr += "torch.nn.ReLU(),\n"
        repr += f"torch.nn.Dropout(p={self.dropout}),\n" if self.dropout != 1 else ""
        repr += f"{layer_type}({self.input_size}, {self.hidden_units}, {'' if not self.cmm else str(self.cmm_args)}),\n"
        repr += f"torch.nn.Sigmoid(),\n" if self.output == "sigmoid" else f"torch.nn.Softmax(),\n" if self.output == "softmax" else ""

        return f"torch.nn.Sequential({repr}\n)"
        


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