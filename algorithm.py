import torch
from collections.abc import Iterable

from experiment import ETExperiment

class ETAlgorithm:
    def __init__(self, training_function, training_function_args=None):
        self.training_function = training_function
        self.training_function_args = training_function_args

    def __call__(self, experiment):
        return self.training_function(experiment, **self.training_function_args)

class ETPipeline:
    def __init__(self, pipeline, name):
        if not isinstance(pipeline, Iterable):
            raise TypeError("The pipeline should be an iterator of ETAlgorithm/ETPipeline.")
        if not isinstance(name, str):
            raise TypeError("The name of the pipeline should be a str.")

        self.pipeline = list()
        for algorithm in pipeline:
            if not isinstance(algorithm, ETAlgorithm) and not isinstance(algorithm, ETPipeline):
                raise TypeError("The pipeline should be an iterator of ETAlgorithm/ETPipeline.")
            self.pipeline.append(algorithm)

        self.name = name
    
    def __call__(self, experiment):
        if not isinstance(experiment, ETExperiment):
            raise TypeError("The argument 'experiment' should be an instance of ETExperiment.")

        pipeline_results = {self.name: {}}
        for algorithm in self.pipeline:
            pipeline_results[self.name].update(algorithm(experiment))

        return pipeline_results

def train(
    experiment,
    epoch_number=1,
    optimizer="adam",
    lr=0.01,
    weight_decay=0,
    batch_size=None,
    shuffle=True,
    device="cpu",
    dataloader_args=None
):
    """Train the network using the specified options.
    The training is single-pass, and the evaluation is performed at the end of each class.
    Args:
        data: an instance of ETDataset containg the training data in the form (image, label).
        optimizer: "adam" or "sgd".
        lr: learning rate.
        weight_decay: weight multiplying the weight decay regularizaion term.
    """

    data = experiment.training_set
    
    model = experiment.model.to(torch.device(device))

    num_examples = len(data)

    batch_size = num_examples if batch_size == None else batch_size
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, **dataloader_args)

    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
        
    # save the train/eval mode of the network and change it to training mode
    training = model.training
    model.train()

    # use cross-entropy loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # create the optimizer selected by the caller of the function
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer "{optimizer}" not defined.')
    
    training_log = f"""Number of examples: {num_examples}
Epochs: {epoch_number}
Loss function: categorical cross-entropy
Optimizer: {optimizer}
Learning rate: {lr}
Weight decay: {weight_decay}
Batch size: {batch_size},
Shuffle: {shuffle},
Device: {device},
Dataloader args: {dataloader_args}


"""

    for epoch in range(1, 1 + epoch_number):
        print(f"Start epoch {epoch}")
        training_log += f"Start epoch {epoch}\n"
        n_examples = 0
        for img_mini_batch, label_mini_batch in dataloader:
            # send the mini-batch to the device memory
            img_mini_batch = img_mini_batch.to(device)
            label_mini_batch = label_mini_batch.to(device)

            # forward step
            # compute the output (actually the logist) of the model on the current example
            logits = model(img_mini_batch)

            # compute the loss function
            loss = loss_function(logits, label_mini_batch)

            n_examples += img_mini_batch.shape[0]

            # print the loss function, once every 100 epochs
            print(f"Epoch = {epoch}\t\tExample = {n_examples}\t\tLoss = {loss.item():<.4f}")
            training_log += f"Epoch = {epoch}\t\tExample = {n_examples}\t\tLoss = {loss.item():<.4f}\n"

            # perform the backward step and the optimization step
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        print(f"End epoch {epoch}")
        training_log += f"End epoch {epoch}\n"

    # recover the initial train/eval mode
    if not training: model.eval()

    return dict(training_log = training_log, trained_model = model) 