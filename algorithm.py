import torch
from collections.abc import Iterable

from experiment import ETExperiment

loss_functions = dict(
    cross_entropy_loss = torch.nn.CrossEntropyLoss
)

optimizers = dict(
    adam = torch.optim.Adam,
    sgd = torch.optim.SGD
)

class ETAlgorithm:
    def __init__(self, training_function, training_function_args=None):
        self.training_function = training_function
        self.training_function_args = training_function_args

    def __call__(self, experiment):
        return self.training_function(experiment, **self.training_function_args)

class ETPipeline:
    def __init__(self, name, pipeline):
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
    loss_function_name="cross_entropy_loss",
    optimizer_name="adam",
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

    # select loss function
    if loss_function_name in loss_functions:
        loss_function = loss_functions[loss_function_name]()
    else:
        raise ValueError(f'Loss function "{loss_function_name}" not defined.')

    # create the optimizer selected by the caller of the function
    if optimizer_name in optimizers:
        optimizer = optimizers[optimizer_name](experiment.model.parameters(), lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer "{optimizer_name}" not defined.')
    
    training_log = f"""Number of examples: {num_examples}
Epochs: {epoch_number}
Loss function: {loss_function_name}
Optimizer: {optimizer_name}
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
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"End epoch {epoch}")
        training_log += f"End epoch {epoch}\n"

    # recover the initial train/eval mode
    if not training: model.eval()

    return dict(training_log = training_log, trained_model = model)

def evaluate(experiment,
    metrics="accuracy",
    batch_size=None,
    shuffle=True,
    device="cpu",
    dataloader_args=None
):
    data = experiment.test_set
    
    model = experiment.model.to(torch.device(device))

    num_examples = len(data)

    batch_size = num_examples if batch_size == None else batch_size
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, **dataloader_args)
        
    # save the train/eval mode of the network and change it to eval mode
    training = model.training
    model.eval()
    
    test_log = f"""Number of examples: {num_examples}
Batch size: {batch_size},
Shuffle: {shuffle},
Device: {device},
Dataloader args: {dataloader_args}


"""
    if metrics != "accuracy": raise NotImplementedError
    accuracy = 0
    with torch.no_grad():
        for img_mini_batch, label_mini_batch in dataloader:
            # send the mini-batch to the device memory
            img_mini_batch = img_mini_batch.to(device)
            label_mini_batch = label_mini_batch.to(device)

            # forward step
            # compute the output (actually the logist) of the model on the current example
            logits = model(img_mini_batch)
            predicted_labels = torch.argmax(logits, dim=1)
            accuracy += torch.sum(torch.eq(predicted_labels, label_mini_batch))
    
    accuracy = accuracy / num_examples

    # recover the initial train/eval mode
    if training: model.train()

    return dict(accuracy = accuracy.item())

def classwise_train(
    experiment,
    epoch_number=1,
    loss_function_name="cross_entropy_loss",
    optimizer_name="adam",
    lr=0.01,
    weight_decay=0,
    device="cpu",
    batch_size=1,
    dataloader_args=None,
    evaluate=True
):
    """Train the network using the specified options.
    The training is single-pass, and the evaluation is performed at the end of each class.
    Args:
        data: a dataset or dataloader containg the training data in the form (image, label).
        test_data: a dataset or dataloader (as the previous one) containing the test data.
        optimizer: "adam" or "sgd".
        lr: learning rate.
        weight_decay: weight multiplying the weight decay regularizaion term.
        plot: if True, the history is plotted at the end of the training procedure.
    """

    data = experiment.training_set
    
    model = experiment.model.to(torch.device(device))

    num_examples = len(data)

    batch_size = num_examples if batch_size == None else batch_size
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **dataloader_args)

    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)

    # save the train/eval mode of the network and change it to training mode
    training = model.training
    model.train()

    # select loss function
    if loss_function_name in loss_functions:
        loss_function = loss_functions[loss_function_name]()
    else:
        raise ValueError(f'Loss function "{loss_function_name}" not defined.')

    # create the optimizer selected by the caller of the function
    if optimizer_name in optimizers:
        optimizer = optimizers[optimizer_name](experiment.model.parameters(), lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer "{optimizer_name}" not defined.')
    
    training_log = f"""Number of examples: {num_examples}
Epochs: {epoch_number}
Loss function: {loss_function_name}
Optimizer: {optimizer_name}
Learning rate: {lr}
Weight decay: {weight_decay}
Batch size: {batch_size},
Device: {device},
Dataloader args: {dataloader_args}


"""

    # select the initial label, assuming it to be 0
    current_label = 0

    # initializing the torch tensor that will contain the history of the evaluation during training
    history = torch.empty((experiment.dataset.num_classes, experiment.dataset.num_classes), dtype=torch.float32)

    print(f"Start training.")
    training_log += f"Start training.\n"
    n_examples = 0
    for img_mini_batch, label_mini_batch in dataloader:
        # send the mini-batch to the device memory
        img_mini_batch = img_mini_batch.to(device)
        label_mini_batch = label_mini_batch.to(device)

        mini_batch_loss = torch.zeros((1,), requires_grad=False)

        # loop over the examples of the current mini-batch
        for img, label in zip(img_mini_batch, label_mini_batch):
            # if we passed to another class, evalute the model on the whole test set and save the results
            if evaluate and label != current_label:
                history[current_label] = evaluate_class_by_class(experiment.test_set)
                current_label = label

            # forward step
            # compute the output (actually the logist) of the model on the current example
            logits = model(img.view((1, *img.shape)))

            # compute the loss function
            loss = loss_function(logits, label.view((1,)))
            mini_batch_loss += loss
            n_examples += 1

            # perform the backward step and the optimization step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # print the loss function, once every 100 examples
        print(f"Example = {n_examples}\t\tLoss = {mini_batch_loss.item():<.4f}")
        training_log += f"Example = {n_examples}\t\tLoss = {mini_batch_loss.item():<.4f}\n"
    
    print(f"End training.")
    training_log += f"End training.\n"

    # recover the initial train/eval mode
    if not training: model.eval()

    output = dict(training_log = training_log, trained_model = model)

    if evaluate:
        history[current_label] = evaluate_class_by_class(
            model,
            experiment.test_set,
            metrics="accuracy",
            device=device,
            batch_size=batch_size,
            dataloader_args=dataloader_args
        )
        plot = plot(plot=plot, savefig=savefig, **kwargs)
        output["training_history"] = history
        output["evaluation_during_training_plot"] = plot

    return output

def evaluate_class_by_class(model, test_data, metrics="accuracy", device="cpu", batch_size=1, dataloader_args=None):
    """Compute and retrn the accuracy of the classifier on each single class.

    Returns: a torch.Tensor containg the accuracy on each class.
    """
    
    if metrics != "accuracy": raise NotImplementedError

    # save the train/eval mode of the network and change it to evaluation mode
    training = model.training
    model.eval()

    # get the targets and the number of classes
    classes = sorted(list(test_data.class_map.values()))
    
    # initializing the counters for the class-by-class accuracy
    true_positive = torch.zeros((test_data.num_classes,)).to(device)
    total = torch.zeros((test_data.num_classes,)).to(device)
    
    num_examples = len(test_data)

    batch_size = num_examples if batch_size == None else batch_size
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **dataloader_args)

    with torch.no_grad():
        # loop over the mini-batches
        for img, label in dataloader:
            # send the mini-batch to the device memory
            img = img.to(device)
            label = label.to(device)

            # compute the output of the model on the current mini-batch
            logits = model(img)

            # decision rule
            output_label = torch.argmax(logits, dim=-1).to(device)

            # update the counters
            for class_index, c in enumerate(classes):
                true_positive[class_index] += torch.sum(torch.logical_and(label==c, output_label==c))
                total[class_index] += torch.sum(label==c)

    # recover the initial train/eval mode
    if training: model.train()

    # return the 1D tensor of accuracies of each class
    return true_positive / total

