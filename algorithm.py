from asyncio.format_helpers import _format_callback_source
import torch
from collections.abc import Iterable
import matplotlib.pyplot as plt
from experiment import ETExperiment
from math import ceil

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
    optimizer_args={},
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
        optimizer = optimizers[optimizer_name](experiment.model.parameters(), **optimizer_args)
    else:
        raise ValueError(f'Optimizer "{optimizer_name}" not defined.')
    
    training_log = f"""Number of examples: {num_examples}
Epochs: {epoch_number}
Loss function: {loss_function_name}
Optimizer: {optimizer_name}
Optimizer args: {optimizer_args}
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

def task_incremental_train(
    experiment,
    epoch_number=1,
    loss_function_name="cross_entropy_loss",
    optimizer_name="adam",
    optimizer_args={},
    device="cpu",
    batch_size=1,
    dataloader_args=None,
    evaluate=True
):
    data = experiment.training_set
    num_classes = data.num_classes

    model = experiment.model.to(torch.device(device))

    num_examples = len(data)

    batch_size = 1 if batch_size == None else batch_size

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
        optimizer = optimizers[optimizer_name](experiment.model.parameters(), **optimizer_args)
    else:
        raise ValueError(f'Optimizer "{optimizer_name}" not defined.')
    
    training_log = f"""Number of examples: {num_examples}
Loss function: {loss_function_name}
Optimizer: {optimizer_name}
Optimizer args: {optimizer_args}
Batch size: {batch_size},
Device: {device},
Dataloader args: {dataloader_args}


"""

    # initializing the torch tensor that will contain the history of the evaluation during training
    history = torch.empty((num_classes, num_classes), dtype=torch.float32)

    for current_label in data.class_map.values():
        print(f"Start training label {current_label}.")
        training_log += f"Start training label {current_label}.\n"
        current_label_data = data.get_subset_by_label(current_label)
        dataloader = torch.utils.data.DataLoader(current_label_data, batch_size=batch_size, shuffle=False, **dataloader_args)
        for epoch in range(epoch_number):
            print(f"\nEpoch: {epoch}.\n")
            training_log += f"\nEpoch: {epoch}.\n"
            n_examples = 0
            for img_mini_batch, label_mini_batch in dataloader:
                # send the mini-batch to the device memory
                img_mini_batch = img_mini_batch.to(device)
                label_mini_batch = label_mini_batch.to(device)
                
                logits = model(img_mini_batch)
                mini_batch_loss = loss_function(logits, label_mini_batch)

                n_examples += batch_size

                # perform the backward step and the optimization step
                mini_batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"Example = {n_examples}\t\tLoss = {mini_batch_loss.item():<.4f}")
                training_log += f"Example = {n_examples}\t\tLoss = {mini_batch_loss.item():<.4f}\n"

        if evaluate:
            history[current_label] = evaluate_class_by_class(
                model,
                experiment.test_set,
                metrics = "accuracy",
                device = device,
                batch_size = batch_size,
                dataloader_args = dataloader_args
            )

    print(f"End training.")
    training_log += f"End training.\n"

    # recover the initial train/eval mode
    if not training: model.eval()

    output = dict(training_log = training_log, trained_model = model)

    if evaluate:
        plot1, plot2 = task_incremental_plot(history, range(1, 1 + num_classes))
        output["training_history"] = str(history)
        output["training_history_tensor"]
        output["macro_accuracy"] = str(history.mean(dim=1))
        output["macro_accuracy_tensor"] = history.mean(dim=1)
        output["task_by_task_class_accuracy"] = plot1
        output["task_by_task_total_accuracy"] = plot2

    return output

def single_pass_online_train(
    experiment,
    loss_function_name="cross_entropy_loss",
    optimizer_name="adam",
    optimizer_args={},
    device="cpu",
    batch_size=1,
    dataloader_args=None,
    evaluate=True,
    evaluation_step="mini_batch"
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
    data.dataset_wise_sort_by_label()

    num_classes = data.num_classes
    
    model = experiment.model.to(torch.device(device))

    num_examples = len(data)

    batch_size = 1 if batch_size == None else batch_size
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
        optimizer = optimizers[optimizer_name](experiment.model.parameters(), **optimizer_args)
    else:
        raise ValueError(f'Optimizer "{optimizer_name}" not defined.')
    
    training_log = f"""Number of examples: {num_examples}
Loss function: {loss_function_name}
Optimizer: {optimizer_name}
Optimizer args: {optimizer_args}
Batch size: {batch_size},
Device: {device},
Dataloader args: {dataloader_args}


"""

    # select the initial label, assuming it to be 0
    current_label = 0

    evaluation_steps = 0
    if isinstance(evaluation_step, int): evaluation_steps = ceil(len(dataloader) / evaluation_step)
    elif evaluation_step == "task": evaluation_steps = num_classes
    elif evaluation_step == "mini_batch": evaluation_steps = len(dataloader)
    else: raise ValueError(f"evaluation_step = '{evaluation_step}', expected int or str in ['task', 'mini_batch'].")

    # initializing the torch tensor that will contain the history of the evaluation during training
    history = torch.empty((evaluation_steps, num_classes), dtype=torch.float32)
    ticks = torch.empty((evaluation_steps,), dtype=torch.float32)

    print(f"Start training.")
    training_log += f"Start training.\n"
    n_examples = 0
    current_class = 0
    current_evaluation_step = 0
    for mini_batch_idx, (img_mini_batch, label_mini_batch) in enumerate(dataloader):
        do_evaluation = True if evaluate and (
            isinstance(evaluation_step, int) and (mini_batch_idx + 1) % evaluation_step == 0 or
            evaluation_step == "task" and label_mini_batch[-1].item() != current_class or
            evaluation_step == "mini_batch" or
            mini_batch_idx == len(dataloader) - 1
            ) else False

        # send the mini-batch to the device memory
        img_mini_batch = img_mini_batch.to(device)
        label_mini_batch = label_mini_batch.to(device)

        logits = model(img_mini_batch)

        mini_batch_loss = loss_function(logits, label_mini_batch)

        n_examples += batch_size

        # perform the backward step and the optimization step
        mini_batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Example = {n_examples}\t\tLoss = {mini_batch_loss.item():<.4f}")
        training_log += f"Example = {n_examples}\t\tLoss = {mini_batch_loss.item():<.4f}\n"

        while do_evaluation:
            history[current_evaluation_step] = evaluate_class_by_class(
                model,
                experiment.test_set,
                metrics = "accuracy",
                device = device,
                batch_size = batch_size,
                dataloader_args = dataloader_args
            )
            
            ticks[current_evaluation_step] = n_examples

            current_evaluation_step += 1

            if label_mini_batch[-1].item() != current_class:
                current_class += 1

            if evaluation_step != "task" or label_mini_batch[-1].item() == current_class:
                do_evaluation = False
    
    print(f"End training.")
    training_log += f"End training.\n"

    # recover the initial train/eval mode
    if not training: model.eval()

    output = dict(training_log = training_log, trained_model = model)

    if evaluate:
        plot1, plot2 = task_incremental_plot(history, ticks)
        output["training_history"] = str(history)
        output["training_history_tensor"] = history
        output["macro_accuracy"] = str(history.mean(dim=1))
        output["macro_accuracy_tensor"] = history.mean(dim=1)
        output["step_by_step_class_accuracy"] = plot1
        output["step_by_step_total_accuracy"] = plot2

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

def task_incremental_plot(history, ticks):
    num_classes = history.shape[1]

    macro_accuracy = history.mean(dim=1)
    
    f = plt.figure()
    ax = f.add_subplot(111)
    
    for h in history.transpose(0, 1):
        ax.plot(ticks, h)
    
    print(ticks)

    if len(ticks) > 50:
        ticks = torch.sort(ticks[torch.randperm(len(ticks))][:50])

    print(ticks)

    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(ticks)
    ax.set_xlabel("computed after training step #")
    ax.set_ylabel("accuracy")
    ax.set_title("Class-wise accuracy at different training steps")

    f_macro = plt.figure()
    ax = f_macro.add_subplot(111)
    ax.plot(ticks, macro_accuracy)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(ticks)
    ax.set_xlabel("computed after training step #")
    ax.set_ylabel("accuracy")
    ax.set_title("Total accuracy at different training steps")

    return f, f_macro