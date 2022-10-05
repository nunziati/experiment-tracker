import torch
from collections.abc import Iterable
import matplotlib.pyplot as plt
from math import ceil

loss_functions = dict(
    cross_entropy_loss = torch.nn.CrossEntropyLoss,
    binary_cross_entropy_loss = torch.nn.BCELoss,
    binary_cross_entropy_loss_with_logits = torch.nn.BCEWithLogitsLoss
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
        pipeline_results = {self.name: {}}
        for algorithm in self.pipeline:
            pipeline_results[self.name].update(algorithm(experiment))

        return pipeline_results

def train(
    experiment,
    epoch_number=1,
    loss_function_name="cross_entropy_loss",
    loss_function_args={},
    optimizer_name="adam",
    optimizer_args={},
    batch_size=None,
    shuffle=True,
    device="cpu",
    dataloader_args=None,
    validation=None,
    **kwargs
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
    
    if validation != None:
        training_data, validation_data = data.split((1 - validation, validation)) 
    else:
        training_data, validation_data = data, None
    
    model = experiment.model.to(torch.device(device))

    num_examples = len(training_data)

    batch_size = num_examples if batch_size == None else batch_size
    training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=shuffle, **dataloader_args)
    validation_dataloader = (
        torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=shuffle, **dataloader_args)
        if validation_data != None
        else None
    )

    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.01, 0.01)
        
    # save the train/eval mode of the network and change it to training mode
    training = model.training
    model.train()

    # select loss function
    if loss_function_name in loss_functions:
        loss_function = loss_functions[loss_function_name](**loss_function_args)
    else:
        raise ValueError(f'Loss function "{loss_function_name}" not defined.')

    # create the optimizer selected by the caller of the function
    if optimizer_name in optimizers:
        optimizer = optimizers[optimizer_name](experiment.model.parameters(), **optimizer_args)
    else:
        raise ValueError(f'Optimizer "{optimizer_name}" not defined.')
    
    training_log = f"""Number of total examples: {len(data)}
Number of training examples: {num_examples}
Number of validation examples: {0 if validation == None else len(validation_data)}
Epochs: {epoch_number}
Loss function: {loss_function_name}
Optimizer: {optimizer_name}
Optimizer args: {optimizer_args}
Batch size: {batch_size},
Shuffle: {shuffle},
Device: {device},
Dataloader args: {dataloader_args}
Validation: {0.0 if validation == None else validation * 100:<.2f} % of the training examples

"""
    learning_curve_x = torch.empty((epoch_number * len(training_dataloader),), dtype=torch.float32)
    learning_curve_y = torch.empty((epoch_number * len(training_dataloader),), dtype=torch.float32)
    
    if validation != None:
        generalization_curve_x = torch.empty((epoch_number,), dtype=torch.float32)
        generalization_curve_y = torch.empty((epoch_number,), dtype=torch.float32)

    best_loss = None
    for epoch in range(1, 1 + epoch_number):
        print(f"Start epoch {epoch}")
        training_log += f"Start epoch {epoch}\n"
        n_examples = 0
        mini_batch_idx = 0
        for img_mini_batch, label_mini_batch in training_dataloader:
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

            learning_curve_x[(epoch - 1) * len(training_dataloader) + mini_batch_idx] = (epoch - 1) + n_examples / len(training_data)
            learning_curve_y[(epoch - 1) * len(training_dataloader) + mini_batch_idx] = loss.item()

            mini_batch_idx += 1
            
        # recover the initial train/eval mode
        if not training: model.eval()

        if validation_dataloader != None:
            # validation
            with torch.no_grad():
                loss = torch.zeros((len(validation_dataloader),), device=device)
                for mini_batch_idx, (img_mini_batch, label_mini_batch) in enumerate(validation_dataloader):
                    # send the mini-batch to the device memory
                    img_mini_batch = img_mini_batch.to(device)
                    label_mini_batch = label_mini_batch.to(device)

                    # forward step
                    # compute the output (actually the logist) of the model on the current example
                    logits = model(img_mini_batch)

                    # compute the loss function
                    loss[mini_batch_idx] = loss_function(logits, label_mini_batch)

                loss = torch.mean(loss)

            if best_loss == None or loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_model = model.state_dict()

            print(f"\nEpoch = {epoch}\t\tValidation loss = {loss.item():<.4f}\t\t{'(BEST)' if loss == best_loss else ''}\n")
            training_log += f"\nEpoch = {epoch}\t\tValidation loss = {loss.item():<.4f}\t\t{'(BEST)' if loss == best_loss else ''}\n\n"

            generalization_curve_x[epoch - 1] = epoch
            generalization_curve_y[epoch - 1] = loss.item()

        else:
            best_loss = None
            best_model = model.state_dict()
            best_epoch = None

        print(f"End epoch {epoch}")
        training_log += f"End epoch {epoch}\n"

    model.load_state_dict(best_model)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(learning_curve_x[len(training_dataloader):], learning_curve_y[len(training_dataloader):])

    if validation != None:
        ax.plot(generalization_curve_x, generalization_curve_y)

    return dict(
        training_log = training_log,
        trained_model = model,
        best_epoch = best_epoch,
        best_validation_loss = best_loss,
        learning_curve = f    
    )

def evaluate(experiment,
    metrics="accuracy",
    batch_size=None,
    shuffle=True,
    device="cpu",
    dataloader_args=None,
    **kwargs
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

    return dict(test_log = test_log, accuracy = accuracy.item())

def task_incremental_train(
    experiment,
    epoch_number=1,
    loss_function_name="cross_entropy_loss",
    loss_function_args={},
    optimizer_name="adam",
    optimizer_args={},
    device="cpu",
    batch_size=1,
    evaluation_batch_size=None,
    dataloader_args=None,
    **kwargs
):
    data = experiment.training_set
    num_classes = data.num_classes

    model = experiment.model.to(torch.device(device))

    num_examples = len(data)

    batch_size = 1 if batch_size == None else batch_size
    if evaluation_batch_size == None: evaluation_batch_size = batch_size

    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.01, 0.01)

    # save the train/eval mode of the network and change it to training mode
    training = model.training
    model.train()

    # select loss function
    if loss_function_name in loss_functions:
        loss_function = loss_functions[loss_function_name](**loss_function_args)
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

        history[current_label] = evaluate_task_by_task(
            model,
            experiment.test_set,
            metrics = "accuracy",
            device = device,
            batch_size = evaluation_batch_size,
            dataloader_args = dataloader_args
        )

    print(f"End training.")
    training_log += f"End training.\n"

    # recover the initial train/eval mode
    if not training: model.eval()

    output = dict(training_log = training_log, trained_model = model)

    plot1, plot2 = task_incremental_plot(history, range(1, 1 + num_classes), use_ticks=True)
    output["training_history"] = str(history)
    output["training_history_tensor"] = history
    output["macro_accuracy"] = str(history.mean(dim=1))
    output["macro_accuracy_tensor"] = history.mean(dim=1)
    output["task_by_task_class_accuracy"] = plot1
    output["task_by_task_total_accuracy"] = plot2

    return output

def single_pass_online_train(
    experiment,
    sorted=True,
    loss_function_name="cross_entropy_loss",
    loss_function_args={},
    optimizer_name="adam",
    optimizer_args={},
    device="cpu",
    batch_size=1,
    dataloader_args=None,
    evaluation_step="mini_batch",
    evaluation_batch_size=None,
    **kwargs
):

    data = experiment.training_set
    if sorted: data.dataset_wise_sort_by_label()

    num_classes = data.num_classes
    
    model = experiment.model.to(torch.device(device))

    num_examples = len(data)

    tasks = data.tasks
    num_tasks = len(tasks)
    
    batch_size = 1 if batch_size == None else batch_size
    if evaluation_batch_size == None: evaluation_batch_size = batch_size
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=not sorted, **dataloader_args)

    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.01, 0.01)

    # save the train/eval mode of the network and change it to training mode
    training = model.training
    model.train()

    # select loss function
    if loss_function_name in loss_functions:
        loss_function = loss_functions[loss_function_name](**loss_function_args)
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

    evaluation_steps = 0
    if isinstance(evaluation_step, int): evaluation_steps = ceil(len(dataloader) / evaluation_step)
    elif evaluation_step == "task": evaluation_steps = num_tasks
    elif evaluation_step == "mini_batch": evaluation_steps = len(dataloader)
    else: raise ValueError(f"evaluation_step = '{evaluation_step}', expected int or str in ['task', 'mini_batch'].")

    # initializing the torch tensor that will contain the history of the evaluation during training
    class_history = torch.empty((evaluation_steps, num_classes), dtype=torch.float32)
    
    if sorted:
        task_history = torch.empty((evaluation_steps, num_tasks), dtype=torch.float32)
        past_tasks_accuracy = torch.empty((evaluation_steps), dtype=torch.float32)
        current_task_accuracy = torch.empty((evaluation_steps), dtype=torch.float32)
        next_tasks_accuracy = torch.empty((evaluation_steps), dtype=torch.float32)
    
    ticks = torch.empty((evaluation_steps,), dtype=torch.float32)

    print(f"Start training.")
    training_log += f"Start training.\n"
    n_examples = 0
    if evaluation_step == "task": current_task_idx = 0
    current_evaluation_step = 0
    for mini_batch_idx, (img_mini_batch, label_mini_batch) in enumerate(dataloader):
        do_evaluation = (
            isinstance(evaluation_step, int) and (mini_batch_idx + 1) % evaluation_step == 0 or
            evaluation_step == "task" and label_mini_batch[-1].item() not in tasks[current_task_idx] or
            evaluation_step == "task" and mini_batch_idx != len(dataloader) - 1 and data[(mini_batch_idx + 1) * batch_size][1] not in tasks[current_task_idx] or
            evaluation_step == "mini_batch" or
            mini_batch_idx == len(dataloader) - 1
        )

        mini_batch_part_size = 10

        mini_batch_parts = ceil(img_mini_batch.shape[0] / mini_batch_part_size)

        for mini_batch_part_idx in range(mini_batch_parts):
            img_mini_batch_part = img_mini_batch[mini_batch_part_idx * mini_batch_part_size:(mini_batch_part_idx + 1) * mini_batch_part_size]
            label_mini_batch_part = label_mini_batch[mini_batch_part_idx * mini_batch_part_size:(mini_batch_part_idx + 1) * mini_batch_part_size]
            
            # send the mini-batch part to the device memory
            img_mini_batch_part = img_mini_batch_part.to(device)
            label_mini_batch_part = label_mini_batch_part.to(device)

            logits = model(img_mini_batch_part)
            
            # compute the loss function
            if mini_batch_part_idx == 0:
                if loss_function_name == "cross_entropy_loss":
                    mini_batch_loss = loss_function(logits, label_mini_batch_part)
                else:
                    one_hot_label_mini_batch_part = torch.nn.functional.one_hot(label_mini_batch_part, num_classes=num_classes).float()
                    mini_batch_loss = loss_function(logits, one_hot_label_mini_batch_part)
            else:
                if loss_function_name == "cross_entropy_loss":
                    mini_batch_loss += loss_function(logits, label_mini_batch_part)
                else:
                    one_hot_label_mini_batch_part = torch.nn.functional.one_hot(label_mini_batch_part, num_classes=num_classes).float()
                    mini_batch_loss += loss_function(logits, one_hot_label_mini_batch_part)

        n_examples += batch_size

        # perform the backward step and the optimization step
        mini_batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Example = {n_examples}\t\tLoss = {mini_batch_loss.item():<.4f}")
        training_log += f"Example = {n_examples}\t\tLoss = {mini_batch_loss.item():<.4f}\n"

        while do_evaluation:
            evaluation_results = evaluate_task_by_task(
                model,
                experiment.test_set,
                metrics = "accuracy",
                device = device,
                batch_size = evaluation_batch_size,
                dataloader_args = dataloader_args,
                current_task = None if evaluation_step != "task" else current_task_idx
            )

            class_history[current_evaluation_step] = evaluation_results["class_history"]
            if sorted:
                task_history[current_evaluation_step] = evaluation_results["task_history"]
                past_tasks_accuracy[current_evaluation_step] = evaluation_results["past_tasks_accuracy"]
                current_task_accuracy[current_evaluation_step] = evaluation_results["current_task_accuracy"]
                next_tasks_accuracy[current_evaluation_step] = evaluation_results["next_tasks_accuracy"]

            ticks[current_evaluation_step] = n_examples

            current_evaluation_step += 1

            if evaluation_step == "task":
                if (
                    label_mini_batch[-1].item() not in tasks[current_task_idx] or
                    mini_batch_idx != len(dataloader) - 1 and data[(mini_batch_idx + 1) * batch_size][1] not in tasks[current_task_idx]
                ):
                    current_task_idx += 1
            else:
                do_evaluation = False
            
            if evaluation_step == "task" and (
                label_mini_batch[-1].item() in tasks[current_task_idx] or
                mini_batch_idx != len(dataloader) - 1 and data[(mini_batch_idx + 1) * batch_size][1] in tasks[current_task_idx]
            ):
                do_evaluation = False
    
    print(f"End training.")
    training_log += f"End training.\n"

    # recover the initial train/eval mode
    if not training: model.eval()

    # output = dict(training_log = training_log, trained_model = model)
    output = dict(training_log = training_log)
    
    plot1, plot2 = task_incremental_plot(class_history, ticks)
    output["training_class_history"] = str(class_history)
    output["training_class_history_tensor"] = class_history
    output["class_macro_accuracy"] = str(class_history.mean(dim=1))
    output["class_macro_accuracy_tensor"] = class_history.mean(dim=1)
    output["step_by_step_class_accuracy"] = plot1
    output["step_by_step_total_accuracy_by_class"] = plot2

    if sorted:
        plot3, plot4 = task_incremental_plot(task_history, ticks)
        output["training_task_history"] = str(task_history)
        output["training_task_history_tensor"] = task_history
        output["task_macro_accuracy"] = str(task_history.mean(dim=1))
        output["task_macro_accuracy_tensor"] = task_history.mean(dim=1)
        output["step_by_step_task_accuracy"] = plot3
        output["step_by_step_total_accuracy_by_task"] = plot4

        output["past_tasks_accuracy"] = get_plot(ticks, past_tasks_accuracy, "Past tasks accuracy")
        output["current_task_accuracy"] = get_plot(ticks, current_task_accuracy, "Current task accuracy")
        output["next_tasks_accuracy"] = get_plot(ticks, next_tasks_accuracy, "Next tasks accuracy")

    return output

def evaluate_task_by_task(model, test_data, metrics="accuracy", device="cpu", batch_size=1, dataloader_args=None, current_task=None):        
    if metrics != "accuracy": raise NotImplementedError

    # save the train/eval mode of the network and change it to evaluation mode
    training = model.training
    model.eval()

    # get the targets and the number of classes
    classes = sorted(list(test_data.class_map.values()))
    if current_task is not None:
        tasks = test_data.tasks
    
    # initializing the counters for the metrics
    class_true_positive = torch.zeros((test_data.num_classes,)).to(device)
    class_total = torch.zeros((test_data.num_classes,)).to(device)
    if current_task is not None:
        task_true_positive = torch.zeros((len(tasks),)).to(device)
        task_total = torch.zeros((len(tasks),)).to(device)
    
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
                class_true_positive[class_index] += torch.sum(torch.logical_and(label==c, output_label==c))
                class_total[class_index] += torch.sum(label==c)

        if current_task is not None:
            for class_index, c in enumerate(classes):
                for task_idx, task in enumerate(tasks):
                    if c in task:
                        task_true_positive[task_idx] += class_true_positive[class_index]
                        task_total[task_idx] += class_total[class_index]
            
            past_tasks_true_positive = float('nan') if current_task == 0 else torch.sum(task_true_positive[:current_task])
            past_tasks_total = float('nan') if current_task == 0 else torch.sum(task_total[:current_task])
            current_task_true_positive = task_true_positive[current_task]
            current_task_total = task_total[current_task]
            next_tasks_true_positive = float('nan') if current_task == len(tasks) - 1 else torch.sum(task_true_positive[current_task + 1:])
            next_tasks_total = float('nan') if current_task == len(tasks) - 1 else torch.sum(task_total[current_task + 1:])

    # recover the initial train/eval mode
    if training: model.train()

    evaluation_results = dict(class_history = class_true_positive / class_total)

    if current_task is not None:
        evaluation_results.update(
            dict(
                task_history = task_true_positive / task_total,
                past_tasks_accuracy = past_tasks_true_positive / past_tasks_total,
                current_task_accuracy = current_task_true_positive / current_task_total,
                next_tasks_accuracy = next_tasks_true_positive / next_tasks_total
            )
        )

    # return the 1D tensor of accuracies of each class
    return evaluation_results

def get_plot(x, y, title="TITLE"):
    f = plt.figure()
    ax = f.add_subplot(111)
    marker = "x" if len(x) <= 20 else None
    ax.plot(x, y, marker=marker)
    
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("computed after training step #")
    ax.set_ylabel("accuracy")
    ax.set_title(title)

    return f

def task_incremental_plot(history, ticks, use_ticks=False):
    num_tasks = history.shape[1]
    macro_accuracy = history.mean(dim=1)
    
    f = plt.figure()
    ax = f.add_subplot(111)
    marker = "x" if len(ticks) <= 20 else None
    for h in history.transpose(0, 1):
        ax.plot(ticks, h, marker=marker)

    ax.set_ylim([-0.1, 1.1])
    if use_ticks: ax.set_xticks(ticks)
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

def get_memory_model_parameters(experiment,
    batch_size=None,
    shuffle=True,
    device="cpu",
    dataloader_args=None,
    **kwargs
):
    data = experiment.test_set
    
    model = experiment.model.to(torch.device(device))

    num_examples = len(data)

    batch_size = num_examples if batch_size == None else batch_size
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, **dataloader_args)
        
    # save the train/eval mode of the network and change it to eval mode
    training = model.training
    model.eval()
    
    def accumulate_in_dict(current, acc={}, reduce="none", accumulate="sum"):
        assert reduce in ["none", "sum"]
        assert accumulate in ["none", "sum"]

        if isinstance(current, torch.Tensor):
            if acc is None:
                if reduce == "none":
                    acc = current
                elif reduce == "sum":
                    acc = torch.sum(current, 0)
                    
            elif isinstance(acc, torch.Tensor):
                if reduce == "sum":
                    current = torch.sum(current, 0)

                if accumulate == "sum":
                    acc += current
                elif accumulate == "none":
                    acc = current
            else:
                raise Exception

        elif isinstance(current, dict) and isinstance(acc, dict):
            for key in current:
                if key not in acc:
                    acc[key] = {} if isinstance(current[key], dict) else None

                acc[key] = accumulate_in_dict(current[key], acc[key], reduce=reduce, accumulate=accumulate)

                if isinstance(current[key], torch.Tensor) and isinstance(acc[key], torch.Tensor):
                    acc[key + "_text"] = repr(acc[key])
        else:
            raise Exception

        return acc

    memory_model_parameters = {}
    with torch.no_grad():
        for img_mini_batch, label_mini_batch in dataloader:
            # send the mini-batch to the device memory
            img_mini_batch = img_mini_batch.to(device)
            label_mini_batch = label_mini_batch.to(device)

            # forward step
            # compute the output (actually the logist) of the model on the current example
            output = model.compute_extended_output(img_mini_batch)
            current_parameters = output["memory_model_parameters"]
            memory_model_parameters = accumulate_in_dict(current_parameters, memory_model_parameters, reduce="sum")

    # recover the initial train/eval mode
    if training: model.train()

    continual_model_parameters = model.get_continual_model_parameters()
    memory_model_parameters = accumulate_in_dict(continual_model_parameters, memory_model_parameters, accumulate="none")

    return dict(memory_model_parameters = memory_model_parameters)