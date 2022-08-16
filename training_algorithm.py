import torch

class Training_Algorithm:
    def __init__(self):
        pass

def batch_mode(model, data, epoch_number, device, optimizer="adam", lr=0.01, weight_decay=0):
    """Train the network using the specified options.
    The training is single-pass, and the evaluation is performed at the end of each class.
    Args:
        data: a dataset or dataloader containg the training data in the form (image, label).
        optimizer: "adam" or "sgd".
        lr: learning rate.
        weight_decay: weight multiplying the weight decay regularizaion term.
    """

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
    
    for epoch in range(0, epoch_number):
        n_examples = 0
        for img_mini_batch, label_mini_batch in data:
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

            # perform the backward step and the optimization step
            loss.backward()
            opt.step()
            opt.zero_grad()

    # recover the initial train/eval mode
    if not training: model.eval()