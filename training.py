import numpy as np
import torch
import torch.nn as nn


#Loss function, optimizer, and training loop

def train(net, dataset, dataloader, epochs, lr):

    loss_fn = nn.BCEWithLogitsLoss()  # includes Sigmoid within itself
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.325, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, verbose=False, step_size=6, gamma=1/2
    )
    acc_train = []
    acc_val = []
    loss_train = []
    
    for epoch in range(epochs):
        epoch_loss = []
        corrects = 0
        net.train()
    
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_hat = net(x_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
    
            # Calculate training accuracy
            corrects += torch.sum(net.round_sigmoid(y_hat) == y_batch)
    
        acc_train.append(corrects / len(dataset))
    
        net.eval()
        y_hat_val = net(dataset.X_val)
        acc_val.append((torch.sum(net.round_sigmoid(y_hat_val) == dataset.Y_val) / len(dataset.Y_val)).item())
    
        val_loss = loss_fn(y_hat_val, dataset.Y_val)
        scheduler.step()
        loss_train.append(val_loss.item())
    
        print(
            "Epoch {}: \t Loss: {:.6f}   Train: {:.6f}   Valid: {:.6f}".format(
                epoch + 1, np.mean(epoch_loss), acc_train[-1], acc_val[-1]
            )
        )
    
    return acc_train, acc_val, loss_train
    

def test(net, dataset):
    net.eval()
    acc_test = torch.sum(net.round_sigmoid(net(dataset.X_test)) == dataset.Y_test) / len(dataset.Y_test)
    print("Test set accuracy: {:.6f}".format(acc_test))
    return acc_test.item()
