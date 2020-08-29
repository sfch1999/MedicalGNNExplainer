import pickle
from Models import LatentNet, Net
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

with open('train_data.pickle', 'rb') as f:
    X_, y_, train_mask_, test_mask_, weight_ = pickle.load(f)  # Load the data
# print(train_mask_[:,1])

# number of epochs to train the model
n_epochs = 1000

for fold in range(10):
    print('\n\n\n-----------fold {} ------------\n'.format(fold))

    # initialize the NN
    # model = LatentNet(3).double()
    model = LatentNet(3).double()
    print(model)

    # specify loss function
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1)

    lr_scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.8, patience=50, verbose=True)

    model.train()  # prep model for training

    X = torch.tensor(X_[:, :, fold]).double()
    y = torch.tensor(np.argmax(y_[:, :, fold], axis=1)).long()
    train_mask = torch.tensor(train_mask_[:, fold] / np.mean(train_mask_[:, fold]))
    test_mask = torch.tensor(test_mask_[:, fold] / np.mean(test_mask_[:, fold]))
    weight = np.squeeze(weight_[:, fold])
    train_mask = train_mask * weight
    test_mask = test_mask * weight
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Start Training')

    for epoch in range(n_epochs):
        # monitor training loss
        ###################
        # train the model #
        ###################
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(X)
        y_hat = torch.argmax(output, dim=1)
        # if epoch%1==0:
        #     print('#####\n', output, '\n######\n')
        #     print(y_hat)
        # if epoch == 2:
        #     exit()

        train_correct = 0
        val_correct = 0

        for i in range(len(y_hat)):
            if train_mask_[i, fold] == 1 and y_hat[i] == y[i]:
                train_correct += 1
            elif test_mask_[i, fold] == 1 and y_hat[i] == y[i]:
                val_correct += 1

        # calculate the loss
        # print(output.shape)
        # print(y.shape)
        # exit()
        loss = criterion(output, y)
        train_loss = torch.mean(loss * train_mask)
        validation_loss = torch.mean(loss * test_mask)
        # backward pass: compute gradient of the loss with respect to model parameters
        train_loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        # lr_scheduler.step(train_loss)
        # update running training loss

        # print training statistics
        # calculate average loss over an epoch
        print(
            'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f} \tValidation '
            'Accuracy: {:.6f}'.format(
                epoch + 1,
                train_loss,
                validation_loss,
                train_correct / np.sum(train_mask_[:, fold]),
                val_correct / np.sum(test_mask_[:, fold])
            ))
