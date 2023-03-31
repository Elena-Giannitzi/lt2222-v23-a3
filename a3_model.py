import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.nn.functional as F

# You can implement classes and helper functions here too.

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

##end of class 

def model_training(train_set, test_set, epochs, batch_size, output_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = MLP(input_size=66, output_size=output_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = torch.argmax(targets, dim=1)  # convert targets to class indices
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        with torch.no_grad():
            model.eval()
            test_loss = 0.0
            correct = 0

            for inputs, targets in test_loader:
                outputs = model(inputs)
                targets = torch.argmax(targets, dim=1)  # convert targets to class indices
                loss = loss_fn(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()

            test_loss /= len(test_loader)
            test_accuracy = correct / len(test_set)

        print(f'Epoch {epoch}: train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | test_accuracy={test_accuracy:.4f}')

    return model

# model = model_training(train_set, test_set, epochs=4, batch_size=10, output_size=14)

#creating the confusion matrix
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score

def confusion_matrix_printed(model, test, train):
    model.eval()
    pred = model(test).argmax(dim=1)
    cm = confusion_matrix(test_labels.argmax(axis=1), pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlabel='Predicted label',
           ylabel='True label',
           title='Confusion matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    
    # save the figure as a PNG file
    fig.savefig('confusion_matrix.png')
    
    # display the balanced accuracy and weighted F1 score
    bal_acc = balanced_accuracy_score(test_labels.argmax(axis=1), pred)
    f1 = f1_score(test_labels.argmax(axis=1), pred, average='weighted')
    print("Balanced accuracy:", bal_acc)
    print("Weighted F1 score:", f1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefiletrain", type=str, help="The file containing the table of instances and features training.")
    parser.add_argument("featurefiletest", type=str, help="The file containing the table of instances and features test.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()
    train_data = pd.read_csv(args.featurefiletrain)
    test_data = pd.read_csv(args.featurefiletest)
    encoder = preprocessing.LabelEncoder()
    label_train = encoder.fit_transform(train_data.label.astype(str).fillna('NA'))
    label_test = encoder.transform(test_data.label.astype(str).fillna('NA'))
    num_classes = 14

    # Encode labels as one-hot tensors
    label_train = torch.nn.functional.one_hot(torch.Tensor(label_train).to(torch.int64), num_classes=num_classes)
    label_test = torch.nn.functional.one_hot(torch.Tensor(label_test).to(torch.int64), num_classes=num_classes)

    # create TensorDatasets
    train_set = torch.utils.data.TensorDataset(torch.Tensor(train_data.iloc[:,:-1].values), label_train)
    test_set = torch.utils.data.TensorDataset(torch.Tensor(test_data.iloc[:,:-1].values), label_test)

    print("Reading {}&{}...".format(args.featurefiletest,args.featurefiletrain))
    
    model = model_training(train_set, test_set, epochs=4, batch_size=10, output_size=14)
    test_vectors, test_labels = test_set.tensors
    confusion_matrix_printed(model, test_vectors, test_labels)

    # implement everything you need here
