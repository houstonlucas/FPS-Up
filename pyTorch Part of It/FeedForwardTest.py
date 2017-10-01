from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time


def main():
    input_size = 2
    output_size = 1

    num_examples = 500
    num_train = num_examples / 2
    num_epochs = 100000

    x = np.random.rand(num_examples, input_size)
    y = toy_data(x, num_examples, output_size)

    x_train = x[:num_train]
    y_train = y[:num_train]

    x_test = x[num_train:]
    y_test = y[num_train:]

    net = Net()

    loss_fun = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

    # train
    for epoch in range(num_epochs):
        inputs = Variable(torch.from_numpy(x_train).float())
        targets = Variable(torch.from_numpy(y_train).float())

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fun(outputs, targets)
        loss.backward()
        optimizer.step()

        # Status Report
        if (epoch + 1) % 2000 == 0:
            print('Epoch [%d/%d]\n\tTraining Loss: %.4f'
                  % (epoch + 1, num_epochs, loss.data[0]))

            # Test data
            inputs = Variable(torch.from_numpy(x_test).float())
            outputs = net(inputs)
            targets = Variable(torch.from_numpy(y_test).float())
            loss = loss_fun(outputs, targets)
            print("\tTesting Loss: %.4f" % (loss.data[0]))

    # Visualize results

    # Actual data
    inputs = Variable(torch.from_numpy(x_train).float())
    outputs = net(inputs)
    targets = Variable(torch.from_numpy(y_train).float())
    loss = loss_fun(outputs, targets)
    print("Training Loss: %.4f" % (loss.data[0]))

    ax = prepare_scatter_plot(x_train, y_train)

    ax = prepare_scatter_plot(x_train, outputs.data.numpy(), '^', 'g', ax)

    # Train data
    inputs = Variable(torch.from_numpy(x_test).float())
    outputs = net(inputs)
    targets = Variable(torch.from_numpy(y_test).float())
    loss = loss_fun(outputs, targets)
    print("Testing Loss: %.4f" % (loss.data[0]))

    # print(type(outputs))
    prepare_scatter_plot(x_train, outputs.data.numpy(), 'x', 'r', ax)

    plt.show()


def toy_data(x, n, output_size):
    y = np.zeros([n, output_size])
    for i in range(n):
        a, b = x[i]
        y[i] = np.asarray([10.314 * a ** 2 + 15.52 * b ** 2])
    return y


def prepare_scatter_plot(input_values, output_values, mark='o', color='b', ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    x_graph = []
    y_graph = []
    z_graph = []
    num_examples = len(input_values)
    for index in range(num_examples):
        x_graph.append(input_values[index][0])
        y_graph.append(input_values[index][1])
        z_graph.append(output_values[index])

    ax.scatter(x_graph, y_graph, z_graph, c=color, marker=mark)

    return ax


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_size = 5

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h1 = self.fc1(x)
        a1 = F.sigmoid(h1)
        y = self.fc2(a1)
        return y


if __name__ == '__main__':
    main()
