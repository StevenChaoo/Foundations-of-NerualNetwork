# Author:StevenChaoo
# -*- coding:UTF-8 -*-


import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class RNN(nn.Module):
    '''
    Build RNN model. Applies a multi-layer Elman RNN with tanh or ReLU

    Var:
        TIME_STEP:   Time step of training
        INPUT_SIZE:  Number of input layer, type is tensor
        INIT_LR:     Learning rate
        N_EPOCHS:    Rounds programs runs
        input_size:  The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers:  Number of recurrent layers
    '''

    def __init__(self):
        super(RNN, self).__init__()
        self.TIME_STEP = 10
        self.INPUT_SIZE = 1
        self.INIT_LR = 0.02
        self.N_EPOCHS = 100
        self.rnn = nn.RNN(
            input_size=self.INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h):
        '''
        Forward propagation

        Param:
            x: The number of input layer
            h: The number of hidden layer
        Return:
            prediction: The result of hidden layer as output
            h:          Hidden layer itself
        '''
        out, h = self.rnn(x, h)
        prediction = self.out(out)
        return prediction, h


def dataPrepare(step, TIME_STEP):
    '''
    Set data as numpy. Here are few funtions needed to explain.
    numpy.linspace() function is used to return evenly spaced numbers within the
    specified interval. The torch.from_numpy() method converts the array into a
    tensor, and the two share memory. If the tensor is modified such as
    reassignment, the original array will also be changed accordingly

    Param:
        step:      Current epoch
        TIME_STEP: Time step
    Return:
        x:     Input in tensor type
        y:     Output in tensor type
        y_np:  Output in numpy type
        steps: ALL steps
    '''
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP,
                        dtype=np.float32, endpoint=False)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[:, np.newaxis, np.newaxis])
    y = torch.from_numpy(y_np[:, np.newaxis, np.newaxis])
    return x, y, y_np, steps


def train(h_state, rnn):
    '''
    Train model

    Param:
        h_state: Hidden layer itself
    Return:
        prediction: The result of hidden layer as output
        h_state:    Current hidden layer being update
    '''
    prediction, h_state = rnn(x, h_state)
    h_state = h_state.detach()
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return prediction, h_state


def draw(y_np, steps, prediction):
    '''
    Show results

    Param:
        y_np:       Output in numpy
        steps:      All steps
        prediction: The result of hidden layer as output
    '''
    plt.cla()
    plt.plot(steps, y_np, 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.1)


if __name__ == "__main__":
    # Initialize RNN class
    rnn = RNN()

    # Set optimizer and loss function
    optimizer = torch.optim.Adam(rnn.parameters(), lr=rnn.INIT_LR)
    loss_func = nn.MSELoss()

    # Initialize hidden layer as none
    h_state = None

    # Initialize matplotlib board
    plt.figure()
    plt.ion()

    # Train and draw each epoch loss
    for step in range(rnn.N_EPOCHS):
        x, y, y_np, steps = dataPrepare(step, rnn.TIME_STEP)
        prediction, h_state = train(h_state, rnn)
        draw(y_np, steps, prediction)

    # Show final loss
    plt.ioff()
    plt.show()
