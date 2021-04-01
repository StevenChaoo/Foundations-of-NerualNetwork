# Long Short-Term Memory

> **Author: [StevenChaoo](https://github.com/StevenChaoo)**

![vscode](https://img.shields.io/badge/visual_studio_code-007acc?style=flat-square&logo=visual-studio-code&logoColor=ffffff)![neovim](https://img.shields.io/badge/Neovim-57a143?style=flat-square&logo=Neovim&logoColor=ffffff)![git](https://img.shields.io/badge/Git-f05032?style=flat-square&logo=git&logoColor=ffffff)
![python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=Python&logoColor=ffffff)![pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=ffffff)

This blog is written by **Neovim** and **Visual Studio Code**. You may need to clone this repository to your local and use **Visual Studio Code** to read. ***Markdown Preview Enhanced*** plugin is necessary as well. You can also read it [here](https://stevenchaoo.github.io/2021/03/30/Foundations-of-LSTM/).

**This blog is an interpretation of the official code of pytorch. There may be some errors. Welcome to communicate and correct.**

## Contents

- [Long Short-Term Memory](#long-short-term-memory)
  - [Contents](#contents)
  - [RNN](#rnn)
    - [Requirement](#requirement)
    - [Import Information](#import-information)
    - [Initilize RNN model](#initilize-rnn-model)
    - [Data Prepare Process](#data-prepare-process)
    - [Train](#train)
    - [Summary and Draw in Picture](#summary-and-draw-in-picture)
  - [LSTM](#lstm)
  - [GRU](#gru)

## RNN

In this demo, we will achieve a simple task: Using sin function to predict the cos function.

### Requirement

- torch >= 1.7.1
- numpy >= 1.19.5
- matplotlib >= 3.3.4

### Import Information

```python
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
```

### Initilize RNN model

The core code of RNN in PyTorch details that `torch.nn.RNN()` applies a multi-layer Elman RNN with $\tanh$ or $\mathrm{ReLU}$ non-linearity to an input sequence. For each element in the input sequence, each layer computes the following function:
$$h_t=\tanh(w_{ih}x_t+b_{ih}+w_{hh}h_{(t-1)}+b_{hh}) \tag{1}$$

where $h_t$ is the hidden state at time $t-1$ or the initial hidden state at time $o$. If **nonlinearity** is $\mathrm{ReLU}$, then $\mathrm{ReLU}$ is used instead of $\tanh$.

```python
class RNN(nn.Module):
    '''
    Build RNN model.

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
```

### Data Prepare Process

```python
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
```

### Train

```python
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
```

### Summary and Draw in Picture

```python
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
```

## LSTM

Only the model definition is different here, and the rest of the data processing and training are roughly the same. So only the model building is introduced.

`torch.nn.LSTM()` applies a multi-layer long short-term memory RNN to an input sequence. For each element in the input sequence, each layer computes the following function:
$$
\begin{align}
i_t&=\sigma(W_{ii}x_t+b_{ii}+W_{hi}h_{t-1}+b_{hi})\\
g_t&=\sigma(W_{if}x_t+b_{if}+W_{hf}h_{t-1}+b_{hf})\\
g_t&=\tanh(W_{ig}x_t+b_{ig}+W_{hg}h_{t-1}+b_{hg})\\
o_t&=\sigma(W_{io}x_t+b_{io}+W_{ho}h_{t-1}+b_{ho})\\
c_t&=f_t\bigodot c_{t-1}+i_t\bigodot g_t\\
h_t&=o_t\bigodot\tanh(c_t) \tag{2}
\end{align}
$$

where $h_t$ is the hidden state at time $t$, $c_t$ is the cell state at time $t$, $x_t$ is the input at time $t$, $h_{t-1}$ is the hidden state of the layer at time $t-1$ or the initial hidden state at time $o$, and $i_t$, $f_t$, $g_t$, $o_t$ are the input, forge, cell, and ouytput gates, respectively. $\sigma$ is the sigmoid function, and $\bigodot$ is the Hadamard product.

```python
class LSTM(nn.Module):
    '''
    Build LSTM model.

    Var:
        input_size:  The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers:  Number of recurrent layers
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    '''
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=50,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        print('lstm out size:')
        print(r_out.shape)
        out = self.out(r_out[:, -1, :])
        return out
```

## GRU

Only the model definition is different here, and the rest of the data processing and training are roughly the same. So only the model building is introduced.

`torch.nn.GRU()` applies a multi-layer gated recurrent unit RNN to an input sequence. For each element in the input sequence, each layer computes the following function:
$$
\begin{align}
r_t&=\sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr})\\
z_t&=\sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz})\\
n_t&=\tanh(W_{in}x_t+b_{in}+r_t\cdot(W_{hn}h_{t-1}+b_{hn}))\\
h_t&=(1-z_t)\cdot n_t+z_t\cdot h_{t-1} \tag{3}
\end{align}
$$

where $h_t$, $x_t$ is the hidden state and input at time $t$, $h_{t-1}$ is the hidden state of the layer at time $t-1$ or the initial hidden state at time $o$, and $r_t$, $z_t$, $n_t$ are the reset, update, and new gates, respectively. $\sigma$ is the sigmoid function, and $\cdot$ is the Hadamard product.

```python
class GRU(nn.Module):
    '''
    Build GRU model.

    Var:
        input_size:  The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    '''
    def __init__(self, output_num):
        super(GRU, self).__init__()
        self.hidden_size = 64
        self.gru = nn.GRU(
            input_size=50, 
            hidden_size=64, 
            batch_first=True
        )
        self.out = nn.Linear(64, output_num)
        self.hidden = None

    def forward(self, x):
        x, self.hidden = self.gru(x)
        x = self.out(x)
        return x, self.hidden
```
