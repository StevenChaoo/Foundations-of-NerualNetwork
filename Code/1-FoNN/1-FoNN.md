# Foundations of Nerual Network

> **Author: [StevenChaoo](https://github.com/StevenChaoo)**

## Contents

- [Foundations of Nerual Network](#foundations-of-nerual-network)
  - [Contents](#contents)
  - [Requirment](#requirment)
  - [Set Hyper-parameters](#set-hyper-parameters)
  - [Initialize parameters](#initialize-parameters)
  - [Build model](#build-model)
  - [Loss function](#loss-function)
  - [Back propogation](#back-propogation)
  - [Update parameters](#update-parameters)
  - [Summary of main function](#summary-of-main-function)

## Requirment

- torch >= 1.7.1
- numpy >= 1.19.5

## Set Hyper-parameters

Sample/batch size is 64. Number of neurons in the input layer is 1000. Number of neurons in the hidden layer is 100 and output size is 10.

Initialize randomly input and standard output x and y with function `torch.randn(int rows, int columns)`. 

Set learning rate 0.01 and this program will run 500 epochs.

```python
if __name__ == "__main__":
    M = 64
    input_size = 1000
    hidden_size = 100
    output_size = 10

    x = torch.randn(M, input_size)
    y = torch.randn(hidden, output_size)

    learning_rate = 0.01
    epoch = 500
```

## Initialize parameters

This nerual network has three layers: input layer, hidden layer and output layer. Therefore there are for parameters needed initialized: w1, b1 and w2, b2. This function is about initializing these four parameters randomly with function `torch.randn(int rows, int columns)` which returned a dictionary `{"w1": w1, "w2": w2, "b1": b1, "b2": b2}`.

```python
def init_parameters(input_size, hidden_size, output_size):
    w1 = torch.randn(input size, hidden_size)
    w2 = torch.randn(hidden_size, output_size)
    b1 = torch.randn(1, hidden_size)
    b2 = torch.randn(1, output_size)
    return {"w1": w1, "w2": w2, "b1": b1, "b2": b2}
```

## Build model

We need to calculate three vectors named respectively $ Z_1 $, $ A_1 $ and $ Z_2 $.

- $ Z_1 = xw_1+b_1 $
- $ A_1 = \sigma(Z_1) $
- $ Z_2 = A_1w_2+b_2 $.

We record $ Z_1 $ and $ A_1 $ in dictionary `cache` for updating and $ Z_2 $ for calculating loss. This function will return two parameters: $ Z_2 $ and `{"w1": w1, "w2": w2, "b1": b1, "b2": b2}`.

```python
def model(x, parameters):
    Z1 = x.mm(parameters["w1"]) + parameters["b1"]
    A1 = Z1.clamp(min=0)
    Z2 = A1.mm(parameters["w2"]) + parameters["b2"]
    cache = {"Z1": Z1, "A1": A1}
    return Z2, cache
```

## Loss function

We use Mean square error(MSE) to calculate loss. $ \mathrm{loss}=(\hat{y}-y)^2 $

```python
def loss_fn(y_pred, y):
    loss = (y_pred - y).pow(2).sum()
    return loss
```

## Back propogation

`torch.t()`: Transpose a two-dimensional matrix
`torch.sum(tensor input, dim = 0/1, bool keepdim)`: Compression a matrix. Dim equal to 0 means vertical compression and 1 means horizontal compression. Keepdim is used to choose if former dim wound been kept.

```python
def backpropogation(x, y, y_pred, cache, parameters):
    m = y.size()[0]
    d_y_pred = 1/m * (y_pred - y)
    d_w2 = 1/m * cache["A1"].t().mm(d_y_pred)
    d_b2 = 1/m * torch.sum(d_y_pred, 0, keepdim=True)
    d_A1 = d_y_pred.mm(parameters["w2"].t())

    d_Z1 = d_A1.clone()
    d_Z1[cache["Z1"] < 0] = 0

    d_w1 = 1/m * x.t().mm(d_Z1)
    d_b1 = 1/m * torch.sum(d_Z1, 0, keepdim=True)
    grads = {
        "d_w1": d_w1,
        "d_b1": d_b1,
        "d_w2": d_w2,
        "d_b2": d_b2
    }
    return grads
```

## Update parameters

Let current parameters minus the result of gradient of parameters times learning rate. Return new parameters.

```python
def update(lr, parameters, grads):
    parameters["w1"] -= lr * grads["d_w1"]
    parameters["w2"] -= lr * grads["d_w2"]
    parameters["b1"] -= lr * grads["d_b1"]
    parameters["b2"] -= lr * grads["d_b2"]
    return parameters
```

## Summary of main function

```python
if __name__ == "__main__":
    M, input_size, hidden_size, output_size = 64, 1000, 100, 10
    x = torch.randn(M, input_size)
    y = torch.randn(M, output_size)

    learning_rate = 1e-2
    EPOCH = 500

    parameters = init_parameters(input_size, hidden_size, output_size)

    for t in range(EPOCH):
        y_pred, cache = model(x, parameters)

        loss = loss_fn(y_pred, y)
        if (t+1) % 100 == 0:
            print(loss)
        grads = backpropogation(x, y, y_pred, cache, parameters)
        parameters = update(learning_rate, parameters, grads)
```