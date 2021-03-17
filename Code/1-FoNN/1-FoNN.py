# Author:STEVEN
# -*- coding:UTF-8 -*-


import torch


def init_parameters(input_size, hidden_size, output_size):
    w1 = torch.randn(input_size, hidden_size)
    w2 = torch.randn(hidden_size, output_size)
    b1 = torch.randn(1, hidden_size)
    b2 = torch.randn(1, output_size)
    return {"w1": w1, "w2": w2, "b1": b1, "b2": b2}


def model(x, parameters):
    Z1 = x.mm(parameters["w1"]) + parameters["b1"]
    A1 = Z1.clamp(min=0)
    Z2 = A1.mm(parameters["w2"]) + parameters["b2"]
    cache = {"Z1": Z1, "A1": A1}
    return Z2, cache


def loss_fn(y_pred, y):
    loss = (y_pred - y).pow(2).sum()
    return loss


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


def update(lr, parameters, grads):
    parameters["w1"] -= lr * grads["d_w1"]
    parameters["w2"] -= lr * grads["d_w2"]
    parameters["b1"] -= lr * grads["d_b1"]
    parameters["b2"] -= lr * grads["d_b2"]
    return parameters


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
