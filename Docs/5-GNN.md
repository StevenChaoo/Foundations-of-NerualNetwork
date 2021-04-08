# Graph Neural Network

> **Author: [StevenChaoo](https://github.com/StevenChaoo)**

![vscode](https://img.shields.io/badge/visual_studio_code-007acc?style=flat-square&logo=visual-studio-code&logoColor=ffffff)![neovim](https://img.shields.io/badge/Neovim-57a143?style=flat-square&logo=Neovim&logoColor=ffffff)![git](https://img.shields.io/badge/Git-f05032?style=flat-square&logo=git&logoColor=ffffff)

This blog is written by **Neovim** and **Visual Studio Code**. You may need to clone this repository to your local and use **Visual Studio Code** to read. ***Markdown Preview Enhanced*** plugin is necessary as well. You can also read it [here]().

## Contents

- [Graph Neural Network](#graph-neural-network)
  - [Contents](#contents)
  - [1. Task Defination](#1-task-defination)
  - [2. Dataset & Benchmark](#2-dataset--benchmark)
    - [2.1 Tasks](#21-tasks)
    - [2.2 Common dataset](#22-common-dataset)
  - [3. Spatial-based GNN](#3-spatial-based-gnn)
    - [3.1 What is spatial-based GNN?](#31-what-is-spatial-based-gnn)
    - [3.2 Models of spatial-based GNN](#32-models-of-spatial-based-gnn)
      - [3.2.1 Neural Networks for Graph (NN4G)](#321-neural-networks-for-graph-nn4g)
      - [3.2.2 Diffusion-Convolution Neural Network (DCNN)](#322-diffusion-convolution-neural-network-dcnn)
      - [3.2.3 Diffusion Graph Convolution (DGC)](#323-diffusion-graph-convolution-dgc)
      - [3.2.4 Mixture Model Networks (MoNET)](#324-mixture-model-networks-monet)
      - [3.2.5 Graph Sample and Aggregate (GrapgSAGE)](#325-graph-sample-and-aggregate-grapgsage)
      - [3.2.6 Graph Attention Networks (GAT)](#326-graph-attention-networks-gat)
      - [3.2.7 Graph Isomorphism Network (GIN)](#327-graph-isomorphism-network-gin)
  - [4. Spectral-based GNN](#4-spectral-based-gnn)
  - [5. Graph Generation](#5-graph-generation)
  - [6. Applications](#6-applications)
  - [REFERENCE](#reference)

## 1. Task Defination

In general, we know multilayer perceptron very well. Input a series of vectors, go through a series of hidden layers and output a series of vectors. We all know that and it really works. Beginning with FFNN, you may know CNN, RNN or even more fashion model Transformer.

But we won't discuss these neural network here. Different from that, graph is a kind of structure with defination on two vital elements: nodes and edges. Each node will store a lot of informations and a specific relation between every two nodes defines a relationship. And therefore, we want to find out all relations between nodes and not forget the information inside the node as well, graph is a necessary structure we need to use.

But here are the questions:

- How do we utilize the structures and relationship to help our model?
- What if the graph is larger, like 20k nodes?
- What if we don't have the all the labels?

The answer is **using neighbors information**.

<div align="center">
  <image src="../Pics/34.jpeg" width="70%">
</div>

Another question has arosed: how to embed node into a feature space using convolution? There are two solutions we could follow:

- Generalize the concept of convolution to graph $\to$ **Spatial-based convolution**
- Back to the definition of convolution in signal processing $\to$ **Spectral-based convolution**

<div align="center">
  <image src="../Pics/35.jpeg" width="70%">
</div>

## 2. Dataset & Benchmark

### 2.1 Tasks

- Semi-supervised node classification
- Regression
- Graph classification
- Graph representation learning
- Link prediction

### 2.2 Common dataset

- CORA: citation network. 2.7k nodes and 5.4k links
- TU-MUTAG: 188 molecules with 18 nodes on average

## 3. Spatial-based GNN

### 3.1 What is spatial-based GNN?

As for spatial-based GNN, we can regard it as a traditional convolutional neural network. There are two important concept we need to know.

- **Aggregate**: updating hidden state in current layer with all neighbor feature in previous layer.
- **Readout**: using one feature to represent the whole graph which collected features of all nodes.

<div align="center">
  <image src="../Pics/36.jpeg" width="90%">
</div>

Generated vector $h_G$ can be used on classification or prediction.

### 3.2 Models of spatial-based GNN

#### 3.2.1 Neural Networks for Graph (NN4G)

<div align="center">
  <image src="../Pics/37.jpeg" width="70%">
</div>

#### 3.2.2 Diffusion-Convolution Neural Network (DCNN)

<div align="center">
  <image src="../Pics/38.jpeg" width="70%">
</div>

#### 3.2.3 Diffusion Graph Convolution (DGC)

<div align="center">
  <image src="../Pics/39.jpeg" width="70%">
</div>

#### 3.2.4 Mixture Model Networks (MoNET)

Using this model we need to make this two concept:

- Define a measure on node distances
- Use weighted sum instead of simple summing up neighbor features

<div align="center">
  <image src="../Pics/40.jpeg" width="70%">
</div>

#### 3.2.5 Graph Sample and Aggregate (GrapgSAGE)

- Can work on both transductive and inductive setting
- GraphSAGE learns how to embed node features from neighbors
- Aggregation: mean, max-pooling, LSTM

<div align="center">
  <image src="../Pics/41.jpeg" width="70%">
</div>

#### 3.2.6 Graph Attention Networks (GAT)

- Input: node features $\mathbf{h}=\{h_1,h_2,\cdots,h_N\},h_i\in\mathbb{R}^F$
- Calculate energy: $e_{ij}=a(\mathbf{W}h_i,\mathbf{W}h_j)$
- Attention score: $\alpha_{ij}=\frac{\exp(\mathrm{LeakyReLU(a^\intercal[\mathbf{W}h_i\parallel\mathbf{W}h_j])})}{\sum_{k\in \mathcal{N}_i}\exp(\mathrm{LeakyReLU(a^\intercal[\mathbf{W}h_i\parallel\mathbf{W}h_k])})}$

<div align="center">
  <image src="../Pics/42.jpeg" width="70%">
</div>

#### 3.2.7 Graph Isomorphism Network (GIN)

- A GNN can be at most as powerful as WL isomorphic text
- Theoretical proofs were provided
- Updating method: $h_v^{(k)}=\mathrm{MLP}^{(k)}((1+\epsilon^{(k)}\cdot h_v^{(k-1)}+\sum_{u\in\mathcal{N}(v)}h_u^{(k-1)}$
- Sum instead of mean or max
- MLP instead of 1-layer

## 4. Spectral-based GNN

## 5. Graph Generation

## 6. Applications

## REFERENCE

1. Related courses at [National Taiwan University](https://www.ntu.edu.tw) [Hung-yi Lee](https://speech.ee.ntu.edu.tw/~tlkagk/)