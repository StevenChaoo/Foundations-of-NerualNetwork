# Graph Neural Network

> **Author: [StevenChaoo](https://github.com/StevenChaoo)**

![vscode](https://img.shields.io/badge/visual_studio_code-007acc?style=flat-square&logo=visual-studio-code&logoColor=ffffff)![neovim](https://img.shields.io/badge/Neovim-57a143?style=flat-square&logo=Neovim&logoColor=ffffff)![git](https://img.shields.io/badge/Git-f05032?style=flat-square&logo=git&logoColor=ffffff)

This blog is written by **Neovim** and **Visual Studio Code**. You may need to clone this repository to your local and use **Visual Studio Code** to read. ***Markdown Preview Enhanced*** plugin is necessary as well. You can also read it [here](https://stevenchaoo.github.io/2021/04/10/GNN/).

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
    - [4.1 Fourier Transform](#41-fourier-transform)
      - [4.1.1 Signal and System](#411-signal-and-system)
      - [4.1.2 Fourier Series Representation](#412-fourier-series-representation)
      - [4.1.3 Spectral Graph Theory](#413-spectral-graph-theory)
    - [4.2 ChebNet](#42-chebnet)
    - [4.3 Graph Convolutional Network (GCN)](#43-graph-convolutional-network-gcn)
  - [5. Summary](#5-summary)
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

- **Semi-supervised node classification**: Stochastic Block Model dataset
- **Regression**: ZINC molecule graphs dataset
- **Graph classification**: SuperPixel MNIST and CIFAR10
- **Graph representation learning**
- **Link prediction**

### 2.2 Common dataset

- **CORA**: citation network. 2.7k nodes and 5.4k links
- **TU-MUTAG**: 188 molecules with 18 nodes on average

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

Main idea of spectral-based GNN is reagarding all features as special signal. The convolution kernel and the input are converted into a signal through Fourier transform, and then the signal is multiplied and added. Finally, it is converted into a graph through the inverted Fourier transform.

### 4.1 Fourier Transform

#### 4.1.1 Signal and System

In the signal and system, each signal can be regarded as a basis. Adding different weights and offsets to the components of basis can finally simulate the signal we need. **We call this synthesis**.
$$\vec{A}=\sum_{k=1}^Na_k\hat{v}_k \tag{1}$$

Conversely, we can calculate the weight and bias of this components of basis by taking the inner product of the signal and each component. **We call this analysis**.
$$a_j = \vec{A}\cdot\hat{v}_j \tag{2}$$

#### 4.1.2 Fourier Series Representation

We can expand it into fourier series as followed:
$$x(t)=\sum_{k=-\infty}^\infty a_ke^{jk\omega_0t}=\sum_{k=-\infty}^\infty a_k\phi_k(t) \tag{3}$$

For the same signal, we can expand it into different domains, usually we use the time domain and frequency domain. The same signal will also have different expansion methods. For example, we know the vector $v_i$ and give the offset $a_i$. But such vectors and offsets may not be easy to derive in the reverse direction, so we usually use another form of expansion. Using the following derivation we can expand the original with vector $u_i$ and offset $b_i$. This may make subsequent calculations easier.
$$
\begin{align}
\vec{A}&=\sum_ia_i\vec{v}_i\\
x(t)&=\int_{-\infty}^\infty x(\tau)\delta(t-\tau)d\tau\\
x(t)&=\frac{1}{2\pi}\int_{-\infty}^\infty {\color{red}X(j\omega)}e^{j\omega t}d\omega\\
&=\sum_k b_k\vec{u}_k \tag{4}\\\\
{\color{red}X(j\omega)}&=\int_{-\infty}^\infty x(t)e^{-j\omega t}dt
\end{align}
$$

#### 4.1.3 Spectral Graph Theory

First of all, let me show you some Symbolic expression.

- Graph: $G=(V,E),\ N=|V|$
- Adjacency Matrix: $A\in\mathbb{R}^{N\times N},\ A_{i,j}=0\ \mathbf{if}\ e_{i,j}\notin E,\ \mathbf{else}\ A_{i,j}=w(i,j)$
- Degree Matrix: $D\in\mathbb{R}^{N\times N},\ D_{i,j}=d(i)\ \mathbf{if}\ i=j,\ \mathbf{else}\ D_{i,j}=0$
- Signal on Graph: $f:V\to\mathbb{R}^N$
- Graph Laplacian: $L=D-A,\ L\succcurlyeq0$. $L$ is symmetric and all positive semidefinite for undirected graph
- $L=U\Lambda U^\intercal$
- $\Lambda=\mathrm{diag}(\lambda_0,\cdots,\lambda_{N-1})\in\mathbb{R}^{N\times N}$
- $U=[u_0,\cdots,u_{N-1}]\in\mathbb{R}^{N\times N}$, orthonormal
- $\lambda_l$ is the frequency, $u_l$ is the basis corresponding to $\lambda_l$

<div align="center">
  <image src="../Pics/43.jpeg" width="70%">
</div>

Assuming that we have a initial signal $f$ and its adjacency matrix $A$ and degree matrix $D$. We can calculate its Laplacian matrix $L$ and components of it $\Lambda$ and $U$
$$
\begin{align}
\left[\begin{array}{c:c:c}
f&D&A
\end{array}\right]&=
\left[\begin{array}{c:cccc:cccc}
4&2&0&0&0&0&1&1&0\\
2&0&3&0&0&1&0&1&1\\
4&0&0&2&0&1&1&0&0\\
-3&0&0&0&1&0&1&0&0
\end{array}\right]\\
&\downarrow\\
L&=
\left[\begin{array}{cccc}
2&-1&-1&0\\
-1&3&-1&-1\\
-1&-1&2&0\\
0&-1&0&1
\end{array}\right]\\
&\downarrow\\
\left[\begin{array}{c:c}
\Lambda&U
\end{array}\right]&=
\left[\begin{array}{cccc:cccc}
0&0&0&0&0.5&-0.41&0.71&-0.29\\
0&1&0&0&0.5&0&0&0.87\\
0&0&3&0&0.5&-0.41&-0.71&-0.29\\
0&0&0&4&0.5&0.82&0&-0.29
\end{array}\right]
\end{align} \tag{5}
$$

And thus we can get the following table:
$$
\begin{array}{|c|c|c|c|c|}
\hline
\lambda&0&1&3&4\\
\hline
u&\begin{bmatrix}
0.5\\0.5\\0.5\\0.5
\end{bmatrix}&\begin{bmatrix}
-0.41\\0\\-0.41\\0.82
\end{bmatrix}&\begin{bmatrix}
0.71\\0\\-0.71\\0
\end{bmatrix}&\begin{bmatrix}
-0.29\\0.87\\-0.29\\-0.29
\end{bmatrix}\\
\hline
\end{array}
$$

From the Fourier transform of the discrete time series, it can be seen that the greater the frequency, the greater the difference between two adjacent signals. Based on this theory, we can build graph fourier transform of signal $x:\hat{x}=U^\intercal x$. $x$ is signal in time domain, $\hat{x}$ is signal in frequence domain.

**What I need to remind is that everything we described before is to find a way to filter on the graph, and we need to define this way. We mapped the injury information of the vertex domain to the frequency domain, and found the intensity difference between different points on different frequencies.**

In frequency domain, we have another matrix called frequency responce matrix to transform $\hat{x}$ with $\hat{y}=g_\theta(\Lambda)\hat{x}$. Finally we got the signal in spectral domain.

Another thing we need to focus on is that how to reduce signal from spectral domain to vertex domain. The approche is multiple $U$.

<div align="center">
  <image src="../Pics/44.jpeg" width="70%">
</div>

All the model needs to do is to learn the value of $g_\theta(\Lambda)$, in other words it is to learn $L$.
$$y=g_\theta(U\Lambda U^\intercal)x=g_\theta(L)x \tag{6}$$

$g_\theta(\cdot)$ can be any function. For example:
$$g_\theta(L)=\log(I+L)=L=\frac{L^2}{2}+\frac{L^3}{3}-\cdots,\ \lambda_{\max}<1 \tag{7}$$

This method sounds feasible. Convert the input graph information to the frequency domain and then to the spectral domain for learning calculations, and then transfer the learned information back to the graph domain. But there is a problem, that is, **the number of learning parameters is related to the input size of the graph**. In addition, **the choice of $g_\theta(\cdot)$ will also affect the final result. After multiple iterations, $L^N$ will have a global impact, usually the $g_\theta(\cdot)$ function is not localize**.

### 4.2 ChebNet

Model ChebNet is a very quick model which can solve problem. It uses polynomial to parametrize $g_\theta(L)$. ChebNet stipulates that G must be a Laplace polynomial. By restricting the polynomial, the operation scale can be restricted to K-localize. At the same time, because the size of k is limited, the parameters to be learned are also limited to k. We only need to learn these parameters to achieve the goal.
$$
\begin{align}
g_\theta(L)&=\sum_{k=0}^K\theta_kL^k\\
g_\theta(\Lambda)&=\sum_{k=0}^K\theta_k\Lambda^k\\
y=Ug_\theta(\Lambda)U^\intercal x&=U(\sum_{k=0}^K\theta_k\Lambda^k)U^\intercal x
\end{align} \tag{8}
$$

From the above formula, we can see that this will cause another problem: its computational cost is very high. The time complexity is $N^2$. Its solution to this problem is using a polynomial function that can be computed recuresively from $L$ -- **Chebyshev Polynomial**.
$$
\begin{align}
T_0(x)&=1\\
T_1(x)&=x\\
T_k(x)&=2xT_{k-1}(x)-T_{k-2}(x),\ x\in[-1,1]
\end{align} \tag{9}
$$

Now, we change the Laplace function $L$ to $\tilde{L}$ after a certain adjustment (such as the following adjustment). Use it as the input of the Chebyshev polynomial. In this way, the parameter we want to learn changes from $g_\theta(\Lambda)=\sum_{k=0}^K\theta_k\Lambda^k\to g_{\theta^{'}}(\tilde{\Lambda})=\sum_{k=0}^K\theta_k^{'}T_k(\tilde{\Lambda})$. This will greatly reduce the time complexity of our operations.
$$
\begin{align}
T_0(\tilde{\Lambda})&=I\\
T_1(\tilde{\Lambda})&=\tilde{\Lambda}\\
T_k(\tilde{\Lambda})&=2\tilde{\Lambda}T_{k-1}(\tilde{\Lambda})-T_{k-2}(\tilde{\Lambda})\\
(\mathrm{where}\ \tilde{\Lambda}&=\frac{2\Lambda}{\lambda_{\max}}-I,\ \tilde{\Lambda}\in[-1,1])
\end{align} \tag{10}
$$

In this way, we rewrite the derivation formula of $y$ into the form of Chebyshev polynomials. Time complexity will drop to O(KE).
$$
\begin{align}
y&=g_{\theta^{'}}(L)x=\sum_{k=0}^K\theta_k^{'}T_k(\tilde{L})x\\
&=\theta_0^{'}{\color{red}T_0(\tilde{L})x}+\theta_1^{'}{\color{red}T_1(\tilde{L})x}+\cdots+\theta_K^{'}{\color{red}T_K(\tilde{L})x}\\
&\downarrow\\
&T_0(\tilde{L})x=x\\
&T_1(\tilde{L})x=\tilde{L}x\\
&T_k(\tilde{L})x=2\tilde{L}T_{k-1}(\tilde{L})x-T_{k-2}(\tilde{L})x\\
&\downarrow\\
&\bar{x}_0=x\\
&\bar{x}_1=\tilde{L}x\\
&\bar{x}_k=2\tilde{L}\bar{x}_{k-1}=2-\bar{x}_{k-2}\\
&\downarrow\\
y&=\theta_0^{'}{\color{red}\bar{x}_0}+\theta_1^{'}{\color{red}\bar{x}_1}+\cdots+\theta_K^{'}{\color{red}\bar{x}_K}\\
&={\color{red}\begin{bmatrix}
\bar{x}_0&\bar{x}_1&\cdots&\bar{x}_K
\end{bmatrix}}\begin{bmatrix}
\theta_0^{'}&\theta_1^{'}&\cdots&\theta_K^{'}
\end{bmatrix}^\intercal
\end{align} \tag{11}
$$

<div align="center">
  <image src="../Pics/45.jpeg" width="70%">
</div>

### 4.3 Graph Convolutional Network (GCN)

GCN is very similar to chebnet, and the specific mathematical derivation is as follows:
$$
\begin{align}
y=g_{\theta^{'}}(L)x&=\sum_{k=0}^K\theta_k^{'}T_k(\tilde{L})x\qquad let\ \ K=1\\
&=\theta_0^{'}x+\theta_1^{'}\tilde{L}x\\&\qquad{\color{red}\because}\ \tilde{L}=\frac{2L}{\lambda_{\max}}-I\\
&=\theta_0^{'}x+\theta_1^{'}(\frac{2L}{\lambda_{\max}}-I)x\\&\qquad{\color{red}\because}\ \lambda_{\max}\approx2\\
&=\theta_0^{'}x+\theta_1^{'}(L-I)x\\&\qquad{\color{red}\because}\ L=I-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\\
&=\theta_0^{'}x-\theta_1^{'}(D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x\\&\qquad{\color{red}\because}\ \theta=\theta_0^{'}=-\theta_1^{'}\\
&=\theta(I+D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x\\
&\downarrow\\
I_N+D^{-\frac{1}{2}}AD^{-\frac{1}{2}}&\to\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}\\
&\downarrow\\
H^{l+1}&=\sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^lW^l)\\
&\downarrow\\
h_v&=f(\frac{1}{|\mathcal{N(v)}|}\sum_{u\in\mathcal{N}(v)}\mathbf{W}x_u+b),\ \forall v\in \mathcal{V}
\end{align} \tag{12}
$$

The last formula means to multiply the hidden vector of this layer by a weight $\mathbf{W}$, and then add it to all its neighbors including himself to get the average, add a bias and pass a linear transformation to get the hidden vector of the next layer vector.

## 5. Summary

- GAT and GCN are the most popular GNNs
- Although GCN is mathematically driven, we tend to ignore its math
- GNN suffers from information lose while getting deeper
- Many deep learning models can be slightly modified and designed to fit graph data, such as Deep Graph InfoMax, Graph Transformer, GraphBert
- Throretical analysis must be dealt with in the future
- GNN can be applied to a variety of tasks

## REFERENCE

1. Related courses at [National Taiwan University](https://www.ntu.edu.tw) [Hung-yi Lee](https://speech.ee.ntu.edu.tw/~tlkagk/)