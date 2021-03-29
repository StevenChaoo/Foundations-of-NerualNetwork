# Long Short-Term Memory

> **Author: [StevenChaoo](https://github.com/StevenChaoo)**

![vscode](https://img.shields.io/badge/visual_studio_code-007acc?style=flat-square&logo=visual-studio-code&logoColor=ffffff)![neovim](https://img.shields.io/badge/Neovim-57a143?style=flat-square&logo=Neovim&logoColor=ffffff)![git](https://img.shields.io/badge/Git-f05032?style=flat-square&logo=git&logoColor=ffffff)

This blog is written by **Neovim** and **Visual Studio Code**. You may need to clone this repository to your local and use **Visual Studio Code** to read. ***Markdown Preview Enhanced*** plugin is necessary as well. You can also read it [here](https://stevenchaoo.github.io/2021/03/30/Foundations-of-LSTM).

## Contents

- [Long Short-Term Memory](#long-short-term-memory)
  - [Contents](#contents)
  - [1. Slot Filling Task](#1-slot-filling-task)
    - [1.1 Task Definition](#11-task-definition)
    - [1.2 Feedforward Neural Network](#12-feedforward-neural-network)
      - [1.2.1 Input](#121-input)
      - [1.2.2 Output](#122-output)
      - [1.2.3 Drawbacks](#123-drawbacks)
  - [2. Recurrent Neural Network](#2-recurrent-neural-network)
    - [2.1 A quick demo for RNN](#21-a-quick-demo-for-rnn)
    - [2.2 Other RNN models](#22-other-rnn-models)
      - [2.2.1 Elman Network & Jordan Network](#221-elman-network--jordan-network)
      - [2.2.2 Bidirectional RNN](#222-bidirectional-rnn)
  - [3. Long Short-Term Memory](#3-long-short-term-memory)
  - [REFERENCE](#reference)

## 1. Slot Filling Task

### 1.1 Task Definition

What is slot? In this sentence: *I would like to arrive Taipei on November 2nd*. We got two important informations--**destination** and **time of arrival**. We call this two functional position **slot**. The next thing is obvious, if there are no words in these two positions, then we need to fill these two slots. That is **Slot Filling Task**.

Through the example above, we know that **Taipei** should be filled into the slot of the **destination** and **November 2nd** should be filled into **time of arrival**.

### 1.2 Feedforward Neural Network

First of all, we can easily think of using feedforward neural network to accomplish this task.

![12](../Pics/12.jpeg)

Network above describes the process of feedforward neural network. Program will go through **input train** and **output**.

#### 1.2.1 Input

At the very beginning, we need to input a series of vectors. There are lots of embedding ways to convert a integer to a vector. In this task, each word is represented as a vector.

One of the most common is **One-Hot encoding**. The vector is lexicon size. Each dimension corresponds to a word in the lexicon. The dimension for the word is 1, and others are 0. If the word is a new one for this dictionary, put it into **other** dimension.
$$\mathrm{lexicon}=\{apple, bag, cat, dog, elephant, other\}\\
\downarrow\\
\begin{bmatrix}
apple\\bag\\cat\\dog\\elephant\\other
\end{bmatrix}=
\begin{bmatrix}
1&\cdots&0\\
\vdots&\ddots&\vdots\\
0&\cdots&1
\end{bmatrix} \tag{1}$$

#### 1.2.2 Output

We expect we can recieve a set of probability distribution that the input word belonging to the slots so that we can determine which words should we put in the slot.

#### 1.2.3 Drawbacks

FFNN has a fatal error that this **model can not recognize elements beyond the slot sets**.

For example, two sentences has completly meaning but FFNN may consider same words thould be filled in the same slot.

<div><center>... <b>arrive</b> Taipei on November 2nd ...</center>
<center>... <b>leave</b> Taipei on November 2nd ...</center></div>

And thus, we hope our FFNN has memory. It would know the previous word before it see the current word.

**We call FFNN with memory a recurrent neural network.**

## 2. Recurrent Neural Network

### 2.1 A quick demo for RNN

The main idea of RNN is that the ouput of hidden layer are stored in the memory. And memory can be considered as another input.

For ease of understanding, we consider that all the weights are 1 and there is no bias in this network.

![13](../Pics/13.jpeg)

Initialize $a$ is 0, our **input sequence** is:
$$\left[
    \begin{matrix}
    1 & 1 & 2 \\
    1 & 1 & 2
\end{matrix}\ \ \ \cdots
\right] \tag{2}$$

- EPOCH 1
  - STEP 1: ![14](../Pics/14.jpeg)
  - STEP 2: ![15](../Pics/15.jpeg)
  - STEP 3: ![16](../Pics/16.jpeg)
- EPOCH 2
  - STEP 1: ![17](../Pics/17.jpeg)
  - STEP 2: ![18](../Pics/18.jpeg)
  - STEP 3: ![19](../Pics/19.jpeg)
  - STEP 4: ![20](../Pics/20.jpeg)

After multiple rounds of training, we get the **output sequence** as:
$$\left[
    \begin{matrix}
    4 & 12 & 32 \\
    4 & 12 & 32
\end{matrix}\ \ \ \cdots
\right] \tag{3}$$

There is a very important thing we need to know that **changing the sequence order will change the output. Even same input will get different output under different memory.**

Going back to the previous example, through RNN, we may be able to solve the above problems because the values stored in the memory is different so that we may get different output.

### 2.2 Other RNN models

#### 2.2.1 Elman Network & Jordan Network

<div align="center">
  <image src="../Pics/21.jpeg" width=70%>
</div>

#### 2.2.2 Bidirectional RNN

<div align="center">
  <image src="../Pics/22.jpeg" width=70%>
</div>

## 3. Long Short-Term Memory

## REFERENCE

1. Related courses at [National Taiwan University](https://www.ntu.edu.tw) [Hung-yi Lee](https://speech.ee.ntu.edu.tw/~tlkagk/)
