# 🧠 YZ50 — AI Research Journey

My implementations, experiments, and technical notes throughout the **YZ50 — Türkiye'nin Yapay Zeka Araştırmacıları** program.

This repository documents my journey from the fundamental building blocks of neural networks to modern language models and GPT-style architectures, following a **from-scratch and first-principles approach inspired by Andrej Karpathy's "Zero to Hero" philosophy**.

Rather than treating AI systems as black boxes, the goal is to understand what happens underneath — how neural networks represent information, how gradients flow through computation graphs, how models learn from data, and how modern language models are built.

---

## 🚀 About YZ50

**YZ50** is an intensive AI research program focused on developing the next generation of AI researchers in Türkiye.

The program follows a hands-on, implementation-oriented learning path. Concepts are not only studied theoretically but also explored by **building core components from scratch**, experimenting with them, and gradually moving toward increasingly sophisticated architectures.

Following the **"Zero to Hero" approach popularized by Andrej Karpathy**, the learning journey starts with a single neuron and progressively builds toward modern language models:

**Neural Networks → Backpropagation → Language Models → MLPs → Deep Networks → WaveNet → Transformers → GPT**
---

## 📚 Learning Journey

The repository follows the progression of the YZ50 curriculum.

### 🧠 Neural Network Foundations

The journey begins with the fundamental components of neural networks:

- Neurons and layers
- Parameters and weights
- Biases and activation functions
- Forward propagation
- Loss functions
- Numerical derivatives
- Gradient descent
- Optimization landscapes
- Model training

The first implementations are intentionally built with basic Python data structures to make the underlying mathematics and computation as transparent as possible.

---

### 🔄 Backpropagation & Automatic Differentiation

The next step is understanding **how neural networks actually learn**.

Topics include:

- Computational graphs
- Chain rule
- Local gradients
- Backpropagation
- Numerical vs. analytical derivatives
- Topological sorting
- Automatic differentiation
- Building a minimal autograd engine

The goal is to move beyond simply using frameworks such as PyTorch and understand the mechanisms that make automatic differentiation possible.

---

### 🔤 Language Modeling

The program then introduces the fundamental ideas behind language models.

Topics explored include:

- Character-level language models
- Bigram models
- Token representations
- Embeddings
- Context windows
- Probability distributions
- Cross-entropy loss
- Training language models

These concepts provide the foundation for understanding how neural networks can learn to predict sequences of tokens.

---

### 🏗️ MLPs & Deep Neural Networks

The next stage focuses on scaling the ideas from individual neurons into deeper neural architectures.

Topics include:

- Multilayer perceptrons
- Hidden layers
- Embedding representations
- Activation functions
- Parameter initialization
- Training dynamics
- Gradient statistics
- Activation statistics
- Batch normalization
- Deep network optimization

The emphasis is on understanding not only how to train deeper models, but also **why certain initialization and normalization techniques are necessary**.

---

### 🌊 WaveNet

The learning path then moves toward more advanced sequence modeling through **WaveNet-style architectures**.

This stage explores:

- Context-based prediction
- Embedding layers
- Dilated convolutions
- Hierarchical receptive fields
- Deep sequential architectures
- Language modeling with increasingly large contexts

This provides a bridge between simpler MLP-based language models and the architectures used in modern generative AI.

---

### ⚡ Transformers

The next major step is understanding the architecture behind modern language models.

Topics include:

- Self-attention
- Query, Key, and Value representations
- Attention scores
- Multi-head attention
- Positional information
- Causal masking
- Transformer blocks
- Residual connections
- Layer normalization

The goal is to understand the Transformer not as a black-box architecture, but as a collection of understandable mathematical operations.

---

### 🤖 GPT

The final stages bring the previous concepts together into a GPT-style language model.

Topics include:

- Token embeddings
- Positional embeddings
- Causal self-attention
- Transformer blocks
- Feed-forward networks
- Language-model objectives
- Training and optimization
- Text generation
- Model evaluation
- Experimentation

The culmination of the learning path is understanding and building a **small GPT-style model from scratch**.

---

## 🛠️ Repository Structure

The repository is organized chronologically according to the weekly progression of the program.

Each week's work is kept in a separate Jupyter Notebook following the `w1_`, `w2_`, `w3_`, ... naming convention.

The notebooks contain a combination of:

- 📖 Technical explanations
- 🧮 Mathematical derivations
- 💻 From-scratch implementations
- 📊 Experiments and visualizations
- 🧪 Model training experiments
- 📝 Personal learning notes

This structure allows the repository to serve both as a record of my progress through YZ50 and as a reference for revisiting the concepts later.

---

## 🎯 Learning Philosophy

The central idea behind this repository is:

> **Don't just use the model. Understand the model.**

Instead of immediately relying on high-level abstractions, the learning process starts from simple mathematical operations and gradually builds toward complex architectures.

For example:

```text
Neuron
   ↓
Layer
   ↓
Neural Network
   ↓
Backpropagation
   ↓
Autograd
   ↓
Language Model
   ↓
MLP
   ↓
Deep Network
   ↓
WaveNet
   ↓
Attention
   ↓
Transformer
   ↓
GPT