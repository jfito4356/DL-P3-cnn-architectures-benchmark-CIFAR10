# 🧠 Comparative Analysis of Convolutional Architectures: The Impact of Normalization, Regularization, and GAP on CIFAR-10

---
**University of Guanajuato (DCEA)** **Department of Administrative Information Systems (LSIA)** **Course:** Deep Learning  
**Developed by:** Noé Israel Felipe Pérez  
---

This repository contains the experimental development and rigorous analysis of deep convolutional topographies evaluated on the **CIFAR-10** dataset. The primary objective is to evaluate stability, convergence speed, and generalization capabilities across multiple architecture variants, contrasting them against a traditional Multi-Layer Perceptron (MLP) baseline.

This project empirically validates how modern deep learning design paradigms —such as **Batch Normalization (BN)**, **Dropout**, **Global Average Pooling (GAP)**, and **Data Augmentation**— solve gradient degradation and overfitting bottlenecks when tackling computer vision tasks.

---

## 🏛️ Theoretical Framework & Introduction

As the depth of traditional linear architectures (*plain networks* or MLPs) increases for vision tasks, training accuracy quickly saturates and degrades due to the structural flattening of images, which destroys spatial relationships.

**Convolutional Neural Networks (CNNs)** overcome this mathematical obstacle by exploiting three fundamental properties:
1. **Spatial Locality:** Small structural filters (e.g., $3\times3$) efficiently detect highly correlated local patterns (edges, textures) regardless of their global position.
2. **Weight Sharing:** The same filters slide across the entire image space, drastically reducing the overall parameter count compared to fully connected layers.
3. **Hierarchical Representations:** Early layers abstract low-level primitives, while deeper layers progressively combine these signals to recognize complex geometric shapes and entire objects.

---

## 🧱 Architectural Topology & Model Complexity

To guarantee a scientifically sound and controlled environment, the performance of a baseline multi-layer perceptron, five convolutional network variants of incremental complexity, and a final optimized architecture were progressively evaluated.

### Block Specifications & Complexity Mapping

| Model | Structural Highlights & Processing Pipeline | Trainable Parameters |
| :--- | :--- | :---: |
| **Proposed MLP** | `Linear(3072 $\rightarrow$ 1024)` + `ReLU` + `Linear(1024 $\rightarrow$ 512)` + `ReLU` + `Linear(512 $\rightarrow$ 256)` + `ReLU` + `Linear(256 $\rightarrow$ 10)` | **3,809,034** |
| **CNN-1** | 1 Block: `Conv2d(3, 32)` + `ReLU` + `MaxPool2d` + Linear Classifier | **32** *(Base filters)* |
| **CNN-2** | 2 Sequential Conv Blocks (`32` $\rightarrow$ `64` channels) + `MaxPool2d` + `Linear` | **60,362** |
| **CNN-3** | 2 Blocks with Internal Double Convolution (`32` $\rightarrow$ `64` channels) + `Dropout(0.25)` | **106,538** |
| **CNN-4** | 3 Deep Blocks + Layer-wise **Batch Normalization** + Incremental `Dropout` | **667,178** |
| **CNN-5** | Cascaded Convolutional Blocks + **Global Average Pooling (GAP)** | **373,386** |
| **Efficient CNN** | Advanced Block Optimization + GAP + **Data Augmentation** (Best Model) | **373,386** |

*Note: Transitioning to Global Average Pooling (GAP) instead of flat `Flatten` dense layers allows a massive reduction in trainable parameters (as observed from CNN-4 to CNN-5), drastically mitigating overfitting risks while preserving mathematical expressiveness.*

---

## 📊 Dataset Specifications (CIFAR-10)

| Attribute | Specification |
| :--- | :--- |
| **Total Volume** | 60,000 color images |
| **Resolution & Format** | $32 \times 32$ pixels, RGB channels (3 color bands) |
| **Target Classes** | 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Experimental Splits** | 50,000 Training samples / 10,000 Test samples |
| **Standard Normalization** | Means: `(0.4914, 0.4822, 0.4465)` \| Stds: `(0.2470, 0.2435, 0.2616)` |

---

## 🧪 Experimental Protocols & Hyperparameters

Each network was subjected to strict training under uniform environment variables to isolate architectural impact and measure true generalization:

* **Optimizer:** AdamW (Variant with decoupled weight decay regularization).
* **Learning Rate (LR):** Initialized at `0.001` with controlled adaptive decay tracking.
* **Batch Size:** 64 / 128 optimized based on convolutional memory constraints.
* **Duration:** 30 Training Epochs with intermediate validation checkpoints.
* **Data Augmentation (Exclusive to Efficient CNN):** Runtime application of `RandomHorizontalFlip` and `RandomCrop(32, padding=4)` to force translation invariance.

### 📈 Test Performance Summary

Empirical results demonstrate the absolute superiority of convolutional designs over the dense MLP baseline and highlight how combining Batch Normalization and regularizers prevents optimization flatlining:

| Rank | Evaluated Model | Final Test Accuracy |
| :---: | :--- | :---: |
| **1st** | **Efficient CNN (GAP + Data Augmentation)** | **83.71%** |
| 2nd | CNN-4 (Deep Architecture with Batch Normalization) | 81.35% |
| 3rd | CNN-3 (Double Convolution per Block) | 80.08% |
| 4th | CNN-2 (Two-Block Structure) | 69.29% |
| 5th | CNN-5 (GAP Block Base without Augmentation) | 67.32% |
| 6th | CNN-1 (Control Baseline Block) | 64.38% |
| 7th | Proposed MLP (Dense Multi-Layer Perceptron) | 59.13% |

---

## 🔬 Visual Diagnostics: Feature Maps Exploration

The repository features an advanced diagnostics module utilizing **PyTorch Hooks** to intercept and extract internal hidden layer activations during the inference pass.

Through this visualization pipeline, the following phenomena are visually demonstrated:
- **Early Layers** react strongly to sharp local gradient shifts, directional edges, and basic contrasts.
- **Intermediate Layers** start structuring abstract geometric patterns and combinations of textures.
- **Final Layers (Pre-GAP)** condense information into global feature activation heatmaps, proving the internal interpretability of the learned concepts.

---

## 🛠️ Technology Stack

* **Language:** Python 3.10+
* **Core Framework:** PyTorch & Torchvision
* **Data Engineering:** NumPy, Pandas
* **Graphics & Diagnostics:** Matplotlib, Seaborn
* **Runtime Utility:** TQDM (Interactive logging tracking)
* **Hardware Acceleration:** CUDA Execution Environment for NVIDIA GPUs

---

## 🚀 Execution & Replication Guide

#### 1. Clone the Repository
```bash
git clone [https://github.com/jfito4356/DL-P3-cnn-architectures-benchmark-CIFAR10](https://github.com/jfito4356/DL-P3-cnn-architectures-benchmark-CIFAR10)
cd your-repo-name