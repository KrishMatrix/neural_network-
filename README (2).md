# 🧠 MNIST & Neural Networks — From Perceptron to Deep Learning

> Exploring neural network fundamentals — starting with a single-layer **Perceptron** on synthetic data, then building up to MNIST digit classification.  
> Part of my hands-on learning journey through classical ML → neural networks.

---

## 📌 What This Project Does

This project walks through the progression from the simplest neural unit (a Perceptron) to a full neural network for handwritten digit recognition:

1. **`visualing_minst.ipynb`** — Perceptron classifier on synthetic binary classification data (the "hello world" of neural networks).
2. **`minst_nn.ipynb`** — Neural network for MNIST digit classification *(coming soon)*.

---

## 📓 Notebook 1: `visualing_minst.ipynb` — Perceptron Classifier

### What's a Perceptron?

A Perceptron is the **simplest possible neural network** — a single neuron with no hidden layers. It learns a linear decision boundary by iteratively adjusting weights based on misclassified examples. It's the building block that eventually led to deep learning.

```
Inputs (x₁, x₂, ... x₂₀)
       │
       ▼
  ┌──────────────┐
  │  Σ (wᵢxᵢ + b)│ ──► step function ──► 0 or 1
  └──────────────┘
       Perceptron
```

### Step-by-step Walkthrough

| Step | Code | What's Happening |
|---|---|---|
| **1. Generate data** | `make_classification(n_samples=1000, n_features=20, n_classes=2)` | Creates a synthetic dataset: 1000 samples, 20 features, 2 classes |
| **2. Train/test split** | `train_test_split(test_size=0.2)` | 800 train / 200 test samples |
| **3. Create Perceptron** | `Perceptron(max_iter=1000, eta0=0.1, tol=1e-3, shuffle=True)` | Single-layer linear classifier |
| **4. Train** | `clf.fit(X_train, y_train)` | Adjusts weights over up to 1000 epochs |
| **5. Evaluate** | `clf.score(X_test, y_test)` | **Accuracy: 80.5%** |

### Hyperparameters Explained

| Parameter | Value | What It Does |
|---|---|---|
| `max_iter=1000` | Max training epochs — stops early if converged |
| `eta0=0.1` | Learning rate — how big each weight update step is |
| `tol=1e-3` | Convergence tolerance — stops if loss improvement < 0.001 |
| `shuffle=True` | Shuffles training data each epoch to avoid order bias |
| `random_state=42` | Reproducible results |

### Result

```
Accuracy: 0.805 (80.5%)
```

**Why only 80.5%?** A Perceptron can only learn **linearly separable** boundaries. If the data has any non-linear patterns (which `make_classification` often generates), the Perceptron will struggle. This is exactly why we need multi-layer networks (MLPs) — which is where `minst_nn.ipynb` comes in.

---

## 📓 Notebook 2: `minst_nn.ipynb` — MNIST Neural Network

> ⚠️ **Placeholder** — notebook file was empty on upload. Will be documented once re-uploaded.

This notebook will cover building a neural network for the MNIST handwritten digit classification task (0–9, 28×28 pixel grayscale images, 70,000 samples).

---

## 🧪 The Learning Progression

```
Perceptron (single neuron, linear)
    │
    │  "80.5% accuracy — can't learn non-linear patterns"
    │
    ▼
Multi-Layer Perceptron / Neural Network (hidden layers, non-linear activations)
    │
    │  "Can learn complex patterns like handwritten digits"
    │
    ▼
MNIST digit classification (10 classes, image data)
```

### Why This Order Matters

Understanding a Perceptron first makes neural networks intuitive:

- A **Perceptron** = 1 neuron, linear decision boundary, simple weight updates
- An **MLP** = many Perceptrons stacked in layers + non-linear activation functions (ReLU, sigmoid)
- **Backpropagation** = the chain rule applied to update weights across all layers (what makes deep learning "deep")

The jump from 80.5% Perceptron accuracy to 97%+ neural network accuracy on real image data demonstrates *why* depth and non-linearity matter.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

### Run

```bash
jupyter notebook visualing_minst.ipynb
# Run all cells → see Perceptron accuracy on synthetic data

jupyter notebook minst_nn.ipynb
# Run all cells → MNIST digit classification (coming soon)
```

---

## 🗂️ Project Structure

```
mnist-neural-networks/
│
├── visualing_minst.ipynb    # Perceptron on synthetic binary classification
├── minst_nn.ipynb           # Neural network for MNIST digits (WIP)
│
└── README.md
```

---

## 📝 Notes & Learnings

- **Perceptron limitations are the whole point.** Seeing it cap at ~80% on non-linear data motivates why hidden layers and activation functions exist.
- **`make_classification` is great for quick experiments.** No file downloads needed — generates clean synthetic data with controlled difficulty.
- **`eta0` (learning rate) matters.** Too high → overshoots and oscillates. Too low → takes forever to converge. 0.1 is a reasonable starting point.
- **`shuffle=True` prevents order dependence.** Without shuffling, the Perceptron can get stuck cycling through the same misclassification patterns.
- **Perceptron convergence theorem:** If data IS linearly separable, the Perceptron is guaranteed to find a solution. If it's NOT separable, it will never converge — it just hits `max_iter` and stops.

---

## 🔮 Next Steps

- [ ] Complete and upload `minst_nn.ipynb` with MNIST classification
- [ ] Visualize MNIST digits with `matplotlib`
- [ ] Compare Perceptron vs MLP vs CNN accuracy on MNIST
- [ ] Experiment with different learning rates and plot convergence curves
- [ ] Try `MLPClassifier` from sklearn as an intermediate step before full PyTorch/TensorFlow

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13.7 | Runtime |
| scikit-learn | Perceptron, data generation, train/test split |
| NumPy | Numerical operations |
| Jupyter | Interactive development |
