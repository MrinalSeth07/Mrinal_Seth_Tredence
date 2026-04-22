# 🔥 Gated Sparsity with L1 Regularization (CIFAR-10)

## 📌 Overview

This project implements **structured pruning using learnable gates with L1 regularization** on a feed-forward neural network (MLP) trained on CIFAR-10.

Each weight in the network is associated with a learnable gate, and sparsity is induced by penalizing these gates during training.

---

## 🧠 Key Idea

Each weight is modified as:

```
effective weight = weight × sigmoid(gate_score / temperature)
```

* Gates ∈ (0,1)
* L1 penalty pushes gates toward 0
* Small gate → weight effectively removed

---

## ⚙️ Methodology

### 1. Model

* MLP architecture:

  * 3072 → 1024 → 512 → 256 → 10
* Each linear layer replaced with **PrunableLinear**

---

### 2. Loss Function

Total loss:

```
L = CrossEntropy + λ × Σ(gates)
```

* CrossEntropy → classification
* L1 on gates → sparsity

---

### 3. Training Strategy

* Warm-up phase (no sparsity for first few epochs)
* λ scheduling to avoid early collapse
* Temperature scaling for sharper gating

---

### 4. Sparsity Measurement

Sparsity is computed using:

```
gate < 0.2 → pruned
```

This threshold ensures reliable estimation and avoids counting minor fluctuations as pruning.

---

## 📊 Results

| Lambda | Train Accuracy (%) | Test Accuracy (%) | Sparsity (%) |
| ------ | ------------------ | ----------------- | ------------ |
| 0      | ~55                | 54.77             | ~0           |
| 1e-5   | ~56                | **55.91**         | ~76          |
| 1e-4   | ~53                | 53.34             | ~92          |
| 5e-4   | ~52                | 52.77             | ~95          |

---

## 🔍 Key Observations

* Moderate λ improves generalization (regularization effect)
* High λ leads to over-pruning → accuracy drop
* Model shows strong redundancy (can remove >70% weights)
* Trade-off between sparsity and accuracy is clearly observed

---

## 📈 Visualizations

The repository includes:

* Gate distribution plot (`gate_distribution.png`)
* Lambda vs Accuracy (`lambda_analysis.png`)
* Pruning effect (`pruning_effect.png`)

---

## ⚠️ Important Notes

### Baseline Sparsity Fix

Gates are initialized to positive values to ensure:

```
λ = 0 → no sparsity
```

This avoids artificial sparsity caused by random initialization.

---

### Multiple Experiment Ranges

Experiments were conducted across:

* Low λ range → optimal trade-off
* High λ range → extreme pruning behavior

Some results (e.g., λ = 0.01) are from a separate notebook included in the repository.

---

## 🧠 Gate Distribution Insight

The gate distribution is highly skewed:

* Strong peak near 0 → pruned weights
* Smooth tail → active weights

This indicates **soft pruning**, not strictly bimodal behavior.

---

## 🚀 Future Improvements

* Post-pruning fine-tuning
* Hard thresholding during training
* Using CNNs for better redundancy
* Larger models for stronger sparsity patterns

---

## ▶️ How to Run

```bash
pip install torch torchvision matplotlib
python main.py
```

---

## 📌 Key Takeaway

> Moderate sparsity improves generalization, while excessive sparsity leads to underfitting.

---

## 📚 Reference

This work is inspired by:

**“Structure Optimization of Neural Networks with L1 Regularization on Gates”**

---

