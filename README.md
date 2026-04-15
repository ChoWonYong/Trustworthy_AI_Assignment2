# Assignment 2 — Differential Testing with DeepXplore (CIFAR-10)

**Course:** Reliable and Trustworthy Artificial Intelligence  
**Due:** April 22, 2026

---

## Overview

This project applies [DeepXplore](https://dl.acm.org/doi/10.1145/3132747.3132785) (Pei et al., SOSP 2017) to perform differential testing on two independently trained ResNet50 models on CIFAR-10. The goal is to automatically find inputs where the two models disagree, which indicates potential model failures.

---

## Project Structure

```
trustworthy2/
├── data/                        # CIFAR-10 dataset (auto-downloaded)
├── models/                      # Trained model checkpoints
│   ├── resnet50_cifar10_model1.pth
│   └── resnet50_cifar10_model2.pth
├── results/                     # Output visualizations from test.py
│   └── disagreement_XX_m1_<class>_m2_<class>.png
├── deepxplore/                  # Original DeepXplore repository (reference)
├── deepxplore_cifar.py          # DeepXplore adapted for CIFAR-10 + PyTorch
├── train_models.py              # Trains two ResNet50 models on CIFAR-10
├── test.py                      # Runs DeepXplore and saves results
├── requirements.txt
└── README.md
```

---

## Environment Setup

### 1. Create and activate conda environment

```bash
conda create -n trustworthy_ai_2 python=3.10
conda activate trustworthy_ai_2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** A CUDA-capable GPU is strongly recommended. The code automatically detects and uses CUDA if available; it will fall back to CPU otherwise (significantly slower for training).

---

## Running the Code

### Step 1 — Train the two ResNet50 models

```bash
python train_models.py
```

This trains two ResNet50 models on CIFAR-10 using different optimizers and random seeds, then saves them to `models/`:

| Model | Optimizer | LR    | Seed |
|-------|-----------|-------|------|
| 1     | SGD + momentum + cosine LR | 0.01 | 42  |
| 2     | Adam + step decay          | 0.001 | 123 |

Training takes approximately 20–40 minutes per model on a GPU. Pre-trained checkpoints can be reused if already present.

### Step 2 — Run DeepXplore differential testing

```bash
python test.py
```

This samples seed images from the CIFAR-10 test set, runs gradient-ascent-based differential testing, and saves results to `results/`.

**Key hyperparameters** (configurable at the top of `test.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_SEEDS` | 100 | Number of seed images sampled from CIFAR-10 test set |
| `GRAD_ITERATIONS` | 100 | Max gradient ascent iterations per seed |
| `TRANSFORMATION` | `'light'` | Perturbation type: `light`, `occl`, or `blackout` |
| `WEIGHT_DIFF` | 2.0 | Weight for the differential-behavior loss term |
| `WEIGHT_NC` | 0.1 | Weight for the neuron-coverage loss term |
| `STEP` | 1.0 | Gradient ascent step size |
| `THRESHOLD` | 0.5 | Neuron activation threshold for coverage tracking |

**Output:**
- `results/disagreement_XX_m1_<class>_m2_<class>.png` — individual disagreement images

---

## Modifications to DeepXplore

The original DeepXplore (Pei et al., 2017) targets Keras/TensorFlow models on ImageNet-scale datasets. The following adaptations were made for this assignment:

1. **Framework:** Reimplemented in PyTorch using `register_forward_hook` for neuron coverage tracking instead of Keras layer outputs.

2. **Dataset & models:** CIFAR-10 (10 classes, 32×32 images resized to 224×224) with two ResNet50 models instead of the original VGG16/VGG19/ResNet50 on ImageNet.

3. **Neuron coverage scope:** Coverage is tracked at the channel level over ResNet50's four main layer blocks (`layer1`–`layer4`, ~3840 channels total) rather than individual neurons, to avoid hooking shared inplace-ReLU modules.

4. **Gradient ascent:** Implemented via PyTorch autograd (`requires_grad_(True)` + `.backward()`). Gradient is normalized by its L2 norm before applying the step, following the original paper.

5. **Transformation constraints:** All three transformations from the original paper are implemented:
   - `light` — global brightness shift (uniform gradient scaled by mean)
   - `occl` — rectangular occlusion patch
   - `blackout` — random small black patch

---

## Expected Results

After running `test.py`, the `results/` directory will contain PNG visualizations of inputs where the two models produce different predictions. The summary printed to stdout shows:

- Total number of disagreement-inducing inputs found
- Neuron coverage achieved for each model (fraction of tracked channels activated)
- Breakdown of which class pairs caused disagreements
