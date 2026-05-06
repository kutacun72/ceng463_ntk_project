# CENG463 Term Project: Neural Tangent Kernel vs Finite Networks

This repository contains the implementation for the CENG463 Introduction to Machine Learning term project.

## Project Topic

**Project 12: Neural Tangent Kernel vs Finite Networks**

The goal of this project is to empirically compare classical machine learning baselines, finite-width neural networks, and Neural Tangent Kernel-based approaches on a supervised classification task.

At the current progress report stage, the implemented experiments focus on:

- Dataset preparation
- Baseline model comparison
- Finite MLP implementation
- MLP width ablation
- Initial error analysis using confusion matrices

The Neural Tangent Kernel component will be further developed in the next phase of the project.

## Dataset

The project uses the **Fashion-MNIST** dataset.

Fashion-MNIST contains 28x28 grayscale images from 10 clothing categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

For the current experiments, a smaller subset is used for faster development:

- Training subset: 10,000 samples
- Test subset: 2,000 samples

The dataset is automatically downloaded through `torchvision`.

## Repository Structure

```text
ceng463_ntk_project/
│
├── data/                         # Automatically downloaded dataset, ignored by Git
├── figures/                      # Generated plots and confusion matrices
├── results/                      # CSV result files
├── src/
│   ├── data_loader.py             # Fashion-MNIST loading and preprocessing
│   ├── baselines.py               # Logistic Regression, Linear SVM, RBF SVM
│   ├── mlp_train.py               # Finite MLP training
│   ├── mlp_width_ablation.py      # MLP hidden-size ablation study
│   ├── combine_results.py         # Combines result CSV files
│   ├── plot_results.py            # Generates result plots
│   └── confusion_matrices.py      # Generates confusion matrices
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Implemented Models

The following models have been implemented so far:

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline |
| Linear SVM | Linear margin-based baseline |
| RBF SVM | Nonlinear kernel-based baseline |
| MLP | Finite-width neural network |

## Initial Results

Main model comparison on the Fashion-MNIST subset:

| Model | Accuracy | Macro-F1 | Runtime |
|---|---:|---:|---:|
| Logistic Regression | 0.8145 | 0.8152 | 10.54 s |
| Linear SVM | 0.7990 | 0.7981 | 32.96 s |
| RBF SVM | 0.8665 | 0.8664 | 14.99 s |
| MLP hidden=256 | 0.8475 | 0.8491 | 25.37 s |

The RBF SVM currently gives the best performance among the implemented models. The finite MLP model performs better than the linear baselines but slightly below the RBF SVM.

## MLP Width Ablation

The following hidden layer sizes were tested:

| Hidden Size | Accuracy | Macro-F1 |
|---:|---:|---:|
| 64 | 0.8430 | 0.8407 |
| 128 | 0.8495 | 0.8497 |
| 256 | 0.8525 | 0.8526 |
| 512 | 0.8380 | 0.8390 |

The best result was obtained with hidden size 256. Increasing the hidden size to 512 did not improve performance under the current training setup.

## Error Analysis

Initial confusion matrix analysis shows that most classification errors occur among visually similar upper-body clothing classes:

- T-shirt/top
- Pullover
- Coat
- Shirt

The Shirt class is particularly difficult for the MLP model. This suggests that the current finite neural network has difficulty learning discriminative features for fine-grained clothing categories.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## How to Run

Load and test the dataset:

```powershell
python src/data_loader.py
```

Run baseline models:

```powershell
python src/baselines.py
```

Train the finite MLP model:

```powershell
python src/mlp_train.py
```

Run MLP width ablation:

```powershell
python src/mlp_width_ablation.py
```

Combine result tables:

```powershell
python src/combine_results.py
```

Generate result plots:

```powershell
python src/plot_results.py
```

Generate confusion matrices:

```powershell
python src/confusion_matrices.py
```

## Reproducibility

The repository contains:

- Source code for all current experiments
- `requirements.txt` dependency file
- Dataset download instructions through code
- CSV result files
- Generated figures
- Clear running instructions

The `data/` directory is ignored by Git because Fashion-MNIST is automatically downloaded when the scripts are executed.

## Next Steps

The next phase of the project will focus on:

- Implementing or approximating the Neural Tangent Kernel model
- Comparing NTK-based classification with finite MLP models
- Extending the ablation study
- Improving regularization and hyperparameter tuning
- Performing deeper error analysis
- Preparing the final academic report