# ContainerKids

This repository is related to a publication that investiges the role of sleep in the formation of spatial context representations in toddlers aged 2-3 years. 
It includes a PyTorch-Geometric pipeline for classifying behavioral graphs (sleep vs. wake) and explaining model decisions using GNNExplainer.  
This repository contains:

- **`config.py`**  
  Centralized hyperparameters and file‐paths.  

- **`model.py`**  
  Defines the `GCNWithEdgeWeights` graph classifier and a `WrappedModel` for explainability.  

- **`data_utils.py`**  
  Loading graphs (`.pt` files), splitting into train/val/test, factory functions for `DataLoader`.  

- **`train.py`**  
  - `run_cv(...)`: 10-fold cross‐validation with early stopping.  
  - `train_final(...)`: train final model on 90% of data.  
  - `evaluate(...)`: compute accuracy & confusion matrix.  

- **`perm_test.py`**  
  Functions for running a permutation test on held-out data.  

- **`explain.py`**  
  Build a `GNNExplainer`, extract node‐importance masks, and save subgraph visualizations.  

- **`visualize.py`**  
  Plotting utilities: confusion matrix, U–V edge‐pair histograms.  

- **`main.py`**  
  CLI entry point. Supports modes:
  - `cv`      – run cross‐validation  
  - `test`   – train final model  
  - `perm`    – run permutation test  
  - `explain` – generate GNN explanations  

---

## 🚀 Quick Start

1. **Clone & install**  
   ```bash
   git clone https://github.com/LisaBastian/ContainerKids.git
   cd ContainerKids
   pip install -r requirements.txt
