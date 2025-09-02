

# Project Directory Overview

This directory contains scripts, data, and results for activity recognition experiments using deep learning and federated learning approaches. Below is a description of all files and folders in this workspace.

## Folders and Files

### pamap/
- `pamap_fl.py` — Federated learning simulation script for the PAMAP2 dataset.
- `train_pamap.py` — Centralized training script for the PAMAP2 dataset.
- `PAMAP2_Dataset/` — Contains data files and documentation for the PAMAP2 dataset.
	- `Protocol/` — Main data files (per subject).
	- `Optional/` — Additional data files (per subject).
	- `*.pdf` — Dataset documentation.
- `results_fl_pamap2_simulation/` — Results from federated learning experiments (plots, confusion matrices).
- `results_pamap2/` — Results from centralized training (model files, reports, plots).

### UCI_HAR/
- `tcn_fl.py` — Federated learning simulation script for the UCI HAR dataset.
- `train_UCR_HAR.py` — Centralized and hybrid model training script for UCI HAR.
- `train_egtcn.py` — EGTCN-like model training script for UCI HAR.
- `data/UCI HAR Dataset/` — Contains data files and documentation for the UCI HAR dataset.
	- `test/` and `train/` — Data splits for testing and training.
	- `Inertial Signals/` — Raw sensor signal files.
	- `*.txt` — Dataset documentation and features.
- `results_egetcn_like/` — Results from EGTCN-like model experiments.
- `results_fl_hybrid_advanced/` — Results from advanced federated learning experiments.
- `results_fl_hybrid_simulation/` — Results from hybrid federated learning simulations.
- `results_fl_improved_model/` — (Empty) Reserved for improved FL model results.
- `results_fl_paper_comparison/` — (Empty) Reserved for paper comparison results.
- `results_fl_simulation/` — (Empty) Reserved for FL simulation results.
- `results_fl_simulation_fast/` — Results from fast FL simulation runs.
- `results_uci_har/` — Results from centralized UCI HAR experiments.
- `results_uci_har_hybrid/` — Results from hybrid model experiments.
- `results_uci_har_hybrid_improved/` — Results from improved hybrid model experiments.

### Readme.md
This file. Provides an overview of all files and folders in the workspace.

---
For details on how to use each script, please refer to the comments and documentation within the respective Python files.

