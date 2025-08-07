# DLQPM_v2: Deep-Learning Quasi-Parton gas Model
The the implementation of Phys.Lett.B 868 (2025) 139692

A PyTorch-based neural network for predicting thermodynamic properties of Quark-Gluon Plasma (QGP) using residual networks and lattice QCD-inspired constraints.

## Features

### Physics-Informed ML: Models quark and gluon masses (Mud, Ms, Mgluon) as functions of temperature (T) and baryon chemical potential (μB).
### Thermodynamic Predictions: Computes:
  1. Energy density (ε), pressure (P), entropy (S), and trace anomaly (Δ = ε - 3P).
  2. Baryon susceptibility (χ_B^2, χ_B^4).
### Visualization: Auto-generates plots during training (e.g., Δ/T^4 vs T/Tc).
### High-Temperature Constraints: Enforces HTL (Hard Thermal Loop) mass relations at high T.
  
## Usage

### Run the model locally (CPU or Cuda):
  * python3 DLQPM.py
### SLURM Submission (HPC):
  * sbatch submit.sh

## Outputs

### Models: Saved in ./model/ (Mud_model.pt, Ms_model.pt, Mg_model.pt).
### Plots: Generated in ./pic/ (e.g., Delta_vs_T_epoch5000.jpg).
### Loss Log: Loss_all.npy tracks training progress.

## Code Structure

* Core Network: Net_Mass (Residual NN) with 8 hidden layers and sigmoid activations.
* Physics Functions:
  1. Partition functions (lnZ_q, lnZ_g) via Gauss-Laguerre quadrature.
  2. Loss terms: Lattice QCD data (s_true, D_true) and HTL constraints.
  3. Visualization: Functions like plot_Delta(), plot_ed() log training progress.

## Data Requirements

Input CSV files (from lattice QCD and Thermal-FIST ):
hotqcd_1407.6387_noerrbar_allT.csv: Thermodynamic observables (s/T^3, P/T^4, etc.).
fig1_data.csv: Baryon susceptibilities (χ_B^2, χ_B^4).

## Notes for Users:

1. Hardware: Defaults to CPU; modify device = torch.device('cuda') for GPU support.
2. Hyperparameters: Adjust epochs, learning_rate in DLQPM.py as needed.
3. If this program has been utilized in your work, please cite the following two original publications in the acknowledgments and references section:
   * Phys.Lett.B 868 (2025) 139692 and Phys.Lett.B 844 (2023) 138088
