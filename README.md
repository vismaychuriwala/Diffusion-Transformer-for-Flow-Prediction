# DiffFluid: Predicting Flow Dynamics with Diffusion Models

This repository contains our implementation of a diffusion-based transformer model for simulating fluid dynamics, inspired by the [DiffFluid paper (2024)](https://arxiv.org/abs/2401.07196).

We reproduce and extend the original results on Navier-Stokes simulations and evaluate the model’s performance on the Lattice Boltzmann Method (LBM), demonstrating the versatility of denoising diffusion probabilistic models (DDPMs) as general-purpose solvers for flow dynamics.

---

## File Overview

- `difffluid.ipynb`: Main notebook for training and evaluating the DiffFluid model on both Navier-Stokes and LBM datasets.
- `navier_stokes.py`: Dataset generation and simulation of 2D Navier-Stokes flows using spectral methods.
- `lattice_boltzmann.py`: LBM-based simulation using the D2Q9 lattice for modeling shear instability.
- `model.py`: Diffusion-based transformer model implementation and training loop.

---

## Project Goals

- Reproduce results from the DiffFluid paper using a U-Net + Transformer-based diffusion model.
- Test the model’s generalizability on different simulation frameworks (Navier-Stokes and LBM).
- Identify pitfalls in training, including loss formulation errors, and correct them.
- Evaluate model performance with both quantitative metrics (MSE, L2 error) and qualitative visualizations.

---

## ⚙️ Methodology

### Simulation Types:
- **Navier-Stokes**: Solved using a spectral solver with Crank-Nicholson time integration. Dataset consists of vorticity fields on a 32×32 grid.
- **Lattice Boltzmann (D2Q9)**: Simulates 2D shear instability. The dataset contains density snapshots over 50 timesteps.

### Model Architecture:
- U-Net-style encoder-decoder with Transformer layers for global context.
- Predicts the added noise in the DDPM framework.
- Trained to denoise 10-frame input sequences and predict the next timestep.

### Loss Function:
- Combined **Mean Squared Error (MSE)** and **L1 Loss** (weighted).
- **Gradient clipping** and **dropout** used to stabilize training.
- Training with **AdamW optimizer** and **ReduceLROnPlateau** scheduler.

---

## 📊 Results

| Simulation Type     | Relative L2 Error |
|---------------------|------------------|
| Navier-Stokes       | 0.07967          |
| Lattice-Boltzmann   | 0.0598           |

- Correcting the loss formulation improved both MSE and visual output dramatically.
- Visualizations showed sharp and physically coherent predictions post-fix.
- Validation loss stabilized around **0.05** after ~100 epochs.

---

## How to Use

1. Clone the repository:
```bash
git clone https://github.com/your_username/difffluid-flow-sim.git
cd difffluid-flow-sim
