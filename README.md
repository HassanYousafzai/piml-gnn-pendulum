# PIML-GNN for Pendulum Dynamics

## Overview
This repository implements a **Physics-Informed Graph Neural Network (PIML-GNN)** in PyTorch for simulating a constrained pendulum, enforcing **Differential Algebraic Equations (DAEs)** to model port-Hamiltonian dynamics. Across multiple runs, the code consistently achieves a **physics loss of ~0.10–0.11** and **data loss of ~0.25–0.27** after 180 epochs, with visualizations demonstrating accurate predictions of position, momentum, and energy conservation.

This work supports my research in physics-informed machine learning, targeting applications in engineering (e.g., robotic control, vehicle dynamics).

## Features
- **Physics-Informed GNN**: Combines Graph Convolutional Networks (GCNs) with DAEs to enforce physical constraints (e.g., port-Hamiltonian systems: dq/dt = p/m, dp/dt = -mgl*sin(q)).
- **DAE Integration**: Ensures energy conservation and accurate dynamics for pendulum simulation.
- **Visualizations**: Generates plots for predicted vs. true position, momentum, energy (`plots/pendulum_results_epoch_X.png`), and loss trends (`plots/loss_trends.png`).
- **Interactive Plot**: Includes a Chart.js loss plot (`loss_plot.json`) for interactive visualization of training metrics.
- **Robustness**: Consistent performance across multiple runs, with physics loss stabilizing at ~0.10–0.11.
- **Extensibility**: Framework adaptable to real datasets (e.g., CFD Simulation Dataset) and industry applications.

## Installation
To run the code, install the required dependencies:

```bash
pip install torch torch-geometric torch-scatter torch-sparse numpy matplotlib
```

For GPU support, ensure PyTorch and PyTorch Geometric versions match your CUDA version (e.g., PyTorch 2.0.0, torch-geometric 2.3.0):

```bash
pip install torch==2.0.0
pip install torch-geometric==2.3.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu116.html
```

*Note*: Replace `cu116` with your CUDA version (e.g., `cu118`) or use `cpu` for CPU-only setups.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/hassan-yousafzai/piml-gnn-pendulum.git
   cd piml-gnn-pendulum
   ```
2. Run the script:
   ```bash
   python piml_gnn_pendulum.py
   ```
3. Outputs:
   - **Console**: Loss metrics every 20 epochs (e.g., `Epoch 180, Loss: 0.3536, Data Loss: 0.2504, Physics Loss: 0.1032` from a recent run).
   - **Plots**: Visualizations in `plots/`:
     - `pendulum_results_epoch_X.png`: Predicted vs. true position, momentum, and energy.
     - `loss_trends.png`: Total, data, and physics loss trends.
   - **Interactive Plot**: `loss_plot.json` (Chart.js) for loss visualization (view via a Chart.js-compatible viewer).

## Results
- **Performance** (across multiple runs at 180 epochs):
  - **Physics Loss**: ~0.10–0.11 (e.g., 0.1032, 0.1019, 0.0966, 0.1082), enforcing DAE constraints and energy conservation.
  - **Data Loss**: ~0.25–0.27 (e.g., 0.2504, 0.2542, 0.2662, 0.2478), reflecting accurate position/momentum predictions.
  - **Total Loss**: ~0.35–0.36 (e.g., 0.3536, 0.3561, 0.3628, 0.3560), indicating robust convergence.
- **Visualizations**:
  - Accurate predictions of pendulum position and momentum against synthetic data.
  - Stable energy conservation, aligning with port-Hamiltonian dynamics.
  - Loss trends show rapid physics loss reduction and stable data loss.
- **Applications**: Extensible to engineering domains (e.g., robotic control, vehicle dynamics) and real datasets for industry collaboration (e.g., Bosch, BMW).
- **Robustness**: Consistent performance across runs, with minor variations due to random initialization (controlled via `torch.manual_seed(42)`).

## Files
- `piml_gnn_pendulum.py`: Main script implementing the PIML-GNN model, training loop, and visualizations.
- `plots/`: Directory containing:
  - `pendulum_results_epoch_X.png`: Position, momentum, and energy plots for each 20th epoch.
  - `loss_trends.png`: Total, data, and physics loss trends.
- `loss_plot.json`: Chart.js configuration for interactive loss visualization.

## Future Work
- Integrate real-world datasets (e.g., CFD Simulation Dataset) for enhanced applicability.
- Optimize hyperparameters (e.g., `lambda_physics`, hidden dimensions) to reduce data loss (~0.25).
- Extend to other dynamic systems (e.g., fluid dynamics, electrical circuits) for broader impact.
- Incorporate Retrieval-Augmented Generation (RAG) for physics literature integration.


## Contact
Hassan Mahmood Yousafzai  
Email: hassan.yousafzai@gmail.com  

