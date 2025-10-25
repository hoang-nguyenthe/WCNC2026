# ⚙️ Optimizing Erasure Coding and Node Allocation for Blockchain Storage (GDA)

This project implements a **Guided Decomposition Algorithm (GDA)** to optimize **Erasure Coding** and **Node Allocation** for blockchain storage, as presented in the paper:

> 🧾 *"Optimizing Erasure Coding and Node Allocation for Blockchain Storage: A Guided Decomposition Approach"*

---

## 🏗️ Project Overview

This project simulates and evaluates the **GDA** algorithm for solving a **Mixed-Integer Nonlinear Programming (MINLP)** problem (P1) that aims to jointly optimize:
- **Erasure Coding parameters** $(n, k)$  
- **Data shard placement** $(x_i, z_i)$  
to minimize the **total storage cost**.

The proposed **GDA** algorithm (implemented as `solve_exact_decomposition`) decomposes P1 into a series of **MIP feasibility sub-problems** (P2), achieving near-global optimal solutions efficiently.

---

## 🧩 Two-Tier Blockchain Storage Framework

The implemented system adopts a **two-tier storage design** to enhance scalability:

| Tier | Description | Storage Strategy |
|------|--------------|------------------|
| 🟢 **Active Tier** | Frequently accessed recent blocks | Full Replication |
| 🔵 **Archive Tier** | Historical blockchain data | MDS-based Erasure Coding |

---

## 📁 Project Structure

```
PERFORMANCE_EVALUATION_PROJECT/
├── __pycache__/
├── venv/
├── baselines.py               # Greedy baseline algorithm
├── config.py                  # System constants (S_A, S_H, T_max, etc.)
├── evaluation_results_v3.csv  # Raw evaluation data
├── exact_solver.py            # Core implementation of GDA (Algorithm 1 & P2)
├── plot_scen1_cost.pdf        # Cost vs. Number of Nodes (Fig. 2)
├── plot_scen1_time.pdf        # Runtime vs. Number of Nodes (Fig. 3)
├── plot_scen2_3D_cost.pdf     # 3D Sensitivity Analysis (Fig. 4)
├── plot_scen2_3D_stem.pdf
├── plot_scen2_3D_sum_z.pdf
├── requirements.txt           # Dependencies
└── run_evaluation.py          # Main script for experiments and plotting
```

---

## 📘 File Descriptions

### 🔧 `config.py`
Defines global constants such as:
- `S_A`, `S_H`: Archive and Active Tier sizes
- `T_max`: Maximum audit time
- `tau_H`, `tau_A`: Fault tolerance thresholds
- `BIG_M`: Big-M constant for linearization

### 🧮 `exact_solver.py`
Core of the **Guided Decomposition Algorithm (GDA)**, implementing Algorithm 1 from the paper.

- `check_feasibility(...)`: Builds and solves each **MIP feasibility subproblem (P2)** using `pulp`, implementing constraints (3, 4, 7–10, 14, 16).
- `solve_exact_decomposition(...)`: Executes the 3-level search loops over $(n, k, Z = \sum z_i)$ to find the optimal solution.

### ⚙️ `baselines.py`
Contains the **Greedy Baseline** (`solve_greedy_baseline`), where $n = 70\% \cdot N$ and $k = n-1$. Used for performance comparison.

### 📊 `run_evaluation.py`
Main experiment script that:
1. Generates synthetic network data (`create_problem_instance(...)`)
2. Runs both solvers:
   - `solve_monolithic(...)`: Gurobi-based commercial solver for MINLP
   - `solve_exact_decomposition(...)`: Proposed GDA method
3. Executes two experiment scenarios:
   - **Scenario 1 (Scalability):** Vary $N$ (number of nodes) from 20 → 100
   - **Scenario 2 (Sensitivity):** Vary $S_A$ and $p_{\mu}$ at fixed $N=50$
4. Plots Figures 2–4 automatically

---

## 🧠 Dependencies

Make sure Python and the required libraries are installed:

```bash
pip install -r requirements.txt
```

**Main libraries used:**
- 🧮 `pulp`: MILP solver for GDA
- 🧠 `gurobipy`: Commercial solver (requires valid Gurobi license)
- 📈 `matplotlib`: Plot generation
- 📊 `pandas`: Data management
- 🔢 `numpy`: Randomized data generation

---

## 🚀 Run Experiments

To reproduce all experiments and generate figures (`.pdf` / `.png`):

```bash
python run_evaluation.py
```

⏳ *Note:* The `solve_monolithic` (Gurobi) method may take up to 1000s per test case, while the proposed **GDA** algorithm runs significantly faster.

---

## 🧭 Results Overview

| Figure | Description | File |
|---------|--------------|------|
| 🖼️ **Fig. 2** | Total cost vs. number of nodes | `plot_scen1_cost.pdf` |
| ⏱️ **Fig. 3** | Runtime comparison (GDA vs Gurobi) | `plot_scen1_time.pdf` |
| 🌐 **Fig. 4** | Sensitivity analysis in 3D | `plot_scen2_3D_cost.pdf` |

---

## 🧩 Credits

Developed as a simulation of the **Guided Decomposition Algorithm (GDA)** described in the research paper:

> *Optimizing Erasure Coding and Node Allocation for Blockchain Storage: A Guided Decomposition Approach*

---

### 🧠 Author
**Nguyen The Hoang**  
📍 Ho Chi Minh City University of Technology (HCMUT – VNU-HCM)  
📧 Contact: *hoang.nguyenthe@hcmut.edu.vn*
