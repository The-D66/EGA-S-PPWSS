# EGA-S: Enhanced Genetic Algorithm with Shifting for Pumped Pipeline Water Supply Systems

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

EGA-S is a domain-specific optimization algorithm for day-ahead pump scheduling in long-distance pressurized water supply systems (PPWSSs). It features:

- **Shifting Operator**: A novel cyclic permutation operator that preserves the total water volume constraint by construction, eliminating the most binding constraint from the penalty landscape.
- **Heuristic Feasible Initialization**: Generates initial populations guided by the Time-of-Use (TOU) tariff structure.
- **Dual-Strategy Comparison**: Systematically compares COST-minimization and ENERGY-minimization scheduling strategies.

## Quick Start

### Local Installation

```bash
git clone https://github.com/The-D66/EGA-S-PPWSS.git
cd EGA-S-PPWSS
pip install -r requirements.txt
```

### Run a Single Optimization

```bash
# COST mode, 15 m³/s target
python main.py --area sA-sB --method eco --aim_vol 1320000
```

### Run Benchmark (All Conditions)

```bash
python run_benchmark.py
```

### Docker

```bash
# Build
docker build -t ega-s-ppwss .

# Run benchmark
docker run --rm ega-s-ppwss

# Run specific condition
docker run --rm ega-s-ppwss python main.py --area sA-sB --method eco --aim_vol 1320000
```

## Project Structure

```
├── main.py                      # Main entry point
├── run_benchmark.py             # Batch benchmark script (GA, PSO, DE, EGA-S, MILP)
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker build file
├── data/
│   ├── config.json              # Algorithm hyperparameters
│   └── area/                    # System configuration files
│       ├── sA-sB.json           # Base condition (n=0.014)
│       └── sA-sB-{n}.json      # Roughness variants
├── pump_opt/
│   ├── api.py                   # CLI argument parser
│   ├── problem.py               # Optimization problem wrapper
│   ├── optimization/
│   │   ├── panning_GA.py        # EGA-S algorithm (core)
│   │   ├── panning_DE.py        # DE baseline
│   │   ├── milp_model.py        # MILP baseline
│   │   └── milp_utils.py        # MILP utilities
│   └── simulation/
│       ├── model.py             # Hydraulic system model
│       ├── pump.py              # Pump station model
│       ├── pipe.py              # Pipeline friction model
│       └── tank.py              # Reservoir model
```

## Data Availability

The data files provided in this repository are **synthetic benchmark datasets** derived from the actual system. All station names, geographic identifiers, and sensitive operational parameters have been anonymized and transformed (scaled + offset) to protect proprietary information under the data-sharing agreement with the operating authority.

The synthetic data preserves the **structural characteristics** of the original system (nonlinear pump curves, quadratic friction losses, reservoir dynamics) to allow reproduction of the key findings reported in the paper.

### Mapping to Paper Conditions

| Condition | Volume (m³) | Description |
|-----------|-------------|-------------|
| C1 | 1,320,000 | 25% capacity |
| C2 | 2,112,000 | 40% capacity |
| C3 | 3,168,000 | 60% capacity |
| C4 | 4,224,000 | 80% capacity |

## Reproducibility

Random seeds used in the paper experiments: `[0, 1, 2, 3, 4]`

To reproduce the benchmark results:
```bash
python run_benchmark.py --seeds 0 1 2 3 4 --pop_size 2000 --max_iter 100
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
