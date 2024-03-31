# Learning Linear Causal Representations Using Higher-Order Cumulants
This repository contains code written in Python and Julia. The latter is only required to reproduce the non-identifiability results for soft interventions.
## Python
### Setup
Before installing the required packages, we recommend setting up a virtual environment to isolate this project's dependencies.

To install the required packages, run the following command in your terminal:
```bash
pip install -r requirements.txt
```
### Parameter recovery from population cumulants
To test the performance of the method using population cumulants as input, run:

```bash
python3 run_pop.py
```

### Parameter recovery from sample cumulants
To test the performance of the method using sample cumulants as input, run:

```bash
python3 run_sample.py [arguments]
```

#### Arguments
- `--nlatent`: Number of latent variables. (Default: 4)
- `--nobserved`: Number of observed variables. (Default: 5)
- `--nsamples_list`: List of number of samples to consider. (Default: [2500, 5000, 10000, 30000, 50000, 100000, 250000])
- `--nonlinear_X`: Flag to add nonlinearity in the transformation from latent to observed variables. If provided, nonlinearity will be added; if omitted, it won't.
- `--alpha_X`: Coefficient quantifying how much nonlinearity to add in the transformation from latent to observed variables. (Default: 0.1 if `nonlinear_X` is provided. None otherwise)
- `--nonlinear_Z`: Flag to add nonlinearity in the latent space. If provied, nonlinearity will be added; if ommited, it won't.
- `--alpha_Z`: Coefficient quantifying how much nonlinearity to add in the latent space. (Default: 0.1 if `nonlinear_Z` is provided. None otherwise)

## Julia
### Setup
The Julia programming language can be installed [here](https://julialang.org/downloads/). After installation, run the following commands in your terminal to install the required packages
```bash
julia
]
add LinearAlgebra
add Oscar@v1.0.0
```
### Non-identifiability for soft interventions
To reproduce the non-identifiability results for soft interventions, run the following command in your terminal:
```bash
julia worstcase_soft.jl
```