# AiNed Project

This repository contains all the code for the simulation of a fully asynchronous event-based neural network based on JAX and mpi4jax.

## Installation

First install the JAX and mpi4jax libraries by following the steps on the official sites [JAX](https://docs.jax.dev/en/latest/installation.html), [mpi4jax](https://mpi4jax.readthedocs.io/en/latest/installation.html) (this requires to have an MPI library installed) the code is only tested on the cpu versions of both libraries (verify which version is used in the requirements.txt file)\

Then install the rest of the packages.

```bash
pip install -r requirements.txt
```

## Repository Overview
There are multiple main files inside the repository (AED = Asynchronous Event-Driven):
* `async_MLP.py` contains the main code for running [AED MLP](#aed-mlp-and-cnn).

* `async_CNN.py` contains the main code for running [AED CNN](#aed-mlp-and-cnn).

* `MPI_general.py` contains the main code for a generalized implementation of data/model [parallelism](#general-parallelism) for MLP (CNN not supported).

* `optuna_MLP.py` contains the main code for running [hyper-parameter tuning](#optuna-hyper-parameters-tuning) on MLP (CNN is not supported yet) 

* `MLP_finite_diff_verification_test.py` and `CNN_finite_diff_verification_test.py` contains test codes to verify the backpropagation computation's correctness using [finite differences](#finite-differences-verification).

* The `other_helpers/` directory contains the helpers for MPI communication, backpropagation computation and more.

* The `dataset_helpers/` directory contains all the dataloaders for the datasets used in this project (MNIST, NMNIST, SHD, DVSGesture). The `mnist_helper.py` and `cnn_mnist.py` contain the code for running the synchronous NNs and store the results for AED network initialization.


## Usage
## AED MLP and CNN
The files `async_MLP.py` and `async_CNN.py` contain the code to run inference, training and retraining of neural networks.\
The different parameters that can be used are explained in the `Params` data structure.

Run command:
```bash
JAX_PLATFORMS=cpu mpirun -n <NUM_PROCESSES> python <SCRIPT>.py
```
- `NUM_PROCESSES`: must be a multiple of the number of layers in the network.
- `SCRIPT`: script to run (e.g. async_MLP.py)

Example:
```bash
JAX_PLATFORMS=cpu mpirun -n 3 python async_MLP.py
```

## Optuna Hyper Parameters tuning
The file `optuna_MLP.py`contains the code for running hyper-parameter tuning.

## General Parallelism
The file `MPI_general.py` contains the main code for a generalized implementation of data/model parallelism, the MPI communication is independently handled by `general_MPI_helpers.py` completely decoupled fron the main file and different mapping strategies can be implemented in the helper file. Currently supported mapping:
- Batch Parallelism: Same principles as in `async_MLP.py` and `async_CNN.py`
- Model Parallelism: Each layer can be mapped onto a different number of processes and each partial layer computes its own partial gradient (Problem: Not hardware realistic implementation) 

## Finite differences verification
The files `MLP_finite_diff_verification_test.py` and `CNN_finite_diff_verification_test.py` contains the finite difference method for comparing the gradient computed in the async network and the Central Difference Method for gradient verification 

