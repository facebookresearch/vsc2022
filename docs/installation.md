# Installation

This codebase has been tested with conda environments.
We document the environments we've used for testing below:

## Evaluation (without VCSL)

```
conda create --name vsc -c pytorch -c nvidia -c conda-forge pytorch \
  torchvision scikit-learn numpy pandas matplotlib faiss-gpu tqdm \
  pytorch-cuda=11.7
conda activate vsc
pip install einops
```

We don't need pytorch for the codebase currently; this is just the environment I used.

Initializing git submodules is not required for this type of installation.

## Baselines (including VCSL)

The [VCSL](https://github.com/alipay/VCSL) codebase is used to localize matches for our baseline matching methods.

```
conda create --name vsc-vcsl -c pytorch -c nvidia -c conda-forge pytorch \
  torchvision scikit-learn numpy pandas matplotlib faiss-gpu tqdm \
  networkx loguru numba cython h5py pytorch-cuda=11.7
conda activate vsc-vcsl
pip install tslearn einops
```

h5py is not needed, but installing it stops some log spam.

VCSL can be used by installing it on your system, or by initializing
git submodules, which adds it locally at `vcsl_module/`:
```
git submodule init
```

Run [tests](testing.md) to check that VCSL localization tests are no longer skipped.
