# Video Similarity Challenge 2022 Codebase

This is the codebase for the 2022 Video Similarity Challenge and
the associated dataset.

## Installation (without VCSL)

```
conda create --name vsc -c pytorch -c conda-forge pytorch torchvision \
  scikit-learn numpy pandas matplotlib faiss tqdm
```

We don't need pytorch for the codebase currently; this is just the environment I used.

Initializing git submodules is not required for this type of installation.

## Installation with VCSL

The [VCSL](https://github.com/alipay/VCSL) codebase is used to localize matches for our baseline matching methods.

```
conda create --name vsc-vcsl -c pytorch -c conda-forge pytorch torchvision \
  scikit-learn numpy pandas matplotlib faiss networkx loguru numba cython \
  h5py tqdm
conda activate vsc-vcsl
pip install tslearn
```

h5py is not needed, but installing it stops some log spam.

## Running tests

```
$ python -m unittest discover
..ss.................
----------------------------------------------------------------------
Ran 21 tests in 0.060s

OK (skipped=2)
```

The skipped tests are localization tests that only run if VCSL is installed.

When run, localization tests warn about unclosed multiprocessing pools.

## Descriptor eval

```
$ python -m vsc.descriptor_eval --query_features vsc_eval_data/queries.npz --ref_features vsc_eval_data/refs.npz --ground_truth vsc_eval_data/gt.csv
Starting Descriptor level eval
...
2022-10-20 12:23:09 INFO     Descriptor track micro-AP (uAP): 0.7894
```

## Matching track eval

```
$ python -m vsc.matching_eval --predictions vsc_eval_data/matches.csv --ground_truth vsc_eval_data/gt.csv
Matching track segment AP: 0.5048
```
