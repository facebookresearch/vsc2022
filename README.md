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
$ python -m vsc.descriptor_eval --query_path vsc_eval_data/queries.npz --ref_path vsc_eval_data/refs.npz --gt_path vsc_eval_data/gt.csv
Starting Descriptor level eval
Loaded 997 query features
Loaded 4974 ref features
Performing KNN with k: 5
Loaded GT from ../vsc_eval_data/gt.csv
Micro AP : 0.79020867778352
```

## Matching track eval

```
$ python -c 'from vsc.metrics import evaluate_matching_track; metrics = evaluate_matching_track("vsc_eval_data/gt.csv", "vsc_eval_data/matches.csv"); print(f"matching ap = {metrics.segment_ap_v2.ap}")'
matching ap = 0.5047952268737359
```
