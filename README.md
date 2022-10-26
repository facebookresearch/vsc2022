# Video Similarity Challenge 2022 Codebase

This is the codebase for the [2022 Video Similarity Challenge](https://vsc.drivendata.org/) and
the associated dataset.

## Documentation

The [docs](docs) folder contains additional documentation for this codebase:

### Installation

See [installation](docs/installation.md)

### Running tests

See [testing](docs/testing.md)

### Reproducing baseline results

See [baseline](docs/baseline.md)

## Running evaluations

### Descriptor eval

```
$ ./descriptor_eval.py --query_features vsc_eval_data/queries.npz --ref_features vsc_eval_data/refs.npz --ground_truth vsc_eval_data/gt.csv
Starting Descriptor level eval
...
2022-10-20 12:23:09 INFO     Descriptor track micro-AP (uAP): 0.7894
```

### Matching track eval

```
$ ./matching_eval.py --predictions vsc_eval_data/matches.csv --ground_truth vsc_eval_data/gt.csv
Matching track segment AP: 0.5048
```

## License

The VSC codebase is released under the [MIT license](LICENSE).
