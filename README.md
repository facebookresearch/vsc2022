# 2023 Video Similarity Challenge Codebase

This is the codebase for the [2023 Video Similarity Challenge](https://sites.google.com/view/vcdw2023/video-similarity-challenge) and
the associated dataset.

The Video Similarity Challenge will be featured at the
[VCDW Workshop at CVPR 2023](https://sites.google.com/view/vcdw2023/video-similarity-challenge)!

The design and results of the challenge can be found in our
[paper](https://drive.google.com/file/d/1MujZDupmVVJC1h9GTU1LS4az-KoqbpZm/view).

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
2022-11-09 17:00:33 INFO     Descriptor track micro-AP (uAP): 0.4754
```

### Matching track eval

```
$ ./matching_eval.py --predictions vsc_eval_data/matches.csv --ground_truth vsc_eval_data/gt.csv
Matching track segment AP: 0.3650
```

## Baseline results on training set

| method                        | score norm         | Descriptor μAP | Matching μAP |
|-------------------------------|--------------------|----------------|--------------|
| [SSCD](docs/baseline.md)      | :x:                | 0.4754         | 0.3650       |
| [SSCD](docs/baseline.md)      | :heavy_check_mark: | 0.6499         | 0.4692       |
| [DINO](docs/baseline_dino.md) | :heavy_check_mark: | _0.4402_       | 0.3393       |
| [DnS](docs/baseline_dns.md)   | :heavy_check_mark: | _0.4129_       | 0.3211       |

**Note**: Numbers in _italics_ do not conform to challenge rules

## License

The VSC codebase is released under the [MIT license](LICENSE).
