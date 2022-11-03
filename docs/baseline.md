# SSCD baseline

We provide a baseline implementation as a starting point to demonstrate
how to use this dataset.

Our baseline builds on the [SSCD](https://github.com/facebookresearch/sscd-copy-detection)
fingerprint, a 512 dimension global feature designed for copy detection.

We implement two types of matching: one that uses a score normalization
technique (discussed in the [Image Similarity Challenge](https://arxiv.org/abs/2106.09672) paper),
and one that uses raw features without L2 normalization.

To support both types of matching, we remove L2 normalization from the
SSCD model before running inference.
If you only plan on matching using score normalization, you can skip that step.

## Download the SSCD model

We used the `sscd_disc_mixup` model in our experiments.
A torchscript version of that model can be found
[here](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt).

## Adapt the SSCD model

Run the following from the root of this repository:
```
python -m vsc.baseline.adapt_sscd_model \
    --input_torchscript ./sscd_disc_mixup.torchscript.pt \
    --output_torchscript ./sscd_disc_mixup.no_l2_norm.torchscript.pt
```

## Inference

The baseline inference script extracts descriptors at one frame per
second (1 fps).

### Training dataset

Run inference on queries and references for the training dataset.

```
python -m vsc.baseline.inference \
    --accelerator cuda --processes 2 \
    --torchscript_path ./sscd_disc_mixup.no_l2_norm.torchscript.pt \
    --output_file ./output/training_queries.npz \
    --dataset_path ./training_dataset/queries
```
```
python -m vsc.baseline.inference \
    --accelerator cuda --processes 2 \
    --torchscript_path ./sscd_disc_mixup.no_l2_norm.torchscript.pt \
    --output_file ./output/training_refs.npz \
    --dataset_path ./training_dataset/refs
```

For GPU inference, set `--accelerator cuda` and set `--processes` to
the number of GPUs on the system.

For CPU inference, set `--accelerator cpu` and set `--processes` to
the desired number of inference processes.

Multiple machine distributed inference is not tested here,
but should be possible with a bit of work.

### Validation references

If you plan on running the score normalization baseline, we'll use the
validation references as the "noise" dataset when evaluating on the
training dataset, so do inference on that dataset too.
(For validation predictions, we'll use the training references as the
"noise" predictions.)

```
python -m vsc.baseline.inference \
    --accelerator gpu --processes 2 \
    --torchscript_path ./sscd_disc_mixup.no_l2_norm.torchscript.pt \
    --output_file ./output/validation_refs.npz \
    --dataset_path ./validation_dataset/refs
```

**Note**: any models, including simple modeling like PCA whitening
or codecs like ITQ, must only be trained on the training dataset.

## Matching

The `sscd_baseline.py` script performs all phases of matching:
retrieval, localization, and assigning scores to localized matches.

Retrieval uses the [FAISS](https://github.com/facebookresearch/faiss)
library, and will use GPU accelerated search if GPUs are found and
the library is installed with GPU support.

Localization uses the [VCSL](https://github.com/alipay/VCSL) library.
We specifically use the temporal network flow (TN) method that VCSL
provides.
We mostly use off-the-shelf settings.
The score normalization evaluation adjusts similarity since this
method expects positive similarity in matching regions.

The baseline script also runs a matching track evaluation, and a descriptor micro AP.
The descriptor micro AP estimates the descriptor track score, but
does not enforce the same limits as the `descriptor_eval.py` evaluation.

The matching script provides several outputs in the `--output_path`
directory, including precision-recall plots,
score-normalized descriptor files, and both retrieved candidate pairs
and localized matches in `.csv` formats.

### With score normalization

```
python -m vsc.baseline.sscd_baseline \
    --query_features ./output/training_queries.npz \
    --ref_features ./output/training_refs.npz \
    --score_norm_features ./output/validation_refs.npz \
    --ground_truth ./training_dataset/ground_truth.csv \
    --output_path ./output/sscd_score_norm

...
2022-10-20 17:02:38 INFO     Candidate uAP: 0.6540
2022-10-20 17:02:41 INFO     Matching track metric: 0.4601
```

### Without score normalization

```
python -m vsc.baseline.sscd_baseline \
    --query_features ./output/training_queries.npz \
    --ref_features ./output/training_refs.npz \
    --ground_truth ./training_dataset/ground_truth.csv \
    --output_path ./output/sscd_raw

...
2022-10-21 10:27:04 INFO     Candidate uAP: 0.4790
2022-10-21 10:27:05 INFO     Matching track metric: 0.3497
```

## Suggestions for improving on this baseline

### Descriptors

There were many promising submissions to the Image Similarity Challenge
whose features could be explored in the video domain.
See the [results paper](https://proceedings.mlr.press/v176/papakipos22a.html).

### Localization and scoring

We use a relatively simple localization method with very little tuning.
It is likely possible to significantly improve localization.

We use a very simple method to assign a score to predicted match segments:
the maximum similarity of the predicted match region.
It should be possible to improve the calibration of segment scores
either by using better score heuristics, or by training a scoring model.
