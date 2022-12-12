# DINO baseline
The baseline can be also implemented with [DINO](https://arxiv.org/abs/2104.14294)
features, using the pretrained ViT-B/8 provided in the official [repo](https://github.com/facebookresearch/dino).

This baseline produces global frame descriptors with 1536 dimensions and builds on the same process as the previous
[SSCD baseline](baseline.md) for video retrieval and localization.

Note that due to the large dimensionality of the generated descriptors, this baseline would not be eligible
for the Descriptor Track of the challenge. Hence, it is considered only for the Matching Track.

Due to its poor performance when no normalization is applied, this baseline is benchmarked only with the
use of score normalization.

## Inference

We use the same inference script to extract DINO descriptors at one frame per
second (1 fps).

### Training dataset
Run inference on queries and references for the training dataset providing `dino` to the `--baseline` argument.
```
python -m vsc.baseline.inference \
    --baseline dino \
    --accelerator cuda --processes 2 \
    --output_file ./output/training_queries_dino.npz \
    --dataset_path ./training_dataset/queries
```
```
python -m vsc.baseline.inference \
    --baseline dino \
    --accelerator cuda --processes 2 \
    --output_file ./output/training_refs_dino.npz \
    --dataset_path ./training_dataset/refs
```
For GPU inference, set `--accelerator cuda` and set `--processes` to
the number of GPUs on the system.

For CPU inference, set `--accelerator cpu` and set `--processes` to
the desired number of inference processes.

Multiple machine distributed inference is not tested here,
but should be possible with a bit of work.

### Validation references
Since this baseline is used only with score normalization baseline, we'll use the
validation references as the "noise" dataset when evaluating on the
training dataset, so do inference on that dataset too.
(For validation predictions, we'll use the training references as the
"noise" predictions.)

Run inference on references of the validation dataset to generate the "noise" dataset.
```
python -m vsc.baseline.inference \
    --baseline dino \
    --accelerator cuda --processes 2 \
    --output_file ./output/validation_refs_dino.npz \
    --dataset_path ./validation_dataset/refs
```

**Note**: any models, including simple modeling like PCA whitening
or codecs like ITQ, must only be trained on the training dataset.

## Matching
Similar to the [SSCD baseline](baseline.md), the `sscd_baseline.py` script performs all
phases of matching:retrieval, localization, and assigning scores to localized matches.


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

Run the matching script providing the files with DINO descriptors to get
the final results of the baseline. Also, provide a feature file for score normalization.
```
python -m vsc.baseline.sscd_baseline \
    --query_features ./output/training_queries_dino.npz \
    --ref_features ./output/training_refs_dino.npz \
    --score_norm_features ./output/validation_refs_dino.npz \
    --ground_truth ./training_dataset/ground_truth.csv \
    --output_path ./output/dino_score_norm

...
2022-12-07 19:10:59 INFO     Candidate uAP: 0.4784
2022-12-07 19:11:00 INFO     Matching track metric: 0.3653
```
