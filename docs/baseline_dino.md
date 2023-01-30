# DINO baseline
The baseline can also be implemented with [DINO](https://arxiv.org/abs/2104.14294)
features, using the pretrained ViT-S/16 provided in the official [repo](https://github.com/facebookresearch/dino) 
that has similar computational requirements to a ResNet50, so as to meet the limitation of the challenge.
We follow the feature extraction process for copy detection, as described in the original paper. 

This baseline produces global frame descriptors with 768 dimensions and builds on the same process as the previous
[SSCD baseline](baseline.md) for video retrieval and localization.

Note that due to the large dimensionality of the generated descriptors, this baseline would not be eligible
for the Descriptor Track of the challenge. Hence, it is considered only for the Matching Track.

Due to its poor performance when no normalization is applied, this baseline is benchmarked only with the
use of score normalization.

## Download the DINO model

We provide the model used in our experiments. The torchscript version of the model can be found
[here](https://mever.iti.gr/vsc2022/dino_vits16_cdpool.torchscript.pt).

## Inference

Use the same inference script to extract DINO descriptors at one frame per
second (1 fps).

### Training dataset
Run inference on queries and references for the training dataset providing `dino` to the `--baseline` argument.
Also, select the `RESIZE_224_SQUARE` for the `--transforms` to use the appropriate preprocessing for this approach. 

```
python -m vsc.baseline.inference \
    --baseline dino \
    --torchscript_path ./dino_vits16_cdpool.torchscript.pt \
    --transforms RESIZE_224_SQUARE \
    --accelerator cuda --processes 2 \
    --output_file ./output/training_queries_dino.npz \
    --dataset_path ./training_dataset/queries
```
```
python -m vsc.baseline.inference \
    --baseline dino \
    --torchscript_path ./dino_vits16_cdpool.torchscript.pt \
    --transforms RESIZE_224_SQUARE \
    --accelerator cuda --processes 2 \
    --output_file ./output/training_refs_dino.npz \
    --dataset_path ./training_dataset/refs
```

### Validation references
Since this baseline is used only with score normalization baseline, we'll use the
validation references as the "noise" dataset when evaluating on the
training dataset.

Run inference on references of the validation dataset to generate the "noise" dataset.
```
python -m vsc.baseline.inference \
    --baseline dino \
    --torchscript_path ./dino_vits16_cdpool.torchscript.pt \
    --transforms RESIZE_224_SQUARE \
    --accelerator cuda --processes 2 \
    --output_file ./output/validation_refs_dino.npz \
    --dataset_path ./validation_dataset/refs
```

**Note**: any models, including simple modeling like PCA whitening
or codecs like ITQ, must only be trained on the training dataset.

## Matching
Similar to the [SSCD baseline](baseline.md), the `sscd_baseline.py` script performs all phases of matching, 
relying on the [FAISS](https://github.com/facebookresearch/faiss) and [VCSL](https://github.com/alipay/VCSL) 
libraries for retrieval and localization, respectively.

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
2022-12-07 19:10:59 INFO     Candidate uAP: 0.4402
2022-12-07 19:11:00 INFO     Matching track metric: 0.3393
```
