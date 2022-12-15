# DnS baseline

Additionally, we provide another baseline based on the [DnS](https://arxiv.org/abs/2106.13266) method.
The off-the-shelf pretrained models and a modified version of the code from the official
[repo](https://github.com/mever-team/distill-and-select) are used, which have been repurposed to facilitate  
retrieval and localization.

This baseline first extracts features using a ResNet50 pretrained on ImageNet, which are then used on an indexing
process to generate coarse and fine video representations using the corresponding students of the DnS method.

For video retrieval, it builds based on the coarse-grained student, producing global frame descriptors 
with 1024 dimensions.

For video localization, the fine-grained students are employed. They generate refined similarity matrices, 
which are combined with the similarity matrices generated with the coarse representations and ultimately
used for localization.

Similar to the [DINO baseline](baseline_dino.md), this baseline would not be eligible for the Descriptor Track of the challenge,
due to the large dimensionality of the generated descriptors. Hence, it is considered only for the Matching Track. 
Also, due to its poor performance when no normalization is applied, it is benchmarked only with the use
of score normalization.


## Inference

Similar to the previous approaches, we use the inference script to extract descriptors at one frame per
second (1 fps).

This extracts ResNet50 features that will be used later to generate coarse and fine video representations.

### Training dataset

Run inference on queries and references for the training dataset providing `dns` to the `--baseline` argument.
Also, select the `RESIZE_224_SQUARE` for the `--transforms` to use the appropriate preprocessing for this approach. 

```
python -m vsc.baseline.inference \
    --baseline dns \
    --transforms RESIZE_224_SQUARE \
    --store_fp16 --batch_size 128 \
    --accelerator cuda --processes 2 \
    --output_file ./output/training_queries_dns.npz \
    --dataset_path ./training_dataset/queries 
```

```
python -m vsc.baseline.inference \
    --baseline dns \
    --transforms RESIZE_224_SQUARE \
    --store_fp16 --batch_size 128 \
    --accelerator cuda --processes 2 \
    --output_file ./output/training_refs_dns.npz \
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
    --baseline dns \
    --transforms RESIZE_224_SQUARE \
    --store_fp16 --batch_size 128 \
    --accelerator cuda --processes 2 \
    --output_file ./output/validation_refs_dns.npz \
    --dataset_path ./validation_dataset/refs
```

**Note**: to avoid excessive RAM-memory consumption, we recommend using `--store_fp16` flag to store features
in half-precision float (float16) format and a large `--batch_size` to avoid the overloading of the dataloader.

## Indexing

Once the initial features have been extracted, use them for indexing the videos. The `dns_index.py` script
indexes the videos based on a student from the DnS approach, given the extracted feature files of each dataset.

The indexing script will generate in the provided `--output_path` directory the corresponding files 
where the representation are stored in `.npz` format.

### Coarse representations with score normalization

Run indexing with the coarse-grained student, providing also the "noise" dataset for score normalization.

```
python -m vsc.baseline.dns_index \
    --student coarse --accelerator cuda \
    --query_features ./output/training_queries_dns.npz \
    --ref_features ./output/training_refs_dns.npz \
    --score_norm_features ./output/validation_refs_dns.npz \
    --output_path ./output/dns
```

### Fine representations with the fine-grained students

Run indexing with a fine-grained student. Pass either `fine_att` or `fine_bin` for the `--student` argument
to either use the fine-grained attention or binarization student, respectively
(refer to the [paper](https://arxiv.org/abs/2106.13266) for more details). 

```
python -m vsc.baseline.dns_index \
    --student fine_att --accelerator cuda \
    --query_features ./output/training_queries_dns.npz \
    --ref_features ./output/training_refs_dns.npz \
    --output_path ./output/dns
```

## Matching

The `dns_baseline.py` script performs all phases of matching: retrieval, localization, 
and assigning scores to localized matches.

Retrieval uses the coarse video representations with score normalization to generate candidate pairs 
for the localization process. Again, the [FAISS](https://github.com/facebookresearch/faiss) library 
is used with GPU acceleration if supported.

Localization uses the fine video representations to generate similarity matrices between videos of the candidate pairs,
combined with ones generated with the coarse video representations.
Then, the temporal network flow (TN) method from the [VCSL](https://github.com/alipay/VCSL) library is used for
the final localization of the copied segments.

The baseline script also runs a matching track evaluation, and a descriptor micro AP.
The descriptor micro AP estimates the descriptor track score, but
does not enforce the same limits as the `descriptor_eval.py` evaluation.

The matching script provides several outputs in the `--output_path`
directory, including precision-recall plots, and both retrieved candidate pairs
and localized matches in `.csv` formats.

Run the matching script with one of the two fine-grained students. Pass either `fine_att` or `fine_bin` for 
the `--student` argument providing the corresponding feature file for query and reference videos. 

### Fine-grained Attention Student

```
python -m vsc.baseline.dns_baseline \
    --student fine_att --accelerator cuda \
    --query_fine_features ./output/dns/queries_fg_att_student.npz \
    --ref_fine_features ./output/dns/refs_fg_att_student.npz \
    --query_coarse_features ./output/dns/queries_cg_student_sn.npz \
    --ref_coarse_features ./output/dns/refs_cg_student_sn.npz \
    --ground_truth ./training_dataset/ground_truth.csv \
    --output_path ./output/dns/fg_att_results

...
2022-12-07 16:50:30 INFO     Candidate uAP: 0.4129
2022-12-07 16:50:30 INFO     Matching track metric: 0.3211
```

### Fine-grained Binarization Student

```
python -m vsc.baseline.dns_baseline \
    --student fine_bin --accelerator cuda \
    --query_fine_features ./output/dns/queries_fg_bin_student.npz \
    --ref_fine_features ./output/dns/refs_fg_bin_student.npz \
    --query_coarse_features ./output/dns/queries_cg_student_sn.npz \
    --ref_coarse_features ./output/dns/refs_cg_student_sn.npz \
    --ground_truth ./training_dataset/ground_truth.csv \
    --output_path ./output/dns/fg_bin_results
    
...
2022-12-07 17:03:55 INFO     Candidate uAP: 0.4129
2022-12-07 17:03:55 INFO     Matching track metric: 0.3090
```
