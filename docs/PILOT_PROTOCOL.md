# 10% ROI-disjoint pilot protocol

This branch adds a fixed fast-screening protocol for SEN12MS-CR experiments.
The audited full-data trainer is left unchanged.

## Purpose

The standard pilot uses:

- 10% of the official UnCRtainTS train split;
- 10% of the official UnCRtainTS validation split;
- 10% of the official UnCRtainTS test split;
- 25 training epochs;
- the same model, optimizer, bridge schedule, batch size, and metric code as the
  controlled DB-CR baseline unless an experiment explicitly changes one of
  those factors.

Relative training work versus a 100%-data / 50-epoch run is approximately

    0.10 * 25 / 50 = 0.05

or about 1/20 of the full training work.

## Geographic and seasonal constraints

The pilot never constructs a new random train/val/test split.

First, the complete dataset is partitioned using the frozen UnCRtainTS
ROI-disjoint split. Only then is each split downsampled independently.

Downsampling is stratified by the season-scoped ROI key:

    <season>_s1/s1_<roi>

For each train/val/test split:

1. compute a 10% target count;
2. allocate that target proportionally across all ROI groups;
3. require at least one retained patch from every ROI group;
4. choose patches inside each ROI by a deterministic SHA-256 ranking using
   subset seed 42.

This preserves the original train/val/test ROI disjointness, retains all ROI
groups represented by each split, and approximately preserves the full
seasonal distribution instead of allowing the pilot to collapse onto only one
or two seasons.

With the currently frozen reference split, the expected sample counts are:

| split | full | pilot |
|---|---:|---:|
| train | 107,142 | 10,714 |
| val | 7,176 | 718 |
| test | 7,899 | 790 |

The ROI-group counts remain 155 / 10 / 10 for train / val / test.

## Training

Example Original DB-CR pilot:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mscloudr.cli.pilot_train \
  --data-root /path/to/SEN12MS-CR \
  --run-name DBCR_Original_pilot10_seed42 \
  --schedule original \
  --device cuda \
  --batch-size 4 \
  --num-workers 4
```

The default is 25 epochs. To run the matched MR_r3 control:

```bash
CUDA_VISIBLE_DEVICES=1 python -m mscloudr.cli.pilot_train \
  --data-root /path/to/SEN12MS-CR \
  --run-name DBCR_MR_r3_pilot10_seed42 \
  --schedule mr_r3 \
  --device cuda \
  --batch-size 4 \
  --num-workers 4
```

Each run writes:

```text
<run-dir>/run_config.json
<run-dir>/pilot_subset_manifest.json
<run-dir>/checkpoints/...
```

`pilot_subset_manifest.json` stores the exact selected sample IDs plus counts,
season coverage, ROI counts, and SHA-256 fingerprints. The compact fingerprint
and split audit are also saved in checkpoint provenance.

Pilot checkpoints are intentionally marked `paper_grade=false`; they are for
screening experimental directions, not final paper tables.

## Evaluation

Evaluate the matched 10% held-out test subset with:

```bash
python -m mscloudr.cli.pilot_eval \
  --checkpoint outputs/DBCR_Original_pilot10_seed42/checkpoints/best_endpoint.pt \
  --data-root /path/to/SEN12MS-CR \
  --device cuda
```

The default output is:

```text
<run-dir>/pilot_eval_metrics.json
```

The evaluator reconstructs the same deterministic pilot subset and rejects the
checkpoint if the sample-ID fingerprint or compact split audit differs.

## Experimental rule

Compare pilot methods only against a baseline trained on this same pilot
protocol. Do not compare a 10%-trained experimental checkpoint directly against
a 100%-trained baseline checkpoint.
