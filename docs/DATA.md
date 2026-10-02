# SEN12MS-CR data handling

The SEN12MS-CR imagery is **not stored in this Git repository**.  The dataset is
hundreds of gigabytes and should remain in a shared/external data directory.

## Recommended local setup

Point the code at the existing dataset using an environment variable:

```bash
export MSCLOUDR_DATA_ROOT='/path/to/Cloud Removal/data'
```

The resolver accepts either:

- the directory that directly contains `ROIs2017_winter_s2_cloudy`, etc.; or
- its parent when the dataset lives in a `SEN12MS-CR/` child directory.

For the current HPDRC-style checkout, a shell configuration can therefore use:

```bash
export MSCLOUDR_DATA_ROOT='/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Removal/data'
```

Quotes are required because the path contains spaces and parentheses.

Do **not** copy or move the dataset into the Git repository.

## Optional symlink

A local convenience symlink is also safe because `data/` is gitignored:

```bash
cd '/aul/homes/mmazi007/Desktop/Source Code (Research)/mscloudr'
mkdir -p data
ln -s '/aul/homes/mmazi007/Desktop/Source Code (Research)/Cloud Removal/data' \
  data/external
```

The environment-variable approach is preferred because it is explicit and
portable across machines.

## Audit the installation

After installing the package in editable mode:

```bash
cd '/aul/homes/mmazi007/Desktop/Source Code (Research)/mscloudr'
python -m pip install -e .
python -m mscloudr.cli.audit_dataset \
  --output outputs/dataset_audit.json \
  --write-sample-ids outputs/sen12mscr_sample_ids.txt
```

The audit walks filenames and counterpart paths only; it does **not** load the
hundreds of gigabytes of TIFF data into memory or copy them.

The output reports:

- total discovered aligned triplets;
- counts by season;
- missing S1/S2 counterparts;
- known ignored/corrupt samples;
- a SHA-256 fingerprint of the ordered sample IDs.

The sample-ID list can later be used to create a frozen train/validation/test
manifest.

## Stable identities

Split manifests use dataset-relative cloudy paths such as:

```text
ROIs1868_summer_s2_cloudy/s2_cloudy_146/ROIs1868_summer_s2_cloudy_146_p202.tif
```

They never store `/aul/...` absolute paths.  This allows the same manifest to be
used on another server where the dataset lives somewhere else.

## Preprocessing currently preserved from the historical implementation

Optical S2:

```text
clip to [0, 10000] -> divide by 10000 -> [0, 1]
```

SAR:

```text
VV: clip [-25, 0]   -> [0, 1]
VH: clip [-32.5, 0] -> [0, 1]
```

These constants are currently migrated for historical reproducibility.  Their
benchmark provenance will be checked separately before the final paper
protocol is frozen.
