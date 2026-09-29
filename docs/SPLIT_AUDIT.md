# SEN12MS-CR split audit

The clean paper protocol uses the ROI-level split encoded in the public
UnCRtainTS loader (`PatrickTUM/UnCRtainTS`, commit
`5e1f1b58e993645e765b64e10b6e9c7ff828b36f`).

The split contains 175 geographically separated ROI groups:

- train: 155 ROI groups
- validation: 10 ROI groups
- test: 10 ROI groups

On the audited local SEN12MS-CR installation, after excluding the one versioned
known-invalid sample, the exact counts are:

- train: 107,142
- validation: 7,176
- test: 7,899
- total: 122,217

The ignored sample belongs to the training split. Therefore the corresponding
full-dataset counts before exclusion are:

- train: 107,143
- validation: 7,176
- test: 7,899
- total: 122,218

This reconciles the dataset total and the reference ROI split.

## DB-CR paper count discrepancy

DB-CR states that SEN12MS-CR contains 122,218 patches and reports split counts
of 114,056 / 7,176 / 7,899. Those printed counts sum to 129,131 and therefore
cannot all be correct.

The validation and test counts exactly match the audited UnCRtainTS ROI split.
The audited split plus the single known-invalid training sample implies a
training count of 107,143 for the complete dataset, not 114,056.

For this repository, we therefore treat the public UnCRtainTS ROI definitions
as the executable source of truth, and record the DB-CR training count as a
paper-text inconsistency rather than reproducing it literally.

## Verified fingerprints

See `manifests/sen12mscr_reference_split_audit_v1.json` for the exact sample
counts and SHA-256 fingerprints recorded from the audited installation.
