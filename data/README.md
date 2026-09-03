# Project data

The project uses two selected categories of data. Keep original files unchanged,
retain source and license information, and write cleaned or derived data to
`processed/` rather than overwriting raw inputs. Dataset contents are ignored by
Git; this file records the project-level provenance and handling rules.

## A. Photovoltaic operational/time-series data

The downloaded version 1 files are stored under
`operational/solar-power-dataset/`. This raw directory is excluded from Git.

- **Dataset:** [Solar Power Dataset](https://www.kaggle.com/datasets/s1nister/solar-power-generation-dataset)
- **Publisher:** `s1nister` on Kaggle
- **License:** CC0 / Public Domain
- **Observed sampling information:** exactly two minutes between every
  consecutive timestamp in the downloaded CSV
- **Role in this project:** initial source of normal operational observations

The dataset is described as containing environmental and solar measurements,
including solar radiation, temperature, humidity, and wind information, as well
as electrical measurements used to determine generated power. This description
does not establish exact column names or units.

The inspected file is `solar_data.csv` (1,009 rows x 14 columns). Its complete
verified schema, source hash, units, ranges, and quality findings are maintained
in [`docs/operational_dataset.md`](../docs/operational_dataset.md). To inspect it
again, run:

```bash
python src/data_processing/inspect_dataset.py data/operational/solar-power-dataset
```

The downloaded CSV remains the authority. If a later Kaggle version is obtained,
repeat the inspection and update the verified schema rather than assuming its
columns match version 1.

### Timestamp policy

The source does not clearly specify a timezone. Preserve timestamps as
timezone-naive and describe the timezone in reports and metadata as **"not
specified by source"**. Do not assume UTC, localize timestamps, or assign a
geographical timezone without additional authoritative metadata. The reported
approximately two-minute source description was checked: all consecutive values
in the downloaded version 1 file are exactly 120 seconds apart.

### Future synthetic anomalies

Treat the selected operational dataset as normal data at the initial modeling
stage. Synthetic anomalies may be introduced later for controlled testing, but
their generation must be reproducible: use documented parameters and random
seeds, retain a clean copy of the source data, and store anomaly labels and
generation provenance separately. No synthetic anomalies have been generated
yet.

## B. Solar-panel images/video

Store the downloaded export in `images/`, retaining its original directory and
split structure.

- **Dataset:** [Detection for defects in solar panels](https://universe.roboflow.com/solar-panels-yolo/detection-for-defects-in-solar-panels-9qenv)
- **Creator:** Solar Panels YOLO
- **Platform:** Roboflow Universe
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Task:** Object detection
- **Required export format:** YOLO annotations compatible with Ultralytics

The source currently presents relevant conditions such as `clean`, `dusty`,
`bird-drop`, `electrical-damage`, `physical-damage`, and `snow-covered`. Do not
rename, normalize, merge, or reorder classes until the downloaded `data.yaml`
has been inspected. For the version actually downloaded, record:

- the `names` mapping and class order from `data.yaml`;
- the `train`, `val`, and `test` paths that are present;
- the image and label counts in each supplied split;
- missing, empty, malformed, or out-of-range label files;
- export/version details and any preprocessing recorded by the source.

The downloaded `data.yaml` and annotation files are authoritative. The class
list above is contextual information, not a claim about an uninspected export.

### CC BY 4.0 attribution

When the dataset or derived material is published, presented, or redistributed,
provide appropriate credit, a source link, a CC BY 4.0 license link, and a clear
indication of modifications. A suitable project attribution is:

> Detection for defects in solar panels, created by Solar Panels YOLO and
> sourced from Roboflow Universe, licensed under CC BY 4.0. Changes: [describe
> changes, or state none].

Do not imply endorsement by the creator. Keep this attribution in reports,
presentations, dataset documentation, and other distributed artifacts that use
the licensed material.

### Image and video split policy

- Preserve the original train/validation/test split supplied by the Roboflow
  export whenever available.
- Do not randomly reshuffle the supplied images across splits.
- Do not randomly mix extracted video frames across splits.
- For future drone footage, assign each source video wholly to one split:
  train, validation, or test. Every extracted frame must inherit its source
  video's split.
- Retain a video identifier in future frame manifests so this grouping can be
  audited and temporal/data leakage can be prevented.

## Processed data

Use `processed/` for reproducibly generated outputs such as cleaned operational
tables, validated annotations, dataset manifests, or extracted frame indexes.
Generated data should be recreated by scripts whenever practical. Processed
manifests should preserve links to source files, original splits, licenses, and
transformation history.

Large and potentially private data files are ignored by Git by default. Only
this documentation and the empty-directory placeholders are committed.
