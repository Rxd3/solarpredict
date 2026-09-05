# Computer-vision dataset record

## Source and verified configuration

- **Dataset:** [Detection for defects in solar panels](https://universe.roboflow.com/solar-panels-yolo/detection-for-defects-in-solar-panels-9qenv)
- **Creator:** Solar Panels YOLO on Roboflow Universe
- **Version:** 1
- **Task:** object detection
- **Format:** YOLO text annotations compatible with Ultralytics
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Local root:** `data/vision/solar_panel_defects/`
- **YAML:** `data/vision/solar_panel_defects/data.yaml`
- **Train images:** `data/vision/solar_panel_defects/train/images`
- **Validation images:** `data/vision/solar_panel_defects/valid/images`
- **Test images:** `data/vision/solar_panel_defects/test/images`

The downloaded YAML initially used `../train/images`, `../valid/images`, and
`../test/images`, but those locations did not exist; the extracted split folders
are directly beside `data.yaml`. Only these three proven structural paths were
corrected to `train/images`, `valid/images`, and `test/images`. The class mapping
and every annotation file remain unchanged.

Published or redistributed use must credit the dataset title and creator, link
the source and CC BY 4.0 license, state whether changes were made, and avoid
implying creator endorsement.

## Actual class mapping

| ID | Exact `data.yaml` name | Images containing class | Boxes | Share of all boxes |
| -: | --- | ---: | ---: | ---: |
| 0 | `bird-drop` | 157 | 1,518 | 26.35% |
| 1 | `clean` | 192 | 1,107 | 19.22% |
| 2 | `dusty` | 194 | 1,460 | 25.35% |
| 3 | `electrical-damage` | 87 | 207 | 3.59% |
| 4 | `physical-damage` | 61 | 169 | 2.93% |
| 5 | `snow-covered` | 112 | 1,299 | 22.55% |

The largest class by boxes is `bird-drop`; the smallest is
`physical-damage`. Their box-count ratio is 1,518 / 169 = **8.98:1**. This is a
material imbalance to consider during later evaluation, but Day 24 does not
rebalance, oversample, rename, or otherwise modify the data.

## Split inventory

| Split | Images | Image share | Label files | Valid boxes |
| --- | ---: | ---: | ---: | ---: |
| Train | 600 | 73.08% | 600 | 4,359 |
| Validation | 120 | 14.62% | 120 | 761 |
| Test | 101 | 12.30% | 101 | 640 |
| **Total** | **821** | **100.00%** | **821** | **5,760** |

All 821 images are JPEG files. There are 452 unique resolutions. Across the
actual files, width ranges from 149 to 6,240 pixels with median 768; height
ranges from 110 to 5,376 pixels with median 675. The complete resolution
frequency table is retained in
`outputs/computer_vision/dataset_summary.json`.

## Annotation validation

Every non-empty line was checked as:

```text
class_id x_center y_center width height
```

Results:

- invalid annotation lines: **0**;
- images missing a label file: **0**;
- label files without a matching image: **0**;
- unreadable images: **0**;
- empty label files: **25** (18 train, 5 validation, 2 test).

An empty YOLO label is valid for a negative/background image, so these 25 files
are not counted as invalid. Their images should nevertheless be reviewed before
training to confirm that the absence of boxes is intentional rather than an
annotation omission. No label was silently repaired.

## Split-integrity and duplicate review

- Duplicate image filenames across splits: **0**.
- Duplicate label filenames across splits: **0**.
- Exact duplicate image SHA-256 groups across splits: **1**.
- Exact duplicate label-content SHA-256 groups across splits: **2**.

The exact duplicated image (`c6c7d608…a2361bc`) occurs once in train and once
in test under different names:

- train: `Physical-damaged-48-_jpg.rf.270e3124c478a269b63d42bb9037807f.jpg`
- test: `Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3.jpg`

Their label files are not identical: both use class ID 0 (`bird-drop`), but the
train copy has 12 boxes and the test copy has 9 with different coordinates.
This creates direct train/test leakage and inconsistent annotation coverage.
Training should remain blocked until a human decides how to resolve this pair
without randomly redesigning the supplied split.

Of the two cross-split label-content hash groups, one is the shared SHA-256 of
the 25 empty files and does not by itself indicate duplicated images. The other
is one identical non-empty annotation text shared by five different image files
(three train and two validation); identical normalized text alone is not proof
that those images are duplicates. Exact file lists and hashes are recorded in
the machine-readable summary.

The original split is otherwise preserved. Future frames extracted from any one
source video must all remain in exactly one split to prevent temporal leakage.

## Reproducible commands

```powershell
.\.venv\Scripts\python.exe -m src.computer_vision.inspect_vision_dataset
.\.venv\Scripts\python.exe -m src.computer_vision.visualize_annotations
.\.venv\Scripts\python.exe -m src.computer_vision.train_detector --dry-run
```

