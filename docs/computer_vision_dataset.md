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
| 0 | `bird-drop` | 156 | 1,509 | 26.24% |
| 1 | `clean` | 192 | 1,107 | 19.25% |
| 2 | `dusty` | 194 | 1,460 | 25.39% |
| 3 | `electrical-damage` | 87 | 207 | 3.60% |
| 4 | `physical-damage` | 61 | 169 | 2.94% |
| 5 | `snow-covered` | 112 | 1,299 | 22.59% |

The largest class by boxes is `bird-drop`; the smallest is
`physical-damage`. Their box-count ratio is 1,509 / 169 = **8.93:1**. This is a
material imbalance to consider during later evaluation, but Day 24 does not
rebalance, oversample, rename, or otherwise modify the data.

## Split inventory

| Split | Images | Image share | Label files | Valid boxes |
| --- | ---: | ---: | ---: | ---: |
| Train | 600 | 73.17% | 600 | 4,359 |
| Validation | 120 | 14.63% | 120 | 761 |
| Test | 100 | 12.20% | 100 | 631 |
| **Total** | **820** | **100.00%** | **820** | **5,751** |

All 820 active images are JPEG files. There are 452 unique resolutions. Across the
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
are not counted as invalid. Visual review could not confidently exclude the six
broad configured classes from any of them; all 25 are marked
`NEEDS_MANUAL_REVIEW`. No label was silently repaired.

## Split-integrity and duplicate review

- Duplicate image filenames across splits: **0**.
- Duplicate label filenames across splits: **0**.
- Exact duplicate image SHA-256 groups across active splits: **0** after quarantine.
- Exact duplicate label-content SHA-256 groups across splits: **2**.

The previously duplicated image (`c6c7d608…a2361bc`) occurred once in train and
once in test under different names:

- train: `Physical-damaged-48-_jpg.rf.270e3124c478a269b63d42bb9037807f.jpg`
- former test: `Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3.jpg`

Their label files are not identical: both use class ID 0 (`bird-drop`), but the
train copy has 12 boxes and the test copy has 9 with different coordinates.
This created direct train/test leakage and inconsistent annotation coverage.
The training copy was retained, while the test image and its nine-box label were
moved intact to recoverable quarantine. See
[computer_vision_dataset_cleanup.md](computer_vision_dataset_cleanup.md).

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
