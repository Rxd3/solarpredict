# Computer-vision dataset record

## Source and verified configuration

- **Dataset:** [Detection for defects in solar panels](https://universe.roboflow.com/solar-panels-yolo/detection-for-defects-in-solar-panels-9qenv)
- **Creator:** Solar Panels YOLO on Roboflow Universe
- **Downloaded version:** 1
- **Task:** object detection
- **Format:** YOLO text annotations compatible with Ultralytics
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Local root:** `data/vision/solar_panel_defects/`
- **YAML:** `data/vision/solar_panel_defects/data.yaml`

The export's class names were retained exactly. Only three invalid relative paths
in the downloaded `data.yaml` were corrected to the locally verified
`train/images`, `valid/images`, and `test/images` locations. Attribution must
include the dataset title and creator, source and license links, a statement of
changes, and no implication of creator endorsement.

## Final active inventory

| Split | Images | Share | Labels | Boxes |
| --- | ---: | ---: | ---: | ---: |
| Train | 582 | 73.21% | 582 | 4,359 |
| Validation | 115 | 14.47% | 115 | 761 |
| Test | 98 | 12.33% | 98 | 631 |
| **Total** | **795** | **100.00%** | **795** | **5,751** |

## Actual class mapping and distribution

| ID | Exact `data.yaml` name | Images containing class | Boxes | Box share |
| -: | --- | ---: | ---: | ---: |
| 0 | `bird-drop` | 156 | 1,509 | 26.24% |
| 1 | `clean` | 192 | 1,107 | 19.25% |
| 2 | `dusty` | 194 | 1,460 | 25.39% |
| 3 | `electrical-damage` | 87 | 207 | 3.60% |
| 4 | `physical-damage` | 61 | 169 | 2.94% |
| 5 | `snow-covered` | 112 | 1,299 | 22.59% |

`bird-drop` is the largest class by box count and `physical-damage` the
smallest, a remaining ratio of **8.93:1**. No oversampling, rebalancing, class
renaming, or split reassignment was performed.

## Per-split class coverage

Each cell is `images / boxes`.

| Class | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `bird-drop` | 109 / 1,132 | 19 / 160 | 28 / 217 |
| `clean` | 135 / 834 | 38 / 193 | 19 / 80 |
| `dusty` | 165 / 1,255 | 7 / 53 | 22 / 152 |
| `electrical-damage` | 60 / 131 | 17 / 35 | 10 / 41 |
| `physical-damage` | 41 / 101 | 13 / 40 | 7 / 28 |
| `snow-covered` | 78 / 906 | 22 / 280 | 12 / 113 |

All six classes remain represented in every split. The two minority classes
have enough labeled training support for a prototype fine-tuning attempt
(`electrical-damage`: 60 images/131 boxes; `physical-damage`: 41 images/101
boxes), but their low counts and the 8.93:1 imbalance require per-class metrics
and cautious conclusions. Adequacy for production cannot be established before
training and independent evaluation.

## Final integrity checks

- Empty labels: **0**
- Images without labels: **0**
- Labels without images: **0**
- Unreadable images: **0**
- Invalid YOLO lines: **0**
- Cross-split exact image-hash groups: **0**
- Duplicate filenames across splits: **0**
- Classes missing from any split: **0**

The one cross-split duplicate and all 25 uncertain empty-label pairs remain
recoverable in quarantine. See
[computer_vision_dataset_cleanup.md](computer_vision_dataset_cleanup.md) and
`data/vision/quarantine/empty_labels/manifest.csv`.

The original split is otherwise preserved. Every frame extracted from one future
source video must remain entirely in one split to prevent temporal leakage.

## Reproducible checks

```powershell
.\.venv\Scripts\python.exe -m src.computer_vision.inspect_vision_dataset
.\.venv\Scripts\python.exe -m src.computer_vision.visualize_annotations
.\.venv\Scripts\python.exe -m src.computer_vision.train_detector --dry-run
```
