# Computer-vision setup

## Objective and completed Day 24 scope

The module will locate visible solar-panel conditions in images and return YOLO
bounding boxes, class names, and confidence values. Day 24 now includes the real
dataset inventory, exhaustive annotation checks, class-distribution review,
ground-truth visualization, training dry run, and future evaluation interface.
It does not fit a model, process video, perform trained-model inference, or build
the dashboard.

The active version 1 dataset has 820 images, 820 label files, six classes,
and 5,751 valid boxes after quarantine. Its exact class mapping and split findings are documented
in [computer_vision_dataset.md](computer_vision_dataset.md).

## Verified environment and dry run

| Item | Verified result |
| --- | --- |
| Python | 3.12.4 |
| PyTorch | 2.14.0+cpu |
| Ultralytics | 8.4.138 |
| CUDA available | No |
| GPU | None exposed to PyTorch |
| Pretrained checkpoint | `yolo11n.pt`, resolved successfully |
| Dry-run dataset paths | Train, validation, and test all exist |
| Dry-run class mapping | All six names loaded |
| `model.train()` called | No |

The real dry run completed successfully with no errors and reported
`ready_for_training: true` for technical configuration/path checks. It does not
assess semantic label quality; the cross-split duplicate has been removed from
the active test split and preserved in recoverable quarantine.

## CPU feasibility and Day 25 recommendation

The prepared configuration is YOLO11n, 50 epochs, 640-pixel input, batch 8,
patience 10, CPU, workers 0, and seed 42. It is technically feasible for 600
training images, but CPU-only training at 640 pixels for 50 epochs is likely to
be unnecessarily slow. The source resolutions are unusually heterogeneous
(452 unique sizes, up to thousands of pixels per side), although Ultralytics
will resize/letterbox inputs to the configured training size.

One conservative Day 25 recommendation is:

| Setting | Recommended value |
| --- | ---: |
| Model | `yolo11n.pt` |
| Epochs | 30 |
| Image size | 512 |
| Batch size | 8 |
| Patience | 8 |
| Device | CPU |
| Workers | 0 |
| Seed | 42 |

This retains every source image and annotation while reducing nominal
epoch-and-pixel work to roughly 38% of the current 50×640 plan. Batch 8 is kept;
reduce it only if a later monitored run demonstrates memory pressure. No
training was executed to benchmark runtime, so this remains an engineering
estimate rather than a measured duration.

## Ground-truth visualization

`outputs/computer_vision/figures/day24_dataset_samples.png` was generated from
real annotations and visually checked. The nine-image contact sheet includes
examples covering all six YAML classes. It shows ground truth only, not model
predictions.

## Pre-training quality gate and limitations

Before model fitting:

1. review the 25 empty label files to confirm they are intentional negative
   samples;
2. rerun the inspector, visualization, dry run, and complete tests if any
   human-approved dataset change is made; and
3. record the approved Day 25 CPU configuration.

The 8.93:1 largest/smallest box imbalance must be reflected in later per-class
evaluation. Generalization to drone altitude, glare, motion, new sites, and new
panel types remains unverified. The original split should otherwise remain
unchanged, and all future frames from one source video must inherit one split.
