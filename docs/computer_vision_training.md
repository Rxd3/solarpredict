# Day 25 computer-vision prototype training

## Objective and controls

One YOLO11n object-detection model was fine-tuned from `yolo11n.pt` on the final
582-image training split. Ultralytics used the 115-image validation split during
training and for model selection. The 98-image test split was not evaluated or
used for selection. No split, class, image, or annotation was changed.

The active dataset contains 795 images, 5,751 boxes, and the six verified classes:
`bird-drop`, `clean`, `dusty`, `electrical-damage`, `physical-damage`, and
`snow-covered`.

## Configuration and runtime

| Setting | Value |
| --- | ---: |
| Pretrained model | `yolo11n.pt` |
| Epochs requested | 30 |
| Epochs completed | 28 |
| Best epoch | 20 |
| Image size | 512 |
| Batch size | 8 |
| Patience | 8 |
| Device / workers | CPU / 0 |
| Random seed | 42 |
| Observed wall-clock run time | 3,290.3 s (54 min 50.3 s) |
| Ultralytics cumulative training time | 3,267.9 s (54 min 27.9 s) |
| Average per completed epoch | 116.7 s |

Training stopped normally after epoch 28 because validation fitness had not
improved for eight epochs after the epoch-20 best. It was not restarted. CPU
throughput varied during the run, but there was no technical failure.

`best.pt`, `last.pt`, and periodic checkpoints at epochs 5, 10, 15, 20, and 25
are preserved in the original Ultralytics run directory. The selected best
checkpoint was copied to `models/computer_vision/solar_panel_detector_best.pt`;
the original was not overwritten. Its SHA-256 is
`b7ad8d7fca947527b30fb24f8c11e9135fbe62aa1e7f44e8a54100933141cd89`.

## Explicit validation-only results

These metrics come from reloading the selected `best.pt` and running the
evaluation interface on `split=val` only.

| Scope | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.3083 | 0.2462 | 0.2004 | 0.0807 |
| bird-drop | 0.2888 | 0.3688 | 0.2296 | 0.0618 |
| clean | 0.2788 | 0.4611 | 0.3214 | 0.1839 |
| dusty | 0.1781 | 0.0189 | 0.0283 | 0.0059 |
| electrical-damage | 0.1744 | 0.2000 | 0.1965 | 0.0710 |
| physical-damage | 0.6901 | 0.1676 | 0.2284 | 0.0947 |
| snow-covered | 0.2394 | 0.2607 | 0.1981 | 0.0668 |

By mAP50-95, `clean` is strongest and `physical-damage` is second, although the
latter's high precision accompanies very low recall. `dusty` is clearly weakest;
`bird-drop` is second-lowest by mAP50-95. `electrical-damage` also has weak
precision and recall. These findings are consistent with the known imbalance
and limited minority-class training support.

## Curves and confusion behavior

Training box and classification loss continued to decline through epoch 28.
Validation losses declined overall but became noisy and roughly plateaued late
in the run. mAP rose strongly from the early epochs, peaked in model-selection
fitness at epoch 20, and did not sustain a later improvement. The curves do not
show a dramatic loss divergence, but the late training/validation separation
and plateau justify the selected earlier checkpoint and early stop.

The validation confusion matrix is dominated by missed objects classified as
background: 118/160 bird-drop, 112/193 clean, 53/53 dusty, 28/35 electrical,
28/40 physical, and 218/280 snow boxes at the matrix operating point. Among
minority classes, five physical-damage boxes were predicted as
electrical-damage and four as bird-drop; only three were counted as correct.
Electrical damage had seven correct and 28 missed. False-positive detections on
background are also substantial. This is weak prototype behavior, not a basis
for changing the class definitions today.

The six-image validation prediction figure uses solid predicted boxes with
class/confidence labels and white dashed ground truth. It covers all six ground-
truth classes and uses no test image.

## Artifacts and limitations

- Original run: `outputs/computer_vision/training/yolo11n_solar_panel_defects_prototype/`
- Selected checkpoint: `models/computer_vision/solar_panel_detector_best.pt`
- Metadata: `models/computer_vision/solar_panel_detector_metadata.json`
- Machine-readable summary: `outputs/computer_vision/training_summary.json`
- Validation metrics: `outputs/computer_vision/validation_metrics.json`
- Validation prediction records: `outputs/computer_vision/validation_prediction_summary.json`
- Sample figure: `outputs/computer_vision/figures/day25_validation_predictions.png`

The class imbalance remains 8.93:1 by box count. The small validation support for
some classes—especially seven dusty images—also makes per-class metrics unstable.
Independent test evaluation, a reusable image-inference interface, drone/video
processing, and dashboard integration remain pending.

**Validation results describe the current prototype and do not establish
production-level defect-detection performance.**
