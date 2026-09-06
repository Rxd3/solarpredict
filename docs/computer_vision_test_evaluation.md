# Independent computer-vision test evaluation

## Evaluation boundary

The frozen checkpoint `models/computer_vision/solar_panel_detector_best.pt`
(SHA-256 `b7ad8d7fca947527b30fb24f8c11e9135fbe62aa1e7f44e8a54100933141cd89`)
was selected from validation results before this evaluation. The exact settings
were recorded in `config/computer_vision_test_evaluation.yaml` before any test
metrics were calculated.

**The test split was evaluated once after model selection and was not used for
model tuning.** No checkpoint, annotation, split, confidence threshold, IoU
threshold, or training parameter was changed in response. Evaluation used all
98 active test images, image size 512, batch size 8, CPU, and the normal
Ultralytics validation confidence/IoU behavior (neither threshold was
overridden).

## Test results

| Scope | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.3878 | 0.2356 | 0.2183 | 0.1062 |
| bird-drop | 0.2457 | 0.1567 | 0.1299 | 0.0399 |
| clean | 0.3393 | 0.7875 | 0.6017 | 0.3634 |
| dusty | 0.5256 | 0.0221 | 0.0910 | 0.0338 |
| electrical-damage | 0.2922 | 0.1009 | 0.0893 | 0.0317 |
| physical-damage | 0.6682 | 0.1429 | 0.2361 | 0.1138 |
| snow-covered | 0.2557 | 0.2035 | 0.1616 | 0.0548 |

The test split contains 631 labelled instances. The saved confidence-0.25
prediction pass produced 324 boxes across its 98 images; this count is a
viewing/inference artifact, not an alternate test metric or tuned operating
point.

## Validation versus test

| Metric | Validation | Test | Test − validation |
| --- | ---: | ---: | ---: |
| Precision | 0.3083 | 0.3878 | +0.0795 |
| Recall | 0.2462 | 0.2356 | −0.0106 |
| mAP50 | 0.2004 | 0.2183 | +0.0179 |
| mAP50–95 | 0.0807 | 0.1062 | +0.0256 |

Overall test performance is broadly similar to validation: precision and both
AP summaries are modestly higher, while recall is essentially unchanged and
slightly lower. This does not establish production generalization.

- `clean` changes most positively (test recall +0.3264 and mAP50 +0.2803).
- `bird-drop` changes most negatively in recall (−0.2121), and its mAP50 falls
  by 0.0998.
- `electrical-damage` mAP50 falls by 0.1072 and recall is only 0.1009.
- `dusty` remains especially difficult: its test recall is 0.0221 despite
  precision of 0.5256, meaning very few labelled dusty instances are detected.
- The lower-support `electrical-damage` and `physical-damage` classes remain
  limited. `electrical-damage` has the lowest test mAP50–95 (0.0317), while
  `physical-damage` retains low recall (0.1429).
- `clean` is strongest by mAP50 and mAP50–95. The confusion matrix also shows a
  large background/missed-detection count across the non-clean classes.

These observations are descriptive only. They did not trigger retraining,
threshold selection, checkpoint replacement, or annotation changes.

## Preserved artifacts

- Machine-readable metrics and confusion counts:
  `outputs/computer_vision/test_evaluation.json`
- Confidence-0.25 per-image predictions:
  `outputs/computer_vision/test_prediction_summary.json`
- Ultralytics confusion matrices, PR/P/R/F1 curves, and labelled/predicted
  batches: `outputs/computer_vision/evaluation/test_evaluation/`
- Representative documentation figure:
  `outputs/computer_vision/figures/day26_test_predictions.png`

The documentation figure reads the saved prediction summary rather than
rerunning the test evaluator. It covers all six ground-truth classes and
deliberately includes missed cases. Its internal title contains no internship
day number.

## Limitations

This is a small, imbalanced prototype dataset and a lightweight CPU-trained
model. Absence of a detection must be reported as `NO_DETECTION`, never as proof
that a panel is healthy or fault-free. The results do not validate performance
on new sites, real drone footage, different cameras, or changing environmental
conditions.
