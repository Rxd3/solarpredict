# Day 24 computer-vision dataset cleanup

## Decision

One byte-identical image appeared in both train and test. The training copy was
retained unchanged. The test image and its matching test label were moved—not
deleted—to `data/vision/quarantine/cross_split_duplicate/` because an image
already seen during training cannot be an independent test example.

## Exact duplicate and annotation disagreement

- Image SHA-256: `c6c7d608c8e1ed598fa95fe95a54df7ed5eeaeb5f73fc6bd823d9cab4a2361bc`
- Retained train image:
  `train/images/Physical-damaged-48-_jpg.rf.270e3124c478a269b63d42bb9037807f.jpg`
- Retained train label:
  `train/labels/Physical-damaged-48-_jpg.rf.270e3124c478a269b63d42bb9037807f.txt`
- Train label SHA-256:
  `4ab95a69eb36b1b96068c8dedd65e38c4c3e68abcaef6da5dd991d27e8342221`
- Quarantined test image:
  `images/Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3.jpg`
- Quarantined test label:
  `labels/Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3.txt`
- Test label SHA-256:
  `5ce3c28c8f1fb24f08e62661a6290add1f144b6643c9f5debeb0f86196c8900f`

Both annotation files contain only class ID 0 (`bird-drop`): 12 train boxes and
9 test boxes. There are no exact box-coordinate matches. A best-IoU comparison
found only one test box above 0.5 IoU with any train box (0.764); the other best
matches were at most 0.403, including two at zero. The test annotation is not a
coordinate-identical subset; it is a materially different annotation of the
same pixels. The side-by-side evidence is saved as
`outputs/computer_vision/figures/day24_duplicate_review.png`.

The quarantine README records original/new paths and hashes. No training file,
class mapping, or bounding box was changed.

## Before and after

| Check | Before | Active after quarantine |
| --- | ---: | ---: |
| Train images | 600 | 600 |
| Validation images | 120 | 120 |
| Test images | 101 | 100 |
| Total images | 821 | 820 |
| Label files | 821 | 820 |
| Bounding boxes | 5,760 | 5,751 |
| Cross-split exact image-hash groups | 1 | 0 |

The removed test label contributed nine `bird-drop` boxes. The active
`bird-drop` count is therefore 1,509 boxes in 156 images; all other class counts
are unchanged.

## Empty-label review

All 25 empty-label images are listed with split, path, filename, dimensions, and
status in `outputs/computer_vision/empty_label_review.csv`. Three contact sheets
show every image without fake boxes.

The broad class schema contains `clean`, `dusty`, and `bird-drop`, and visual
review showed solar panels in every image; several show visible dirt or debris.
It is therefore not reliable to assert that any image contains no configured
class. Conservative result:

- `LIKELY_BACKGROUND`: **0**
- `NEEDS_MANUAL_REVIEW`: **25**

No annotation was invented or rewritten. Human approval or source correction of
these empty labels remains the only dataset-quality gate before training.

## Post-cleanup validation

The complete inspector found 600/120/100 active images, zero invalid lines,
zero missing image/label pairs, zero orphan labels, zero unreadable images, zero
duplicate filenames, and zero cross-split duplicate image hashes. The active
ground-truth sample figure was regenerated, and the YOLO dry run succeeded
without calling `model.train()`.

