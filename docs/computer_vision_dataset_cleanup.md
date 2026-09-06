# Day 24 computer-vision dataset cleanup

## Conservative decisions

Two recoverable cleanup actions were completed. No image or label was permanently
deleted, no box or class was invented, and the supplied split membership of all
remaining active files was preserved.

1. One byte-identical image appeared in train and test. The training pair was
   retained and the test image plus its differing nine-box label were moved to
   `data/vision/quarantine/cross_split_duplicate/` to remove train/test leakage.
2. The 25 reviewed empty-label images all visibly contain solar panels. Since
   `clean` is a detectable object class and the publisher has not confirmed these
   as intentional negatives, each image and its matching empty label were moved
   to `data/vision/quarantine/empty_labels/`.

The empty-label manifest records original/quarantine paths, original split,
image and label SHA-256 digests, and this exact reason:

> Visible solar panels with empty annotation; cannot safely treat as background because clean is an object class.

## Exact duplicate audit

- Image SHA-256: `c6c7d608c8e1ed598fa95fe95a54df7ed5eeaeb5f73fc6bd823d9cab4a2361bc`
- Retained train image: `train/images/Physical-damaged-48-_jpg.rf.270e3124c478a269b63d42bb9037807f.jpg`
- Quarantined test image: `images/Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3.jpg`
- Quarantined test label SHA-256: `5ce3c28c8f1fb24f08e62661a6290add1f144b6643c9f5debeb0f86196c8900f`

The retained train label has 12 `bird-drop` boxes; the former test label has nine
different `bird-drop` boxes. The comparison figure is
`outputs/computer_vision/figures/day24_duplicate_review.png`.

## Empty-label split impact

| Original split | Pairs quarantined |
| --- | ---: |
| Train | 18 |
| Validation | 5 |
| Test | 2 |
| **Total** | **25** |

## Dataset progression

| Check | Original export | After duplicate quarantine | Final active dataset |
| --- | ---: | ---: | ---: |
| Train images | 600 | 600 | 582 |
| Validation images | 120 | 120 | 115 |
| Test images | 101 | 100 | 98 |
| Total images/labels | 821 | 820 | 795 |
| Bounding boxes | 5,760 | 5,751 | 5,751 |
| Empty labels | 25 | 25 | 0 |
| Cross-split exact image-hash groups | 1 | 0 | 0 |

Empty label files contain no boxes, so their quarantine reduced active pair
counts but did not change any class or bounding-box count.

## Final validation

The final active dataset has zero empty labels, missing labels, orphan labels,
unreadable images, invalid YOLO lines, cross-split duplicate image hashes, and
duplicate filenames across splits. All six configured classes remain present in
train, validation, and test. The inspector's training quality gate is open.

