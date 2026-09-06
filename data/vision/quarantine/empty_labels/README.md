# Empty-label quarantine

This directory contains 25 recoverable image/label pairs removed from the active
Roboflow train, validation, and test splits during the Day 24 dataset-integrity
review. The original split structure is retained under `train/`, `valid/`, and
`test/`.

All 25 images visibly contain solar panels, while their matching YOLO label files
are empty. Because `clean` is an object class in this dataset, these files cannot
be safely interpreted as intentional background/negative examples without
confirmation from the source publisher. They were therefore excluded from active
training rather than assigned invented boxes or classes.

No quarantined annotation was rewritten and no file was permanently deleted.
`manifest.csv` records each original path, quarantine path, original split, image
and label SHA-256 digest, and the exact quarantine reason. These files must remain
outside active splits unless corrected annotations or authoritative source
information becomes available.

Quarantined pairs by original split:

| Original split | Pairs |
| --- | ---: |
| Train | 18 |
| Validation (`valid/`) | 5 |
| Test | 2 |
| **Total** | **25** |
