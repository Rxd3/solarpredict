# Quarantined cross-split duplicate

The test image and label below were moved out of the active dataset on Day 24.
The image bytes duplicate a retained training image, so the test copy cannot be
an independent evaluation example. The files were moved rather than deleted and
their contents were not changed.

## Quarantined test pair

- Original image: `data/vision/solar_panel_defects/test/images/Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3.jpg`
- New image: `data/vision/quarantine/cross_split_duplicate/images/Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3.jpg`
- Image SHA-256: `c6c7d608c8e1ed598fa95fe95a54df7ed5eeaeb5f73fc6bd823d9cab4a2361bc`
- Original label: `data/vision/solar_panel_defects/test/labels/Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3.txt`
- New label: `data/vision/quarantine/cross_split_duplicate/labels/Bird-184-_jpg.rf.4929f26fc5a5f175af527f49cefe25b3.txt`
- Label SHA-256: `5ce3c28c8e1fb24f08e62661a6290add1f144b6643c9f5debeb0f86196c8900f`
- Label contents: 9 boxes, all class ID 0 (`bird-drop`).

## Retained training pair

- Image: `data/vision/solar_panel_defects/train/images/Physical-damaged-48-_jpg.rf.270e3124c478a269b63d42bb9037807f.jpg`
- Image SHA-256: `c6c7d608c8e1ed598fa95fe95a54df7ed5eeaeb5f73fc6bd823d9cab4a2361bc`
- Label: `data/vision/solar_panel_defects/train/labels/Physical-damaged-48-_jpg.rf.270e3124c478a269b63d42bb9037807f.txt`
- Label SHA-256: `4ab95a69eb36b1b96068c8dedd65e38c4c3e68abcaef6da5dd991d27e8342221`
- Label contents: 12 boxes, all class ID 0 (`bird-drop`).

The class IDs agree, but the box sets do not: there are zero exact box matches,
and only one of the nine test boxes has a best train-box IoU above 0.5. The test
annotation is therefore not a coordinate-identical subset of the train
annotation. No box was selected, combined, invented, or rewritten.

