# System architecture

## Operational anomaly path — prototype complete

```text
Operational Dataset Replay
                  |
                  v
Operational Processing / validated seven-feature rows
                  |
                  v
Frozen Anomaly Detector + Threshold
                  |
                  v
Score / NORMAL / ALERT ----> Exact-2-minute event grouping
                  |
                  v
Streamlit Dashboard
```

`src/anomaly_detection/inference.py` is the boundary for future consumers. It
verifies artifact hashes, validates the exact seven inputs, performs no fitting,
and returns a stable dashboard-friendly schema. `src/dashboard/data_service.py`
validates and caches the recorded 337-row replay, verifies the saved scores
against that inference boundary, and supplies the Overview and Operational
Monitoring pages. Preprocessing and feature preparation remain separate because
monitoring inputs must supply `rtd_mean` and `rtd_std` consistently before
scoring.

## Computer-vision image/video path — prototype complete

```text
Image / Video
        |
        v
Frame / Image Input
        |
        v
Frozen YOLO11n Detector
        |
        v
Bounding Boxes + Class + Confidence
        |
        v
Detection Results / Condition Summary / Detection Log
        |
        v
Streamlit Dashboard (UI integration pending Day 28)
```

The final active six-class dataset, all 795 image/label pairs, and all 5,751 boxes
have been inspected after recoverable duplicate and empty-label quarantine.
Annotation syntax is valid, every class remains present in every split, and the
ground-truth contact sheet is complete. One YOLO11n prototype was selected on
validation and evaluated once on the untouched test split without subsequent
tuning. Read-only image/frame inference, annotated output, cautious condition
summaries, and sequential video processing are now implemented. The video smoke
test used a clearly identified validation-image demo sequence because no real
inspection video was present.

## Remaining integration placeholders

- **Computer-vision model:** retain the frozen checkpoint and one-time test
  results; do not tune from the test split.
- **Drone video data:** obtain representative real footage, retain source-video
  identity, and keep every frame from a video entirely within one dataset split.
- **Dashboard image/video integration:** connect the existing CV inference
  modules to the prepared placeholder pages, with real inputs and the earlier
  validation-image demo clearly distinguished.

The operational dashboard path is functional. Image/video UI integration and
final cross-module testing remain future work.
