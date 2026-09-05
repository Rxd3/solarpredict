# System architecture

## Operational anomaly path — prototype complete

```text
Operational CSV / future monitoring row
                  |
                  v
Conservative preprocessing (raw source retained)
                  |
                  v
Feature preparation (exact NO_TIME_7 contract)
                  |
                  v
Frozen training-only StandardScaler
                  |
                  v
Frozen Isolation Forest
                  |
                  v
Continuous anomaly score (-score_samples)
                  |
                  v
Frozen threshold (strict score > 0.7135242760182695)
                  |
                  v
NORMAL / ALERT ----+----> Exact-2-minute event grouping
                  |
                  v
Future dashboard / event log
```

`src/anomaly_detection/inference.py` is the boundary for future consumers. It
verifies artifact hashes, validates the exact seven inputs, performs no fitting,
and returns a stable dashboard-friendly schema. Preprocessing and feature
preparation remain separate because monitoring inputs must supply `rtd_mean` and
`rtd_std` consistently before scoring.

## Computer-vision image path — dataset validated, training pending

```text
Solar Panel Image
        |
        v
YOLO Object Detector
        |
        v
Bounding Boxes + Class + Confidence
        |
        v
Inspection Result
        |
        v
Future Dashboard
```

The active six-class dataset, all 820 image/label pairs, and all 5,751 boxes have
been inspected after recoverable duplicate quarantine. Annotation syntax is
valid and the ground-truth contact sheet is complete. Training/evaluation entry
points are prepared; 25 empty labels remain queued for human approval.

## Remaining integration placeholders

- **Computer-vision model:** obtain human approval for empty-label handling,
  then train and evaluate with per-class reporting that reflects the imbalance.
- **Drone video processing:** retain source-video identity and keep every frame
  from a video entirely within one dataset split.
- **Dashboard integration:** combine operational scores/events and visual
  detections, with real-pipeline and simulation modes clearly distinguished.

Video processing and dashboard integration remain future placeholders; Day 24
does not implement them.
