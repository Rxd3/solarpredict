# System architecture

## Integrated prototype

```text
Recorded Operational Dataset Replay
              |
              v
Validated operational rows --> Frozen anomaly detector + threshold
              |                              |
              |                              v
              |                    score / NORMAL / ALERT
              |                              |
              +------------------------------+
                             |
                             v
                    Streamlit Dashboard
                             ^
                             |
               visual results and downloads
                             |
Uploaded image/video --> safe validation --> Frozen YOLO11n detector
                                                |
                                                v
                              boxes / classes / confidence
```

The Streamlit application is the presentation boundary for both established
prototype paths. It does not fit, tune, calibrate, or otherwise mutate either
model.

## Operational path

`src/anomaly_detection/inference.py` verifies the frozen scaler, Isolation
Forest, metadata, and threshold artifacts; validates the exact seven inputs;
and returns a stable scoring schema. `src/dashboard/data_service.py` joins the
recorded 337-row evaluation baseline with its saved anomaly feed and verifies
the saved scores and statuses against the frozen inference boundary. Exact
two-minute threshold crossings are grouped into events for display.

## Image path

`src/dashboard/vision_service.py` validates the upload extension and size,
decodes the actual bytes with OpenCV, verifies the frozen checkpoint hash, and
calls `src/computer_vision/inference.py`. The existing prediction and drawing
functions return actual bounded boxes, class names, confidences, condition
summaries, and an annotated PNG held in memory for display/download.

## Video path

The visual dashboard service validates container metadata and a readable frame,
then places the upload in a generated isolated temporary workspace. It calls
`src/computer_vision/process_video.py`, which processes frames sequentially,
preserves every output frame, and writes detection CSV, per-frame JSON, and
summary JSON. A bundled FFmpeg binary supplied through `imageio-ffmpeg`
transcodes the validated OpenCV output to browser-compatible H.264/yuv420p when
available. Output bytes are returned to Streamlit before the temporary
workspace is automatically removed.

## Data and security boundaries

- Raw operational and vision datasets remain unchanged.
- Uploaded media is size-limited, content-validated, and never permanently
  written by the dashboard.
- Untrusted client filenames are not used as storage paths.
- Model resources are cached, but upload results are invalidated when content
  hashes or inference settings change.
- The operational source timezone remains unspecified and timestamps remain
  timezone-naive.
- Video frames derived from a future source video must remain entirely within
  one train, validation, or test split to prevent temporal leakage.

## Remaining system work

The operational, anomaly-inference, image-inference, video-processing, and
dashboard integration paths are complete as engineering prototypes. Remaining
work includes representative real-drone validation, broader end-to-end and
performance testing, persistent inspection records, live telemetry ingestion,
authentication, deployment hardening, and final demonstration/handoff
materials. The frozen checkpoint and one-time test results must not be tuned
from the test split.
