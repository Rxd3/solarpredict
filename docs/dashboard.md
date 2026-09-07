# Streamlit dashboard

## Purpose and current scope

The Streamlit application is an integrated engineering prototype for the
**Solar Panel Monitoring and Fault Detection System**. It combines a recorded
operational-data replay, frozen anomaly-model results, frozen YOLO image
inspection, and frame-by-frame video inspection.

The operational source is consistently labelled **Recorded Operational Dataset
Replay** or **Monitoring Data Replay**. It is not live plant telemetry. Visual
predictions are screening results for review, not a professional inspection or
a certification that panels are healthy.

## Application sections

### Overview

The overview reports the selected operational reading, actual alert count and
status, grouped-event count, recorded-data scope, and the readiness of the four
prototype modules. The default 337-row feed contains zero threshold crossings;
the application does not fabricate demonstration alerts.

### Operational Monitoring

This section provides a chronological replay slider; Previous, Next, and Reset
controls; operational measurements; frozen anomaly score and threshold; four
interactive charts; and actual grouped-event output. Timestamps remain
timezone-naive because the dataset source does not specify a timezone.

### Panel Inspection

The image workflow accepts JPG, JPEG, and PNG uploads up to 20 MB. The browser
extension filter is followed by decoding the actual bytes with OpenCV. After a
successful preview, inference starts only when **Analyze Image** is pressed.

The page uses the existing frozen YOLO11n checkpoint and lets the user select a
display/deployment confidence threshold from 0.05 to 0.95 (default 0.25). This
control changes which boxes are returned for the current upload; it is not a
model-accuracy value and does not tune or retrain the checkpoint.

Actual outputs include:

- original and annotated images;
- total detections, highest confidence, detected-condition count, and cautious
  `ATTENTION`, `CLEAN`, or `NO_DETECTION` status;
- a confidence-ordered table with class, confidence, and box coordinates;
- detected-class counts;
- an annotated PNG download.

The uploaded image and generated annotation remain in memory for the current
session and are not permanently saved by the application.

### Video Inspection

The video workflow accepts MP4, AVI, and MOV uploads up to 200 MB. It validates
the extension, file size, container metadata, FPS, dimensions, reported frame
count, and an actual decoded frame before enabling processing. Inference starts
only when **Process Video** is pressed.

The confidence control has the same 0.05 to 0.95 range and 0.25 default as image
inspection. Frame stride is limited to 1 through 10. Stride 1 analyzes every
frame; a higher value analyzes every nth frame to reduce CPU work. Every source
frame is still written to the annotated output, and skipped inference frames
remain explicitly marked in the frame JSON rather than being treated as
no-detection frames.

Processing reuses the existing sequential video API in an isolated,
randomly-named temporary directory. Client filenames are reduced to safe
basenames for display only and are never used as trusted storage paths. The
temporary upload and intermediate files are removed automatically after their
bytes have been collected for the session.

Actual outputs include:

- total, analyzed, and output-frame counts;
- frame stride, total detections, processing time, and approximate inference
  FPS;
- cautious aggregate visual-condition status;
- six-class counts, frames containing each class, and maximum confidence;
- a detection-log preview;
- downloadable annotated MP4, complete detection CSV, per-frame JSON, and
  summary JSON.

OpenCV first creates and validates an MP4V result. The dashboard then uses the
project dependency `imageio-ffmpeg` to create H.264/yuv420p, fast-start MP4 for
browser playback. If conversion is unavailable, the validated OpenCV output is
still downloadable and the UI reports that playback may not work in the
browser. Generated download names contain a random run identifier and never
reuse an untrusted upload path.

## Frozen model boundaries

The anomaly service verifies its frozen artifacts and rescans the saved feed
without fitting. The visual service verifies the selected checkpoint SHA-256
before loading it and checks the exact six-class map. Streamlit caches each
model resource so normal reruns do not repeatedly load weights. Neither
dashboard workflow calls training, fitting, threshold calibration, or test-set
tuning code.

## Error handling and session behavior

Expected missing artifacts, corrupt uploads, unsupported formats, unreadable
video containers, invalid metadata, inference failures, and encoding failures
are shown as readable messages rather than normal-user tracebacks. A result is
invalidated when its upload hash, confidence, or video stride changes, which
prevents stale results from being shown under new settings.

## Launch

From the repository root with the project environment activated:

```powershell
.\.venv\Scripts\streamlit.exe run src/dashboard/app.py
```

Or, after activation:

```powershell
streamlit run src/dashboard/app.py
```

## Prototype limitations

- Operational monitoring replays 337 recorded readings rather than consuming
  live telemetry.
- The frozen anomaly detector produced no alerts for the final controlled
  moderate synthetic evaluation events.
- The computer-vision detector has low overall test performance and is
  especially weak for dusty, electrical-damage, and physical-damage classes.
- `NO_DETECTION` does not establish that a panel is healthy or fault-free, and
  a clean-only output is not a physical-health certification.
- Uploaded-video processing is synchronous and CPU-oriented; a large video can
  take substantial time even with frame stride.
- No representative real drone footage is bundled. The existing engineering
  demo was constructed from validation images and must not be described as
  real drone footage.
- Authentication, persistent inspection history, live ingestion, background
  job processing, and deployment hardening remain outside this prototype.

## Verification

Automated Streamlit tests render all pages and exercise the operational replay.
Service tests validate real image inference, corrupt uploads, safe filenames,
the frozen checkpoint hash, frame-stride accounting, generated logs, H.264
conversion, and unchanged anomaly artifacts. A live local-server smoke test is
also used to verify navigation and real upload workflows.
