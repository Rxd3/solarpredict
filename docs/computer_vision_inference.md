# Computer-vision inference interfaces

## Frozen image/frame API

`src/computer_vision/inference.py` exposes the frozen six-class checkpoint
through:

- `load_detector()` — cached loading plus exact class-map validation;
- `predict_image()` — inference from a readable image path;
- `predict_frame()` — inference from an OpenCV BGR array;
- `predict_images()` — repeated image-path inference with one model load;
- `draw_detections()` and `save_annotated_image()` — non-destructive rendering;
- `summarize_conditions()` — dashboard-ready condition aggregation.

Every detection contains `class_id`, `class_name`, `confidence`, `x1`, `y1`,
`x2`, and `y2`. Image results also record dimensions, detection count, unique
classes, highest confidence, and the condition summary. Coordinates are clipped
to image bounds and confidences are constrained to `[0, 1]` for a stable output
contract.

The viewing/deployment confidence defaults to 0.25. It is configurable for a
future application, but was predeclared and was **not** tuned from test results.

Condition-summary rules are intentionally cautious:

- one or more non-clean detections → `ATTENTION`;
- detections containing only `clean` → `CLEAN`;
- no detections → `NO_DETECTION`.

`NO_DETECTION` is not interpreted as healthy or fault-free.

## Video processor

`src/computer_vision/process_video.py` opens an input video through OpenCV,
reads sequentially, performs CPU inference, overlays labels/boxes, and writes an
annotated video at the source FPS and dimensions when OpenCV provides them.
Default confidence is 0.25 and default `frame_stride` is 1.

When stride is greater than one, every source frame is still written. Skipped
inference frames are visibly marked in the annotated video and explicitly
recorded with `analyzed: false` in the frame log; they are not silently treated
as zero detections. Video time is calculated as `frame_index / source_fps`.

Outputs are:

- annotated video;
- detection CSV with `frame_index`, `video_time_seconds`, class fields,
  confidence, and bounding-box coordinates;
- frame JSON containing a record for every source frame, nested detections, and
  condition summary for every analyzed frame;
- summary JSON with duration, FPS/dimensions, total/output/analyzed frames,
  stride, class counts, frames per class, maximum class confidences, and timing.

## Pipeline smoke test and performance

No `.mp4`, `.avi`, or `.mov` inspection video existed in the repository before
the smoke test. `src/computer_vision/create_demo_video.py` therefore created a
12-frame, 640×480, 2 FPS input from six class-covering validation images.

**Demo sequence created from validation images for video-pipeline testing.** It
is not real drone footage, and each frame is visibly labelled as a demo.

The real processing path analyzed all 12 frames at stride 1 and wrote all 12
output frames. It recorded 68 detections. On this run, detector calls averaged
0.1468 seconds per analyzed frame, approximately 6.81 inference FPS; total
processing took 1.8347 seconds (about 6.54 end-to-end FPS). These figures are a
small local CPU smoke-test measurement, not a throughput guarantee. A larger
`frame_stride` may be useful for long footage, with the explicit coverage tradeoff
retained in both logs and the video.

Artifacts are under `outputs/computer_vision/video/`:

- `demo_input.mp4` and `demo_input_metadata.json`;
- `demo_annotated.mp4`;
- `demo_detections.csv`;
- `demo_frames.json`;
- `demo_summary.json`.

Generated video binaries are ignored by Git; the small metadata and logs remain
reproducible project records.

## Streamlit image integration

`src/dashboard/vision_service.py` adapts the frozen APIs for uploads without
changing their inference behavior. The image page accepts JPG, JPEG, and PNG
files up to 20 MB, checks the suffix, decodes the actual bytes with OpenCV, and
runs inference only after the user selects **Analyze Image**. Uploaded image
bytes and annotated PNG output remain session-resident rather than being saved
as project data.

The page exposes a display/deployment confidence threshold from 0.05 to 0.95,
defaulting to the predeclared 0.25. Changing it invalidates the old result. It
does not retrain the detector or redefine the independent test metrics. The UI
shows the actual annotated image, cautious condition status, confidence-ordered
detections and box coordinates, class counts, and an annotated PNG download.

## Streamlit video integration

The video page accepts MP4, AVI, and MOV uploads up to 200 MB. It validates an
actual decoded frame plus FPS, dimensions, and frame-count metadata before the
explicit **Process Video** action. Uploads are placed under generated filenames
inside an isolated temporary directory; client filenames are display-only and
cannot choose storage locations. All temporary media and logs are removed after
their result bytes are collected.

The user can choose confidence from 0.05 to 0.95 and frame stride from 1 to 10.
Every source frame remains in the annotated output at the source FPS and
dimensions. When stride exceeds one, only every nth frame receives inference;
the per-frame JSON explicitly records skipped frames as `analyzed: false`.

The normal OpenCV output is validated and then transcoded with the bundled
`imageio-ffmpeg` executable to H.264/yuv420p with fast-start metadata for browser
playback. A failed conversion is surfaced clearly; the validated OpenCV output
remains downloadable. The page provides the annotated MP4, full detection CSV,
per-frame JSON, and summary JSON with unique generated filenames.

The frozen checkpoint SHA-256 is verified before dashboard model loading:

```text
b7ad8d7fca947527b30fb24f8c11e9135fbe62aa1e7f44e8a54100933141cd89
```

No dashboard path invokes training or test-set evaluation. The established
weak-class and no-real-drone-footage limitations remain visible beside the
results.
