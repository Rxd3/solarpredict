# Streamlit dashboard

## Purpose and current scope

The dashboard is an engineering prototype for the **Solar Panel Monitoring and
Fault Detection System**. Its operational side is functional and replays
recorded evaluation-baseline measurements through previously calculated,
verified outputs from the frozen anomaly detector.

The interface consistently labels this source **Recorded Operational Dataset
Replay** or **Monitoring Data Replay**. It is not live plant telemetry and must
not be presented as a production monitoring deployment.

## Application sections

### Overview

The overview displays:

- 337 monitored replay readings;
- power and solar radiation for the selected reading;
- the actual number of alert rows (zero in the default feed);
- the selected row's `NORMAL` or `ALERT` result;
- operational prototype status and grouped-event count;
- computer-vision readiness and its important weak-class limitation;
- recorded-dataset scope and sampling information;
- concise frozen-model information.

### Operational Monitoring

This section provides:

- replay slider plus Previous Reading, Next Reading, and Reset Replay controls;
- separate measurement and model-result panels;
- selected power, solar radiation, air temperature, relative humidity, wind
  speed, RTD mean/std, anomaly score, frozen threshold, and status;
- interactive power and solar-radiation time series;
- anomaly score with the unchanged frozen threshold;
- selectable, single-unit environmental charts;
- a selected-timestamp indicator on each chart;
- actual alert-row and grouped-event summaries.

The default feed contains zero threshold-crossing rows, so the event area states
that no events were detected. The dashboard does not add demonstration alerts.

### Panel Inspection

This is a deliberate placeholder. It lists the verified six classes and model
limitations, but exposes no inactive upload control and generates no prediction.
Image upload, annotated results, and condition summaries are planned for Day 28.

### Drone Video Inspection

This is also a deliberate placeholder. It describes the prepared frame-by-frame
processor, annotated video, class counts, timestamps, and detection logs. No real
drone footage is bundled; the earlier engineering smoke test used a clearly
labelled sequence created from validation images. Video upload and execution are
planned for Day 28.

## Data and anomaly integration

`src/dashboard/data_service.py` joins two existing, timestamp-aligned sources:

- `data/model_ready/core_evaluation_baseline.csv` provides the recorded seven
  model features and display measurements;
- `outputs/dashboard_anomaly_feed_example.csv` provides saved anomaly score,
  threshold, margin, and status values.

Both contain the same 337 evaluation-baseline timestamps. The service validates
their schemas, exact timestamp order, numerical fields, power/radiation
agreement, threshold margins, and strict status mapping. It then rescans the
rows through `src/anomaly_detection/inference.py` and requires the frozen scores,
threshold, and statuses to match the saved feed. Model loading is cached, and
Streamlit caches the verified replay across normal reruns. No fitting occurs.

Timestamps use the verified `%Y-%m-%d %H:%M:%S` representation and remain
timezone-naive because the dataset source does not specify a timezone.

## Replay behavior

Replay position is held in Streamlit session state. The slider permits direct
inspection; Previous and Next move exactly one recorded row; Reset returns to
the first row. All operations clamp safely to `[0, 336]`. Autoplay is omitted to
keep the prototype deterministic and reliable.

## Error handling

The application displays a readable error and recovery guidance when a feed or
artifact is missing, required columns are absent, timestamps are invalid,
sources disagree, or frozen-artifact verification fails. Expected data errors
do not expose a Python traceback to normal dashboard users.

## Launch

From the repository root with the project virtual environment activated:

```powershell
streamlit run src/dashboard/app.py
```

Equivalent explicit Windows command:

```powershell
.\.venv\Scripts\streamlit.exe run src/dashboard/app.py
```

Streamlit and Plotly were already present in the minimal project requirements
and verified in the virtual environment.

## Prototype limitations

- The 337-row replay is recorded data, not live telemetry.
- The underlying source spans approximately 33.6 hours, not long-term or
  seasonal operation.
- The frozen threshold is a prototype boundary, not a validated plant alarm.
- The final moderate synthetic evaluation produced zero detected events; this
  limitation remains visible in the model-information panel.
- The computer-vision detector has low overall performance and is especially
  weak for dusty, electrical-damage, and physical-damage conditions.
- Image/video dashboard integration, representative real-drone validation,
  cross-module testing, authentication, persistence, and deployment hardening
  remain future work.

## Verification

Streamlit's application test harness renders every navigation section, exercises
the slider and Next control, verifies all four operational charts, confirms the
selected reading updates, and confirms the default feed remains at zero alerts.
The application also launches as a local Streamlit server without an exception.
