# Solar Panel Monitoring and Fault Detection System

University internship project for **TECHNOVOLT**. The long-term objective is to
build a system that brings photovoltaic operating data and visual panel
inspections together, making it easier to monitor performance and identify
possible faults.

This repository currently contains the project scaffold plus the completed
operational-dataset acquisition, inspection, data-quality, and initial
exploratory-analysis stage, followed by conservative operational-data
preprocessing and validation. The unusual-measurement review and Day 8 feature
plan are also complete. Basic operational features and the selected 10-, 30-,
and 60-minute trailing rolling features have been implemented and validated.
Causal first-difference, rate, safeguarded relative-change, and 30-minute
baseline-deviation features are also complete. The 81-column feature inventory,
redundancy review, missingness classification, recurring-transition review, and
candidate model-input plan are complete. Day 12 adds seven disabled synthetic
scenario specifications and a reproducible evaluation plan. Day 13 implements
the generator and a seed-42 validation prototype with seven events. Day 14
reviews direct and propagated effects and proposes two chronological split
strategies. Day 15 adopts Option B and creates untouched baseline and unscaled
Core matrices for training, calibration, and evaluation. Day 16 fits one
training-only scaler and an initial Isolation Forest, then generates continuous
scores for the three untouched baseline partitions. Day 17 reproduces those
scores without refitting and investigates feature support, clock-time, light
context, RTD spread, and high-score baseline observations. Day 18 performs a
controlled four-variant feature comparison and recommends `NO_TIME_7` from
training/calibration evidence only. Day 19 confirms its stability across five
predeclared random seeds and freezes the existing seed-42 `NO_TIME_7` artifact
as the prototype configuration. Day 20 uses untouched calibration-baseline
scores to freeze a conservative prototype threshold, then performs a strictly
post-freeze evaluation-baseline sanity check. Day 21 creates and validates the
final unscaled synthetic evaluation copy using full causal history and seed
2026. Day 22 applies the unchanged detector and threshold to that copy and
records strict point-level, event-level, propagated-context, and paired-baseline
results. Day 23 documents the zero-detection result, closes the frozen
operational anomaly component as **PROTOTYPE COMPLETE**, and adds a verified,
read-only inference API plus an honest zero-alert dashboard-feed example. No
parameter was changed in response to the evaluation. The next major component
is **COMPUTER VISION SOLAR-PANEL INSPECTION**. Day 24 prepares its YOLO dataset
inspector, label validator, ground-truth visualization, CPU-safe training
configuration, dry-run entry point, and evaluation interface. The downloaded
Roboflow export has now been inspected and cleaned conservatively: 795 active
image/label pairs, six actual classes, and 5,751 valid boxes. The one exact
train/test duplicate and 25 uncertain empty-label pairs were moved intact to
recoverable quarantine. The final active dataset passes its integrity checks and
Day 25 completed one controlled CPU YOLO11n run: early stopping ended at epoch
28, with epoch 20 selected from validation. The selected prototype reached
validation precision 0.3083, recall 0.2462, mAP50 0.2004, and mAP50-95 0.0807.
Day 26 froze the evaluation settings, evaluated the 98-image test split exactly
once, and added reusable image/frame inference plus a sequential video interface.
The unchanged checkpoint reached test precision 0.3878, recall 0.2356, mAP50
0.2183, and mAP50-95 0.1062. No tuning followed the test result. The operational
anomaly and computer-vision components are now **PROTOTYPE COMPLETE**. Day 27
adds a working Streamlit application with a verified 337-row recorded-data
replay, operational measurements, frozen anomaly scores/status, event handling,
and prepared image/video navigation. Image/video UI integration and final
cross-module testing remain pending.

The repository directory name is **`solarpredict`**. The project and display
name remains **Solar Panel Monitoring and Fault Detection System**.

## Planned system components

1. **Operational monitoring** — ingest and visualize photovoltaic time-series
   measurements such as power, voltage, current, irradiance, and temperature.
2. **Time-series anomaly detection** — detect unusual operating patterns using
   normal operational data and, initially, reproducible synthetic anomalies.
3. **Computer-vision inspection** — detect visible solar-panel conditions with
   object-detection annotations.
4. **Drone/video processing** — extract and analyze video frames while retaining
   links to their source footage and timestamps.
5. **Unified web dashboard** — combine operational trends, anomaly results, and
   computer-vision findings in one interface.

## Project structure

```text
solarpredict/
|-- data/
|   |-- operational/       # Raw photovoltaic time-series files
|   |-- images/            # Images and, later, source video datasets
|   |-- processed/         # Cleaned or transformed datasets
|   `-- README.md          # Dataset scope, licenses, and policies
|-- notebooks/             # Exploratory analysis notebooks
|-- docs/                  # Verified dataset and schema documentation
|-- config/                # Disabled scenario designs and future configuration
|-- src/
|   |-- data_processing/   # Dataset inspection, loading, and quality utilities
|   |-- anomaly_detection/ # Frozen prototype evaluation and inference code
|   |-- computer_vision/   # Frozen YOLO evaluation and image/video inference
|   `-- dashboard/         # Streamlit replay UI, services, components, and charts
|-- models/                # Generated model artifacts (not committed)
|-- outputs/
|   |-- figures/           # Generated plots and figures
|   `-- predictions/       # Generated prediction files
|-- tests/                 # Automated tests
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Selected datasets and data policies

### Operational time-series data

- **Dataset:** [Solar Power Dataset](https://www.kaggle.com/datasets/s1nister/solar-power-generation-dataset)
- **Source:** Kaggle, published by `s1nister`
- **License:** CC0 / Public Domain
- **Observed sampling information:** the downloaded version 1 CSV has an exact
  two-minute interval between every consecutive timestamp.
- **Known measurement scope:** environmental and solar measurements such as
  solar radiation, temperature, humidity, and wind information, together with
  electrical measurements used to determine generated power.

The downloaded file is `solar_data.csv`, with 1,009 rows and 14 columns. Its
verified schema, units, date range, hashes, descriptive results, and quality
findings are recorded in [docs/operational_dataset.md](docs/operational_dataset.md).
The source metadata does not specify a timezone. Timestamps remain timezone-naive
and are documented as **"not specified by source"**. They must not be labeled as
UTC or assigned a geographical timezone without new source evidence.

This dataset initially represents normal operational behavior. A later stage
will add synthetic anomalies through a documented, seeded, and reproducible
generation process. Synthetic records and their provenance must remain
distinguishable from the original observations.

### Computer-vision data

- **Dataset:** [Detection for defects in solar panels](https://universe.roboflow.com/solar-panels-yolo/detection-for-defects-in-solar-panels-9qenv)
- **Source/creator:** Solar Panels YOLO on Roboflow Universe
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Task:** Object detection
- **Annotation format:** YOLO format compatible with Ultralytics

The relevant conditions advertised by the source include `clean`, `dusty`,
`bird-drop`, `electrical-damage`, `physical-damage`, and `snow-covered`. These
names are not a substitute for inspecting the downloaded export. The export's
`data.yaml` is authoritative for the actual class names, ordering, dataset
paths, and available splits; classes must not be renamed before that inspection.

CC BY 4.0 attribution must accompany published or redistributed use of the
dataset and derived materials. Credit the dataset title and creator, link to the
Roboflow source and CC BY 4.0 license, and indicate whether changes were made.
Attribution must not imply that the creator endorses TECHNOVOLT or this project.

### Split policy

- Preserve the image dataset's original train/validation/test split whenever it
  is supplied. Do not create a new random split merely to reshuffle the data.
- Do not randomly distribute extracted video frames across splits.
- For future drone footage, assign each complete source video to exactly one
  split. Every frame derived from that video must inherit the same split to
  prevent temporal and data leakage.

## Current project status

Completed in the operational-data stage:

- project structure and Python environment setup;
- operational dataset selection and CC0 provenance documentation;
- anonymous acquisition of Kaggle dataset version 1;
- inspection of every downloaded operational CSV;
- verified operational schema and timezone-naive timestamp loader;
- machine-readable data-quality analysis;
- eight exploratory figures and a reproducible EDA notebook;
- conservative operational-data preprocessing and before/after validation;
- operational data-quality review of negative radiation, wind direction,
  temporal coverage, and variable relationships;
- a documented feature-engineering plan for the next development stage;
- basic time, RTD-summary, electrical-consistency, and low-light-context feature
  engineering;
- full-window, trailing-only rolling feature engineering for selected signals;
- causal change/rate-of-change and safeguarded deviation feature engineering;
- feature review and model-input planning across all 81 columns, including
  redundancy, missingness, and recurring 17:00 transition evidence;
- synthetic anomaly scenario design and anomaly evaluation planning (Day 12);
- synthetic anomaly generator, dependent-feature recomputation, and one
  reproducible prototype for engineering validation (Day 13);
- synthetic prototype review and chronological split planning (Day 14);
- final chronological split selection and baseline model-dataset preparation
  using the adopted 336/336/337-row Option B boundaries (Day 15);
- training-only StandardScaler fitting, initial Isolation Forest training, and
  continuous untouched-baseline score generation (Day 16);
- read-only baseline score and distribution-shift investigation using the saved
  Day 16 artifacts (Day 17);
- controlled comparison of four predeclared Isolation Forest feature sets using
  training/calibration selection evidence only (Day 18);
- five-seed `NO_TIME_7` score/rank stability review, focused RTD sensitivity
  check, and seed-42 prototype configuration freeze (Day 19);
- calibration-only comparison of six predeclared threshold rules and freeze of
  the 97.5th-percentile prototype threshold (Day 20);
- evaluation-only generation and validation of the final synthetic dataset,
  including explicit direct/propagated effect context (Day 21);
- frozen-detector scoring of the final synthetic dataset, including strict
  point-level and event-level evaluation and paired-baseline review (Day 22);
- evidence-based final review, frozen read-only inference API, exact two-minute
  event grouping, and dashboard-ready baseline export (Day 23);
- computer-vision directory structure, YOLO dataset-validation tooling,
  real dataset/YAML inspection, exhaustive annotation validation, class and
  split review, ground-truth visualization, environment verification, successful
  non-training dry run, duplicate cleanup, empty-label quarantine, final training
  readiness review, and evaluation interface (Day 24);
- one controlled YOLO11n CPU training run, validation-only evaluation, training
  curve/confusion review, and fixed validation prediction figure (Day 25);
- one-time independent test evaluation of the frozen checkpoint, descriptive
  validation/test comparison, and preserved test curves/confusion matrix;
- frozen six-class image/frame inference, non-destructive annotations, cautious
  condition summaries, sequential video processing, structured frame/detection
  logs, and a labelled validation-image demo smoke test (Day 26);
- Streamlit application structure, system overview, operational monitoring,
  session-state replay controls, selected-reading indicators, frozen anomaly
  status/events, and nonfunctional CV integration placeholders (Day 27);
- loader, inspection, preprocessing, quality-review, basic-feature, rolling,
  change-feature, feature-review, scenario-configuration, and generator tests.

Not completed yet:

- image inspection dashboard integration;
- video inspection dashboard integration;
- final operational/computer-vision dashboard integration testing;
- validation on representative real drone footage;
- final full-system testing and deployment review.

### Operational anomaly component: PROTOTYPE COMPLETE

The completed operational path covers data preparation, feature engineering,
model selection, stability review, threshold calibration, controlled synthetic
evaluation, and a reusable inference API. Its important known limitation is
equally explicit: the frozen prototype detected none of the three moderate
synthetic evaluation events on Day 22. This module is closed for further model
tuning; future use must retain the frozen artifacts and report actual output.

### Computer vision: PROTOTYPE COMPLETE

Completed: acquisition, actual class/split discovery, validation of every YOLO
line, image/annotation inventory, class distribution, cross-split hash review,
real ground-truth sample visualization, CPU/Ultralytics verification,
lightweight YOLO11n configuration, successful non-training dry run, and a future
evaluation interface. The exact train/test duplicate and its differing label
were documented, compared visually, and moved intact to quarantine. All 25
uncertain empty-label pairs were also quarantined after review, without inventing
annotations. The final 582/115/98 active splits contain 795 paired files and
5,751 boxes; every class remains present in every split and the data-quality gate
is open. One approved YOLO11n run completed 28 epochs before normal early
stopping, selecting epoch 20. Validation limitations remain explicit. The frozen
checkpoint was then evaluated once on all 98 test images and was not changed or
tuned afterward. Image and OpenCV-frame prediction, bounded structured boxes,
annotated output, conservative `ATTENTION`/`CLEAN`/`NO_DETECTION` summaries, and
sequential video processing are implemented. Dusty recall remains particularly
weak, electrical and physical damage remain limited, and the video smoke test
used a labelled validation-image demo rather than real drone footage. Dashboard
integration remains pending.

### Dashboard: OPERATIONAL MONITORING COMPLETE

The Streamlit dashboard now provides a functional Overview and Operational
Monitoring experience over the 337-row evaluation-baseline replay. It validates
and joins the recorded operational measurements with the unchanged saved anomaly
feed, verifies those scores against the frozen detector, and preserves the real
zero-alert result. Replay position is controlled by a slider and bounded
Previous/Next/Reset controls. Power, solar radiation, anomaly score/threshold,
and individually selected environmental charts all mark the chosen timestamp.
Panel and drone-video sections are intentionally nonfunctional placeholders for
the next integration stage; no upload control or fabricated prediction is shown.

Operational-stage artifacts:

- [Verified schema and findings](docs/operational_dataset.md)
- [Data-quality JSON](outputs/operational_data_quality.json)
- [EDA notebook](notebooks/01_operational_data_exploration.ipynb)
- [Preprocessing decisions and validation](docs/operational_preprocessing.md)
- [Preprocessing summary JSON](outputs/preprocessing_summary.json)
- [Day 7 quality review and feature plan](docs/feature_planning.md)
- [Day 7 machine-readable review](outputs/operational_quality_review.json)
- [Day 8 basic feature definitions and validation](docs/basic_feature_engineering.md)
- [Day 8 machine-readable feature summary](outputs/basic_feature_summary.json)
- [Day 9 rolling feature definitions and validation](docs/rolling_feature_engineering.md)
- [Day 9 machine-readable rolling summary](outputs/rolling_feature_summary.json)
- [Day 10 change feature definitions and validation](docs/change_feature_engineering.md)
- [Day 10 machine-readable change summary](outputs/change_feature_summary.json)
- [Day 11 feature inventory and model-input plan](docs/model_feature_review.md)
- [Day 11 machine-readable feature review](outputs/model_feature_review.json)
- [Day 11 recurring 17:00 transition review](outputs/recurring_transition_review.json)
- [Day 12 synthetic scenario design and evaluation plan](docs/synthetic_anomaly_design.md)
- [Day 12 disabled example scenario configuration](config/synthetic_anomaly_scenarios.yaml)
- [Day 13 generator and prototype validation](docs/synthetic_anomaly_generator.md)
- [Day 13 event metadata and source preservation](outputs/synthetic_anomaly_metadata.json)
- Synthetic prototype CSV: `data/processed/operational_synthetic_prototype.csv`
  (ignored by Git; not the final model test dataset)
- [Day 13 synthetic example figure](outputs/figures/operational/day13_synthetic_anomaly_example.png)
- [Day 14 prototype review and split recommendation](docs/synthetic_prototype_review.md)
- [Day 14 machine-readable chronological review](outputs/chronological_split_review.json)
- [Day 14 proposed split timeline](outputs/figures/operational/day14_split_timeline.png)
- [Day 15 adopted split configuration](config/chronological_split.yaml)
- [Day 15 model-dataset preparation](docs/model_dataset_preparation.md)
- [Day 15 machine-readable preparation report](outputs/model_dataset_preparation.json)
- Baseline and unscaled Core partition CSVs: `data/model_ready/` (ignored by
  Git and reproducible from the unchanged 81-column feature dataset)
- [Day 16 model configuration](config/anomaly_model.yaml)
- [Day 16 initial detector review](docs/initial_detector_review.md)
- [Day 16 machine-readable detector summary](outputs/initial_detector_summary.json)
- Continuous baseline score CSVs: `outputs/anomaly_scores_*.csv`
- Training-scaled Core matrices: `data/model_ready/scaled/` (generated and
  ignored by Git)
- Fitted Day 16 scaler, Isolation Forest, and metadata: `models/` (generated
  and ignored by Git)
- [Day 16 baseline score figure](outputs/figures/operational/day16_baseline_anomaly_scores.png)
- [Day 17 baseline score investigation](docs/baseline_score_investigation.md)
- [Day 17 machine-readable investigation](outputs/baseline_score_investigation.json)
- [Day 17 light-context score figure](outputs/figures/operational/day17_score_context_review.png)
- [Day 18 predeclared comparison configuration](config/model_feature_comparison.yaml)
- [Day 18 controlled feature comparison](docs/model_feature_comparison.md)
- [Day 18 machine-readable comparison](outputs/model_feature_comparison.json)
- [Day 18 comparison figure](outputs/figures/operational/day18_model_feature_comparison.png)
- Day 18 comparison scalers, models, and metadata: `models/comparisons/`
  (generated and ignored by Git)
- [Day 19 stability review and prototype freeze](docs/model_stability_and_freeze.md)
- [Day 19 predeclared stability configuration](config/model_stability_review.yaml)
- [Frozen prototype configuration](config/frozen_anomaly_detector.yaml)
- [Day 19 machine-readable stability review](outputs/model_stability_review.json)
- Frozen model metadata: `models/frozen_anomaly_detector_metadata.json`
  (references the existing Day 18 seed-42 artifacts and records their hashes)
- [Day 19 seed-stability figure](outputs/figures/operational/day19_seed_stability.png)
- [Day 20 threshold calibration documentation](docs/anomaly_threshold_calibration.md)
- [Day 20 predeclared threshold candidates](config/anomaly_threshold_calibration.yaml)
- [Frozen prototype threshold](config/frozen_anomaly_threshold.yaml)
- [Day 20 machine-readable threshold review](outputs/anomaly_threshold_calibration.json)
- Frozen threshold metadata: `models/frozen_anomaly_threshold_metadata.json`
  (contains references and hashes, not a trained model)
- [Day 20 threshold-calibration figure](outputs/figures/operational/day20_threshold_calibration.png)
- [Day 21 final synthetic evaluation documentation](docs/final_synthetic_evaluation.md)
- [Day 21 final evaluation configuration](config/final_synthetic_evaluation.yaml)
- [Day 21 event metadata and validation](outputs/final_synthetic_evaluation_metadata.json)
- [Day 21 generated-power validation figure](outputs/figures/operational/day21_final_synthetic_evaluation.png)
- Final full-schema and unscaled `NO_TIME_7` evaluation CSVs:
  `data/model_ready/final_synthetic_evaluation*.csv` (generated and ignored by Git)
- [Day 22 final detector evaluation](docs/final_anomaly_detector_evaluation.md)
- [Day 22 machine-readable results](outputs/final_anomaly_detector_evaluation.json)
- [Day 22 prediction rows](outputs/final_synthetic_anomaly_predictions.csv)
- [Day 22 paired baseline score comparison](outputs/final_synthetic_score_comparison.csv)
- [Day 22 detection-results figure](outputs/figures/operational/day22_synthetic_detection_results.png)
- [Day 23 final operational anomaly review](docs/anomaly_module_final_review.md)
- [Frozen dashboard inference interface](src/anomaly_detection/inference.py)
- [Dashboard real/simulation separation policy](docs/dashboard_simulation_policy.md)
- [System architecture and future placeholders](docs/system_architecture.md)
- [Final operational-module status](outputs/anomaly_module_final_status.json)
- [Dashboard anomaly-feed example](outputs/dashboard_anomaly_feed_example.csv)
- [Day 24 verified dataset inventory and integrity review](docs/computer_vision_dataset.md)
- [Day 24 dataset cleanup and quarantine audit](docs/computer_vision_dataset_cleanup.md)
- [Day 24 computer-vision setup](docs/computer_vision_setup.md)
- [Day 25 YOLO training and initial validation](docs/computer_vision_training.md)
- [YOLO prototype training configuration](config/computer_vision_training.yaml)
- [Independent test evaluation](docs/computer_vision_test_evaluation.md)
- [Frozen test-evaluation settings](config/computer_vision_test_evaluation.yaml)
- [Machine-readable test metrics](outputs/computer_vision/test_evaluation.json)
- [Image and video inference interfaces](docs/computer_vision_inference.md)
- [Computer-vision prototype status](outputs/computer_vision/computer_vision_module_status.json)
- [Representative test prediction figure](outputs/computer_vision/figures/day26_test_predictions.png)
- [Machine-readable CV dataset/setup status](outputs/computer_vision/dataset_summary.json)
- [Streamlit dashboard documentation](docs/dashboard.md)
- [Dashboard application](src/dashboard/app.py)
- [Validated dashboard data service](src/dashboard/data_service.py)
- Processed CSV: `data/processed/operational_cleaned.csv` (ignored by Git and
  reproducible from the raw file)
- Basic feature CSV: `data/processed/operational_features_basic.csv` (ignored by
  Git and reproducible from the cleaned file)
- Rolling feature CSV: `data/processed/operational_features_rolling.csv`
  (ignored by Git and reproducible from the basic feature file)
- Change feature CSV: `data/processed/operational_features_change.csv` (ignored
  by Git and reproducible from the rolling feature file)
- [Day 9 power rolling-mean validation figure](outputs/figures/operational/day9_power_rolling_mean.png)
- Generated figures: `outputs/figures/operational/` (ignored by Git and
  reproducible from the analysis script)

## Setup

Python 3.12 is recommended for a stable scientific-Python environment. From the
project root:

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset inspection

The public operational dataset was acquired anonymously with `kagglehub`. To
repeat that download after installing the requirements, run this from the
project root:

```powershell
python -c "from pathlib import Path; import kagglehub; kagglehub.dataset_download('s1nister/solar-power-generation-dataset', output_dir=str(Path('data/operational/solar-power-dataset').resolve()))"
```

Raw operational files are excluded by `.gitignore` and must not be committed.

Inspect every downloaded operational CSV:

```bash
python src/data_processing/inspect_dataset.py data/operational/solar-power-dataset
```

Use `--head` to change the number of preview rows:

```bash
python src/data_processing/inspect_dataset.py data/operational/solar-power-dataset --head 10
```

Regenerate the machine-readable quality report and exploratory figures:

```bash
python src/data_processing/analyze_operational_data.py data/operational/solar-power-dataset
```

Regenerate the conservatively processed operational dataset and its validation
summary:

```bash
python src/data_processing/preprocess_operational_data.py
```

Reproduce the read-only operational quality review and relationship analysis:

```bash
python src/data_processing/review_operational_quality.py
```

Regenerate and validate the bounded Day 8 basic feature dataset:

```bash
python src/data_processing/engineer_basic_features.py
```

Regenerate the Day 9 trailing rolling features, summary, and validation figure:

```bash
python src/data_processing/engineer_rolling_features.py
```

Regenerate the Day 10 causal change features and transition summary:

```bash
python src/data_processing/engineer_change_features.py
```

Regenerate the read-only Day 11 feature inventory, correlation review, candidate
sets, and recurring-transition evidence:

```bash
python src/data_processing/review_model_features.py
```

The Day 12 scenario template remains disabled. Day 13 uses an explicit per-run
override to create the combined, unsplit generator-validation prototype:

```bash
python -m src.anomaly_detection.generate_synthetic_anomalies --prototype --scenarios all --seed 42
```

Validate the configuration, generator, and complete existing suite:

```bash
python -m pytest -q -p no:cacheprovider
```

Regenerate the Day 14 synthetic-prototype review and proposed split timeline:

```bash
python -m src.anomaly_detection.review_synthetic_prototype
```

Create and validate the adopted Day 15 baseline and unscaled Core partitions:

```bash
python -m src.anomaly_detection.create_chronological_split
```

Reproduce the Day 16 training-only scaler, initial detector, and continuous
baseline scores:

```bash
python -m src.anomaly_detection.train_baseline_detector
```

Reproduce the Day 17 read-only baseline score investigation:

```bash
python -m src.anomaly_detection.investigate_baseline_scores
```

Reproduce the Day 18 controlled feature comparison:

```bash
python -m src.anomaly_detection.compare_detector_features
```

Reproduce the Day 19 five-seed stability review and prototype freeze checks:

```bash
python -m src.anomaly_detection.review_model_stability
```

Reproduce the Day 20 calibration-only threshold freeze and post-freeze baseline
sanity check:

```bash
python -m src.anomaly_detection.calibrate_anomaly_threshold
```

Reproduce the Day 21 final synthetic evaluation copy and validation artifacts:

```bash
python -m src.anomaly_detection.generate_final_synthetic_evaluation
```

Apply the unchanged detector and threshold to the Day 21 dataset and reproduce
the Day 22 evaluation:

```bash
python -m src.anomaly_detection.evaluate_final_synthetic_detector
```

Reproduce the Day 23 dashboard-feed example and final module-status record using
the read-only frozen inference API:

```bash
python -m src.anomaly_detection.finalize_operational_module
```

After downloading the Roboflow YOLO export, inspect it, render a ground-truth
sample sheet, and validate the training setup without fitting:

```bash
python -m src.computer_vision.inspect_vision_dataset
python -m src.computer_vision.visualize_annotations
python -m src.computer_vision.train_detector --dry-run
```

The independent test evaluation has already been completed once. Do **not**
rerun it or use its results to tune this frozen prototype. To recreate only the
labelled engineering demo and exercise the normal video inference interface:

```bash
python -m src.computer_vision.create_demo_video
python -m src.computer_vision.process_video outputs/computer_vision/video/demo_input.mp4 --output-video outputs/computer_vision/video/demo_annotated.mp4 --detection-log outputs/computer_vision/video/demo_detections.csv --frame-log outputs/computer_vision/video/demo_frames.json --summary outputs/computer_vision/video/demo_summary.json
```

Launch the operational monitoring dashboard:

```bash
streamlit run src/dashboard/app.py
```

Run the complete automated suite:

```bash
python -m pytest -q -p no:cacheprovider
```

Launch the reproducible notebook environment:

```bash
jupyter lab notebooks/01_operational_data_exploration.ipynb
```

Inspect an image dataset recursively:

```bash
python src/data_processing/inspect_images.py data/images
```

The image utility counts files by containing directory, reports one readable
sample's dimensions, and lists files that OpenCV cannot read. It does not train
or run a model.

## Planned development stages

1. **Project setup and dataset inventory (operational scope completed)**
   - Establish the repository structure and development environment.
   - Download the selected datasets under their respective license terms.
   - Inspect the operational CSV and the image export's `data.yaml` and
     directory organization.
   - Record verified schemas, units, labels, split definitions, and known
     quality issues without altering source files.
2. **Operational data cleaning, quality review, and feature planning (completed)**
   - Define schemas and validation rules.
   - Clean timestamps, missing values, duplicates, and invalid sensor readings.
   - Keep source timestamps timezone-naive unless better metadata is obtained.
   - Review image quality, verified class definitions, annotations, and label
     balance while preserving the source split.
   - Parse and normalize the verified operational schema, preserve missing and
     suspicious values, and validate the processed output against the raw file.
   - Investigate retained measurements and plan time, rolling, change,
     electrical-relationship, and environmental features without creating them.
3. **Feature engineering, operational baseline, and anomaly detection (feature
   construction, model-input planning, scenario design, prototype generation,
   chronological split planning, baseline model-dataset preparation, and the
   initial training-only detector, baseline score investigation, controlled
   feature comparison, prototype detector/threshold freezes, and final synthetic
   evaluation generation, frozen-detector evaluation, final review, and
   inference interface completed through Day 23; prototype closed for tuning)**
   - Create the approved time, RTD-summary, electrical-consistency, and
     low-light-context features while preserving every original value.
   - Add selected 10-, 30-, and 60-minute rolling statistics using complete,
     trailing-only windows.
   - Add causal differences, fixed-interval rates, denominator safeguards, and
     deviations from the existing 30-minute trailing means.
   - Inventory all features, review missingness and redundancy, and define
     compact model-input candidates without fitting preprocessing or a model.
   - Establish transparent baseline methods and time-aware evaluation splits.
   - Define reproducible synthetic anomaly scenarios separately from the normal
     source data.
   - Confirm seed stability and freeze a reproducible prototype feature/model
     configuration before threshold calibration.
   - Compare a small predeclared threshold set on calibration baseline scores,
     freeze one prototype rule, and keep later evaluation from changing it.
   - Generate the final synthetic evaluation copy from complete causal history,
     without applying or changing the frozen detector.
   - Apply the frozen detector and threshold once, report strict point/event
     metrics and paired score changes, and do not tune from those results.
   - Expose the frozen artifacts through a read-only interface and preserve the
     zero-detection Day 22 result without post-evaluation tuning.
4. **Computer-vision modeling (prototype complete)**
   - Validate the downloaded YOLO annotations and `data.yaml`.
   - Train one detector, select it using validation, then evaluate the frozen
     checkpoint once on the preserved test split.
5. **Drone/video inference pipeline (prototype interface complete)**
   - Process frames sequentially, retain frame/time provenance, and emit
     dashboard-ready detection and frame summaries.
   - Validate against representative real drone footage in a future stage.
6. **Dashboard and integration (operational dashboard complete)**
   - Present recorded operational measurements, frozen anomaly scores/status,
     and real grouped-event output in Streamlit.
   - Connect the existing image/video inference modules in the next stage.
7. **Testing, documentation, and handoff**
   - Add reproducible tests, configuration, deployment notes, and a final report.

## Current limitations

- The operational file spans only 33 hours 36 minutes, so it cannot support
  seasonal or long-term conclusions.
- Units and exact physical interpretation are not specified by the source for
  the five `RTD_*` channels, `Array_Voltage`, or `Array_Current`.
- Negative solar-radiation values and wind directions above 360 degrees are
  retained as documented quality flags; they have not been corrected or labeled
  as anomalies.
- The 5 W/m^2 low-light threshold is an exploratory project context flag, not a
  manufacturer specification or anomaly boundary.
- Full rolling windows intentionally introduce 4, 14, or 29 initial NaNs per
  feature; these values have not been filled or removed.
- Solar relative-change and normalized-deviation features intentionally retain
  NaN when the previous value or trailing baseline is below the documented
  absolute denominator floor.
- The verified computer-vision prototype has low overall test recall (0.2356)
  and mAP50-95 (0.1062). `dusty` recall is 0.0221; electrical and physical
  damage also remain limited. These weaknesses were not used for post-test
  tuning.
- The video smoke test used a labelled validation-image demo because no real
  inspection/drone video was available. Its short-run CPU throughput is not a
  deployment guarantee.
- `NO_DETECTION` does not establish that a panel is healthy or fault-free.
- The seed-42 `NO_TIME_7` model is frozen as a reproducible prototype
  configuration. Its calibration 97.5th-percentile threshold is also frozen,
  and its controlled Day 22 evaluation result is explicitly reported below.
- Every selected-threshold calibration alert is daylight-like, and 7 of 9 are
  among the 16 extreme training-scaled `rtd_std` rows. This is retained as a
  prototype limitation rather than interpreted as a physical fault.
- Only three of seven final synthetic scenarios could be safely placed under
  the unchanged evaluation-only, daylight, gap, and protected-transition rules;
  the remaining four are explicitly recorded as skipped.
- The frozen detector produced no alerts on the three placed events (0/21 direct
  rows and 0/3 events). This negative prototype result is retained without
  changing the detector or threshold.
- The synthetic prototype contains controlled perturbations; their future
  detection performance will not establish performance on real photovoltaic faults.
- The adopted 336/336/337-row split is suitable only for this internship
  prototype; approximately 33.6 hours and one complete daily cycle cannot
  provide independent daily partitions.
- `models/` contains the generated Day 16 scaler, initial Isolation Forest, and
  metadata; these artifacts are ignored by Git and reproducible from the
  adopted training matrix.
