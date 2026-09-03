# Solar Panel Monitoring and Fault Detection System

University internship project for **TECHNOVOLT**. The long-term objective is to
build a system that brings photovoltaic operating data and visual panel
inspections together, making it easier to monitor performance and identify
possible faults.

This repository currently contains the project scaffold plus the completed
operational-dataset acquisition, inspection, data-quality, and initial
exploratory-analysis stage. No anomaly-detection model, computer-vision model,
video pipeline, or dashboard has been implemented or trained yet.

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
|-- src/
|   |-- data_processing/   # Dataset inspection, loading, and quality utilities
|   |-- anomaly_detection/ # Future time-series modeling code
|   |-- computer_vision/   # Future image/video modeling code
|   `-- dashboard/         # Future Streamlit application code
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
- basic loader and inspection tests.

Not completed yet:

- feature engineering;
- synthetic anomaly generation;
- anomaly-detection implementation or training;
- computer-vision dataset acquisition or training;
- drone/video processing;
- web dashboard implementation.

Operational-stage artifacts:

- [Verified schema and findings](docs/operational_dataset.md)
- [Data-quality JSON](outputs/operational_data_quality.json)
- [EDA notebook](notebooks/01_operational_data_exploration.ipynb)
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
2. **Operational data preparation and exploratory analysis (initial scope completed)**
   - Define schemas and validation rules.
   - Clean timestamps, missing values, duplicates, and invalid sensor readings.
   - Keep source timestamps timezone-naive unless better metadata is obtained.
   - Review image quality, verified class definitions, annotations, and label
     balance while preserving the source split.
3. **Operational baseline and anomaly detection**
   - Establish transparent baseline methods and time-aware evaluation splits.
   - Define reproducible synthetic anomaly scenarios separately from the normal
     source data.
   - Train and evaluate anomaly-detection candidates without data leakage.
4. **Computer-vision modeling**
   - Validate the downloaded YOLO annotations and `data.yaml`.
   - Train and evaluate object-detection models using the preserved source
     train/validation/test split.
5. **Drone/video inference pipeline**
   - Extract frames, track provenance, and manage duplicate or near-duplicate
     frames.
6. **Dashboard and integration**
   - Present monitoring data and validated model outputs in Streamlit.
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
- The computer-vision class mapping and split paths remain unverified until the
  downloaded `data.yaml` is inspected.
- The repository contains no trained models or experimental results.
- The `models/` and `outputs/` directories are placeholders for later stages.
