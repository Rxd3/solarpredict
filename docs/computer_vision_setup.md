# Computer-vision setup

## Day 24 outcome

The real Roboflow version 1 dataset was cleaned conservatively and made ready for
the Day 25 prototype training attempt. Its final active inventory is 795 paired
images/labels, six classes, and 5,751 valid boxes. This document records the
pre-training state; actual Day 25 results are in
[computer_vision_training.md](computer_vision_training.md).

The sample figure at
`outputs/computer_vision/figures/day24_dataset_samples.png` was regenerated from
only the final active dataset and includes examples covering all six classes.

## Verified environment and final dry run

| Item | Verified result |
| --- | --- |
| Python | 3.12.4 |
| PyTorch | 2.14.0+cpu |
| Ultralytics | 8.4.138 |
| CUDA available | No |
| Checkpoint | `yolo11n.pt` resolved |
| Split counts | 582 train / 115 validation / 98 test |
| Classes | All six loaded and present in every split |
| Active data-quality problems | 0 |
| `model.train()` called | No |
| Dry-run readiness | `true`, no errors |

## Final Day 25 configuration

| Setting | Value |
| --- | ---: |
| Model | `yolo11n.pt` |
| Epochs | 30 |
| Image size | 512 |
| Batch size | 8 |
| Patience | 8 |
| Device | CPU |
| Workers | 0 |
| Seed | 42 |
| Save checkpoints | Yes |
| Periodic save | Every 5 epochs |

CPU-only training may take significant time; no duration is claimed because no
training benchmark has been run. Ultralytics will preserve `best.pt` and
`last.pt`, plus a periodic checkpoint every five epochs. The training entry
point supports `--resume`, which loads the configured run's `last.pt` if an
interrupted run is resumed later.

Day 24 verification command:

```powershell
.\.venv\Scripts\python.exe -m src.computer_vision.train_detector --dry-run
```

Future Day 25 commands (do not run during Day 24):

```powershell
# Start the configured run
.\.venv\Scripts\python.exe -m src.computer_vision.train_detector

# Resume only after last.pt exists
.\.venv\Scripts\python.exe -m src.computer_vision.train_detector --resume
```

## Remaining limitations

The 8.93:1 largest/smallest box imbalance remains. Minority classes have
prototype training support but production adequacy is unproven. Generalization
to drone altitude, glare, motion, new sites, and new panel types also remains
unverified. Training, trained-model evaluation, inference, video processing,
and dashboard integration are future work.
