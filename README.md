# Temporal Robustness Evaluation of YOLO-Based Traffic-Light Detection under Synthetic Degradation

[한국어 안내](README_ko.md)

This repository is the reproducibility package for the paper **“Temporal Robustness Evaluation of YOLO-Based Traffic-Light Detection under Synthetic Degradation.”** It compares YOLOv8n, YOLO11n, and YOLO12n on the DriveU Traffic Light Dataset (DTLD), under clean images and nine synthetic degradation settings.

The repository contains the paper split manifests, data-preparation and evaluation code, compact result tables, and training metadata. It intentionally does **not** contain DTLD images, converted images, synthetically degraded images, per-frame prediction dumps, or model weights.

## What is reported in the paper

- Four circular traffic-light states: `red`, `yellow`, `green`, and `red_yellow`
- Route-disjoint split: 34 train routes, 4 validation routes, and 5 test routes
- Images: 32,699 train, 4,035 validation, and 4,244 test
- Models: YOLOv8n, YOLO11n, and YOLO12n
- Training: image size 1280, batch size 4, AdamW, initial learning rate 0.00125, weight decay 0.0005, seed 42, deterministic mode, AMP, at most 50 epochs, and early-stopping patience 10
- Synthetic degradation:
  - fog alpha: 0.15, 0.30, 0.45
  - low-light gamma: 1.4, 2.0, 2.8
  - horizontal motion-blur kernel: 3, 7, 11 pixels
- Primary temporal settings: confidence 0.25, matching IoU 0.50, 2-second stable-detection window, 80% success threshold, at least 3 observations, maximum 2-second observation gap, and 5-second AUC horizon
- Sensitivity settings: confidence 0.10/0.25/0.50, IoU 0.30/0.40/0.50, and stable-detection windows of 1/2/3 seconds
- Final inference: 100,000 two-sided sequence-cluster sign-flip permutations with Holm correction

The two temporal summaries are:

- **AUC5**: normalized area under the cumulative stable-detection-onset curve from 0 to 5 seconds; higher is better.
- **Mean longest complete red-miss duration** (D-bar): for each track, measure its longest complete run of missed red detections and then average across tracks; lower is better.

## Repository contents

```text
.
├── configs/                     # Human-readable experiment and path templates
├── common/                      # Shared portable path configuration
├── data_preparation/            # Dataset inspection, conversion, and degradation
├── figures/                     # Regenerated SVG figures from included metrics
├── model_evaluation/            # Clean/degraded evaluation and prediction export
├── model_training/              # YOLOv8n, YOLO11n, and YOLO12n training
├── results/
│   ├── framewise/               # Compact clean-test summary
│   ├── temporal/
│   │   ├── track_metrics/       # Corrected per-track AUC5/red-miss inputs
│   │   ├── sensitivity/         # Confidence, IoU, and 1/2/3-second checks
│   │   └── sequence_clustered/  # Final paper-level clustered inference
│   └── training_metadata/       # Ultralytics arguments and epoch histories
├── splits/                      # Exact route lists used in the paper
├── temporal_analysis/           # AUC5, red-miss, sensitivity, and inference
├── tests/                       # Regression tests for core metric definitions
├── visualization/               # Figure-generation code
├── weights/README.md            # Optional release-asset names and checksums
├── CODE_STRUCTURE.md            # Full numbered-code map
├── CITATION.cff
├── requirements.txt
├── README.md
├── README_ko.md
├── VERIFICATION.md
└── THIRD_PARTY_NOTICES.md
```

The checked-in result tables are small verification artifacts, not substitutes for DTLD. They allow the final statistical stage to be inspected without redistributing the dataset or large prediction files.

## Data access and redistribution

DTLD is provided by Ulm University. Obtain it from the official pages and comply with the dataset terms:

- Dataset page: <https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/>
- Registration form and terms: <https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/registration-form-dtld/>
- Official parser: <https://github.com/julimueller/dtld_parsing>

**Do not commit or redistribute DTLD images or annotations through this repository.** Each user must obtain access from the dataset provider. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

By default, the scripts expect:

```text
data/raw/
└── DTLD_Labels_v2.0/
    └── v2.0/
        └── DTLD_all.json
```

The TIFF files must be available under the root referenced by the paths inside `DTLD_all.json`. If your layout differs, set `DTLD_RAW_ROOT` and `DTLD_LABEL_JSON` explicitly.

## Installation

Create an isolated Python environment. Install a PyTorch build compatible with your CUDA driver first, then install the pinned project dependencies:

```bash
python -m pip install -r requirements.txt
```

Training and model-inference scripts require an NVIDIA CUDA GPU and stop when CUDA is unavailable. Dataset inspection and scripts 45, 47, and 48 can run on CPU once their required inputs exist.

Ultralytics 8.4.108 was used for the reported experiment. Exact software settings are recorded in `configs/experiment.yaml` and the files under `results/training_metadata/`.

## Portable paths

All paths are centralized in `common/project_paths.py`. Defaults remain inside the repository; environment variables can point to data and output locations on any machine.

| Variable | Default | Purpose |
|---|---|---|
| `DTLD_RAW_ROOT` | `data/raw` | Registered DTLD download root |
| `DTLD_LABEL_JSON` | `data/raw/DTLD_Labels_v2.0/v2.0/DTLD_all.json` | DTLD v2 annotation JSON |
| `DTLD_DATASET_ROOT` | `data/processed/DTLD_YOLO_4class` | Converted YOLO dataset |
| `DTLD_DEGRADED_ROOT` | `data/processed/DTLD_YOLO_4class_degraded` | Nine degraded test sets |
| `DTLD_OUTPUT_ROOT` | `outputs` | Training, inference, and analysis outputs |
| `DTLD_WEIGHTS_ROOT` | `weights` | Optional released `*_best.pt` files |
| `DTLD_CORRECTED_TEMPORAL_OUTPUT_ROOT` | `outputs/corrected_temporal_analysis` | Script 45 output |
| `DTLD_CORRECTED_SENSITIVITY_OUTPUT_ROOT` | `outputs/corrected_temporal_analysis/sensitivity` | Script 47 output |
| `DTLD_SEQUENCE_CLUSTER_OUTPUT_ROOT` | `outputs/sequence_clustered_statistics` | Script 48 final output |
| `DTLD_CLUSTER_PERMUTATION_ITERATIONS` | `100000` | Two-sided sequence-cluster sign-flip count |

Example, PowerShell:

```powershell
$env:DTLD_RAW_ROOT = "<path-to-DTLD>"
$env:DTLD_LABEL_JSON = "<path-to-DTLD-all-json>"
$env:DTLD_OUTPUT_ROOT = "<path-to-output-directory>"
```

Example, POSIX shell:

```bash
export DTLD_RAW_ROOT=/data/DTLD
export DTLD_LABEL_JSON=/data/DTLD/DTLD_Labels_v2.0/v2.0/DTLD_all.json
export DTLD_OUTPUT_ROOT=/data/dtld_outputs
```

## Reproduction workflow

Run commands from the repository root. The numbered filenames show the original experimental order; gaps correspond to exploratory or superseded scripts that were deliberately excluded.

### 1. Inspect DTLD and retain the paper split

```bash
python -m data_preparation.01_inspect_dataset
python -m data_preparation.02_session_class_summary
python -m data_preparation.08_check_pixel_range
```

The exact route-disjoint split used in the paper is already stored in `splits/`. Use those files for strict reproduction. `03_make_split.py` is included to document how the split search was performed, but it writes the manifest files; run it only in a clean copy when auditing split generation:

```bash
python -m data_preparation.03_make_split
```

### 2. Convert and validate the four-class YOLO dataset

```bash
python -m data_preparation.12_build_full_yolo_dataset
python -m data_preparation.13_validate_full_dataset
```

The conversion maps the 16-bit TIFF input to 8-bit single-channel PNG while preserving the four paper classes and route-disjoint manifests.

### 3. Train or provide the three final weights

To retrain:

```bash
python -m model_training.19_train_yolov8_final
python -m model_training.20_train_yolo11_final
python -m model_training.21_train_yolo12_final
```

Alternatively, obtain the three optional GitHub Release assets described in [weights/README.md](weights/README.md) and place them in `weights/`. The main Git repository should not contain the `.pt` files.

### 4. Evaluate clean images and export temporal predictions

```bash
python -m model_evaluation.22_test_all_models
python -m model_evaluation.23_analyze_test_sequences
python -m model_evaluation.24_export_test_predictions
```

Predictions are exported from confidence 0.01 so later analyses can apply confidence thresholds of 0.10, 0.25, and 0.50 without rerunning inference.

### 5. Build, validate, and evaluate the degraded test sets

`32_make_degradation_previews.py` is optional and creates a visual check. The remaining scripts generate and verify all nine degraded sets, run frame-level evaluation, and export temporal predictions.

```bash
python -m data_preparation.32_make_degradation_previews
python -m data_preparation.33_build_degraded_test_sets
python -m data_preparation.34_validate_degraded_test_sets
python -m model_evaluation.35_evaluate_degraded_test_sets
python -m model_evaluation.37_export_degraded_temporal_predictions
```

### 6. Recalculate the corrected temporal metrics

```bash
python -m temporal_analysis.45_recalculate_corrected_temporal_metrics
```

Script 45 applies the paper’s primary definition, including continuous-window coverage and the 2-second maximum gap. Script 25 is retained because script 45 imports its parsing and matching helpers; script 25’s older exploratory window list is **not** the source of the paper’s final 1/2/3-second sensitivity result.

### 7. Run sensitivity analysis

```bash
python -m temporal_analysis.47_corrected_temporal_sensitivity
```

Script 47 is the definitive sensitivity analysis for 1-, 2-, and 3-second stable-detection windows, as well as the reported confidence and IoU settings.

### 8. Run the final sequence-clustered inference

```bash
python -m temporal_analysis.48_sequence_clustered_temporal_statistics
```

Script 48 is the authoritative paper-level analysis. It clusters tracks by driving sequence, performs 100,000 two-sided sequence-level sign-flip permutations, and applies Holm correction separately within the direct-comparison and Clean-change families for each metric. No confidence interval is computed. The checked-in files under `results/temporal/sequence_clustered/` were produced with those settings.

To make the iteration counts explicit:

```powershell
$env:DTLD_CLUSTER_PERMUTATION_ITERATIONS = "100000"
python -m temporal_analysis.48_sequence_clustered_temporal_statistics
```

```bash
DTLD_CLUSTER_PERMUTATION_ITERATIONS=100000 \
python -m temporal_analysis.48_sequence_clustered_temporal_statistics
```

### 9. Regenerate the paper curve figure

```bash
python -m visualization.generate_fig4_step_no_ci_svg
```

## Verify the included statistical artifacts without DTLD

The corrected per-track CSV files are included, so the final statistical stage can be checked without raw images or prediction JSONL files. Write regenerated files to `outputs/` to keep the checked-in reference tables unchanged.

PowerShell:

```powershell
$env:DTLD_SEQUENCE_CLUSTER_OUTPUT_ROOT = "outputs\sequence_clustered_statistics"
$env:DTLD_CLUSTER_PERMUTATION_ITERATIONS = "100000"
python -m temporal_analysis.48_sequence_clustered_temporal_statistics
```

POSIX shell:

```bash
DTLD_SEQUENCE_CLUSTER_OUTPUT_ROOT=outputs/sequence_clustered_statistics \
DTLD_CLUSTER_PERMUTATION_ITERATIONS=100000 \
python -m temporal_analysis.48_sequence_clustered_temporal_statistics
```

Compare the four regenerated CSV/JSON files with `results/temporal/sequence_clustered/`.

## Included and intentionally excluded artifacts

Included:

- exact paper split manifests and split summary;
- source scripts required for data preparation, training, evaluation, corrected temporal analysis, sensitivity analysis, and final clustered inference;
- compact clean-test, per-track, sensitivity, and sequence-clustered result tables;
- final training arguments and epoch histories;
- checksums and placement instructions for optional trained weights.

Intentionally excluded:

- all DTLD files and annotations;
- converted YOLO images/labels and all nine degraded image sets;
- per-image/per-frame JSONL predictions and bulky evaluation runs;
- caches, temporary files, preview exports, logs, and machine-specific paths;
- intermediate and `last.pt` checkpoints;
- presentation files, manuscript PDFs, screenshots, and local archives;
- superseded exploratory scripts 26–31 and 38–44, including the outdated package builder and pre-correction temporal statistics.

These exclusions keep the repository reviewable without removing anything required to understand or rerun the reported pipeline.

## Model weights

The three `best.pt` files are kept outside the Git repository under `release_assets/model_weights/` in the submission package. If redistribution is permitted, attach them to a GitHub Release and verify their SHA-256 values using [weights/README.md](weights/README.md). Do not commit them to the main Git history.

## License status

No open-source license has been selected for this package. A `LICENSE` file is therefore intentionally absent. Before making the repository public, the authors must select a suitable code license and confirm that every distributed artifact, especially trained weights, can be redistributed under the applicable third-party terms. The absence of a license does not grant permission to reuse or redistribute the code beyond applicable law or explicit paper-review arrangements.

## Citation

Use the paper citation once its final bibliographic information is available. `CITATION.cff` contains only the currently known title and author metadata; it does not invent a DOI or repository URL.
