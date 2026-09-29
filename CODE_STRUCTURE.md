# Code structure

The folders follow the experiment workflow shown in the paper. Run every
module from the repository root with `python -m <folder>.<module>`. Number gaps
are intentional: exploratory or superseded files were excluded.

## Data preparation

| Module | Role |
|---|---|
| `data_preparation.01_inspect_dataset` | Inspect the registered DTLD annotation hierarchy and class counts. |
| `data_preparation.02_session_class_summary` | Summarize image/object counts by route. |
| `data_preparation.03_make_split` | Recreate the deterministic route-disjoint split search. |
| `data_preparation.08_check_pixel_range` | Audit sampled TIFF bit depth and pixel range. |
| `data_preparation.12_build_full_yolo_dataset` | Convert DTLD to the four-class 8-bit YOLO dataset. |
| `data_preparation.13_validate_full_dataset` | Validate converted images, labels, classes, and counts. |
| `data_preparation.32_make_degradation_previews` | Optionally render degradation previews. |
| `data_preparation.33_build_degraded_test_sets` | Build the nine fixed synthetic-degradation sets. |
| `data_preparation.34_validate_degraded_test_sets` | Validate degraded images and copied labels. |

## Model training

| Module | Role |
|---|---|
| `model_training.19_train_yolov8_final` | Train YOLOv8n with the paper settings. |
| `model_training.20_train_yolo11_final` | Train YOLO11n with identical settings. |
| `model_training.21_train_yolo12_final` | Train YOLO12n with identical settings. |

## Model evaluation

| Module | Role |
|---|---|
| `model_evaluation.22_test_all_models` | Run clean test-set frame-level evaluation. |
| `model_evaluation.23_analyze_test_sequences` | Audit test sequence and timestamp structure. |
| `model_evaluation.24_export_test_predictions` | Export clean per-frame predictions. |
| `model_evaluation.35_evaluate_degraded_test_sets` | Evaluate all clean and degraded conditions. |
| `model_evaluation.37_export_degraded_temporal_predictions` | Export degraded predictions for temporal analysis. |

## Temporal analysis

| Module | Role |
|---|---|
| `temporal_analysis.25_analyze_temporal_metrics` | Shared GT/prediction parsing and matching helpers. |
| `temporal_analysis.45_recalculate_corrected_temporal_metrics` | Compute the primary AUC5 and complete red-miss metrics. |
| `temporal_analysis.47_corrected_temporal_sensitivity` | Run confidence, IoU, and 1/2/3-second sensitivity analyses. |
| `temporal_analysis.48_sequence_clustered_temporal_statistics` | Run 100,000 two-sided sequence-cluster sign-flip tests with Holm correction. |

## Shared code and visualization

| Module | Role |
|---|---|
| `common.project_paths` | Portable repository paths and model-weight lookup. |
| `visualization.generate_fig4_step_no_ci_svg` | Regenerate the paper's onset-curve figure. |
