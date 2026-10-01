# Model weights

The trained weights are intentionally excluded from the Git repository. If redistribution is permitted, upload the following files as **GitHub Release assets**, not as regular Git-tracked files.

| Release asset | Size (bytes) | SHA-256 |
|---|---:|---|
| `yolov8n_best.pt` | 6,302,378 | `C30BE70E15F529324FBAEF6BB91BD383B7E86FCDCCC5FFE848125107316B5778` |
| `yolo11n_best.pt` | 5,524,954 | `8A3C6FCA1D367C7F9F7C2254F6DC7B13206581FC9D872B9890D32736FCFB95B1` |
| `yolo12n_best.pt` | 5,571,930 | `BB600BEF7B1290A70FADA35A1D72D15DD04B24EDBA4E133DE26B1E48018ED332` |

After downloading the release assets, place them in this directory:

```text
weights/
├── yolov8n_best.pt
├── yolo11n_best.pt
└── yolo12n_best.pt
```

`src/temporal_robustness/repository_paths.py` first looks for these filenames in `DTLD_WEIGHTS_ROOT` (this directory by default). If a released file is absent, it falls back to the corresponding locally trained `outputs/full_training/<model>_final/weights/best.pt`.

Verify on PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 weights\yolov8n_best.pt
Get-FileHash -Algorithm SHA256 weights\yolo11n_best.pt
Get-FileHash -Algorithm SHA256 weights\yolo12n_best.pt
```

Verify on Linux or macOS:

```bash
sha256sum weights/yolov8n_best.pt weights/yolo11n_best.pt weights/yolo12n_best.pt
```

The files in the local submission package are staged under `release_assets/model_weights/`. Confirm the hashes again immediately before creating the release.

## Redistribution notice

These weights were produced using Ultralytics software and pretrained model initialization. Their presence in the local submission package is not, by itself, a statement that public redistribution is authorized. Before publishing them, confirm the applicable Ultralytics terms, the selected repository license, the paper venue’s policy, and any institutional requirements. If redistribution is uncertain, publish the training scripts and checksums only and instruct reviewers to retrain the models.
