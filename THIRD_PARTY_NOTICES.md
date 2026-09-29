# Third-party notices

This file identifies the principal third-party data and software used by the experiment. It is informational and does not replace the original providers’ current licenses or terms.

## DriveU Traffic Light Dataset (DTLD)

The experiment uses the DriveU Traffic Light Dataset provided by Ulm University.

- Official dataset page: <https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/>
- Registration form and terms: <https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/registration-form-dtld/>
- Official parsing utilities: <https://github.com/julimueller/dtld_parsing>

No DTLD image or annotation file is distributed in this repository or in its intended GitHub Release. Users must obtain DTLD directly from the provider, accept the applicable terms, follow the provider’s citation requirements, and keep the data outside Git history. Converted PNG files, YOLO labels derived from DTLD, synthetic degraded variants, and per-frame prediction files are also excluded to avoid redistributing dataset-derived bulk content.

## Ultralytics and YOLO models

The experiment used the `ultralytics` Python package, version 8.4.108, and the YOLOv8n, YOLO11n, and YOLO12n model families.

- Project: <https://github.com/ultralytics/ultralytics>
- Documentation: <https://docs.ultralytics.com/>

Ultralytics software, pretrained initialization weights, and any trained derivative weight files remain subject to the applicable Ultralytics terms. This repository does not restate or modify those terms. Confirm redistribution rights before attaching the trained `.pt` files to a public release.

## Python ecosystem dependencies

The code also relies on PyTorch, NumPy, Pillow, and OpenCV. Each project is governed by its own license. The dependency list in `requirements.txt` does not grant any additional rights, and no third-party source code is vendored here.

## Project code

No open-source license has yet been selected for the paper-specific source code. A `LICENSE` file is intentionally absent until the authors make and document that decision. Nothing in this notice grants rights beyond those provided by the relevant copyright holders, licenses, dataset terms, applicable law, or explicit paper-review arrangements.
