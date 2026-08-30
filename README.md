# DAGL-MIL

DAGL-MIL is a breast-level weakly supervised framework for paired-view
mammography classification and lesion localization. A sample is one breast
with its available ipsilateral CC and MLO views. The network predicts two
independent targets—mass and suspicious calcification—and reconstructs
class-specific full-resolution response maps without using lesion boxes during
training.

## Method overview

The implementation follows one deterministic data-to-result path:

1. DICOM polarity, laterality, breast foreground and P1–P99 intensity are
   standardized.
2. Each view is represented by a variable-length bag of 512 × 512 patches;
   patches below 50% breast coverage are excluded.
3. A frozen Mammo-CLIP EfficientNet-B2 encoder produces 352-dimensional patch
   features.
4. Anatomical descriptors, class-conditional prototypes, a sparse dynamic
   graph and directional cross-view consistency refine candidate tokens.
5. Top-K MIL aggregation produces breast-level mass and calcification scores.
6. Overlapping candidate responses are averaged into heatmaps and converted to
   connected-component localization boxes for post-hoc evaluation.

The complete mapping from manuscript concepts to source files is provided in
[`docs/METHOD_CODE_MAP.md`](docs/METHOD_CODE_MAP.md).

## Repository structure

```text
configs/                 DAGL, five baselines and seven ablations
src/dagl_mil/data/       manifest, DICOM preprocessing and patch bags
src/dagl_mil/models/     DAGL-MIL, baselines and frozen Mammo-CLIP encoder
src/dagl_mil/engine/     training, checkpoints, feature encoding and inference
src/dagl_mil/evaluation/ classification, localization, bootstrap and efficiency
scripts/                 data gates, cache builders and experiment wrappers
tests/                   unit and protocol-level regression tests
artifacts/splits/        split counts, anomalies and public assignment snapshot
```

## Data and weights

Download VinDr-Mammo 1.0.0 from
[PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/) under its own data
use terms. Download the EfficientNet-B2 checkpoint from the official
[Mammo-CLIP project](https://github.com/batmanlab/Mammo-CLIP) under its upstream
license. Data and weights are deliberately excluded from Git.

Place or link them at the defaults below, or edit `configs/dagl.yaml`:

```text
data/vindr-mammo-1.0.0/
weights/model-epoch-10.repacked.pt
```

## Installation

The formal experiments target Python 3.10+ and a CUDA-capable NVIDIA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Reproduce the experiment

Build the examination-grouped seed-2026 split and deterministic preprocessing
cache:

```bash
dagl-mil prepare
python scripts/precache_vindr.py --workers 8
python scripts/audit_patch_feasibility.py --workers 8
python scripts/validate_data_gates.py
```

Run a real-DICOM smoke test:

```bash
dagl-mil smoke --resume --skip-completed
```

Train, infer and evaluate DAGL-MIL:

```bash
dagl-mil train --config configs/dagl.yaml --resume --skip-completed
dagl-mil infer --config configs/dagl.yaml --skip-completed
dagl-mil evaluate --config configs/dagl.yaml --skip-completed
dagl-mil efficiency --config configs/dagl.yaml --skip-completed
```

Run the complete comparison matrix:

```bash
dagl-mil run-all --resume --skip-completed
```

Additional protocol details and expected artifacts are documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Tests

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

Tests using the complete VinDr-Mammo tree are skipped when the dataset is not
available. Formal training commands fail closed when their CUDA and protocol
requirements are not satisfied.

## Third-party components

The frozen image encoder contains code derived from Mammo-CLIP and
EfficientNet-PyTorch. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`MAMMO_CLIP_LICENSE_NOTICE.txt`](MAMMO_CLIP_LICENSE_NOTICE.txt). The repository
does not redistribute Mammo-CLIP weights or VinDr-Mammo data.

## License status

This repository is being prepared privately. An explicit project license will
be selected before public release; until then, all rights are reserved. The
third-party files retain their original upstream terms.

