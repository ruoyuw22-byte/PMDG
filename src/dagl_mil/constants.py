import os
from pathlib import Path

CLASS_NAMES = ("mass", "calcification")
PROTOTYPE_NAMES = ("normal", "mass", "calcification")
VIEW_NAMES = ("CC", "MLO")

PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("DAGL_MIL_PROJECT_ROOT", PACKAGE_PROJECT_ROOT)
).expanduser()
DEFAULT_VINDR_ROOT = Path(
    os.environ.get(
        "DAGL_MIL_VINDR_ROOT",
        DEFAULT_PROJECT_ROOT / "data" / "vindr-mammo-1.0.0",
    )
).expanduser()
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "DAGL_MIL_BACKBONE_CHECKPOINT",
        DEFAULT_PROJECT_ROOT / "weights" / "model-epoch-10.repacked.pt",
    )
).expanduser()

EXPECTED_SPLIT_COUNTS = {
    "train": {"breasts": 6400, "mass": 393, "calcification": 139, "both": 71},
    "val": {"breasts": 1600, "mass": 97, "calcification": 36, "both": 18},
    "test": {"breasts": 2000, "mass": 119, "calcification": 54, "both": 22},
}
