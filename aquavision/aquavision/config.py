import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AQUAVISION_DATA", ROOT / "data"))
FRONTEND_DIR = ROOT / "frontend"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
MAX_UPLOAD_MB = 500

# Defaults mirror the original desktop app ("Balanced" mode: conf 0.28, imgsz 576).
DEFAULT_SETTINGS = {
    "confidence": 0.28,
    "enhance_default": False,
    "class_mode": "auto",   # "auto" = original behaviour, "all" = every class the weights know
    "video_stride": 2,      # run the model on every Nth video frame
}
IMGSZ = 576
MAX_DET = 50

# Weight search order. Override with AQUAVISION_WEIGHTS=/path/to/best.pt
WEIGHT_CANDIDATES = [
    ROOT / "models" / "best.pt",
    ROOT / "runs" / "train" / "diver_detector" / "weights" / "best.pt",
    ROOT / "models" / "yolo11s.pt",
    ROOT / "models" / "yolo11n.pt",
]
