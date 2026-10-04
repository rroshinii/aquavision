"""YOLO inference wrapper. Reuses the model and logic from the original underwater.py."""
import os
import threading
from pathlib import Path

import cv2

from . import config

# Same alias set as the original desktop app: COCO "person" is what a diver looks like.
PERSON_ALIASES = {"person", "diver", "swimmer", "human"}
STOCK_NAMES = {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"}

_PALETTE = [(212, 227, 95), (92, 190, 255), (122, 112, 255), (140, 220, 120),
            (255, 170, 90), (230, 120, 220), (90, 230, 240), (130, 150, 255)]


class ModelUnavailable(RuntimeError):
    pass


def resolve_weights():
    env = os.environ.get("AQUAVISION_WEIGHTS")
    if env:
        p = Path(env)
        if not p.exists():
            raise ModelUnavailable(f"AQUAVISION_WEIGHTS points to a missing file: {p}")
        return p
    for p in config.WEIGHT_CANDIDATES:
        if p.exists():
            return p
    raise ModelUnavailable("No model weights found. Put a YOLO .pt file in models/ (see README).")


class Detector:
    def __init__(self, weights=None):
        self._weights = Path(weights) if weights else None
        self._model = None
        self._lock = threading.Lock()
        self.device = "cpu"
        self.load_error = None

    # ---- loading -------------------------------------------------------
    def load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from ultralytics import YOLO
            except Exception as exc:
                self.load_error = f"ultralytics/torch not importable: {exc}"
                raise ModelUnavailable(self.load_error)
            path = self._weights or resolve_weights()
            self.device = 0 if torch.cuda.is_available() else "cpu"
            try:
                self._model = YOLO(str(path))
                self._weights = path
            except Exception as exc:
                self.load_error = f"Could not load {path.name}: {exc}"
                raise ModelUnavailable(self.load_error)

    @property
    def names(self):
        self.load()
        return dict(self._model.names)

    def person_only(self, mode):
        """Original behaviour: with stock COCO weights only people (divers) are reported."""
        n = {v.lower() for v in self.names.values()}
        return mode == "auto" and "person" in n and "diver" not in n

    def classes(self, mode="auto"):
        if self.person_only(mode):
            return ["Diver"]
        return list(self.names.values())

    def info(self, mode="auto"):
        self.load()
        stock = self._weights.name in STOCK_NAMES
        return {
            "weights": self._weights.name,
            "task": getattr(self._model, "task", "detect"),
            "device": "cuda" if self.device != "cpu" else "cpu",
            "total_model_classes": len(self.names),
            "classes": self.classes(mode),
            "class_mode": mode,
            "stock_coco_weights": stock,
        }

    # ---- inference -----------------------------------------------------
    def predict(self, bgr, conf, mode="auto"):
        self.load()
        keep_people = self.person_only(mode)
        with self._lock:
            res = self._model.predict(
                bgr, conf=conf, imgsz=config.IMGSZ, max_det=config.MAX_DET,
                device=self.device, half=self.device != "cpu", verbose=False,
            )[0]
        out = []
        for b in res.boxes:
            name = self._model.names[int(b.cls[0])]
            if keep_people:
                if name.lower() not in PERSON_ALIASES:
                    continue
                name = "Diver"
            x1, y1, x2, y2 = (round(float(v), 1) for v in b.xyxy[0].tolist())
            out.append({"class": name, "confidence": round(float(b.conf[0]), 4), "bbox": [x1, y1, x2, y2]})
        return out


def color_for(name):
    return _PALETTE[sum(map(ord, name)) % len(_PALETTE)]


def annotate(bgr, detections):
    out = bgr.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 900)
    thick = max(2, int(round(2 * scale)))
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        c = color_for(d["class"])
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thick)
        label = f'{d["class"]} {d["confidence"]:.2f}'
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, thick - 1 or 1)
        top = max(0, y1 - th - base - 6)
        cv2.rectangle(out, (x1, top), (x1 + tw + 8, top + th + base + 6), c, -1)
        cv2.putText(out, label, (x1 + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (20, 20, 20), max(1, thick - 1), cv2.LINE_AA)
    return out
