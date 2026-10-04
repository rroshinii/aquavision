"""API tests. FakeDetector exists ONLY here, to test everything around the model without PyTorch."""
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aquavision.app import create_app
from aquavision.detector import ModelUnavailable


class FakeDetector:
    def load(self): pass
    def classes(self, mode="auto"): return ["Diver"]
    def info(self, mode="auto"):
        return {"weights": "fake.pt", "task": "detect", "device": "cpu", "total_model_classes": 1,
                "classes": ["Diver"], "class_mode": mode, "stock_coco_weights": False}
    def predict(self, bgr, conf, mode="auto"):
        h, w = bgr.shape[:2]
        return [{"class": "Diver", "confidence": 0.9, "bbox": [w * .2, h * .2, w * .6, h * .8]},
                {"class": "Diver", "confidence": 0.5, "bbox": [1, 1, 20, 20]}]


class BrokenDetector(FakeDetector):
    def load(self): raise ModelUnavailable("weights missing")
    def predict(self, *a, **k): raise ModelUnavailable("weights missing")
    def info(self, mode="auto"): raise ModelUnavailable("weights missing")


def png(w=160, h=120):
    ok, buf = cv2.imencode(".png", np.random.randint(0, 255, (h, w, 3), np.uint8))
    return io.BytesIO(buf.tobytes())


@pytest.fixture
def client(tmp_path):
    return create_app(FakeDetector(), tmp_path).test_client()


def test_pages_and_model(client):
    assert client.get("/").status_code == 200 and client.get("/app").status_code == 200
    assert client.get("/api/model").get_json()["classes"] == ["Diver"]


def test_image_flow_stats_history_delete(client):
    r = client.post("/api/detect/image", data={"file": (png(), "a.png"), "enhance": "1"})
    j = r.get_json(); assert r.status_code == 200, j
    assert j["total_objects"] == 2 and j["max_confidence"] == 0.9 and j["enhanced_url"]
    assert j["summary"]["objects"][0] == {"name": "Diver", "count": 2, "avg_confidence": 0.7, "max_confidence": 0.9}
    assert client.get(j["result_url"]).status_code == 200
    st = client.get("/api/stats").get_json()
    assert st["images_analyzed"] == 1 and st["total_detections"] == 2 and st["most_detected"]["name"] == "Diver"
    assert st["average_confidence"] == 0.7
    assert len(client.get("/api/history").get_json()) == 1
    assert b"AV-00001" in client.get("/api/history/export.csv").data
    assert client.get(f"/api/history/{j['id']}/report.json").status_code == 200
    assert client.delete(f"/api/history/{j['id']}").status_code == 200
    assert client.get("/api/stats").get_json()["total_detections"] == 0


def test_empty_stats_are_zero_not_fake(client):
    st = client.get("/api/stats").get_json()
    assert st["total_detections"] == 0 and st["most_detected"] is None and st["average_confidence"] is None


def test_bad_inputs(client):
    assert client.post("/api/detect/image", data={"file": (io.BytesIO(b"x"), "a.txt")}).status_code == 400
    assert client.post("/api/detect/image", data={"file": (io.BytesIO(b"notimage"), "a.png")}).status_code == 400
    assert client.post("/api/detect/image").status_code == 400
    assert client.get("/api/history/999").status_code == 404


def test_model_unavailable_is_clean_error(tmp_path):
    c = create_app(BrokenDetector(), tmp_path).test_client()
    r = c.post("/api/detect/image", data={"file": (png(), "a.png")})
    assert r.status_code == 503 and "weights missing" in r.get_json()["error"]
    assert c.get("/api/model").status_code == 503
    assert c.get("/api/stats").get_json()["failed"] == 1       # failure recorded, not hidden


def test_settings_roundtrip(client):
    s = client.post("/api/settings", json={"confidence": 0.5, "video_stride": 3, "class_mode": "all"}).get_json()
    assert s["confidence"] == 0.5 and s["video_stride"] == 3 and s["class_mode"] == "all"
    assert client.post("/api/settings", json={"class_mode": "bogus"}).status_code == 400


def test_live_frame(client):
    r = client.post("/api/detect/frame", data={"frame": (png(), "f.png")}).get_json()
    assert len(r["detections"]) == 2 and r["width"] == 160


def test_video_job(client, tmp_path):
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(12): w.write(np.random.randint(0, 255, (120, 160, 3), np.uint8))
    w.release()
    r = client.post("/api/detect/video", data={"file": (p.open("rb"), "v.mp4"), "stride": "3"})
    assert r.status_code == 202
    import time
    for _ in range(100):
        j = client.get(f"/api/jobs/{r.get_json()['job_id']}").get_json()
        if j["status"] != "processing": break
        time.sleep(0.2)
    assert j["status"] == "completed", j
    rec = j["record"]
    assert rec["summary"]["frames_total"] == 12 and rec["summary"]["frames_analyzed"] == 4 and rec["total_objects"] == 8
    assert client.get(rec["result_url"]).status_code == 200
    assert client.get("/api/stats").get_json()["videos_analyzed"] == 1
