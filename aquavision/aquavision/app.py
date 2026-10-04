"""AquaVision Flask backend."""
import csv
import io
import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from . import config
from .detector import Detector, ModelUnavailable, annotate
from .enhance import enhance_underwater
from .store import Store


def summarize(dets):
    by = {}
    for d in dets:
        by.setdefault(d["class"], []).append(d["confidence"])
    objects = [{"name": k, "count": len(v), "avg_confidence": round(sum(v) / len(v), 4),
                "max_confidence": round(max(v), 4)} for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))]
    confs = [d["confidence"] for d in dets]
    hist = [0] * 10
    for c in confs:
        hist[min(int(c * 10), 9)] += 1
    return {"objects": objects, "hist": hist, "total": len(dets),
            "avg_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "max_confidence": round(max(confs), 4) if confs else None}


def create_app(detector=None, data_dir=None):
    data = Path(data_dir or config.DATA_DIR)
    for sub in ("uploads", "results"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    store = Store(data / "aquavision.db")
    det = detector or Detector()
    jobs = {}

    app = Flask(__name__, static_folder=str(config.FRONTEND_DIR), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

    # ---- helpers -----------------------------------------------------------
    def err(msg, code=400):
        return jsonify({"error": msg}), code

    def public(rec):
        rec = dict(rec)
        for k in ("original", "enhanced", "result"):
            rec[k + "_url"] = f"/media/{rec[k]}" if rec.get(k) else None
            rec.pop(k, None)
        return rec

    def read_settings_form():
        s = store.settings()
        conf = float(request.form.get("confidence", s["confidence"]))
        enh = request.form.get("enhance", "1" if s["enhance_default"] else "0") in ("1", "true", "on")
        return s, min(max(conf, 0.01), 0.99), enh

    def decode_upload(f, exts):
        if not f or not f.filename:
            raise ValueError("No file was uploaded.")
        ext = Path(f.filename).suffix.lower()
        if ext not in exts:
            raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(exts))}")
        return ext

    def fail_record(kind, name, msg, ms=0):
        return store.add(input_type=kind, filename=name, status="failed", processing_ms=ms, error=msg)

    @app.errorhandler(HTTPException)
    def http_err(e):
        if request.path.startswith("/api"):
            msg = "File too large." if e.code == 413 else e.description
            return err(msg, e.code)
        return e

    @app.errorhandler(ModelUnavailable)
    def model_err(e):
        return err(str(e), 503)

    # ---- pages -------------------------------------------------------------
    @app.get("/")
    def landing():
        return send_from_directory(config.FRONTEND_DIR, "index.html")

    @app.get("/app")
    def dashboard():
        return send_from_directory(config.FRONTEND_DIR, "app.html")

    @app.get("/media/<kind>/<path:name>")
    def media(kind, name):
        if kind not in ("uploads", "results"):
            abort(404)
        return send_from_directory(data / kind, name, as_attachment=request.args.get("download") == "1",
                                   conditional=True)

    # ---- model / settings --------------------------------------------------
    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/model")
    def model_info():
        try:
            return jsonify(det.info(store.settings()["class_mode"]))
        except ModelUnavailable as e:
            return err(str(e), 503)

    @app.route("/api/settings", methods=["GET", "POST"])
    def settings():
        if request.method == "POST":
            b = request.get_json(silent=True) or {}
            clean = {}
            if "confidence" in b: clean["confidence"] = min(max(float(b["confidence"]), 0.01), 0.99)
            if "enhance_default" in b: clean["enhance_default"] = bool(b["enhance_default"])
            if "class_mode" in b:
                if b["class_mode"] not in ("auto", "all"): return err("class_mode must be 'auto' or 'all'")
                clean["class_mode"] = b["class_mode"]
            if "video_stride" in b: clean["video_stride"] = min(max(int(b["video_stride"]), 1), 30)
            return jsonify(store.save_settings(clean))
        return jsonify(store.settings())

    # ---- image -------------------------------------------------------------
    @app.post("/api/detect/image")
    def detect_image():
        f = request.files.get("file")
        try:
            ext = decode_upload(f, config.IMAGE_EXT)
        except ValueError as e:
            return err(str(e))
        raw = f.read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return err("This file could not be read as an image.")
        s, conf, enh = read_settings_form()
        uid = uuid.uuid4().hex[:12]
        orig = f"uploads/{uid}{ext}"
        (data / orig).write_bytes(raw)
        t0 = time.perf_counter()
        try:
            work = enhance_underwater(img) if enh else img
            enhanced = None
            if enh:
                enhanced = f"results/{uid}_enhanced.jpg"
                cv2.imwrite(str(data / enhanced), work, [cv2.IMWRITE_JPEG_QUALITY, 92])
            dets = det.predict(work, conf, s["class_mode"])
            result = f"results/{uid}_result.jpg"
            cv2.imwrite(str(data / result), annotate(work, dets), [cv2.IMWRITE_JPEG_QUALITY, 92])
        except ModelUnavailable as e:
            fail_record("image", f.filename, str(e))
            return err(str(e), 503)
        except Exception as e:  # inference failure
            fail_record("image", f.filename, f"Inference failed: {e}")
            return err(f"Detection failed: {e}", 500)
        ms = int((time.perf_counter() - t0) * 1000)
        sm = summarize(dets)
        sm.update({"detections": dets, "width": img.shape[1], "height": img.shape[0],
                   "enhanced": enh, "confidence_threshold": conf})
        rec = store.add(input_type="image", filename=f.filename, status="completed", total_objects=sm["total"],
                        avg_confidence=sm["avg_confidence"], max_confidence=sm["max_confidence"],
                        processing_ms=ms, summary=sm, original=orig, enhanced=enhanced, result=result)
        return jsonify(public(rec))

    # ---- live frame (webcam). Not stored: the client can save a snapshot via /api/detect/image ----
    @app.post("/api/detect/frame")
    def detect_frame():
        f = request.files.get("frame")
        if not f:
            return err("Missing 'frame'.")
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return err("Frame could not be decoded.")
        s, conf, enh = read_settings_form()
        t0 = time.perf_counter()
        work = enhance_underwater(img) if enh else img
        dets = det.predict(work, conf, s["class_mode"])
        h, w = img.shape[:2]
        return jsonify({"detections": dets, "width": w, "height": h,
                        "processing_ms": int((time.perf_counter() - t0) * 1000)})

    @app.post("/api/enhance")
    def enhance_preview():
        f = request.files.get("file")
        try:
            decode_upload(f, config.IMAGE_EXT)
        except ValueError as e:
            return err(str(e))
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return err("This file could not be read as an image.")
        ok, buf = cv2.imencode(".jpg", enhance_underwater(img), [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), mimetype="image/jpeg")

    # ---- video -------------------------------------------------------------
    def run_video(job_id, path, name, conf, enh, stride, mode):
        job = jobs[job_id]
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(str(path))
        writer = None
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            uid = path.stem
            raw_out = data / "results" / f"{uid}_raw.mp4"
            final = f"results/{uid}_result.mp4"
            all_dets, timeline, last, idx, first_hit, fallback = [], [], [], 0, None, None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] > 1280:
                    r = 1280 / frame.shape[1]
                    frame = cv2.resize(frame, (1280, int(frame.shape[0] * r)))
                if idx % stride == 0:
                    work = enhance_underwater(frame) if enh else frame
                    last = det.predict(work, conf, mode)
                    all_dets.extend(last)
                    timeline.append(len(last))
                    frame = work
                    if last and first_hit is None:
                        first_hit = annotate(frame, last)
                elif enh:
                    frame = enhance_underwater(frame)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                shown = annotate(frame, last)
                if fallback is None:
                    fallback = shown
                writer.write(shown)
                idx += 1
                job["progress"] = round(idx / total, 3) if total else 0.0
            if writer is None:
                raise RuntimeError("The video contained no readable frames.")
            writer.release(); writer = None
            # transcode to H.264 so browsers can play it; fall back to the mp4v file
            out_path = data / final
            try:
                import imageio_ffmpeg
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw_out),
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)],
                               check=True, timeout=1800)
                raw_out.unlink(missing_ok=True)
            except Exception:
                shutil.move(str(raw_out), str(out_path))
            thumb = f"results/{uid}_preview.jpg"
            cv2.imwrite(str(data / thumb), first_hit if first_hit is not None else fallback, [cv2.IMWRITE_JPEG_QUALITY, 88])
            sm = summarize(all_dets)
            sm.update({"frames_total": idx, "frames_analyzed": len(timeline), "timeline": timeline,
                       "fps": round(fps, 2), "frame_stride": stride, "enhanced": enh, "confidence_threshold": conf,
                       "note": "Counts are detections summed over analysed frames; one object seen in many frames is counted each time."})
            rec = store.add(input_type="video", filename=name, status="completed", total_objects=sm["total"],
                            avg_confidence=sm["avg_confidence"], max_confidence=sm["max_confidence"],
                            processing_ms=int((time.perf_counter() - t0) * 1000), summary=sm,
                            original=f"uploads/{path.name}", enhanced=thumb, result=final)
            job.update(status="completed", progress=1.0, record_id=rec["id"])
        except Exception as e:
            fail_record("video", name, str(e), int((time.perf_counter() - t0) * 1000))
            job.update(status="failed", error=str(e))
        finally:
            cap.release()
            if writer is not None:
                writer.release()

    @app.post("/api/detect/video")
    def detect_video():
        f = request.files.get("file")
        try:
            ext = decode_upload(f, config.VIDEO_EXT)
        except ValueError as e:
            return err(str(e))
        det.load()  # fail fast with a clear 503 if the model is unavailable
        path = data / "uploads" / f"{uuid.uuid4().hex[:12]}{ext}"
        f.save(path)
        probe = cv2.VideoCapture(str(path))
        ok = probe.isOpened() and probe.read()[0]
        probe.release()
        if not ok:
            path.unlink(missing_ok=True)
            return err("This file could not be read as a video.")
        s, conf, enh = read_settings_form()
        stride = min(max(int(request.form.get("stride", s["video_stride"])), 1), 30)
        job_id = uuid.uuid4().hex[:10]
        jobs[job_id] = {"status": "processing", "progress": 0.0}
        threading.Thread(target=run_video, args=(job_id, path, f.filename, conf, enh, stride, s["class_mode"]),
                         daemon=True).start()
        return jsonify({"job_id": job_id}), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        j = jobs.get(job_id)
        if not j:
            return err("Unknown job.", 404)
        out = dict(j)
        if j.get("record_id"):
            out["record"] = public(store.get(j["record_id"]))
        return jsonify(out)

    # ---- history / stats ---------------------------------------------------
    @app.get("/api/stats")
    def stats():
        return jsonify(store.stats())

    @app.get("/api/history")
    def history():
        limit = request.args.get("limit", type=int)
        return jsonify([public(r) | {"summary": {"objects": r["summary"].get("objects", [])}}
                        for r in store.list(limit, request.args.get("type"))])

    @app.get("/api/history/export.csv")
    def export_csv():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "created_at", "input_type", "filename", "status", "objects", "avg_confidence",
                    "max_confidence", "processing_ms", "classes"])
        for r in reversed(store.list()):
            w.writerow([r["display_id"], r["created_at"], r["input_type"], r["filename"], r["status"],
                        r["total_objects"], r["avg_confidence"], r["max_confidence"], r["processing_ms"],
                        "; ".join(f'{o["name"]}:{o["count"]}' for o in r["summary"].get("objects", []))])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=aquavision_history.csv"})

    @app.get("/api/history/<int:rid>")
    def history_one(rid):
        r = store.get(rid)
        return jsonify(public(r)) if r else err("Detection not found.", 404)

    @app.get("/api/history/<int:rid>/report.json")
    def report(rid):
        r = store.get(rid)
        if not r:
            return err("Detection not found.", 404)
        body = json.dumps(public(r), indent=2)
        return Response(body, mimetype="application/json",
                        headers={"Content-Disposition": f"attachment; filename={r['display_id']}.json"})

    def remove_files(recs):
        for r in recs:
            for k in ("original", "enhanced", "result"):
                if r.get(k):
                    (data / r[k]).unlink(missing_ok=True)

    @app.delete("/api/history/<int:rid>")
    def history_delete(rid):
        r = store.delete(rid)
        if not r:
            return err("Detection not found.", 404)
        remove_files([r])
        return jsonify({"deleted": rid})

    @app.delete("/api/history")
    def history_clear():
        recs = store.clear()
        remove_files(recs)
        return jsonify({"deleted": len(recs)})

    return app
