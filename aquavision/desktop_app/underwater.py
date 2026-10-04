import importlib
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk

try:
    import torch
except Exception:  # pragma: no cover - runtime dependency check
    torch = None

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime dependency check
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None


REQUIRED_PACKAGES = [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
    ("torch", "torch"),
    ("ultralytics", "ultralytics"),
]


def ensure_dependencies():
    missing = []
    for package_name, module_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(package_name)

    if missing:
        message = "Installing required packages: " + ", ".join(missing)
        print(message)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *missing])


ensure_dependencies()


class DetectionProcessor:
    def __init__(self, callback=None):
        self.callback = callback
        self.model = None
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.running = False
        self.thread = None
        self.cap = None
        self.writer = None
        self.recording = False
        self.record_path = None
        self.frame = None
        self.frame_ready = False
        self.source_type = "camera"
        self.source = 0
        self.fps = 20.0
        self.model_name = "yolo11m.pt"
        self.confidence = 0.28
        self.imgsz = 640
        self.max_det = 100
        self.mode = "Balanced"

    def load_model(self):
        if YOLO is None:
            raise RuntimeError(f"ultralytics could not be imported: {YOLO_IMPORT_ERROR}")
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Please install the requirements first.")

        if self.model is None:
            self._configure_mode()
            self._notify_status(f"Loading {self.model_name} on {self.device}...")
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self._notify_status(f"Model ready: {self.model_name}")

    def _configure_mode(self):
        if self.mode == "Fast":
            self.model_name = "yolo11n.pt"
            self.confidence = 0.35
            self.imgsz = 480
            self.max_det = 25
        elif self.mode == "Accurate":
            self.model_name = "yolo11m.pt"
            self.confidence = 0.24
            self.imgsz = 640
            self.max_det = 100
        else:
            self.model_name = "yolo11s.pt"
            self.confidence = 0.28
            self.imgsz = 576
            self.max_det = 50

    def start(self, source_type, source, record_enabled=False, record_path=None, mode="Balanced"):
        self.stop()
        self.mode = mode
        self.load_model()
        self.running = True
        self.source_type = source_type
        self.source = source
        self.recording = record_enabled
        self.record_path = record_path
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.recording = False
        self.frame_ready = False
        self.frame = None

    def _process_loop(self):
        self.cap = cv2.VideoCapture(self.source if self.source_type == "camera" else self.source)
        if not self.cap.isOpened():
            self._notify_status("Unable to open the selected source.")
            self.running = False
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._notify_status("Streaming started")
        self.fps = 20.0 if self.source_type == "camera" else max(self.cap.get(cv2.CAP_PROP_FPS), 10.0)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self._notify_status("Stream finished or camera disconnected.")
                break

            annotated = self._detect_divers(frame)
            self._handle_recording(annotated)
            self.frame = annotated
            self.frame_ready = True
            self._notify_frame()

        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.running = False
        self._notify_status("Streaming stopped")

    def _detect_divers(self, frame):
        annotated = frame.copy()
        try:
            results = self.model(
                frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                stream=False,
                max_det=self.max_det,
                half=(self.device == "cuda"),
                augment=False,
            )
            result = results[0]
        except Exception as exc:
            self._notify_status(f"Detection error: {exc}")
            return annotated

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            if cls_name.lower() not in {"person", "diver", "swimmer", "human"}:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"Diver {box.conf[0]:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return annotated

    def _handle_recording(self, frame):
        if not self.recording or not self.record_path:
            return
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.record_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
        if self.writer is not None:
            self.writer.write(frame)

    def _notify_frame(self):
        if self.callback is not None:
            self.callback()

    def _notify_status(self, message):
        if self.callback is not None:
            self.callback(message=message)


class UnderwaterDiverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Underwater Diver Detection")
        self.geometry("1150x780")
        self.configure(bg="#0f172a")

        self.processor = DetectionProcessor(callback=self.handle_callback)
        self.source_mode = tk.StringVar(value="Camera")
        self.camera_index = tk.StringVar(value="0")
        self.video_path = tk.StringVar(value="")
        self.record_path = tk.StringVar(value="")
        self.record_enabled = tk.BooleanVar(value=False)
        self.performance_mode = tk.StringVar(value="Balanced")
        self.status_text = tk.StringVar(value="Ready to start streaming")
        self._build_ui()
        self.after(50, self.refresh_frame)

    def _build_ui(self):
        main_frame = ttk.Frame(self, padding=16)
        main_frame.pack(fill="both", expand=True)

        title = ttk.Label(main_frame, text="Underwater Diver Detection", font=("Segoe UI", 22, "bold"))
        title.pack(anchor="w", pady=(0, 12))

        controls = ttk.LabelFrame(main_frame, text="Controls", padding=12)
        controls.pack(fill="x", pady=(0, 12))

        source_row = ttk.Frame(controls)
        source_row.pack(fill="x", pady=4)
        ttk.Label(source_row, text="Source:").pack(side="left")
        ttk.Combobox(source_row, textvariable=self.source_mode, values=["Camera", "Video"], width=12).pack(side="left", padx=(8, 12))

        ttk.Label(source_row, text="Camera index:").pack(side="left")
        ttk.Entry(source_row, textvariable=self.camera_index, width=8).pack(side="left", padx=(4, 12))

        self.video_entry = ttk.Entry(source_row, textvariable=self.video_path, width=40)
        self.video_entry.pack(side="left", padx=(4, 8))
        ttk.Button(source_row, text="Browse", command=self.choose_video_file).pack(side="left")

        tune_row = ttk.Frame(controls)
        tune_row.pack(fill="x", pady=6)
        ttk.Label(tune_row, text="Performance:").pack(side="left")
        ttk.Combobox(tune_row, textvariable=self.performance_mode, values=["Fast", "Balanced", "Accurate"], width=12).pack(side="left", padx=(8, 12))

        action_row = ttk.Frame(controls)
        action_row.pack(fill="x", pady=8)
        ttk.Button(action_row, text="Start Streaming", command=self.start_streaming).pack(side="left")
        ttk.Button(action_row, text="Stop Streaming", command=self.stop_streaming).pack(side="left", padx=(8, 0))

        record_row = ttk.Frame(controls)
        record_row.pack(fill="x", pady=4)
        ttk.Checkbutton(record_row, text="Record video", variable=self.record_enabled).pack(side="left")
        ttk.Entry(record_row, textvariable=self.record_path, width=40).pack(side="left", padx=(8, 8))
        ttk.Button(record_row, text="Choose Record File", command=self.choose_record_file).pack(side="left")

        status_row = ttk.Frame(main_frame)
        status_row.pack(fill="x", pady=(0, 8))
        ttk.Label(status_row, text="Status:").pack(side="left")
        ttk.Label(status_row, textvariable=self.status_text, foreground="#2563eb").pack(side="left", padx=(8, 0))

        self.video_label = ttk.Label(main_frame, anchor="center")
        self.video_label.pack(fill="both", expand=True)

    def choose_video_file(self):
        path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv")])
        if path:
            self.video_path.set(path)
            self.source_mode.set("Video")

    def choose_record_file(self):
        path = filedialog.asksaveasfilename(
            title="Save recording as",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")],
        )
        if path:
            self.record_path.set(path)

    def start_streaming(self):
        if self.processor.running:
            messagebox.showinfo("Info", "Streaming is already running.")
            return

        if self.source_mode.get() == "Video" and not self.video_path.get():
            messagebox.showwarning("Warning", "Please select a video file before starting.")
            return

        try:
            source = int(self.camera_index.get()) if self.source_mode.get() == "Camera" else self.video_path.get()
        except ValueError:
            messagebox.showwarning("Warning", "Camera index must be a number.")
            return

        record_path = self.record_path.get() if self.record_enabled.get() else None
        self.processor.start(
            source_type="camera" if self.source_mode.get() == "Camera" else "video",
            source=source,
            record_enabled=self.record_enabled.get(),
            record_path=record_path,
            mode=self.performance_mode.get(),
        )
        self.status_text.set("Streaming is running...")

    def stop_streaming(self):
        self.processor.stop()
        self.status_text.set("Streaming stopped")

    def refresh_frame(self):
        if self.processor.frame_ready and self.processor.frame is not None:
            frame = self.processor.frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = image.resize((900, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            self.processor.frame_ready = False
        self.after(50, self.refresh_frame)

    def handle_callback(self, message=None):
        if message is not None:
            self.status_text.set(message)


def main():
    try:
        app = UnderwaterDiverApp()
        app.mainloop()
    except Exception as exc:
        messagebox.showerror("Startup error", str(exc))


if __name__ == "__main__":
    main()
