import argparse
import importlib
import subprocess
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
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
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *missing])


def build_data_yaml(dataset_root: Path, classes: list[str]) -> Path:
    dataset_root.mkdir(parents=True, exist_ok=True)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    train_images = images_dir / "train"
    train_labels = labels_dir / "train"
    val_images = images_dir / "val"
    val_labels = labels_dir / "val"

    for folder in [train_images, train_labels, val_images, val_labels]:
        folder.mkdir(parents=True, exist_ok=True)

    if not any(train_images.iterdir()) or not any(val_images.iterdir()):
        print("No training or validation images were found. Please add labeled image files to the dataset folders.")
        print(f"Expected directories: {train_images}, {val_images}")

    data_yaml_path = dataset_root / "data.yaml"
    content = (
        "train: images/train\n"
        "val: images/val\n"
        "test: \n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    data_yaml_path.write_text(content, encoding="utf-8")
    return data_yaml_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom YOLO diver detector from a YOLO-style dataset")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the dataset root folder")
    parser.add_argument("--classes", type=str, nargs="+", default=["diver"], help="Class names in the dataset")
    parser.add_argument("--model", type=str, default="models/yolo11s.pt", help="YOLO model checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Number of training workers")
    parser.add_argument("--project", type=str, default="runs/train", help="Output folder for training results")
    parser.add_argument("--name", type=str, default="diver_detector", help="Name of the training run")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dependencies()

    from ultralytics import YOLO
    import torch

    dataset_root = Path(args.dataset_root).resolve()
    data_yaml = build_data_yaml(dataset_root, args.classes)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")
    print(f"Using dataset config: {data_yaml}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        device=device,
        pretrained=True,
        exist_ok=True,
    )

    print("Training finished.")
    print(f"Best weights are in: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
