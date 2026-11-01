"""Train YOLOv8 on a pothole dataset (YOLOv8 format, has data.yaml).
Colab: Runtime > T4 GPU, then:  !python train.py --data /content/dataset/data.yaml
"""
import argparse
import os
import shutil

from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="path to data.yaml")
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--base", default="yolov8n.pt", help="yolov8n.pt (fast) / yolov8s.pt (better)")
a = p.parse_args()

model = YOLO(a.base)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            project="runs", name="pothole", exist_ok=True)
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")

best = os.path.join("runs", "pothole", "weights", "best.pt")
os.makedirs("models", exist_ok=True)
shutil.copy(best, "models/best.pt")
print("Saved -> models/best.pt")
