"""Run on a video file or webcam.
python detect.py --source road.mp4
python detect.py --source 0 --show
"""
import argparse

from ultralytics import YOLO

from utils import process_video

p = argparse.ArgumentParser()
p.add_argument("--source", required=True, help="video path or 0 for webcam")
p.add_argument("--model", default="models/best.pt")
p.add_argument("--conf", type=float, default=0.4)
p.add_argument("--out", default="output.mp4")
p.add_argument("--show", action="store_true", help="live window (not on Colab)")
a = p.parse_args()

src = int(a.source) if a.source.isdigit() else a.source
out, n = process_video(YOLO(a.model), src, a.out, a.conf, a.show)
print(f"Done. Saved {out} | unique potholes: {n}")
