"""Shared helpers: severity, drawing, DIP pipeline, video processing."""
import os
import shutil
import subprocess
import time

import cv2
import numpy as np


# Filters against false positives (sky, trees, huge boxes)
MAX_AREA = 0.35   # box larger than this share of the frame is not a pothole
ROI_TOP = 0.40    # ignore detections whose centre is above this fraction of frame height


def severity(box, shape):
    """Approximate severity from box area / frame area (camera-distance dependent)."""
    x1, y1, x2, y2 = box
    h, w = shape[:2]
    ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
    if ratio < 0.02:
        return "LOW", (0, 200, 0)
    if ratio < 0.08:
        return "MEDIUM", (0, 165, 255)
    return "HIGH", (0, 0, 255)


def annotate(frame, result):
    """Draw boxes + severity labels. Returns (image, severity_counts, track_ids)."""
    out = frame.copy()
    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    ids = []
    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        conf = float(b.conf[0])
        fh, fw = frame.shape[:2]
        area = (x2 - x1) * (y2 - y1) / float(fw * fh)
        cy = (y1 + y2) / 2.0 / fh
        if area > MAX_AREA or cy < ROI_TOP:
            continue
        label, color = severity((x1, y1, x2, y2), frame.shape)
        counts[label] += 1
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(out, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if b.id is not None:
            ids.append(int(b.id[0]))
    return out, counts, ids


def draw_hud(frame, counts, total, fps=None):
    """Top-left info panel (modifies frame in place)."""
    lines = [f"Potholes (total): {total}",
             f"LOW: {counts['LOW']}  MED: {counts['MEDIUM']}  HIGH: {counts['HIGH']}"]
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    cv2.rectangle(frame, (8, 8), (300, 12 + 24 * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (14, 30 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)


def dip_steps(img_bgr):
    """Classical Digital Image Processing pipeline, step by step."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg = img_bgr.copy()
    big = [c for c in contours if cv2.contourArea(c) > 500]
    cv2.drawContours(seg, big, -1, (0, 255, 255), 2)
    return [("1. Grayscale", gray), ("2. Gaussian Blur", blur),
            ("3. Canny Edges", edges), ("4. Morphology (Close)", morph),
            ("5. Contour Segmentation", seg)]


def process_video(model, source, out_path, conf=0.4, show=False, progress=None):
    """Run detection + tracking on a video/webcam, save annotated video.
    Returns (out_path, unique_pothole_count)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open source: {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    unique, n, t0 = set(), 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # persist=False on the first frame resets the tracker between videos
        res = model.track(frame, persist=(n > 0), conf=conf, verbose=False)[0]
        out, counts, ids = annotate(frame, res)
        unique.update(ids)
        n += 1
        draw_hud(out, counts, len(unique), n / (time.time() - t0))
        writer.write(out)
        if show:
            cv2.imshow("Pothole Detection (q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if progress and total:
            progress(min(n / total, 1.0))
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    return out_path, len(unique)


def to_h264(src):
    """Re-encode to H.264 so browsers can play it (needs ffmpeg; else returns src)."""
    if not shutil.which("ffmpeg"):
        return src
    dst = src.replace(".mp4", "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p", dst],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst if os.path.exists(dst) else src
