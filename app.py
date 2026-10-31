"""Gradio demo (local or HuggingFace Spaces). Run: python app.py"""
import os
import tempfile

import cv2
import gradio as gr
from ultralytics import YOLO

from utils import annotate, dip_steps, draw_hud, process_video, to_h264

model = YOLO(os.getenv("POTHOLE_MODEL", "models/best.pt"))


def run_image(img, conf):
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    res = model.predict(bgr, conf=conf, verbose=False)[0]
    out, counts, _ = annotate(bgr, res)
    draw_hud(out, counts, sum(counts.values()))
    steps = []
    for title, im in dip_steps(bgr):
        code = cv2.COLOR_GRAY2RGB if im.ndim == 2 else cv2.COLOR_BGR2RGB
        steps.append((cv2.cvtColor(im, code), title))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), steps


def run_video(path, conf, progress=gr.Progress()):
    if path is None:
        return None, "Upload a video first."
    out_path = tempfile.mktemp(suffix=".mp4")
    _, n = process_video(model, path, out_path, conf, progress=progress)
    return to_h264(out_path), f"Unique potholes detected: {n}"


with gr.Blocks(title="Pothole Detection") as demo:
    gr.Markdown("# 🕳️ Pothole Detection — YOLOv8 + OpenCV\n"
                "Detection, severity (LOW/MEDIUM/HIGH), unique counting & DIP pipeline.")
    conf = gr.Slider(0.1, 0.9, 0.4, step=0.05, label="Confidence threshold")
    with gr.Tab("Image"):
        i_in = gr.Image(type="numpy", label="Road image")
        i_btn = gr.Button("Detect", variant="primary")
        i_out = gr.Image(label="Detection result")
        i_gal = gr.Gallery(label="DIP pipeline steps", columns=5)
        i_btn.click(run_image, [i_in, conf], [i_out, i_gal])
    with gr.Tab("Video"):
        v_in = gr.Video(label="Road video")
        v_btn = gr.Button("Detect", variant="primary")
        v_out = gr.Video(label="Result")
        v_txt = gr.Textbox(label="Summary")
        v_btn.click(run_video, [v_in, conf], [v_out, v_txt])

if __name__ == "__main__":
    demo.launch()
