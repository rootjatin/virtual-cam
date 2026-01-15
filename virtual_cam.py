#!/usr/bin/env python3
import argparse
import os
import signal
import time

import cv2
import numpy as np
import pyfakewebcam
from PIL import Image, ImageSequence

running = True

def stop(sig, frame):
    global running
    running = False

def parse_hex_color(s: str):
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError("Background color must be like #RRGGBB")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))

def fit_rgb(img_rgb: Image.Image, out_w: int, out_h: int, keep_aspect: bool, bg_rgb):
    if keep_aspect:
        src_w, src_h = img_rgb.size
        scale = min(out_w / src_w, out_h / src_h)
        new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
        resized = img_rgb.resize((new_w, new_h), Image.BICUBIC)
        canvas = Image.new("RGB", (out_w, out_h), bg_rgb)
        canvas.paste(resized, ((out_w - new_w)//2, (out_h - new_h)//2))
        img_rgb = canvas
    else:
        img_rgb = img_rgb.resize((out_w, out_h), Image.BICUBIC)
    return np.array(img_rgb, dtype=np.uint8)  # RGB uint8

def stream_gif(path, cam, out_w, out_h, fps, keep_aspect, bg_rgb, preview):
    gif = Image.open(path)
    frame_interval = 1.0 / max(fps, 1e-6)

    while running:
        for frame in ImageSequence.Iterator(gif):
            if not running:
                return
            t0 = time.perf_counter()

            rgba = frame.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, bg_rgb + (255,))
            comp = Image.alpha_composite(bg, rgba).convert("RGB")

            rgb = fit_rgb(comp, out_w, out_h, keep_aspect, bg_rgb)
            cam.schedule_frame(rgb)

            if preview:
                cv2.imshow("Virtual Cam Preview", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    return

            dt = time.perf_counter() - t0
            if frame_interval - dt > 0:
                time.sleep(frame_interval - dt)

        try:
            gif.seek(0)
        except Exception:
            pass

def stream_video(path, cam, out_w, out_h, fps, keep_aspect, bg_rgb, preview):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    use_fps = fps if fps > 0 else (src_fps if src_fps and src_fps > 0 else 30.0)
    frame_interval = 1.0 / max(use_fps, 1e-6)

    while running:
        ok, bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            continue

        t0 = time.perf_counter()
        rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_np)

        rgb = fit_rgb(img, out_w, out_h, keep_aspect, bg_rgb)
        cam.schedule_frame(rgb)

        if preview:
            cv2.imshow("Virtual Cam Preview", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        dt = time.perf_counter() - t0
        if frame_interval - dt > 0:
            time.sleep(frame_interval - dt)

    cap.release()

def main():
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="GIF or video file (mp4/mkv/etc)")
    ap.add_argument("--device", default="/dev/video2")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0, help="For video: set 0 to use source fps")
    ap.add_argument("--keep-aspect", action="store_true")
    ap.add_argument("--bg", default="#000000")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        raise SystemExit(f"File not found: {args.source}")
    if not os.path.exists(args.device):
        raise SystemExit(f"Device not found: {args.device} (load v4l2loopback)")

    bg_rgb = parse_hex_color(args.bg)
    cam = pyfakewebcam.FakeWebcam(args.device, args.width, args.height)

    ext = os.path.splitext(args.source.lower())[1]
    print(f"Feeding {args.source} -> {args.device} @ {args.width}x{args.height}")
    print("Ctrl+C to stop. If preview is on, press 'q' to close preview.")

    if ext == ".gif":
        stream_gif(args.source, cam, args.width, args.height, args.fps, args.keep_aspect, bg_rgb, args.preview)
    else:
        stream_video(args.source, cam, args.width, args.height, args.fps, args.keep_aspect, bg_rgb, args.preview)

    if args.preview:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

