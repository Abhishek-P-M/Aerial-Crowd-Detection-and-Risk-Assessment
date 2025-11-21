# src/image_analytics.py
import argparse, os, glob, math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# VisDrone class names (order matters)
VISDRONE_NAMES = [
    "pedestrian","people","bicycle","car","van","truck",
    "tricycle","awning-tricycle","bus","motor","others"
]

# Vehicles set (for proximity risk to pedestrians)
VEHICLE_SET = {
    "car","van","truck","bus","motor","tricycle","awning-tricycle","bicycle"
}

# Weights for Congestion Index (tuneable)
CI_WEIGHTS = {
    "pedestrian":0.8, "people":0.8, "bicycle":0.5, "car":1.0, "van":1.2, "truck":1.5,
    "tricycle":0.7, "awning-tricycle":0.9, "bus":2.0, "motor":0.6, "others":0.5
}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _safe_textbbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """
    Pillow 10+ removed textsize(); prefer textbbox(), with fallbacks.
    Returns (left, top, right, bottom)
    """
    try:
        return draw.textbbox((0, 0), text, font=font)
    except Exception:
        try:
            fb = font.getbbox(text)  # (x0,y0,x1,y1)
            return (0, 0, fb[2]-fb[0], fb[3]-fb[1])
        except Exception:
            w = int(draw.textlength(text, font=font)) if hasattr(draw, "textlength") else len(text)*8
            h = 16
            return (0, 0, w, h)

def draw_overlay(img_pil, boxes, classes, scores, class_id_to_name):
    im = img_pil.copy()
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    H, W = im.size[1], im.size[0]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cls = int(classes[i])
        name = class_id_to_name.get(cls, str(cls))
        lab  = f"{name} {scores[i]:.2f}"

        # box
        dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        # label size
        l, t, r, b = _safe_textbbox(dr, lab, font)
        tw, th = (r - l), (b - t)

        # background behind label (clamped to image top)
        by1 = max(0, int(y1) - th - 6)
        by2 = int(y1)
        bx1 = max(0, int(x1))
        bx2 = min(W, int(x1) + tw + 6)
        dr.rectangle([bx1, by1, bx2, by2], fill=(255, 0, 0))
        dr.text((bx1 + 3, by1 + 3), lab, fill=(255, 255, 255), font=font)

    return im

def heatmap_from_points(H, W, points, sigma=15):
    """
    Simple density heatmap: impulse map + Gaussian blur.
    points: list of (x,y) pixel coords
    """
    if len(points) == 0:
        return np.zeros((H, W), dtype=np.float32)
    acc = np.zeros((H, W), dtype=np.float32)
    for (x, y) in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < H and 0 <= xi < W:
            acc[yi, xi] += 1.0
    # Blur
    k = max(3, int(sigma*3)//2*2+1)
    hm = cv2.GaussianBlur(acc, (k, k), sigmaX=sigma, sigmaY=sigma)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
    return hm

def colorize_heatmap(hm):
    """0..1 -> colored RGBA overlay using OpenCV colormap."""
    hm_u8 = (hm * 255).astype(np.uint8)
    color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)  # BGR
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGBA)
    color[..., 3] = (hm * 200).astype(np.uint8)  # alpha
    return color  # HxWx4

def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def run(model_path, source, out_dir, conf=0.25, imgsz=960, do_heatmap=True):
    ensure_dir(out_dir)
    over_dir = os.path.join(out_dir, "overlays"); ensure_dir(over_dir)
    hm_dir   = os.path.join(out_dir, "heatmaps"); ensure_dir(hm_dir)

    m = YOLO(model_path)
    # names can be dict (id->name) or list
    if isinstance(m.names, dict):
        names = {int(i): n for i, n in m.names.items()}
    else:
        names = {i: n for i, n in enumerate(VISDRONE_NAMES)}

    # collect images
    if os.path.isdir(source):
        images = [p for p in glob.glob(os.path.join(source, "*"))
                  if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    else:
        images = [source]

    rows = []  # for CSV
    for ip in images:
        # --- predict ---
        res = m.predict(source=ip, conf=conf, imgsz=imgsz, save=False, verbose=False)[0]
        H, W = res.orig_shape
        if res.boxes is not None and len(res.boxes) > 0:
            boxes  = res.boxes.xyxy.cpu().numpy()
            clses  = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy()
        else:
            boxes  = np.zeros((0, 4), np.float32)
            clses  = np.zeros((0,), int)
            scores = np.zeros((0,), np.float32)

        # --- overlay image ---
        img = Image.open(ip).convert("RGB")
        overlay = draw_overlay(img, boxes, clses, scores, names)
        base = os.path.basename(ip)
        overlay.save(os.path.join(over_dir, base))

        # --- per-class counts ---
        cnt_ids = Counter(clses.tolist())
        counts = {names.get(k, str(k)): int(v) for k, v in cnt_ids.items()}

        # --- density heatmap (all detections) ---
        centers = [center_of(b) for b in boxes]
        if do_heatmap:
            hm = heatmap_from_points(H, W, centers, sigma=max(8, int(0.015 * max(H, W))))
            rgba = colorize_heatmap(hm)
            bg = np.array(img.convert("RGBA"))
            blend = bg.copy()
            alpha = rgba[..., 3:4].astype(np.float32) / 255.0
            blend[..., :3] = (alpha * rgba[..., :3] + (1 - alpha) * blend[..., :3]).astype(np.uint8)

            # ✅ Save as PNG to support RGBA (no JPEG alpha error)
            hm_out = Path(hm_dir) / (Path(base).stem + "_heatmap.png")
            Image.fromarray(blend, mode="RGBA").save(hm_out)

        # --- metrics ---
        # Congestion Index: sum(weights per detection)
        ci = 0.0
        for cls in clses.tolist():
            nm = names.get(int(cls), "others")
            ci += CI_WEIGHTS.get(nm, 1.0)

        # Proximity Risk Index: vehicles close to pedestrians in a single frame
        veh_centers = [center_of(b) for b, c in zip(boxes, clses) if names.get(int(c), "") in VEHICLE_SET]
        ped_centers = [center_of(b) for b, c in zip(boxes, clses) if names.get(int(c), "") in {"pedestrian", "people"}]
        diag = math.hypot(W, H)
        thr = 0.08 * diag  # ~8% of diagonal; tune for your data
        pri = 0.0
        min_dists = []
        if veh_centers and ped_centers:
            vc = np.array(veh_centers); pc = np.array(ped_centers)
            for p in pc:
                d = np.sqrt(((vc - p) ** 2).sum(axis=1)).min()
                min_dists.append(d)
                pri += max(0.0, (thr - d) / thr)  # closer => higher risk

        # Occupancy (sum of bbox area / image area)
        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clip(min=0)
        occupancy = float(areas.sum() / (W * H + 1e-6))

        row = {
            "image": base,
            "congestion_index": round(ci, 3),
            "proximity_risk_index": round(pri, 3),
            "occupancy_frac": round(occupancy, 4),
            "total_detections": int(len(boxes)),
        }
        # attach per-class counts (columns like count_car, count_pedestrian, ...)
        for n, v in counts.items():
            row[f"count_{n}"] = v

        # extras: average min distance ped→vehicle (pixels)
        row["avg_min_ped_vehicle_px"] = round(float(np.mean(min_dists)) if min_dists else 0.0, 2)
        rows.append(row)

    # write CSV sorted by risk/CI
    df = pd.DataFrame(rows).fillna(0)
    if not df.empty:
        df = df.sort_values(["proximity_risk_index", "congestion_index"], ascending=False)
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote metrics: {out_csv}")
    print(f"📂 Overlays: {over_dir}")
    if do_heatmap:
        print(f"🔥 Heatmaps: {hm_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/train/weights/best.pt",
                    help="Path to .pt weights (use yolov8n.pt to sanity-check pipeline)")
    ap.add_argument("--source", required=True, help="Folder of images OR a single image path")
    ap.add_argument("--out", default="outputs/analytics")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--no-heatmap", action="store_true")
    a = ap.parse_args()
    run(a.model, a.source, a.out, a.conf, a.imgsz, do_heatmap=not a.no_heatmap)
