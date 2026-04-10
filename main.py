"""
MagicCamera Pipeline — YOLOv8-seg + Depth Estimation → blocks_<stem>.json

Usage:
    python main.py <image_path> [--output path.json] [--conf 0.5] [--show]

Defaults: blocks_<picture_stem>.json and (with --show) output_<picture_stem>.png
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import DPTForDepthEstimation, DPTImageProcessor
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────

MAX_IMAGE_SIZE = 1024
DEFAULT_CONF = 0.5
DEPTH_MODEL_NAME = "Intel/dpt-large"


# ── Segmentation ─────────────────────────────────────────────────────────────

def segment(image_path: str, conf: float = DEFAULT_CONF):
    """Run YOLOv8-seg and return results."""
    model = YOLO("yolov8m-seg.pt")
    results = model(image_path, conf=conf, verbose=False)
    return results[0]


# ── Depth Estimation ─────────────────────────────────────────────────────────

def estimate_depth(image: Image.Image) -> np.ndarray:
    """Return a depth map (H, W) normalised to [0, 1]."""
    processor = DPTImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
    model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME)
    model.eval()

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth  # (1, H', W')

    # Interpolate to original image size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0),
        size=image.size[::-1],  # (H, W)
        mode="bicubic",
        align_corners=False,
    ).squeeze().numpy()

    # Normalise to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


# ── Geometry ─────────────────────────────────────────────────────────────────

def compute_geometry(boxes, masks, class_names, depth_map, img_h, img_w):
    """Derive position, scale, rotation, depth for each detected object."""
    blocks = []

    for i, (box, mask, label) in enumerate(zip(boxes, masks, class_names)):
        x1, y1, x2, y2 = box

        # Centre of bounding box, normalised to [-1, 1]
        cx = ((x1 + x2) / 2 / img_w) * 2 - 1
        cy = -(((y1 + y2) / 2 / img_h) * 2 - 1)  # flip Y for Unity

        # Sample depth inside mask
        mask_bool = mask.astype(bool)
        if mask_bool.any():
            obj_depth = float(np.median(depth_map[mask_bool]))
        else:
            obj_depth = 0.5

        # Scale from bounding box size relative to image
        sx = (x2 - x1) / img_w
        sy = (y2 - y1) / img_h
        sz = 0.1  # placeholder Z thickness

        area = int(mask_bool.sum())

        blocks.append({
            "id": i,
            "type": label,
            "position": [round(float(cx), 4), round(float(cy), 4), round(obj_depth, 4)],
            "rotation": [0, 0, 0],
            "scale": [round(float(sx), 4), round(float(sy), 4), round(float(sz), 4)],
            "depth": round(obj_depth, 4),
            "area": area,
        })

    return blocks


# ── Visualisation ────────────────────────────────────────────────────────────

def visualise(image_bgr, result, depth_map, png_path: str):
    """Show YOLO detections + depth map side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: YOLO detections
    annotated = result.plot()
    axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("YOLOv8-seg Detections")
    axes[0].axis("off")

    # Right: Depth map
    axes[1].imshow(depth_map, cmap="inferno")
    axes[1].set_title("Depth Map (DPT)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MagicCamera pipeline")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: blocks_<picture_stem>.json)",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="YOLO confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show visualisation")
    args = parser.parse_args()

    stem = Path(args.image).stem
    json_path = args.output if args.output is not None else f"blocks_{stem}.json"
    png_path = f"output_{stem}.png"

    print(f"[1/5] Loading image: {args.image}")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        print(f"Error: cannot read image '{args.image}'")
        sys.exit(1)
    img_h, img_w = image_bgr.shape[:2]
    print(f"      Original size: {img_w}×{img_h}")

    # Resize if too large
    if max(img_h, img_w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(img_h, img_w)
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale)
        img_h, img_w = image_bgr.shape[:2]
        print(f"      Resized to: {img_w}×{img_h}")
        cv2.imwrite("/tmp/_magic_cam_resized.png", image_bgr)
        yolo_input = "/tmp/_magic_cam_resized.png"
    else:
        yolo_input = args.image

    print(f"[2/5] Running YOLOv8-seg segmentation (conf={args.conf})...")
    result = segment(yolo_input, conf=args.conf)

    if result.boxes is None or len(result.boxes) == 0:
        print("      No objects detected. Exiting.")
        json.dump({"blocks": []}, open(json_path, "w"), indent=2)
        sys.exit(0)

    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = [result.names[cid] for cid in class_ids]
    print(f"      Detected {len(boxes)} objects: {class_names}")

    # Get masks resized to image dimensions
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        if masks.shape[1:] != (img_h, img_w):
            masks_resized = []
            for m in masks:
                masks_resized.append(cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST))
            masks = np.array(masks_resized)
        print(f"      Masks shape: {masks.shape}")
    else:
        print("      No masks returned — using bounding boxes as fallback")
        masks = np.zeros((len(boxes), img_h, img_w), dtype=np.uint8)
        for i, (x1, y1, x2, y2) in enumerate(boxes.astype(int)):
            masks[i, y1:y2, x1:x2] = 1

    print(f"[3/5] Estimating depth with {DEPTH_MODEL_NAME}...")
    pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    depth_map = estimate_depth(pil_image)
    print(f"      Depth map range: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    print(f"[4/5] Computing geometry for {len(boxes)} objects...")
    blocks = compute_geometry(boxes, masks, class_names, depth_map, img_h, img_w)
    for b in blocks:
        print(f"      [{b['id']}] {b['type']:20s}  pos={b['position']}  depth={b['depth']}")

    print(f"[5/5] Writing {json_path}...")
    with open(json_path, "w") as f:
        json.dump({"blocks": blocks}, f, indent=2)
    print(f"      Done — {len(blocks)} blocks saved.")

    if args.show:
        print("      Opening visualisation...")
        visualise(image_bgr, result, depth_map, png_path)
    else:
        print("      (use --show to see visualisation)")


if __name__ == "__main__":
    main()
