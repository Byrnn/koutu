"""Run the matting harness for semantic UI components in one pass.

The component boxes are deliberately explicit for this first test.  This keeps
the segmentation result auditable: the detector can be replaced later without
changing Trimap, alpha matting, foreground reconstruction, or the manifest
format.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from process import (
    build_trimap,
    composite,
    crop_to_alpha,
    estimate_alpha_cf,
    estimate_foreground_ml,
    grabcut_mask,
    load_rgb,
    parse_bbox,
    save_gray,
    save_rgb,
    save_rgba,
)


def clip_bbox(bbox, width, height):
    x, y, bw, bh = bbox
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    bw = max(1, min(int(bw), width - x))
    bh = max(1, min(int(bh), height - y))
    return x, y, bw, bh


def decodable_height(image):
    """Find the useful prefix of Pillow's fill for a truncated JPEG."""
    a = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if a.shape[0] < 2:
        return a.shape[0]
    # A truncated JPEG decoded by Pillow is filled with one constant row.
    for y in range(1, a.shape[0]):
        if bool(np.all(a[y:] == a[y])):
            return y
    return a.shape[0]


def write_status(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_component(image, component, out_root, unknown_width, truncated, valid_height):
    height, width = image.shape[:2]
    component_id = str(component["id"])
    kind = str(component.get("type", "unknown"))
    bbox = clip_bbox(parse_bbox(component["bbox"]), width, height)
    x, y, bw, bh = bbox
    out = out_root / component_id
    out.mkdir(parents=True, exist_ok=True)

    missing_rows = truncated and (y >= valid_height or y + bh > valid_height)
    roi = image[y : y + bh, x : x + bw]
    roi_flat = float(np.std(roi)) < 1e-6
    if missing_rows or roi_flat:
        status = {
            "id": component_id,
            "type": kind,
            "bbox": [x, y, bw, bh],
            "status": "blocked_source_missing" if missing_rows else "blocked_flat_source",
            "message": (
                "The reference JPEG is truncated before this component; no chest pixels "
                "are available to segment. Replace reference_ui.jpg and rerun."
                if missing_rows
                else "The selected source region contains no usable image variation."
            ),
        }
        write_status(out / "status.json", status)
        return status

    # Work on the component ROI so the matting solve is fast and the output is
    # naturally cropped.  Leave a one-pixel margin for GrabCut's rectangle init.
    local_rect = (1, 1, max(1, bw - 2), max(1, bh - 2))
    mask = grabcut_mask(roi, local_rect)
    # Small, high-contrast UI buttons can be rejected when GrabCut sees only
    # the tight ROI.  Re-run once against the full reference for context.
    if int(mask.sum()) < max(4, int(mask.size * 0.005)):
        full_mask = grabcut_mask(image, bbox)
        mask = full_mask[y : y + bh, x : x + bw]
    # A valid matting solve needs both definite foreground and background.
    # Keep the failure visible if neither segmentation pass can find one.
    if int(mask.sum()) == 0 or int(mask.sum()) == mask.size:
        status = {
            "id": component_id,
            "type": kind,
            "bbox": [x, y, bw, bh],
            "status": "segmentation_failed",
            "message": "GrabCut did not produce both foreground and background pixels.",
        }
        write_status(out / "status.json", status)
        return status
    trimap = None
    # Keep a real 3–15px unknown band when the component is large, but do not
    # erode a small button completely.  The first usable trimap is recorded in
    # the manifest so the choice remains inspectable.
    tried_widths = []
    for candidate in [unknown_width, min(3, unknown_width), 2, 1]:
        candidate = max(1, int(candidate))
        if candidate in tried_widths:
            continue
        tried_widths.append(candidate)
        candidate_trimap, _, _ = build_trimap(mask, candidate)
        if np.any(candidate_trimap >= 0.9) and np.any(candidate_trimap <= 0.1):
            trimap = candidate_trimap
            break
    if trimap is None:
        status = {
            "id": component_id,
            "type": kind,
            "bbox": [x, y, bw, bh],
            "status": "trimap_failed",
            "message": "Mask did not leave both definite foreground and background pixels.",
        }
        write_status(out / "status.json", status)
        return status
    alpha = np.clip(estimate_alpha_cf(roi, trimap), 0.0, 1.0)
    foreground = np.clip(estimate_foreground_ml(roi, alpha), 0.0, 1.0)

    save_gray(out / "target_mask.png", mask.astype(np.float64))
    save_gray(out / "trimap.png", trimap)
    save_gray(out / "alpha.png", alpha)
    save_rgb(out / "foreground.png", foreground)
    save_rgba(out / "final_rgba_full.png", foreground, alpha)

    _, alpha_c, foreground_c = crop_to_alpha(roi, alpha, foreground)
    save_rgba(out / "final_rgba.png", foreground_c, alpha_c)
    for name, color in {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
    }.items():
        save_rgb(out / f"test_{name}.png", composite(foreground_c, alpha_c, color))

    # Keep an original-coordinate canvas for direct FGUI placement.
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgb8 = np.clip(foreground * 255.0, 0, 255).astype(np.uint8)
    a8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    rgba[y : y + bh, x : x + bw, :3] = rgb8
    rgba[y : y + bh, x : x + bw, 3] = a8
    Image.fromarray(rgba, mode="RGBA").save(out / "final_rgba_canvas.png")

    status = {
        "id": component_id,
        "type": kind,
        "bbox": [x, y, bw, bh],
        "status": "ok",
        "alpha_coverage": round(float((alpha > 0.01).mean()), 6),
        "unknown_coverage": round(float(((trimap > 0.0) & (trimap < 1.0)).mean()), 6),
        "outputs": sorted(p.name for p in out.glob("*.png")),
    }
    write_status(out / "status.json", status)
    return status


def main():
    ap = argparse.ArgumentParser(description="Batch semantic UI alpha-matting test")
    ap.add_argument("--config", default="components.json")
    ap.add_argument("--output", default="components_output")
    args = ap.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = Path(config.get("input", "reference_ui.jpg"))
    image, truncated = load_rgb(input_path)
    valid_height = decodable_height(image) if truncated else image.shape[0]
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for component in config["components"]:
        results.append(
            process_component(
                image,
                component,
                out_root,
                int(config.get("unknown_width", 5)),
                truncated,
                valid_height,
            )
        )

    report = {
        "input": str(input_path),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "jpeg_truncated": truncated,
        "decodable_height": int(valid_height),
        "components": results,
    }
    write_status(out_root / "report.json", report)
    layers = []
    for result in results:
        if "bbox" not in result:
            continue
        x, y, width, height = result["bbox"]
        layers.append(
            {
                "id": result["id"],
                "type": result["type"],
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "z_order": len(layers),
                "status": result["status"],
                "asset": f"{result['id']}/final_rgba.png",
                "canvas_asset": f"{result['id']}/final_rgba_canvas.png",
            }
        )
    write_status(out_root / "layers.json", layers)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
