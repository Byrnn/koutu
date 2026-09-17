import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile
from pymatting import estimate_alpha_cf, estimate_foreground_ml

# GitHub can contain partially uploaded JPEGs while a reference asset is being
# replaced.  Pillow normally aborts on those files; allowing the decoder to
# return the decodable prefix lets the pipeline produce a truthful diagnostic
# instead of silently fabricating missing UI pixels.
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_rgb(path):
    """Load an RGB image and report whether its byte stream is truncated."""
    raw = Path(path).read_bytes()
    truncated = False
    if raw[:2] == b'\xff\xd8' and b'\xff\xd9' not in raw:
        truncated = True
    image = Image.open(path).convert('RGB')
    image.load()
    return np.asarray(image, dtype=np.float64) / 255.0, truncated


def parse_bbox(text: str):
    vals = [int(v.strip()) for v in text.split(',')]
    if len(vals) != 4:
        raise ValueError('bbox must be x,y,w,h')
    return tuple(vals)


def save_gray(path: Path, arr: np.ndarray):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(path)


def save_rgb(path: Path, arr: np.ndarray):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='RGB').save(path)


def save_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray):
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    a8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb8, a8])
    Image.fromarray(rgba, mode='RGBA').save(path)


def grabcut_mask(image_rgb: np.ndarray, bbox):
    h, w = image_rgb.shape[:2]
    x, y, bw, bh = bbox
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))

    image_bgr = cv2.cvtColor((image_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, mask, (x, y, bw, bh), bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)


def build_trimap(mask: np.ndarray, unknown_width: int):
    k = max(1, int(unknown_width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    fg = cv2.erode(mask, kernel, iterations=1)
    bg = 1 - cv2.dilate(mask, kernel, iterations=1)

    trimap = np.full(mask.shape, 0.5, np.float64)
    trimap[bg == 1] = 0.0
    trimap[fg == 1] = 1.0
    return trimap, fg.astype(np.float64), bg.astype(np.float64)


def crop_to_alpha(rgb, alpha, foreground, pad=8):
    ys, xs = np.where(alpha > 0.005)
    if len(xs) == 0:
        return rgb, alpha, foreground
    x0, x1 = max(0, xs.min() - pad), min(alpha.shape[1], xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(alpha.shape[0], ys.max() + 1 + pad)
    return rgb[y0:y1, x0:x1], alpha[y0:y1, x0:x1], foreground[y0:y1, x0:x1]


def composite(fg: np.ndarray, alpha: np.ndarray, bg_rgb):
    bg = np.zeros_like(fg)
    bg[:] = np.array(bg_rgb, dtype=np.float64)
    return fg * alpha[..., None] + bg * (1.0 - alpha[..., None])


def main():
    ap = argparse.ArgumentParser(description='Game UI alpha-matting pipeline')
    ap.add_argument('--input', required=True)
    ap.add_argument('--bbox', required=True, help='x,y,w,h of target UI')
    ap.add_argument('--unknown-width', type=int, default=8, help='trimap unknown band width in px')
    ap.add_argument('--output', default='output')
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    image, truncated = load_rgb(args.input)
    if truncated:
        print('WARNING: input JPEG has no EOI marker; results only cover the decodable pixels.')
    bbox = parse_bbox(args.bbox)

    mask = grabcut_mask(image, bbox)
    trimap, _, _ = build_trimap(mask, args.unknown_width)

    alpha = estimate_alpha_cf(image, trimap)
    alpha = np.clip(alpha, 0.0, 1.0)
    foreground = estimate_foreground_ml(image, alpha)
    foreground = np.clip(foreground, 0.0, 1.0)

    save_gray(out / 'target_mask.png', mask.astype(np.float64))
    save_gray(out / 'trimap.png', trimap)
    save_gray(out / 'alpha.png', alpha)
    save_rgb(out / 'foreground.png', foreground)
    save_rgba(out / 'final_rgba_full.png', foreground, alpha)

    _, alpha_c, foreground_c = crop_to_alpha(image, alpha, foreground)
    save_rgba(out / 'final_rgba.png', foreground_c, alpha_c)

    tests = {
        'black': (0.0, 0.0, 0.0),
        'white': (1.0, 1.0, 1.0),
        'red': (1.0, 0.0, 0.0),
        'blue': (0.0, 0.0, 1.0),
    }
    for name, color in tests.items():
        save_rgb(out / f'test_{name}.png', composite(foreground_c, alpha_c, color))

    print(f'Done. Results written to: {out.resolve()}')


if __name__ == '__main__':
    main()
