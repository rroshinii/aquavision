"""Underwater image enhancement: colour-cast correction + contrast + light denoise + sharpen."""
import cv2
import numpy as np


def enhance_underwater(bgr):
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)               # B, G, R
    gains = np.clip(means.mean() / np.maximum(means, 1e-3), 0.7, 2.5)  # restores red lost to absorption
    img = np.clip(img * gains, 0, 255).astype(np.uint8)

    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    out = cv2.bilateralFilter(out, 5, 35, 35)             # reduce noise, keep edges
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    return cv2.addWeighted(out, 1.3, blur, -0.3, 0)       # mild unsharp mask
