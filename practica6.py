from pathlib import Path
import json
import math

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent


def read_image(name):
    data = np.fromfile(ROOT / name, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(name)
    return img


def write_jpg(path, img, quality=95):
    ok, data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"No se pudo codificar {path.name}")
    data.tofile(path)


def resize_to_height(img, height):
    scale = height / img.shape[0]
    width = int(round(img.shape[1] * scale))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def resize_cover(img, size):
    target_w, target_h = size
    scale = max(target_w / img.shape[1], target_h / img.shape[0])
    resized = cv2.resize(
        img,
        (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    y = (resized.shape[0] - target_h) // 2
    x = (resized.shape[1] - target_w) // 2
    return resized[y : y + target_h, x : x + target_w].copy()


def soft_white_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray < 245).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def keep_largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 20).astype(np.uint8), 8)
    if count <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean = np.where(labels == largest, 255, 0).astype(np.uint8)
    clean = cv2.GaussianBlur(clean, (7, 7), 0)
    return clean


def grabcut_mask(img, rect_margin=8):
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (
        rect_margin,
        rect_margin,
        img.shape[1] - rect_margin * 2,
        img.shape[0] - rect_margin * 2,
    )
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def overlay_alpha(base, fg, mask, x, y):
    h, w = fg.shape[:2]
    roi = base[y : y + h, x : x + w].astype(np.float32)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    blended = fg.astype(np.float32) * alpha + roi * (1.0 - alpha)
    base[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)


def add_waldo_marks(img):
    """Turn one real person in the crowd into a small Waldo-like target."""
    # Person at the right side of the selected video frame.
    x0, y0, w, h = 1682, 430, 125, 300

    # Shirt: red/white horizontal stripes clipped to a torso-like quadrilateral.
    torso = np.array(
        [
            [x0 + 15, y0 + 140],
            [x0 + 112, y0 + 130],
            [x0 + 122, y0 + 292],
            [x0 + 5, y0 + 292],
        ],
        dtype=np.int32,
    )
    torso_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(torso_mask, torso, 255)
    stripe_layer = img.copy()
    for yy in range(y0 + 128, y0 + 300, 20):
        color = (245, 245, 245) if ((yy - y0) // 20) % 2 == 0 else (35, 35, 210)
        cv2.rectangle(stripe_layer, (x0, yy), (x0 + w + 15, yy + 20), color, -1)
    a = (torso_mask.astype(np.float32) / 255.0)[:, :, None] * 0.88
    img[:] = np.clip(stripe_layer.astype(np.float32) * a + img.astype(np.float32) * (1 - a), 0, 255)

    # Hat and glasses: enough detail to be recognizable without making it cartoon-only.
    cv2.ellipse(img, (x0 + 67, y0 + 69), (55, 13), -8, 0, 360, (245, 245, 245), -1)
    cv2.ellipse(img, (x0 + 67, y0 + 55), (35, 31), -6, 180, 360, (245, 245, 245), -1)
    cv2.rectangle(img, (x0 + 32, y0 + 55), (x0 + 103, y0 + 67), (35, 35, 210), -1)
    cv2.circle(img, (x0 + 52, y0 + 103), 11, (30, 30, 30), 2)
    cv2.circle(img, (x0 + 82, y0 + 101), 11, (30, 30, 30), 2)
    cv2.line(img, (x0 + 63, y0 + 102), (x0 + 72, y0 + 102), (30, 30, 30), 2)

    return (x0, y0, w, h)


def add_subtle_shadow(base, x, y, w, h):
    shadow = np.zeros(base.shape[:2], dtype=np.uint8)
    center = (x + w // 2, y + h - 20)
    cv2.ellipse(shadow, center, (w // 2, max(12, h // 10)), 0, 0, 360, 120, -1)
    shadow = cv2.GaussianBlur(shadow, (41, 41), 0)
    a = (shadow.astype(np.float32) / 255.0)[:, :, None] * 0.35
    base[:] = np.clip(base.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)


def main():
    out_dir = ROOT / "practica6_generadas"
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(ROOT / "personas.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
    ok, crowd = cap.read()
    if not ok:
        raise RuntimeError("No se pudo leer personas.mp4")

    width, height = 1920, 1200
    big = np.zeros((height, width, 3), dtype=np.uint8)

    crowd_panel = resize_cover(crowd, (width, 760))
    big[:760, :] = crowd_panel

    table = read_image("tornillos.jpg")
    table_panel = resize_cover(table, (width, 440))
    table_panel = cv2.GaussianBlur(table_panel, (3, 3), 0)
    table_panel = cv2.convertScaleAbs(table_panel, alpha=0.92, beta=-8)
    big[760:, :] = table_panel

    # Make the transition look like a simple photo collage/worksheet, not a blank split.
    cv2.rectangle(big, (0, 746), (width, 775), (35, 35, 35), -1)

    waldo_box = add_waldo_marks(big)

    gear_src = read_image("engranaje2.jpg")
    gear = gear_src[0:500, 355:875].copy()
    gear = resize_to_height(gear, 385)
    gear_mask = keep_largest_component(soft_white_mask(gear))
    gear_x, gear_y = 1160, 795
    add_subtle_shadow(big, gear_x, gear_y, gear.shape[1], gear.shape[0])
    overlay_alpha(big, gear, gear_mask, gear_x, gear_y)

    # A realistic screw crop from the supplied photo.
    screws = read_image("tornillos.jpg")
    screw_crop = screws[260:650, 185:625].copy()
    screw_crop = cv2.resize(screw_crop, (405, 360), interpolation=cv2.INTER_AREA)
    screw_mask = grabcut_mask(screw_crop, 18)
    screw_x, screw_y = 205, 810
    add_subtle_shadow(big, screw_x, screw_y, screw_crop.shape[1], screw_crop.shape[0])
    overlay_alpha(big, screw_crop, screw_mask, screw_x, screw_y)

    # Exact templates are cropped from the final composed image.
    templates = {
        "plantilla_tornillo.jpg": (screw_x + 42, screw_y + 28, 300, 285),
        "plantilla_agujero_engranaje.jpg": (gear_x + 222, gear_y + 133, 100, 100),
        "plantilla_waldo.jpg": (waldo_box[0] + 12, waldo_box[1] + 48, 105, 238),
    }

    write_jpg(out_dir / "imagen_grande.jpg", big, 94)
    metadata = {"imagen_grande.jpg": {"width": width, "height": height}, "plantillas": {}}
    for filename, (x, y, w, h) in templates.items():
        crop = big[y : y + h, x : x + w].copy()
        write_jpg(out_dir / filename, crop, 96)
        metadata["plantillas"][filename] = {"x": x, "y": y, "width": w, "height": h}

    with open(out_dir / "ubicaciones.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Quick verification for the practice: each small template should be found.
    gray_big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    print("Archivos generados en:", out_dir)
    for filename in templates:
        tpl_color = read_image(str(Path("practica6_generadas") / filename))
        tpl = cv2.cvtColor(tpl_color, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_big, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print(f"{filename}: confianza={max_val:.4f}, ubicacion={max_loc}")


if __name__ == "__main__":
    main()
