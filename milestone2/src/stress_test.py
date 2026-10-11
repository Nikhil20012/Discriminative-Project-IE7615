import argparse
import json
import os
import random

import pandas as pd
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png"}

SCENARIOS = {
    "3x3_plain": dict(grid=(3, 3), tile=224),
    "2x2_plain": dict(grid=(2, 2), tile=224),
    "3x3_gap_white": dict(grid=(3, 3), tile=224, gap=12, bg=(255, 255, 255)),
    "3x3_gap_black": dict(grid=(3, 3), tile=224, gap=12, bg=(0, 0, 0)),
    "2x2_gap_white": dict(grid=(2, 2), tile=224, gap=12, bg=(255, 255, 255)),
    "3x3_small_tiles": dict(grid=(3, 3), tile=128),
    "3x3_large_tiles": dict(grid=(3, 3), tile=320),
    "2x2_large_tiles": dict(grid=(2, 2), tile=320),
    "3x3_on_640_canvas": dict(grid=(3, 3), tile=190, gap=8, canvas=(640, 640), bg="random"),
    "3x3_on_wide_canvas": dict(grid=(3, 3), tile=200, canvas=(1280, 720), bg=(128, 128, 128)),
    "3x3_partial": dict(grid=(3, 3), tile=224, fill=(5, 8), bg=(200, 200, 200)),
    "2x2_partial": dict(grid=(2, 2), tile=224, fill=(2, 3), bg=(200, 200, 200)),
    "3x3_jittered": dict(grid=(3, 3), tile=224, jitter=True, bg=(255, 255, 255)),
    "3x3_low_jpeg": dict(grid=(3, 3), tile=224, quality=35),
    "4x4_dense": dict(grid=(4, 4), tile=160),
    "1x4_row": dict(grid=(1, 4), tile=224),
}


def load_pools(src):
    pools = {}
    for cls in sorted(os.listdir(src)):
        d = os.path.join(src, cls)
        if os.path.isdir(d):
            pools[cls] = [os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in EXTS]
    return pools


def compose(cfg, pools, class_to_idx, rng):
    rows, cols = cfg["grid"]
    t, g = cfg["tile"], cfg.get("gap", 0)
    grid_w, grid_h = cols * t + (cols + 1) * g, rows * t + (rows + 1) * g
    W, H = cfg.get("canvas", (grid_w, grid_h))
    bg = cfg.get("bg", (0, 0, 0))
    if bg == "random":
        bg = tuple(rng.randint(60, 220) for _ in range(3))
    canvas = Image.new("RGB", (W, H), bg)
    ox, oy = (W - grid_w) // 2, (H - grid_h) // 2

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    if "fill" in cfg:
        cells = rng.sample(cells, rng.randint(*cfg["fill"]))
    lines = []
    for (r, c), cls in zip(cells, rng.sample(sorted(pools), len(cells))):
        x, y, s = ox + g + c * (t + g), oy + g + r * (t + g), t
        if cfg.get("jitter"):
            s = rng.randint(int(t * 0.65), t)
            x, y = x + rng.randint(0, t - s), y + rng.randint(0, t - s)
        tile = Image.open(rng.choice(pools[cls])).convert("RGB").resize((s, s))
        canvas.paste(tile, (x, y))
        lines.append(f"{class_to_idx[cls]} {(x + s / 2) / W:.6f} {(y + s / 2) / H:.6f} {s / W:.6f} {s / H:.6f}")
    return canvas, lines


def build(a, class_to_idx):
    pools = load_pools(a.src)
    names = {v: k for k, v in class_to_idx.items()}
    rng = random.Random(a.seed)
    for name, cfg in SCENARIOS.items():
        root = os.path.join(a.out, name)
        img_dir, lbl_dir = os.path.join(root, "images"), os.path.join(root, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(a.n):
            canvas, lines = compose(cfg, pools, class_to_idx, rng)
            canvas.save(os.path.join(img_dir, f"{name}_{i:03d}.jpg"), quality=cfg.get("quality", 95))
            with open(os.path.join(lbl_dir, f"{name}_{i:03d}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
        with open(os.path.join(root, "data.yaml"), "w") as f:
            f.write(f"path: {os.path.abspath(root)}\ntrain: images\nval: images\ntest: images\n")
            f.write(f"nc: {len(names)}\nnames:\n")
            for k in sorted(names):
                f.write(f"  {k}: {names[k]}\n")
    print(f"built {len(SCENARIOS)} scenarios x {a.n} images in {a.out}")


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def clean(preds, one_per_class):
    if not one_per_class:
        return [(k, b) for k, b, _ in preds]
    best = {}
    for k, b, c in preds:
        if k not in best or c > best[k][1]:
            best[k] = (b, c)
    return [(k, b) for k, (b, _) in best.items()]


def score_images(model, root, conf, agnostic=False, one_per_class=False):
    img_dir, lbl_dir = os.path.join(root, "images"), os.path.join(root, "labels")
    ok = gt_total = found = pred_total = 0
    files = sorted(os.listdir(img_dir))
    for f in files:
        r = model.predict(os.path.join(img_dir, f), conf=conf, agnostic_nms=agnostic, verbose=False)[0]
        H, W = r.orig_shape
        gts = []
        for line in open(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")):
            if line.strip():
                k, cx, cy, w, h = line.split()
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                gts.append((int(k), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        preds = clean([(int(c), tuple(b), s) for c, b, s in
                       zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(), r.boxes.conf.tolist())], one_per_class)
        used, hit = set(), 0
        for k, gb in gts:
            best, best_j = 0.5, None
            for j, (pk, pb) in enumerate(preds):
                if j not in used and pk == k and iou(gb, pb) >= best:
                    best, best_j = iou(gb, pb), j
            if best_j is not None:
                used.add(best_j)
                hit += 1
        ok += hit == len(gts) and len(preds) == len(gts)
        gt_total += len(gts)
        found += hit
        pred_total += len(preds)
    return {
        "images": len(files),
        "image_acc": round(ok / len(files), 3),
        "recall": round(found / max(gt_total, 1), 3),
        "precision": round(found / max(pred_total, 1), 3),
    }


def evaluate(a):
    from ultralytics import YOLO

    rows = []
    for w in a.weights:
        tag = os.path.basename(os.path.dirname(os.path.dirname(w))) or os.path.splitext(os.path.basename(w))[0]
        model = YOLO(w)
        for name in SCENARIOS:
            root = os.path.join(a.out, name)
            row = {"model": tag, "scenario": name}
            if not a.skip_map:
                m = model.val(data=os.path.join(root, "data.yaml"), split="test", conf=0.001, plots=False,
                              verbose=False, project=os.path.abspath(os.path.join(a.results, "val")), name=f"{tag}_{name}", exist_ok=True)
                row.update({"mAP50": round(m.box.map50, 3), "mAP50_95": round(m.box.map, 3)})
            row.update(score_images(model, root, a.conf, a.agnostic, a.one_per_class))
            rows.append(row)
            print(row)
            if a.examples:
                imgs = sorted(os.listdir(os.path.join(root, "images")))[: a.examples]
                model.predict([os.path.join(root, "images", f) for f in imgs], conf=a.conf, save=True,
                              project=os.path.abspath(os.path.join(a.results, "examples", tag)), name=name, exist_ok=True, verbose=False)

    df = pd.DataFrame(rows)
    os.makedirs(a.results, exist_ok=True)
    df.to_csv(os.path.join(a.results, f"summary{a.tag}.csv"), index=False)
    print("\nimage accuracy (every object found with the right ID and box, nothing extra):")
    print(df.pivot(index="scenario", columns="model", values="image_acc").to_string())
    print("\nobject recall:")
    print(df.pivot(index="scenario", columns="model", values="recall").to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="*", default=[])
    p.add_argument("--src", default="../data/processed_224/test")
    p.add_argument("--classes", default="../data/processed_224/class_to_idx.json")
    p.add_argument("--out", default="../data/stress")
    p.add_argument("--results", default="results/stress")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--examples", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--agnostic", action="store_true")
    p.add_argument("--one_per_class", action="store_true")
    p.add_argument("--skip_map", action="store_true")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    with open(a.classes) as f:
        class_to_idx = json.load(f)
    if a.rebuild or not os.path.isdir(a.out):
        build(a, class_to_idx)
    if a.weights:
        evaluate(a)


if __name__ == "__main__":
    main()
