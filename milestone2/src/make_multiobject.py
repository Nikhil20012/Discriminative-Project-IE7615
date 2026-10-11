import argparse
import csv
import json
import os
import random

from PIL import Image, ImageEnhance

EXTS = {".jpg", ".jpeg", ".png"}


def list_images(split_dir):
    out = {}
    for cls in sorted(os.listdir(split_dir)):
        d = os.path.join(split_dir, cls)
        if os.path.isdir(d):
            files = [os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in EXTS]
            if files:
                out[cls] = files
    return out


def prep_tile(path, side, rng):
    im = Image.open(path).convert("RGB").resize((side, side), Image.BILINEAR)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.8, 1.2))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.2))
    return im


def grid_layout(n, size, rng, shape=None, tight=False):
    if shape:
        rows, cols = shape
    else:
        cols = 1 if n == 1 else (2 if n <= 4 else 3)
        rows = -(-n // cols)
        rows, cols = (rows, cols) if rng.random() < 0.5 else (cols, rows)
        while rows * cols < n:
            rows += 1
    cw, ch = size / cols, size / rows
    cells = rng.sample([(r, c) for r in range(rows) for c in range(cols)], n)
    boxes = []
    for r, c in cells:
        if tight:
            s = int(min(cw, ch))
            x, y = int(c * cw), int(r * ch)
        else:
            s = int(min(cw, ch) * rng.uniform(0.85, 1.0))
            x = int(c * cw + rng.uniform(0, cw - s))
            y = int(r * ch + rng.uniform(0, ch - s))
        boxes.append((x, y, s))
    return boxes


def collage_layout(n, size, rng, max_overlap=0.1, tries=60):
    lo, hi = (0.2, 0.45) if n <= 4 else (0.15, 0.32)
    boxes = []
    for _ in range(n):
        for _ in range(tries):
            s = int(size * rng.uniform(lo, hi))
            x, y = rng.randint(0, size - s), rng.randint(0, size - s)
            ok = True
            for bx, by, bs in boxes:
                iw = max(0, min(x + s, bx + bs) - max(x, bx))
                ih = max(0, min(y + s, by + bs) - max(y, by))
                if iw * ih > max_overlap * min(s * s, bs * bs):
                    ok = False
                    break
            if ok:
                boxes.append((x, y, s))
                break
    return boxes


LAYOUTS = ["grid9", "grid4", "grid", "collage"]


def pick_boxes(lay, size, min_objs, max_objs, n_classes, rng):
    if lay == "grid9":
        return grid_layout(9, size, rng, shape=(3, 3), tight=rng.random() < 0.5)
    if lay == "grid4":
        return grid_layout(4, size, rng, shape=(2, 2), tight=rng.random() < 0.5)
    n = rng.randint(min_objs, min(max_objs, n_classes))
    return grid_layout(n, size, rng) if lay == "grid" else collage_layout(n, size, rng)


def make_split(split, src, out, class_to_idx, n_images, size, min_objs, max_objs, weights, rng):
    pools = list_images(os.path.join(src, split))
    classes = list(pools)
    img_dir = os.path.join(out, "images", split)
    lbl_dir = os.path.join(out, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    rows = []
    by_layout = {lay: [] for lay in LAYOUTS}
    for i in range(n_images):
        lay = rng.choices(LAYOUTS, weights=weights)[0]
        boxes = pick_boxes(lay, size, min_objs, max_objs, len(classes), rng)
        chosen = rng.sample(classes, len(boxes))
        bg = tuple(rng.randint(90, 200) for _ in range(3))
        canvas = Image.new("RGB", (size, size), bg)
        lines = []
        for cls, (x, y, s) in zip(chosen, boxes):
            src_path = rng.choice(pools[cls])
            canvas.paste(prep_tile(src_path, s, rng), (x, y))
            k = class_to_idx[cls]
            lines.append(f"{k} {(x + s / 2) / size:.6f} {(y + s / 2) / size:.6f} {s / size:.6f} {s / size:.6f}")
            rows.append([f"{split}_{i:05d}", split, lay, cls, k, src_path, x, y, s])
        name = f"{split}_{i:05d}"
        canvas.save(os.path.join(img_dir, name + ".jpg"), quality=92)
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        by_layout[lay].append(f"./images/{split}/{name}.jpg")

    for lay, paths in by_layout.items():
        with open(os.path.join(out, f"{split}_{lay}.txt"), "w") as f:
            f.write("\n".join(paths) + "\n")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="../data/processed_224")
    p.add_argument("--out", default="../data/multiobject")
    p.add_argument("--n_train", type=int, default=3000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--min_objs", type=int, default=2)
    p.add_argument("--max_objs", type=int, default=9)
    p.add_argument("--weights", type=float, nargs=4, default=[0.35, 0.25, 0.2, 0.2],
                   help="share of grid9 (3x3), grid4 (2x2), random grid, collage")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = random.Random(a.seed)
    with open(os.path.join(a.src, "class_to_idx.json")) as f:
        class_to_idx = json.load(f)
    names = {v: k for k, v in class_to_idx.items()}

    manifest = []
    for split, n in (("train", a.n_train), ("val", a.n_val), ("test", a.n_test)):
        manifest += make_split(split, a.src, a.out, class_to_idx, n, a.size,
                               a.min_objs, a.max_objs, a.weights, rng)
        print(f"{split}: {n} images")

    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "layout", "object_id", "class_idx", "source", "x", "y", "side"])
        w.writerows(manifest)

    root = os.path.abspath(a.out)
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n")
        f.write(f"nc: {len(names)}\nnames:\n")
        for k in sorted(names):
            f.write(f"  {k}: {names[k]}\n")
    print("wrote", os.path.join(a.out, "data.yaml"))


if __name__ == "__main__":
    main()