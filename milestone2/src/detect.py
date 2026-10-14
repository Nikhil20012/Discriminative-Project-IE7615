import argparse
import os

from PIL import Image
from ultralytics import YOLO

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def expand(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(IMAGE_TYPES))
        else:
            files.append(p)
    return files


def keep_best_per_class(r):
    best = {}
    for i, (k, c) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
        if int(k) not in best or c > best[int(k)][1]:
            best[int(k)] = (i, c)
    return r[sorted(i for i, _ in best.values())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="image files and/or folders of images")
    p.add_argument("--weights", default="results/detect/yolov8m_v2/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="results/detect_preds")
    p.add_argument("--all_boxes", action="store_true", help="turn off the one-box-per-object cleanup")
    a = p.parse_args()

    files = expand(a.images)
    if not files:
        raise SystemExit("No images found.")
    os.makedirs(a.out, exist_ok=True)

    model = YOLO(a.weights)
    for path in files:
        r = model.predict(path, conf=a.conf, imgsz=a.imgsz, agnostic_nms=not a.all_boxes, verbose=False)[0]
        if not a.all_boxes and len(r.boxes):
            r = keep_best_per_class(r)
        name = os.path.basename(path)
        Image.fromarray(r.plot()[..., ::-1]).save(os.path.join(a.out, name))

        h = r.orig_shape[0]
        row_height = max(h / 12, 1)
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            dets.append((round(y1 / row_height), x1, model.names[int(b.cls)], float(b.conf), (x1, y1, x2, y2)))
        print(f"{name}: {len(dets)} objects")
        for _, _, obj, c, box in sorted(dets, key=lambda d: (d[0], d[1])):
            print(f"  {obj}  conf={c:.2f}  box={box}")


if __name__ == "__main__":
    main()