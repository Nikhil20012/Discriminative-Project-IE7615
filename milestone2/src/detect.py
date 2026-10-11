import argparse
import os

from PIL import Image
from ultralytics import YOLO


def keep_best_per_class(r):
    best = {}
    for i, (k, c) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())):
        if int(k) not in best or c > best[int(k)][1]:
            best[int(k)] = (i, c)
    return r[sorted(i for i, _ in best.values())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--weights", default="results/detect/yolov8m_v2/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="results/detect_preds")
    p.add_argument("--all_boxes", action="store_true", help="turn off the one-box-per-object cleanup")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = YOLO(a.weights)
    for r in model.predict(a.images, conf=a.conf, agnostic_nms=not a.all_boxes, stream=True, verbose=False):
        if not a.all_boxes and len(r.boxes):
            r = keep_best_per_class(r)
        name = os.path.basename(r.path)
        Image.fromarray(r.plot()[..., ::-1]).save(os.path.join(a.out, name))

        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            dets.append((y1, x1, model.names[int(b.cls)], float(b.conf), (x1, y1, x2, y2)))
        print(f"{name}: {len(dets)} objects")
        for _, _, obj, c, box in sorted(dets, key=lambda d: (round(d[0] / 50), d[1])):
            print(f"  {obj}  conf={c:.2f}  box={box}")


if __name__ == "__main__":
    main()
