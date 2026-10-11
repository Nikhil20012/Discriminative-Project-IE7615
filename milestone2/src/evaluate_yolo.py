import argparse
import csv
import os
import re

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="results/detect/yolov8m_v2/weights/best.pt")
    p.add_argument("--data", default="../data/multiobject/data.yaml")
    p.add_argument("--test_dir", default="../data/multiobject")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="results/detect_eval")
    p.add_argument("--layout", choices=["grid9", "grid4", "grid", "collage"],
                   help="score only test images of this layout, e.g. grid9 for the 3x3 demo case")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = a.layout or "all"

    img_dir = os.path.join(a.test_dir, "images", "test")
    lbl_dir = os.path.join(a.test_dir, "labels", "test")
    data = a.data
    if a.layout:
        y = re.sub(r"(?m)^test:.*$", f"test: test_{a.layout}.txt", open(a.data).read())
        data = os.path.join(a.test_dir, f"data_{a.layout}.yaml")
        open(data, "w").write(y)
        files = [os.path.basename(l.strip()) for l in open(os.path.join(a.test_dir, f"test_{a.layout}.txt")) if l.strip()]
    else:
        files = sorted(os.listdir(img_dir))

    model = YOLO(a.weights)
    m = model.val(data=data, split="test", conf=0.001, plots=True, project=os.path.abspath(a.out), name=tag, exist_ok=True)
    print(f"[{tag}] {len(files)} images")
    print(f"mAP50 {m.box.map50:.4f} | mAP50-95 {m.box.map:.4f} | P {m.box.mp:.4f} | R {m.box.mr:.4f}")

    with open(os.path.join(a.out, f"per_class_ap_{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "object_id", "ap50", "ap50_95"])
        for i, k in enumerate(m.box.ap_class_index):
            w.writerow([int(k), model.names[int(k)], f"{m.box.ap50[i]:.4f}", f"{m.box.ap[i]:.4f}"])

    exact = tp = fp = fn = n_img = 0
    for fn_ in files:
        gt = {int(l.split()[0]) for l in open(os.path.join(lbl_dir, os.path.splitext(fn_)[0] + ".txt")) if l.strip()}
        r = model.predict(os.path.join(img_dir, fn_), conf=a.conf, verbose=False)[0]
        pred = {int(c) for c in r.boxes.cls.tolist()}
        exact += pred == gt
        tp += len(pred & gt); fp += len(pred - gt); fn += len(gt - pred); n_img += 1
    print(f"image-level: exact set match {exact / n_img:.3f} | ID precision {tp / max(tp + fp, 1):.3f} "
          f"| ID recall {tp / max(tp + fn, 1):.3f} (conf>={a.conf})")


if __name__ == "__main__":
    main()