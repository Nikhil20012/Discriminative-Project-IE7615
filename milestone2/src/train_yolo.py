import argparse
import os

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/multiobject/data.yaml")
    p.add_argument("--weights", default="yolov8s.pt")   # try yolov8n.pt / yolov8m.pt too
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--freeze", type=int, default=0, help="freeze first N layers (10 = backbone)")
    p.add_argument("--name", default="yolov8s")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    model = YOLO(a.weights)  # COCO-pretrained; the head is rebuilt for our 73 classes
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed,
        freeze=a.freeze or None, project=os.path.abspath("results/detect"), name=a.name,
        patience=15, fliplr=0.5, flipud=0.0, degrees=0.0, close_mosaic=10,
    )


if __name__ == "__main__":
    main()
