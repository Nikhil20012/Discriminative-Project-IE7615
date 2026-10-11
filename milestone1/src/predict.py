import argparse
import json
import os

import torch
from PIL import Image, ImageOps

from data import get_transforms
from models import build_model
from train import get_device

EXTS = {".jpg", ".jpeg", ".png"}


def load_model(name, results_dir, num_classes, device):
    with open(os.path.join(results_dir, f"{name}_summary.json")) as f:
        model_type = json.load(f)["model"]
    model = build_model(model_type, num_classes, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(results_dir, f"{name}.pt"), map_location="cpu"))
    return model.to(device).eval()


def load_image(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    return ImageOps.pad(im, (224, 224), color=(0, 0, 0))


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(
                os.path.join(p, f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in EXTS
            )
        else:
            files.append(p)
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--name", default="resnet50_tuned")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--classes", default="../data/processed_224/class_to_idx.json")
    p.add_argument("--topk", type=int, default=3)
    a = p.parse_args()

    with open(a.classes) as f:
        idx_to_class = {i: c for c, i in json.load(f).items()}

    device = get_device()
    model = load_model(a.name, a.results_dir, len(idx_to_class), device)
    _, tf = get_transforms(augment=False)

    with torch.no_grad():
        for path in collect(a.images):
            x = tf(load_image(path)).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1)[0].cpu()
            top = probs.topk(a.topk)
            preds = ", ".join(
                f"{idx_to_class[i.item()]} ({v.item():.1%})" for v, i in zip(top.values, top.indices)
            )
            print(f"{path}: {preds}")


if __name__ == "__main__":
    main()