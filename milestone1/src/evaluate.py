import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import get_loaders
from models import build_model, count_params
from train import get_device


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def predict_all(model, loader, device):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def ms_per_image(model, loader, device, n_images=200):
    model.eval()
    batches, count = [], 0
    for x, _ in loader:
        batches.append(x)
        count += len(x)
        if count >= n_images:
            break
    images = torch.cat(batches)[:n_images].to(device)

    with torch.no_grad():
        for img in images[:10]:
            model(img.unsqueeze(0))
        sync(device)
        start = time.perf_counter()
        for img in images:
            model(img.unsqueeze(0))
        sync(device)
    return (time.perf_counter() - start) * 1000 / len(images)


def top_confusions(cm, classes, k=10):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off.ravel())[::-1][:k]
    rows = []
    for flat in order:
        i, j = np.unravel_index(flat, off.shape)
        if off[i, j] > 0:
            rows.append({"true": classes[i], "predicted": classes[j], "count": int(off[i, j])})
    return pd.DataFrame(rows)


def plot_confusion(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=5)
    ax.set_yticklabels(classes, fontsize=5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate(name, args, test_loader, classes, device, out_dir):
    with open(os.path.join(args.results_dir, f"{name}_summary.json")) as f:
        summary = json.load(f)
    ckpt = os.path.join(args.results_dir, f"{name}.pt")

    model = build_model(summary["model"], len(classes), pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)

    y_true, y_pred = predict_all(model, test_loader, device)
    labels = list(range(len(classes)))

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(os.path.join(out_dir, f"{name}_per_class.csv"))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(out_dir, f"{name}_confusion.csv"))
    plot_confusion(cm, classes, f"{name} test confusion matrix", os.path.join(out_dir, f"{name}_confusion.png"))
    top_confusions(cm, classes).to_csv(os.path.join(out_dir, f"{name}_top_confusions.csv"), index=False)

    result = {
        "name": name,
        "model": summary["model"],
        "params": count_params(model),
        "size_mb": round(os.path.getsize(ckpt) / 1e6, 2),
        "best_epoch": summary["best_epoch"],
        "best_val_acc": round(summary["best_val_acc"], 4),
        "test_acc": round(accuracy_score(y_true, y_pred), 4),
        "test_macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "ms_per_image": round(ms_per_image(model, test_loader, device), 2),
        "train_minutes": summary["train_minutes"],
    }
    print(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", default=["custom", "mobilenet", "resnet50"])
    p.add_argument("--data_dir", default="../data/processed_224")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = get_device()
    out_dir = os.path.join(args.results_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)
    _, _, test_loader, classes = get_loaders(args.data_dir, args.batch_size, args.num_workers, augment=False)

    rows = [evaluate(name, args, test_loader, classes, device, out_dir) for name in args.names]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)
    print("\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()