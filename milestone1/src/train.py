import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data import get_loaders
from models import MODEL_NAMES, build_model, count_params, freeze_backbone, frozen_bn_to_eval, unfreeze_top

DEFAULTS = {
    "custom": {"epochs": 40, "lr": 1e-3},
    "mobilenet": {"epochs": 20, "lr": 1e-3, "ft_lr": 1e-4, "unfreeze": 4},
    "resnet50": {"epochs": 20, "lr": 1e-3, "ft_lr": 1e-4, "unfreeze": 2},
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    if training:
        frozen_bn_to_eval(model)

    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def make_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train(args, trial=None):
    set_seed(args.seed)
    device = get_device()
    train_loader, val_loader, _, classes = get_loaders(
        args.data_dir, args.batch_size, args.num_workers, augment=not args.no_augment
    )
    model = build_model(args.model, len(classes)).to(device)
    if args.init_ckpt:
        model.load_state_dict(torch.load(args.init_ckpt, map_location=device))
    criterion = nn.CrossEntropyLoss()

    if args.model == "custom":
        phases = [(args.epochs, args.lr, None)]
    else:
        freeze_backbone(model, args.model)
        phases = [
            (args.head_epochs, args.lr, None),
            (args.epochs - args.head_epochs, args.ft_lr, args.unfreeze),
        ]

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.name}.pt")
    last_path = os.path.join(args.out_dir, f"{args.name}_last.pt")
    head_path = os.path.join(args.out_dir, f"{args.name}_head.pt")
    history, best_acc, best_epoch, epoch = [], 0.0, 0, 0
    start = time.time()
    print(f"[{args.name}] device={device} classes={len(classes)} params={count_params(model):,}")

    state = None
    if args.resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        history, best_acc, best_epoch, epoch = state["history"], state["best_acc"], state["best_epoch"], state["epoch"]
        print(f"[{args.name}] resumed from epoch {epoch}, best val_acc so far {best_acc:.4f}")

    for phase, (n_epochs, lr, unfreeze) in enumerate(phases):
        if unfreeze:
            unfreeze_top(model, args.model, unfreeze)
        if state and phase < state["phase"]:
            continue
        optimizer = make_optimizer(model, lr, args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))

        first = 0
        if state and phase == state["phase"]:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            first = state["epoch_in_phase"]

        for ep in range(first, n_epochs):
            epoch += 1
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, args.max_batches)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device, max_batches=args.max_batches)
            scheduler.step()

            history.append({
                "epoch": epoch, "lr": lr, "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss, "val_acc": va_acc,
            })
            print(f"[{args.name}] epoch {epoch}/{args.epochs} "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

            if va_acc > best_acc:
                best_acc, best_epoch = va_acc, epoch
                torch.save(model.state_dict(), ckpt_path)

            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "phase": phase, "epoch_in_phase": ep + 1,
                "epoch": epoch, "best_acc": best_acc, "best_epoch": best_epoch, "history": history,
            }, last_path)

            if trial is not None:
                trial.report(va_acc, epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

        if phase == 0 and args.model != "custom" and n_epochs > 0:
            torch.save(model.state_dict(), head_path)

    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, f"{args.name}_history.csv"), index=False)
    summary = {
        "name": args.name,
        "model": args.model,
        "params": count_params(model),
        "augment": not args.no_augment,
        "epochs": args.epochs,
        "head_epochs": None if args.model == "custom" else args.head_epochs,
        "lr": args.lr,
        "ft_lr": args.ft_lr,
        "unfreeze": args.unfreeze,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "train_minutes": round((time.time() - start) / 60, 2),
    }
    with open(os.path.join(args.out_dir, f"{args.name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.name}] best val_acc={best_acc:.4f} at epoch {best_epoch}, saved {ckpt_path}")
    return best_acc


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--data_dir", default="../data/processed_224")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--name")
    p.add_argument("--epochs", type=int)
    p.add_argument("--head_epochs", type=int, default=3)
    p.add_argument("--lr", type=float)
    p.add_argument("--ft_lr", type=float)
    p.add_argument("--unfreeze", type=int)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_batches", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init_ckpt")
    args = p.parse_args(argv)

    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.name is None:
        args.name = args.model + ("_noaug" if args.no_augment else "")
    return args


if __name__ == "__main__":
    train(parse_args())