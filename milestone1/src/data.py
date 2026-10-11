import json
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(augment=True):
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if not augment:
        return eval_tf, eval_tf

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def get_loaders(data_dir, batch_size=32, num_workers=4, augment=True):
    train_tf, eval_tf = get_transforms(augment)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)

    with open(os.path.join(data_dir, "class_to_idx.json")) as f:
        expected = json.load(f)
    for ds in (train_ds, val_ds, test_ds):
        assert ds.class_to_idx == expected, "class mapping does not match class_to_idx.json"

    loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    return (
        loader(train_ds, True),
        loader(val_ds, False),
        loader(test_ds, False),
        train_ds.classes,
    )