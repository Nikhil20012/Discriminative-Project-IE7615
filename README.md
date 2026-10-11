# IE 7615 Discriminative Deep Learning Project

This repo has our group's work for the IE 7615 discriminative deep learning project. The goal of the project is a system that can identify objects in images: first a single object per image, then several objects in one image along with where they are.

The dataset is the shared class dataset: 73 objects, about 100 photos each, taken by students in Sections 1 and 3. Every object has an Object ID (OBJ001 to OBJ073) from the class Google Sheet, and those IDs are the labels everywhere in this repo.


## Milestones

| Milestone | Task | What we built | Best result |
|---|---|---|---|
| [Milestone 1](milestone1/README.md) | Single-object classification: given one image, predict its Object ID | A custom CNN, MobileNetV3-Large and ResNet-50, plus augmentation, Optuna tuning and seed experiments (11 trained models) | ResNet-50: 98.2% test accuracy |
| [Milestone 2](milestone2/README.md) | Object detection: given an image with several objects, find each one and predict its Object ID and location | A generator that builds multi-object images with labels, and YOLOv8s and YOLOv8m trained on them, plus a stress test with 16 image formats | YOLOv8m: 0.983 test mAP@0.5, about 97% of objects found with the right ID in 3x3 images |

Each milestone folder has its own README with the full details: what we did, how to run it, and the results.


## Repo structure

```
README.md            this file
requirements.txt     Python packages for both milestones
milestone1/          single-object classification
  README.md
  notebooks/         preprocessing notebook (Colab)
  src/               data loading, models, training, tuning, evaluation, prediction
  results/           training history, evaluation and tuning results
  logs/              output of every run
milestone2/          multi-object detection with YOLOv8
  README.md
  notebooks/         first YOLOv8 training run (Colab)
  src/               dataset generator, training, evaluation, stress test, detection
  results/           training runs, evaluation, stress test and sample detections
data/                datasets (not in the repo, see below)
```


## Setup

```bash
pip install -r requirements.txt
```

The datasets and trained model weights aren't in the repo because of their size.

- **Milestone 1 data:** the raw images come from the class shared Google Drive folder. `milestone1/notebooks/Preprocessing.ipynb` cleans them and produces `data/processed_224/`.
- **Milestone 2 data:** `milestone2/src/make_multiobject.py` builds the multi-object dataset in `data/multiobject/` from `data/processed_224/`.
- **Model weights:** saved as `.pt` files in each milestone's `results/` folder when you train. They are excluded from git.

All scripts are run from inside their milestone folder, for example:

```bash
cd milestone2
python src/detect.py path/to/image.jpg
```
