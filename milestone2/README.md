# Milestone 2: Multi-Object Detection with YOLOv8

## Overview

For Milestone 2 the task is object detection. Given one image with several objects in it, the model has to find every object, say which Object ID it is, and draw a box around it.

There is no ready-made detection dataset for our objects, so we built one. We took the single-object photos from Milestone 1 and pasted several of them onto one image. Since our code decides where each photo goes, it also knows the exact box and Object ID of every object, so the labels come for free.

We then fine-tuned two pretrained YOLOv8 models (YOLOv8s and YOLOv8m) on this dataset, tested them on held-out images, and ran a stress test with 16 different image formats to check how they hold up on the kinds of images we might get on demo day.

Our final model is YOLOv8m. On the test set it gets 0.984 mAP@0.5 and finds about 97% of the objects in 3x3 images with the right Object ID.


## 1. What the TAs asked for

These points from the TA announcements shaped our approach:

- At least 100 multi-object images per student, so at least 400 for our group of 4. We generated 4,000.
- The demo will mostly use 2x2 and 3x3 layouts (4 and 9 objects), so the training data needs plenty of both, plus other layouts and object counts so the model doesn't only learn one pattern.
- Background-only images are not needed for detection, so we didn't use any (they were already removed in Milestone 1).
- The Object IDs from the class Google Sheet are the labels.


## 2. Building the multi-object dataset (`make_multiobject.py`)

### How each image is made

1. Pick a layout (see below) and how many objects go in the image.
2. Pick that many different Object IDs at random. An Object ID never appears twice in one image.
3. For each one, pick a random photo of that object from the Milestone 1 data.
4. Resize each photo to its tile size, flip it horizontally half the time, and change its brightness and contrast a little (0.8 to 1.2 times).
5. Paste the photos onto a 640x640 canvas filled with a random mid-tone color (each RGB channel between 90 and 200).
6. Write one label line per photo: the class number and the box (center x, center y, width, height) as fractions of the image size. The box is exactly where the photo was pasted.

### Layouts

| Layout | Share of images | Objects | How the photos are placed |
|---|---|---|---|
| 3x3 grid | 35% | 9 | Half of these have photos edge to edge with no gaps (like a plain concatenation of photos). The other half have each photo at 85 to 100% of its cell size, shifted to a random spot inside the cell |
| 2x2 grid | 25% | 4 | Same as the 3x3 grid |
| Random grid | 20% | 2 to 9 | Grid shape depends on the count (1x2 or 2x1 for 2 objects, 2x2 for 3 to 4, 2x3 or 3x2 for 5 to 6, 3x3 for 7 to 9), some cells left empty, photos at 85 to 100% of the cell with random offsets |
| Collage | 20% | 2 to 9 | Photos at random positions and random sizes. Each photo is 20 to 45% of the canvas width when there are up to 4 objects, and 15 to 32% when there are more, so they still fit. Photos can overlap by at most 10% of the smaller one |

### No leakage between train and test

The Milestone 1 data is already split into train, val and test. Training images are built only from Milestone 1 training photos, validation images only from validation photos, and test images only from test photos. So the model is never tested on a photo it saw during training.

### Dataset size

| Split | Images | Built from |
|---|---|---|
| Train | 3,000 | Milestone 1 train photos (4,863) |
| Val | 500 | Milestone 1 val photos (1,042) |
| Test | 500 | Milestone 1 test photos (1,043) |

That is 4,000 images in total, with about 6.5 objects per image on average (the 500 validation images have 3,265 objects). The test set has 186 3x3 images, 122 2x2 images and 192 random grid and collage images.

The generator uses a fixed seed (42), so the dataset can be rebuilt exactly. It writes:

```
data/multiobject/
  images/train/  images/val/  images/test/    the composite images
  labels/train/  labels/val/  labels/test/    one .txt label file per image
  manifest.csv                                every pasted photo: source file, Object ID, position, size, layout
  data.yaml                                   dataset paths and the 73 class names, for YOLO
  test_grid9.txt, test_grid4.txt, ...         image lists for each layout, so each layout can be scored on its own
```

### Our first version

Before the TAs told us the demo would mostly use 3x3 images, our first dataset had 1,500 training images with 2 to 6 objects each, using only random grids and collages. The largest grid was 2x3, so the model never saw a 9-object image. We trained YOLOv8s on it (in `notebooks/Milestone2_YOLOv8s_Training.ipynb`) and got 0.966 mAP@0.5 on its own test set, with 72.3% of test images fully correct. We then added the 3x3 and 2x2 layouts, raised the maximum to 9 objects, doubled the dataset and retrained. Everything below is about the second version.


## 3. Training (`train_yolo.py`)

We fine-tuned YOLOv8 models that were pretrained on the COCO dataset. Ultralytics replaces the original 80-class output with a new 73-class one, and all layers are trained (nothing frozen), since the model has to learn fine differences between our objects that COCO doesn't cover.

| Setting | Value |
|---|---|
| Models | YOLOv8s (about 11M parameters, 22.6 MB) and YOLOv8m (25.9M parameters, 52.1 MB) |
| Starting weights | COCO-pretrained (`yolov8s.pt`, `yolov8m.pt`) |
| Epochs | up to 100, with early stopping if validation doesn't improve for 15 epochs |
| Image size | 640 |
| Batch size | 16 |
| Seed | 42 |
| Model selection | Weights from the epoch with the best validation score |
| Hardware | NVIDIA RTX 5090 (32 GB) rented on Vast.ai, PyTorch 2.14, CUDA 13.0, Ultralytics 8.4 |

On top of the augmentation in the generator, YOLO adds its own augmentation while training:

| Augmentation | Setting |
|---|---|
| Mosaic (4 training images combined into one) | on, turned off for the last 10 epochs |
| Random scaling | up to 50% |
| Random shift | up to 10% |
| Color shifts | hue, saturation and brightness (YOLO defaults) |
| Horizontal flip | 50% chance |
| Vertical flip, rotation | off, since the demo images will be upright |

```bash
python src/make_multiobject.py
python src/train_yolo.py --name yolov8s_v2 --epochs 100
python src/train_yolo.py --name yolov8m_v2 --weights yolov8m.pt --epochs 100
```


## 4. Evaluation (`evaluate_yolo.py`)

Each model's best weights are evaluated once on the 500 test images, and again on just the 3x3 and just the 2x2 test images. We report:

- **mAP@0.5 and mAP@0.5:0.95:** the standard detection scores. A box counts as correct if it has the right Object ID and overlaps the true box enough (IoU of at least 0.5, or averaged over 0.5 to 0.95)
- **Precision and recall**
- **AP for every class** (saved as CSV)
- **Image-level check:** for each image, does the set of predicted Object IDs exactly match the true set? This is the closest to what matters in the demo, since one mistake anywhere makes the whole image wrong. It uses a confidence threshold of 0.25.

```bash
python src/evaluate_yolo.py --weights results/detect/yolov8m_v2/weights/best.pt --out results/eval/eval_yolov8m_v2
python src/evaluate_yolo.py --weights results/detect/yolov8m_v2/weights/best.pt --out results/eval/eval_yolov8m_v2 --layout grid9
python src/evaluate_yolo.py --weights results/detect/yolov8m_v2/weights/best.pt --out results/eval/eval_yolov8m_v2 --layout grid4
```


## 5. Results

### Validation (500 images, 3,265 objects)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8s | 0.977 | 0.971 | 0.989 | 0.989 |
| YOLOv8m | 0.978 | 0.974 | 0.990 | 0.990 |

### Test

| | YOLOv8s | YOLOv8m |
|---|---|---|
| **All test images (500)** | | |
| mAP@0.5 | 0.986 | 0.984 |
| mAP@0.5:0.95 | 0.985 | 0.983 |
| Precision / recall | 0.982 / 0.962 | 0.977 / 0.964 |
| Images with every Object ID right | 76.6% | 78.4% |
| **3x3 images (186)** | | |
| mAP@0.5 | 0.985 | 0.982 |
| Object ID precision / recall | 0.975 / 0.970 | 0.969 / 0.977 |
| Images with all 9 right | 71.0% | 72.6% |
| **2x2 images (122)** | | |
| mAP@0.5 | 0.985 | 0.985 |
| Object ID precision / recall | 0.967 / 0.973 | 0.952 / 0.973 |
| Images with all 4 right | 83.6% | 81.1% |

### What we found

1. **Both models are very accurate and almost tied.** The differences between them are 1 to 2 points at most, which on 186 3x3 images is only about 3 images.
2. **mAP@0.5:0.95 is almost the same as mAP@0.5.** Usually the stricter score is a lot lower. Here the objects are sharp-edged photos on a plain background, so finding the box is easy and the boxes are almost perfect. The hard part is picking the right Object ID, not finding the object.
3. **Object-level and image-level numbers tell different stories.** About 97% of individual objects in 3x3 images are found with the right ID, but only about 72% of 3x3 images have all 9 right. With 9 objects per image, even a small error rate per object adds up to roughly one mistake every 3 to 4 images.
4. **Adding 3x3 data fixed the main gap of the first version.** The first model had never seen a 9-object image. The new models handle 3x3 images about as well as the rest of the test set.


## 6. Stress test (`stress_test.py`)

We don't know exactly how the demo images will be made, so we tested both models on 16 other ways someone could put 2x2 and 3x3 images together. Each format has 40 images, built only from Milestone 1 test photos.

| Format | What it looks like |
|---|---|
| 3x3 plain, 2x2 plain | Photos edge to edge at their original 224x224 size, no background |
| 3x3 and 2x2 with gaps | 12 pixel white or black gaps between photos |
| 3x3 small, 3x3 large, 2x2 large | Photos at 128 or 320 pixels |
| 3x3 on a 640x640 canvas, 3x3 on a 1280x720 canvas | The grid centered on a larger colored or gray image |
| 3x3 partial, 2x2 partial | Some cells empty (5 to 8 of 9, or 2 to 3 of 4 filled) |
| 3x3 jittered | Photos at random sizes and positions inside their cells |
| 3x3 low quality | Saved as a heavily compressed JPEG |
| 4x4, 1x4 row | 16 objects, or 4 objects in one row |

For each image we check if every object was found with the right Object ID and a box overlapping the true one (IoU of at least 0.5), with no extra boxes. The tables show the share of images that pass.

### Post-processing

The first run showed that recall was high (96 to 99% in most formats) but full-image accuracy was much lower, so most failures were extra wrong boxes, not missed objects. Sometimes the model put two different Object IDs on the same photo. Since in these images every photo holds exactly one object and no Object ID appears twice, we added two simple rules after the model runs:

1. **One box per location:** class-agnostic NMS, so overlapping boxes are removed even if they have different Object IDs.
2. **One box per Object ID:** if the same ID shows up twice, keep only the most confident one.

### Results (share of images where everything is right)

| Format | YOLOv8s | YOLOv8m | YOLOv8s + post-processing | YOLOv8m + post-processing |
|---|---|---|---|---|
| 3x3 plain | 60.0% | 60.0% | 65.0% | **72.5%** |
| 3x3 white gaps | 72.5% | 75.0% | 77.5% | **87.5%** |
| 3x3 black gaps | 70.0% | 72.5% | 77.5% | **85.0%** |
| 3x3 small photos | 65.0% | 67.5% | 65.0% | **70.0%** |
| 3x3 large photos | 77.5% | 62.5% | **80.0%** | 75.0% |
| 3x3 on 640x640 canvas | 70.0% | 72.5% | **75.0%** | **75.0%** |
| 3x3 on 1280x720 canvas | 35.0% | 52.5% | 42.5% | **60.0%** |
| 3x3 partial | 77.5% | 67.5% | **80.0%** | 75.0% |
| 3x3 jittered | 65.0% | 75.0% | 67.5% | **82.5%** |
| 3x3 low quality JPEG | 47.5% | 30.0% | **55.0%** | 47.5% |
| 2x2 plain | 75.0% | 85.0% | 82.5% | **87.5%** |
| 2x2 white gaps | 70.0% | 75.0% | 80.0% | **90.0%** |
| 2x2 large photos | 77.5% | 72.5% | **82.5%** | **82.5%** |
| 2x2 partial | 85.0% | 87.5% | 90.0% | **92.5%** |
| 4x4 | 35.0% | 42.5% | 45.0% | **52.5%** |
| 1x4 row | 75.0% | 82.5% | 82.5% | **92.5%** |

With post-processing, YOLOv8m still finds 92 to 98% of all objects with the right Object ID in every format.

### What we found

1. **Post-processing helped in every format** for YOLOv8m, by 2.5 to 17.5 points, without retraining.
2. **YOLOv8m is the more robust model.** It has higher object recall than YOLOv8s in 15 of 16 formats, and with post-processing it has the better image score in 11 of 16 formats and ties in 2 more.
3. **The weak spots are very small photos and low image quality.** On the 1280x720 canvas each photo ends up about 100 pixels wide after YOLO resizes the image to 640, which is smaller than anything in training. Heavy JPEG compression was never part of training either. Both could be fixed by adding smaller tiles and compressed images to the generator.


## 7. Final model

**YOLOv8m with post-processing.** It is tied with YOLOv8s on the test set, but it is more robust across the different formats in the stress test, which matters most since the demo format is not fixed. YOLOv8s is a good smaller alternative at less than half the size.


## 8. Running detection (`detect.py`)

This is the script for the demo. Give it one or more images or a folder:

```bash
python src/detect.py path/to/image.jpg
python src/detect.py path/to/folder/
```

For each image it saves a copy with the boxes and Object IDs drawn on it (in `results/detect_preds/`), and prints every detected object with its confidence and box, in reading order (top row left to right, then the next row). Post-processing is on by default, `--all_boxes` turns it off, and `--weights` picks a different model. It uses YOLOv8m by default.

Sample detections on test images are in `results/report_samples/`.


## 9. Folder structure

```
milestone2/
  README.md
  notebooks/
    Milestone2_YOLOv8s_Training.ipynb   first YOLOv8s run on the first dataset (Colab)
  src/
    make_multiobject.py   builds the multi-object dataset and labels
    train_yolo.py         fine-tunes a pretrained YOLOv8
    evaluate_yolo.py      test metrics, per-class AP and the image-level check
    stress_test.py        builds the 16 stress test formats and scores models on them
    detect.py             runs a model on new images (used in the demo)
  results/
    detect/               training runs for yolov8s_v2 and yolov8m_v2 (curves, confusion matrices, weights)
    eval/                 test logs and per-class AP files
    eval_plots/           test set precision-recall curves and confusion matrices
    stress/               stress test summaries and example detections
    report_samples/       sample detections on test images
    logs/                 training logs
```

All commands are run from inside the `milestone2/` folder. The scripts look for the data in `../data/` by default. The model weights (`.pt` files) are not in the repo because of their size.

```bash
python src/stress_test.py --weights results/detect/yolov8s_v2/weights/best.pt results/detect/yolov8m_v2/weights/best.pt
python src/stress_test.py --weights results/detect/yolov8s_v2/weights/best.pt results/detect/yolov8m_v2/weights/best.pt --agnostic --one_per_class --skip_map --tag _fixed
```


## 10. Limitations

- **Boxes cover the whole photo, not the object.** Each box is the tile the photo was pasted into. In close-up photos that is almost the same as the object, but in far shots the object is small inside the tile. This comes from building the dataset by combining photos.
- **The demo format isn't fixed.** We covered a wide range in the stress test, but the model is weaker on very small photos and heavily compressed images.
- **The composites are simpler than real scenes.** Photos sit on plain backgrounds with clean edges, which is why finding the boxes is so easy. Real photos with several objects would be harder.
- **The test photos repeat.** The 500 test images reuse the 1,043 Milestone 1 test photos in different combinations, so the test images are not fully independent of each other.
- **Same environment issue as Milestone 1.** Each object was photographed by one student in their own home, so the model may partly use the background of each photo to recognize the object.
