# Milestone 1: Single-Object Classification

## Overview

For Milestone 1 we built image classifiers that take a single 224x224 image of an object and predict its Object ID. There are 73 objects (classes) in the class dataset.

We trained three CNN models and compared them:

1. A custom CNN trained from scratch
2. MobileNetV3-Large with transfer learning
3. ResNet-50 with transfer learning

We also ran a few extra experiments: training with and without augmentation, training the custom CNN for longer, tuning hyperparameters with Optuna, and repeating some runs with different random seeds. In total we trained 11 models.

Our best model was ResNet-50 at about 98.2 to 98.3% test accuracy. MobileNetV3 (tuned) got 97.7% while being about 5.5 times smaller.


## 1. Dataset

We used the merged dataset from Sections 1 and 3. Each student took around 100 photos of one object, with different backgrounds, lighting, distances and angles, and at least one cluttered scene. Each student also added about 5 background-only photos (the same scenes without the object).

- 73 classes, each labeled with its Object ID from the class Google Sheet (for example OBJ047 is a backpack)
- Images are 224x224, JPEG, RGB
- The label of each image comes from the folder it is in

Based on what the TA told us, we used the Object ID as the class label and left the background-only images out of the classification task (we did not make a separate background class).


## 2. Preprocessing

Preprocessing was done in a Google Colab notebook (`notebooks/Preprocessing.ipynb`). We followed the order the TAs recommended:

raw dataset, validation, background removal, duplicate check, resizing, RGB conversion, train/val/test split

| Step | What we did | Result |
|---|---|---|
| Load | Copied the final shared dataset from Google Drive | 7,345 files |
| Validation | Tried to open every file to make sure it was a valid image, skipped system files | 0 corrupted, 1 non-image file skipped |
| Labels | Took the Object ID from the folder name (`OBJ###`) | 73 classes, no unlabeled images |
| Background images | Removed files with `bg` or `background` in the filename | 379 removed |
| Duplicates | Computed an MD5 hash for every file and removed exact copies, before splitting so the same image can't end up in both train and test | 16 removed, none across different classes |
| Resizing | Images that weren't 224x224 were scaled down and padded to keep their shape (not stretched). EXIF rotation was fixed first | 2 images resized |
| Color | Converted to RGB where needed | All images were already RGB |
| Split | Stratified split by class, 70/15/15, seed 42 | 4,863 train, 1,042 val, 1,043 test |

After preprocessing we had **6,948 images in 73 classes**. Every class shows up in train, val and test, and no image is in more than one split.

Most classes had 5 background images. A few had more (OBJ005: 9, OBJ009: 11, OBJ014: 7, OBJ017: 7, OBJ025: 6, OBJ049: 6, OBJ053: 10, OBJ063: 6, OBJ069: 7). OBJ004 and OBJ065 had none, which we checked by looking through the folders.

The notebook outputs a `processed_224` folder, which goes in `data/` at the root of the repo:

```
data/processed_224/
  train/OBJ001/ ... train/OBJ073/
  val/OBJ001/   ... val/OBJ073/
  test/OBJ001/  ... test/OBJ073/
  manifest.csv                 original path, label, size, hash and split for each image
  class_to_idx.json            Object ID to number mapping
  preprocessing_report.json    the counts from the table above
```


## 3. Folder structure

```
milestone1/
  README.md
  notebooks/
    Preprocessing.ipynb   preprocessing (Colab)
  src/
    data.py               data loading and augmentation
    models.py             the three models and fine-tuning helpers
    train.py              training
    tune.py               Optuna tuning
    evaluate.py           test set evaluation and comparison
    predict.py            predicts the Object ID for new images
  results/                training history, summaries, evaluation and tuning results
  logs/                   output of every run
```

All commands below are run from inside the `milestone1/` folder. The scripts look for the data in `../data/processed_224` by default.

Libraries: torch, torchvision, numpy, pandas, pillow, scikit-learn, matplotlib, tqdm, optuna, gdown (all in `requirements.txt` at the root of the repo)


## 4. Data loading and augmentation (`data.py`)

We load images with PyTorch's `ImageFolder`, which takes the label from the folder name. We also check that PyTorch's label mapping matches `class_to_idx.json` so the Object IDs line up correctly.

All images are normalized with the ImageNet mean and standard deviation since the pretrained models expect that. We used the same normalization for the custom CNN to keep things consistent.

Augmentation is only applied to the training images, and it happens on the fly while training, so every epoch sees slightly different versions of each image. We don't save any new images, so the dataset is still 6,948 images.

| Augmentation | Setting |
|---|---|
| Random resized crop | scale 0.8 to 1.0, output 224 |
| Horizontal flip | 50% chance |
| Rotation | up to 15 degrees |
| Color jitter | brightness, contrast and saturation 0.2 |

We didn't change hue, because color is what separates some of the classes (like orange and apple). Validation and test images are only normalized.


## 5. Models (`models.py`)

| Model | Approach | Parameters | Model file size |
|---|---|---|---|
| Custom CNN | Trained from scratch | 407,657 | 1.6 MB |
| MobileNetV3-Large | Transfer learning (ImageNet) | 4,295,545 | 17.4 MB |
| ResNet-50 | Transfer learning (ImageNet) | 23,657,609 | 95.0 MB |

### Why we picked these

We wanted to compare a model trained from scratch against pretrained ones, and also see how much accuracy we lose by using a smaller model. These three go from very small (0.4M parameters) to fairly large (23.7M).

- **Custom CNN:** our baseline. It only learns from our own ~4,900 training images, so it shows what the data alone can do and how much pretraining adds.
- **MobileNetV3-Large:** a small pretrained model built for phones and low-power devices. We picked it to see how close a lightweight model can get.
- **ResNet-50:** one of the most common pretrained models for image classification. We used it as the stronger, heavier option.

### Architectures

**Custom CNN:** 4 convolution blocks. Each block has a 3x3 convolution, batch normalization, ReLU and 2x2 max pooling. The blocks have 32, 64, 128 and 256 filters, and each one halves the image size (224, 112, 56, 28, 14). After that there is global average pooling, dropout of 0.3, and a final layer with 73 outputs. We kept it small on purpose since there are only about 67 training images per class and a bigger network would just memorize them.

**MobileNetV3-Large:** made of inverted residual blocks that use depthwise separable convolutions. Instead of one normal convolution, it does a 3x3 convolution on each channel separately and then a 1x1 convolution to combine the channels, which needs a lot fewer calculations. Some blocks also have squeeze-and-excitation modules that give more weight to the more useful channels, and it uses the hard-swish activation. The layout of the network was found with an automated architecture search for mobile devices. At the end there is a classifier (960 to 1,280 features, hard-swish, dropout 0.2) followed by our 73-class output layer.

**ResNet-50:** 50 layers. It starts with a 7x7 convolution and max pooling, then has 4 stages of bottleneck residual blocks (3, 4, 6 and 3 blocks). Each block has a 1x1, 3x3 and 1x1 convolution, plus a skip connection that adds the block's input to its output. The skip connections help gradients flow back through the network, which is what makes it possible to train a network this deep. Global average pooling gives 2,048 features, which go into our 73-class output layer.

For both pretrained models we replaced the original 1,000-class ImageNet output layer with a new 73-class layer. That's why the parameter counts are a bit lower than the usual numbers (around 5.4M and 25.6M). Their earlier layers were already trained on 1.2 million ImageNet images, so they already pick up general things like edges, textures and shapes. The models mostly just need to learn how to use those features for our 73 objects.


## 6. Training (`train.py`)

### Custom CNN
Trained all layers from the start, for 40 epochs with a learning rate of 1e-3. We also did an 80 epoch run (see section 9).

### Pretrained models
We fine-tuned them in two stages:

1. **Head training (3 epochs):** the pretrained layers are frozen and only the new output layer is trained, with a learning rate of 1e-3. This way the new layer learns something useful before we start changing the pretrained weights.
2. **Fine-tuning (17 epochs):** we unfreeze the top part of the network (the last 2 of ResNet-50's 4 stages, and the last 4 of MobileNet's 17 blocks) and train it together with the output layer at a lower learning rate of 1e-4, so the pretrained weights only change a little.

Batch norm layers in the frozen part are kept in eval mode so their statistics don't change during training.

### Settings used for all models

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate schedule | Cosine decay within each stage |
| Loss | Cross-entropy |
| Batch size | 32 |
| Seed | 42 |
| Model selection | Kept the weights from the epoch with the best validation accuracy |
| Hardware | NVIDIA RTX 5090 (32 GB) rented on Vast.ai, PyTorch 2.14, CUDA 13.0 |

For every run we save the best weights, a CSV with loss and accuracy for each epoch, and a JSON summary with the settings and results. A checkpoint is also saved after every epoch so a run can be resumed if it gets interrupted. The test set is not used at all during training or tuning.

```bash
python src/train.py --model custom
python src/train.py --model mobilenet
python src/train.py --model resnet50
python src/train.py --model resnet50 --no_augment
python src/train.py --model custom --epochs 80 --name custom_80
```


## 7. Hyperparameter tuning with Optuna (`tune.py`)

We used Optuna to tune the fine-tuning stage for both pretrained models. Each trial trains a model with one set of hyperparameters. Optuna's TPE sampler looks at how earlier trials did and uses that to pick the next set of values to try. A median pruner stops trials early if they are doing worse than the earlier trials, which saves time.

| Setting | Value |
|---|---|
| Goal | Highest validation accuracy |
| Trials | 30 per model |
| Epochs per trial | 10 (fine-tuning stage only) |
| Starting point | Every trial starts from the same checkpoint (saved after head training), so the trials are fair to compare |
| Sampler | TPE, seed 42 |
| Pruner | Median pruner. A trial is stopped if its validation accuracy is below the median of earlier trials at the same epoch. Starts after the first 3 trials, from epoch 3 |

Search space:

| Hyperparameter | Range |
|---|---|
| Fine-tuning learning rate | 1e-5 to 1e-3 (log scale) |
| Weight decay | 1e-5 to 1e-2 (log scale) |
| Number of unfrozen stages | ResNet-50: 1 to 4, MobileNet: 2 to 10 |
| Batch size | 32 or 64 |

Once the search was done, we retrained each model with the best settings for the full 17 fine-tuning epochs, starting from the same head checkpoint. This gives it the same training budget as the baseline (3 + 17 epochs), so the comparison is fair.

Best settings:

| | ResNet-50 | MobileNetV3 |
|---|---|---|
| Best trial | #16 (val 99.0%) | #5 (val 98.2%) |
| Learning rate | 2.3e-4 (default was 1e-4) | 3.7e-4 (default was 1e-4) |
| Weight decay | 1.1e-4 (default was 1e-4) | 4.0e-5 (default was 1e-4) |
| Unfrozen stages | 4 of 4 (default was 2) | 6 of 17 (default was 4) |
| Batch size | 32 | 32 |
| Trials finished / pruned | 13 / 17 | 7 / 23 |

Both models did better with a higher learning rate and more unfrozen layers than our defaults, so letting more of the pretrained network adjust to our objects helped. More than half the trials were pruned, which kept tuning fast.

The best trial scores (like 99.0% for ResNet-50) are a bit higher than what you'd normally get, since they are the best out of 30 tries. The full retrain gave 98.7% on validation, which is the more realistic number.

```bash
python src/tune.py --model resnet50 --trials 30 --trial_epochs 10
python src/tune.py --model mobilenet --trials 30 --trial_epochs 10
```


## 8. Evaluation (`evaluate.py`)

Each model's best checkpoint is evaluated once on the test set (1,043 images). For each model we get:

- Test accuracy and macro F1 (F1 averaged over all 73 classes, each class counted equally)
- Precision, recall and F1 for every class
- Confusion matrix (CSV and image)
- The 10 most confused class pairs
- Number of parameters, model file size, and time per image

Time per image was measured by running 200 test images one at a time, after a short warm up. All models were evaluated together on the same GPU. The combined results are in `results/eval/comparison.csv`.

```bash
python src/evaluate.py --names custom custom_80 mobilenet mobilenet_s1 mobilenet_s2 mobilenet_tuned resnet50 resnet50_s1 resnet50_s2 resnet50_noaug resnet50_tuned
```


## 9. Experiments

1. **Model comparison:** custom CNN vs MobileNetV3 vs ResNet-50, all with the same data, split, augmentation and evaluation.
2. **Augmentation:** ResNet-50 trained with and without augmentation.
3. **Longer training:** custom CNN trained for 80 epochs instead of 40, to see if it was held back by training time or by its size.
4. **Hyperparameter tuning:** Optuna for MobileNetV3 and ResNet-50 (section 7).
5. **Different seeds:** MobileNetV3 and ResNet-50 trained again with seeds 1 and 2, to see how much results change between runs. The split stays the same, only the random weight initialization, shuffling and augmentation change.


## 10. Results

### All runs

| Run | Best val acc | Best epoch | Test acc | Test macro F1 | Training time (min) |
|---|---|---|---|---|---|
| Custom CNN (40 epochs) | 69.4% | 36 | 69.2% | 0.686 | 1.8 |
| Custom CNN (80 epochs) | 78.6% | 72 | 77.4% | 0.773 | 3.5 |
| MobileNetV3 | 97.4% | 18 | 97.1% | 0.971 | 1.1 |
| MobileNetV3, seed 1 | 97.5% | 15 | 97.4% | 0.974 | 1.1 |
| MobileNetV3, seed 2 | 97.4% | 19 | 97.0% | 0.971 | 1.0 |
| MobileNetV3, tuned | 98.0% | 11 | 97.7% | 0.977 | 1.0 |
| ResNet-50, no augmentation | 97.8% | 13 | 97.3% | 0.973 | 1.5 |
| ResNet-50 | 98.6% | 13 | 98.3% | 0.983 | 1.5 |
| ResNet-50, seed 1 | 98.5% | 15 | 97.6% | 0.976 | 1.5 |
| ResNet-50, seed 2 | 98.6% | 15 | 98.3% | 0.983 | 1.5 |
| ResNet-50, tuned | 98.7% | 12 | 98.2% | 0.982 | 1.8 |

Time per image was about 0.3 ms for the custom CNN and about 2.5 ms for both MobileNetV3 and ResNet-50 on the RTX 5090. MobileNet isn't faster here because on a GPU this powerful, running one image at a time is mostly overhead. On a CPU or a phone, MobileNet would be a lot faster than ResNet-50.

### Seed runs (default settings, test accuracy)

| Model | Seed 42 | Seed 1 | Seed 2 | Mean ± std |
|---|---|---|---|---|
| MobileNetV3 | 97.12% | 97.41% | 97.03% | 97.19% ± 0.20 |
| ResNet-50 | 98.27% | 97.60% | 98.27% | 98.05% ± 0.39 |

### What we found

1. **Pretraining made the biggest difference.** The best custom CNN got 77.4% on test while both pretrained models got around 97 to 98%, so roughly a 20 point gap.
2. **ResNet-50 is more accurate than MobileNetV3.** ResNet-50's worst seed (97.6%) is still higher than MobileNet's best seed (97.4%), so the ~0.9 point difference isn't just random variation.
3. **Augmentation helped.** ResNet-50 got 98.3% on test with augmentation and 97.3% without. Without augmentation it hit 100% training accuracy by epoch 6 and then stopped improving on validation.
4. **The custom CNN needed more training time.** Going from 40 to 80 epochs improved test accuracy from 69.2% to 77.4%. Its training and validation accuracy stayed close together the whole time, so it was underfitting, not overfitting. (Training accuracy was a bit lower than validation because the training images are augmented and dropout is on during training.)
5. **Tuning helped MobileNet but not ResNet-50.** Tuned MobileNet (97.7%) was better than all three untuned MobileNet runs (97.0 to 97.4%). Tuned ResNet-50 (98.2%) was inside the range of the untuned runs (97.6 to 98.3%), so our default settings were already about as good as it gets for ResNet-50.
6. **Validation and test results were close.** Every model's test accuracy was within about 1 point of its validation accuracy, so picking the best epoch based on validation didn't make the results look much better than they are.
7. **The pretrained models overfit a little.** They reach close to 100% training accuracy while validation is 97 to 98.7%, but the validation loss stays flat instead of going up.


## 11. Error analysis

Tuned ResNet-50 got 19 out of 1,043 test images wrong. The most common mistakes:

| Actual object | Predicted as | Count |
|---|---|---|
| Card wallet (OBJ061) | Comb (OBJ059) | 3 |
| Mobile phone (OBJ013) | Headphones (OBJ062) | 2 |
| Comb (OBJ059) | Card wallet (OBJ061) | 1 |
| Neck pillow (OBJ068) | Backpack (OBJ047) | 1 |
| Mobile phone (OBJ013) | Footstool (OBJ058) | 1 |

- Card wallet and comb were the only pair mixed up both ways. Both are small, flat and thin, so at 224x224 they can look similar, especially in medium and far shots.
- Neck pillow and backpack are both soft, rounded and made of fabric.
- Mobile phone was the hardest class. It's small, dark and plain, without much that makes it stand out.

The dataset also has some groups of similar objects: Sunglasses (OBJ014) and Wayfarer Sunglasses (OBJ045), Watch (OBJ025), iWatch (OBJ046) and Analog watch (OBJ065), Card wallet (OBJ061) and Men's Wallet (OBJ067), Power Bank (OBJ057) and Wireless powerbank (OBJ048), and Mug (OBJ011) and Water Tumbler (OBJ015). None of these showed up in the top mistakes.


## 12. Final model

We picked models based on validation accuracy and report their test accuracy.

**Final model: ResNet-50 (tuned).** It had the highest validation accuracy (98.7%) and got 98.2% on test, which is basically the same as the untuned ResNet-50 (98.3%).

**Smaller alternative: MobileNetV3 (tuned).** It got 97.7% on test, only about 0.5 points lower than ResNet-50, with 5.5 times fewer parameters (4.3M vs 23.7M) and a much smaller file (17 MB vs 95 MB). It would be the better choice if the model has to run on a phone or limited hardware.


## 13. Limitations

- **Backgrounds and environments.** Each object was photographed by one student in their own home. So the model might be using things like the room, furniture or lighting along with the object itself. This could also be part of why similar looking objects were easy to tell apart, since they were photographed in different places. To check this properly, we would need to test on photos of the same objects taken somewhere else.
- **One physical object per class.** Each class is one specific item, so the model learns that item and not the general category (one particular backpack, not backpacks in general).
- **Small test set per class.** There are only about 14 test images per class, so one wrong image changes that class's accuracy by about 7%.
- **Variation between runs.** The same model with different seeds varied by up to about 0.7 points, so differences smaller than that shouldn't be taken too seriously.
- **Speed was measured on a very fast GPU,** which hides the speed difference between the models that you would see on a CPU or phone.


## 14. Running predictions (`predict.py`)

```bash
python src/predict.py path/to/image.jpg
python src/predict.py path/to/folder/ --name mobilenet_tuned
```

The script processes the image the same way as in preprocessing (fixes rotation, converts to RGB, pads to 224x224) and prints the top 3 predicted Object IDs with their confidence, for example:

```
image.jpg: OBJ047 (94.2%), OBJ012 (3.1%), OBJ030 (0.8%)
```

By default it uses tuned ResNet-50. You can pick another model with `--name`. The trained weights (`.pt` files) are not in the repo because of their size, so you need to train the model first or get the weights from us.
