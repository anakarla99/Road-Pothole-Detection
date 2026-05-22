# 🛣️ Road Pothole Detection — Satellite Image Segmentation

Semantic segmentation of road conditions in Havana, Cuba using satellite imagery and deep learning. The model classifies each pixel of a satellite image as background, good street, or deteriorated street.

**Authors:** Ana Karla Caballero (C-411) · Alejandro Camacho (C-412) — University of Havana
**Framework:** PyTorch · torchvision
**Model:** DeepLabv3 (ResNet-101 backbone)

---

## 🎯 Problem

Detecting road deterioration at scale using satellite imagery, to support urban planning and infrastructure management in Cuba. Manual annotation is labour-intensive and error-prone — this project automates pixel-level classification.

**Classes** (`class_names.txt`):

| Class | Label | Colour (overlay) |
|-------|-------|-----------------|
| 0 | Background | Black |
| 1 | Good street | Red |
| 2 | Bad street | Green |

---

## 📁 Structure

```
├── main.py          ← Training script (DeepLabv3, CrossEntropyLoss, Adam)
├── test.py          ← Inference + confusion matrix generation
├── rotate.py        ← Data augmentation: 90/180/270° rotations via ImageMagick
├── class_names.txt  ← Class labels
├── Reporte.tex      ← Full project report (LaTeX)
├── dataset/
│   ├── images/      ← Satellite images (gitignored)
│   └── masks/       ← Labelme-annotated masks (gitignored)
├── models/          ← Saved .pth checkpoints (gitignored)
└── predicted/       ← Inference output: prediction masks + overlays + confusion matrices (gitignored)
```

---

## 🧠 Model & Training

`main.py` fine-tunes the last convolutional layer of **DeepLabv3 ResNet-101** on the road dataset:

- Input resize: 315×315
- Batch size: 4
- Optimiser: Adam (lr = 1e-6), only `classifier[-1]` trained
- Loss: `CrossEntropyLoss` with class weights `[0.00001, 2.0, 1.5]` (heavy background suppression)
- Epochs: 10 (best results from extended runs of 100 epochs)
- Device: CUDA if available, else CPU

```bash
python main.py
# Saves: model_deeplabv3_resnet101_ts315,315_bs4_ep10_w.pth
```

---

## 🔍 Inference

`test.py` runs all `.pth` models in `./models/` against images in `./testing/images/` and saves per-model:

- `<image> prediction.png` — colour-coded segmentation mask
- `<image> overlay.png` — prediction overlaid on original (60% opacity)
- `<image>_original_mask.png` — ground truth mask copy
- `<image>_confusion_matrix.png` — heatmap (sklearn + seaborn)

```bash
python test.py
```

---

## 🗂 Dataset

37 satellite images collected from Kaggle, Mapillary, and Google Maps, covering different areas of Havana. Expanded to **108 images** via augmentation (rotations with `rotate.py`, flips). Split: 88 training / 20 validation.

Masks annotated manually with [Labelme](https://github.com/labelmeai/labelme).

> Dataset, models, and prediction outputs are gitignored. Add your own images to `dataset/images/` and masks to `dataset/masks/` to train.

---

## 🚀 Setup

```bash
pip install torch torchvision pillow numpy matplotlib scikit-learn seaborn
# For rotate.py: sudo apt install imagemagick

python main.py   # train
python test.py   # evaluate
```
