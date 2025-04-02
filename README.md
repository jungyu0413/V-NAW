# V-NAW: Video-based Noise-aware Adaptive Weighting for Facial Expression Recognition (CVPR Workshop 2025 ABAW)

This repository contains the official implementation of the paper:  
**["V-NAW: Video-based Noise-aware Adaptive Weighting for Facial Expression Recognition"](https://arxiv.org/abs/2503.15970)**

---

## Abstract

Facial Expression Recognition (FER) plays a crucial role in human affective analysis and has been widely applied in computer vision tasks such as human-computer interaction and psychological assessment. The 8th Affective Behavior Analysis in-the-Wild (ABAW) Challenge aims to assess human emotions using the video-based Aff-Wild2 dataset. This challenge includes various tasks, including the video-based EXPR recognition track, which is our primary focus. In this paper, we demonstrate that addressing label ambiguity and class imbalance, which are known to cause performance degradation, can lead to meaningful performance improvements. Specifically, we propose Video-based Noise-aware Adaptive Weighting (V-NAW), which adaptively assigns importance to each frame in a clip to address label ambiguity and effectively capture temporal variations in facial expressions. Furthermore, we introduce a simple and effective augmentation strategy to reduce redundancy between consecutive frames, which is a primary cause of overfitting. Through extensive experiments, we validate the effectiveness of our approach, demonstrating significant improvements in video-based FER performance.

---

## Challenge Result

This model ranked **5th place** in the **Facial Expression Recognition Track** of the  
[CVPR 2025 8th ABAW Challenge](https://affective-behavior-analysis-in-the-wild.github.io/8th/).

---

## Main Architecture

<img src="https://github.com/user-attachments/assets/f697ce89-8c2b-4b3e-81d5-047770b5a677" width="700" alt="Architecture" />

---

## Performance Results

<img src="https://github.com/user-attachments/assets/d56bf7d3-1734-42d6-bb31-95badf6881f5" width="700" alt="Main Results" />

---

## Ablation Study

<img src="https://github.com/user-attachments/assets/70f43306-28ff-4be4-8520-40803d833eff" width="700" alt="Ablation Study" />

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/jungyu0413/V-NAW-Video-FER.git
cd V-NAW-Video-FER
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

- Download the [Aff-Wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/) dataset.
- Organize the data as described in `DATASET.md`.

### 4. Train

We provide two training scripts depending on your environment:

#### Option A: Multi-GPU Training (Distributed Data Parallel)

To train using multiple GPUs via PyTorch DDP (no torchrun required):

```bash
python DDP_train_exp.py --config configs/vnaw_config.yaml
```

- Automatically uses all available GPUs on a single node
- No need for `torchrun` or additional launcher
- Efficient for large-scale training

> Ensure that your environment supports NCCL backend for multi-GPU communication

#### Option B: Single-GPU Training

To run training on a single GPU:

```bash
python train_exp.py --config configs/vnaw_config.yaml
```

- Easy to use for debugging or small-scale experiments

### 5. Inference

```bash
python inference.py --video path_to_video.mp4
```

Runs expression recognition inference on the input video using the trained model.

---

## Project Structure

```
.
├── configs/              # Configuration files
├── data/                 # Dataset processing and loading
├── models/               # Model architectures
├── modules/              # Loss functions, metrics, and utilities
├── train_exp.py          # Single-GPU training script
├── DDP_train_exp.py      # Multi-GPU training script (DDP)
├── inference.py          # Inference script
├── requirements.txt
└── README.md
```

---

## Contact

- Email: [jungyu0413@gmail.com](mailto:jungyu0413@gmail.com)
- GitHub: [github.com/jungyu0413](https://github.com/jungyu0413)
- Google Scholar: [scholar.google.com/citations?user=2pnIJggAAAAJ&hl=ko](https://scholar.google.com/citations?user=2pnIJggAAAAJ&hl=ko)
- LinkedIn: [linkedin.com/in/jungyu-lee-0315sb](https://www.linkedin.com/in/jungyu-lee-0315sb/)


