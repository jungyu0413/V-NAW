# V-NAW: Video-based Noise-aware Adaptive Weighting for Facial Expression Recognition (CVPRW 2025)

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.15970)
[![Challenge](https://img.shields.io/badge/CVPR_2025-ABAW_Challenge-blue)](https://affective-behavior-analysis-in-the-wild.github.io/8th/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper **"V-NAW: Video-based Noise-aware Adaptive Weighting for Facial Expression Recognition"**, presented at the **CVPR 2025 8th ABAW Challenge Workshop**.

🏆 **Challenge Result:** Our model ranked **5th place** in the Facial Expression Recognition (EXPR) Track of the [8th ABAW Challenge](https://affective-behavior-analysis-in-the-wild.github.io/8th/).

> **Abstract:** Facial Expression Recognition (FER) plays a crucial role in human affective analysis and has been widely applied in computer vision tasks such as human-computer interaction and psychological assessment. The 8th Affective Behavior Analysis in-the-Wild (ABAW) Challenge aims to assess human emotions using the video-based Aff-Wild2 dataset. In this paper, we demonstrate that addressing label ambiguity and class imbalance, which are known to cause performance degradation, can lead to meaningful performance improvements. Specifically, we propose Video-based Noise-aware Adaptive Weighting (V-NAW), which adaptively assigns importance to each frame in a clip to address label ambiguity and effectively capture temporal variations in facial expressions. Furthermore, we introduce a simple and effective augmentation strategy to reduce redundancy between consecutive frames, which is a primary cause of overfitting.

---

## 📂 Repository Structure

```text
V-NAW/
│
├── configs/
│   └── vnaw_config.yaml   # Hyperparameters and training configurations
├── data/                  # Directory for Aff-Wild2 dataset
├── src/
│   ├── dataset.py         # Dataloaders for video datasets
│   ├── loss.py            # Implementation of V-NAW
│   ├── model.py           # Network architecture (Temporal Encoder)
│   ├── resnet.py          # ResNet base functions
│   ├── resnet18.py        # ResNet-18 backbone
│   ├── test.py            # Evaluation functions
│   ├── train.py           # Training loops and optimization
│   └── utils.py           # Evaluation metrics and helper functions
│
├── DDP_train_exp.py       # Multi-GPU training script (Distributed Data Parallel)
├── train_exp.py           # Single-GPU training script
├── inference.py           # Script for running inference on videos
└── README.md
```

---

## ⚙️ Environment Setup

We recommend using Anaconda to manage the environment.

```bash
# Clone the repository
git clone https://github.com/jungyu0413/V-NAW-Video-FER.git
cd V-NAW-Video-FER

# Create and activate a conda environment
conda create -n vnaw python=3.8
conda activate vnaw

# Install dependencies
pip install -r requirements.txt
```

---

## 💾 Data Preparation

1. Download the **[Aff-Wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)** dataset.
2. Organize the data as described in the dataset guidelines (or refer to `DATASET.md` if available).
3. Ensure the dataset path is correctly configured in your `configs/vnaw_config.yaml`.

---

## 🚀 How to Run

### Training

We provide two training scripts depending on your hardware environment:

**Option A: Multi-GPU Training (Distributed Data Parallel)**
To train using multiple GPUs via PyTorch DDP. This automatically uses all available GPUs on a single node without requiring `torchrun` or additional launchers, making it highly efficient for large-scale training. *(Ensure that your environment supports the NCCL backend for multi-GPU communication)*:

```bash
python DDP_train_exp.py --config configs/vnaw_config.yaml
```

**Option B: Single-GPU Training**
Easy to use for debugging or small-scale experiments on a single GPU:

```bash
python train_exp.py --config configs/vnaw_config.yaml
```

### Inference

Run expression recognition inference on a custom input video using the trained model:

```bash
python inference.py --video path_to_video.mp4
```

---

## 📊 Framework & Results

### Main Architecture
<img src="[https://github.com/user-attachments/assets/f697ce89-8c2b-4b3e-81d5-047770b5a677](https://github.com/user-attachments/assets/f697ce89-8c2b-4b3e-81d5-047770b5a677)" width="700" alt="Architecture" />

### Performance Results
<img src="[https://github.com/user-attachments/assets/d56bf7d3-1734-42d6-bb31-95badf6881f5](https://github.com/user-attachments/assets/d56bf7d3-1734-42d6-bb31-95badf6881f5)" width="700" alt="Main Results" />

### Ablation Study
<img src="[https://github.com/user-attachments/assets/70f43306-28ff-4be4-8520-40803d833eff](https://github.com/user-attachments/assets/70f43306-28ff-4be4-8520-40803d833eff)" width="700" alt="Ablation Study" />

---

## 📜 Citation

If you find this repository or our paper useful for your research, please consider citing:

```bibtex
@inproceedings{lee2025vnaw,
  title={V-NAW: Video-based Noise-aware Adaptive Weighting for Facial Expression Recognition},
  author={Lee, JunGyu and Lee, Kunyoung and Park, Haesol and Kim, Ig-Jae and Nam, Gi Pyo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2025}
}
```

## 🙏 Acknowledgements
This research was supported by the KIST Institutional Program (Project No. 2E33612).




