# RAML_Siamese-Neural-Network

# Siamese Neural Networks: Comparative Study on Architecture and Loss Functions

## Overview

This repository contains the implementation and experiments for a research project studying the impact of different backbone architectures and loss functions on Siamese Neural Networks (SNNs) across two distinct domains:

1. **Face Verification**  
   - Datasets: Labeled Faces in the Wild (LFW) and Georgia Tech Faces dataset  
   - Architectures: MLP, SimpleCNN, ResNet18, ResNet50, VGG16, MobileNetV3  
   - Loss Functions: Contrastive Loss, Triplet Loss, Cosine Embedding Loss, InfoNCE Loss

2. **Book Recommendation**  
   - Dataset: Amazon Book Reviews subset (30,000 user-book interactions)  
   - Architectures: MLP (metadata only), CNN (1D text), BERT (DistilBERT), Hybrid (combining CNN and MLP)  
   - Loss Functions: Contrastive Loss, Triplet Loss, Circle Loss

The goal is to understand how architecture and loss function choices interact, vary by data modality and domain, and affect model performance, computational efficiency, and practical deployment considerations.

## Repository Structure

```
├── notebooks/                # Jupyter notebooks used for exploratory data analysis and experiments
│
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

## Installation and Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Kugelblitz-26/RAML_Siamese-Neural-Network.git
   cd RAML_Siamese-Neural-Network
   ```

2. **Install dependencies**

   We recommend using a Python virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare datasets**

   - Download and prepare LFW, Georgia Tech Faces, and Amazon Book Reviews.
   - Ensure data directories are correctly set in training scripts or config files.

## Usage

- **Train Face Verification Models**

  ```bash
  python final.ipynb
  ```

- **Train Book Recommendation Models**

  ```bash
  python book/ex.ipynb
  ```

- **Evaluate Models and Visualize Embeddings**

  Use provided utilities in `utils/` or notebooks for t-SNE visualization and performance metrics.

- **Advanced Training**

  Implementations include easy extensions for hard negative mining, dynamic pair generation, and model checkpointing.

## Face Verification

This section evaluates the performance of Siamese Neural Networks (SNNs) on face recognition using the Labeled Faces in the Wild (LFW) and Georgia Tech Faces (GTech) datasets. The goal is to understand how different loss functions and architectures impact face verification performance.

### 📁 `case1/Experiment1`: Architecture and Loss Function Comparison

This experiment investigates how different backbone architectures and loss functions affect the performance of SNNs for face recognition. The folder contains:

```
case1/Experiment1/
├── analysis.py         # Code to evaluate performance metrics (e.g., accuracy, ROC-AUC, etc.)
├── architectures.py    # Definitions of backbone networks like ResNet18, VGG16, SimpleCNN, etc.
├── Gtech/              # Contains the Georgia Tech Face dataset and preprocessing code
├── LFW/                # Contains the LFW dataset and preprocessing code
├── loss.py             # Implementation of loss functions (Contrastive, Triplet, etc.)
├── sampling.py         # Triplet and pair sampling logic for Siamese training
├── train.py            # 🔰 Entry point: Training script with model instantiation and training loop
└── visualise.py        # Code to visualize learned embeddings using t-SNE, PCA, etc.
```

### ▶️ Running the Code

The main script to initiate training is:

```bash
python train.py
```

This file imports components from other modules:
- **Architectures** from `architectures.py`
- **Loss functions** from `loss.py`
- **Dataset loading and preprocessing** from the `LFW/` and `Gtech/` directories
- **Sampling strategies** from `sampling.py`
- **Visualization tools** from `visualise.py`

You **must configure the dataset paths** within the respective dataset folders (`LFW/` and `Gtech/`) to ensure correct data loading before training.

### 🔧 Dependencies

Ensure all required libraries are included in `requirements.txt`. Common dependencies include:

```txt
torch
torchvision
numpy
matplotlib
scikit-learn
pandas
tqdm
```

(Please verify your local environment and update this file accordingly.)

## Key Contributions and Extensions

- Integration and extension of third-party codebases:
  - Base Siamese MLP implementation adapted from [MLP-Siamese-Network GitHub repository](https://github.com/Setarehkhaleghian/MLP-Siamese-Network) (extended for multimodal data and supported extra architectures).
  - Face recognition Siamese pipeline inspired by Kaggle notebook [Face Recognition using Siamese Networks](https://www.kaggle.com/code/vijaykrishnan1905/face-recognition-using-siamese-networks).
- Novel hybrid architecture combining CNN-based text encoding with MLP-based metadata for book recommendation.
- Comprehensive comparative study evaluating multiple loss functions across different domains and architectures.
- Detailed analysis of computational trade-offs to inform deployment decisions.

See [REPORT.md](./notebooks/REPORT.md) for the full research write-up and results.

## References

- Schroff et al., 2015. FaceNet: A unified embedding for face recognition and clustering.  
- Chopra et al., 2005. Learning a similarity metric discriminatively, with application to face verification.  
- Sanh et al., 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.  
- He et al., 2017. Neural Collaborative Filtering.  
- Sun et al., 2020. Circle loss: A unified perspective of pair similarity optimization.  
- Official PyTorch models and torchvision libraries.

Please see the full report for a complete bibliography.
