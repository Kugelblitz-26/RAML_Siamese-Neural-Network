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
   cd siamese-networks-comparison
   ```

2. **Install dependencies**

   We recommend using a Python virtual environment.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare datasets**

   - download and preparing LFW, Georgia Tech Faces, and Amazon Book Reviews.
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



---

