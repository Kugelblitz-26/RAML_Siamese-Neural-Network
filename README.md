# RAML_Siamese-Neural-Network
# Impact of Sample-Selection Strategies on Siamese Neural Network Performance

Abstract
This repository contains the implementation and experiments for the research question: “What is the impact of the training-sample selection strategy on the performance of a Siamese Neural Network?”** We compare random sampling, hard-negative mining, semi-hard negative mining, and distance-weighted sampling across multiple datasets.

---

## 📁 Repository Structure

```
your-repo/
├─ README.md
├─ notebooks/
│  ├─ 01_data_exploration.ipynb
│  ├─ 02_random_sampling_experiment.ipynb
│  ├─ 03_hard_mining_experiment.ipynb
│  ├─ 04_semi_hard_vs_distance_weighted.ipynb
│  └─ 05_results_analysis.ipynb
├─ src/
│  ├─ model.py            # Siamese network architectures
│  ├─ sampling.py         # Sampling strategy implementations
│  └─ train.py            # Training loop and evaluation
└─ results/
   ├─ figures/            # Plots of loss curves and retrieval metrics
   └─ tables/             # Summary tables (Recall, accuracy)
```

---

## 🛠️ Setup Instructions

1. Clone the repository
2. **Create and activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\\Scripts\\activate   # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Download datasets**

   * Omniglot, MNIST, Fashion-MNIST via torchvision
   * LFW: `bash scripts/download_lfw.sh`

---

## 🚀 Usage

### 1. Jupyter Notebooks

Each notebook in `notebooks/` is self-contained:

```bash
jupyter lab notebooks/01_data_exploration.ipynb
```

Follow the markdown instructions and run cells sequentially to reproduce experiments.

### 2. Command-line Scripts

To run training with a specific sampling strategy:

```bash
python src/train.py --strategy semi_hard --dataset omniglot --epochs 50
```

Output models and logs will be saved under `results/`.

---

## 📊 Results Snapshot

![Recall@1 Comparison](results/figures/recall1_comparison.png)

| Strategy             | Recall\@1 (Omniglot) | Recall\@1 (LFW) |
| -------------------- | -------------------- | --------------- |
| Random Sampling      | 72.4%                | 68.1%           |
| Hard-Negative Mining | 85.2%                | 78.5%           |
| Semi-Hard Mining     | 88.3%                | 80.7%           |
| Distance-Weighted    | 90.1%                | 82.4%           |

---

## 📈 GitHub Activity


## 📚 References

1. Schroff, Kalenichenko, and Philbin. “FaceNet: A Unified Embedding for Face Recognition and Clustering.” CVPR 2015.
2. Wu et al. “Sampling Matters in Deep Embedding Learning.” ICCV 2017.
3. Hermans, Beyer, and Leibe. “In Defense of the Triplet Loss for Person Re-Identification.” arXiv 2017.
4. Sohn. “Improved Deep Metric Learning with Multi-class N-pair Loss Objective.” NIPS 2016.

---

