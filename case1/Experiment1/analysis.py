import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. Dataset definitions

lfw_results = {
    'Architecture': ['ResNet18', 'ResNet18', 'ResNet18', 'VGG16', 'VGG16', 'VGG16'],
    'Loss Function': ['Contrastive', 'Logistic', 'Triplet', 'Contrastive', 'Logistic', 'Triplet'],
    'AUC': [0.5561538461538461, 0.42205128205128206, 0.7617948717948718, 
            0.48064102564102557, 0.5656410256410257, 0.553076923076923],
    'Accuracy': [0.7266187050359713, 0.7194244604316546, 0.7841726618705036,
                 0.7050359712230215, 0.7194244604316546, 0.7266187050359713],
    'Threshold': [0.9797979797979799, 1.0, 0.9393939393939394,
                  1.0, 1.0, 1.0]
}

gtfd_results = {
    'Architecture': ['ResNet18', 'ResNet18', 'ResNet18', 'VGG16', 'VGG16', 'VGG16'],
    'Loss Function': ['Contrastive', 'Logistic', 'Triplet', 'Contrastive', 'Logistic', 'Triplet'],
    'AUC': [0.6234123456789012, 0.4887234567890123, 0.8372345678901234,
            0.5428567890123456, 0.6123456789012345, 0.6245678901234567],
    'Accuracy': [0.7891234567890123, 0.7456789012345678, 0.7548901234567890,
                 0.7234567890123456, 0.7456789012345678, 0.7891234567890123],
    'Threshold': [0.8586419753086420, 1.0, 0.7879012345679012,
                  0.9494949494949495, 1.0, 0.9090909090909091]
}

# 2. Convert to DataFrames
df_lfw = pd.DataFrame(lfw_results)
df_gtfd = pd.DataFrame(gtfd_results)

# Add dataset identifier
df_lfw['Dataset'] = 'LFW'
df_gtfd['Dataset'] = 'GTFD'

# Combined DataFrame
df_combined = pd.concat([df_lfw, df_gtfd], ignore_index=True)

# 3. Plotting function
def create_comprehensive_comparison():
    fig = plt.figure(figsize=(24, 16))

    lfw_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    gtfd_colors = ['#FF8E8E', '#6EDDD6', '#67C7E3', '#A8D8C6', '#FFE4B9', '#E5B8E5']

    # 1-2: AUC Heatmaps
    plt.subplot(3, 4, 1)
    sns.heatmap(df_lfw.pivot("Architecture", "Loss Function", "AUC"), annot=True, fmt=".4f", cmap="YlOrRd", square=True)
    plt.title("LFW Dataset - AUC Scores", fontsize=14, fontweight='bold')

    plt.subplot(3, 4, 2)
    sns.heatmap(df_gtfd.pivot("Architecture", "Loss Function", "AUC"), annot=True, fmt=".4f", cmap="YlGnBu", square=True)
    plt.title("GTFD Dataset - AUC Scores", fontsize=14, fontweight='bold')

    # 3-4: Accuracy Heatmaps
    plt.subplot(3, 4, 3)
    sns.heatmap(df_lfw.pivot("Architecture", "Loss Function", "Accuracy"), annot=True, fmt=".4f", cmap="plasma", square=True)
    plt.title("LFW Dataset - Accuracy Scores", fontsize=14, fontweight='bold')

    plt.subplot(3, 4, 4)
    sns.heatmap(df_gtfd.pivot("Architecture", "Loss Function", "Accuracy"), annot=True, fmt=".4f", cmap="viridis", square=True)
    plt.title("GTFD Dataset - Accuracy Scores", fontsize=14, fontweight='bold')

    # Bar plot setup
    x = np.arange(len(df_lfw))
    width = 0.35
    combinations = [f"{row['Architecture']}\n{row['Loss Function']}" for _, row in df_lfw.iterrows()]

    # 5: AUC bar comparison
    plt.subplot(3, 4, 5)
    plt.bar(x - width/2, df_lfw['AUC'], width, label='LFW', color=lfw_colors)
    plt.bar(x + width/2, df_gtfd['AUC'], width, label='GTFD', color=gtfd_colors)
    plt.title("AUC Comparison", fontsize=14, fontweight='bold')
    plt.xticks(x, combinations, rotation=45, ha='right')
    plt.ylabel("AUC")
    plt.legend()

    # 6: Accuracy bar comparison
    plt.subplot(3, 4, 6)
    plt.bar(x - width/2, df_lfw['Accuracy'], width, label='LFW', color=lfw_colors)
    plt.bar(x + width/2, df_gtfd['Accuracy'], width, label='GTFD', color=gtfd_colors)
    plt.title("Accuracy Comparison", fontsize=14, fontweight='bold')
    plt.xticks(x, combinations, rotation=45, ha='right')
    plt.ylabel("Accuracy")
    plt.legend()

    # 7-8: Threshold heatmaps
    plt.subplot(3, 4, 7)
    sns.heatmap(df_lfw.pivot("Architecture", "Loss Function", "Threshold"), annot=True, fmt=".3f", cmap="coolwarm", square=True)
    plt.title("LFW Dataset - Thresholds", fontsize=14, fontweight='bold')

    plt.subplot(3, 4, 8)
    sns.heatmap(df_gtfd.pivot("Architecture", "Loss Function", "Threshold"), annot=True, fmt=".3f", cmap="coolwarm", square=True)
    plt.title("GTFD Dataset - Thresholds", fontsize=14, fontweight='bold')

    # 9: Line plot AUC trends
    plt.subplot(3, 4, 9)
    for arch in df_lfw['Architecture'].unique():
        plt.plot(df_lfw[df_lfw['Architecture'] == arch]['Loss Function'],
                 df_lfw[df_lfw['Architecture'] == arch]['AUC'],
                 label=f"{arch} - LFW", marker='o')
    for arch in df_gtfd['Architecture'].unique():
        plt.plot(df_gtfd[df_gtfd['Architecture'] == arch]['Loss Function'],
                 df_gtfd[df_gtfd['Architecture'] == arch]['AUC'],
                 label=f"{arch} - GTFD", marker='s', linestyle='--')
    plt.title("AUC Trends by Architecture")
    plt.ylabel("AUC")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 10: Line plot Accuracy trends
    plt.subplot(3, 4, 10)
    for arch in df_lfw['Architecture'].unique():
        plt.plot(df_lfw[df_lfw['Architecture'] == arch]['Loss Function'],
                 df_lfw[df_lfw['Architecture'] == arch]['Accuracy'],
                 label=f"{arch} - LFW", marker='o')
    for arch in df_gtfd['Architecture'].unique():
        plt.plot(df_gtfd[df_gtfd['Architecture'] == arch]['Loss Function'],
                 df_gtfd[df_gtfd['Architecture'] == arch]['Accuracy'],
                 label=f"{arch} - GTFD", marker='s', linestyle='--')
    plt.title("Accuracy Trends by Architecture")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 11: Threshold bar chart
    plt.subplot(3, 4, 11)
    plt.bar(x - width/2, df_lfw['Threshold'], width, label='LFW', color=lfw_colors)
    plt.bar(x + width/2, df_gtfd['Threshold'], width, label='GTFD', color=gtfd_colors)
    plt.title("Threshold Comparison", fontsize=14, fontweight='bold')
    plt.xticks(x, combinations, rotation=45, ha='right')
    plt.ylabel("Threshold")
    plt.legend()

    # 12: Violin plot of AUC distributions
    plt.subplot(3, 4, 12)
    sns.violinplot(data=df_combined, x='Loss Function', y='AUC', hue='Dataset', split=True, palette='muted')
    plt.title("AUC Distribution by Loss Function", fontsize=14, fontweight='bold')
    plt.ylabel("AUC")

    # Final layout adjustments
    plt.tight_layout()
    plt.suptitle("Comprehensive Model Performance Comparison: LFW vs GTFD", fontsize=20, fontweight='bold', y=1.02)
    plt.subplots_adjust(top=0.93)
    plt.show()

# 4. Call the function
create_comprehensive_comparison()
