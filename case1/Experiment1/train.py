import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


from architectures import ResNet18Backbone, VGGBackbone, DeepFaceBackbone
from loss import ContrastiveLoss, TripletLoss, LogisticLoss
from LFW.processingLFW import LFWDataset, SiameseDataset, SiameseNetwork
from train import train_contrastive, train_triplet, SiameseNetwork
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def evaluate_model(model, test_loader, device):
    model.eval()
    similarities = []
    labels = []
    
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2 = img1.to(device), img2.to(device)
            embedding1, embedding2 = model(img1, img2)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(embedding1, embedding2)
            
            similarities.extend(similarity.cpu().numpy())
            labels.extend(label.numpy())
    
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Compute metrics
    auc = roc_auc_score(labels, similarities)
    
    # Find best threshold
    thresholds = np.linspace(-1, 1, 100)
    best_acc = 0
    best_threshold = 0
    
    for threshold in thresholds:
        predictions = (similarities > threshold).astype(int)
        acc = accuracy_score(labels, predictions)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    return auc, best_acc, best_threshold

# Main Experiment Function
def run_experiment(lfw_root_dir, num_epochs=10, batch_size=32):
    """
    Run the complete experiment comparing different architectures and loss functions
    """
    
    # Create datasets
    print("Loading LFW dataset...")
    train_dataset = LFWDataset(lfw_root_dir, transform=transform_train, mode='train')
    test_dataset = LFWDataset(lfw_root_dir, transform=transform_test, mode='test')
    
    # Split into train/test (you might want to use official LFW splits)
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_subset, test_subset = torch.utils.data.random_split(
        train_dataset, [train_size, test_size])
    
    # Create Siamese datasets
    train_siamese = SiameseDataset(train_subset, num_pairs=5000)
    test_siamese = SiameseDataset(test_subset, num_pairs=1000)
    
    train_loader = DataLoader(train_siamese, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_siamese, batch_size=batch_size, shuffle=False)
    
    # Define architectures
    architectures = {
        'ResNet18': lambda: ResNet18Backbone(embedding_dim=128),
        'VGG16': lambda: VGGBackbone(embedding_dim=128),
        'DeepFace': lambda: DeepFaceBackbone(embedding_dim=128)
    }
    
    # Define loss functions
    loss_functions = {
        'Contrastive': ContrastiveLoss(margin=2.0),
        'Logistic': LogisticLoss()
    }
    
    # Results storage
    results = []
    
    # Run experiments
    for arch_name, arch_fn in architectures.items():
        for loss_name, loss_fn in loss_functions.items():
            print(f"\n{'='*60}")
            print(f"Training {arch_name} with {loss_name} Loss")
            print(f"{'='*60}")
            
            # Create model
            backbone = arch_fn()
            model = SiameseNetwork(backbone).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                
                if loss_name == 'Triplet':
                    # Special handling for triplet loss
                    triplet_loss = TripletLoss(margin=1.0)
                    avg_loss = train_triplet(model, train_subset, triplet_loss, 
                                           optimizer, device, batch_size)
                else:
                    avg_loss = train_contrastive(model, train_loader, loss_fn, 
                                               optimizer, device)
                
                print(f"Average Loss: {avg_loss:.4f}")
            
            # Evaluation
            print("Evaluating model...")
            auc, accuracy, threshold = evaluate_model(model, test_loader, device)
            
            results.append({
                'Architecture': arch_name,
                'Loss Function': loss_name,
                'AUC': auc,
                'Accuracy': accuracy,
                'Threshold': threshold
            })
            
            print(f"Results - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    
    # Handle Triplet Loss separately (needs different training approach)
    triplet_loss = TripletLoss(margin=1.0)
    
    for arch_name, arch_fn in architectures.items():
        print(f"\n{'='*60}")
        print(f"Training {arch_name} with Triplet Loss")
        print(f"{'='*60}")
        
        backbone = arch_fn()
        model = SiameseNetwork(backbone).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            avg_loss = train_triplet(model, train_subset, triplet_loss, 
                                   optimizer, device, batch_size)
            print(f"Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        print("Evaluating model...")
        auc, accuracy, threshold = evaluate_model(model, test_loader, device)
        
        results.append({
            'Architecture': arch_name,
            'Loss Function': 'Triplet',
            'AUC': auc,
            'Accuracy': accuracy,
            'Threshold': threshold
        })
        
        print(f"Results - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
    
    return results

# Visualization Functions
def plot_results(results):
    """
    Create visualizations of the experiment results
    """
    df = pd.DataFrame(results)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Siamese Network Performance Comparison on LFW Dataset', fontsize=16)
    
    # 1. AUC Heatmap
    auc_pivot = df.pivot(index='Architecture', columns='Loss Function', values='AUC')
    sns.heatmap(auc_pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0,0])
    axes[0,0].set_title('AUC Scores')
    
    # 2. Accuracy Heatmap
    acc_pivot = df.pivot(index='Architecture', columns='Loss Function', values='Accuracy')
    sns.heatmap(acc_pivot, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[0,1])
    axes[0,1].set_title('Accuracy Scores')
    
    # 3. AUC Bar Plot
    df_melted = df.melt(id_vars=['Architecture', 'Loss Function'], 
                       value_vars=['AUC'], var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='Architecture', y='Score', hue='Loss Function', ax=axes[1,0])
    axes[1,0].set_title('AUC Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Accuracy Bar Plot
    df_melted_acc = df.melt(id_vars=['Architecture', 'Loss Function'], 
                           value_vars=['Accuracy'], var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted_acc, x='Architecture', y='Score', hue='Loss Function', ax=axes[1,1])
    axes[1,1].set_title('Accuracy Comparison')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print("\nSummary Results:")
    print("="*80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Find best combinations
    best_auc = df.loc[df['AUC'].idxmax()]
    best_acc = df.loc[df['Accuracy'].idxmax()]
    
    print(f"\nBest AUC: {best_auc['Architecture']} + {best_auc['Loss Function']} = {best_auc['AUC']:.4f}")
    print(f"Best Accuracy: {best_acc['Architecture']} + {best_acc['Loss Function']} = {best_acc['Accuracy']:.4f}")









# Example usage
if __name__ == "__main__":
    # Set your LFW dataset path here
    LFW_ROOT_DIR = "/path/to/lfw/dataset"  # Update this path
    
    # Check if path exists
    if not os.path.exists(LFW_ROOT_DIR):
        print(f"Please update LFW_ROOT_DIR to point to your LFW dataset directory")
        print("Expected structure: /path/to/lfw/PersonName/image1.jpg")
    else:
        # Run the complete experiment
        print("Starting Siamese Network Architecture Comparison Experiment")
        print("This will compare ResNet18, VGG16, and DeepFace architectures")
        print("with Contrastive, Triplet, and Logistic loss functions")
        
        results = run_experiment(LFW_ROOT_DIR, num_epochs=5, batch_size=16)
        
        # Visualize results
        plot_results(results)
        
        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv('siamese_network_results.csv', index=False)
        print("\nResults saved to 'siamese_network_results.csv'")
