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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preprocessing and augmentation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset for LFW
class LFWDataset(Dataset):
    def __init__(self, root_dir, pairs_file=None, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Get all person directories
        self.people = [d for d in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, d))]
        
        # Create image paths and labels
        self.image_paths = []
        self.labels = []
        
        for person_idx, person in enumerate(self.people):
            person_dir = os.path.join(root_dir, person)
            images = [f for f in os.listdir(person_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img in images:
                self.image_paths.append(os.path.join(person_dir, img))
                self.labels.append(person_idx)
        
        self.num_classes = len(self.people)
        print(f"Dataset loaded: {len(self.image_paths)} images, {self.num_classes} people")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image and label
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, 0

# Siamese Dataset for pairs
class SiameseDataset(Dataset):
    def __init__(self, base_dataset, num_pairs=10000):
        self.base_dataset = base_dataset
        self.num_pairs = num_pairs
        
        # Group images by label
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(base_dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        # Generate pairs
        self.pairs = []
        self.labels = []
        
        # Generate positive pairs (same person)
        for _ in range(num_pairs // 2):
            label = np.random.choice(list(self.label_to_indices.keys()))
            if len(self.label_to_indices[label]) >= 2:
                idx1, idx2 = np.random.choice(self.label_to_indices[label], 2, replace=False)
                self.pairs.append((idx1, idx2))
                self.labels.append(1)  # Same person
        
        # Generate negative pairs (different people)
        for _ in range(num_pairs // 2):
            label1, label2 = np.random.choice(list(self.label_to_indices.keys()), 2, replace=False)
            idx1 = np.random.choice(self.label_to_indices[label1])
            idx2 = np.random.choice(self.label_to_indices[label2])
            self.pairs.append((idx1, idx2))
            self.labels.append(0)  # Different people
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        label = self.labels[idx]
        
        img1, _ = self.base_dataset[idx1]
        img2, _ = self.base_dataset[idx2]
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Architecture Definitions

# 1. ResNet18 Backbone
class ResNet18Backbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ResNet18Backbone, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Remove the final classification layer
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)
        
    def forward(self, x):
        return F.normalize(self.resnet(x), p=2, dim=1)

# 2. VGG Backbone
class VGGBackbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super(VGGBackbone, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        # Modify classifier
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, embedding_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.vgg(x), p=2, dim=1)

# 3. DeepFace-like Architecture
class DeepFaceBackbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super(DeepFaceBackbone, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            
            # Conv Block 2
            nn.Conv2d(32, 16, kernel_size=9, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            
            # Conv Block 3
            nn.Conv2d(16, 16, kernel_size=9, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            
            # Conv Block 4
            nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            
            # Conv Block 5
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
        )
        
        # Calculate the size after convolutions
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16 * 13 * 13, 4096),  # Adjust based on actual output size
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, embedding_dim)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, backbone, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        
    def forward(self, x1, x2):
        embedding1 = self.backbone(x1)
        embedding2 = self.backbone(x2)
        return embedding1, embedding2
    
    def get_embedding(self, x):
        return self.backbone(x)

# Loss Functions

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embedding1, embedding2, label):
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss

class LogisticLoss(nn.Module):
    def __init__(self):
        super(LogisticLoss, self).__init__()
        
    def forward(self, embedding1, embedding2, label):
        # Compute similarity score
        similarity = F.cosine_similarity(embedding1, embedding2)
        # Convert to probability
        prob = torch.sigmoid(similarity * 10)  # Scale for better gradients
        # Binary cross entropy
        loss = F.binary_cross_entropy(prob, label)
        return loss

# Training Functions

def train_contrastive(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (img1, img2, label) in enumerate(dataloader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        
        optimizer.zero_grad()
        embedding1, embedding2 = model(img1, img2)
        loss = criterion(embedding1, embedding2, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def train_triplet(model, base_dataset, criterion, optimizer, device, batch_size=32):
    model.train()
    total_loss = 0
    num_batches = 100  # Number of triplet batches per epoch
    
    # Group samples by label
    label_to_indices = {}
    for idx in range(len(base_dataset)):
        _, label = base_dataset[idx]
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    labels_with_multiple = [label for label, indices in label_to_indices.items() 
                           if len(indices) >= 2]
    
    for batch_idx in range(num_batches):
        anchors, positives, negatives = [], [], []
        
        for _ in range(batch_size):
            # Select anchor label (must have at least 2 samples)
            anchor_label = np.random.choice(labels_with_multiple)
            anchor_idx, positive_idx = np.random.choice(
                label_to_indices[anchor_label], 2, replace=False)
            
            # Select negative label
            negative_label = np.random.choice(
                [l for l in labels_with_multiple if l != anchor_label])
            negative_idx = np.random.choice(label_to_indices[negative_label])
            
            anchor_img, _ = base_dataset[anchor_idx]
            positive_img, _ = base_dataset[positive_idx]
            negative_img, _ = base_dataset[negative_idx]
            
            anchors.append(anchor_img)
            positives.append(positive_img)
            negatives.append(negative_img)
        
        anchors = torch.stack(anchors).to(device)
        positives = torch.stack(positives).to(device)
        negatives = torch.stack(negatives).to(device)
        
        optimizer.zero_grad()
        anchor_emb = model.get_embedding(anchors)
        positive_emb = model.get_embedding(positives)
        negative_emb = model.get_embedding(negatives)
        
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 20 == 0:
            print(f'Triplet Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches

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
