import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# For InceptionV3 (needs 299x299)
transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hook class to capture intermediate layer outputs
class LayerActivationHook:
    def __init__(self):
        self.activations = OrderedDict()
        self.hooks = []
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def register_hooks(self, model, layer_names):
        for name, module in model.named_modules():
            if name in layer_names:
                hook = self.hooks.append(module.register_forward_hook(self.get_activation(name)))
    
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = OrderedDict()

# Architecture Definitions with Layer Visualization Support

class ResNet18Backbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ResNet18Backbone, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embedding_dim)
        
        # Define layer names for visualization
        self.layer_names = [
            'resnet.conv1',           # Initial conv
            'resnet.bn1',             # Batch norm
            'resnet.layer1.0.conv1',  # First residual block
            'resnet.layer1.1.conv2',  # Second residual block
            'resnet.layer2.0.conv1',  # Third layer start
            'resnet.layer2.1.conv2',  # Third layer end
            'resnet.layer3.0.conv1',  # Fourth layer start
            'resnet.layer3.1.conv2',  # Fourth layer end
            'resnet.layer4.0.conv1',  # Fifth layer start
            'resnet.layer4.1.conv2',  # Fifth layer end
            'resnet.avgpool',         # Global average pooling
            'resnet.fc'               # Final embedding
        ]
        
    def forward(self, x):
        return F.normalize(self.resnet(x), p=2, dim=1)

class VGGBackbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super(VGGBackbone, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, embedding_dim)
        )
        
        # Define layer names for visualization
        self.layer_names = [
            'vgg.features.0',    # Conv 1-1
            'vgg.features.2',    # Conv 1-2
            'vgg.features.5',    # Conv 2-1
            'vgg.features.7',    # Conv 2-2
            'vgg.features.10',   # Conv 3-1
            'vgg.features.12',   # Conv 3-2
            'vgg.features.14',   # Conv 3-3
            'vgg.features.17',   # Conv 4-1
            'vgg.features.19',   # Conv 4-2
            'vgg.features.21',   # Conv 4-3
            'vgg.features.24',   # Conv 5-1
            'vgg.features.26',   # Conv 5-2
            'vgg.features.28',   # Conv 5-3
            'vgg.classifier.0',  # FC 1
            'vgg.classifier.3',  # FC 2
            'vgg.classifier.6'   # Final embedding
        ]
        
    def forward(self, x):
        return F.normalize(self.vgg(x), p=2, dim=1)

class EfficientNetB7Backbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super(EfficientNetB7Backbone, self).__init__()
        self.efficientnet = models.efficientnet_b7(pretrained=True)
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(self.efficientnet.classifier[1].in_features, embedding_dim)
        )
        
        # Define key layer names for visualization
        self.layer_names = [
            'efficientnet.features.0.0',     # Initial conv
            'efficientnet.features.1.0.block.0.0',  # First MBConv
            'efficientnet.features.2.0.block.0.0',  # Second MBConv
            'efficientnet.features.3.0.block.0.0',  # Third MBConv
            'efficientnet.features.4.0.block.0.0',  # Fourth MBConv
            'efficientnet.features.5.0.block.0.0',  # Fifth MBConv
            'efficientnet.features.6.0.block.0.0',  # Sixth MBConv
            'efficientnet.features.7.0.block.0.0',  # Seventh MBConv
            'efficientnet.features.8.0',     # Final conv
            'efficientnet.avgpool',          # Global average pooling
            'efficientnet.classifier.1'     # Final embedding
        ]
        
    def forward(self, x):
        return F.normalize(self.efficientnet(x), p=2, dim=1)

class InceptionV3Backbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super(InceptionV3Backbone, self).__init__()
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_dim)
        
        # Define layer names for visualization
        self.layer_names = [
            'inception.Conv2d_1a_3x3.conv',      # Initial conv
            'inception.Conv2d_2a_3x3.conv',      # Second conv
            'inception.Conv2d_2b_3x3.conv',      # Third conv
            'inception.Conv2d_3b_1x1.conv',      # Fourth conv
            'inception.Conv2d_4a_3x3.conv',      # Fifth conv
            'inception.Mixed_5b.branch1x1.conv', # First inception
            'inception.Mixed_5c.branch1x1.conv', # Second inception
            'inception.Mixed_5d.branch1x1.conv', # Third inception
            'inception.Mixed_6a.branch3x3.conv', # Reduction A
            'inception.Mixed_6b.branch1x1.conv', # Fourth inception
            'inception.Mixed_7a.branch3x3.conv', # Reduction B
            'inception.Mixed_7b.branch1x1.conv', # Final inception
            'inception.avgpool',                 # Global average pooling
            'inception.fc'                       # Final embedding
        ]
        
    def forward(self, x):
        # InceptionV3 expects 299x299 input
        if x.size(2) != 299 or x.size(3) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return F.normalize(self.inception(x), p=2, dim=1)

# Siamese Network with Visualization
class SiameseNetworkViz(nn.Module):
    def __init__(self, backbone, loss_type='contrastive'):
        super(SiameseNetworkViz, self).__init__()
        self.backbone = backbone
        self.loss_type = loss_type
        
        # Initialize loss-specific layers if needed
        if loss_type == 'logistic':
            self.similarity_head = nn.Sequential(
                nn.Linear(backbone.layer_names[-1].split('.')[-1] 
                         if hasattr(backbone, 'layer_names') else 128, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x1, x2=None):
        if x2 is not None:
            # Siamese forward pass
            embedding1 = self.backbone(x1)
            embedding2 = self.backbone(x2)
            
            if self.loss_type == 'logistic':
                # Compute similarity score
                similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
                return embedding1, embedding2, similarity
            else:
                return embedding1, embedding2
        else:
            # Single image forward pass for visualization
            return self.backbone(x1)

# Visualization Functions

def denormalize_image(tensor):
    """Denormalize image tensor for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return tensor * std + mean

def visualize_feature_maps(activations, layer_names, max_channels=16):
    """Visualize feature maps from each layer"""
    n_layers = len(activations)
    
    # Create a large figure
    fig = plt.figure(figsize=(20, 4 * n_layers))
    
    for idx, (layer_name, activation) in enumerate(activations.items()):
        # Skip if activation is None or has wrong dimensions
        if activation is None or len(activation.shape) < 2:
            continue
            
        # Handle different activation shapes
        if len(activation.shape) == 4:  # Conv layers: (batch, channels, height, width)
            activation = activation[0]  # Take first batch
            n_channels = min(activation.shape[0], max_channels)
            
            # Create subplot grid for this layer
            start_idx = idx * max_channels + 1
            
            for ch in range(n_channels):
                plt.subplot(n_layers, max_channels, start_idx + ch)
                
                feature_map = activation[ch].cpu().numpy()
                plt.imshow(feature_map, cmap='viridis')
                plt.axis('off')
                
                if ch == 0:  # Add layer name only to first channel
                    plt.title(f'{layer_name}\nChannel {ch}', fontsize=8)
                else:
                    plt.title(f'Ch {ch}', fontsize=8)
                    
        elif len(activation.shape) == 2:  # FC layers: (batch, features)
            plt.subplot(n_layers, 1, idx + 1)
            activation_1d = activation[0].cpu().numpy()
            plt.plot(activation_1d)
            plt.title(f'{layer_name} - Embedding Vector (Size: {len(activation_1d)})')
            plt.xlabel('Feature Index')
            plt.ylabel('Activation Value')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_layer_progression(model, image_tensor, architecture_name, loss_type):
    """Visualize how image transforms through each layer"""
    
    # Setup hooks
    hook_manager = LayerActivationHook()
    hook_manager.register_hooks(model.backbone, model.backbone.layer_names)
    
    # Forward pass to capture activations
    model.eval()
    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0))
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(25, 30))
    
    # Original image
    plt.subplot(6, 4, 1)
    orig_img = denormalize_image(image_tensor.unsqueeze(0))[0]
    orig_img = torch.clamp(orig_img, 0, 1)
    plt.imshow(orig_img.permute(1, 2, 0).cpu().numpy())
    plt.title(f'Original Image\n{architecture_name} + {loss_type.title()} Loss', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Layer activations
    plot_idx = 2
    for layer_name, activation in hook_manager.activations.items():
        if activation is None:
            continue
            
        plt.subplot(6, 4, plot_idx)
        
        if len(activation.shape) == 4:  # Conv layer
            # Take first batch, average across channels for visualization
            feature_map = activation[0].mean(dim=0).cpu().numpy()
            plt.imshow(feature_map, cmap='plasma')
            plt.title(f'{layer_name.split(".")[-1]}\nShape: {tuple(activation.shape[1:])}', fontsize=9)
            
        elif len(activation.shape) == 2:  # FC layer
            # Visualize as bar plot for embeddings
            embedding = activation[0].cpu().numpy()
            if len(embedding) <= 50:  # Small embeddings - bar plot
                plt.bar(range(len(embedding)), embedding)
                plt.title(f'{layer_name.split(".")[-1]}\nEmbedding Size: {len(embedding)}', fontsize=9)
            else:  # Large embeddings - line plot
                plt.plot(embedding)
                plt.title(f'{layer_name.split(".")[-1]}\nEmbedding Size: {len(embedding)}', fontsize=9)
            plt.xlabel('Feature Index')
            plt.ylabel('Value')
            
        plt.axis('off' if len(activation.shape) == 4 else 'on')
        plot_idx += 1
        
        if plot_idx > 24:  # Limit to 24 subplots
            break
    
    plt.suptitle(f'{architecture_name} Feature Progression with {loss_type.title()} Loss', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Clean up hooks
    hook_manager.clear_hooks()
    
    return fig, hook_manager.activations

def compare_pair_similarity(model1, model2, img1, img2, arch1_name, arch2_name, loss_type):
    """Compare similarity scores between two models on a pair of images"""
    
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        # Get embeddings from both models
        emb1_model1, emb2_model1 = model1(img1.unsqueeze(0), img2.unsqueeze(0))
        emb1_model2, emb2_model2 = model2(img1.unsqueeze(0), img2.unsqueeze(0))
        
        # Compute similarities
        sim1 = F.cosine_similarity(emb1_model1, emb2_model1).item()
        sim2 = F.cosine_similarity(emb1_model2, emb2_model2).item()
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    img1_show = denormalize_image(img1.unsqueeze(0))[0]
    img2_show = denormalize_image(img2.unsqueeze(0))[0]
    img1_show = torch.clamp(img1_show, 0, 1)
    img2_show = torch.clamp(img2_show, 0, 1)
    
    axes[0, 0].imshow(img1_show.permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Image 1', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img2_show.permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title('Image 2', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Similarity comparison
    models = [arch1_name, arch2_name]
    similarities = [sim1, sim2]
    colors = ['skyblue', 'lightcoral']
    
    axes[0, 2].bar(models, similarities, color=colors)
    axes[0, 2].set_title(f'Cosine Similarity\n({loss_type.title()} Loss)', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Similarity Score')
    axes[0, 2].set_ylim(-1, 1)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add similarity values on bars
    for i, sim in enumerate(similarities):
        axes[0, 2].text(i, sim + 0.05, f'{sim:.3f}', ha='center', fontweight='bold')
    
    # Embedding visualizations
    emb1_1 = emb1_model1[0].cpu().numpy()
    emb1_2 = emb1_model2[0].cpu().numpy()
    
    # Plot embeddings
    axes[1, 0].plot(emb1_1[:50], 'b-', label=f'{arch1_name}', linewidth=2)
    axes[1, 0].plot(emb1_2[:50], 'r-', label=f'{arch2_name}', linewidth=2)
    axes[1, 0].set_title('Image 1 Embeddings (First 50 dims)', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    emb2_1 = emb2_model1[0].cpu().numpy()
    emb2_2 = emb2_model2[0].cpu().numpy()
    
    axes[1, 1].plot(emb2_1[:50], 'b-', label=f'{arch1_name}', linewidth=2)
    axes[1, 1].plot(emb2_2[:50], 'r-', label=f'{arch2_name}', linewidth=2)
    axes[1, 1].set_title('Image 2 Embeddings (First 50 dims)', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Embedding distance visualization
    distances = [
        np.linalg.norm(emb1_1 - emb2_1),  # L2 distance for model 1
        np.linalg.norm(emb1_2 - emb2_2)   # L2 distance for model 2
    ]
    
    axes[1, 2].bar(models, distances, color=colors)
    axes[1, 2].set_title('L2 Distance Between Embeddings', fontsize=10)
    axes[1, 2].set_ylabel('L2 Distance')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add distance values on bars
    for i, dist in enumerate(distances):
        axes[1, 2].text(i, dist + 0.1, f'{dist:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def load_sample_images(lfw_path, num_images=2):
    """Load sample images from LFW dataset"""
    images = []
    image_paths = []
    
    # Find first available person directory
    for person_dir in os.listdir(lfw_path):
        person_path = os.path.join(lfw_path, person_dir)
        if os.path.isdir(person_path):
            # Get images from this person
            img_files = [f for f in os.listdir(person_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in img_files[:num_images]:
                img_path = os.path.join(person_path, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                    image_paths.append(img_path)
                    
                    if len(images) >= num_images:
                        return images, image_paths
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
    
    return images, image_paths

# Main Visualization Function
def run_layer_visualization_experiment(lfw_path):
    """Run comprehensive layer visualization experiment"""
    
    print("🎨 Starting Layer Visualization Experiment")
    print("="*60)
    
    # Load sample images
    print("📸 Loading sample images...")
    images, image_paths = load_sample_images(lfw_path, num_images=2)
    
    if len(images) < 2:
        print("❌ Could not load sufficient images from dataset")
        return
    
    print(f"✅ Loaded images from: {image_paths}")
    
    # Define architectures and loss functions
    architectures = {
        'ResNet18': ResNet18Backbone,
        'VGG16': VGGBackbone,
        'EfficientNet-B7': EfficientNetB7Backbone,
        'InceptionV3': InceptionV3Backbone
    }
    
    loss_functions = ['contrastive', 'triplet']
    
    # Process each architecture with each loss function
    all_figures = []
    
    for arch_name, arch_class in architectures.items():
        print(f"\n🏗️  Processing {arch_name} Architecture...")
        
        for loss_type in loss_functions:
            print(f"   🔧 With {loss_type.title()} Loss...")
            
            # Create model
            backbone = arch_class(embedding_dim=128)
            model = SiameseNetworkViz(backbone, loss_type=loss_type).to(device)
            
            # Prepare image (special handling for InceptionV3)
            if arch_name == 'InceptionV3':
                img_tensor = transform_inception(images[0]).to(device)
            else:
                img_tensor = transform(images[0]).to(device)
            
            # Generate layer visualization
            try:
                fig, activations = visualize_layer_progression(
                    model, img_tensor, arch_name, loss_type)
                
                # Save figure
                fig_name = f'{arch_name}_{loss_type}_layers.png'
                fig.savefig(fig_name, dpi=150, bbox_inches='tight')
                all_figures.append((fig_name, fig))
                
                print(f"     ✅ Generated visualization: {fig_name}")
                
            except Exception as e:
                print(f"     ❌ Error with {arch_name} + {loss_type}: {e}")
                continue
    
    # Generate pair comparison
    print(f"\n🔍 Generating Pair Similarity Comparison...")
    
    try:
        # Compare ResNet18 vs EfficientNet with contrastive loss
        resnet_model = SiameseNetworkViz(
            ResNet18Backbone(embedding_dim=128), 'contrastive').to(device)
        efficientnet_model = SiameseNetworkViz(
            EfficientNetB7Backbone(embedding_dim=128), 'contrastive').to(device)
        
        img1_tensor = transform(images[0]).to(device)
        img2_tensor = transform(images[1]).to(device)
        
        comparison_fig = compare_pair_similarity(
            resnet_model, efficientnet_model,
            img1_tensor, img2_tensor,
            'ResNet18', 'EfficientNet-B7', 'contrastive'
        )
        
        comparison_fig.savefig('pair_similarity_comparison.png', dpi=150, bbox_inches='tight')
        all_figures.append(('pair_similarity_comparison.png', comparison_fig))
        
        print("✅ Generated pair similarity comparison")
        
    except Exception as e:
        print(f"❌ Error generating pair comparison: {e}")
    
    # Summary
    print(f"\n🎉 Experiment Complete!")
    print(f"📊 Generated {len(all_figures)} visualizations")
    print("📁 Files saved:")
    for fig_name, _ in all_figures:
        print(f"   - {fig_name}")
    
    # Display all figures
    plt.show()
    
    return all_figures

# Quick test function for single architecture
def quick_layer_viz(image_path, architecture='ResNet18', loss_type='contrastive'):
    """Quick visualization for single image and architecture"""
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    
    if architecture == 'InceptionV3':
        img_tensor = transform_inception(img).to(device)
    else:
        img_tensor = transform(img).to(device)
    
    # Create model
    arch_classes = {
        'ResNet18': ResNet18Backbone,
        'VGG16': VGGBackbone,
        'EfficientNet-B7': EfficientNetB7Backbone,
        'InceptionV3': InceptionV3Backbone
    }
    
    backbone = arch_classes[architecture](embedding_dim=128)
    model = SiameseNetworkViz(backbone, loss_type=loss_type).to(device)
    
    # Generate visualization
    fig, activations = visualize_layer_progression(model, img_tensor, architecture, loss_type)
    
    return fig, activations

# Example usage
if __name__ == "__main__":
    # Set your LFW dataset path
    LFW_PATH = r"C:\Users\phani\Desktop\RAML Datasets\lfw_filtered"  # Update this path
    
    if not os.path.exists(LFW_PATH):
        print("🚨 Please update LFW_PATH to point to your LFW dataset directory")
        print("Expected structure: /path/to/lfw/PersonName/image1.jpg")
        print("\n🔧 Available for visualization:")
        print("   Architectures: ResNet18, VGG16, EfficientNet-B7, InceptionV3")
        print("   Loss Functions: Contrastive, Triplet, Logistic")
        print("   Outputs: Layer-wise feature maps, embedding progressions, similarity comparisons")
        
        # Demo with dummy image if no dataset available
        print("\n🎭 Running demo with dummy image...")
        dummy_img = torch.randn(3, 224, 224)
        
        # Save dummy image
        dummy_pil = transforms.ToPILImage()(torch.clamp(dummy_img * 0.5 + 0.5, 0, 1))
        dummy_pil.save('dummy_image.jpg')
        
        try:
            fig, activations = quick_layer_viz('dummy_image.jpg', 'ResNet18', 'contrastive')
            fig.savefig('demo_layer_visualization.png', dpi=150, bbox_inches='tight')
            print("✅ Demo visualization saved as 'demo_layer_visualization.png'")
            plt.show()
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            
    else:
        # Run full experiment
        figures = run_layer_visualization_experiment(LFW_PATH)
        print(f"\n🎨 All visualizations complete! Generated {len(figures)} figures.")
