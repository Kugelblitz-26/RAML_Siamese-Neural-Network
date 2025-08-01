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