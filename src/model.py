# model.py - Contains the structure of the Siamese Network model

import torch
import torch.nn as nn

# Define a Siamese Network for learning image embeddings
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        # Convolutional feature extractor (CNN backbone)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),   # 1 input channel (e.g., grayscale), 32 output channels, 5x5 filter
            nn.ReLU(inplace=True),             # Activation function
            nn.MaxPool2d(2),                   # Downsample by a factor of 2
            nn.Conv2d(32, 64, kernel_size=5),  # 32 input channels, 64 output channels, 5x5 filter
            nn.ReLU(inplace=True),             # Activation function
            nn.MaxPool2d(2),                   # Downsample by a factor of 2
        )

        # Fully connected layers to produce the final embedding
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),        # Flattened output from conv layers (assumes 28x28 input size)
            nn.ReLU(inplace=True),             # Activation function
            nn.Linear(256, embedding_dim)      # Final embedding of specified dimension
        )

    # Forward pass through the network for a single input
    def forward_once(self, x):
        x = self.conv(x)                       # Extract features using CNN
        x = x.view(x.size(0), -1)              # Flatten the feature maps
        x = self.fc(x)                         # Get the embedding vector
        return x

    # Forward pass for a pair of inputs (used in Siamese setup)
    def forward(self, input1, input2):
        out1 = self.forward_once(input1)       # Embedding for first input
        out2 = self.forward_once(input2)       # Embedding for second input
        return out1, out2                      # Return both embeddings
