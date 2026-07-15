import torch
import torch.nn as nn
import torch.nn.functional as F

class SilhouetteNet(nn.Module):
    def __init__(self):
        super(SilhouetteNet, self).__init__()
        
        # Input: 1 channel (grayscale), Size: 64 x 192
        
        # Conv Block 1: Extracts low-level shape contours
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Block 2: Captures broader geometric structures
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Conv Block 3: Deep features (word lengths, spacing patterns)
        self.conv3 = nn.Conv2d(64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling to shrink spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout to prevent overfitting on specific font types
        self.dropout = nn.Dropout(0.4)
        
        # Fully Connected Layers
        # After three MaxPool operations (64x192 -> 32x96 -> 16x48 -> 8x24)
        # Flattened size = 128 channels * 8 height * 24 width = 24576
        self.fc1 = nn.Linear(128 * 8 * 24, 128)
        self.fc2 = nn.Linear(128, 1) # Binary output (Logit for BCEWithLogitsLoss)

    def forward(self, x):
        # Layer 1: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Layer 2: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Layer 3: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the feature maps for the dense layers
        x = x.view(-1, 128 * 8 * 24)
        
        # Fully Connected Block
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output raw logit
        
        return x

# Quick verification check
if __name__ == "__main__":
    model = SilhouetteNet()
    # Mock Batch of 4 images, 1 channel, 64x192 resolution
    mock_input = torch.randn(4, 1, 64, 192)
    output = model(mock_input)
    print(f"Model successfully initialized! Output shape for batch: {output.shape}")