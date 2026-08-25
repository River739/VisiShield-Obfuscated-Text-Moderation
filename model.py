class SilhouetteNet(nn.Module):
    def __init__(self):
        super(SilhouetteNet, self).__init__()

        # Conv Block 1: Input (1, 64, 192) -> Output (32, 32, 96)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Conv Block 2: Input (32, 32, 96) -> Output (64, 16, 48)
        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Conv Block 3: Input (64, 16, 48) -> Output (128, 8, 24)
        self.conv3 = nn.Conv2d(64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)

        # Fully Connected Block
        # Calculated flat size: 128 channels * 8 height * 24 width = 24,576
        self.fc1 = nn.Linear(128 * 8 * 24, 128)
        self.fc2 = nn.Linear(128, 1) # Raw Logit for Binary Classification

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(-1, 128 * 8 * 24)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x