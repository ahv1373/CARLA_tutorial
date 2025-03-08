import torch
import torch.nn as nn
import torch.nn.functional as F


class RoadTypeCNNClassifier(nn.Module):
    def __init__(self, input_channels: int = 3):
        super(RoadTypeCNNClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32 * 32, 256)  # 128 x 32 x 32 is the size of the tensor after the conv layers
        self.fc2 = nn.Linear(256, 1)  # Binary classification output

        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, start_dim=1)  # start_dim=1 to flatten all dimensions except batch (leave dimension 0)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for binary classification
        return x


if __name__ == "__main__":
    # Test the model
    model = RoadTypeCNNClassifier()
    print(model)
    x_ = torch.randn(4, 3, 256, 256)  # Standard size of the input vector is batch_size x channels x height x width
    y = model(x_)
    print(y)  # Output size should be 1x1 which is the binary classification output
    print(y.shape)