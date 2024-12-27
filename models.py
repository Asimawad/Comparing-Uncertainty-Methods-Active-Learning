# Models for comparing uncertainty methods
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    print(f"youre on Cuda {torch.cuda}")
elif torch.backends.mps.is_available():
    device = "mps"


# LeNet5 with additional dropout layers for MNIST
class BayesianMNISTCNN(nn.Module):
    """LeNet5 with additional dropout layers."""

    def __init__(self, num_classes=10, p=0.5):
        super(BayesianMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.dropout1 = nn.Dropout2d(p=p)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout2 = nn.Dropout2d(p=p)
        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.dropout3 = nn.Dropout(p=p)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x, _ = F.max_pool2d(x, kernel_size=2, return_indices=True)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x, _ = F.max_pool2d(x, kernel_size=2, return_indices=True)
        x = self.dropout2(x)

        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)

        embedding = x
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output, embedding

def create_model_optimizer(seed) -> tuple[BayesianMNISTCNN, torch.optim.Optimizer]:
    """Create a model and optimizer for MNIST."""
    torch.manual_seed(seed)
    model = BayesianMNISTCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    return model, optimizer