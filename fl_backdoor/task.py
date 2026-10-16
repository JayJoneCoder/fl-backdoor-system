"""fl_backdoor: A Flower / PyTorch app."""

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from fl_backdoor.dataset import get_dataset


class Net(nn.Module):
    """CNN model that adapts to dataset input shape and number of classes."""

    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        super().__init__()
        C, H, W = input_shape

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(C, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Fixed output size regardless of input
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_data(partition_id: int, num_partitions: int, batch_size: int, dataset_name: str = "cifar10", *, seed=42, shuffle_seed=None):
    """Load partition data for a specific client."""
    dataset = get_dataset(dataset_name)
    return dataset.load_partition(partition_id, num_partitions, batch_size, seed=seed, shuffle_seed=shuffle_seed)


def load_centralized_dataset(dataset_name: str = "cifar10", batch_size: int = 128):
    """Load centralized test set for global evaluation."""
    dataset = get_dataset(dataset_name)
    return dataset.load_centralized_test(batch_size)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


@contextmanager
def evaluation_mode(net: nn.Module) -> Iterator[None]:
    """Temporarily disable training behavior and gradients, restoring every layer."""
    training_modes = [(module, module.training) for module in net.modules()]
    try:
        net.eval()
        with torch.no_grad():
            yield
    finally:
        # Calling net.train() here would overwrite individually frozen layer modes.
        for module, was_training in training_modes:
            module.training = was_training


def test(net, testloader, device):
    """Validate without updating model state or changing its original layer modes."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with evaluation_mode(net):
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy