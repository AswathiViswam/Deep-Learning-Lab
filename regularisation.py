#apply L1,L2 and Dropout regularisation
#implement weight initialisation technique

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

class FeedForwardNN(nn.Module):
    def __init__(self, dropout=False):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

def init_weights(model, mode='default'):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if mode == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif mode == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

def train(model, train_loader, test_loader, lr=0.001, weight_decay=0.0, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss / len(train_loader))
        acc = evaluate(model, test_loader)
        accuracies.append(acc)
        print(f"Epoch {epoch+1}: Loss={losses[-1]:.4f}, Accuracy={acc:.2f}%")
    return losses, accuracies

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100 * correct / total

def run_experiment(name, init='default', dropout=False, l2=0.0):
    print(f"\n=== {name} ===")
    model = FeedForwardNN(dropout=dropout)
    if init != 'default':
        init_weights(model, init)
    _, accuracies = train(model, train_loader, test_loader, weight_decay=l2)
    final_acc = accuracies[-1]
    return name, final_acc

experiments = [
    {"name": "Baseline"},
    {"name": "Xavier Initialization", "init": "xavier"},
    {"name": "Kaiming Initialization", "init": "kaiming"},
    {"name": "Dropout Regularization", "dropout": True},
    {"name": "L2 Regularization", "l2": 1e-4},
]

summary = []
for exp in experiments:
    name, acc = run_experiment(**exp)
    summary.append((name, acc))

print("\n=== Experiment Summary ===")
for name, acc in summary:
    print(f"{name:30s} -> Accuracy: {acc:.2f}%")
