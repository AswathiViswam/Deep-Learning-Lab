#Implement a feedforward neural network with training and evaluation on CIFAR-10 dataset, supporting multiple activation functions and configurations.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

activation_map = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid()
}

class FeedForwardNN(nn.Module):
    def __init__(self, hidden_units, activation):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4 = nn.Linear(hidden_units[2], 10)
        self.act = activation_map[activation]

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.fc4(x)
        return x

def train(model, train_loader, criterion, optimizer):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

configs = [
    ((512, 256, 128), 'relu'),
    ((512, 256, 128), 'tanh'),
    ((512, 256, 128), 'sigmoid'),
    ((256, 128, 64), 'relu'),
    ((256, 128, 64), 'tanh'),
    ((256, 128, 64), 'sigmoid'),
    ((1024, 512, 256), 'relu'),
    ((1024, 512, 256), 'tanh'),
    ((1024, 512, 256), 'sigmoid')
]

results = []
best_acc = 0
best_model = None
best_config = None

for idx, (hidden_units, activation) in enumerate(configs):
    print(f"\nRun {idx+1}: Hidden={hidden_units}, Activation={activation}")
    model = FeedForwardNN(hidden_units, activation).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        train(model, train_loader, criterion, optimizer)

    acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {acc:.2f}%")
    results.append((idx + 1, hidden_units, activation, acc))

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_config = (hidden_units, activation)

print("\n=== Final Results ===")
for run_id, hidden_units, activation, acc in results:
    print(f"Run {run_id}: Hidden={hidden_units}, Activation={activation}, Accuracy={acc:.2f}%")

print(f"\nBest Model Config: Hidden={best_config[0]}, Activation={best_config[1]}, Accuracy={best_acc:.2f}%")

best_model.eval()
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

outputs = best_model(images)
_, predicted = torch.max(outputs, 1)

images = images.cpu()
predicted = predicted.cpu()
labels = labels.cpu()

classes = train_data.classes

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    img = images[i].permute(1, 2, 0).numpy()  
    img = img * 0.5 + 0.5  
    plt.imshow(img)
    plt.title(f"P: {classes[predicted[i]]}\nT: {classes[labels[i]]}")
    plt.axis('off')
plt.suptitle("Predictions on Test Images - Best Model")
plt.show()
