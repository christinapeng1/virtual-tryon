from dataset import SwipeDataset
from model import GestureCNN
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import torch
import numpy as np
import matplotlib.pyplot as plt

dataset = SwipeDataset("data/Jester_Event/Train", "data/Jester_Event/Train.csv", max_samples=10000)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train, val = random_split(dataset, [train_size, val_size])

train_labels = np.array([train[i][1] for i in range(len(train))])

class_counts = np.bincount(train_labels)

for i, c in enumerate(class_counts):
    print(f"Class {i}: {c}")

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights, dtype=torch.float32)

sample_weights = class_weights[train_labels]
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train, batch_size=16, sampler=sampler)
val_loader = DataLoader(val, batch_size=16)

model = GestureCNN(num_classes=3).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

train_accs = []
val_accs = []

for epoch in range(10):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        pred = model(x)
        loss = loss_fn(pred, y)

        preds = pred.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

    train_acc = correct / total
    train_accs.append(train_acc)

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()

            pred = model(x)
            preds = pred.argmax(dim=1)

            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_acc = val_correct / val_total
    val_accs.append(val_acc)

    print(f"Epoch {epoch} | Loss: {total_loss / len(train_loader):.4f} | Train Accuracy {train_acc:.4f} | Val Accuracy {val_acc:.4f}")

plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.show()

torch.save(model.state_dict(), "gesture_model.pth")
