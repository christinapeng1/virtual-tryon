from dataset import SwipeDataset
from model import GestureCNN
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np

dataset = SwipeDataset("data/Jester_Event/Train", "data/Jester_Event/Train.csv", max_samples=10000)

labels = np.array([dataset[i][1] for i in range(len(dataset))])
class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
class_weights = torch.tensor(class_weights, dtype=torch.float32)

sample_weights = class_weights[labels]
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

loader = DataLoader(dataset, batch_size=16, sampler=sampler)

model = GestureCNN(num_classes=3).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
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

    acc = correct / total
    print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.4f} | Accuracy {acc:.4f}")

torch.save(model.state_dict(), "gesture_model.pth")
