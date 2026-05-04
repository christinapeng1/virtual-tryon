# train.py

import torch
from torch.utils.data import DataLoader
from dataset import GestureDataset
from model import GestureCNN
from utils import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = GestureDataset(split="train")
val_ds = GestureDataset(split="validation")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

model = GestureCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    total_loss = 0

    for v, l in train_loader:
        v, l = v.to(device), l.to(device)

        preds = model(v)
        loss = criterion(preds, l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    accs = []

    with torch.no_grad():
        for v, l in val_loader:
            v, l = v.to(device), l.to(device)
            preds = model(v)
            accs.append(accuracy(preds, l))

    print(f"Epoch {epoch}")
    print("Loss:", total_loss / len(train_loader))
    print("Val Acc:", sum(accs) / len(accs))

torch.save(model.state_dict(), "gesture.pt")