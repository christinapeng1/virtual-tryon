import torch

def accuracy(preds, labels):
    return (preds.argmax(1) == labels).float().mean().item()