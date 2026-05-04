from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch
from decord import VideoReader, cpu

labels = {"Swiping Left": 0, "Swiping Right": 1}

class GestureDataset(Dataset):
    def __init__(self, split="train", num_frames=8):
        self.dataset = load_dataset("Ishara5/20bn-jester-event")[split]
        self.dataset = self.dataset.filter(lambda x: x["label"] in labels)

        self.num_frames = num_frames

        self.dataset = self.dataset.map(lambda x: {"label": labels[x["label"]]})

    def __len__(self):
        return len(self.dataset)

    def _load_video(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        idx = np.linspace(0, len(vr) - 1, self.num_frames).astype(int)
        frames = vr.get_batch(idx).asnumpy()
        frames = torch.tensor(frames).permute(3, 0, 1, 2) / 255.0
        return frames.float()

    def __getitem__(self, idx):
        item = self.dataset[idx]

        video = self._load_video(item["video"])
        label = item["label"]

        return video, torch.tensor(label)