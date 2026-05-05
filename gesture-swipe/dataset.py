import glob
import h5py
import torch
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import Dataset

LEFT = {11, 16}
RIGHT = {12, 17}

H, W = 100, 176

class SwipeDataset(Dataset):
    def __init__(self, root, csv_path, max_samples=1000):
        self.files = glob.glob(root + "/**/events.h5", recursive=True)

        df = pd.read_csv(csv_path)
        self.label_lookup = dict(zip(df["video_id"], df["label_id"]))

        self.samples = []

        for path in self.files[:max_samples]:
            sample_id = int(os.path.basename(os.path.dirname(path)))

            if sample_id in self.label_lookup:
                label = self.label_lookup[sample_id]
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def map_label(self, label_str):
        if label_str in LEFT:
            return 0
        elif label_str in RIGHT:
            return 1
        else:
            return 2

    def events_to_tensor(self, events, num_bins=3):
        if len(events) == 0:
            return np.zeros((2 * num_bins, H, W), dtype=np.float32)
        x = events[:, 0].astype(int)
        y = events[:, 1].astype(int)
        p = events[:, 3]
        t = events[:, 2]

        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

        t_min, t_max = t.min(), t.max()
        bins = np.linspace(t_min, t_max, num_bins + 1)

        frames = []

        for i in range(num_bins):
            mask = (t >= bins[i]) & (t < bins[i + 1])

            img = np.zeros((2, H, W), dtype=np.float32)

            pos = mask & (p > 0)
            neg = mask & (p <= 0)

            img[0, y[pos], x[pos]] = 1.0
            img[1, y[neg], x[neg]] = 1.0

            frames.append(img)

        return np.concatenate(frames, axis=0)

    def __getitem__(self, idx):
        path = self.files[idx]

        sample_id = int(path.split("/")[-2])

        label_str = self.label_lookup[sample_id]
        label = self.map_label(label_str)

        with h5py.File(path, "r") as f:
            events = f["events"][:]

        img = self.events_to_tensor(events)
        img = torch.tensor(img, dtype=torch.float32)

        return img, label
