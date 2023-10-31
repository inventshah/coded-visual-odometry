import os

import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

SIZE = (480, 640)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ICLDataset(Dataset):
    def __init__(
        self, base: str, path: str, codedDir: str = "Coded", cache: bool = True
    ):
        self.path = path

        self.transform = transforms.Compose([transforms.CenterCrop(SIZE)])
        self.data = []
        dir_path = os.path.join(base, path, codedDir)
        for j in range(len([p for p in os.listdir(dir_path) if p.endswith(".png")])):
            file = f"{j}.png"
            coded_file = os.path.join(base, path, codedDir, file)
            depth_file = os.path.join(base, path, "depth", file)

            if cache:
                self.data.append(self.process(coded_file, depth_file))
            else:
                self.data.append((coded_file, depth_file))

        self.cache = cache

        self.len = self.data.__len__()

    def process(self, coded_file: str, depth_file: str):
        coded = torch.from_numpy(cv2.imread(coded_file)).moveaxis(-1, 0)
        metric_depth = torch.from_numpy(
            cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 5000
        )

        return {
            "Coded": self.transform(coded.to(torch.float32)) / 255.0,
            "Depth": self.transform(metric_depth.to(torch.float32)),
        }

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.cache:
            return self.data[idx]
        else:
            return self.process(*self.data[idx])

    def __repr__(self):
        return f"ICLDataset(path={self.path}, n={self.len})"
