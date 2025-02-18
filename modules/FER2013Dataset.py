from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, usage="Training", transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            usage (str): One of "Training", "PublicTest", or "PrivateTest".
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["Usage"] == usage]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = int(row["emotion"])
        pixels = row["pixels"]
        pixels = np.array([int(p) for p in pixels.split()]).reshape(48, 48).astype(np.uint8)
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label