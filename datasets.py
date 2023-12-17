import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, images:list[torch.Tensor], labels:torch.Tensor) -> None:
        self.img_labels = labels
        self.img_store = images

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image  = self.img_store[idx]
        label = self.img_labels[idx]
        return image, label