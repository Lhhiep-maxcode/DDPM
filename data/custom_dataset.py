import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_path, train_dir=None, test_dir=None, val_dir=None, transform=None):
        self.root_path = root_path
        self.transform = transform

        test_dir = os.path.join(root_path, test_dir)
        train_dir = os.path.join(root_path, train_dir)
        val_dir = os.path.join(root_path, val_dir)

        test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]
        train_images = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
        val_images = [os.path.join(val_dir, img) for img in os.listdir(val_dir)]
        
        self.images = test_images + train_images + val_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
