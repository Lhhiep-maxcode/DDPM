import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class StanfordCarDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        test_dir = os.path.join(root, "cars_test/cars_test")
        train_dir = os.path.join(root, "cars_train/cars_train")

        test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir)]
        train_images = [os.path.join(train_dir, img) for img in os.listdir(train_dir)]
        
        self.images = test_images + train_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
