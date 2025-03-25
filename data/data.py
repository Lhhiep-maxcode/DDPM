import torch
import torchvision
import os
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image

class StanfordCarDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        test_images = os.listdir("/kaggle/input/stanford-cars-dataset/cars_test/cars_test")
        test_images = [os.path.join("cars_test/cars_test", i) for i in test_images]
        
        train_images = os.listdir("/kaggle/input/stanford-cars-dataset/cars_train/cars_train")
        train_images = [os.path.join("cars_train/cars_train", i) for i in train_images]
        
        self.images = test_images + train_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_path = os.path.join(self.root, self.images[idx])
        image = Image.open(im_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image