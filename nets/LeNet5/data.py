import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.dataset import download_dataset

LABEL_DICT = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

class FashionMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        for label_str in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label_str)
            if not os.path.isdir(label_dir):
                continue
            label = LABEL_DICT[int(label_str)]
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label


def load_data():
    download_dataset('fashion-mnist')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = FashionMNISTDataset(root_dir='./datasets/fashion-mnist/train', transform=transform)
    test_set = FashionMNISTDataset(root_dir='./datasets/fashion-mnist/test', transform=transform)
    
    return train_set, test_set

def prepare_data(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)