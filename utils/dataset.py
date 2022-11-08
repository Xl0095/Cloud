from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import torch
import random


class CloudDataset(Dataset):
    def __init__(self, data_infos, train, imsize=224):
        self.data_infos = data_infos
        self.train = train
        self.to_tensor = transforms.PILToTensor()
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.3),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(15),
                transforms.RandomAutocontrast(),
                transforms.RandomResizedCrop(imsize),
            ]
            )
        else:
            self.transform = transforms.Compose([
                transforms.Resize(imsize),
            ]
            )

    def __getitem__(self, i):
        img_path = os.path.join('data/Train', self.data_infos[i][0])
        img = self.to_tensor(Image.open(img_path).convert('RGB'))
        img = img.float() / 255
        img = self.transform(img)
        return img, torch.tensor(int(self.data_infos[i][1].split(';')[0]) - 1)

    def __len__(self):
        return len(self.data_infos)


def get_train_val_dataset():
    df = pd.read_csv('data/Train_label.csv')
    tot_infos = list(zip(df['FileName'], df['Code']))
    random.shuffle(tot_infos)

    train_infos = tot_infos[: - len(tot_infos) // 5]
    val_infos = tot_infos[- len(tot_infos) // 5:]

    train_dataset = CloudDataset(train_infos, train=True)
    val_dataset = CloudDataset(val_infos, train=False)

    return train_dataset, val_dataset
