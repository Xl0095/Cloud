from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
import torch
import random
from tqdm import tqdm
import lmdb
from prepare_dataset import main_generation
import pickle


class CloudDataset(Dataset):
    def __init__(self, data_infos, train, load_method, imsize=224):
        assert load_method in ['lmdb', 'preload', 'basic']
        self.load_method = load_method
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
                transforms.Resize((imsize, imsize)),
            ]
            )
        if self.load_method == 'preload':
            self.preload()
        elif self.load_method == 'lmdb':
            self.init_lmdb()

    def init_lmdb(self):
        if not os.path.exists('data/lmdb/train'):
            raise OSError('Please generate lmdb files first.')
        env_name = 'data/lmdb/train' if self.train else 'data/lmdb/val'
        self.env = lmdb.Environment(env_name)
        self.txn = self.env.begin()

    def preload(self):
        preload_img = []
        preload_label = []
        print('Preloading Dataset. Waiting ... ')
        for i in tqdm(range(len(self.data_infos))):
            img_path = os.path.join('data/Train', self.data_infos[i][0])
            img = self.to_tensor(Image.open(img_path).convert('RGB'))
            img = img.float() / 255
            label = torch.tensor(int(self.data_infos[i][1].split(';')[0]) - 1)
            preload_img.append(img)
            preload_label.append(label)
        self.preload_img = torch.stack(preload_img, dim=0)
        self.preload_label = torch.stack(preload_label, dim=0)

    def __getitem__(self, i):
        if self.load_method == 'basic':
            img_path = os.path.join('data/Train', self.data_infos[i][0])
            img = self.to_tensor(Image.open(img_path).convert('RGB')).cuda()
            img = img.float() / 255
            img = self.transform(img)
            return img, torch.tensor(int(self.data_infos[i][1].split(';')[0]) - 1).cuda()
        elif self.load_method == 'preload':
            img = self.transform(self.preload_img[i].cuda())
            label = self.preload_label[i]
            return img, label
        elif self.load_method == 'lmdb':
            img, label = pickle.loads(self.txn.get(str(i).encode()))
            img = self.transform(img.cuda())
            return img, label

    def __len__(self):
        return len(self.data_infos)


def get_train_val_dataset(args):
    df = pd.read_csv('data/Train_label.csv')
    tot_infos = list(zip(df['FileName'], df['Code']))
    random.shuffle(tot_infos)

    train_infos = tot_infos[: - int(len(tot_infos) // 5)]
    val_infos = tot_infos[- int(len(tot_infos) // 5):]

    train_dataset = CloudDataset(train_infos, train=True, load_method=args.load_method)
    val_dataset = CloudDataset(val_infos, train=False, load_method=args.load_method)

    return train_dataset, val_dataset
