import random
import pandas as pd
import lmdb
import os
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import pickle
import shutil


def dump2lmdb(infos, path):
    if not os.path.exists(path):
        os.mkdir(path)
    env = lmdb.open(path, map_size=1.4e11)
    txn = env.begin(write=True)
    to_tensor = transforms.PILToTensor()
    print('Generating lmdb file to: ' + path)

    for i in tqdm(range(len(infos))):
        data_id, label = infos[i]
        img_path = os.path.join('data/Train', data_id)
        img = to_tensor(Image.open(img_path).convert('RGB'))
        img = img.float() / 255
        label = torch.tensor(int(label.split(';')[0]) - 1)
        txn.put(key=str(i).encode(), value=pickle.dumps((img, label)))

    txn.commit()
    env.close()


def main_generation():
    df = pd.read_csv('data/Train_label.csv')
    tot_infos = list(zip(df['FileName'], df['Code']))
    random.shuffle(tot_infos)

    train_infos = tot_infos[: - int(len(tot_infos) // 5)]
    val_infos = tot_infos[- int(len(tot_infos) // 5):]

    if os.path.exists("./data/lmdb/train"):
        shutil.rmtree("./data/lmdb/train")
    if os.path.exists("./data/lmdb/val"):
        shutil.rmtree("./data/lmdb/val")

    dump2lmdb(train_infos, "./data/lmdb/train")
    dump2lmdb(val_infos, "./data/lmdb/val")


if __name__ == '__main__':
    main_generation()
