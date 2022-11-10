import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from sklearn.metrics import f1_score, recall_score, accuracy_score

import utils.models
from utils.dataset import get_train_val_dataset, get_test_dataset
from utils.utils import Accumulator


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=3e-3)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--logging_path', default='logging/default_experiment')
    parser.add_argument('--load_method', default='basic', help='You can choose different methods for loading dataset.'
                                                               ' [basic, lmdb, preload] are supported.')

    args = parser.parse_args()
    return args


def lr_scheduler_func(epoch_num):
    # lr scheduler if you use LambdaLR
    if epoch_num < 40:
        return 3e-4
    elif epoch_num < 60:
        return 1e-4
    elif epoch_num < 80:
        return 1e-5
    else:
        return 1e-6


def get_metrics(y, y_pred, cuda):
    # Calculate corresponding metrics.
    if cuda:
        y, y_pred = y.cpu(), y_pred.cpu()
    acc = accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    return {'Accuracy': acc, 'Recall': rec, 'F1': f1}


def dump2file(model, args):
    # Use model to generate submission file.
    test_dataset = get_test_dataset(args)
    model.eval()
    result = {'FileName': [], 'Code': []}
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img_id = test_dataset.data_infos[i]
            img = test_dataset[i]
            y_pred = torch.argmax(model(img.unsqueeze(0)), dim=-1).item()
            result['FileName'].append(img_id)
            result['Code'].append(y_pred + 1)
    df = pd.DataFrame(result)
    df.to_csv('my_submission.csv', index=False)


def main():
    args = parse_arg()
    train_dataset, val_dataset = get_train_val_dataset(args)
    tb_logger = SummaryWriter(args.logging_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    model = utils.models.ResNet18(class_num=28, pretrained=False)
    if args.cuda:
        model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Choose different lr schedulers.
    # scheduler = LambdaLR(optimizer=optim, lr_lambda=lr_scheduler_func)
    scheduler = CosineAnnealingLR(optimizer=optim, T_max=args.epoch, eta_min=1e-6)

    for i in tqdm(range(args.epoch)):
        # Train
        model.train()
        print('Training ... ')
        accumulator1 = Accumulator()

        for _, (x, y) in enumerate(train_loader):
            o = model(x)
            loss = criterion(o, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            y_pred = torch.argmax(o, dim=-1)

            metrics = get_metrics(y, y_pred, args.cuda)
            metrics['Loss'] = loss.item()
            accumulator1.append(metrics)
        train_metrics = accumulator1.mean()
        print('    '.join(['EPOCH {}'.format(i)] + ['{}: {:.3f}'.format(k, train_metrics[k]) for k in train_metrics]))

        # Validation
        model.eval()
        print('Validating ... ')
        accumulator2 = Accumulator()
        with torch.no_grad():
            for _, (x, y) in enumerate(val_loader):
                if args.cuda:
                    x, y = x.cuda(), y.cuda()
                o = model(x)
                loss = criterion(o, y)
                y_pred = torch.argmax(o, dim=-1)

                metrics = get_metrics(y, y_pred, args.cuda)
                metrics['Loss'] = loss.item()
                accumulator2.append(metrics)
        val_metrics = accumulator2.mean()
        print('    '.join(['EPOCH {}'.format(i)] + ['{}: {:.3f}'.format(k, val_metrics[k]) for k in val_metrics]))

        scheduler.step()
        for k in train_metrics:
            tb_logger.add_scalar('train/{}'.format(k), train_metrics[k], i)
        for k in val_metrics:
            tb_logger.add_scalar('validation/{}'.format(k), val_metrics[k], i)
        tb_logger.add_scalar('learning_rate', optim.state_dict()['param_groups'][0]['lr'], i)

    dump2file(model, args)


if __name__ == '__main__':
    main()
