import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from utils.dataset import get_train_val_dataset
import utils.models
from utils.utils import Accumulator


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=3e-4)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--cuda', default=True)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--logging_path', default='logging/default_experiment')

    args = parser.parse_args()
    return args


def lr_scheduler_func(epoch_num):
    if epoch_num < 40:
        return 3e-4
    elif epoch_num < 60:
        return 1e-4
    elif epoch_num < 80:
        return 1e-5
    else:
        return 1e-6


def main():
    args = parse_arg()
    train_dataset, val_dataset = get_train_val_dataset()
    tb_logger = SummaryWriter(args.logging_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model = utils.models.ResNet18(class_num=29, pretrained=False)
    if args.cuda:
        model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Choose different lr schedulers.
    scheduler = LambdaLR(optimizer=optim, lr_lambda=lr_scheduler_func)
    # scheduler = CosineAnnealingLR(optimizer=optim, T_max=args.batch_size, eta_min=1e-6)

    for i in range(args.epoch):
        # Train
        model.train()
        accumulator1 = Accumulator()
        for _, (x, y) in enumerate(train_loader):
            print('aaa')
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            o = model(x)
            loss = criterion(o, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            acc = torch.sum(torch.argmax(o, dim=-1) == y).item() / o.shape[0]
            accumulator1.append(loss.item(), acc)
        train_loss, train_acc = accumulator1.mean()
        print('EPOCH {}    TRAIN_LOSS: {:.3f}    TRAIN_ACC: {:.3f}'.format(i, train_loss, train_acc))

        # Validation
        model.eval()
        accumulator2 = Accumulator()
        with torch.no_grad():
            for _, (x, y) in enumerate(val_loader):
                if args.cuda:
                    x, y = x.cuda(), y.cuda()
                o = model(x)
                loss = criterion(o, y)
                acc = torch.sum(torch.argmax(o, dim=-1) == y).item() / len(o.shape[0])
                accumulator2.append(loss.item(), acc)
        val_loss, val_acc = accumulator2.mean()
        print('EPOCH {}    VAL_LOSS: {:.3f}    VAL_ACC: {:.3f}'.format(i, val_loss, val_acc))

        scheduler.step()

        tb_logger.add_scalar('train/acc', train_acc, i)
        tb_logger.add_scalar('train/loss', train_loss, i)
        tb_logger.add_scalar('validation/acc', val_acc, i)
        tb_logger.add_scalar('validation/loss', val_loss, i)
        tb_logger.add_scalar('learning_rate', optim.state_dict()['param_groups'][0]['lr'], i)


if __name__ == '__main__':
    main()
