import argparse
import copy
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

from network import NET
from utils import AverageMeter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-dir', type=str, default='model')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    train_data = dataset.MNIST(root="mnist",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

    test_data = dataset.MNIST(root="mnist",
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)

    train_subset, eval_subset = torch.utils.data.random_split(
        train_data, [50000, 10000], generator=torch.Generator().manual_seed(args.seed))

    train_dataloader = data_utils.DataLoader(dataset=train_subset, shuffle=True, batch_size=args.batch_size)
    eval_dataloader = data_utils.DataLoader(dataset=eval_subset, shuffle=False, batch_size=args.batch_size)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = NET().to(device)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_accu = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_subset) - len(train_subset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        accu_num = 0

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)
                preds = preds.argmax(dim=1)

            accu_num += (preds == labels).sum()

        print('eval accuracy: {:.2%}'.format(accu_num/len(eval_subset)))

        if accu_num > best_accu:
            best_epoch = epoch
            best_accu = accu_num
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2%}'.format(best_epoch, best_accu))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))