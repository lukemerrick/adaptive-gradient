from mlp import MLP
from wideresnet import WideResNet
import dnn_utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

from tqdm import tqdm

import tensorboard_logger


epochs = 500
lrs = [0.001, 0.0033, 0.01, 0.033]
momentum = 0.9
adam_lrs = [lr/10 for lr in lrs]
for i, lr in enumerate(lrs):
    print('[{}/{}] SGD Running Momentum={} LR={}'.format(i+1, len(lrs), momentum, lr))
    train_loader, val_loader = dnn_utils.get_cifar10_loaders()
    tlog = tensorboard_logger.Logger('runs/MLP_0.998decay_{}lr'.format(lr))
    decay = dnn_utils.get_lr_decay_function(0.988, 1, tlog.log_value)
    model = MLP([32*32*3, 512, 10])
    model = model.cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in tqdm(range(epochs)):
        decay(optimizer, epoch)
        dnn_utils.train(train_loader, model, criterion, optimizer, epoch,
            total_epochs=epochs,
            performance_stats={'train_err': dnn_utils.top1error},
            verbose=False, tensorboard_log_function=tlog.log_value,
            tensorboard_stats=['train_loss', 'train_err'])
        dnn_utils.validate(val_loader, model, criterion, epoch,
            total_epochs=epochs,
            performance_stats={'val_err': dnn_utils.top1error},
            verbose=False, tensorboard_log_function=tlog.log_value,
            tensorboard_stats=['val_loss', 'val_err'])

for i, lr in enumerate(adam_lrs):
    print('[{}/{}] Adam Running Momentum={} LR={}'.format(i+1, len(lrs), momentum, lr))
    train_loader, val_loader = dnn_utils.get_cifar10_loaders()
    tlog = tensorboard_logger.Logger('runs/MLP_Adam_{}lr'.format(lr))
    decay = dnn_utils.get_lr_decay_function(0.988, 1, tlog.log_value)
    model = MLP([32*32*3, 512, 10])
    model = model.cuda()
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        decay(optimizer, epoch)
        dnn_utils.train(train_loader, model, criterion, optimizer, epoch,
            total_epochs=epochs,
            performance_stats={'train_err': dnn_utils.top1error},
            verbose=False, tensorboard_log_function=tlog.log_value,
            tensorboard_stats=['train_loss', 'train_err'])
        dnn_utils.validate(val_loader, model, criterion, epoch,
            total_epochs=epochs,
            performance_stats={'val_err': dnn_utils.top1error},
            verbose=False, tensorboard_log_function=tlog.log_value,
            tensorboard_stats=['val_loss', 'val_err'])
