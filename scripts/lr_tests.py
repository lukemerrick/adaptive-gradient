from mlp import MLP
from wideresnet import WideResNet
import dnn_utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import pytorch4adam

from tqdm import tqdm

import tensorboard_logger

def get_experiement_name(model_name, schedule_name, adaptive, amsgrad, momentum, lr):
    if adaptive and momentum:
        raise('Cannot run momentum on adaptive')
    if amsgrad and not adaptive:
        raise('Cannot run amsgrad on non adaptive')
    train_loader, val_loader = dnn_utils.get_cifar10_loaders()
    if adaptive:
        if amsgrad:
            method_name = 'AMSGrad'
        else:
            method_name = 'ADAM'
    else:
        if momentum:
            method_name = 'Nest_SGD'
        else:
            method_name = 'SGD'
    lr_name = 'lr={}'.format(lr)
    experiment_name = '_'.join([model_name, method_name,
                                schedule_name, lr_name])
    return experiment_name

def run_experiment(model, lr, decay_delta=1.0, decay_k=1, epochs=400,
                   adaptive=False, amsgrad=False, momentum=0.9, nesterov=True,
                   model_name='default_model', schedule_name='default_schedule',
                   logdir='../runs'):
    has_momentum = momentum != 0
    nesterov = nesterov and has_momentum
    experiment_name = get_experiement_name(model_name, schedule_name, adaptive,
                                           amsgrad, has_momentum, lr)
    tlog = tensorboard_logger.Logger(logdir + '/' + experiment_name)
    train_loader, val_loader = dnn_utils.get_cifar10_loaders()
    decay_lr = dnn_utils.get_lr_decay_function(decay_delta, decay_k, tlog.log_value)
    cudnn.benchmark = True
    if adaptive:
        optimizer = pytorch4adam.Adam(model.parameters(), lr=lr, amsgrad=amsgrad)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                nesterov=nesterov)
    for epoch in tqdm(range(epochs)):
        if not adaptive:
            decay_lr(optimizer, epoch)
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


epochs = 10
adam_lrs = [0.0001, 0.0005, 0.001]
sgd_lrs = [0.05, 0.1, 0.5]
momentum = 0.9
def get_resnet():
    return WideResNet(depth=28, num_classes=10).cuda()
criterion = nn.CrossEntropyLoss().cuda()
for adaptive in [True, False]:
    for amsgrad in [False]:
        for momentum in [0.0, 0.9]:
            if (momentum > 0) and adaptive:
                continue
            lrs = adam_lrs if adaptive else sgd_lrs
            for lr in lrs:
                run_experiment(get_resnet(), lr, decay_delta=0.988, decay_k=1, epochs=epochs,
                        adaptive=adaptive, amsgrad=amsgrad, momentum=momentum,
                        model_name='wideresnet_28', schedule_name='.988decay',
                        logdir='../runs/deciding_lr')
