import dnn_utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import pytorch4adam

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

def run_experiment(model, lr, criterion=nn.CrossEntropyLoss().cuda(),
                    decay_delta=1.0, decay_k=1, epochs=400,
                    adaptive=False, amsgrad=False, momentum=0.9, nesterov=True,
                    task = 'cifar10',
                    model_name='default_model', schedule_name='default_schedule',
                    logdir='../runs'):
    has_momentum = (momentum != 0) and (not adaptive)
    nesterov = nesterov and has_momentum
    experiment_name = get_experiement_name(model_name, schedule_name, adaptive,
                                           amsgrad, has_momentum, lr)
    tlog = tensorboard_logger.Logger(logdir + '/' + experiment_name)
    if task == 'cifar10':
        train_loader, val_loader = dnn_utils.get_cifar10_loaders()
    elif task == 'mnist':
        train_loader, val_loader = dnn_utils.get_mnist_loaders()
    else:
        raise Exception('only cifar10 and minist supported')
    decay_lr = dnn_utils.get_lr_decay_function(decay_delta, decay_k, tlog.log_value)
    cudnn.benchmark = True
    if adaptive:
        optimizer = pytorch4adam.Adam(model.parameters(), lr=lr, amsgrad=amsgrad)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                nesterov=nesterov)
    for epoch in range(epochs):
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
