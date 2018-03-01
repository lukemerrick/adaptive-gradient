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

epochs = 400
adam_lr = 0.0005
sgd_lr = 0.05
momentum = 0.9
def get_resnet():
    return WideResNet(depth=28, num_classes=10).cuda()
criterion = nn.CrossEntropyLoss().cuda()
nonadaptive_decay_params = [('smooth_decay_0.,985', 0.985, 1), ('step_decay_0.1_100', 0.1, 100)]

# adaptive method tests
for amsgrad in [True, False]:
    run_experiment(get_resnet(), adam_lr, epochs=epochs, adaptive=True, amsgrad=amsgrad,
            model_name='wideresnet_28', schedule_name='no_decay', logdir='../runs/adaptive/')

# nonadaptive tests
for use_momentum in [True, False]:
    current_momentum = momentum if use_momentum else 0.0
    for schedule_name, delta, k in nonadaptive_decay_params:
        run_experiment(get_resnet(), sgd_lr, delta, k, epochs, momentum=current_momentum,
                model_name='wideresnet_28', schedule_name=schedule_name,
                logdir='../runs/nonadaptive')


