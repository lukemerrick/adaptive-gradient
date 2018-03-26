from mlp import MLP
from wideresnet import WideResNet
import dnn_utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import pytorch4adam

import tensorboard_logger

from concurrent.futures import ProcessPoolExecutor

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
    has_momentum = (momentum != 0) and (not adaptive)
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

# experimental config
epochs = 200
momentum = 0.9
logdir_adaptive = '../runs/adaptive_mlp/'
logdir_nonadaptive = '../runs/nonadaptive_mlp/'

def get_resnet():
    return WideResNet(depth=28, num_classes=10, widen_factor=10, drop_rate=0.5).cuda()
def get_single_layer_mlp():
    return MLP([32*32*3, 4096, 10]).cuda()
def get_double_layer_mlp():
    return MLP([32*32*3, 4096, 4096, 10]).cuda()

criterion = nn.CrossEntropyLoss().cuda()
nonadaptive_decay_params = [('smooth_decay_0.,98', 0.98, 1), ('step_decay_0.1_80', 0.1, 80)]
resnet_model_list = [('wideresnet_28_10', get_resnet)]
mlp_model_list = [('mlp_4096_single', get_single_layer_mlp), ('mlp_4096_double', get_double_layer_mlp)]

# define a generator to allow for lazy creation of models to pipe into multiprocessing code below
def train_args_generator(model_list):
    for model_name, model_func in model_list:
        if 'resnet' in model_name:
            adam_lr = 0.0003
            sgd_lr = 0.5
        else:
            adam_lr = 0.0003
            sgd_lr = 0.03

        # adaptive method tests
        for amsgrad in [True, False]:
            kwargs = dict(model=model_func, lr=adam_lr, epochs=epochs, adaptive=True, amsgrad=amsgrad,
                    model_name=model_name, schedule_name='no_decay', logdir=logdir_adaptive)
            yield kwargs

        # nonadaptive tests
        for use_momentum in [True, False]:
            current_momentum = momentum if use_momentum else 0.0
            for schedule_name, delta, k in nonadaptive_decay_params:
                kwargs = dict(model=model_func, lr=sgd_lr, decay_delta=delta, decay_k=k,
                        epochs=epochs, momentum=current_momentum,
                        model_name=model_name, schedule_name=schedule_name,
                        logdir=logdir_nonadaptive)
                yield kwargs

# use multiprocessing to complete experimentation much faster (MLP models tend to use a quarter GB of vram and 5-10% of GPU compute power)
def run_on_kwargs(kwargs):
    '''
    slight hack to not initialize the model until experiment time
    '''
    kwargs['model'] = kwargs['model']()
    run_experiment(**kwargs)
resnet_n_threads = 1
mlp_n_threads = 25
# with ProcessPoolExecutor(max_workers=resnet_n_threads) as executor:
#     executor.map(run_on_kwargs, train_args_generator(resnet_model_list))
with ProcessPoolExecutor(max_workers=mlp_n_threads) as executor:
    executor.map(run_on_kwargs, train_args_generator(mlp_model_list))
