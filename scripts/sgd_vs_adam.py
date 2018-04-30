from mlp import MLP
from lenet import LeNet
from wideresnet import WideResNet
import dnn_utils
import experiment_utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import pytorch4adam

import tensorboard_logger

import itertools
from concurrent.futures import ProcessPoolExecutor


def get_resnet(multi_gpu=False):
    model = WideResNet(depth=16, num_classes=10, widen_factor=2, drop_rate=0.0)
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    return model.cuda()

def get_lenet():
    return LeNet().cuda()

def get_single_layer_mlp():
    return MLP([28*28, 1024, 10]).cuda()

def get_double_layer_mlp():
    return MLP([28*28, 1024, 1024, 10]).cuda()


# define experiment generator for paralellization
def experiment_batch_generator(trials=10, model_func=get_lenet, model_name='lenet',
        sgd_lr=0.03, adam_lr=0.0003, task='mnist', epochs=40, logdir='../runs/'):
    for trial in range(trials):
        batch = []
        config_model_name = '[trial_{}]{}_'.format(trial, task) + model_name
        # run ADAM
        kwargs = dict(lr=adam_lr, epochs=epochs,
                task='mnist',
                adaptive=True, amsgrad=False, model_name=config_model_name,
                schedule_name='no_lr_decay', logdir=logdir)
        batch.append((model_func, kwargs))

        # run amsgrad
        kwargs = dict(lr=adam_lr, epochs=epochs,
                task='mnist',
                adaptive=True, amsgrad=True, model_name=config_model_name,
                schedule_name='no_lr_decay', logdir=logdir)
        batch.append((model_func, kwargs))

        # run SGD smooth decay
        kwargs = dict(lr=sgd_lr, epochs=epochs,
                task='mnist',
                decay_delta=0.96, decay_k=1, model_name=config_model_name,
                schedule_name='smooth_0.96', logdir=logdir)
        batch.append((model_func, kwargs))

        yield batch
def run_experiment_from_generator(tpl):
    model_func, kwargs = tpl
    model = model_func()
    experiment_utils.run_experiment(model, **kwargs)

def run_paralell_experiment(generator, n_processes=10):
    criterion = nn.CrossEntropyLoss().cuda()
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        executor.map(run_experiment_from_generator, generator)

sgd_lrs = [0.01, 0.033, 0.1, 0.33, 1]
adam_lrs = [0.0001, 0.00033, 0.001, 0.0033, 0.01]

common_config = dict(trials=30, task='mnist',
                     epochs=100, logdir = '../runs/mnist_sgd_vs_adam/')
model_configs = (('lenet', get_lenet, 0.033, 0.0033),
                  ('mlp_1024', get_single_layer_mlp, 0.1, 0.00033))
#                  ('mlp_1024_1024', get_double_layer_mlp, 0.033, 0.0001))

for model_name, model_func, sgd_lr, adam_lr in model_configs:
    generator = experiment_batch_generator(model_func=model_func, model_name=model_name,
                        sgd_lr=sgd_lr, adam_lr=adam_lr, **common_config)
    for batch in generator:
        run_paralell_experiment(batch, n_processes=5)
