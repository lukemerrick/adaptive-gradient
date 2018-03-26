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
                decay_delta=0.95, decay_k=1, model_name=config_model_name,
                schedule_name='smooth_0.95', logdir=logdir)
        batch.append((model_func, kwargs))

         # run SGD stepped decay
        kwargs = dict (lr=sgd_lr, epochs=epochs,
                task='mnist',
                decay_delta=0.5, decay_k=10, model_name=config_model_name,
                schedule_name='step_.5_10', logdir=logdir)
        batch.append((model_func, kwargs))

        # run with different step
        kwargs = dict (lr=sgd_lr, epochs=epochs,
                task='mnist',
                decay_delta=0.2, decay_k=15, model_name=config_model_name,
                schedule_name='step_.2_15', logdir=logdir)
        batch.append((model_func, kwargs))
        yield batch
def run_experiment_from_generator(tpl):
    model_func, kwargs = tpl
    model = model_func()
    experiment_utils.run_experiment(model, **kwargs)

def run_paralell_experiment(generator, n_processes=10, trials=30):
    criterion = nn.CrossEntropyLoss().cuda()
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        executor.map(run_experiment_from_generator, generator)

common_config = dict(trials=30, sgd_lr=0.03, adam_lr=0.0003, task='mnist',
                     epochs=40, logdir = '../runs/mnist_full/')
models = (('lenet',get_lenet), ('mlp_1024', get_single_layer_mlp), ('mlp_1024_1024', get_double_layer_mlp))
generators = [experiment_generator(model_func=func, model_name=name, **common_config) for name, func in models]
for gen in generators:
    for batch in gen:
        run_paralell_experiment(batch)
# experiments = [exp for gen in generators for exp in gen]
# run_paralell_experiment(experiments)
