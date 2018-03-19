from mlp import MLP
from wideresnet import WideResNet
import dnn_utils
import experiment_utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import pytorch4adam

import tensorboard_logger

from concurrent.futures import ProcessPoolExecutor

# experimental config
epochs = 200
logdir = '../runs/adam_tuning/'

def get_resnet(multi_gpu=True):
    model = WideResNet(depth=16, num_classes=10, widen_factor=4, drop_rate=0.5)
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    return model.cuda()
def get_mlp(size=512):
    return MLP([32*32*3, size, size, 10]).cuda()

criterion = nn.CrossEntropyLoss().cuda()
lrs = [0.0001, 0.0003, 0.001]
beta_ones = [0.8, 0.9, 0.95]
beta_twos = [0.998, 0.999, 0.9995]

# define a generator to perform grid search
def train_args_generator(lrs, beta_ones, beta_twos, amsgrads=[True, False]):
    for lr in lrs:
        for b1 in beta_ones:
            for b2 in beta_twos:
                for use_amsgrad in amsgrads:
                    yield lr, b1, b2, use_amsgrad

# run the corresponding experiments
model_name = 'wideresnet_16_4'
for lr, b1, b2, use_amsgrad in train_args_generator(lrs, beta_ones, beta_twos):
    config_model_name = model_name + '_b1={}_b2={}'.format(b1,b2)
    experiment_utils.run_experiment(model=get_resnet(multi_gpu=False), lr=lr, epochs=epochs,
            adaptive=True, amsgrad=use_amsgrad, model_name=config_model_name,
            schedule_name='no_lr_decay', logdir=logdir)
