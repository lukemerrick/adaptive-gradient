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
epochs = 100
logdir = '../runs/sgd_vs_adam/'

def get_resnet(multi_gpu=True):
    model = WideResNet(depth=16, num_classes=10, widen_factor=2, drop_rate=0.0)
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    return model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
adam_lr = 0.0003
sgd_lr = 0.5
model_name = 'wideresnet_16_2'

for trial in range(5):
    config_model_name = '[trial_{}]'.format(trial) + model_name
    # run ADAM
    experiment_utils.run_experiment(model=get_resnet(multi_gpu=False), lr=adam_lr, epochs=epochs,
            adaptive=True, amsgrad=False, model_name=config_model_name,
            schedule_name='no_lr_decay', logdir=logdir)

    # run SGD smooth decay
    experiment_utils.run_experiment(model=get_resnet(multi_gpu=False), lr=sgd_lr, epochs=epochs,
            decay_delta=0.95, decay_k=1, model_name=config_model_name,
            schedule_name='smooth_0.95', logdir=logdir)
    # # run SGD stepped decay
    # experiment_utils.run_experiment(model=get_resnet(multi_gpu=False), lr=sgd_lr, epochs=epochs,
    #         decay_delta=0.01, decay_k=40, model_name=config_model_name,
    #         schedule_name='step_.01_40', logdir=logdir)



