import time

from average_meter import AverageMeter

from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Loosely based upon training code in https://github.com/xternalz/WideResNet-pytorch
'''

def train(train_loader, model, criterion, optimizer, epoch, 
          total_epochs=-1, performance_stats={},
          verbose=True, print_freq=10,
          tensorboard_log_function=None,
          tensorboard_stats=['train_loss']):
    '''
    Trains for one epoch. 
    
    x, y are input and target
    y_hat is the predicted output
    
    performance_stats is a dictionary of name:function pairs
    where the function calculates some performance score from y and
    y_hat
    
    see the docs for the 'display_training_stats' function for
    info on verbose, print_freq, tensorboard_log_function, and
    tensorboard_stats
    '''
    
    base_stats = {'batch_time' : AverageMeter(), 'train_loss' : AverageMeter()}
    other_stats = {name:AverageMeter() for name in performance_stats.keys()}
    stats = {**base_stats, **other_stats}
    
    # enter training mode
    model.train()
    
    # begin timing the epoch
    stopwatch = time.time()
    
    # iterate over the batches of the epoch
    for i, (x, y) in enumerate(train_loader):
        y = y.cuda(async=True)
        x = x.cuda()
        # wrap as Variables
        x_var = torch.autograd.Variable(x)
        y_var = torch.autograd.Variable(y)
        
        # forward pass
        y_hat = model(x_var)
        loss = criterion(y_hat, y_var)
        
        # track loss and performance stats
        stats['train_loss'].update(loss.data[0], x.size(0))
        for stat_name, stat_func in performance_stats.items():
            stats[stat_name].update(stat_func(y_hat.data, y), x.size(0))
        
        # compute gradient and do backwards pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # track batch time
        stats['batch_time'].update(time.time() - stopwatch)
        stopwatch = time.time()
        
        # display progress
        if verbose:
            print_stats('train', stats, i, len(train_loader),
                        epoch, total_epochs, print_freq=print_freq)
    
    # print results
    if verbose:
        print_stats('train', stats, len(train_loader), len(train_loader),
                    epoch, total_epochs, print_freq=1)
    
    if tensorboard_log_function is not None:
        stats_to_log = {k:v for k,v in stats.items() if k in tensorboard_stats}
        log_stats_to_tensorboard(stats_to_log, tensorboard_log_function, epoch)
        
def validate(val_loader, model, criterion, epoch, total_epochs=-1,
             performance_stats={}, verbose=True, print_freq=10,
             tensorboard_log_function=None,
             tensorboard_stats=['val_loss']):
    '''
    Evaluates the model on the validation set.
    
    x, y are input and target
    y_hat is the predicted output
    
    performance_stats is a dictionary of name:function pairs
    where the function calculates some performance score from y and
    y_hat
    
    see the docs for the 'display_training_stats' function for
    info on verbose, print_freq, tensorboard_log_function, and
    tensorboard_stats
    '''
    
    base_stats = {'batch_time' : AverageMeter(), 'val_loss' : AverageMeter()}
    other_stats = {name:AverageMeter() for name in performance_stats.keys()}
    stats = {**base_stats, **other_stats}
    
    # enter evaluation mode
    model.eval()
    
    # begin timing the epoch
    stopwatch = time.time()
    
    # iterate over the batches of the single validation epoch
    for i, (x, y) in enumerate(val_loader):
        y = y.cuda(async=True)
        x = x.cuda()
        # wrap as Variables
        x_var = torch.autograd.Variable(x, volatile=True)
        y_var = torch.autograd.Variable(y, volatile=True)
        
        # forward pass
        y_hat = model(x_var)
        loss = criterion(y_hat, y_var)
        
        # track loss and performance stats
        stats['val_loss'].update(loss.data[0], x.size(0))
        for stat_name, stat_func in performance_stats.items():
            stats[stat_name].update(stat_func(y_hat.data, y), x.size(0))
        
        # track batch time
        stats['batch_time'].update(time.time() - stopwatch)
        stopwatch = time.time()
        
        # display progress
        if verbose:
            print_stats('val', stats, i, len(val_loader),
                        epoch, total_epochs, print_freq=print_freq)
    
    # print results
    if verbose:
        print_stats('val', stats, len(val_loader), len(val_loader),
                    epoch, total_epochs, print_freq=1)
    
    if tensorboard_log_function is not None:
        stats_to_log = {k:v for k,v in stats.items() if k in tensorboard_stats}
        log_stats_to_tensorboard(stats_to_log, tensorboard_log_function, epoch)
        
def print_stats(phase, stats, batch, ttl_batches, 
                epoch=1, ttl_epochs=1, 
                print_freq=10):
    '''
    Handles the logging of training and validation statistics to
    printed output and tensorboard.
    
    phase is a string, typically 'tain' or 'val'
    
    batch, ttl_batches, epoch, and ttl_epochs are integers indicating
    the current epoch & batch and the total number of epochs and batches
    
    prints results every print_freq batches
    '''
    
    if batch % print_freq == 0:
        msgs = ['{name} {meter.mean:.4f} ({meter.mean:.4f})'.format(
                            name=name, meter=meter)\
                            for name, meter in stats.items()]
        stats_report = '\t'.join(msgs)
        print('[{phase}] epoch: [{epoch}/{ttl_epochs}]\t'
              'batch: [{batch}/{ttl_batches}]\n'.format(
                  phase=phase, epoch=epoch, ttl_epochs=ttl_epochs,
                  batch=batch, ttl_batches=ttl_batches) + stats_report)
            
def log_stats_to_tensorboard(stats, tensorboard_log_function, epoch):
    '''
    all stats passed in will be logged to tensorboard
    using the tensorboard_log_function
    '''
    for name, meter in stats.items():
        tensorboard_log_function(name, meter.mean, epoch)
        
def get_cifar10_loaders(data_dir='../data', batch_size=128):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__['CIFAR10'](data_dir, train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__['CIFAR10'](data_dir, train=False, 
                                     transform=transform_test),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader

def top1accuracy(y_pred, y_true):
    batch_size = y_true.size(0)
    _, pred_top = y_pred.topk(1, 1, True, True)
    correct = pred_top.view(-1).eq(y_true)
    # index 0th element to extract value from the Tensor
    return correct.sum(0).float().div_(batch_size)[0]

def top1error(y_pred, y_true):
    return 1.0 - top1accuracy(y_pred, y_true)

def get_lr_decay_function(delta, k, tensorboard_log_function=None):
    '''
    Returns a function that decays (scales) lr by delta every k epochs,
    if tensorboard_log_function is provided, the returned function will
    log learning rate when called
    
    The returned function will require inputs:
    optimizer, epoch
    '''
    def adjust_learning_rate(optimizer, epoch):
        lr = optimizer.param_groups[0]['lr']
        if (epoch != 0) and (epoch % k == 0):
            lr = lr * delta
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if tensorboard_log_function is not None:
            tensorboard_log_function('lr', lr, epoch)
            
    return adjust_learning_rate

def get_exponential_lr_decay_function(halflife, tensorboard_log_function=None):
    '''
    Wrapper to get a lr decay function based upon halflife (where k = 1)
    '''
    k = 1
    delta = (1/2)**(1/halflife)
    return get_lr_decay_function(delta, k, tensorboard_log_function)