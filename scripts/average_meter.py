class AverageMeter():
    '''Computes and stores the mean and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.mean = self.sum / self.count
