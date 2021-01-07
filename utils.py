class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':.6f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_list = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.avg_list.append(self.avg)

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class FocalTverskyLoss(nn.Module):
    def __init__(self, beta=0.7, gamma=0.75):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, mask, smooth=1.):
        pred = pred.view(-1)
        mask = mask.view(-1)

        true_pos = (pred * mask).sum()
        false_neg = ((1 - pred) * mask).sum()
        false_pos = (pred * (1 - mask)).sum()
        tversky = (true_pos + smooth) / (true_pos + self.beta * false_neg + (1 - self.beta) * false_pos + smooth)
        focal_tversky_loss = torch.pow((1 - tversky), self.gamma)

        return focal_tversky_loss


class DiceCoef(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pr, gt):
        intersection = (pr * gt).sum()
        dice = (2 * intersection + self.smooth) / (pr.sum() + gt.sum() + self.smooth)

        return dice