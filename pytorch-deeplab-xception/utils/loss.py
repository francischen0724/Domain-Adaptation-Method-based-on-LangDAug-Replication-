import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=None, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal' or 'bce' or 'bce-dice']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'bce':
            return self.BCEWithLogitsLoss
        elif mode == 'bce-dice':
            return self.BCEDiceLoss
        else:
            raise NotImplementedError
        
    def dice_coeff(self, logit, target):
        smooth = 1.
        pred = torch.sigmoid(logit)
        bs = pred.size(0)
        m1 = pred.contiguous().view(bs, -1)
        m2 = target.contiguous().view(bs, -1)
        intersection = (m1 * m2).sum(axis=-1)
        score = 1 - ((2. * intersection + smooth) / (m1.sum(axis=-1) + m2.sum(axis=-1) + smooth))
        # print(score.shape)
        # 1/0
        return score
    
    def BCEDiceLoss(self, logit, target, is_aug):
        n, c, h, w = logit.size()
        bce = nn.BCEWithLogitsLoss(weight=self.weight, reduction='none')

        if self.cuda:
            bce = bce.cuda()
        dice_loss = self.dice_coeff(logit, target)
        dice_loss = dice_loss * is_aug
        dice_loss = dice_loss.mean()

        ce = bce(logit, target)
        ce = ce.contiguous().view(n, -1).mean(dim=-1)
        ce = ce * is_aug
        ce = ce.mean()

        loss = ce + dice_loss

        return loss
    
    def BCEWithLogitsLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.BCEWithLogitsLoss(weight=self.weight, size_average=self.size_average, reduction='none')
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target)

        if self.batch_average:
            loss /= n

        return loss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target)

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":

    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




