import numpy as np


class Evaluator(object):
    def __init__(self, num_class, dataset):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.dataset = dataset
        self.smooth = 1e-6
        self.disc_dices = []
        self.cup_dices = []
        self.dices = []
        self.disc_mious = []
        self.cup_mious = []
        self.mious = []

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.smooth)
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self.smooth)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix) + self.smooth)
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def DICE(self):
        if self.dataset == 'fundus':
            disc_dice = (np.diag(self.confusion_matrix)[0] * 2) / (np.sum(self.confusion_matrix, axis=1)[0] + np.sum(self.confusion_matrix, axis=0)[0] + self.smooth)
            cup_dice = (np.diag(self.confusion_matrix)[1] * 2) / (np.sum(self.confusion_matrix, axis=1)[1] + np.sum(self.confusion_matrix, axis=0)[1] + self.smooth)
            return disc_dice, cup_dice
        else:
            dice = 2 * np.diag(self.confusion_matrix) / (
                        np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) + self.smooth)
            dice = np.nanmean(dice)
            return dice
    
    def DICE_alt(self):
        if self.dataset == 'fundus':
            return np.mean(self.disc_dices), np.mean(self.cup_dices)
        elif self.dataset == 'prostate':
            return np.mean(self.dices)
    
    def MIoU_alt(self):
        if self.dataset == 'fundus':
            return np.mean(self.disc_mious), np.mean(self.cup_mious), np.mean(self.mious)  
        elif self.dataset == 'prostate':
            return np.mean(self.mious)

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.smooth)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix) + self.smooth)

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
    
    def _positive_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def _negative_sigmoid(self, x):
        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)
        return exp / (exp + 1)


    def sigmoid(self, x):
        positive = x >= 0
        # Boolean array inversion is faster than another comparison
        negative = ~positive

        # empty contains junk hence will be faster to allocate
        # Zeros has to zero-out the array after allocation, no need for that
        # See comment to the answer when it comes to dtype
        result = np.empty_like(x, dtype=np.float32)
        result[positive] = self._positive_sigmoid(x[positive])
        result[negative] = self._negative_sigmoid(x[negative])

        return result
    
    def dice_metric(self, pred, label):
        batch_size = pred.shape[0]
        smooth = 1e-6

        if self.dataset == 'fundus':
            disc_dices, cup_dices = [], []
            

            for batch in range(batch_size):
                disc_intersection = (pred[batch][0] * label[batch][0]).sum()
                disc_dice = (2 * disc_intersection) / (pred[batch][0].sum() + label[batch][0].sum() + smooth)
                cup_intersection = (pred[batch][-1] * label[batch][-1]).sum()
                cup_dice = (2 * cup_intersection) / (pred[batch][-1].sum() + label[batch][-1].sum() + smooth)

                disc_dices.append(disc_dice)
                cup_dices.append(cup_dice)

            return np.mean(disc_dices), np.mean(cup_dices)
        elif self.dataset == 'prostate':
            dices = []
            for batch in range(batch_size):
                intersection = (pred[batch] * label[batch]).sum()
                dice = (2 * intersection) / (pred[batch].sum() + label[batch].sum() + smooth)
                dices.append(dice)
            return np.mean(dices)
    
    def miou_metric(self, pred, label):
        batch_size = pred.shape[0]
        mious = []
        smooth = 1e-6

        if self.dataset == 'fundus':
            disc_mious, cup_mious = [], []
            

            for batch in range(batch_size):
                disc_intersection = (pred[batch][0] * label[batch][0]).sum()
                disc_miou = (disc_intersection) / (pred[batch][0].sum() + label[batch][0].sum() - disc_intersection+ smooth)
                cup_intersection = (pred[batch][-1] * label[batch][-1]).sum()
                cup_miou = (cup_intersection) / (pred[batch][-1].sum() + label[batch][-1].sum() - cup_intersection+ smooth)
                miou = (disc_miou + cup_miou) / 2
                disc_mious.append(disc_miou)
                cup_mious.append(cup_miou)
                mious.append(miou)

            return np.mean(disc_mious), np.mean(cup_mious), np.mean(mious)
        elif self.dataset == 'prostate':
            for batch in range(batch_size):
                intersection = (pred[batch] * label[batch]).sum()
                miou = (intersection) / (pred[batch].sum() + label[batch].sum() - intersection + smooth)
                mious.append(miou)
            return np.mean(mious)
    
    def _generate_matrix(self, gt_image, pre_image):

        if self.dataset == 'fundus':
            assert gt_image.ndim == 4 and pre_image.ndim == 4
            assert gt_image.shape[1] == self.num_class and pre_image.shape[1] == self.num_class
            
            pre_image = self.sigmoid(pre_image)
            pre_image = (pre_image >= 0.5).astype(np.int32)

            disc_dice, cup_dice = self.dice_metric(pre_image, gt_image)
            disc_miou, cup_miou, miou = self.miou_metric(pre_image, gt_image)
            self.disc_dices.append(disc_dice)
            self.cup_dices.append(cup_dice)
            self.disc_mious.append(disc_miou)
            self.cup_mious.append(cup_miou)
            self.mious.append(miou)
            # Get batch size and other dimensions
            B, num_class, N, N = gt_image.shape
            
            # Flatten the arrays to simplify calculations
            gt_flat = gt_image.reshape(B, num_class, -1)
            pre_flat = pre_image.reshape(B, num_class, -1)
            
            # Create an empty confusion matrix
            confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int32)
            
            # Iterate over each class
            for gt_class in range(self.num_class):
                for pre_class in range(self.num_class):
                    # Count the number of times a pixel is labeled as gt_class in ground truth
                    # and as pre_class in prediction across all batches
                    true_positive = np.sum((gt_flat[:, gt_class, :] == 1) & (pre_flat[:, pre_class, :] == 1))
                    confusion_matrix[gt_class, pre_class] += true_positive
        elif self.dataset == 'prostate':
            assert gt_image.ndim == 4 and pre_image.ndim == 4
            assert gt_image.shape[1] == self.num_class and pre_image.shape[1] == self.num_class
            
            pre_image = self.sigmoid(pre_image)
            pre_image = (pre_image >= 0.5).astype(np.int32)

            dice = self.dice_metric(pre_image, gt_image)
            miou = self.miou_metric(pre_image, gt_image)
            self.dices.append(dice)
            self.mious.append(miou)

            confusion_matrix = np.zeros((self.num_class, self.num_class), dtype=np.int32)

            confusion_matrix[0,0] = np.sum((gt_image[:,0,:,:] == 1) & (pre_image[:,0,:,:] == 1))
        else:
            mask = (gt_image >= 0) & (gt_image < self.num_class)
            label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
            count = np.bincount(label, minlength=self.num_class**2)
            confusion_matrix = count.reshape(self.num_class, self.num_class)

        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.disc_dices = []
        self.cup_dices = []
        self.mious = []



