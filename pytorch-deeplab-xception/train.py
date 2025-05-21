import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from modeling.deeplabv3.deeplabv3 import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from torchvision.utils import make_grid
from PIL import Image
import torch
import wandb

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        if args.backbone == 'resnet34':
            model = DeepLabV3(num_class=self.nclass, backbone=args.backbone)
        else:    
            model = DeepLab(num_classes=self.nclass,
                            backbone=args.backbone,
                            output_stride=args.out_stride,
                            sync_bn=args.sync_bn,
                            freeze_bn=args.freeze_bn)

        if args.backbone == 'resnet34':
            train_params = add_weight_decay(model, args.weight_decay)
        else:
            train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                            {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(train_params, lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                        lr=args.lr, nesterov=args.nesterov)
        elif args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(train_params, lr=args.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass, args.dataset)
        # Define lr scheduler
        if not args.testing:
            self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                                args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        self.test_best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):

        import time

        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):

            start = time.time()

            image, target = sample['image'], sample['label']

            if self.args.cuda:
                image, target = image.to('cuda',non_blocking=True), target.to('cuda',non_blocking=True)
                aug_wt = sample['aug_wt'].to('cuda',non_blocking=True)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)

            loss = self.criterion(output, target, aug_wt)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            wandb.log({'Iter Loss': loss.item()})

            tbar.set_description('Train loss: %.6f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.6f' % train_loss)

        wandb.log({'Train Epoch': epoch, 'Epoch Loss': train_loss})

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        disc_dice = 0.
        cup_dice = 0.
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            aug_wt = sample['aug_wt']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                aug_wt = sample['aug_wt'].cuda()
            with torch.no_grad():
                output = self.model(image)
                loss = self.criterion(output, target, aug_wt)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.6f' % (test_loss / (i + 1)))
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()

                self.evaluator.add_batch(target, pred)

        # Fast test during the training
        if self.args.dataset != 'prostate':
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
            self.writer.add_scalar('val/mIoU', mIoU, epoch)
            self.writer.add_scalar('val/Acc', Acc, epoch)
            self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
            self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

        if self.args.dataset == 'fundus':
            disc_dice, cup_dice = self.evaluator.DICE()
            alt_disc_dice, alt_cup_dice = self.evaluator.DICE_alt()
            alt_mdice = (alt_disc_dice + alt_cup_dice) / 2
            alt_disc_miou, alt_cup_miou, alt_miou = self.evaluator.MIoU_alt()
            self.writer.add_scalar('val/Disc_Dice', disc_dice, epoch)
            self.writer.add_scalar('val/Cup_Dice', cup_dice, epoch)
            self.writer.add_scalar('val/Alt_Disc_Dice', alt_disc_dice, epoch)
            self.writer.add_scalar('val/Alt_Cup_Dice', alt_cup_dice, epoch)
            self.writer.add_scalar('val/Alt_mDICE', alt_mdice, epoch)
            self.writer.add_scalar('val/Alt_Disc_IoU', alt_disc_miou, epoch)
            self.writer.add_scalar('val/Alt_Cup_IoU', alt_cup_miou, epoch)
            self.writer.add_scalar('val/Alt_mIoU', alt_miou, epoch)
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Disc Dice: {}, Cup Dice: {}, Alt Disc Dice: {}, Alt Cup Dice: {}, Alt mDICE: {}, Alt Disc IoU: {}, Alt Cup IoU: {}, Alt mIoU: {}".format(Acc, Acc_class, mIoU, FWIoU, disc_dice, cup_dice, alt_disc_dice, alt_cup_dice, alt_mdice, alt_disc_miou, alt_cup_miou, alt_miou))

            wandb.log({'Val Epoch': epoch, 'Val Epoch Loss': test_loss,'Val mIoU': alt_miou, 'Val mDICE': alt_mdice, 'Val Disc Dice': alt_disc_dice, 'Val Cup Dice': alt_cup_dice, 'Val Disc IoU': alt_disc_miou, 'Val Cup IoU': alt_cup_miou})
        elif self.args.dataset == 'prostate':
            dice = self.evaluator.DICE_alt()
            mIoU = self.evaluator.MIoU_alt()
            mdice = dice
            print(f'mIoU: {mIoU}, mDICE: {mdice}')

            wandb.log({'Val Epoch': epoch, 'Val Epoch Loss': test_loss,'Val mIoU': mIoU, 'Val mDICE': mdice})
        else:
            dice = self.evaluator.DICE()
            self.writer.add_scalar('val/DICE', dice, epoch)
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Dice: {}".format(Acc, Acc_class, mIoU, FWIoU, dice))

        print('Loss: %.6f' % test_loss)

        if self.args.dataset == 'prostate':
            new_pred = mdice
        elif self.args.dataset == 'fundus':
            new_pred = alt_mdice
        else:
            new_pred = mIoU

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

        wandb.log({'Val Best mDICE': self.best_pred})

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def testing(self, epoch):
        self.model.eval()
        self.evaluator.reset()

        self.print_network()
        
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            aug_wt = sample['aug_wt']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
                aug_wt = sample['aug_wt'].cuda()
            with torch.no_grad():
                output = self.model(image)
                loss = self.criterion(output, target, aug_wt)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                
                if self.args.dataset != 'fundus' and self.args.dataset != 'prostate':
                    pred = np.argmax(pred, axis=1)
                # Add batch sample into evaluator
                self.evaluator.add_batch(target, pred)

        # Fast test during the training
        if self.args.dataset != 'prostate':
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            self.writer.add_scalar('test/total_loss_epoch', test_loss, epoch)
            self.writer.add_scalar('test/mIoU', mIoU, epoch)
            self.writer.add_scalar('test/Acc', Acc, epoch)
            self.writer.add_scalar('test/Acc_class', Acc_class, epoch)
            self.writer.add_scalar('test/fwIoU', FWIoU, epoch)
        print('Testing:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

        if self.args.dataset == 'fundus':
            disc_dice, cup_dice = self.evaluator.DICE()
            alt_disc_dice, alt_cup_dice = self.evaluator.DICE_alt()
            alt_mdice = (alt_disc_dice + alt_cup_dice) / 2
            alt_disc_miou, alt_cup_miou, alt_miou = self.evaluator.MIoU_alt()
            self.writer.add_scalar('test/Disc_Dice', disc_dice, epoch)
            self.writer.add_scalar('test/Cup_Dice', cup_dice, epoch)
            self.writer.add_scalar('test/Alt_Disc_Dice', alt_disc_dice, epoch)
            self.writer.add_scalar('test/Alt_Cup_Dice', alt_cup_dice, epoch)
            self.writer.add_scalar('test/Alt_mDICE', alt_mdice, epoch)
            self.writer.add_scalar('test/Alt_Disc_IoU', alt_disc_miou, epoch)
            self.writer.add_scalar('test/Alt_Cup_IoU', alt_cup_miou, epoch)
            self.writer.add_scalar('test/Alt_mIoU', alt_miou, epoch)
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Disc Dice: {}, Cup Dice: {}, Alt Disc Dice: {}, Alt Cup Dice: {}, Alt mDICE: {}, Alt Disc IoU: {}, Alt Cup IoU: {}, Alt mIoU: {}".format(Acc, Acc_class, mIoU, FWIoU, disc_dice, cup_dice, alt_disc_dice, alt_cup_dice, alt_mdice, alt_disc_miou, alt_cup_miou, alt_miou))

            wandb.log({'Test Epoch': epoch, 'Test Epoch Loss': test_loss,'Test mIoU': alt_miou, 'Test mDICE': alt_mdice, 'Test Disc Dice': alt_disc_dice, 'Test Cup Dice': alt_cup_dice, 'Test Disc IoU': alt_disc_miou, 'Test Cup IoU': alt_cup_miou})

        elif self.args.dataset == 'prostate':
            dice = self.evaluator.DICE_alt()
            mIoU = self.evaluator.MIoU_alt()
            mdice = dice
            print(f'mIoU: {mIoU}, mDICE: {mdice}')

            wandb.log({'Test Epoch': epoch, 'Test Epoch Loss': test_loss,'Test mIoU': mIoU, 'Test mDICE': mdice})
        else:
            dice = self.evaluator.DICE()
            self.writer.add_scalar('test/DICE', dice, epoch)
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Dice: {}".format(Acc, Acc_class, mIoU, FWIoU, dice))

        print('Loss: %.3f' % test_loss)

        if not self.args.testing:

            if self.args.dataset == 'prostate':
                new_pred = mdice
            elif self.args.dataset == 'fundus':
                new_pred = alt_mdice
            else:
                new_pred = mIoU
                
            if new_pred > self.test_best_pred:
                is_best = True
                self.test_best_pred = new_pred
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.test_best_pred,
                }, is_best, filename='test_checkpoint.pth.tar')

            wandb.log({'Test Best mDICE': self.test_best_pred})

    

def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'resnet34', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet(101))')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'fundus', 'prostate'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--data_root', type=str, default='/path/to/data/root', help='dataset root directory')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal', 'bce', 'bce-dice'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--splitid', nargs='+', type=str, default=[1,2,3],
                        help='split id for fundus dataset')
    parser.add_argument('--valid', nargs='+', type=int, default=[4], help='split id for validation dataset')
    parser.add_argument('--testid', nargs='+', type=str, default=[4],
                        help='split id for test dataset')
    parser.add_argument('--testing', action='store_true', default=False,
                    help='run testing only')
    parser.add_argument('--with_L', action='store_true', default=False,
                    help='use L channel replacement set')
    parser.add_argument('--with_LAB_LD', action='store_true', default=False, help='use L channel replacement set with LAB color space')
    parser.add_argument('--aug_wt', type=float, default=1.0, help='augmentation weight')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'fundus': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'fundus': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    wandb.login()
    wandb.init(project='Langevin-Data-Augmentation', config=args, resume='allow')

    if args.testing:
        trainer.testing(trainer.args.start_epoch)
        return
    
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):

            trainer.validation(epoch)
            trainer.testing(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
