import os
import torch
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

        if not os.path.exists(os.path.join(self.directory, 'validation')):
            os.mkdir(os.path.join(self.directory, 'validation'))

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        save_image(grid_image, os.path.join(self.directory, 'validation', str(global_step) + '_image.png'))

        if dataset =='fundus':
            output = torch.sigmoid(output)
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            grid_image = make_grid(decode_seg_map_sequence(output[:3].detach().cpu().numpy(),
                                                        dataset=dataset), 3, normalize=False,)
        else:
            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                        dataset=dataset), 3, normalize=False,
                                                        range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        save_image(grid_image, os.path.join(self.directory, 'validation', str(global_step) + '_pred.png'))

        if dataset == 'fundus':
            grid_image = make_grid(decode_seg_map_sequence(target[:3].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False,)
        else:
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                        dataset=dataset), 3, normalize=False, range=(0, 255))
            
        writer.add_image('Groundtruth label', grid_image, global_step)
        save_image(grid_image, os.path.join(self.directory, 'validation', str(global_step) + '_target.png'))