import argparse
import sys
import os
import cv2
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import time
from torchvision import datasets, transforms, utils

from tqdm import tqdm
from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist
from ulib.utils import create_dir
from ulib.data_utils import mixup_data, cut_mix
from ulib.logger import Logger
from ulib.non_leaking import augment
import kornia.color as color
import albumentations as A
# Added for fundus datasets
from dataloader.ms_fundus.fundus_dataloader import FundusSegmentation
from dataloader.ms_fundus import fundus_transforms as tr
from dataloader.ms_prostate.convert_csv_to_list import convert_labeled_list
from dataloader.ms_prostate.PROSTATE_dataloader import PROSTATE_dataset
from dataloader.ms_prostate.transform import collate_fn_w_transform

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)

def normalize_to_unit_range(tensor):
    """
    Normalize tensor from range (-1, 1) to range (0, 1).
    """
    return (tensor + 1) / 2

def normalize_to_neg_one_one(tensor):
    """
    Normalize tensor from range (0, 1) to range (-1, 1).
    """
    return tensor * 2 - 1

def normalize_lab_tensor(tensor):
    """
    Normalize a B, C, H, W tensor with LAB format to range [-1, 1].
    
    Args:
    tensor (torch.Tensor): Input tensor with shape (B, C, H, W)
    
    Returns:
    torch.Tensor: Normalized tensor with the same shape
    """
    # Check if tensor has the right number of channels
    assert tensor.size(1) == 3, "Input tensor must have 3 channels (L, a, b)"
    
    # Normalize L channel from [0, 100] to [-1, 1]
    tensor[:, 0, :, :] = (tensor[:, 0, :, :] / 50.0) - 1.0
    
    # Normalize a and b channels from [-128, 127] to [-1, 1]
    tensor[:, 1, :, :] = (tensor[:, 1, :, :] + 128) / 127.5 - 1.0
    tensor[:, 2, :, :] = (tensor[:, 2, :, :] + 128) / 127.5 - 1.0
    
    return tensor

def denormalize_lab_tensor(tensor):
    """
    Denormalize a B, C, H, W tensor from range [-1, 1] back to original LAB format.
    
    Args:
    tensor (torch.Tensor): Input tensor with shape (B, C, H, W)
    
    Returns:
    torch.Tensor: Denormalized tensor with the same shape
    """
    # Check if tensor has the right number of channels
    assert tensor.size(1) == 3, "Input tensor must have 3 channels (L, a, b)"
    
    # Denormalize L channel from [-1, 1] to [0, 100]
    tensor[:, 0, :, :] = (tensor[:, 0, :, :] + 1.0) * 50.0
    
    # Denormalize a and b channels from [-1, 1] to [-128, 127]
    tensor[:, 1, :, :] = (tensor[:, 1, :, :] + 1.0) * 127.5 - 128
    tensor[:, 2, :, :] = (tensor[:, 2, :, :] + 1.0) * 127.5 - 128
    
    return tensor

def rgb_to_lab(rgb_tensor):
    """
    Convert a RGB torch tensor in BCHW format from range (-1, 1) to LAB format.
    """
    # Normalize from (-1, 1) to (0, 1)
    rgb_tensor_unit = normalize_to_unit_range(rgb_tensor)
    
    # Convert from RGB to LAB
    lab_tensor = color.rgb_to_lab(rgb_tensor_unit)
    
    return lab_tensor

def lab_to_rgb(lab_tensor):
    """
    Convert a LAB torch tensor in BCHW format to RGB format and normalize back to (-1, 1).
    """
    # Convert from LAB to RGB
    rgb_tensor_unit = color.lab_to_rgb(lab_tensor)
    
    # Normalize from (0, 1) to (-1, 1)
    rgb_tensor = normalize_to_neg_one_one(rgb_tensor_unit)
    
    return rgb_tensor

def train(epoch, loader, model, optimizer, scheduler, device, args):
	if dist.is_primary():
		loader = tqdm(loader)

	criterion = nn.MSELoss()

	latent_loss_weight = 0.25
	sample_size = 8
	avg_mse= 9999
	mse_sum = 0
	mse_n = 0

	for i, img in enumerate(loader):

		model.zero_grad()

		img = img["image"]

		img = img.to(device)
		img_copy = img.clone().detach()
		lam_mixup, lam_cutmix = 1.0, 1.0
		rand_index = torch.arange(0, img.size()[0]).cuda()
		if args.mixup:
			img, rand_index, lam_mixup = mixup_data(x=img, spherical=args.spherical)

		if args.cutmix:
			img, rand_index, lam_cutmix = cut_mix(x=img, args=args)

		if args.ada:
			img = augment(img, p=args.ada_p)

		if args.color_space == "LAB":
			img = normalize_lab_tensor(rgb_to_lab(img))
			img_copy = normalize_lab_tensor(rgb_to_lab(img_copy))

		lam = lam_cutmix * lam_mixup

		out, latent_loss = model(img + args.input_noise * torch.randn_like(img), rand_index=None, lam=lam)

		recon_loss = criterion(out, img)
		latent_loss = latent_loss.mean()
		loss = recon_loss + latent_loss_weight * latent_loss
		loss.backward()

		if scheduler is not None:
			scheduler.step()
		optimizer.step()

		part_mse_sum = recon_loss.item() * img.shape[0]
		part_mse_n = img.shape[0]
		comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
		comm = dist.all_gather(comm)

		for part in comm:
			mse_sum += part["mse_sum"]
			mse_n += part["mse_n"]

		if dist.is_primary():
			lr = optimizer.param_groups[0]["lr"]
			avg_mse =  mse_sum / mse_n

			loader.set_description(
				(	f"epoch: {epoch + 1}/{args.total_epoch}; "
					f"mse: {recon_loss.item():.5f}; "
					f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
					f"lr: {lr:.5f}"
				)
			)

			if epoch % 1000 == 0:
				model.eval()

				sample = img[:sample_size]

				with torch.no_grad():
					out, _ = model(sample, rand_index=None, lam=lam)

				if args.color_space == "LAB":
					sample = lab_to_rgb(denormalize_lab_tensor(sample))
					out = lab_to_rgb(denormalize_lab_tensor(out))


				# utils.save_image(
				# 	torch.cat([sample, out], 0),
				# 	f"{args.sample_path}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
				# 	nrow=sample_size,
				# 	normalize=True,
				# 	range=(-1, 1),
				# )
				utils.save_image(
					torch.cat([sample, out], 0),
					f"{args.sample_path}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
					nrow=sample_size,
					normalize=True,
					value_range=(-1, 1),  # ✅ 正确
				)

				model.train()
		
	return avg_mse


def main(args):
	if dist.is_primary():
		sys.stdout = Logger(os.path.join(args.log_path, f'{args.dir_name}/log.txt'))
		print(args)
	device = "cuda"
	args.distributed = dist.get_world_size() > 1
	
	if args.dataset == "fundus":
		composed_transforms_tr = transforms.Compose([
			tr.RandomScaleCrop(256),
			tr.Normalize_tf(),
			tr.ToTensor()
		])

		# dataloader config
		dataset =  FundusSegmentation(base_dir=args.data_path, phase='train', splitid=args.split_id, transform=composed_transforms_tr)
		sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)

		loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=8, pin_memory=True, drop_last=False)
	elif args.dataset == "prostate":
		source_csv = []
		for s_n in args.domains:
			source_csv.append(s_n + '_train.csv')
		sr_img_list, sr_label_list = convert_labeled_list(args.data_path, source_csv)
		
		composed_transforms_tr = A.Compose([											
											A.RandomSizedCrop(min_max_height=(300, 330), size=(384, 384), p=0.3),
											A.SafeRotate(limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.3),
											A.HorizontalFlip(p=0.5),
										])

		dataset = PROSTATE_dataset(args.data_path, sr_img_list, sr_label_list,
									target_size=args.size, batch_size=args.batch_size, img_normalize=True, transform=composed_transforms_tr)
		sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
		loader = DataLoader(dataset=dataset,
							batch_size=args.batch_size,
							sampler=sampler,
							pin_memory=True,
							num_workers=8,
							drop_last=False)
		
		print("========== [DEBUG VQ-VAE Data Loading] ==========")
		print(f"Dataset root: {args.data_path}")
		print(f"Train CSVs: {source_csv}")    # 如果有 csv list
		print(f"Total images: {len(sr_img_list)}")
		print(f"Total masks : {len(sr_label_list)}")

		# 打印前 5 个样本
		for i in range(min(5, len(sr_img_list))):
			print(f"  {i}: {sr_img_list[i]} | {sr_label_list[i]}")
		print("===============================================")


	model = VQVAE(embed_dim=args.embed_dim, n_embed=args.n_embed, noise=args.noise).to(device)

	if args.resume is not None:
		model.load_state_dict(torch.load(args.resume))

	if args.distributed:
		model = nn.parallel.DistributedDataParallel(
			model,
			device_ids=[dist.get_local_rank()],
			output_device=dist.get_local_rank(),
		)

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = None
	args.total_epoch = args.n_samples // (len(loader) * args.batch_size)
	if dist.is_primary():
		print('Number of Total Epochs: {}'.format(args.total_epoch))
	if args.sched == "cycle":
		scheduler = CycleScheduler(
			optimizer,
			args.lr,
			n_iter=len(loader) * args.total_epoch,
			momentum=None,
			warmup_proportion=0.05,
		)

	used_sample = 0
	epoch = 0
	best_mse= 9999
	best_epoch = 0
	while used_sample < args.n_samples:
		if args.distributed:
			sampler.set_epoch(epoch)
		avg_mse = train(epoch, loader, model, optimizer, scheduler, device, args)

		if dist.is_primary():
			if epoch % 5 == 0:
				torch.save(model.state_dict(), f"{args.ckpt_path}/vqvae_{str(epoch+1).zfill(3)}.pt")
			if avg_mse < best_mse:
				torch.save(model.state_dict(), f"{args.ckpt_path}/vqvae_best.pt")
				best_mse = avg_mse
				best_epoch = epoch
			print("Current MSE is {:.7f}, Best MSE is {:.7f} in Epoch {}".format(avg_mse, best_mse, best_epoch + 1))

		epoch += 1
		used_sample += len(loader) * args.batch_size


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_gpu", type=int, default=1)

	port = (
			2 ** 15
			+ 2 ** 14
			+ hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
	)
	parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

	parser.add_argument("--embed_dim", type=int, default=64)
	parser.add_argument("--n_embed", type=int, default=512)
	parser.add_argument("--noise", action='store_true')
	parser.add_argument("--dataset", type=str, choices=['fundus', 'prostate'], default="fundus")
	parser.add_argument("--domains", type=str, nargs='+', default=['BIDMC', 'BMC', 'HK', 'I2CVB', 'RUNMC', 'UCL'], help="List of domains to use in case of prostate experiments")
	parser.add_argument("--size", type=int, default=256)
	parser.add_argument("--n_samples", type=int, default=10_000_000)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--sched", type=str)
	parser.add_argument("--batch_size", type=int, default=64, help="batch size for each gpu")

	parser.add_argument("--data_path", type=str)
	parser.add_argument("--log_path", type=str, default='logs')

	parser.add_argument("--input_noise", type=float, default=0.0)
	parser.add_argument("--mixup", action="store_true")
	parser.add_argument("--spherical", action="store_true")
	parser.add_argument("--ada", action="store_true")
	parser.add_argument("--ada_p", type=float, default=0.5)
	parser.add_argument("--cutmix", action="store_true")
	parser.add_argument('--beta', default=0, type=float,
						help='hyperparameter beta')
	parser.add_argument('--cutmix_prob', default=0, type=float,
						help='cutmix probability')

	parser.add_argument("--suffix", type=str)
	parser.add_argument("--color_space", type=str, default="RGB", choices=['RGB', 'LAB'])
	parser.add_argument("--resume", type=str, default=None)

	args = parser.parse_args()

	if args.dataset == "fundus":
		split_id = [int(d[-1]) for d in args.suffix]

		args.split_id = split_id

		split_id_str = "".join([str(i) for i in split_id])
	elif args.dataset == "prostate":
		split_id_str = "-".join(args.domains)

	args.dir_name = "-".join([item for item in [
		f"{args.dataset}",
		split_id_str,
		f"embed{args.embed_dim}",
		f"nembed{args.n_embed}",
		"mixup" if args.mixup else None,
		"noise" if args.noise else None,
		"sp" if args.mixup and args.spherical else None,
		f"cutmix-{args.beta}-{args.cutmix_prob}" if args.cutmix else None,
		f"ada{args.ada_p}" if args.ada else None,
		f"noisein{args.input_noise}" if args.input_noise > 0 else None,
		f"{args.color_space}",
		] if item is not None])

	create_dir(os.path.join(args.log_path, args.dir_name))
	sample_path = os.path.join(args.log_path, f'{args.dir_name}/sample')
	ckpt_path = os.path.join(args.log_path, f'{args.dir_name}/checkpoint')
	create_dir(sample_path)
	create_dir(ckpt_path)
	args.sample_path = sample_path
	args.ckpt_path = ckpt_path

	dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
